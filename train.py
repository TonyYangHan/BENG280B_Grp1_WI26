import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

from data import build_case_index, split_cases, NpzSliceDataset
from loss import BCEDiceLoss, dice_score_from_logits
from utils import (
    set_seed, AverageMeter, save_checkpoint,
    load_mae_vitb16_encoder_weights,
    load_yaml_config, deep_get
)


class FrozenMAEEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    @torch.no_grad()
    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, N+1, C]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token tensor [B,N,C], got {tokens.shape}.")
        patch = tokens[:, 1:, :]
        B, N, C = patch.shape
        h = w = int(N ** 0.5)
        if h * w != N:
            raise RuntimeError(f"Patch tokens N={N} is not a square; check input size/patch size.")
        feat = patch.transpose(1, 2).reshape(B, C, h, w)
        return feat


class SimpleUpsampleDecoder(nn.Module):
    def __init__(self, in_ch: int = 768, out_size: int = 224, patch: int = 16, mid_ch: int = 256):
        super().__init__()
        assert out_size % patch == 0, "out_size must be divisible by patch size."

        def gn(c):
            g = 32
            while c % g != 0 and g > 1:
                g //= 2
            return nn.GroupNorm(g, c)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            gn(mid_ch),
            nn.GELU(),
        )

        steps = int(round(torch.log2(torch.tensor(patch)).item()))  # patch=16 -> 4
        self.blocks = nn.ModuleList()
        for _ in range(steps):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
                gn(mid_ch),
                nn.GELU(),
            ))

        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, feat):
        x = self.proj(feat)
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(x)
        return self.head(x)


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, amp: bool):
    encoder.eval()
    decoder.train()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda",enabled=amp):
            feat = encoder(x)
            logits = decoder(feat)
            loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            d = dice_score_from_logits(logits, y).item()

        loss_meter.update(loss.item(), n=x.size(0))
        dice_meter.update(d, n=x.size(0))

    return loss_meter.avg, dice_meter.avg


@torch.no_grad()
def validate(encoder, decoder, loader, criterion, device):
    encoder.eval()
    decoder.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = encoder(x)
        logits = decoder(feat)
        loss = criterion(logits, y)
        d = dice_score_from_logits(logits, y).item()

        loss_meter.update(loss.item(), n=x.size(0))
        dice_meter.update(d, n=x.size(0))

    return loss_meter.avg, dice_meter.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=str)
    ap.add_argument("--out_dir", type=str, default="../runs/mae_decoder")

    # Config loading (defaults to config.yaml next to this train.py)
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: next to train.py)")

    args = ap.parse_args()

    # Load config.yaml
    default_cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg_path = Path(args.config).expanduser().resolve() if args.config else default_cfg_path
    cfg = load_yaml_config(str(cfg_path))
    print(f"Config: {cfg_path} ({'loaded' if cfg else 'not found/empty -> using defaults'})")

    mae_ckpt = deep_get(cfg, "mae_ckpt", None)
    if not mae_ckpt:
        raise ValueError("mae_ckpt must be set in config.yaml")

    # Pull values from config with defaults
    seed = int(deep_get(cfg, "seed", 1337))

    out_size = int(deep_get(cfg, "data.out_size", 224))
    k_slices = int(deep_get(cfg, "data.k_slices", 3))
    val_frac = float(deep_get(cfg, "data.val_frac", 0.2))
    lesion_sampling_cfg = deep_get(cfg, "data.lesion_sampling", 0.7)
    auto_lesion_sampling = isinstance(lesion_sampling_cfg, str) and lesion_sampling_cfg.strip().lower() == "auto"
    lesion_sampling = float(lesion_sampling_cfg) if not auto_lesion_sampling else 0.0
    img_key = str(deep_get(cfg, "data.img_key", "img"))
    mask_key = str(deep_get(cfg, "data.mask_key", "mask"))
    normalize = bool(deep_get(cfg, "data.normalize", True))

    epochs = int(deep_get(cfg, "train.epochs", 30))
    batch_size = int(deep_get(cfg, "train.batch_size", 16))
    num_workers = int(deep_get(cfg, "train.num_workers", 4))
    amp = bool(deep_get(cfg, "train.amp", False))

    lr = float(deep_get(cfg, "optim.lr", 1e-3))
    wd = float(deep_get(cfg, "optim.wd", 0.05))

    bce_w = float(deep_get(cfg, "loss.bce_weight", 0.5))
    dice_w = float(deep_get(cfg, "loss.dice_weight", 0.5))

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Data
    # -----------------
    npz_paths = sorted(str(p) for p in Path(args.data_root).glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No *.npz found in {args.data_root}")

    cases = build_case_index(npz_paths, img_key=img_key, mask_key=mask_key)
    train_cases, val_cases = split_cases(cases, val_frac=val_frac, seed=seed)

    if auto_lesion_sampling:
        total_slices = sum(c.num_slices for c in train_cases)
        lesion_slices = sum(len(c.lesion_slices) for c in train_cases)
        lesion_sampling = lesion_slices / total_slices if total_slices > 0 else 0.0
        lesion_sampling = max(0.0, min(1.0, lesion_sampling))
        print(f"Auto lesion_sampling: {lesion_sampling:.4f} ({lesion_slices}/{total_slices} slices with lesions in train set)")

    train_ds = NpzSliceDataset(
        train_cases,
        out_size=out_size,
        k_slices=k_slices,
        lesion_sampling=lesion_sampling,
        img_key=img_key,
        mask_key=mask_key,
        normalize=normalize,
    )
    val_ds = NpzSliceDataset(
        val_cases,
        out_size=out_size,
        k_slices=k_slices,
        lesion_sampling=0.0,
        img_key=img_key,
        mask_key=mask_key,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------
    # Model
    # -----------------
    # Create ViT-B/16 with img_size matching out_size (so pos_embed length matches)
    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, img_size=out_size)
    vit.to(device)

    epoch_loaded, info = load_mae_vitb16_encoder_weights(vit, mae_ckpt, device=device)
    print(f"Loaded MAE weights (epoch={epoch_loaded}) -> {info}")

    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    encoder = FrozenMAEEncoder(vit).to(device)
    decoder = SimpleUpsampleDecoder(in_ch=768, out_size=out_size, patch=16, mid_ch=256).to(device)

    # -----------------
    # Loss / opt
    # -----------------
    criterion = BCEDiceLoss(bce_weight=bce_w, dice_weight=dice_w).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=wd)

    # -----------------
    # Train
    # -----------------
    best_val_dice = -1.0
    for ep in range(1, epochs + 1):
        tr_loss, tr_dice = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, device, amp=amp)
        va_loss, va_dice = validate(encoder, decoder, val_loader, criterion, device)

        print(f"[{ep:03d}/{epochs}] "
              f"train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
              f"val loss={va_loss:.4f} dice={va_dice:.4f}")

        ckpt = {
            "epoch": ep,
            "decoder": decoder.state_dict(),
            "config_path": str(cfg_path),
            "config": cfg,
            "runtime": {
                "device": device,
                "out_size": out_size,
                "k_slices": k_slices,
                "batch_size": batch_size,
                "lr": lr,
                "wd": wd,
                "amp": amp,
            },
            "val_dice": va_dice,
        }
        save_checkpoint(str(out_dir / "last.pth"), ckpt)

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            save_checkpoint(str(out_dir / "best.pth"), ckpt)
            print(f"  ✓ new best val dice: {best_val_dice:.4f}")

    print("Done. Best val dice:", best_val_dice)


if __name__ == "__main__":
    main()
