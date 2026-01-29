import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import timm

from data import build_case_index, split_cases, NpzSliceDataset
from loss import FocalDiceLoss, dice_score_from_logits, dice_score_ignore_empty
from utils import (
    set_seed, AverageMeter, save_checkpoint,
    load_mae_vitb16_encoder_weights,
    load_yaml_config, deep_get, compute_roc_auc, sample_pixels_for_roc,
    save_roc_plot
)
from model import FrozenMAEEncoder, SimpleUpsampleDecoder


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, amp: bool, dice_thr: float, ignore_empty_dice: bool):
    encoder.eval()
    decoder.train()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            feat = encoder(x)
            logits = decoder(feat)
            loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            if ignore_empty_dice:
                d = dice_score_ignore_empty(logits, y, thr=dice_thr)
            else:
                d = float(dice_score_from_logits(logits, y, thr=dice_thr).item())

        loss_meter.update(loss.item(), n=x.size(0))
        dice_meter.update(d, n=x.size(0))

    return loss_meter.avg, dice_meter.avg


@torch.no_grad()
def validate(
    encoder, decoder, loader, criterion, device,
    dice_thr: float,
    ignore_empty_dice: bool,
    roc_cfg: dict,
    out_dir: Path,
    epoch: int,
):
    encoder.eval()
    decoder.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    roc_enabled = bool(roc_cfg.get("enabled", False))
    sp = int(roc_cfg.get("sample_pixels_per_slice", 2048))
    spp = int(roc_cfg.get("pos_pixels_per_slice", 256))
    max_slices = int(roc_cfg.get("max_slices", 4096))
    save_plot = bool(roc_cfg.get("save_plot", True))

    probs_all = []
    labels_all = []
    slices_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = encoder(x)
        logits = decoder(feat)
        loss = criterion(logits, y)

        if ignore_empty_dice:
            d = dice_score_ignore_empty(logits, y, thr=dice_thr)
        else:
            d = float(dice_score_from_logits(logits, y, thr=dice_thr).item())

        loss_meter.update(loss.item(), n=x.size(0))
        dice_meter.update(d, n=x.size(0))

        if roc_enabled and slices_seen < max_slices:
            probs = torch.sigmoid(logits)
            p_s, y_s = sample_pixels_for_roc(
                probs, y,
                sample_pixels_per_slice=sp,
                pos_pixels_per_slice=spp,
            )
            probs_all.append(p_s)
            labels_all.append(y_s)
            slices_seen += int(x.size(0))

    val_auc = float("nan")
    if roc_enabled and probs_all:
        probs_np = np.concatenate(probs_all, axis=0)
        labels_np = np.concatenate(labels_all, axis=0)
        fpr, tpr, val_auc = compute_roc_auc(probs_np, labels_np)

        if save_plot:
            save_roc_plot(fpr, tpr, val_auc, out_dir=out_dir, epoch=epoch)

    return loss_meter.avg, dice_meter.avg, val_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=str)
    ap.add_argument("--out_dir", "-o", type=str, default="../runs/mae_decoder")
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

    seed = int(deep_get(cfg, "seed", 1337))
    set_seed(seed)

    # Data cfg
    out_size = int(deep_get(cfg, "data.out_size", 224))
    val_frac = float(deep_get(cfg, "data.val_frac", 0.2))
    mask_key = str(deep_get(cfg, "data.mask_key", "mask"))
    normalize = bool(deep_get(cfg, "data.normalize", True))

    channel_mode = str(deep_get(cfg, "data.channel_mode", "modalities")).lower().strip()
    modality_keys = deep_get(cfg, "data.modality_keys", ["dwi", "adc", "flair"])
    img_key = str(deep_get(cfg, "data.img_key", "img"))
    k_slices = int(deep_get(cfg, "data.k_slices", 3))

    empty_slice_prob = float(deep_get(cfg, "data.empty_slice_prob", 0.1))
    val_empty_slice_prob = float(deep_get(cfg, "data.val_empty_slice_prob", 0.0))

    crop_enabled = bool(deep_get(cfg, "data.crop.enabled", False))
    crop_size_native = int(deep_get(cfg, "data.crop.size_native", 96))
    crop_jitter = float(deep_get(cfg, "data.crop.jitter", 0.15))

    # Train cfg
    epochs = int(deep_get(cfg, "train.epochs", 30))
    batch_size = int(deep_get(cfg, "train.batch_size", 16))
    num_workers = int(deep_get(cfg, "train.num_workers", 4))
    amp = bool(deep_get(cfg, "train.amp", False))

    lr = float(deep_get(cfg, "optim.lr", 1e-3))
    wd = float(deep_get(cfg, "optim.wd", 0.05))

    # Loss cfg
    focal_gamma = float(deep_get(cfg, "loss.focal_gamma", 1.0))
    alpha_pos = float(deep_get(cfg, "loss.alpha_pos", 0.75))
    alpha_neg = float(deep_get(cfg, "loss.alpha_neg", 0.25))
    focal_w = float(deep_get(cfg, "loss.focal_w", 0.5))
    dice_w = float(deep_get(cfg, "loss.dice_w", 0.5))

    # Metrics cfg
    dice_thr = float(deep_get(cfg, "metrics.dice_thr", 0.5))
    ignore_empty_in_dice = bool(deep_get(cfg, "metrics.ignore_empty_in_dice", True))
    roc_cfg = deep_get(cfg, "metrics.roc", {}) or {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Data
    # -----------------
    npz_paths = sorted(str(p) for p in Path(args.data_root).glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No *.npz found in {args.data_root}")

    cases = build_case_index(
        npz_paths,
        mask_key=mask_key,
        channel_mode=channel_mode,
        modality_keys=modality_keys,
        img_key=img_key,
    )
    train_cases, val_cases = split_cases(cases, val_frac=val_frac, seed=seed)

    train_ds = NpzSliceDataset(
        train_cases,
        out_size=out_size,
        channel_mode=channel_mode,
        modality_keys=modality_keys,
        img_key=img_key,
        k_slices=k_slices,
        mask_key=mask_key,
        empty_slice_prob=empty_slice_prob,
        normalize=normalize,
        crop_enabled=crop_enabled,
        crop_size_native=crop_size_native,
        crop_jitter=crop_jitter,
    )
    val_ds = NpzSliceDataset(
        val_cases,
        out_size=out_size,
        channel_mode=channel_mode,
        modality_keys=modality_keys,
        img_key=img_key,
        k_slices=k_slices,
        mask_key=mask_key,
        empty_slice_prob=val_empty_slice_prob,
        normalize=normalize,
        crop_enabled=crop_enabled,   # keep consistent; you can set false in cfg if you want
        crop_size_native=crop_size_native,
        crop_jitter=crop_jitter,
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
    criterion = FocalDiceLoss(
        gamma=focal_gamma,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        focal_w=focal_w,
        dice_w=dice_w,
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=wd)

    # -----------------
    # Train
    # -----------------
    best_val_dice = -1.0
    best_val_auc = float("-inf")

    for ep in range(1, epochs + 1):
        tr_loss, tr_dice = train_one_epoch(
            encoder, decoder, train_loader, criterion, optimizer, device,
            amp=amp, dice_thr=dice_thr, ignore_empty_dice=ignore_empty_in_dice
        )
        va_loss, va_dice, va_auc = validate(
            encoder, decoder, val_loader, criterion, device,
            dice_thr=dice_thr,
            ignore_empty_dice=ignore_empty_in_dice,
            roc_cfg=roc_cfg,
            out_dir=out_dir,
            epoch=ep,
        )

        msg = (f"[{ep:03d}/{epochs}] "
               f"train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
               f"val loss={va_loss:.4f} dice={va_dice:.4f}")
        if bool(roc_cfg.get("enabled", False)):
            msg += f" auc={va_auc:.4f}"
        print(msg)

        ckpt = {
            "epoch": ep,
            "decoder": decoder.state_dict(),
            "config_path": str(cfg_path),
            "config": cfg,
            "runtime": {
                "device": device,
                "out_size": out_size,
                "channel_mode": channel_mode,
                "modality_keys": modality_keys,
                "k_slices": k_slices,
                "batch_size": batch_size,
                "lr": lr,
                "wd": wd,
                "amp": amp,
            },
            "val_dice": va_dice,
            "val_auc": va_auc,
        }
        save_checkpoint(str(out_dir / "last.pth"), ckpt)

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            save_checkpoint(str(out_dir / "best.pth"), ckpt)
            print(f"  ✓ new best val dice: {best_val_dice:.4f}")

        # optional: also track best AUROC
        if (not np.isnan(va_auc)) and (va_auc > best_val_auc):
            best_val_auc = va_auc
            save_checkpoint(str(out_dir / "best_auc.pth"), ckpt)
            print(f"  ✓ new best val auc: {best_val_auc:.4f}")

    print("Done. Best val dice:", best_val_dice)
    if bool(roc_cfg.get("enabled", False)):
        print("Done. Best val auc:", best_val_auc)


if __name__ == "__main__":
    main()
