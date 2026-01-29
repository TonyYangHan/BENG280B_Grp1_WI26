import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt

from model import FrozenMAEEncoder, SimpleUpsampleDecoder
from utils import load_yaml_config, deep_get, load_mae_vitb16_encoder_weights


def normalize_channels(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c]
        lo, hi = np.percentile(v, [0.5, 99.5])
        v = np.clip(v, lo, hi)
        mu = float(v.mean())
        sd = float(v.std()) + 1e-6
        out[c] = (v - mu) / sd
    return out


def load_case_multimodal(npz_path: str, modality_keys, mask_key: str):
    d = np.load(npz_path, allow_pickle=False)
    vols = [d[k].astype(np.float32) for k in modality_keys]  # list of [Z,H,W]
    mask = d[mask_key].astype(np.uint8)                      # [Z,H,W]
    return vols, mask


def load_case_zstack(npz_path: str, img_key: str, mask_key: str):
    d = np.load(npz_path, allow_pickle=False)
    img = d[img_key].astype(np.float32)      # [Z,H,W]
    mask = d[mask_key].astype(np.uint8)      # [Z,H,W]
    return img, mask


def stack_slices(vol, z, k):
    z = int(z)
    k = int(k)
    Z = vol.shape[0]
    if k == 1:
        idxs = [z]
    elif k == 3:
        idxs = [max(0, z - 1), z, min(Z - 1, z + 1)]
    elif k == 5:
        idxs = [max(0, z - 2), max(0, z - 1), z, min(Z - 1, z + 1), min(Z - 1, z + 2)]
    else:
        raise ValueError("k_slices must be 1, 3, or 5")
    return vol[idxs, :, :]


def prepare_model(cfg, device):
    out_size = int(deep_get(cfg, "data.out_size", 224))

    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, img_size=out_size)
    vit.to(device)

    mae_ckpt = deep_get(cfg, "mae_ckpt", None)
    if not mae_ckpt:
        raise ValueError("mae_ckpt missing in config")
    load_mae_vitb16_encoder_weights(vit, mae_ckpt, device=device)

    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    encoder = FrozenMAEEncoder(vit).to(device)
    decoder = SimpleUpsampleDecoder(in_ch=768, out_size=out_size, patch=16, mid_ch=256).to(device)
    decoder.eval()
    return encoder, decoder, out_size


def predict_slice(encoder, decoder, x_np: np.ndarray, device, out_size: int, threshold: float):
    """Run model on a single slice stack and return prob map + binary mask."""
    xt = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1,C,H,W]
    xt = F.interpolate(xt, size=(out_size, out_size), mode="bilinear", align_corners=False)

    with torch.no_grad():
        feat = encoder(xt)
        logits = decoder(feat)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (probs > float(threshold)).astype(np.uint8)

    return probs, pred


def visualize(slice_img, gt_mask, pred_mask, out_path, title):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(gt_mask, cmap="Reds", alpha=0.35)
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(pred_mask, cmap="Blues", alpha=0.35)
    plt.title("Pred")
    plt.axis("off")

    plt.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", type=str, help="Path to case npz")
    ap.add_argument("checkpoint", type=str, help="Path to trained decoder checkpoint (best.pth)")
    ap.add_argument("--config", type=str, default=None, help="Optional config.yaml path (fallback to checkpoint config)")
    ap.add_argument("--out_dir", type=str, default="outputs/infer", help="Directory to save visualizations")
    ap.add_argument("--device", type=str, default=None, help="Override device, e.g., cpu or cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")
    ap.add_argument("--z", type=int, default=None, help="Slice index to visualize (default: middle slice)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_yaml_config(args.config) if args.config else ckpt.get("config", {})
    if not cfg:
        raise ValueError("Config not found; provide --config or ensure checkpoint contains config")

    encoder, decoder, out_size = prepare_model(cfg, device)

    # Load decoder weights
    decoder_sd = ckpt.get("decoder", ckpt)
    decoder.load_state_dict(decoder_sd, strict=False)

    channel_mode = str(deep_get(cfg, "data.channel_mode", "modalities")).lower().strip()
    mask_key = str(deep_get(cfg, "data.mask_key", "mask"))
    normalize = bool(deep_get(cfg, "data.normalize", True))

    if channel_mode == "modalities":
        modality_keys = deep_get(cfg, "data.modality_keys", ["dwi", "adc", "flair"])
        vols, mask = load_case_multimodal(args.npz_path, modality_keys, mask_key)

        def build_input(z_idx: int):
            x_np = np.stack([v[z_idx] for v in vols], axis=0)  # [C,H,W]
            return x_np, vols[0][z_idx]  # show DWI by default

    else:
        img_key = str(deep_get(cfg, "data.img_key", "img"))
        k_slices = int(deep_get(cfg, "data.k_slices", 3))
        img, mask = load_case_zstack(args.npz_path, img_key, mask_key)

        def build_input(z_idx: int):
            x_np = stack_slices(img, z_idx, k_slices)  # [C,H,W]
            return x_np, img[z_idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose slices: user-provided z or all slices with non-zero GT mask
    mask_nonzero = (mask.reshape(mask.shape[0], -1).max(axis=1) > 0)
    target_slices = [int(args.z)] if args.z is not None else list(np.where(mask_nonzero)[0])

    if not target_slices:
        print("No non-zero slices in mask; nothing to predict.")
        return

    saved = 0
    for z in target_slices:
        x_np, slice_img = build_input(z)
        y = (mask[z] > 0).astype(np.uint8)

        if normalize:
            x_np = normalize_channels(x_np)

        probs, pred = predict_slice(encoder, decoder, x_np, device, out_size, args.threshold)

        y_t = torch.from_numpy(y[None, None].astype(np.float32))
        y_t = F.interpolate(y_t, size=(out_size, out_size), mode="nearest")[0, 0].numpy().astype(np.uint8)

        slice_img_resized = F.interpolate(
            torch.from_numpy(slice_img[None, None].astype(np.float32)),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        out_path = out_dir / (Path(args.npz_path).stem + f"_z{z:03d}_thr{args.threshold:.2f}.png")
        visualize(
            slice_img=slice_img_resized,
            gt_mask=y_t,
            pred_mask=pred,
            out_path=out_path,
            title=f"z={z} (thr={args.threshold})"
        )
        saved += 1
        print("Saved:", out_path)

    print(f"Done. Saved {saved} slice(s) to {out_dir}.")


if __name__ == "__main__":
    main()
