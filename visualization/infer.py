import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt

from train import FrozenMAEEncoder, SimpleUpsampleDecoder
from utils import load_yaml_config, deep_get, load_mae_vitb16_encoder_weights


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


def normalize_slice_stack(x):
    # x: [C,H,W] float32
    center = x[x.shape[0] // 2]
    lo, hi = np.percentile(center, [0.5, 99.5])
    x = np.clip(x, lo, hi)
    mu = x.mean()
    sd = x.std() + 1e-6
    x = (x - mu) / sd
    return x.astype(np.float32, copy=False)


def load_case(npz_path, img_key, mask_key):
    d = np.load(npz_path, allow_pickle=False)
    img = d[img_key].astype(np.float32)
    mask = d[mask_key].astype(np.uint8)
    return img, mask


def prepare_model(cfg, device):
    out_size = int(deep_get(cfg, "data.out_size", 224))
    amp = bool(deep_get(cfg, "train.amp", False))

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

    return encoder, decoder, out_size, amp


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
    ap.add_argument("npz_path", type=str, help="Path to case npz containing img/mask")
    ap.add_argument("checkpoint", type=str, help="Path to trained decoder checkpoint (best.pth)")
    ap.add_argument("--config", type=str, default=None, help="Optional config.yaml path (fallback to checkpoint config)")
    ap.add_argument("--out_dir", type=str, default="outputs/infer", help="Directory to save visualizations")
    ap.add_argument("--device", type=str, default=None, help="Override device, e.g., cpu or cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")
    ap.add_argument("--verbose", "-v", action="store_true", help="If set, enables verbose logging")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_yaml_config(args.config) if args.config else ckpt.get("config", {})
    if not cfg:
        raise ValueError("Config not found; provide --config or ensure checkpoint contains config")

    encoder, decoder, out_size, amp = prepare_model(cfg, device)

    # Load decoder weights
    decoder_sd = ckpt.get("decoder", ckpt)
    decoder.load_state_dict(decoder_sd, strict=False)

    k_slices = int(deep_get(cfg, "data.k_slices", 3))
    normalize = bool(deep_get(cfg, "data.normalize", True))
    img_key = str(deep_get(cfg, "data.img_key", "img"))
    mask_key = str(deep_get(cfg, "data.mask_key", "mask"))

    img, mask = load_case(args.npz_path, img_key, mask_key)
    Z = img.shape[0]
    lesion_slices = np.where(mask.reshape(Z, -1).sum(axis=1) > 0)[0].tolist()
    if not lesion_slices:
        print("No lesion slices found in this case; nothing to visualize.")
        return

    out_dir = Path(args.out_dir)

    for z in lesion_slices:
        if mask[z].sum() == 0:
            continue
        stack = stack_slices(img, z, k_slices)
        stack_norm = normalize_slice_stack(stack) if normalize else stack.astype(np.float32)

        xt = torch.from_numpy(stack_norm).unsqueeze(0)  # [1,C,H,W]
        xt = F.interpolate(xt, size=(out_size, out_size), mode="bilinear", align_corners=False)
        xt = xt.to(device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=amp and device.startswith("cuda")):
            feat = encoder(xt)
            logits = decoder(feat)
            pred = torch.sigmoid(logits)

        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        if args.verbose:
            print(f"max proba: {pred_np.max():.4f}, min proba: {pred_np.min():.4f}, mean proba: {pred_np.mean():.4f}")
        pred_bin = (pred_np >= args.threshold).astype(np.float32)
        gt_mask = mask[z].astype(np.float32)

        # Resize gt to match out_size for fair overlay
        gt_t = torch.from_numpy(gt_mask)[None, None]
        gt_resized = F.interpolate(gt_t, size=(out_size, out_size), mode="nearest").squeeze().numpy()

        center_slice = stack[k_slices // 2]
        if center_slice.shape != (out_size, out_size):
            center_t = torch.from_numpy(center_slice)[None, None]
            center_slice = F.interpolate(center_t, size=(out_size, out_size), mode="bilinear", align_corners=False).squeeze().numpy()

        out_path = out_dir / f"slice_{z:03d}.png"
        visualize(center_slice, gt_resized, pred_bin, out_path, title=f"z={z} (thr={args.threshold})")
        if args.verbose:
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
