import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import MAESegmenter, FrozenViTMLPClassifier


MIN_NONZERO_RATIO = 0.10


def normalize_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mu = float(x.mean())
    sd = float(x.std())
    if sd == 0:
        return x
    return (x - mu) / (sd + 1e-8)


def load_case(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    dwi = d["dwi"].astype(np.float32)
    adc = d["adc"].astype(np.float32)
    mask = d["mask"].astype(np.uint8)
    return dwi, adc, mask


def build_input(dwi_vol: np.ndarray, adc_vol: np.ndarray, z_idx: int):
    dwi = dwi_vol[z_idx]
    adc = adc_vol[z_idx]
    diff = dwi - adc

    raw_stack = np.stack([dwi, adc, diff], axis=0)
    nonzero_ratio = float(np.count_nonzero(raw_stack)) / float(raw_stack.size)

    dwi = normalize_channel(dwi)
    adc = normalize_channel(adc)
    diff = normalize_channel(diff)

    x_np = np.stack([dwi, adc, diff], axis=0)
    return x_np, dwi_vol[z_idx], nonzero_ratio


def prepare_model(checkpoint_path: str, device):
    out_size = 224
    model = MAESegmenter(num_classes=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, out_size


def _infer_classifier_hidden_dims(state_dict: dict):
    linear_weights = []
    for key, val in state_dict.items():
        if key.startswith("classifier.mlp") and key.endswith(".weight"):
            try:
                layer_idx = int(key.split(".")[2])
            except Exception:  # noqa: BLE001
                continue
            linear_weights.append((layer_idx, val))

    linear_weights.sort(key=lambda item: item[0])
    if len(linear_weights) <= 1:
        return [512, 256]
    return [int(w.shape[0]) for _, w in linear_weights[:-1]]


def prepare_classifier(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    hidden_dims = _infer_classifier_hidden_dims(state_dict)
    classifier = FrozenViTMLPClassifier(hidden_dims=hidden_dims, num_classes=1, dropout=0.0).to(device)
    classifier.load_state_dict(state_dict, strict=False)
    classifier.eval()
    return classifier


def predict_classifier_prob(classifier, x_np: np.ndarray, device, out_size: int):
    xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
    xt = F.interpolate(xt, size=(out_size, out_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        logit = classifier(xt)
        prob = torch.sigmoid(logit)[0, 0].item()
    return float(prob)


def predict_slice(model, x_np: np.ndarray, device, out_size: int, threshold: float):
    """Run model on a single slice stack and return prob map + binary mask."""
    xt = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1,C,H,W]
    xt = F.interpolate(xt, size=(out_size, out_size), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = model(xt)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (probs > float(threshold)).astype(np.uint8)

    return probs, pred


def visualize(slice_img, gt_mask, pred_mask, prob_map, out_path, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(gt_mask, cmap="Reds", alpha=0.35)
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(pred_mask, cmap="Blues", alpha=0.35)
    plt.title("Pred")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    heat = plt.imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("Probability")
    plt.axis("off")
    plt.colorbar(heat, fraction=0.046, pad=0.04)

    plt.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
    iou = intersection / (pred_sum + gt_sum - intersection + 1e-8)
    return float(dice), float(iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", type=str, help="Path to case npz")
    ap.add_argument("checkpoint", type=str, help="Path to trained MAESegmenter checkpoint (best_terence.pth)")
    ap.add_argument("out_dir", type=str, help="Directory to save visualizations")
    ap.add_argument("--cls_ckpt", type=str, default=None, help="Optional checkpoint of trained binary slice classifier")
    ap.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu or cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")
    ap.add_argument("--z", type=int, default=None, help="Slice index to visualize (default: middle slice)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model, out_size = prepare_model(args.checkpoint, device)
    classifier = prepare_classifier(args.cls_ckpt, device) if args.cls_ckpt else None
    dwi_vol, adc_vol, mask = load_case(args.npz_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose slices: only slices where raw input has >20% non-zero pixels
    if args.z is not None:
        _, _, nonzero_ratio = build_input(dwi_vol, adc_vol, int(args.z))
        target_slices = [int(args.z)] if nonzero_ratio > MIN_NONZERO_RATIO else []
    else:
        target_slices = []
        for z_idx in range(mask.shape[0]):
            _, _, nonzero_ratio = build_input(dwi_vol, adc_vol, z_idx)
            if nonzero_ratio > MIN_NONZERO_RATIO:
                target_slices.append(z_idx)

    if not target_slices:
        print(f"No slices with >{int(MIN_NONZERO_RATIO * 100)}% non-zero input pixels found; nothing to predict.")
        return

    if classifier is not None:
        print("Classifier-gated inference enabled.")

    saved = 0
    dice_scores = []
    iou_scores = []
    non_empty_gt_count = 0
    for z in target_slices:
        x_np, slice_img, _ = build_input(dwi_vol, adc_vol, z)
        y = (mask[z] > 0).astype(np.uint8)

        cls_prob = None
        low, high = 0.1, 0.4
        if classifier is not None:
            cls_prob = predict_classifier_prob(classifier, x_np, device, out_size)
            if cls_prob >= high:
                seg_threshold = 0.5
                probs, pred = predict_slice(model, x_np, device, out_size, seg_threshold)
            elif cls_prob > low:
                seg_threshold = 0.9
                probs, pred = predict_slice(model, x_np, device, out_size, seg_threshold)
            else:
                seg_threshold = None
                probs = np.zeros((out_size, out_size), dtype=np.float32)
                pred = np.zeros((out_size, out_size), dtype=np.uint8)
        else:
            seg_threshold = args.threshold
            probs, pred = predict_slice(model, x_np, device, out_size, seg_threshold)

        y_t = torch.from_numpy(y[None, None].astype(np.float32))
        y_t = F.interpolate(y_t, size=(out_size, out_size), mode="nearest")[0, 0].numpy().astype(np.uint8)

        if np.any(y_t):
            dice, iou = compute_dice_iou(pred, y_t)
            dice_scores.append(dice)
            iou_scores.append(iou)
            non_empty_gt_count += 1

        slice_img_resized = F.interpolate(
            torch.from_numpy(slice_img[None, None].astype(np.float32)),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        out_path = out_dir / (Path(args.npz_path).stem + f"_z{z:03d}_thr{args.threshold:.2f}.png")
        if classifier is not None:
            if seg_threshold is None:
                title = f"z={z} cls={cls_prob:.3f} -> empty"
            else:
                title = f"z={z} cls={cls_prob:.3f} thr={seg_threshold:.2f}"
        else:
            title = f"z={z} (thr={seg_threshold})"
        visualize(
            slice_img=slice_img_resized,
            gt_mask=y_t,
            pred_mask=pred,
            prob_map=probs,
            out_path=out_path,
            title=title
        )
        saved += 1

    print(f"Done. Saved {saved} slice(s) to {out_dir}.")
    if dice_scores:
        print(f"Metric slices used (non-empty GT): {non_empty_gt_count}")
        print(f"Average Dice across predicted slices: {np.mean(dice_scores):.4f}")
        print(f"Average IoU across predicted slices: {np.mean(iou_scores):.4f}")
    else:
        print("No non-empty GT slices among predicted slices; Dice/IoU not computed.")


if __name__ == "__main__":
    main()
