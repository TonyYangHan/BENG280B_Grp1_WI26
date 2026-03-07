import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
try:
	from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
	def tqdm(iterable=None, **kwargs):
		return iterable

from model import MAESegmenter, FrozenViTMLPClassifier


MIN_NONZERO_RATIO = 0.10


def normalize_channel(x: np.ndarray) -> np.ndarray:
	x = x.astype(np.float32, copy=False)
	mu = float(x.mean())
	sd = float(x.std())
	if sd == 0:
		return x
	return (x - mu) / (sd + 1e-8)


def load_case(npz_path: Path):
	d = np.load(npz_path, allow_pickle=False)
	dwi = d["dwi"].astype(np.float32)
	adc = d["adc"].astype(np.float32)
	mask = d["mask"].astype(np.uint8)
	return dwi, adc, mask


def build_input(dwi_vol: np.ndarray, adc_vol: np.ndarray, z_idx: int):
	dwi = dwi_vol[z_idx]
	adc = adc_vol[z_idx]
	diff = dwi - adc

	# Slice eligibility is based on non-zero ratio from DWI input only.
	nonzero_ratio = float(np.count_nonzero(dwi)) / float(dwi.size)

	dwi = normalize_channel(dwi)
	adc = normalize_channel(adc)
	diff = normalize_channel(diff)

	x_np = np.stack([dwi, adc, diff], axis=0)
	return x_np, nonzero_ratio


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
	xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
	xt = F.interpolate(xt, size=(out_size, out_size), mode="bilinear", align_corners=False)

	with torch.no_grad():
		logits = model(xt)
		probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
		pred = (probs > float(threshold)).astype(np.uint8)

	return pred


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray):
	pred = (pred_mask > 0).astype(np.uint8)
	gt = (gt_mask > 0).astype(np.uint8)
	intersection = np.logical_and(pred, gt).sum()
	pred_sum = pred.sum()
	gt_sum = gt.sum()
	dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
	return float(dice)


def read_case_basenames(txt_path: Path):
	names = []
	with txt_path.open("r", encoding="utf-8") as f:
		for line in f:
			name = line.strip()
			if not name or name.startswith("#"):
				continue
			names.append(name)
	return names


def resolve_npz_path(npz_dir: Path, basename: str):
	p = npz_dir / basename
	if p.suffix != ".npz":
		p = p.with_suffix(".npz")
	return p


def summarize_case_dice(case_dice_values):
	if not case_dice_values:
		return None
	return float(np.mean(case_dice_values))


def compute_mean_and_se(values):
	arr = np.array(values, dtype=np.float64)
	mean = float(arr.mean())
	if arr.size <= 1:
		se = 0.0
	else:
		se = float(arr.std(ddof=1) / np.sqrt(arr.size))
	median = float(np.median(arr))
	return mean, se, median


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("case_list_txt", type=str, help=".txt file containing one npz basename per line")
	ap.add_argument("npz_dir", type=str, help="Directory containing npz files")
	ap.add_argument("checkpoint", type=str, help="Path to trained MAESegmenter checkpoint")
	ap.add_argument("--cls_ckpt", type=str, default=None, help="Optional checkpoint of trained binary slice classifier")
	ap.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu or cuda:0")
	ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")
	args = ap.parse_args()

	device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

	case_list_txt = Path(args.case_list_txt)
	npz_dir = Path(args.npz_dir)

	model, out_size = prepare_model(args.checkpoint, device)
	classifier = prepare_classifier(args.cls_ckpt, device) if args.cls_ckpt else None
	if classifier is not None:
		print("Classifier-gated inference enabled.")

	basenames = read_case_basenames(case_list_txt)
	if not basenames:
		print(f"No valid basenames found in {case_list_txt}.")
		return

	per_case_mean_dice = []
	all_slice_dice_scores = []

	for basename in tqdm(basenames, desc="Cases", unit="case"):
		npz_path = resolve_npz_path(npz_dir, basename)
		if not npz_path.exists():
			print(f"[SKIP] Missing file: {npz_path}")
			continue

		dwi_vol, adc_vol, mask = load_case(npz_path)

		case_dice_scores = []
		eligible_slices = 0

		for z_idx in tqdm(range(mask.shape[0]), desc=f"Slices ({npz_path.name})", unit="slice", leave=False):
			x_np, nonzero_ratio = build_input(dwi_vol, adc_vol, z_idx)
			y = (mask[z_idx] > 0).astype(np.uint8)

			# Keep only lesion-positive slices with enough DWI support.
			if nonzero_ratio <= MIN_NONZERO_RATIO or not np.any(y):
				continue

			eligible_slices += 1

			low, high = 0.1, 0.4
			if classifier is not None:
				cls_prob = predict_classifier_prob(classifier, x_np, device, out_size)
				if cls_prob >= high:
					seg_threshold = 0.5
					pred = predict_slice(model, x_np, device, out_size, seg_threshold)
				elif cls_prob > low:
					seg_threshold = 0.9
					pred = predict_slice(model, x_np, device, out_size, seg_threshold)
				else:
					pred = np.zeros((out_size, out_size), dtype=np.uint8)
			else:
				pred = predict_slice(model, x_np, device, out_size, args.threshold)

			y_t = torch.from_numpy(y[None, None].astype(np.float32))
			y_t = F.interpolate(y_t, size=(out_size, out_size), mode="nearest")[0, 0].numpy().astype(np.uint8)

			dice = compute_dice(pred, y_t)
			case_dice_scores.append(dice)
			all_slice_dice_scores.append(dice)

		case_mean_dice = summarize_case_dice(case_dice_scores)
		if case_mean_dice is None:
			print(f"[SKIP] {npz_path.name}: no eligible slices (need DWI non-zero ratio > {MIN_NONZERO_RATIO:.2f} and lesion-positive GT)")
			continue

		per_case_mean_dice.append(case_mean_dice)
		print(f"[CASE] {npz_path.name}: avg_dice={case_mean_dice:.4f} over {eligible_slices} eligible slice(s)")

	if not per_case_mean_dice:
		print("No cases produced eligible Dice scores.")
		return

	mean_case_dice, se_case_dice, median_case_dice = compute_mean_and_se(per_case_mean_dice)
	print("\nSummary across npz cases:")
	print(f"Case-average Dice mean: {mean_case_dice:.4f}")
	print(f"Case-average Dice standard error: {se_case_dice:.4f}")
	print(f"Case-average Dice median: {median_case_dice:.4f}")
	print(f"Cases included: {len(per_case_mean_dice)}")

	if all_slice_dice_scores:
		mean_slice_dice, se_slice_dice, median_slice_dice = compute_mean_and_se(all_slice_dice_scores)
		print("\nSummary across all eligible slices:")
		print(f"All-slice Dice mean: {mean_slice_dice:.4f}")
		print(f"All-slice Dice standard error: {se_slice_dice:.4f}")
		print(f"All-slice Dice median: {median_slice_dice:.4f}")
		print(f"Eligible slices included: {len(all_slice_dice_scores)}")


if __name__ == "__main__":
	main()
