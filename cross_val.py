import os
import csv
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MAESegmenter
from loss import CombinedLoss
from utils import set_seed
from data import IslesNpzDataset


DATA_ROOT = "../ISLES-2022-npz-multimodal_clean/all"
OUTPUT_DIR = "../runs/terence_strategy_cv"

FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 224
SAMPLES_PER_PATIENT = 20
NUM_WORKERS = 12
SEED = 42


def list_npz_files(data_root: str):
    files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_root}")
    return sorted(files)


def make_folds(files, n_splits=FOLDS, seed=SEED):
    if len(files) < n_splits:
        raise ValueError(f"Need at least {n_splits} files for {n_splits}-fold CV, found {len(files)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(files)
    folds = np.array_split(shuffled, n_splits)
    return [fold.tolist() for fold in folds]


def write_case_list(case_names, output_path: str):
    with open(output_path, "w", encoding="utf-8") as handle:
        for case_name in case_names:
            handle.write(f"{case_name}\n")


def train_one_fold(fold_idx: int, train_list_path: str, val_list_path: str, device: torch.device):
    train_ds = IslesNpzDataset(DATA_ROOT, train_list_path, img_size=IMG_SIZE, samples_per_patient=SAMPLES_PER_PATIENT)
    val_ds = IslesNpzDataset(DATA_ROOT, val_list_path, img_size=IMG_SIZE, samples_per_patient=SAMPLES_PER_PATIENT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MAESegmenter(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = CombinedLoss()

    best_dice = 0.0
    best_iou = 0.0
    best_combined_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1}/{FOLDS} | Epoch {epoch+1}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs, v_masks = v_imgs.to(device), v_masks.to(device)
                v_out = model(v_imgs)
                v_loss = criterion(v_out, v_masks)
                pred = (torch.sigmoid(v_out) > 0.5).float()
                intersection = (pred * v_masks).sum()
                pred_sum = pred.sum()
                target_sum = v_masks.sum()
                dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
                iou = intersection / (pred_sum + target_sum - intersection + 1e-8)

                val_loss_sum += v_loss.item()
                val_dice_sum += dice.item()
                val_iou_sum += iou.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        avg_val_dice = val_dice_sum / max(len(val_loader), 1)
        avg_val_iou = val_iou_sum / max(len(val_loader), 1)

        if avg_val_loss < best_combined_loss:
            best_combined_loss = avg_val_loss
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou

        print(
            f"Fold {fold_idx+1} | Loss: {avg_train_loss:.4f} | "
            f"Val CombinedLoss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}"
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_terence.pth"))
            print(f"Saved fold {fold_idx+1} best model (Dice: {best_dice:.4f})")

    return {
        "fold": fold_idx + 1,
        "best_combined_loss": best_combined_loss,
        "best_dice": best_dice,
        "best_iou": best_iou,
    }


def save_fold_metrics_csv(metrics_rows, output_path: str):
    fieldnames = ["fold", "best_combined_loss", "best_dice", "best_iou"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_dir = os.path.join(OUTPUT_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = list_npz_files(DATA_ROOT)
    all_cases = [os.path.splitext(os.path.basename(path))[0] for path in all_files]
    folds = make_folds(all_cases, n_splits=FOLDS, seed=SEED)

    fold_metrics = []
    for idx in range(FOLDS):
        val_cases = folds[idx]
        train_cases = [case for j, fold in enumerate(folds) if j != idx for case in fold]

        train_list_path = os.path.join(split_dir, f"fold_{idx+1}_train.txt")
        val_list_path = os.path.join(split_dir, f"fold_{idx+1}_val.txt")
        write_case_list(train_cases, train_list_path)
        write_case_list(val_cases, val_list_path)

        print(f"Starting fold {idx+1}: {len(train_cases)} train files, {len(val_cases)} val files")
        fold_result = train_one_fold(idx, train_list_path, val_list_path, device)
        fold_metrics.append(fold_result)

    csv_path = os.path.join(OUTPUT_DIR, "fold_metrics.csv")
    save_fold_metrics_csv(fold_metrics, csv_path)

    mean_dice = float(np.mean([row["best_dice"] for row in fold_metrics])) if fold_metrics else 0.0
    print(f"5-fold CV complete. Mean best Dice: {mean_dice:.4f}")
    print(f"Saved fold metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
