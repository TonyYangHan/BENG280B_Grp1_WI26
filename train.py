import matplotlib
matplotlib.use('Agg') 
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MAESegmenter
from data import IslesNpzDataset
from loss import CombinedLoss

# --- 队友的新路径 ---
DATA_ROOT = "../ISLES-2022-npz-multimodal_clean/all"
TRAIN_LIST = "../ISLES-2022-npz-multimodal_clean/splits/train.txt"
VAL_LIST = "../ISLES-2022-npz-multimodal_clean/splits/val.txt"
OUTPUT_DIR = "../runs/terence_strategy"

BATCH_SIZE = 16 
EPOCHS = 30
LR = 1e-4

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = IslesNpzDataset(DATA_ROOT, TRAIN_LIST)
    val_ds = IslesNpzDataset(DATA_ROOT, VAL_LIST)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    model = MAESegmenter(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = CombinedLoss()

    best_dice = 0.0
    epoch_metrics = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_loss_sum = 0
        val_dice_sum = 0
        val_iou_sum = 0
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs, v_masks = v_imgs.to(device), v_masks.to(device)
                v_out = model(v_imgs)
                v_loss = criterion(v_out, v_masks)
                pred = (torch.sigmoid(v_out) > 0.5).float()
                intersection = (pred * v_masks).sum()
                pred_sum = pred.sum()
                target_sum = v_masks.sum()
                dice = (2. * intersection) / (pred_sum + target_sum + 1e-8)
                iou = intersection / (pred_sum + target_sum - intersection + 1e-8)
                val_loss_sum += v_loss.item()
                val_dice_sum += dice.item()
                val_iou_sum += iou.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)

        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "dice": avg_val_dice,
            "iou": avg_val_iou,
        })

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}"
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_terence.pth"))
            print(f"Saved Best Model (Dice: {best_dice:.4f})")

    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved training metrics CSV to {metrics_csv_path}")

if __name__ == "__main__":
    main()
