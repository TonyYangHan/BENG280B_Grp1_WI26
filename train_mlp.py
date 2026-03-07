import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import FrozenViTMLPClassifier
from data import IslesAllSlicesDataset
from loss import ClassificationLoss
from utils import compute_accuracy, compute_roc_auc, load_yaml_config, deep_get

CONFIG_PATH = "config_mlp.yaml"
# Defaults (can be overridden via config)
DATA_ROOT = "../ISLES-2022-npz-multimodal_clean/all"
TRAIN_LIST = "../ISLES-2022-npz-multimodal_clean/splits/train.txt"
VAL_LIST = "../ISLES-2022-npz-multimodal_clean/splits/val.txt"
OUTPUT_DIR = "../runs/mlp_slice_classifier"

def main():
    cfg = load_yaml_config(CONFIG_PATH)
    data_root = deep_get(cfg, "data.root", DATA_ROOT)
    train_list = deep_get(cfg, "data.train_list", TRAIN_LIST)
    val_list = deep_get(cfg, "data.val_list", VAL_LIST)
    output_dir = deep_get(cfg, "train.output_dir", OUTPUT_DIR)
    img_size = deep_get(cfg, "data.img_size", 224)

    batch_size = deep_get(cfg, "train.batch_size", 32)
    val_batch_size = deep_get(cfg, "train.val_batch_size", batch_size)
    epochs = deep_get(cfg, "train.epochs", 30)
    lr = deep_get(cfg, "train.lr", 1e-4)
    weight_decay = deep_get(cfg, "train.weight_decay", 1e-4)
    num_workers = deep_get(cfg, "train.num_workers", 8)
    pin_memory = bool(deep_get(cfg, "train.pin_memory", True))

    hidden_dims = deep_get(cfg, "model.hidden_dims", [512, 256])
    dropout = deep_get(cfg, "model.dropout", 0.3)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Datasets
    # Note: We reuse IslesNpzDataset. It returns (img, mask).
    # We will compute the classification label inside the loop dynamically
    # based on whether the mask has any non-zero pixels.
    # Use the new dataset that indexes all valid slices
    train_ds = IslesAllSlicesDataset(data_root, train_list, img_size=img_size)
    val_ds = IslesAllSlicesDataset(data_root, val_list, img_size=img_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Initialize classifier that consumes frozen ViT latents
    model = FrozenViTMLPClassifier(hidden_dims=hidden_dims, num_classes=1, dropout=dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ClassificationLoss()

    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            # Create classification target: 1 if user has lesion in this slice (sum > 0), else 0
            # dataset returns masks of shape [B, 1, H, W]
            targets = (masks.view(masks.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            acc = compute_accuracy(logits, targets)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc.item():.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_probs = []
        val_targets_all = []
        
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs = v_imgs.to(device)
                v_targets = (v_masks.view(v_masks.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1).to(device)
                
                v_logits = model(v_imgs)
                loss = criterion(v_logits, v_targets)
                acc = compute_accuracy(v_logits, v_targets)

                probs = torch.sigmoid(v_logits).squeeze(1)
                val_probs.append(probs.detach().cpu().numpy())
                val_targets_all.append(v_targets.squeeze(1).detach().cpu().numpy())
                
                val_loss += loss.item()
                val_acc += acc.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        val_probs_np = np.concatenate(val_probs, axis=0) if val_probs else np.array([], dtype=np.float32)
        val_targets_np = np.concatenate(val_targets_all, axis=0) if val_targets_all else np.array([], dtype=np.float32)
        _, _, val_auc = compute_roc_auc(val_probs_np, val_targets_np)
        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} AUROC: {val_auc_str}"
        )

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_mlp.pth"))
            print(f"Saved Best MLP (Acc: {best_acc:.4f})")

if __name__ == "__main__":
    main()
