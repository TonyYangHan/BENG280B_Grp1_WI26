import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import cv2
import timm
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# --- 1. 模型定义 (保持不变) ---
class ConvDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.upsample(x)

class MAESegmenter(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=3, global_pool='')
        self.decoder = ConvDecoder(embed_dim=768, num_classes=num_classes)
    def forward(self, x):
        features = self.encoder.forward_features(x)[:, 1:, :]
        B, N, C = features.shape
        H = W = int(N**0.5)
        features = features.transpose(1, 2).reshape(B, C, H, W)
        return self.decoder(features)

# --- 2. 配置 ---
CKPT_PATH = "runs/terence_strategy/best_terence.pth"
DATA_ROOT = "/mnt/c/Users/81493/OneDrive/Desktop/BENG280B_Project/all_npz/all"
OUTPUT_DIR = "vis_quant_metrics_results"
# 指定你要跑的两个病人
TARGET_PATIENTS = ["sub-strokecase0011", "sub-strokecase0168"] 

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    if std == 0: return img
    return (img - mean) / (std + 1e-8)

def calculate_metrics(pred_prob, target_mask):
    """
    计算单张切片的指标
    pred_prob: (H, W) 0~1 float
    target_mask: (H, W) 0 or 1 float
    """
    # 二值化预测
    pred_bin = (pred_prob > 0.5).astype(np.float32)
    
    # Flatten
    p = pred_bin.flatten()
    t = target_mask.flatten()
    prob_flat = pred_prob.flatten()
    
    # 1. Dice Score
    intersection = (p * t).sum()
    dice = (2. * intersection) / (p.sum() + t.sum() + 1e-8)
    
    # 2. IoU (Jaccard)
    union = p.sum() + t.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    # 3. AUROC
    # 如果 Ground Truth 全是 0 (没病灶) 或 全是 1，AUROC 无定义
    if t.sum() == 0 or t.sum() == len(t):
        auc = np.nan # 标记为无效
    else:
        try:
            auc = roc_auc_score(t, prob_flat)
        except ValueError:
            auc = np.nan

    return dice, iou, auc

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Calculating Metrics on {device}...")

    # 加载模型
    model = MAESegmenter(num_classes=1).to(device)
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Error: Weights not found at {CKPT_PATH}")
        return
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    for case_name in TARGET_PATIENTS:
        fname = f"{case_name}.npz"
        path = os.path.join(DATA_ROOT, fname)
        
        if not os.path.exists(path):
            print(f"⚠️ Missing {path}, skipping.")
            continue
            
        print(f"\n📊 Analyzing {case_name}...")
        patient_dir = os.path.join(OUTPUT_DIR, case_name)
        os.makedirs(patient_dir, exist_ok=True)

        data = np.load(path)
        dwi_vol = data['dwi']
        adc_vol = data['adc']
        mask_vol = data['mask']
        num_slices = dwi_vol.shape[0]

        # 记录每层的指标，最后算个平均
        metrics_log = []

        for z in range(num_slices):
            # --- 1. 准备数据 ---
            dwi = dwi_vol[z]
            adc = adc_vol[z]
            msk = mask_vol[z]
            diff = dwi - adc
            
            img_stack = np.stack([normalize(dwi), normalize(adc), normalize(diff)], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

            # --- 2. 推理 ---
            with torch.no_grad():
                logits = model(img_tensor)
                prob_map = torch.sigmoid(logits).squeeze().cpu().numpy() # 224x224
            
            # 为了计算指标，我们需要把 GT 也缩放到 224 或者把 Pred 缩放回原图
            # 这里统一缩放到 224 计算，方便显示
            mask_resized = cv2.resize(msk, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            # --- 3. 计算指标 ---
            dice, iou, auc = calculate_metrics(prob_map, mask_resized)
            
            # 只有当 Mask 有病灶时，Dice/IoU 才有意义用于评估“分割能力”
            # 如果 Mask 是空的，Dice 通常是 0 (除非预测也是空)，这会拉低平均分
            has_lesion = mask_resized.sum() > 0
            if has_lesion:
                metrics_log.append([dice, iou, auc])
                auc_str = f"{auc:.3f}"
            else:
                auc_str = "N/A" # 空切片没有 AUC

            # --- 4. 可视化 ---
            # 准备显示用的图
            viz_dwi = cv2.resize(dwi, (224, 224), interpolation=cv2.INTER_LINEAR)
            viz_dwi = (viz_dwi - viz_dwi.min()) / (viz_dwi.max() - viz_dwi.min() + 1e-8)
            pred_mask = (prob_map > 0.5).astype(np.float32)

            plt.figure(figsize=(16, 5))
            
            # Col 1: DWI
            plt.subplot(1, 4, 1)
            plt.imshow(viz_dwi, cmap='gray')
            plt.title(f"Slice {z}: DWI Input")
            plt.axis('off')

            # Col 2: Ground Truth
            plt.subplot(1, 4, 2)
            plt.imshow(mask_resized, cmap='bone')
            plt.title("Ground Truth")
            plt.axis('off')

            # Col 3: AI Pred + Metrics Title
            plt.subplot(1, 4, 3)
            plt.imshow(pred_mask, cmap='bone')
            # 关键：把分数写在标题里
            if has_lesion:
                title_str = f"Pred (>0.5)\nDice: {dice:.3f} | IoU: {iou:.3f}"
            else:
                title_str = f"Pred (>0.5)\n(No Lesion in GT)"
            plt.title(title_str, color='red' if has_lesion and dice < 0.5 else 'black')
            plt.axis('off')
            
            # Col 4: Heatmap
            plt.subplot(1, 4, 4)
            plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(label="Prob")
            plt.title(f"Heatmap\nAUROC: {auc_str}")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(patient_dir, f"slice_{z:03d}.png"), dpi=100)
            plt.close()

            # 打印到终端
            if has_lesion:
                print(f"   Slice {z:03d}: Dice={dice:.4f}, IoU={iou:.4f}, AUC={auc_str}")
        
        # --- 打印这个病人的平均分 ---
        if len(metrics_log) > 0:
            avg_metrics = np.nanmean(np.array(metrics_log), axis=0)
            print(f"\n🏆 Patient {case_name} Average (Lesion Slices Only):")
            print(f"   Dice: {avg_metrics[0]:.4f}")
            print(f"   IoU : {avg_metrics[1]:.4f}")
            print(f"   AUC : {avg_metrics[2]:.4f}")
        else:
            print(f"\n⚠️ Patient {case_name} has no visible lesion slices.")

    print(f"\n✅ Visualization with metrics saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
