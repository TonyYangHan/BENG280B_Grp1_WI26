import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # <--- 用于解决大小不一的关键
import os
import torch.nn.functional as F
import cv2
from model import MAESegmenter

# --- 配置 ---
CKPT_PATH = "runs/terence_strategy/best_terence.pth"
DATA_ROOT = "/mnt/c/Users/81493/OneDrive/Desktop/BENG280B_Project/all_npz/all"
# 保持原文件夹，运行即覆盖 (Requirement 3)
OUTPUT_DIR = "vis_results_full_patient" 
TARGET_PATIENTS = ["sub-strokecase0011", "sub-strokecase0168"]

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    if std == 0: return img
    return (img - mean) / (std + 1e-8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Optimized Visualization on {device}...")

    # 加载模型
    model = MAESegmenter(num_classes=1).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    for case_name in TARGET_PATIENTS:
        path = os.path.join(DATA_ROOT, f"{case_name}.npz")
        if not os.path.exists(path): continue
            
        print(f"🔍 Processing {case_name}...")
        patient_dir = os.path.join(OUTPUT_DIR, case_name)
        os.makedirs(patient_dir, exist_ok=True)

        data = np.load(path)
        dwi_vol, mask_vol = data['dwi'], data['mask']
        adc_vol = data['adc']

        for z in range(dwi_vol.shape[0]):
            # 数据准备
            dwi, adc = dwi_vol[z], adc_vol[z]
            diff = dwi - adc
            img_stack = np.stack([normalize(dwi), normalize(adc), normalize(diff)], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
            input_224 = F.interpolate(img_tensor, size=(224, 224), mode='bilinear')

            # 推理
            with torch.no_grad():
                logits = model(input_224)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # --- 核心修订：绘图部分 ---
            # 1. 设置画布，调小高度以适配紧凑布局
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            
            # 基础图样处理
            viz_dwi = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
            viz_gt = cv2.resize(mask_vol[z], (224, 224), interpolation=cv2.INTER_NEAREST)
            viz_pred = (prob > 0.5).astype(np.float32)

            # Subplot 1: DWI
            axes[0].imshow(viz_dwi, cmap='gray')
            axes[0].set_title(f"Slice {z}: Input DWI", fontsize=12)
            axes[0].axis('off')

            # Subplot 2: GT
            axes[1].imshow(viz_gt, cmap='bone')
            axes[1].set_title("Ground Truth Mask", fontsize=12)
            axes[1].axis('off')

            # Subplot 3: AI Binary Pred
            axes[2].imshow(viz_pred, cmap='bone')
            axes[2].set_title("AI Prediction (>0.5)", fontsize=12)
            axes[2].axis('off')

            # Subplot 4: Heatmap (解决 Requirement 1 & 2)
            im = axes[3].imshow(prob, cmap='jet', vmin=0, vmax=1)
            axes[3].set_title("Probability Heatmap", fontsize=12)
            axes[3].axis('off')
            
            # 关键：手动为第四张图添加颜色条，而不影响其主图大小
            divider = make_axes_locatable(axes[3])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

            # 调整间距：wspace 控制水平间距，0.05 非常紧凑 (Requirement 2)
            plt.subplots_adjust(wspace=0.05, left=0.02, right=0.95, top=0.9, bottom=0.1)

            # 保存 (Requirement 3: 路径相同会直接覆盖旧图)
            save_path = os.path.join(patient_dir, f"slice_{z:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
            plt.close()

    print(f"✅ All new visualizations saved in {OUTPUT_DIR}. Old files have been replaced.")

if __name__ == "__main__":
    main()