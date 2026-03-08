import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # <--- 解决尺寸一致的关键
import os
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF
from model import MAESegmenter

# --- 配置 ---
CKPT_PATH = "runs/terence_strategy/best_terence.pth"
DATA_ROOT = "/mnt/c/Users/81493/OneDrive/Desktop/BENG280B_Project/all_npz/all"
OUTPUT_DIR = "vis_noise_robustness2" # 新文件夹 (Requirement 2)
TARGET_PATIENTS = ["sub-strokecase0011", "sub-strokecase0168"]
SLICES_TO_VISUALIZE = 20

# 噪声参数保持不变
NOISE_FACTOR = 0.2
BLUR_KERNEL = 5
BLUR_SIGMA = 1.5

def add_noise_and_blur(img_tensor):
    noisy = TF.gaussian_blur(img_tensor, kernel_size=BLUR_KERNEL, sigma=BLUR_SIGMA)
    noise = torch.randn_like(noisy) * NOISE_FACTOR
    return noisy + noise

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    if std == 0: return img
    return (img - mean) / (std + 1e-8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Consistent-Size Noise Comparison on {device}...")

    model = MAESegmenter(num_classes=1).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    for case_name in TARGET_PATIENTS:
        path = os.path.join(DATA_ROOT, f"{case_name}.npz")
        if not os.path.exists(path): continue
            
        patient_dir = os.path.join(OUTPUT_DIR, case_name)
        os.makedirs(patient_dir, exist_ok=True)
        data = np.load(path)
        dwi_vol, adc_vol, mask_vol = data['dwi'], data['adc'], data['mask']
        
        # 挑选病灶明显的 20 张
        lesion_sizes = np.sum(mask_vol, axis=(1, 2))
        target_slices = np.argsort(lesion_sizes)[-SLICES_TO_VISUALIZE:]
        target_slices = sorted(target_slices)

        for z in target_slices:
            dwi, adc, msk = dwi_vol[z], adc_vol[z], mask_vol[z]
            img_stack = np.stack([normalize(dwi), normalize(adc), normalize(dwi-adc)], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
            clean_input = F.interpolate(img_tensor, size=(224, 224), mode='bilinear')
            noisy_input = add_noise_and_blur(clean_input)

            with torch.no_grad():
                logits_c = model(clean_input)
                prob_c = torch.sigmoid(logits_c).squeeze().cpu().numpy()
                logits_n = model(noisy_input)
                prob_n = torch.sigmoid(logits_n).squeeze().cpu().numpy()

            # 绘图逻辑
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 定义辅助函数处理 Colorbar
            def add_fixed_colorbar(ax, im):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)

            # 准备展示数据
            vis_clean_dwi = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
            vis_noisy_dwi = (noisy_input[0,0].cpu().numpy())
            vis_noisy_dwi = (vis_noisy_dwi - vis_noisy_dwi.min()) / (vis_noisy_dwi.max() - vis_noisy_dwi.min() + 1e-8)
            vis_gt = cv2.resize(msk, (224, 224), interpolation=cv2.INTER_NEAREST)
            diff_map = np.abs(prob_c - prob_n)

            # --- Row 1: Clean ---
            axes[0, 0].imshow(vis_clean_dwi, cmap='gray'); axes[0, 0].set_title("Original Input"); axes[0, 0].axis('off')
            
            im1 = axes[0, 1].imshow(prob_c, cmap='jet', vmin=0, vmax=1)
            axes[0, 1].set_title("Prediction (Clean)"); axes[0, 1].axis('off')
            add_fixed_colorbar(axes[0, 1], im1) # (Requirement 1: 挂载 Colorbar)

            axes[0, 2].imshow(vis_gt, cmap='bone'); axes[0, 2].set_title("Ground Truth"); axes[0, 2].axis('off')

            # --- Row 2: Noisy ---
            axes[1, 0].imshow(vis_noisy_dwi, cmap='gray'); axes[1, 0].set_title("Noisy Input"); axes[1, 0].axis('off')

            im2 = axes[1, 1].imshow(prob_n, cmap='jet', vmin=0, vmax=1)
            axes[1, 1].set_title("Prediction (Noisy)"); axes[1, 1].axis('off')
            add_fixed_colorbar(axes[1, 1], im2)

            im3 = axes[1, 2].imshow(diff_map, cmap='inferno', vmin=0, vmax=0.5)
            axes[1, 2].set_title("Difference (Robustness Error)"); axes[1, 2].axis('off')
            add_fixed_colorbar(axes[1, 2], im3)

            # 压缩布局 (Requirement 3: 其他不变)
            plt.tight_layout()
            plt.savefig(os.path.join(patient_dir, f"compare_slice_{z:03d}.png"), bbox_inches='tight')
            plt.close()
            
    print(f"✅ Success! Consistent sub-pictures saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()