import torch
import numpy as np
import os
import torch.nn.functional as F
import cv2
import timm
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots # 用于分屏对比
from skimage import measure
import torchvision.transforms.functional as TF

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
OUTPUT_DIR = "vis_3d_noise_compare_results"
TARGET_PATIENTS = ["sub-strokecase0011", "sub-strokecase0168"]

# --- 噪声参数 (和你之前的实验保持一致) ---
NOISE_FACTOR = 0.2
BLUR_KERNEL = 5
BLUR_SIGMA = 1.5

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    if std == 0: return img
    return (img - mean) / (std + 1e-8)

def add_noise_and_blur(img_tensor):
    """同时添加高斯模糊和高斯噪声"""
    # 1. Blur
    noisy = TF.gaussian_blur(img_tensor, kernel_size=BLUR_KERNEL, sigma=BLUR_SIGMA)
    # 2. Noise
    noise = torch.randn_like(noisy) * NOISE_FACTOR
    noisy = noisy + noise
    return noisy

def volume_to_mesh(vol, color='red', opacity=0.5, name='Mesh'):
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
        x, y, z = verts.T
        i, j, k = faces.T
        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=name,
            showlegend=True, # 开启图例
            lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=0.1, fresnel=0.2)
        )
    except Exception as e: return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting 3D Noise Robustness Comparison on {device}...")

    model = MAESegmenter(num_classes=1).to(device)
    if not os.path.exists(CKPT_PATH):
        print("❌ Checkpoint not found.")
        return
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    for case_name in TARGET_PATIENTS:
        path = os.path.join(DATA_ROOT, f"{case_name}.npz")
        if not os.path.exists(path): continue
            
        print(f"\n🧠 Processing {case_name} (Clean vs Noisy)...")
        data = np.load(path)
        dwi_vol = data['dwi']
        adc_vol = data['adc']
        mask_vol = data['mask'] # GT
        
        # 准备两个空的预测体积
        pred_vol_clean = np.zeros_like(mask_vol)
        pred_vol_noisy = np.zeros_like(mask_vol)
        
        num_slices = dwi_vol.shape[0]
        
        # 逐层推理
        for z in range(num_slices):
            dwi, adc = dwi_vol[z], adc_vol[z]
            diff = dwi - adc
            img_stack = np.stack([normalize(dwi), normalize(adc), normalize(diff)], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
            
            # Resize
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear')
            
            # --- 分支 A: Clean Input ---
            with torch.no_grad():
                logits_c = model(img_tensor)
                prob_c = torch.sigmoid(logits_c).squeeze().cpu().numpy()
            
            # --- 分支 B: Noisy Input ---
            noisy_tensor = add_noise_and_blur(img_tensor)
            with torch.no_grad():
                logits_n = model(noisy_tensor)
                prob_n = torch.sigmoid(logits_n).squeeze().cpu().numpy()

            # Resize back & Binarize
            h, w = mask_vol.shape[1], mask_vol.shape[2]
            
            pred_c_resized = cv2.resize(prob_c, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_vol_clean[z] = (pred_c_resized > 0.5).astype(float)
            
            pred_n_resized = cv2.resize(prob_n, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_vol_noisy[z] = (pred_n_resized > 0.5).astype(float)

        # --- 生成 3D Meshes ---
        brain_vol = (dwi_vol > 10).astype(float) 

        # 创建左右分屏的 Figure
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=(f"Before Noise (Original)", f"After Noise (Blur+Gaussian)")
        )

        # 定义通用元素 (Brain & GT) - 左右两边都加
        # 风格：v2 版本 (Brain=LightGray, GT=Blue, Pred=Red)
        
        for i, col_idx in enumerate([1, 2]):
            scene_str = f"scene{'' if i==0 else i+1}"
            show_legend = (i == 0) # 只在左边显示一次图例，避免重复

            # 1. Brain Contour
            mesh_brain = volume_to_mesh(brain_vol, color='lightgray', opacity=0.05, name='Brain Contour')
            if mesh_brain: 
                mesh_brain.showlegend = False
                fig.add_trace(mesh_brain, row=1, col=col_idx)

            # 2. Ground Truth (Blue)
            mesh_gt = volume_to_mesh(mask_vol, color='blue', opacity=0.3, name='Ground Truth (Doctor)')
            if mesh_gt: 
                mesh_gt.showlegend = show_legend
                fig.add_trace(mesh_gt, row=1, col=col_idx)

        # --- 添加 Prediction (Red) ---
        
        # 左边: Clean Prediction
        mesh_pred_clean = volume_to_mesh(pred_vol_clean, color='red', opacity=0.5, name='AI Prediction')
        if mesh_pred_clean:
            mesh_pred_clean.showlegend = True
            fig.add_trace(mesh_pred_clean, row=1, col=1)
            
        # 右边: Noisy Prediction
        mesh_pred_noisy = volume_to_mesh(pred_vol_noisy, color='red', opacity=0.5, name='AI Prediction (Noisy)')
        if mesh_pred_noisy:
            mesh_pred_noisy.showlegend = False # 颜色一样，不需要重复图例
            fig.add_trace(mesh_pred_noisy, row=1, col=2)

        # --- 设置相机和布局 ---
        camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        
        fig.update_layout(
            title=f"Robustness Test: {case_name}",
            scene1=dict(
                camera=camera, aspectmode='data',
                xaxis=dict(title="X", visible=False), yaxis=dict(title="Y", visible=False),
                zaxis=dict(title="Z (Slices)", visible=True)
            ),
            scene2=dict(
                camera=camera, aspectmode='data',
                xaxis=dict(title="X", visible=False), yaxis=dict(title="Y", visible=False),
                zaxis=dict(title="Z (Slices)", visible=True)
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        save_path = os.path.join(OUTPUT_DIR, f"Compare_Noise_{case_name}.html")
        fig.write_html(save_path)
        print(f"✅ Saved comparison to: {save_path}")

if __name__ == "__main__":
    main()