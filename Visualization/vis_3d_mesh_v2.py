import torch
import numpy as np
import os
import torch.nn.functional as F
import cv2
import timm
import torch.nn as nn
import plotly.graph_objects as go
from skimage import measure

# --- 1. 定义模型 (保持不变) ---
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
OUTPUT_DIR = "vis_3d_mesh_results_labeled" # 结果保存在新文件夹
TARGET_PATIENTS = ["sub-strokecase0011", "sub-strokecase0168"]

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    if std == 0: return img
    return (img - mean) / (std + 1e-8)

def volume_to_mesh(vol, color='red', opacity=0.5, name='Mesh'):
    try:
        # level=0.5 在 0 和 1 之间取等值面
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
        x, y, z = verts.T
        i, j, k = faces.T
        
        return go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=name,
            showlegend=False, # 关闭Mesh自带的图例(通常很难看)，我们用下面的Dummy Scatter代替
            lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=0.1, fresnel=0.2)
        )
    except Exception as e:
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting High-Contrast 3D Reconstruction on {device}...")

    model = MAESegmenter(num_classes=1).to(device)
    if not os.path.exists(CKPT_PATH):
        print("❌ Checkpoint not found.")
        return
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    for case_name in TARGET_PATIENTS:
        path = os.path.join(DATA_ROOT, f"{case_name}.npz")
        if not os.path.exists(path): continue
            
        print(f"\n🧠 Processing {case_name}...")
        data = np.load(path)
        dwi_vol = data['dwi']
        adc_vol = data['adc']
        mask_vol = data['mask']
        
        # 1. 生成预测
        pred_vol = np.zeros_like(mask_vol)
        num_slices = dwi_vol.shape[0]
        
        for z in range(num_slices):
            dwi, adc = dwi_vol[z], adc_vol[z]
            diff = dwi - adc
            img_stack = np.stack([normalize(dwi), normalize(adc), normalize(diff)], axis=0)
            img_tensor = torch.from_numpy(img_stack).float().unsqueeze(0).to(device)
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear')
            
            with torch.no_grad():
                logits = model(img_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            h, w = mask_vol.shape[1], mask_vol.shape[2]
            pred_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_vol[z] = (pred_resized > 0.5).astype(float)

        # 2. 生成大脑轮廓
        brain_vol = (dwi_vol > 10).astype(float) 

        plots = []

        # --- 🎨 颜色修改区域 START ---
        
        # A. 大脑轮廓: 保持浅灰色，但透明度极低，作为背景参照
        mesh_brain = volume_to_mesh(brain_vol, color='lightgray', opacity=0.05, name='Brain')
        if mesh_brain: plots.append(mesh_brain)

        # B. Ground Truth: 改用【亮绿色】，透明度调低一点，让它像个"幽灵外壳"
        mesh_gt = volume_to_mesh(mask_vol, color='lime', opacity=0.25, name='Ground Truth')
        if mesh_gt: plots.append(mesh_gt)

        # C. AI Prediction: 改用【深红色】，透明度高一点，像"实体核心"
        # 这样如果在内部重叠，能看清红色在绿色里面；如果在外部，红色会很扎眼
        mesh_pred = volume_to_mesh(pred_vol, color='crimson', opacity=0.5, name='AI Prediction')
        if mesh_pred: plots.append(mesh_pred)

        # D. 更新图例 (Legend) 的颜色点
        plots.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                                  marker=dict(size=10, color='lime'), name='🟢 Ground Truth (Doctor)'))
        plots.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                                  marker=dict(size=10, color='crimson'), name='🔴 AI Prediction (Model)'))
        plots.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                                  marker=dict(size=10, color='lightgray'), name='⚪ Brain Contour'))
                                  
        # --- 🎨 颜色修改区域 END ---

        # 3. 设置布局
        layout = go.Layout(
            title=f"3D Analysis: {case_name}",
            scene=dict(
                xaxis=dict(title="X", showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(title="Y", showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(
                    title="<b>Slice Index (Z)</b>", 
                    showgrid=True, gridcolor='lightgray', zeroline=False, showticklabels=True
                ),
                aspectmode='data'
            ),
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05, bgcolor="rgba(255, 255, 255, 0.8)"),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig = go.Figure(data=plots, layout=layout)
        
        # 保存到原文件名，直接覆盖或者加个后缀
        save_path = os.path.join(OUTPUT_DIR, f"Mesh_v3_HighContrast_{case_name}.html")
        fig.write_html(save_path)
        print(f"✅ Saved High-Contrast mesh to: {save_path}")

if __name__ == "__main__":
    main()
