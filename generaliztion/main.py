import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import SimpleITK as sitk
import timm


FOLD_ROOTS = [
    r"E:\picai_public_images_fold0",
    r"E:\picai_public_images_fold1",
    r"E:\picai_public_images_fold2",
    r"E:\picai_public_images_fold3",
    r"E:\picai_public_images_fold4",
]
LABEL_ROOT = r"E:\picai_labled"
OUTPUT_DIR = "runs/terence_strategy"

BATCH_SIZE  = 16
EPOCHS      = 60
LR          = 3e-5
VAL_RATIO   = 0.2
SEED        = 42
IMG_SIZE    = 224
EARLY_STOP  = 6

N_LESION_TRAIN  = 9999
N_HEALTHY_TRAIN = 400
N_LESION_VAL    = 100
N_HEALTHY_VAL   = 20


# ============================================================
# 1. MODEL
# ============================================================
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

    def forward(self, x):
        return self.upsample(x)


class MAESegmenter(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=True, in_chans=3, global_pool=''
        )
        self.decoder = ConvDecoder(embed_dim=768, num_classes=num_classes)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        features = features[:, 1:, :]
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features.transpose(1, 2).reshape(B, C, H, W)
        return self.decoder(features)


# ============================================================
# 2. LOSS
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=50.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.dice = DiceLoss()
        self.bce  = None
        self._bce_device = None

    def forward(self, logits, targets):
        if self.bce is None or logits.device != self._bce_device:
            self._bce_device = logits.device
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight_val]).to(logits.device)
            )
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


# ============================================================
# 3. DATASET
# ============================================================
class BalancedDataset(Dataset):
    """
    三通道输入: T2W + ADC + HBV
    fixed=True  → 验证集，初始化后固定不变
    fixed=False → 训练集，每次 resample() 重新随机抽
    """

    def __init__(self, lesion_samples, healthy_samples,
                 n_lesion=500, n_healthy=50,
                 img_size=IMG_SIZE, fixed=False):
        self.lesion_samples  = lesion_samples
        self.healthy_samples = healthy_samples
        self.n_lesion  = n_lesion
        self.n_healthy = n_healthy
        self.img_size  = img_size
        self.fixed     = fixed
        self.samples   = []
        self.resample()

    def resample(self):
        if self.fixed and len(self.samples) > 0:
            return
        n_l = min(self.n_lesion,  len(self.lesion_samples))
        n_h = min(self.n_healthy, len(self.healthy_samples))
        self.samples = (random.sample(self.lesion_samples,  n_l) +
                        random.sample(self.healthy_samples, n_h))
        random.shuffle(self.samples)
        label = '固定验证集' if self.fixed else '训练集'
        print(f"  ({label}): {n_l} + {n_h} = {n_l + n_h} ")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t2w_path, adc_path, hbv_path, lbl_path, slice_idx = self.samples[idx]

        def read_slice(path, s):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
            if arr.ndim == 4:
                arr = arr[0]
            s = min(s, arr.shape[0] - 1)
            return arr[s]

        def normalize(arr):
            std = arr.std()
            return (arr - arr.mean()) / (std + 1e-8) if std > 1e-8 else arr

        def resize(arr):
            img = sitk.GetImageFromArray(arr)
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize([self.img_size, self.img_size])
            orig_size    = img.GetSize()
            orig_spacing = img.GetSpacing()
            resampler.SetOutputSpacing([
                orig_spacing[0] * orig_size[0] / self.img_size,
                orig_spacing[1] * orig_size[1] / self.img_size,
            ])
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetOutputDirection(img.GetDirection())
            return sitk.GetArrayFromImage(resampler.Execute(img)).astype(np.float32)

        t2w = normalize(read_slice(t2w_path, slice_idx))
        adc = normalize(read_slice(adc_path, slice_idx))
        hbv = normalize(read_slice(hbv_path, slice_idx))

        lbl_arr = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))
        if lbl_arr.ndim == 4:
            lbl_arr = lbl_arr[0]
        s = min(slice_idx, lbl_arr.shape[0] - 1)
        lbl = (lbl_arr[s] > 0).astype(np.float32)

        t2w_r = resize(t2w)
        adc_r = resize(adc)
        hbv_r = resize(hbv)
        lbl_r = resize(lbl)

        img_3ch = np.stack([t2w_r, adc_r, hbv_r], axis=0)

        return (torch.from_numpy(img_3ch).float(),
                torch.from_numpy(lbl_r).unsqueeze(0).float())


# ============================================================
# 4. scan
# ============================================================
def build_sample_list(fold_roots, label_root):
    label_index = {}
    for fname in os.listdir(label_root):
        if fname.endswith(".nii.gz"):
            stem = fname.replace(".nii.gz", "")
        elif fname.endswith(".nii"):
            stem = fname.replace(".nii", "")
        else:
            continue
        label_index[stem] = os.path.join(label_root, fname)

    stem_to_mhas = {}
    for fold in fold_roots:
        for root, _, files in os.walk(fold):
            for fname in files:
                if not fname.endswith(".mha"):
                    continue
                parts = fname.replace(".mha", "").rsplit("_", 1)
                if len(parts) != 2:
                    continue
                stem, series = parts[0], parts[1].lower()
                if stem not in label_index:
                    continue
                if stem not in stem_to_mhas:
                    stem_to_mhas[stem] = {}
                stem_to_mhas[stem][series] = os.path.join(root, fname)

    lesion_samples  = []
    healthy_samples = []
    skipped_no_hbv  = 0

    for stem, series_dict in tqdm(stem_to_mhas.items(), desc="扫描切片"):
        lbl_path = label_index[stem]

        t2w_path = series_dict.get("t2w")
        adc_path = series_dict.get("adc")
        hbv_path = series_dict.get("hbv")

        if not t2w_path or not adc_path or not hbv_path:
            skipped_no_hbv += 1
            continue

        try:
            t2w_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2w_path)).astype(np.float32)
            adc_arr = sitk.GetArrayFromImage(sitk.ReadImage(adc_path)).astype(np.float32)
            hbv_arr = sitk.GetArrayFromImage(sitk.ReadImage(hbv_path)).astype(np.float32)
            lbl_arr = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))

            if t2w_arr.ndim == 4: t2w_arr = t2w_arr[0]
            if adc_arr.ndim == 4: adc_arr = adc_arr[0]
            if hbv_arr.ndim == 4: hbv_arr = hbv_arr[0]
            if lbl_arr.ndim == 4: lbl_arr = lbl_arr[0]

            n_slices = min(t2w_arr.shape[0], adc_arr.shape[0],
                           hbv_arr.shape[0], lbl_arr.shape[0])

            for s in range(n_slices):
                sample = (t2w_path, adc_path, hbv_path, lbl_path, s)
                if lbl_arr[s].sum() > 0:
                    lesion_samples.append(sample)
                else:
                    healthy_samples.append(sample)

        except Exception as e:
            print(f"[WARN] 跳过 {t2w_path}: {e}")

    print(f"跳过缺少序列的病例: {skipped_no_hbv}")
    print(f"有病灶切片: {len(lesion_samples)} | 健康切片: {len(healthy_samples)}")
    return lesion_samples, healthy_samples


# ============================================================
# 5.  Ground Truth
# ============================================================
def visualize_prediction(model, val_lesion_samples, device, output_dir, best_dice):
    model.eval()

    sample = random.choice(val_lesion_samples)
    t2w_path, adc_path, hbv_path, lbl_path, slice_idx = sample

    def read_slice(path, s):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        return arr[min(s, arr.shape[0] - 1)]

    def normalize(arr):
        std = arr.std()
        return (arr - arr.mean()) / (std + 1e-8) if std > 1e-8 else arr

    def resize_np(arr, size=IMG_SIZE):
        img = sitk.GetImageFromArray(arr)
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([size, size])
        orig_size    = img.GetSize()
        orig_spacing = img.GetSpacing()
        resampler.SetOutputSpacing([
            orig_spacing[0] * orig_size[0] / size,
            orig_spacing[1] * orig_size[1] / size,
        ])
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        return sitk.GetArrayFromImage(resampler.Execute(img)).astype(np.float32)

    
    t2w = normalize(read_slice(t2w_path, slice_idx))
    adc = normalize(read_slice(adc_path, slice_idx))
    hbv = normalize(read_slice(hbv_path, slice_idx))

    lbl_arr = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))
    if lbl_arr.ndim == 4:
        lbl_arr = lbl_arr[0]
    lbl = (lbl_arr[min(slice_idx, lbl_arr.shape[0] - 1)] > 0).astype(np.float32)

    t2w_r = resize_np(t2w)
    adc_r = resize_np(adc)
    hbv_r = resize_np(hbv)
    lbl_r = resize_np(lbl)

    img_3ch = np.stack([t2w_r, adc_r, hbv_r], axis=0)
    img_tensor = torch.from_numpy(img_3ch).float().unsqueeze(0).to(device)

  
    with torch.no_grad():
        out  = model(img_tensor)
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()
        pred = (prob > 0.5).astype(np.float32)


    p = pred.flatten()
    m = lbl_r.flatten()
    intersection = (p * m).sum()
    dice = (2. * intersection) / (p.sum() + m.sum() + 1e-8)


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Validation Sample  |  Slice Dice: {dice:.4f}  |  Best Val Dice: {best_dice:.4f}",
                 fontsize=13)


    axes[0].imshow(t2w_r, cmap='gray')
    axes[0].contour(lbl_r, levels=[0.5], colors='lime', linewidths=1.5)
    axes[0].set_title("Original (T2W)\n+ GT contour", fontsize=11)
    axes[0].axis('off')


    axes[1].imshow(t2w_r, cmap='gray')
    axes[1].imshow(prob, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[1].contour(pred, levels=[0.5], colors='red', linewidths=1.5)
    axes[1].set_title("Prediction\n(red = predicted lesion)", fontsize=11)
    axes[1].axis('off')


    axes[2].imshow(t2w_r, cmap='gray')
    axes[2].imshow(lbl_r, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
    axes[2].contour(lbl_r, levels=[0.5], colors='lime', linewidths=1.5)
    axes[2].set_title("Ground Truth\n(green = true lesion)", fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "prediction_sample.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化图已保存到: {save_path}")


# ============================================================
# 6. train
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lesion_samples, healthy_samples = build_sample_list(FOLD_ROOTS, LABEL_ROOT)


    all_stems = list({s[0] for s in lesion_samples + healthy_samples})
    random.seed(SEED)
    random.shuffle(all_stems)
    val_stems = set(all_stems[:int(len(all_stems) * VAL_RATIO)])

    def split_by_stem(samples):
        train = [s for s in samples if s[0] not in val_stems]
        val   = [s for s in samples if s[0] in val_stems]
        return train, val

    train_lesion,  val_lesion  = split_by_stem(lesion_samples)
    train_healthy, val_healthy = split_by_stem(healthy_samples)

    print(f"{len(train_lesion)} |  {len(train_healthy)}")
    print(f"{len(val_lesion)}  |  {len(val_healthy)}")

    train_ds = BalancedDataset(train_lesion, train_healthy,
                               n_lesion=N_LESION_TRAIN, n_healthy=N_HEALTHY_TRAIN,
                               fixed=False)
    val_ds   = BalancedDataset(val_lesion,   val_healthy,
                               n_lesion=N_LESION_VAL,   n_healthy=N_HEALTHY_VAL,
                               fixed=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    model     = MAESegmenter(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = CombinedLoss(pos_weight=50.0)

    best_dice  = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        train_ds.resample()


        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        scheduler.step()


        model.eval()
        val_dice_sum = 0.0
        val_count    = 0
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs, v_masks = v_imgs.to(device), v_masks.to(device)
                v_out = model(v_imgs)
                pred  = (torch.sigmoid(v_out) > 0.5).float()

                for i in range(pred.shape[0]):
                    m = v_masks[i].view(-1)
                    if m.sum() == 0:
                        continue
                    p = pred[i].view(-1)
                    intersection = (p * m).sum()
                    dice = (2. * intersection) / (p.sum() + m.sum() + 1e-8)
                    val_dice_sum += dice.item()
                    val_count    += 1

        avg_val_dice = val_dice_sum / max(val_count, 1)
        print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.4f} "
              f"| Val Dice: {avg_val_dice:.4f} (on {val_count} lesion slices)")

        if avg_val_dice > best_dice:
            best_dice  = avg_val_dice
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_terence.pth"))
            print(f"  → Saved Best Model (Dice: {best_dice:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP})")
            if no_improve >= EARLY_STOP:
                print(f"Early stopping! Best Dice: {best_dice:.4f}")
                break


    best_ckpt = os.path.join(OUTPUT_DIR, "best_terence.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    visualize_prediction(model, val_lesion, device, OUTPUT_DIR, best_dice)


if __name__ == "__main__":
    main()
