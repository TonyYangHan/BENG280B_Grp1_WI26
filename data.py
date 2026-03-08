import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_case_list(txt_path):
    with open(txt_path, 'r') as f:
        cases = [line.strip() for line in f.readlines() if line.strip()]
    return cases

class IslesNpzDataset(Dataset):
    def __init__(self, data_root, list_path, img_size=224, samples_per_patient=20):
        self.data_root = data_root
        self.case_list = load_case_list(list_path)
        self.img_size = img_size
        self.samples_per_patient = samples_per_patient # <--- 关键修改：每个病人采样多少次
        
        self.valid_files = []
        for case_name in self.case_list:
            fname = case_name if case_name.endswith('.npz') else f"{case_name}.npz"
            file_path = os.path.join(self.data_root, fname)
            if os.path.exists(file_path):
                self.valid_files.append(file_path)
            else:
                print(f"Dataset Warning: Missing {file_path}")
        print(f"Loaded {len(self.valid_files)} cases from {list_path}")
        print(f"⚡ Training Strategy: {samples_per_patient} random slices per patient per epoch.")

    def __len__(self):
        # 让数据集看起来变大了 20 倍
        return len(self.valid_files) * self.samples_per_patient

    def normalize(self, img):
        mean = np.mean(img)
        std = np.std(img)
        if std == 0: return img
        return (img - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        # 映射回真实的病人索引
        patient_idx = idx % len(self.valid_files)
        path = self.valid_files[patient_idx]
        
        try:
            data = np.load(path)
            dwi_vol = data['dwi']
            adc_vol = data['adc']
            mask_vol = data['mask']
            
            # --- 随机采样策略 ---
            # 优先取有病灶的层 (Lesion slices)
            z_indices = np.where(np.sum(mask_vol, axis=(1, 2)) > 0)[0]
            
            # 也可以加入一些无病灶的层 (Background slices) 来平衡，这里我们先专注于病灶
            if len(z_indices) > 0:
                z = np.random.choice(z_indices)
            else:
                z = dwi_vol.shape[0] // 2
                
            dwi = dwi_vol[z]
            adc = adc_vol[z]
            msk = mask_vol[z]

            # --- 3 Channels: DWI, ADC, DWI-ADC ---
            diff = dwi - adc
            
            dwi = self.normalize(dwi)
            adc = self.normalize(adc)
            diff = self.normalize(diff)
            
            img_stack = np.stack([dwi, adc, diff], axis=0)
            
            img_t = torch.from_numpy(img_stack).float()
            msk_t = torch.from_numpy(msk).float().unsqueeze(0)

            # Resize to 224
            img_t = F.interpolate(img_t.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), size=(self.img_size, self.img_size), mode='nearest').squeeze(0)
            
            return img_t, msk_t
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, 224, 224)), torch.zeros((1, 224, 224))