"""
Advanced Dataset for ISLES-2022 with preprocessing tricks
- Remove small lesions (< 3 voxels)
- Remove empty brain slices
- Upscale to 224x224 for MAE compatibility
- Balance empty/lesion-positive slices
- Multi-modal support (DWI, ADC, DWI-ADC)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy import ndimage


class ISLESAdvancedDataset(Dataset):
    """
    Advanced ISLES-2022 dataset with preprocessing tricks.
    """
    def __init__(
        self,
        npz_dir,
        split_file,
        target_size=224,
        modality='all',  # 'dwi', 'adc', or 'all'
        balance_ratio=0.5,  # ratio of empty slices to keep (0.5 = 50/50 balance)
        min_lesion_size=3,  # minimum lesion size in voxels
        remove_empty_brain=True,
        apply_crop=False,
        crop_margin=10,
        dwi_key='dwi',      # NPZ key for DWI image
        adc_key='adc',      # NPZ key for ADC image
        mask_key='mask'     # NPZ key for segmentation mask (changed from 'seg' to 'mask')
    ):
        super().__init__()
        
        self.npz_dir = npz_dir
        self.target_size = target_size
        self.modality = modality
        self.balance_ratio = balance_ratio
        self.min_lesion_size = min_lesion_size
        self.remove_empty_brain = remove_empty_brain
        self.apply_crop = apply_crop
        self.crop_margin = crop_margin
        
        # NPZ key names (can be customized)
        self.dwi_key = dwi_key
        self.adc_key = adc_key
        self.mask_key = mask_key
        
        # Auto-detect keys from first file if default keys don't work
        self._detect_npz_keys()
        
        # Load subject IDs
        with open(split_file, 'r') as f:
            self.subjects = [line.strip() for line in f.readlines()]
        
        print(f"Loading slices from {len(self.subjects)} subjects...")
        
        # Build slice index
        self.slice_data = []
        self._build_slice_index()
        
        print(f"Total slices after preprocessing: {len(self.slice_data)}")
        print(f"  Lesion-positive slices: {sum(1 for s in self.slice_data if s['has_lesion'])}")
        print(f"  Empty slices: {sum(1 for s in self.slice_data if not s['has_lesion'])}")
    
    def _detect_npz_keys(self):
        """Auto-detect NPZ key names by inspecting the first file"""
        # Get first NPZ file
        first_subject = self.subjects[0] if hasattr(self, 'subjects') else None
        if first_subject is None:
            # If subjects not loaded yet, find any NPZ file
            import os
            npz_files = [f for f in os.listdir(self.npz_dir) if f.endswith('.npz')]
            if npz_files:
                npz_path = os.path.join(self.npz_dir, npz_files[0])
            else:
                print("Warning: No NPZ files found for key detection")
                return
        else:
            npz_path = os.path.join(self.npz_dir, f"{first_subject}.npz")
        
        if not os.path.exists(npz_path):
            print(f"Warning: Cannot find NPZ file for key detection: {npz_path}")
            return
        
        # Load and inspect keys
        data = np.load(npz_path)
        available_keys = list(data.keys())
        
        print(f"\nAuto-detecting NPZ keys from: {os.path.basename(npz_path)}")
        print(f"Available keys: {available_keys}")
        
        # Try to detect keys
        key_mapping = {
            'dwi': ['dwi', 'DWI', 'dwi_image', 'image_dwi'],
            'adc': ['adc', 'ADC', 'adc_image', 'image_adc'],
            'mask': ['mask', 'seg', 'label', 'segmentation', 'lesion', 'gt', 'groundtruth']
        }
        
        detected = {}
        for data_type, possible_keys in key_mapping.items():
            for key in possible_keys:
                if key in available_keys:
                    detected[data_type] = key
                    break
        
        # Update keys if detected
        if 'dwi' in detected:
            self.dwi_key = detected['dwi']
            print(f"  ✓ DWI key: '{self.dwi_key}'")
        else:
            print(f"  ⚠ DWI key not found, using default: '{self.dwi_key}'")
        
        if 'adc' in detected:
            self.adc_key = detected['adc']
            print(f"  ✓ ADC key: '{self.adc_key}'")
        else:
            print(f"  ⚠ ADC key not found, using default: '{self.adc_key}'")
        
        if 'mask' in detected:
            self.mask_key = detected['mask']
            print(f"  ✓ Mask key: '{self.mask_key}'")
        else:
            print(f"  ⚠ Mask key not found, using default: '{self.mask_key}'")
            print(f"  Available keys were: {available_keys}")
            print(f"  You may need to manually specify mask_key parameter")
        
        print()
    
    def _is_valid_brain_slice(self, image_slice):
        """Check if slice contains visualizable brain (not all zeros)"""
        if not self.remove_empty_brain:
            return True
        return np.sum(image_slice > 0) > 100  # At least 100 non-zero pixels
    
    def _clean_small_lesions(self, mask_slice):
        """Remove small 3D components from mask (< min_lesion_size voxels)"""
        if self.min_lesion_size <= 1:
            return mask_slice
        
        # Label connected components
        labeled, num_features = ndimage.label(mask_slice)
        
        if num_features == 0:
            return mask_slice
        
        # Remove small components
        cleaned_mask = np.zeros_like(mask_slice)
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if np.sum(component) >= self.min_lesion_size:
                cleaned_mask[component] = 1
        
        return cleaned_mask
    
    def _compute_brain_crop(self, image_slice):
        """Compute tight crop around brain region"""
        # Find non-zero regions
        nonzero_mask = image_slice > 0
        rows = np.any(nonzero_mask, axis=1)
        cols = np.any(nonzero_mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add margin
        h, w = image_slice.shape
        rmin = max(0, rmin - self.crop_margin)
        rmax = min(h, rmax + self.crop_margin)
        cmin = max(0, cmin - self.crop_margin)
        cmax = min(w, cmax + self.crop_margin)
        
        return (rmin, rmax, cmin, cmax)
    
    def _build_slice_index(self):
        """Build index of all valid slices"""
        lesion_slices = []
        empty_slices = []
        
        for subject_id in self.subjects:
            npz_path = os.path.join(self.npz_dir, f"{subject_id}.npz")
            
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} not found, skipping...")
                continue
            
            # Load data
            data = np.load(npz_path)
            dwi = data[self.dwi_key]  # [D, H, W]
            adc = data[self.adc_key]  # [D, H, W]
            mask = data[self.mask_key]  # [D, H, W]
            
            # Process each slice
            for slice_idx in range(dwi.shape[0]):
                dwi_slice = dwi[slice_idx]
                adc_slice = adc[slice_idx]
                mask_slice = mask[slice_idx]
                
                # Skip if no brain visible
                if not self._is_valid_brain_slice(dwi_slice):
                    continue
                
                # Clean small lesions
                mask_slice = self._clean_small_lesions(mask_slice)
                
                # Compute crop region if needed
                crop_region = None
                if self.apply_crop:
                    crop_region = self._compute_brain_crop(dwi_slice)
                    if crop_region is None:
                        continue
                
                # Check if has lesion
                has_lesion = bool(np.sum(mask_slice) > 0)
                
                slice_info = {
                    'subject_id': subject_id,
                    'slice_idx': slice_idx,
                    'has_lesion': has_lesion,
                    'crop_region': crop_region
                }
                
                if has_lesion:
                    lesion_slices.append(slice_info)
                else:
                    empty_slices.append(slice_info)
        
        # Balance dataset
        print(f"Before balancing:")
        print(f"  Lesion slices: {len(lesion_slices)}")
        print(f"  Empty slices: {len(empty_slices)}")
        
        # Keep all lesion slices
        self.slice_data.extend(lesion_slices)
        
        # Sample empty slices based on balance_ratio
        if self.balance_ratio > 0:
            n_empty_to_keep = int(len(lesion_slices) * (self.balance_ratio / (1 - self.balance_ratio)))
            n_empty_to_keep = min(n_empty_to_keep, len(empty_slices))
            
            # Randomly sample empty slices
            rng = np.random.RandomState(42)
            empty_indices = rng.choice(len(empty_slices), n_empty_to_keep, replace=False)
            sampled_empty = [empty_slices[i] for i in empty_indices]
            self.slice_data.extend(sampled_empty)
        
        # Shuffle
        rng = np.random.RandomState(42)
        rng.shuffle(self.slice_data)
    
    def __len__(self):
        return len(self.slice_data)
    
    def __getitem__(self, idx):
        slice_info = self.slice_data[idx]
        
        # Load NPZ file
        npz_path = os.path.join(self.npz_dir, f"{slice_info['subject_id']}.npz")
        data = np.load(npz_path)
        
        dwi = data[self.dwi_key][slice_info['slice_idx']]  # [H, W]
        adc = data[self.adc_key][slice_info['slice_idx']]  # [H, W]
        mask = data[self.mask_key][slice_info['slice_idx']]  # [H, W]
        
        # Clean mask
        mask = self._clean_small_lesions(mask)
        
        # Apply crop if specified
        if slice_info['crop_region'] is not None:
            rmin, rmax, cmin, cmax = slice_info['crop_region']
            dwi = dwi[rmin:rmax, cmin:cmax]
            adc = adc[rmin:rmax, cmin:cmax]
            mask = mask[rmin:rmax, cmin:cmax]
        
        # Select modality
        if self.modality == 'dwi':
            image = dwi[np.newaxis, ...]  # [1, H, W]
        elif self.modality == 'adc':
            image = adc[np.newaxis, ...]  # [1, H, W]
        else:  # 'all'
            image = np.stack([dwi, adc, dwi - adc], axis=0)  # [3, H, W]
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Upscale to 224x224
        image = TF.resize(image, [self.target_size, self.target_size], 
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask.unsqueeze(0), [self.target_size, self.target_size],
                        interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        
        # Normalize image (0-1 range per channel)
        for c in range(image.shape[0]):
            min_val = image[c].min()
            max_val = image[c].max()
            if max_val > min_val:
                image[c] = (image[c] - min_val) / (max_val - min_val)
        
        return {
            'image': image,
            'mask': mask,
            'has_lesion': bool(slice_info['has_lesion']),
            'subject_id': slice_info['subject_id'],
            'slice_idx': slice_info['slice_idx']
        }


def test_dataset():
    """Test dataset loading"""
    # These paths should match your Google Drive structure
    npz_dir = '/content/drive/MyDrive/BENG_280B/ISLES-2022-npz-multimodal_clean/all'
    splits_dir = '/content/drive/MyDrive/BENG_280B/ISLES-2022-npz-multimodal_clean/splits'
    
    train_file = os.path.join(splits_dir, 'train.txt')
    
    if not os.path.exists(train_file):
        print(f"Split file not found: {train_file}")
        print("Please update paths in test_dataset()")
        return
    
    # Test different configurations
    configs = [
        {'modality': 'dwi', 'balance_ratio': 0.5},
        {'modality': 'adc', 'balance_ratio': 0.5},
        {'modality': 'all', 'balance_ratio': 0.3},  # 30% empty, 70% lesion
    ]
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing config: {config}")
        print('='*70)
        
        dataset = ISLESAdvancedDataset(
            npz_dir=npz_dir,
            split_file=train_file,
            target_size=224,
            **config
        )
        
        # Test loading a sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Has lesion: {sample['has_lesion']}")
        print(f"  Lesion pixels: {sample['mask'].sum().item()}")


if __name__ == "__main__":
    test_dataset()
