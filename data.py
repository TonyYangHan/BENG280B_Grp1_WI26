# data.py
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class CaseIndex:
    path: str
    num_slices: int
    lesion_slices: List[int]


def _load_npz(path: str, img_key: str = "img", mask_key: str = "mask"):
    d = np.load(path, allow_pickle=False)
    img = d[img_key]  # [Z,H,W]
    mask = d[mask_key]  # [Z,H,W]
    return img, mask


def build_case_index(
    npz_paths: List[str],
    img_key: str = "img",
    mask_key: str = "mask",
) -> List[CaseIndex]:
    cases: List[CaseIndex] = []
    for p in npz_paths:
        img, mask = _load_npz(p, img_key, mask_key)
        if img.ndim != 3 or mask.ndim != 3:
            raise ValueError(f"{p}: expected img/mask to be 3D [Z,H,W], got img {img.shape} mask {mask.shape}")
        if img.shape != mask.shape:
            raise ValueError(f"{p}: img/mask shape mismatch: img {img.shape}, mask {mask.shape}")

        lesion = (mask > 0).astype(np.uint8)
        lesion_slices = np.where(lesion.reshape(lesion.shape[0], -1).sum(axis=1) > 0)[0].tolist()

        cases.append(CaseIndex(path=p, num_slices=img.shape[0], lesion_slices=lesion_slices))
    return cases


def split_cases(
    cases: List[CaseIndex],
    val_frac: float = 0.2,
    seed: int = 1337,
) -> Tuple[List[CaseIndex], List[CaseIndex]]:
    rng = random.Random(seed)
    idx = list(range(len(cases)))
    rng.shuffle(idx)
    n_val = int(round(len(cases) * val_frac))
    val_idx = set(idx[:n_val])
    train = [cases[i] for i in range(len(cases)) if i not in val_idx]
    val = [cases[i] for i in range(len(cases)) if i in val_idx]
    return train, val


class NpzSliceDataset(Dataset):
    """
    Returns (x, y):
      x: [C, out_size, out_size] float32
      y: [1, out_size, out_size] float32 (0/1)
    """

    def __init__(
        self,
        cases: List[CaseIndex],
        out_size: int = 224,
        k_slices: int = 3,                 # use 3 to match RGB weights (z-1,z,z+1)
        lesion_sampling: float = 0.7,      # probability to draw a lesion-containing slice (if available)
        img_key: str = "img",
        mask_key: str = "mask",
        normalize: bool = True,
    ):
        if k_slices not in (1, 3, 5):
            raise ValueError("k_slices should be 1, 3, or 5 for simple 2.5D stacks.")
        self.cases = cases
        self.out_size = int(out_size)
        self.k = int(k_slices)
        self.lesion_sampling = float(lesion_sampling)
        self.img_key = img_key
        self.mask_key = mask_key
        self.normalize = normalize

        # index: (case_i, z) for slices that contain lesion only
        self.index: List[Tuple[int, int]] = []
        for ci, c in enumerate(self.cases):
            if not c.lesion_slices:
                continue
            for z in c.lesion_slices:
                self.index.append((ci, z))

        if not self.index:
            raise ValueError("No lesion-containing slices found in provided cases.")

    def __len__(self):
        return len(self.index)

    def _choose_z(self, case: CaseIndex, z_default: int) -> int:
        if self.lesion_sampling <= 0 or not case.lesion_slices:
            return z_default
        if random.random() < self.lesion_sampling:
            return random.choice(case.lesion_slices)
        return z_default

    def _stack_slices(self, vol: np.ndarray, z: int) -> np.ndarray:
        # vol: [Z,H,W]
        Z = vol.shape[0]
        if self.k == 1:
            idxs = [z]
        elif self.k == 3:
            idxs = [max(0, z - 1), z, min(Z - 1, z + 1)]
        else:  # k == 5
            idxs = [max(0, z - 2), max(0, z - 1), z, min(Z - 1, z + 1), min(Z - 1, z + 2)]
        x = vol[idxs, :, :]  # [C,H,W]
        return x

    def __getitem__(self, i: int):
        ci, z0 = self.index[i]
        case = self.cases[ci]
        z = self._choose_z(case, z0)

        img, mask = _load_npz(case.path, self.img_key, self.mask_key)  # [Z,H,W]
        x = self._stack_slices(img.astype(np.float32), z)              # [C,H,W]
        y = (mask[z].astype(np.float32) > 0).astype(np.float32)[None]  # [1,H,W]

        # Optional per-slice normalization (simple and robust)
        if self.normalize:
            # normalize based on the center slice channel
            center = x[x.shape[0] // 2]
            lo, hi = np.percentile(center, [0.5, 99.5])
            x = np.clip(x, lo, hi)
            mu = x.mean()
            sd = x.std() + 1e-6
            x = (x - mu) / sd

        # Ensure float32 (percentile/mean can promote to float64)
        x = x.astype(np.float32, copy=False)

        # Resize to out_size (this is "training upscale"; if you only want display upscale, do it in qc.py)
        xt = torch.from_numpy(x)  # [C,H,W]
        yt = torch.from_numpy(y)  # [1,H,W]
        xt = xt.unsqueeze(0)      # [1,C,H,W]
        yt = yt.unsqueeze(0)      # [1,1,H,W]

        xt = F.interpolate(xt, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        yt = F.interpolate(yt, size=(self.out_size, self.out_size), mode="nearest")

        xt = xt.squeeze(0).contiguous()  # [C, out, out]
        yt = yt.squeeze(0).contiguous()  # [1, out, out]
        return xt, yt
