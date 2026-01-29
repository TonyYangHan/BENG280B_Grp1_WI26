# data.py
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class CaseIndex:
    path: str
    num_slices: int
    lesion_slices: List[int]
    empty_slices: List[int]


def _npz_get(path: str, key: str) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if key not in d:
        raise KeyError(f"{path}: missing key '{key}'. Available: {list(d.keys())}")
    return d[key]


def _load_mask(path: str, mask_key: str) -> np.ndarray:
    m = _npz_get(path, mask_key)
    if m.ndim != 3:
        raise ValueError(f"{path}: expected mask [Z,H,W], got {m.shape}")
    return m


def _load_modalities(path: str, modality_keys: Sequence[str]) -> List[np.ndarray]:
    outs = []
    for k in modality_keys:
        v = _npz_get(path, k)
        if v.ndim != 3:
            raise ValueError(f"{path}: expected '{k}' [Z,H,W], got {v.shape}")
        outs.append(v)
    return outs


def build_case_index(
    npz_paths: List[str],
    mask_key: str = "mask",
    channel_mode: str = "modalities",
    modality_keys: Sequence[str] = ("dwi", "adc", "flair"),
    img_key: str = "img",   # only used in zstack mode
) -> List[CaseIndex]:
    """
    Builds per-case slice indices based on mask.
    """
    cases: List[CaseIndex] = []
    channel_mode = str(channel_mode).lower().strip()

    for p in npz_paths:
        mask = _load_mask(p, mask_key)
        Z = mask.shape[0]

        # sanity check volume shapes
        if channel_mode == "modalities":
            vols = _load_modalities(p, modality_keys)
            for k, v in zip(modality_keys, vols):
                if v.shape != mask.shape:
                    raise ValueError(f"{p}: shape mismatch '{k}' {v.shape} vs mask {mask.shape}")
        elif channel_mode == "zstack":
            img = _npz_get(p, img_key)
            if img.ndim != 3:
                raise ValueError(f"{p}: expected img [Z,H,W], got {img.shape}")
            if img.shape != mask.shape:
                raise ValueError(f"{p}: img/mask shape mismatch: img {img.shape}, mask {mask.shape}")
        else:
            raise ValueError("channel_mode must be 'modalities' or 'zstack'")

        lesion = (mask > 0).astype(np.uint8)
        lesion_slices = np.where(lesion.reshape(Z, -1).sum(axis=1) > 0)[0].tolist()
        empty_slices = [z for z in range(Z) if z not in set(lesion_slices)]

        cases.append(CaseIndex(path=p, num_slices=Z, lesion_slices=lesion_slices, empty_slices=empty_slices))

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


def _pad_crop_2d(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, pad_val: float = 0.0) -> np.ndarray:
    """
    arr: [H,W]
    returns cropped region with padding if needed, shape [(y1-y0),(x1-x0)]
    """
    H, W = arr.shape
    top = max(0, -y0)
    left = max(0, -x0)
    bot = max(0, y1 - H)
    right = max(0, x1 - W)

    yy0 = max(0, y0)
    yy1 = min(H, y1)
    xx0 = max(0, x0)
    xx1 = min(W, x1)

    out = arr[yy0:yy1, xx0:xx1]
    if top or left or bot or right:
        out = np.pad(out, ((top, bot), (left, right)), mode="constant", constant_values=pad_val)
    return out


def _lesion_center(mask2d: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask2d > 0)
    cy = int((ys.min() + ys.max()) // 2)
    cx = int((xs.min() + xs.max()) // 2)
    return cy, cx


def _random_center(H: int, W: int) -> Tuple[int, int]:
    return random.randint(0, H - 1), random.randint(0, W - 1)


def _apply_crop(
    x: np.ndarray,          # [C,H,W]
    y: np.ndarray,          # [1,H,W]
    size_native: int,
    jitter_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    C, H, W = x.shape
    cs = int(size_native)
    if cs <= 0 or cs >= min(H, W):
        return x, y

    mask2d = y[0]
    if mask2d.sum() > 0:
        cy, cx = _lesion_center(mask2d)
    else:
        cy, cx = _random_center(H, W)

    j = int(round(jitter_frac * cs))
    if j > 0:
        cy += random.randint(-j, j)
        cx += random.randint(-j, j)

    y0 = cy - cs // 2
    x0 = cx - cs // 2
    y1 = y0 + cs
    x1 = x0 + cs

    # crop each channel + mask with padding
    xc = np.stack([_pad_crop_2d(x[c], y0, y1, x0, x1, pad_val=0.0) for c in range(C)], axis=0)
    yc = _pad_crop_2d(mask2d, y0, y1, x0, x1, pad_val=0.0)[None]
    return xc, yc


def _normalize_channels(x: np.ndarray) -> np.ndarray:
    """
    x: [C,H,W] float32
    per-channel robust clip + standardize
    """
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c]
        lo, hi = np.percentile(v, [0.5, 99.5])
        v = np.clip(v, lo, hi)
        mu = float(v.mean())
        sd = float(v.std()) + 1e-6
        out[c] = (v - mu) / sd
    return out


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

        # input modes
        channel_mode: str = "modalities",
        modality_keys: Sequence[str] = ("dwi", "adc", "flair"),

        # legacy zstack mode
        img_key: str = "img",
        k_slices: int = 3,

        mask_key: str = "mask",

        # sampling
        empty_slice_prob: float = 0.1,

        # preprocessing
        normalize: bool = True,

        # optional crop
        crop_enabled: bool = False,
        crop_size_native: int = 96,
        crop_jitter: float = 0.15,
    ):
        channel_mode = str(channel_mode).lower().strip()
        if channel_mode not in ("modalities", "zstack"):
            raise ValueError("channel_mode must be 'modalities' or 'zstack'")

        if channel_mode == "zstack" and k_slices not in (1, 3, 5):
            raise ValueError("k_slices should be 1, 3, or 5 for simple 2.5D stacks.")

        self.cases = cases
        self.out_size = int(out_size)

        self.channel_mode = channel_mode
        self.modality_keys = list(modality_keys)

        self.img_key = img_key
        self.k = int(k_slices)

        self.mask_key = mask_key
        self.empty_slice_prob = float(empty_slice_prob)
        self.normalize = bool(normalize)

        self.crop_enabled = bool(crop_enabled)
        self.crop_size_native = int(crop_size_native)
        self.crop_jitter = float(crop_jitter)

        # base index: all lesion slices (by case)
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

    def _stack_z(self, vol: np.ndarray, z: int) -> np.ndarray:
        # vol: [Z,H,W]
        Z = vol.shape[0]
        if self.k == 1:
            idxs = [z]
        elif self.k == 3:
            idxs = [max(0, z - 1), z, min(Z - 1, z + 1)]
        else:  # k == 5
            idxs = [max(0, z - 2), max(0, z - 1), z, min(Z - 1, z + 1), min(Z - 1, z + 2)]
        return vol[idxs, :, :]  # [C,H,W]

    def __getitem__(self, i: int):
        ci, z_lesion = self.index[i]
        case = self.cases[ci]

        # choose whether to inject an empty slice
        use_empty = (self.empty_slice_prob > 0) and (random.random() < self.empty_slice_prob) and bool(case.empty_slices)
        z = random.choice(case.empty_slices) if use_empty else z_lesion

        # load mask slice
        mask_vol = _load_mask(case.path, self.mask_key).astype(np.float32)  # [Z,H,W]
        y = (mask_vol[z] > 0).astype(np.float32)[None]  # [1,H,W]

        # load input channels
        if self.channel_mode == "modalities":
            vols = _load_modalities(case.path, self.modality_keys)  # list of [Z,H,W]
            x = np.stack([v[z].astype(np.float32) for v in vols], axis=0)  # [3,H,W]
        else:
            img_vol = _npz_get(case.path, self.img_key).astype(np.float32)  # [Z,H,W]
            x = self._stack_z(img_vol, z)  # [C,H,W]

        # optional crop in native resolution
        if self.crop_enabled:
            x, y = _apply_crop(x, y, size_native=self.crop_size_native, jitter_frac=self.crop_jitter)

        # normalize
        if self.normalize:
            x = _normalize_channels(x)

        # resize to out_size
        xt = torch.from_numpy(x).unsqueeze(0)  # [1,C,H,W]
        yt = torch.from_numpy(y).unsqueeze(0)  # [1,1,H,W]

        xt = F.interpolate(xt, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        yt = F.interpolate(yt, size=(self.out_size, self.out_size), mode="nearest")

        xt = xt.squeeze(0).contiguous()
        yt = yt.squeeze(0).contiguous()
        return xt, yt