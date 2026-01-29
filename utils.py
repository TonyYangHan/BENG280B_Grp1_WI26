import os, random, matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

import numpy as np
import torch

import yaml


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def save_checkpoint(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def deep_get(cfg: Dict[str, Any], key: str, default=None):
    """
    deep_get(cfg, "train.batch_size", 16)
    """
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(cfg: Dict[str, Any], key: str, value):
    """
    deep_set(cfg, "train.batch_size", 32)
    """
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "model.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def interpolate_pos_embed(vit_model, checkpoint_state_dict: Dict[str, torch.Tensor]):
    if "pos_embed" not in checkpoint_state_dict:
        return

    pos_embed_ckpt = checkpoint_state_dict["pos_embed"]
    pos_embed_model = vit_model.pos_embed

    if pos_embed_ckpt.shape == pos_embed_model.shape:
        return

    cls_ckpt = pos_embed_ckpt[:, :1, :]
    patch_ckpt = pos_embed_ckpt[:, 1:, :]

    patch_model = pos_embed_model[:, 1:, :]

    dim = patch_ckpt.shape[-1]
    num_patches_ckpt = patch_ckpt.shape[1]
    num_patches_model = patch_model.shape[1]

    gs_ckpt = int(num_patches_ckpt ** 0.5)
    gs_model = int(num_patches_model ** 0.5)

    if gs_ckpt * gs_ckpt != num_patches_ckpt or gs_model * gs_model != num_patches_model:
        return

    patch_ckpt = patch_ckpt.reshape(1, gs_ckpt, gs_ckpt, dim).permute(0, 3, 1, 2)
    patch_ckpt = torch.nn.functional.interpolate(patch_ckpt, size=(gs_model, gs_model), mode="bicubic", align_corners=False)
    patch_ckpt = patch_ckpt.permute(0, 2, 3, 1).reshape(1, gs_model * gs_model, dim)

    new_pos = torch.cat([cls_ckpt, patch_ckpt], dim=1)
    checkpoint_state_dict["pos_embed"] = new_pos


def load_mae_vitb16_encoder_weights(vit_model, mae_ckpt_path: str, device: str = "cpu") -> Tuple[int, Dict[str, Any]]:
    ckpt = torch.load(mae_ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = _strip_prefix(state)

    interpolate_pos_embed(vit_model, state)

    model_sd = vit_model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_sd and v.shape == model_sd[k].shape}

    missing, unexpected = vit_model.load_state_dict(filtered, strict=False)

    extra = {
        "ckpt_keys": len(state),
        "loaded_keys": len(filtered),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": missing[:10],
        "unexpected_keys_sample": unexpected[:10],
    }
    epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    return epoch, extra


def compute_roc_auc(probs: np.ndarray, labels: np.ndarray):
    """
    probs: float array [N] in [0,1]
    labels: uint8/bool/int array [N] in {0,1}
    Returns: fpr [M], tpr [M], auc float
    """
    probs = probs.astype(np.float64, copy=False)
    labels = labels.astype(np.int32, copy=False)

    P = int(labels.sum())
    N = int(labels.shape[0] - P)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-probs)  # descending
    y = labels[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    tpr = tp / P
    fpr = fp / N

    # add (0,0) start
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # numpy 2.0 renamed trapz -> trapezoid; keep backward compatibility
    trapz_fn = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)
    auc = float(trapz_fn(tpr, fpr))
    return fpr, tpr, auc


def sample_pixels_for_roc(
    probs: torch.Tensor,     # [B,1,H,W] on GPU
    targets: torch.Tensor,   # [B,1,H,W] on GPU
    sample_pixels_per_slice: int = 2048,
    pos_pixels_per_slice: int = 256,
):
    """
    Stratified sampling per slice:
      - try to sample some positives (if any)
      - fill the rest with negatives
    Returns two CPU numpy arrays (p, y) concatenated across batch.
    """
    probs = probs.detach()
    targets = targets.detach()

    B, _, H, W = probs.shape
    total = H * W
    sp = int(sample_pixels_per_slice)
    spp = int(pos_pixels_per_slice)
    spp = max(0, min(sp, spp))
    spn = sp - spp

    ps = []
    ys = []

    for b in range(B):
        p = probs[b, 0].reshape(-1)
        y = targets[b, 0].reshape(-1)

        pos_idx = torch.where(y > 0.5)[0]
        neg_idx = torch.where(y <= 0.5)[0]

        # sample positives
        if pos_idx.numel() > 0 and spp > 0:
            kpos = min(spp, int(pos_idx.numel()))
            sel_pos = pos_idx[torch.randint(0, pos_idx.numel(), (kpos,), device=pos_idx.device)]
        else:
            sel_pos = None
            kpos = 0

        # sample negatives
        if spn > 0 and neg_idx.numel() > 0:
            kneg = min(spn + (spp - kpos), int(neg_idx.numel()))
            sel_neg = neg_idx[torch.randint(0, neg_idx.numel(), (kneg,), device=neg_idx.device)]
        else:
            sel_neg = None
            kneg = 0

        if sel_pos is not None and sel_neg is not None:
            sel = torch.cat([sel_pos, sel_neg], dim=0)
        elif sel_pos is not None:
            sel = sel_pos
        elif sel_neg is not None:
            sel = sel_neg
        else:
            # extremely unlikely
            sel = torch.randint(0, total, (min(sp, total),), device=p.device)

        ps.append(p[sel].float().cpu().numpy())
        ys.append(y[sel].float().cpu().numpy())

    return np.concatenate(ps, axis=0), np.concatenate(ys, axis=0)

def save_roc_plot(fpr, tpr, auc, out_dir: str, epoch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"roc_val_ep{epoch:03d}.png"
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Val ROC (AUROC={auc:.4f})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
