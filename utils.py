import os, random
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
