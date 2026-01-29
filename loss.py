# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: [B,1,H,W]
    targets: [B,1,H,W] in {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 1.0,
    alpha_pos: float = 0.75,
    alpha_neg: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    targets = targets.float()

    # Standard BCE on logits (stable)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = exp(-bce): equals p for y=1 and (1-p) for y=0
    p_t = torch.exp(-bce)

    alpha_t = alpha_pos * targets + alpha_neg * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss



class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        d = dice_loss_with_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * d


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha_pos=0.75, alpha_neg=0.25, focal_w=0.5, dice_w=0.5):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = float(alpha_neg)
        self.focal_w = float(focal_w)
        self.dice_w = float(dice_w)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = focal_loss_with_logits(logits, targets, gamma=self.gamma, alpha_pos=self.alpha_pos, alpha_neg=self.alpha_neg)
        d = dice_loss_with_logits(logits, targets)
        return self.focal_w * focal + self.dice_w * d


@torch.no_grad()
def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    pred = pred.view(pred.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (pred * targets).sum(dim=1)
    denom = pred.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()

def dice_score_ignore_empty(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    """
    Dice computed only on samples where GT has any positive pixel.
    Prevents empty-slice dice~=1 inflating metrics.
    """
    B = targets.size(0)
    has_pos = (targets.view(B, -1).sum(dim=1) > 0)
    if not torch.any(has_pos):
        return 0.0
    return float(dice_score_from_logits(logits[has_pos], targets[has_pos], thr=thr).item())
