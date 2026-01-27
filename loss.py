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
