"""
Advanced Loss Functions for Stroke Lesion Segmentation
Implements: Focal Loss, Dice Loss, Tversky Loss
Avoids BCE due to class imbalance issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reduces loss for well-classified examples, focuses on hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] - raw logits
            targets: [B, H, W] or [B, 1, H, W] - binary targets
        """
        # Ensure targets have same shape as inputs
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate focal loss
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Directly optimizes the Dice coefficient.
    
    DiceLoss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] - raw logits
            targets: [B, H, W] or [B, 1, H, W] - binary targets
        """
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Ensure targets have same shape as inputs
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.
    Better for handling false positives and false negatives differently.
    
    TL = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    
    When alpha=beta=0.5, reduces to Dice Loss.
    alpha>0.5 penalizes FP more (reduce over-segmentation)
    beta>0.5 penalizes FN more (reduce under-segmentation)
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] - raw logits
            targets: [B, H, W] or [B, 1, H, W] - binary targets
        """
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Ensure targets have same shape as inputs
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate components
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        
        # Calculate Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal + Dice + Tversky
    Weights can be adjusted based on performance
    """
    def __init__(
        self,
        focal_weight=1.0,
        dice_weight=1.0,
        tversky_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        tversky_alpha=0.3,
        tversky_beta=0.7
    ):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    
    def forward(self, inputs, targets, gate_weight=None):
        """
        Args:
            inputs: [B, 1, H, W] - raw logits
            targets: [B, H, W] or [B, 1, H, W] - binary targets
            gate_weight: [B, 1] - gating weights (optional)
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        
        total_loss = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.tversky_weight * tversky
        )
        
        # Apply gating weights if provided
        if gate_weight is not None:
            # Downweight loss for slices predicted to have no lesions
            total_loss = total_loss * gate_weight.mean()
        
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(),
            'tversky': tversky.item(),
            'total': total_loss.item()
        }


class GatingLoss(nn.Module):
    """
    Binary cross-entropy loss for gating head.
    Predicts whether a slice contains lesions.
    """
    def __init__(self):
        super(GatingLoss, self).__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, gate_logits, targets):
        """
        Args:
            gate_logits: [B, 1] - sigmoid outputs from gating head
            targets: [B, H, W] or [B, 1, H, W] - binary masks
        """
        # Create binary labels: 1 if mask has any lesion pixels, 0 otherwise
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Check if each slice has lesions
        has_lesion = (targets.sum(dim=[2, 3]) > 0).float()  # [B, 1]
        
        loss = self.bce(gate_logits, has_lesion)
        return loss


def test_losses():
    """Test all loss functions"""
    batch_size = 4
    height, width = 224, 224
    
    # Create dummy data
    inputs = torch.randn(batch_size, 1, height, width)
    targets = (torch.rand(batch_size, height, width) > 0.9).float()
    gate_logits = torch.sigmoid(torch.randn(batch_size, 1))
    
    print("Testing loss functions...")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Gate logits shape: {gate_logits.shape}")
    print()
    
    # Test individual losses
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    tversky_loss = TverskyLoss()
    gating_loss = GatingLoss()
    
    print(f"Focal Loss: {focal_loss(inputs, targets).item():.4f}")
    print(f"Dice Loss: {dice_loss(inputs, targets).item():.4f}")
    print(f"Tversky Loss: {tversky_loss(inputs, targets).item():.4f}")
    print(f"Gating Loss: {gating_loss(gate_logits, targets).item():.4f}")
    print()
    
    # Test combined loss
    combined_loss = CombinedLoss()
    total, components = combined_loss(inputs, targets, gate_logits)
    print(f"Combined Loss: {total.item():.4f}")
    print(f"Components: {components}")


if __name__ == "__main__":
    test_losses()
