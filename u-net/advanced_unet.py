"""
Advanced U-Net with Gating Head for Stroke Lesion Segmentation
Implements architecture tricks to beat nnU-Net baseline (Dice: 0.660)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use transpose conv for upsampling or bilinear interpolation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GatingHead(nn.Module):
    """
    Gating head to predict whether there are lesions in a slice.
    This helps downweight predictions for empty slices.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        x = self.fc(x)  # [B, 1]
        return x


class AdvancedUNet(nn.Module):
    """
    U-Net with Gating Head for stroke lesion segmentation.
    
    Key features:
    - Multi-channel input support (DWI, ADC, DWI-ADC)
    - Gating head for lesion presence prediction
    - Deep supervision capability
    - 224x224 input size for better MAE encoder compatibility
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, use_gating=True):
        super(AdvancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_gating = use_gating

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Gating head (predicts if slice has lesions)
        if use_gating:
            self.gating_head = GatingHead(1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Gating prediction (binary: has lesion or not)
        gate_logit = None
        if self.use_gating:
            gate_logit = self.gating_head(x5)  # [B, 1]
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if self.use_gating:
            return logits, gate_logit
        else:
            return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)


def test_model():
    """Test the model architecture"""
    model = AdvancedUNet(n_channels=3, n_classes=1, use_gating=True)
    
    # Test with 224x224 input (upscaled from 112x112)
    x = torch.randn(2, 3, 224, 224)
    logits, gate_logit = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {logits.shape}")
    print(f"Gating output shape: {gate_logit.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_model()
