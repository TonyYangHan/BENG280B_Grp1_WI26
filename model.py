
import torch
import torch.nn as nn
import timm

class ConvDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)

class MAESegmenter(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # 关键修改：in_chans=3 (DWI, ADC, DWI-ADC)
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=3, global_pool='')
        self.decoder = ConvDecoder(embed_dim=768, num_classes=num_classes)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        features = features[:, 1:, :] # 移除 CLS token
        B, N, C = features.shape
        H = W = int(N**0.5) 
        features = features.transpose(1, 2).reshape(B, C, H, W)
        return self.decoder(features)
