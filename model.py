
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

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim=3*224*224, hidden_dims=None, num_classes=1, dropout=0.0):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten only when we receive image-like tensors
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.mlp(x)


class FrozenViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', in_chans=3):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, in_chans=in_chans, global_pool='')
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.embed_dim = getattr(self.encoder, "num_features", getattr(self.encoder, "embed_dim", 768))
        self.encoder.eval()

    def forward(self, x):
        # Keep encoder frozen and in eval mode for deterministic latents
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder.forward_features(x)
        cls_token = features[:, 0]
        return cls_token


class FrozenViTMLPClassifier(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=1, dropout=0.1, model_name='vit_base_patch16_224', in_chans=3):
        super().__init__()
        self.encoder = FrozenViTEncoder(model_name=model_name, in_chans=in_chans)
        self.classifier = SimpleMLPClassifier(
            input_dim=self.encoder.embed_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        latents = self.encoder(x)
        return self.classifier(latents)
