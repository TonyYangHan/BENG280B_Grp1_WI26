import torch, torch.nn as nn
import torch.nn.functional as F

class FrozenMAEEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    @torch.no_grad()
    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, N+1, C]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token tensor [B,N,C], got {tokens.shape}.")
        patch = tokens[:, 1:, :]
        B, N, C = patch.shape
        h = w = int(N ** 0.5)
        if h * w != N:
            raise RuntimeError(f"Patch tokens N={N} is not a square; check input size/patch size.")
        feat = patch.transpose(1, 2).reshape(B, C, h, w)
        return feat


class SimpleUpsampleDecoder(nn.Module):
    def __init__(self, in_ch: int = 768, out_size: int = 224, patch: int = 16, mid_ch: int = 256):
        super().__init__()
        assert out_size % patch == 0, "out_size must be divisible by patch size."

        def gn(c):
            g = 32
            while c % g != 0 and g > 1:
                g //= 2
            return nn.GroupNorm(g, c)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            gn(mid_ch),
            nn.GELU(),
        )

        steps = int(round(torch.log2(torch.tensor(patch)).item()))  # patch=16 -> 4
        self.blocks = nn.ModuleList()
        for _ in range(steps):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
                gn(mid_ch),
                nn.GELU(),
            ))

        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, feat):
        x = self.proj(feat)
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(x)
        return self.head(x)