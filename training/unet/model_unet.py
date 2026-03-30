"""
UNet Baseline for MDAS Segmentation
====================================

Standard encoder-decoder UNet with skip connections.
First conv layer adapts to arbitrary input channel count,
so the same architecture works for HySpex (368ch) and S2 (12ch).

Architecture:
    Encoder: input → 64 → 128 → 256 → 512
    Decoder: 512 → 256 → 128 → 64 → num_classes

Designed for small crop sizes (14×14 to 64×64).
Uses padding to preserve spatial dimensions through convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3×3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch = channels from below + skip channels
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        # Upsample x to match skip's spatial size
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet for semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 368 for HySpex, 12 for S2).
    num_classes : int
        Number of output classes (default: 6 for MDAS).
    base_dim : int
        Base feature dimension (default: 64). Encoder doubles at each stage.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        base_dim: int = 64,
    ):
        super().__init__()

        d = base_dim  # 64

        # Encoder
        self.enc1 = ConvBlock(in_channels, d)       # → d
        self.enc2 = DownBlock(d, d * 2)              # → 2d
        self.enc3 = DownBlock(d * 2, d * 4)          # → 4d
        self.enc4 = DownBlock(d * 4, d * 8)          # → 8d (bottleneck)

        # Decoder
        self.dec3 = UpBlock(d * 8 + d * 4, d * 4)   # 8d up + 4d skip → 4d
        self.dec2 = UpBlock(d * 4 + d * 2, d * 2)   # 4d up + 2d skip → 2d
        self.dec1 = UpBlock(d * 2 + d, d)            # 2d up + d skip  → d

        # Head
        self.head = nn.Conv2d(d, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] input tensor.

        Returns:
            logits: [B, num_classes, H, W] — same spatial size as input.
        """
        # Encoder
        e1 = self.enc1(x)   # [B, d, H, W]
        e2 = self.enc2(e1)  # [B, 2d, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 4d, H/4, W/4]
        e4 = self.enc4(e3)  # [B, 8d, H/8, W/8]

        # Decoder
        d3 = self.dec3(e4, e3)  # [B, 4d, H/4, W/4]
        d2 = self.dec2(d3, e2)  # [B, 2d, H/2, W/2]
        d1 = self.dec1(d2, e1)  # [B, d, H, W]

        return self.head(d1)    # [B, num_classes, H, W]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)