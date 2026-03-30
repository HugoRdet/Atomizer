"""
3D UNet for Hyperspectral Segmentation
========================================

Treats the spectral dimension as a third spatial axis:
    Input: [B, 1, C_spectral, H, W]  (single "channel", 3D volume)

3D convolutions learn joint spectral-spatial features. The spectral
dimension is progressively compressed through pooling. After the
bottleneck, spectral dim is collapsed to 1 and the decoder uses
2D convolutions with skip connections.

Interface matches other baselines:
    model = UNet3D(in_channels=242, num_classes=14)
    logits = model(image)  # [B, 242, H, W] → [B, 14, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3D conv layers with GroupNorm and ReLU."""
    def __init__(self, in_ch, out_ch, spectral_kernel=3):
        super().__init__()
        s_pad = spectral_kernel // 2
        self.conv1 = nn.Conv3d(in_ch, out_ch,
                               kernel_size=(spectral_kernel, 3, 3),
                               padding=(s_pad, 1, 1), bias=False)
        self.norm1 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch,
                               kernel_size=(spectral_kernel, 3, 3),
                               padding=(s_pad, 1, 1), bias=False)
        self.norm2 = nn.GroupNorm(min(16, out_ch), out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class ConvBlock2D(nn.Module):
    """Two 2D conv layers with GroupNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(16, out_ch), out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    """
    3D UNet for hyperspectral semantic segmentation.

    Encoder uses 3D convolutions over (spectral, H, W).
    Spectral dimension is pooled progressively.
    After bottleneck, spectral dim is collapsed and decoder uses 2D convs.

    Args:
        in_channels: Number of spectral bands (e.g., 242 for EnMAP).
        num_classes: Number of segmentation classes.
        base_features: Base feature count (doubled at each level).
    """

    def __init__(self, in_channels=242, num_classes=14, base_features=32):
        super().__init__()
        self.in_channels = in_channels
        f = base_features

        # ── 3D Encoder ─────────────────────────────────────────────
        self.enc1 = ConvBlock3D(1, f, spectral_kernel=7)
        self.enc2 = ConvBlock3D(f, f*2, spectral_kernel=5)
        self.enc3 = ConvBlock3D(f*2, f*4, spectral_kernel=3)
        self.enc4 = ConvBlock3D(f*4, f*8, spectral_kernel=3)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # ── Bottleneck (3D) ────────────────────────────────────────
        self.bottleneck_3d = ConvBlock3D(f*8, f*8, spectral_kernel=3)

        # ── Spectral collapse ─────────────────────────────────────
        self.spec_collapse = nn.AdaptiveAvgPool3d((1, None, None))

        # ── 2D Decoder ─────────────────────────────────────────────
        self.up4 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock2D(f*4 + f*4, f*4)

        self.up3 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(f*2 + f*2, f*2)

        self.up2 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(f + f, f)

        self.up1 = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(f, f)

        # ── Output ─────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[UNet3D] in_channels={in_channels}, num_classes={num_classes}")
        print(f"[UNet3D] base_features={base_features}")
        print(f"[UNet3D] Parameters: {n_params:,}")

    def forward(self, image):
        """
        Args:
            image: [B, C_spectral, H, W]

        Returns:
            logits: [B, num_classes, H, W]
        """
        B, C, H, W = image.shape

        # Add channel dim: [B, C, H, W] → [B, 1, C, H, W]
        x = image.unsqueeze(1)

        # ── 3D Encoder ─────────────────────────────────────────────
        e1 = self.enc1(x)
        e1_pool = self.pool(e1)

        e2 = self.enc2(e1_pool)
        e2_pool = self.pool(e2)

        e3 = self.enc3(e2_pool)
        e3_pool = self.pool(e3)

        e4 = self.enc4(e3_pool)
        e4_pool = self.pool(e4)

        # ── Bottleneck ─────────────────────────────────────────────
        b = self.bottleneck_3d(e4_pool)

        # ── Spectral collapse ──────────────────────────────────────
        b_2d = self.spec_collapse(b).squeeze(2)
        e3_2d = self.spec_collapse(e3).squeeze(2)
        e2_2d = self.spec_collapse(e2).squeeze(2)
        e1_2d = self.spec_collapse(e1).squeeze(2)

        # ── 2D Decoder ─────────────────────────────────────────────
        d4 = self.up4(b_2d)
        e3_2d = self._match_size(e3_2d, d4)
        d4 = self.dec4(torch.cat([d4, e3_2d], dim=1))

        d3 = self.up3(d4)
        e2_2d = self._match_size(e2_2d, d3)
        d3 = self.dec3(torch.cat([d3, e2_2d], dim=1))

        d2 = self.up2(d3)
        e1_2d = self._match_size(e1_2d, d2)
        d2 = self.dec2(torch.cat([d2, e1_2d], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, image)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    @staticmethod
    def _match_size(source, target):
        """Interpolate source to match target's spatial dims."""
        if source.shape[-2:] != target.shape[-2:]:
            source = F.interpolate(
                source, size=target.shape[-2:],
                mode='bilinear', align_corners=False)
        return source