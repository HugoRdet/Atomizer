"""
SenPa-MAE Segmentation Baseline
==================================

Adapted from Prexl & Schmitt (2024) "SenPa-MAE: Sensor Parameter Aware
Masked Autoencoder for Multi-Satellite Self-Supervised Pretraining."

Original SenPa-MAE is a MAE for pretraining. We adapt the encoder
architecture for supervised segmentation:
  - Sensor parameter encoding (spectral response + GSD)
  - Channels as separate tokens (each band is a token)
  - No masking (supervised training)
  - Simple upsampling segmentation head

The key idea: each band's spectral response function (2301-dim,
sampled at 1nm over 400–2700nm) is embedded via MLP and ADDED
to patch embeddings. GSD is similarly embedded. This gives the
ViT explicit knowledge of sensor characteristics.

Input: [B, C, H, W] + response_functions [C, 2301] + gsd [scalar]
Output: [B, num_classes, H, W]

For C2Seg baseline: same interface as UNet/ViT.
    model = SenPaSeg(in_channels=242, num_classes=14, img_size=128)
    logits = model(image)  # response functions built from metadata
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_


# =============================================================================
# SENSOR PARAMETER ENCODERS (from SenPa-MAE)
# =============================================================================

class SpectralResponseEncoder(nn.Module):
    """Encode spectral response function (2301-dim) → emb_dim."""
    def __init__(self, emb_dim, response_dim=2301):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(response_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, rf):
        """
        Args:
            rf: [B, C, 2301] spectral response per band
        Returns:
            [B, C, emb_dim]
        """
        return self.net(rf)


class GSDEncoder(nn.Module):
    """Encode ground sampling distance (scalar) → emb_dim."""
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, gsd):
        """
        Args:
            gsd: [B, C] GSD per band (usually same value repeated)
        Returns:
            [B, C, emb_dim]
        """
        return self.net(gsd.unsqueeze(-1))


# =============================================================================
# SPECTRAL RESPONSE FUNCTION BUILDER
# =============================================================================

def build_response_functions(wavelengths, bandwidths, wl_min=400, wl_max=2700):
    """
    Build rectangular spectral response functions from wavelength/bandwidth.

    Args:
        wavelengths: list of central wavelengths (nm), length C
        bandwidths: list of bandwidths (nm), length C
        wl_min: start of spectral range
        wl_max: end of spectral range

    Returns:
        rf: [C, 2301] response functions (0 or 1 for rectangular)
    """
    n_points = wl_max - wl_min + 1  # 2301
    rf = torch.zeros(len(wavelengths), n_points)

    for i, (wl, bw) in enumerate(zip(wavelengths, bandwidths)):
        lo = wl - bw / 2.0
        hi = wl + bw / 2.0
        # Indices into the 2301-dim array
        idx_lo = max(0, int(round(lo - wl_min)))
        idx_hi = min(n_points - 1, int(round(hi - wl_min)))
        if idx_lo <= idx_hi:
            rf[i, idx_lo:idx_hi + 1] = 1.0

    return rf


# =============================================================================
# UPERNET DECODER (matches ViT+UPerNet baseline)
# =============================================================================

def _norm(num_channels, num_groups=16):
    """GroupNorm — works with any batch size (unlike BatchNorm)."""
    return nn.GroupNorm(min(num_groups, num_channels), num_channels)


class PPM(nn.Module):
    """Pyramid Pooling Module. Adapts bins to spatial size at runtime."""
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.reduction_dim = reduction_dim

        self.pool_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for _ in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * len(bins), in_dim,
                      kernel_size=3, padding=1, bias=False),
            _norm(in_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        pyramids = [x]
        for b, conv in zip(self.bins, self.pool_convs):
            pool_size = min(b, H, W)
            pooled = F.adaptive_avg_pool2d(x, output_size=pool_size)
            pooled = conv(pooled)
            pyramids.append(F.interpolate(
                pooled, size=(H, W), mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(pyramids, dim=1))


class UPerNetHead(nn.Module):
    """
    UPerNet segmentation head for transformer features.

    Takes multi-layer features from the transformer, reshapes to spatial,
    applies FPN fusion + PPM, outputs segmentation logits.
    """
    def __init__(self, emb_dim, num_classes, num_patches_side, patch_size,
                 fpn_dim=256):
        super().__init__()
        self.num_patches_side = num_patches_side
        self.patch_size = patch_size
        self.target_size = num_patches_side * patch_size

        # PPM on final features
        self.ppm = PPM(emb_dim, emb_dim // 4)

        # FPN lateral convolutions (for multi-layer features)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(emb_dim, fpn_dim, kernel_size=1, bias=False),
                _norm(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                _norm(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        # Final fusion
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            _norm(fpn_dim),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.cls_head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

    def forward(self, layer_features):
        """
        Args:
            layer_features: list of 4 tensors, each [B, T, D]
                from different transformer layers.

        Returns:
            logits: [B, num_classes, H, W]
        """
        H_p = W_p = self.num_patches_side

        # Reshape each layer's output to spatial: [B, D, H_p, W_p]
        spatial_features = []
        for feat in layer_features:
            feat_2d = rearrange(feat, 'b (h w) d -> b d h w', h=H_p, w=W_p)
            spatial_features.append(feat_2d)

        # Apply PPM to last layer
        spatial_features[-1] = self.ppm(spatial_features[-1])

        # FPN: lateral + top-down
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, spatial_features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False)

        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Resize all to same size
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size,
                mode='bilinear', align_corners=False)

        # Fuse
        fused = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))

        # Classify + upsample
        logits = self.cls_head(fused)
        logits = F.interpolate(
            logits, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False)

        return logits


# =============================================================================
# SENPA SEGMENTATION MODEL
# =============================================================================

class SenPaSeg(nn.Module):
    """
    SenPa-MAE architecture adapted for supervised segmentation.

    Each spectral band is treated as a separate set of spatial tokens.
    Spectral response functions and GSD are encoded and added to
    patch embeddings, giving the model explicit sensor knowledge.

    Args:
        in_channels: Number of input bands.
        num_classes: Number of segmentation classes.
        img_size: Input image size (H = W).
        patch_size: Patch size for tokenization.
        emb_dim: Embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        wavelengths: List of central wavelengths (nm). If None, uses dummy.
        bandwidths: List of bandwidths (nm). If None, uses dummy.
        gsd: Ground sampling distance (m). Default 10.0.
    """

    def __init__(
        self,
        in_channels=242,
        num_classes=14,
        img_size=128,
        patch_size=8,
        emb_dim=256,
        num_layers=6,
        num_heads=8,
        wavelengths=None,
        bandwidths=None,
        gsd=10.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        assert img_size % patch_size == 0
        self.num_patches_side = img_size // patch_size
        self.num_patches = self.num_patches_side ** 2

        # ── Patch embedding (per-channel) ───────────────────────────
        self.patch_embed = nn.Linear(patch_size ** 2, emb_dim)
        self.patchify = Rearrange(
            'b c (h1 h) (w1 w) -> b c (h1 w1) (h w)',
            h1=self.num_patches_side,
            w1=self.num_patches_side,
        )

        # ── Positional embedding (shared across channels) ──────────
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, emb_dim)
        )
        trunc_normal_(self.pos_embedding, std=0.02)

        # ── Sensor parameter encoders ──────────────────────────────
        self.spectral_encoder = SpectralResponseEncoder(emb_dim)
        self.gsd_encoder = GSDEncoder(emb_dim)

        # ── Build and register response functions ──────────────────
        if wavelengths is not None and bandwidths is not None:
            rf = build_response_functions(wavelengths, bandwidths)
        else:
            # Dummy: uniform response across all wavelengths
            rf = torch.ones(in_channels, 2301) / 2301.0
        self.register_buffer("response_functions", rf)  # [C, 2301]
        self.default_gsd = gsd

        # ── Transformer encoder (separate blocks for feature extraction) ─
        self.transformer_blocks = nn.ModuleList(
            [Block(emb_dim, num_heads) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

        # Indices for multi-layer feature extraction (4 evenly spaced)
        self.feat_indices = self._get_feat_indices(num_layers)

        # ── UPerNet segmentation head ─────────────────────────────
        self.seg_head = UPerNetHead(
            emb_dim, num_classes,
            self.num_patches_side, patch_size,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SenPaSeg] in_channels={in_channels}, "
              f"num_classes={num_classes}, img_size={img_size}")
        print(f"[SenPaSeg] patch_size={patch_size}, "
              f"emb_dim={emb_dim}, layers={num_layers}")
        print(f"[SenPaSeg] num_patches={self.num_patches}, "
              f"tokens_per_image={in_channels * self.num_patches}")
        print(f"[SenPaSeg] UPerNet feat layers: {self.feat_indices}")
        print(f"[SenPaSeg] Parameters: {n_params:,}")

    @staticmethod
    def _get_feat_indices(num_layers):
        """Pick 4 evenly spaced layer indices for multi-scale features."""
        if num_layers <= 4:
            return list(range(num_layers))
        step = num_layers / 4
        return [int(round(step * (i + 1))) - 1 for i in range(4)]

    def forward(self, image, response_functions=None, gsd=None):
        """
        Args:
            image: [B, C, H, W]
            response_functions: [B, C, 2301] or None (uses registered)
            gsd: [B, C] or None (uses default)

        Returns:
            logits: [B, num_classes, H, W]
        """
        B, C, H, W = image.shape

        # ── Patchify ───────────────────────────────────────────────
        patches = self.patchify(image)
        patches = self.patch_embed(patches)

        # ── Positional embedding ───────────────────────────────────
        patches = patches + self.pos_embedding

        # ── Sensor parameter embeddings ────────────────────────────
        if response_functions is None:
            rf = self.response_functions[:C].unsqueeze(0).expand(B, -1, -1)
        else:
            rf = response_functions

        if gsd is None:
            gsd_val = torch.full((B, C), self.default_gsd,
                                 device=image.device, dtype=image.dtype)
        else:
            gsd_val = gsd

        spectral_emb = self.spectral_encoder(rf)
        gsd_emb = self.gsd_encoder(gsd_val)

        patches = patches + spectral_emb[:, :, None, :]
        patches = patches + gsd_emb[:, :, None, :]

        # ── Flatten channels into token sequence ───────────────────
        tokens = rearrange(patches, 'b c t d -> b (c t) d')

        # ── Transformer with multi-layer feature extraction ────────
        layer_features = []
        for i, block in enumerate(self.transformer_blocks):
            tokens = block(tokens)
            if i in self.feat_indices:
                # Pool across channels: [B, C*T, D] → [B, T, D]
                normed = self.layer_norm(tokens)
                pooled = rearrange(normed, 'b (c t) d -> b c t d', c=C)
                pooled = pooled.mean(dim=1)  # [B, T, D]
                layer_features.append(pooled)

        # Pad if fewer layers than 4
        while len(layer_features) < 4:
            layer_features.append(layer_features[-1])

        # ── UPerNet segmentation head ─────────────────────────────
        logits = self.seg_head(layer_features)

        return logits