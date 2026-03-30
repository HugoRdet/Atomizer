"""
LTAE — Lightweight Temporal Attention Encoder
==============================================

Temporal aggregation module from Garnot & Landrieu (2020, 2021).
Takes a sequence of per-pixel feature maps and produces a single
temporally-aggregated feature map via multi-head attention.

Components:
    LTAE:         Core temporal attention (operates on [B, T, d])
    TemporalUNet: UNet encoder + LTAE + UNet decoder (U-TAE)

Input flow (TemporalUNet):
    [B, T, C, H, W]
        → per-frame UNet encoder → [B, T, C', H', W'] at each scale
        → LTAE at bottleneck     → [B, C', H', W']
        → UNet decoder           → [B, num_classes, H, W]

References:
    - Garnot & Landrieu, "Lightweight Temporal Self-Attention for
      Classifying Satellite Image Time Series", 2020
    - Garnot & Landrieu, "Panoptic Segmentation of Satellite Image
      Time Series with Convolutional Temporal Attention Networks", 2021
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from day-of-year.

    Maps integer day-of-year (1..366) to a d-dimensional vector using
    sin/cos at geometrically spaced frequencies.

    If no dates are provided, falls back to uniform spacing.
    """

    def __init__(self, d_model: int, max_len: int = 366):
        super().__init__()
        self.d_model = d_model

        # Precompute frequency table
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe)  # [max_len+1, d_model]

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            doy: [B, T] integer day-of-year values (0..366)
        Returns:
            [B, T, d_model] positional embeddings
        """
        doy = doy.clamp(0, self.pe.shape[0] - 1).long()
        return self.pe[doy]  # [B, T, d_model]


# =============================================================================
# LTAE — LIGHTWEIGHT TEMPORAL ATTENTION ENCODER
# =============================================================================

class LTAE(nn.Module):
    """
    Lightweight Temporal Attention Encoder.

    Aggregates a sequence of d-dimensional features across time using
    multi-head attention with learned query vectors.

    Args:
        in_channels:  Input feature dimension (d)
        n_heads:      Number of attention heads
        d_k:          Key dimension per head (default: 4)
        d_model:      Internal projection dimension (default: 256)
        dropout:       Dropout on attention weights
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_heads: int = 16,
        d_k: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_model = d_model

        # Input projection: d → d_model
        self.fc_in = nn.Linear(in_channels, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Learned query: one per head
        self.query = nn.Parameter(torch.randn(n_heads, d_k))
        nn.init.normal_(self.query, mean=0, std=1.0 / math.sqrt(d_k))

        # Key projection (per head): d_model → n_heads * d_k
        self.fc_k = nn.Linear(d_model, n_heads * d_k)

        # Value projection: d_model → n_heads * d_k
        self.fc_v = nn.Linear(d_model, n_heads * d_k)

        # Output projection: n_heads * d_k → in_channels
        self.fc_out = nn.Linear(n_heads * d_k, in_channels)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(
        self,
        x: torch.Tensor,
        doy: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, T, d] — temporal sequence of features
            doy:      [B, T]   — day-of-year (optional, for positional encoding)
            pad_mask: [B, T]   — True = valid, False = padded (optional)

        Returns:
            out:      [B, d] — temporally aggregated features
            attn:     [B, n_heads, T] — attention weights (for viz)
        """
        B, T, d = x.shape

        # Project to d_model
        x_proj = self.fc_in(x)  # [B, T, d_model]

        # Add positional encoding
        if doy is not None:
            x_proj = x_proj + self.pos_enc(doy)
        else:
            # Fallback: uniform spacing
            uniform_doy = torch.linspace(0, 365, T, device=x.device).long()
            uniform_doy = uniform_doy.unsqueeze(0).expand(B, -1)
            x_proj = x_proj + self.pos_enc(uniform_doy)

        # Keys and Values
        K = self.fc_k(x_proj)  # [B, T, n_heads * d_k]
        V = self.fc_v(x_proj)  # [B, T, n_heads * d_k]

        # Reshape for multi-head: [B, n_heads, T, d_k]
        K = K.view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Query: [1, n_heads, 1, d_k] → broadcast over batch
        Q = self.query.unsqueeze(0).unsqueeze(2)  # [1, n_heads, 1, d_k]

        # Attention scores: [B, n_heads, 1, T]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Mask padded timesteps
        if pad_mask is not None:
            # pad_mask: [B, T], True = valid → invert for masking
            mask = ~pad_mask  # True = padded
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)  # [B, n_heads, 1, T]
        attn = self.dropout(attn)

        # Weighted sum of values: [B, n_heads, 1, d_k]
        out = torch.matmul(attn, V)

        # Reshape: [B, n_heads * d_k]
        out = out.squeeze(2).reshape(B, self.n_heads * self.d_k)

        # Output projection + residual (if dims match)
        out = self.fc_out(out)  # [B, d]

        attn_weights = attn.squeeze(2)  # [B, n_heads, T]

        return out, attn_weights


# =============================================================================
# SPATIAL-TEMPORAL LTAE WRAPPER
# =============================================================================

class SpatioTemporalLTAE(nn.Module):
    """
    Applies LTAE pixel-wise over spatial feature maps.

    Input:  [B, T, C, H, W]
    Output: [B, C, H, W]

    Reshapes spatial dims to batch dim, runs LTAE, reshapes back.
    """

    def __init__(self, in_channels: int, n_heads: int = 16, d_k: int = 4,
                 d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ltae = LTAE(
            in_channels=in_channels,
            n_heads=n_heads,
            d_k=d_k,
            d_model=d_model,
            dropout=dropout,
        )

    def forward(self, x, doy=None, pad_mask=None):
        """
        Args:
            x:        [B, T, C, H, W]
            doy:      [B, T]
            pad_mask: [B, T]
        Returns:
            out:  [B, C, H, W]
            attn: [B, n_heads, T, H, W]
        """
        B, T, C, H, W = x.shape

        # Reshape: merge spatial dims into batch → [B*H*W, T, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        # Expand doy and mask to match
        if doy is not None:
            doy_flat = doy.unsqueeze(1).unsqueeze(1).expand(B, H, W, T)
            doy_flat = doy_flat.reshape(B * H * W, T)
        else:
            doy_flat = None

        if pad_mask is not None:
            mask_flat = pad_mask.unsqueeze(1).unsqueeze(1).expand(B, H, W, T)
            mask_flat = mask_flat.reshape(B * H * W, T)
        else:
            mask_flat = None

        # LTAE: [B*H*W, T, C] → [B*H*W, C]
        out_flat, attn_flat = self.ltae(x_flat, doy_flat, mask_flat)

        # Reshape back: [B, H, W, C] → [B, C, H, W]
        out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Attention: [B*H*W, n_heads, T] → [B, n_heads, T, H, W]
        n_heads = attn_flat.shape[1]
        attn = attn_flat.reshape(B, H, W, n_heads, T).permute(0, 3, 4, 1, 2)

        return out, attn


# =============================================================================
# UNET BUILDING BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """Two 3×3 conv + BN + ReLU."""
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample → concat skip → ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        if dy > 0 or dx > 0:
            x = F.pad(x, [0, dx, 0, dy])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# TEMPORAL UNET (U-TAE)
# =============================================================================

class TemporalUNet(nn.Module):
    """
    U-TAE: UNet encoder + LTAE temporal aggregation + UNet decoder.

    Per-frame spatial encoding, LTAE aggregation at bottleneck,
    then spatial decoding to produce segmentation logits.

    Args:
        in_channels:  Number of input bands per frame (e.g., 10 for S2)
        num_classes:  Number of segmentation classes
        base_channels: Base channel width (doubled at each encoder level)
        n_heads:      LTAE attention heads
        d_k:          Key dimension per LTAE head
        d_model:      LTAE internal projection dimension
    """

    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 20,
        base_channels: int = 64,
        n_heads: int = 16,
        d_k: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        c = base_channels

        # ── Encoder (applied per-frame) ─────────────────────────────
        self.enc1 = ConvBlock(in_channels, c)         # → [c, H, W]
        self.enc2 = DownBlock(c, c * 2)               # → [2c, H/2, W/2]
        self.enc3 = DownBlock(c * 2, c * 4)           # → [4c, H/4, W/4]
        self.enc4 = DownBlock(c * 4, c * 8)           # → [8c, H/8, W/8]

        # ── LTAE at bottleneck ──────────────────────────────────────
        self.ltae = SpatioTemporalLTAE(
            in_channels=c * 8,
            n_heads=n_heads,
            d_k=d_k,
            d_model=d_model,
            dropout=dropout,
        )

        # ── Decoder ─────────────────────────────────────────────────
        # Skip connections use temporal mean of encoder features
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)     # → [4c, H/4, W/4]
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)     # → [2c, H/2, W/2]
        self.dec1 = UpBlock(c * 2, c, c)              # → [c, H, W]

        # ── Classification head ─────────────────────────────────────
        self.head = nn.Conv2d(c, num_classes, 1)

    def _encode_frame(self, x):
        """Encode a single frame. Returns features at each scale."""
        e1 = self.enc1(x)   # [B, c, H, W]
        e2 = self.enc2(e1)  # [B, 2c, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 4c, H/4, W/4]
        e4 = self.enc4(e3)  # [B, 8c, H/8, W/8]
        return e1, e2, e3, e4

    def forward(self, x, doy=None, pad_mask=None):
        """
        Args:
            x:        [B, T, C, H, W] — temporal image sequence
            doy:      [B, T] — day-of-year per frame (optional)
            pad_mask: [B, T] — True = valid frame (optional)

        Returns:
            logits: [B, num_classes, H, W]
        """
        B, T, C, H, W = x.shape

        # ── Encode each frame ───────────────────────────────────────
        # Merge B and T for efficient batched encoding
        x_flat = x.reshape(B * T, C, H, W)

        e1, e2, e3, e4 = self._encode_frame(x_flat)

        # Reshape back to [B, T, ...]
        def unflatten(feat):
            _, c, h, w = feat.shape
            return feat.reshape(B, T, c, h, w)

        e1_t = unflatten(e1)  # [B, T, c, H, W]
        e2_t = unflatten(e2)  # [B, T, 2c, H/2, W/2]
        e3_t = unflatten(e3)  # [B, T, 4c, H/4, W/4]
        e4_t = unflatten(e4)  # [B, T, 8c, H/8, W/8]

        # ── Temporal aggregation at bottleneck ──────────────────────
        bottleneck, attn = self.ltae(e4_t, doy=doy, pad_mask=pad_mask)
        # bottleneck: [B, 8c, H/8, W/8]

        # ── Skip connections: temporal mean ─────────────────────────
        if pad_mask is not None:
            # Weighted mean using valid mask
            mask_w = pad_mask.float()  # [B, T]
            mask_sum = mask_w.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            mask_w = mask_w / mask_sum  # [B, T] normalized

            def masked_mean(feat_t):
                # feat_t: [B, T, C, H, W], mask_w: [B, T]
                w = mask_w[:, :, None, None, None]  # [B, T, 1, 1, 1]
                return (feat_t * w).sum(dim=1)
        else:
            def masked_mean(feat_t):
                return feat_t.mean(dim=1)

        skip3 = masked_mean(e3_t)  # [B, 4c, H/4, W/4]
        skip2 = masked_mean(e2_t)  # [B, 2c, H/2, W/2]
        skip1 = masked_mean(e1_t)  # [B, c, H, W]

        # ── Decode ──────────────────────────────────────────────────
        d3 = self.dec3(bottleneck, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.head(d1)