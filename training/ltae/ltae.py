"""
LTAE — Lightweight Temporal Attention Encoder
==============================================

Temporal aggregation module from Garnot & Landrieu (2020).
Takes a sequence of per-pixel feature maps and produces a single
temporally-aggregated feature map via multi-head attention with
learned queries.

Components:
    LTAE:                Core temporal attention (operates on [B, T, d])
    SpatioTemporalLTAE:  Pixel-wise LTAE on [B, T, C, H, W] feature maps
    UNetLTAE:            Classic UNet (per-frame) + LTAE at output resolution

Architecture (UNetLTAE):
    [B, T, C, H, W]
        → per-frame UNet encoder (shared)         → [B, T, F, H, W]
        → per-frame UNet decoder (shared)         → [B, T, base_ch, H, W]
        → SpatioTemporalLTAE (per-pixel)          → [B, base_ch, H, W]
        → 1×1 classification head                 → [B, num_classes, H, W]

Reference:
    - Garnot & Landrieu, "Lightweight Temporal Self-Attention for
      Classifying Satellite Image Time Series", 2020
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.unet.model_unet import UNetEncoder, UNetDecoder


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
        return self.pe[doy]


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
        d_k:          Key dimension per head
        d_model:      Internal projection dimension
        dropout:      Dropout on attention weights
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

        self.fc_in = nn.Linear(in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Learned query: one per head
        self.query = nn.Parameter(torch.randn(n_heads, d_k))
        nn.init.normal_(self.query, mean=0, std=1.0 / math.sqrt(d_k))

        self.fc_k = nn.Linear(d_model, n_heads * d_k)
        self.fc_v = nn.Linear(d_model, n_heads * d_k)
        self.fc_out = nn.Linear(n_heads * d_k, in_channels)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(
        self,
        x: torch.Tensor,
        doy: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
    ):
        """
        Args:
            x:        [B, T, d]
            doy:      [B, T] (optional)
            pad_mask: [B, T] True=valid, False=padded (optional)
        Returns:
            out:  [B, d]
            attn: [B, n_heads, T]
        """
        B, T, d = x.shape

        x_proj = self.fc_in(x)  # [B, T, d_model]

        # Positional encoding
        if doy is not None:
            x_proj = x_proj + self.pos_enc(doy)
        else:
            uniform_doy = torch.linspace(0, 365, T, device=x.device).long()
            uniform_doy = uniform_doy.unsqueeze(0).expand(B, -1)
            x_proj = x_proj + self.pos_enc(uniform_doy)

        # Keys and Values: [B, T, n_heads, d_k] → [B, n_heads, T, d_k]
        K = self.fc_k(x_proj).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.fc_v(x_proj).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Query: [1, n_heads, 1, d_k] — broadcast over batch
        Q = self.query.unsqueeze(0).unsqueeze(2)

        # Attention: [B, n_heads, 1, T]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if pad_mask is not None:
            mask = ~pad_mask  # True = padded
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: [B, n_heads, 1, d_k] → [B, n_heads * d_k]
        out = torch.matmul(attn, V).squeeze(2).reshape(B, self.n_heads * self.d_k)
        out = self.fc_out(out)  # [B, in_channels]

        return out, attn.squeeze(2)  # attn: [B, n_heads, T]


# =============================================================================
# SPATIO-TEMPORAL LTAE WRAPPER
# =============================================================================

class SpatioTemporalLTAE(nn.Module):
    """
    Applies LTAE pixel-wise over spatial feature maps.

    Input:  [B, T, C, H, W]
    Output: [B, C, H, W]
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

        # [B, T, C, H, W] → [B*H*W, T, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        # Expand doy / mask to match
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

        out_flat, attn_flat = self.ltae(x_flat, doy_flat, mask_flat)

        # Back to spatial: [B, H, W, C] → [B, C, H, W]
        out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Attention: [B*H*W, n_heads, T] → [B, n_heads, T, H, W]
        n_heads = attn_flat.shape[1]
        attn = attn_flat.reshape(B, H, W, n_heads, T).permute(0, 3, 4, 1, 2)

        return out, attn


# =============================================================================
# UNET + LTAE — Classic UNet with temporal aggregation at output resolution
# =============================================================================

class UNetLTAE(nn.Module):
    """
    Classic UNet (per-frame, shared weights) + LTAE temporal aggregation
    at full output resolution + classification head.

    Differs from U-TAE:
        - U-TAE places LTAE at the UNet bottleneck (low spatial res, high channels)
        - UNetLTAE places LTAE AFTER full UNet encode-decode (full res)
        - U-TAE uses temporally-attended skip connections
        - UNetLTAE has no temporal handling in skips — each frame goes through
          the full UNet independently, then per-pixel LTAE aggregates outputs

    Forward:
        [B, T, C, H, W]
            → reshape: [B*T, C, H, W]
            → UNet encoder + decoder (shared weights, per-frame)
            → reshape: [B, T, base_ch, H, W]
            → SpatioTemporalLTAE → [B, base_ch, H, W]
            → 1×1 conv head → [B, num_classes, H, W]

    Args:
        in_channels:  Input bands per frame (e.g., 10 for S2, 12 for S2+S1)
        num_classes:  Output segmentation classes
        topology:     UNet feature widths per level
        n_heads:      LTAE attention heads
        d_k:          LTAE key dim per head
        d_model:      LTAE internal projection dim
        dropout:      LTAE attention dropout
    """

    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 20,
        topology=(64, 128, 256, 512, 1024),
        n_heads: int = 16,
        d_k: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # UNet without classification head — we want pre-logit features
        self.encoder = UNetEncoder(in_channels, topology)
        self.decoder = UNetDecoder(topology)

        base_channels = topology[0]

        # Temporal aggregation at full output resolution
        self.ltae = SpatioTemporalLTAE(
            in_channels=base_channels,
            n_heads=n_heads,
            d_k=d_k,
            d_model=d_model,
            dropout=dropout,
        )

        # Final classification head (after temporal aggregation)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x, doy=None, pad_mask=None):
        """
        Args:
            x:        [B, T, C, H, W]
            doy:      [B, T] day-of-year (optional)
            pad_mask: [B, T] True=valid (optional)
        Returns:
            logits: [B, num_classes, H, W]
        """
        B, T, C, H, W = x.shape

        # Per-frame UNet (shared weights, batched via merging B and T)
        x_flat = x.reshape(B * T, C, H, W)
        feats_flat = self.decoder(self.encoder(x_flat))  # [B*T, base_ch, H, W]

        # Reshape back to temporal: [B, T, base_ch, H, W]
        base_ch = feats_flat.shape[1]
        feats_t = feats_flat.reshape(B, T, base_ch, H, W)

        # Per-pixel LTAE: [B, T, base_ch, H, W] → [B, base_ch, H, W]
        agg, _attn = self.ltae(feats_t, doy=doy, pad_mask=pad_mask)

        # Classify
        return self.head(agg)