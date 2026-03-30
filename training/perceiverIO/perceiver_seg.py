"""
Perceiver Segmentation Model
===============================

Wraps PerceiverIO for dense pixel-wise segmentation.

Input: [B, C, H, W] image (any channel count)
Output: [B, num_classes, H, W] logits

Pipeline:
    1. Flatten image: [B, C, H, W] → [B, H*W, C]
    2. Fourier encode (x, y) positions → [H*W, pos_dim]
    3. Concat: [B, H*W, C + pos_dim] → input tokens
    4. Build queries: same Fourier positions, projected → [B, H*W, query_dim]
    5. PerceiverIO: encode(tokens) → decode(queries) → [B, H*W, num_classes]
    6. Reshape: [B, num_classes, H, W]

For C2Seg baseline evaluation: same interface as UNet/ViT.
    model = PerceiverSeg(in_channels=242, num_classes=14, img_size=128)
    logits = model(image)  # [B, 14, 128, 128]
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .fourier import FourierPositionalEncoding
from .perceiver_io import PerceiverIO


class PerceiverSeg(nn.Module):
    """
    Perceiver-IO for semantic segmentation.

    Same interface as UNet/ViT baselines: forward(image) → logits.

    Args:
        in_channels: Number of input bands (e.g., 242 for EnMAP).
        num_classes: Number of segmentation classes.
        img_size: Expected spatial size (H = W). Used for query construction.
        num_latents: Number of latent vectors.
        latent_dim: Latent vector dimension.
        depth: Number of encoder blocks.
        cross_heads: Heads for cross-attention.
        latent_heads: Heads for self-attention.
        cross_dim_head: Dim per cross-attention head.
        latent_dim_head: Dim per self-attention head.
        self_per_cross_attn: Self-attention blocks per cross-attention.
        weight_tie_layers: Share weights across encoder blocks.
        num_freq_bands: Fourier frequency bands for positional encoding.
        max_freq: Maximum Fourier frequency.
        attn_dropout: Attention dropout.
        ff_dropout: Feedforward dropout.
    """

    def __init__(
        self,
        in_channels=242,
        num_classes=14,
        img_size=128,
        num_latents=256,
        latent_dim=256,
        depth=6,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        self_per_cross_attn=1,
        weight_tie_layers=True,
        num_freq_bands=6,
        max_freq=10.0,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Positional encoding
        self.pos_encoder = FourierPositionalEncoding(
            num_bands=num_freq_bands,
            max_freq=max_freq,
        )
        pos_dim = self.pos_encoder.get_output_dim()

        # Input: raw channels + fourier positions
        input_dim = in_channels + pos_dim

        # Query: fourier positions projected to query_dim
        query_dim = latent_dim
        self.query_proj = nn.Linear(pos_dim, query_dim)

        # Core Perceiver-IO
        self.perceiver = PerceiverIO(
            input_dim=input_dim,
            query_dim=query_dim,
            output_dim=num_classes,
            num_latents=num_latents,
            latent_dim=latent_dim,
            depth=depth,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            self_per_cross_attn=self_per_cross_attn,
            weight_tie_layers=weight_tie_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            decoder_ff=True,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[PerceiverSeg] in_channels={in_channels}, "
              f"num_classes={num_classes}, img_size={img_size}")
        print(f"[PerceiverSeg] input_dim={input_dim} "
              f"(channels={in_channels} + pos={pos_dim})")
        print(f"[PerceiverSeg] latents={num_latents}×{latent_dim}, "
              f"depth={depth}, self_per_cross={self_per_cross_attn}")
        print(f"[PerceiverSeg] Parameters: {n_params:,}")

    def forward(self, image):
        """
        Args:
            image: [B, C, H, W]

        Returns:
            logits: [B, num_classes, H, W]
        """
        B, C, H, W = image.shape

        # Flatten spatial: [B, C, H, W] → [B, H*W, C]
        pixels = rearrange(image, 'b c h w -> b (h w) c')

        # Positional encoding: [H*W, pos_dim]
        pos = self.pos_encoder(H, W, device=image.device, dtype=image.dtype)
        pos = repeat(pos, 'n d -> b n d', b=B)

        # Input tokens: [B, H*W, C + pos_dim]
        tokens = torch.cat([pixels, pos], dim=-1)

        # Queries: projected positions [B, H*W, query_dim]
        queries = self.query_proj(pos)

        # Perceiver-IO: encode + decode
        logits = self.perceiver(tokens, queries=queries)  # [B, H*W, num_classes]

        # Reshape to spatial: [B, num_classes, H, W]
        logits = rearrange(logits, 'b (h w) c -> b c h w', h=H, w=W)

        return logits