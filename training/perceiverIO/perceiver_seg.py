"""
Perceiver Segmentation Model — Multi-Task / Multi-Temporal
============================================================

Wraps PerceiverIO for dense pixel-wise segmentation, with native
support for multi-temporal inputs.

Input:  [B, C, H, W]             single-frame
     or [B, T, C, H, W]           multi-temporal
        + doy: [B, T] long        day-of-year per frame (multi-temporal)
                                  or None (single-frame uses learned vec)

Output: [B, num_classes, H, W]    logits (single-frame)
     or [B, T, num_classes, H, W] logits (multi-temporal)

Pipeline:
    Multi-frame (T >= 1):
        1. Flatten: [B, T, C, H, W] -> [B, T*H*W, C]
        2. Per-token spatial position from Fourier (varies per pixel,
           tied across frames at same pixel).
        3. Per-token time encoding from doy, broadcast to all H*W
           pixels of frame t. If doy is None, use a learned vector.
        4. Concat [reflectance, pos, time] -> input tokens.
        5. Build queries: exact 1:1 match with input tokens
           query_input = tokens -> Linear -> [T*H*W, query_dim]
        6. PerceiverIO.encode(tokens).decode(queries)
           -> [B, T*H*W, num_classes]
        7. Reshape -> [B, num_classes, H, W] or [B, T, num_classes, H, W]

Dense Token Queries: By utilizing the exact input features (reflectance +
spatial pos + time) as queries, the decoder acts as a structural
skip-connection. This forces the latent cross-attention to anchor
directly to the raw input space, which is highly effective for dense
reconstruction and resolving autoencoder blurriness.

Single learned time vector: shared across all tasks. Used for
single-frame tasks (no real time).
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .fourier import FourierPositionalEncoding, FourierTimeEncoding
from .perceiver_io import PerceiverIO


class PerceiverSeg(nn.Module):
    """
    Perceiver-IO for semantic segmentation, with native multi-temporal support.

    Args:
        in_channels: Number of input bands per frame (e.g., 15 for the
                     canonical 13 S2 + 2 SAR layout).
        num_classes: Number of segmentation classes (or output channels).
        img_size: Spatial size (H = W).
        num_latents, latent_dim, depth, etc.: standard Perceiver-IO config.
        num_freq_bands, max_freq: Fourier-encoder config (shared by
                                  position and time).
    """

    def __init__(
        self,
        in_channels=15,
        num_classes=14,
        img_size=512,
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

        # ── Encoders ──────────────────────────────────────
        self.pos_encoder = FourierPositionalEncoding(
            num_bands=num_freq_bands, max_freq=max_freq,
        )
        self.time_encoder = FourierTimeEncoding(
            num_bands=num_freq_bands, max_freq=max_freq,
        )
        pos_dim = self.pos_encoder.get_output_dim()
        time_dim = self.time_encoder.get_output_dim()

        # ── Learned "no time" vector ─────────────────────
        # Used for single-frame inputs (no real DOY).
        self.no_time_vector = nn.Parameter(torch.zeros(time_dim))
        nn.init.normal_(self.no_time_vector, std=0.02)

        # ── Token / query dims ───────────────────────────
        # Token = [reflectance(C), pos, time]
        input_dim = in_channels + pos_dim + time_dim

        # Query = projection of the exact same features [reflectance, pos, time]
        query_dim = latent_dim
        self.query_proj = nn.Linear(input_dim, query_dim)

        # ── Core Perceiver-IO ────────────────────────────
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
              f"(channels={in_channels} + pos={pos_dim} + time={time_dim})")
        print(f"[PerceiverSeg] latents={num_latents}x{latent_dim}, "
              f"depth={depth}, self_per_cross={self_per_cross_attn}")
        print(f"[PerceiverSeg] Parameters: {n_params:,}")

    # ─────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────

    def forward(self, image: torch.Tensor, doy: torch.Tensor = None):
        """
        Args:
            image: [B, C, H, W]            (single-frame)
                or [B, T, C, H, W]         (multi-temporal)
            doy:   [B, T] long             day-of-year per frame (only used
                                            when image is 5D)
                or None                    fall back to learned vector

        Returns:
            logits: [B, num_classes, H, W]       (if input was 4D)
                 or [B, T, num_classes, H, W]    (if input was 5D)
        """
        # ── Normalize input rank ─────────────────────────
        if image.dim() == 4:
            # [B, C, H, W] -> [B, T=1, C, H, W]
            image = image.unsqueeze(1)
            T = 1
        elif image.dim() == 5:
            T = image.shape[1]
        else:
            raise ValueError(
                f"Expected 4D or 5D image, got {image.dim()}D shape "
                f"{tuple(image.shape)}"
            )
        B, _, C, H, W = image.shape

        # ── Spatial position encoding (varies per pixel, tied across T) ──
        pos = self.pos_encoder(H, W, device=image.device, dtype=image.dtype)
        # pos: [H*W, pos_dim]

        # ── Per-token time encoding ──────────────────────
        # Either Fourier-encoded DOY (T frames -> T distinct vectors,
        # broadcast over H*W pixels) or the learned vector.
        if doy is not None and T > 1:
            # doy: [B, T] -> [B, T, time_dim]
            time_per_frame = self.time_encoder(doy.to(image.device))
            time_per_frame = time_per_frame.to(image.dtype)
        else:
            # Broadcast the learned vector to [B, T, time_dim]
            time_per_frame = self.no_time_vector.to(image.dtype) \
                .view(1, 1, -1) \
                .expand(B, T, -1)

        # ── Build input tokens: [B, T*H*W, C + pos_dim + time_dim] ──
        # 1. Reflectance: [B, T, C, H, W] -> [B, T*H*W, C]
        pixels = rearrange(image, 'b t c h w -> b (t h w) c')

        # 2. Position: [H*W, pos_dim] -> [B, T*H*W, pos_dim]
        #    Same pos repeated for every (t, h, w) at fixed (h, w).
        pos_for_tokens = repeat(pos, '(h w) d -> b (t h w) d', b=B, t=T, h=H, w=W)

        # 3. Time: [B, T, time_dim] -> [B, T*H*W, time_dim]
        #    Same time vector for all H*W pixels of frame t.
        time_for_tokens = repeat(
            time_per_frame, 'b t d -> b (t h w) d', h=H, w=W,
        )

        tokens = torch.cat([pixels, pos_for_tokens, time_for_tokens], dim=-1)

        # ── Build queries: Exact same as tokens ──
        queries = self.query_proj(tokens)

        # ── Perceiver-IO ─────────────────────────────────
        logits = self.perceiver(tokens, queries=queries)
        # Output shape: [B, T*H*W, num_classes]

        # ── Reshape ──────────────────────────────────────
        if T > 1:
            return rearrange(logits, 'b (t h w) c -> b t c h w', t=T, h=H, w=W)
        else:
            # Drop 't' from the einops pattern entirely since T=1
            return rearrange(logits, 'b (h w) c -> b c h w', h=H, w=W)
