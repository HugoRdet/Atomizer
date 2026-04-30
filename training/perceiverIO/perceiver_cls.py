"""
Perceiver Classification Model
================================

Wraps PerceiverIO for image-level classification. Same input tokenization
as PerceiverSeg (one token per pixel, with Fourier position + time
encoding), but the decoder output is a single learned query token —
equivalent to attention-pooling over the latent set.

This mirrors Atomizer-IO's classification approach (single query token
attending to the processor latents), so the per-task comparison stays
clean: same latent-bottleneck mechanism, only difference is the
input tokenization.

Input:  [B, C, H, W]              single-frame
     or [B, T, C, H, W]           multi-temporal (rare for cls tasks)
        + doy: [B, T] long        day-of-year per frame, or None

Output: [B, num_classes]          classification logits

Pipeline:
    1. Build input tokens (identical to PerceiverSeg):
       per pixel-frame: [reflectance(C), Fourier(x, y), Fourier(t) or learned]
    2. Single learned query token: [1, query_dim]
       -> [B, 1, query_dim] via repeat
    3. PerceiverIO.encode(tokens).decode(query) -> [B, 1, num_classes]
    4. Squeeze -> [B, num_classes]
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .fourier import FourierPositionalEncoding, FourierTimeEncoding
from .perceiver_io import PerceiverIO


class PerceiverCls(nn.Module):
    """
    Perceiver-IO for image-level classification, with native multi-temporal support.

    Args:
        in_channels: Number of input bands per frame.
        num_classes: Number of classification classes.
        img_size:    Spatial size (H = W). Used for token positional encoding.
        num_latents, latent_dim, depth, etc.: standard Perceiver-IO config.
        num_freq_bands, max_freq: Fourier-encoder config (shared by
                                  position and time).
    """

    def __init__(
        self,
        in_channels=15,
        num_classes=10,
        img_size=64,
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
        # Used for single-frame inputs. Shared across all tasks.
        self.no_time_vector = nn.Parameter(torch.zeros(time_dim))
        nn.init.normal_(self.no_time_vector, std=0.02)

        # ── Token / query dims ───────────────────────────
        # Token = [reflectance(C), pos, time]
        input_dim = in_channels + pos_dim + time_dim
        query_dim = latent_dim

        # ── Single learned classification query token ────
        # Acts as a CLS token: attends to the processor latents
        # to pool a global representation for classification.
        self.cls_query = nn.Parameter(torch.zeros(1, query_dim))
        nn.init.normal_(self.cls_query, std=0.02)

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
        print(f"[PerceiverCls] in_channels={in_channels}, "
              f"num_classes={num_classes}, img_size={img_size}")
        print(f"[PerceiverCls] input_dim={input_dim} "
              f"(channels={in_channels} + pos={pos_dim} + time={time_dim})")
        print(f"[PerceiverCls] latents={num_latents}x{latent_dim}, "
              f"depth={depth}, self_per_cross={self_per_cross_attn}")
        print(f"[PerceiverCls] query: single learned CLS token (attention pool)")
        print(f"[PerceiverCls] Parameters: {n_params:,}")

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
            logits: [B, num_classes]
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
        pos_for_tokens = repeat(pos, '(h w) d -> b (t h w) d', b=B, t=T, h=H, w=W)

        # 3. Time: [B, T, time_dim] -> [B, T*H*W, time_dim]
        time_for_tokens = repeat(
            time_per_frame, 'b t d -> b (t h w) d', h=H, w=W,
        )

        tokens = torch.cat([pixels, pos_for_tokens, time_for_tokens], dim=-1)

        # ── Build single CLS query: [B, 1, query_dim] ────
        queries = self.cls_query.to(image.dtype) \
            .view(1, 1, -1) \
            .expand(B, 1, -1)

        # ── Perceiver-IO ─────────────────────────────────
        logits = self.perceiver(tokens, queries=queries)
        # [B, 1, num_classes]

        return logits.squeeze(1)
        # [B, num_classes]