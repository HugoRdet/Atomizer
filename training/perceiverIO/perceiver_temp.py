"""
Perceiver Segmentation Model — PASTIS (per-pixel, time-aggregated)
==================================================================

A dedicated Perceiver-IO for PASTIS crop-type segmentation, where the target
is ONE label per pixel (time-aggregated). Differs from PerceiverSeg:

  - Encoder: unchanged — one token per (t, h, w), time encoded in the token.
  - Query:   ONE query per PIXEL (h, w), not per (t, h, w). The query is the
             faithful Perceiver-IO "input feature at the query location",
             instantiated for temporal inputs as a 1x1 conv (single Linear)
             over the channel-concatenated T*C features of that pixel,
             concatenated with the pixel's position encoding.
  - Output:  [B, num_classes, H, W]  (one label per pixel).

The pixel's T temporal features are sliced DIRECTLY from the image tensor
(image[:, :, :, h, w] -> [B, T, C]); no gather index is needed because the
input is gridded.

TEMPORAL DROPOUT (the experiment):
  - Encoder: dropped-timestep tokens are MASKED out of the token set (natural).
  - Query 1x1 conv: FIXED-T, so dropped timesteps are ZERO-FILLED back to T
    (forced by the channel-concat conv). Pass `frame_valid` [B, T] to mark
    real vs dropped frames; the conv zeroes dropped slots before concat.
  This token-removal (encoder) + zero-fill (query conv) is the architectural
  rigidity the experiment measures, shared with channel-concat baselines
  (ResNet TimeMerge); Atomiser by contrast removes tokens everywhere.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .fourier import FourierPositionalEncoding, FourierTimeEncoding
from .perceiver_io import PerceiverIO


class PerceiverSegPASTIS(nn.Module):
    def __init__(
        self,
        in_channels=13,
        num_classes=20,
        img_size=128,
        num_frames=10,                 # fixed T (channel-concat width = T*C)
        agg_dim=128,                   # 1x1 conv aggregator output dim
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
        self.img_size    = img_size
        self.num_frames  = num_frames

        # ── Encoders ──────────────────────────────────────
        self.pos_encoder  = FourierPositionalEncoding(num_bands=num_freq_bands, max_freq=max_freq)
        self.time_encoder = FourierTimeEncoding(num_bands=num_freq_bands, max_freq=max_freq)
        pos_dim  = self.pos_encoder.get_output_dim()
        time_dim = self.time_encoder.get_output_dim()

        self.no_time_vector = nn.Parameter(torch.zeros(time_dim))
        nn.init.normal_(self.no_time_vector, std=0.02)

        # encoder token = [reflectance(C), pos, time]
        input_dim = in_channels + pos_dim + time_dim
        query_dim = latent_dim

        # ── 1x1 conv temporal aggregator for the QUERY ───────────────
        # pure 1x1 conv over channel-concat T*C  ==  single Linear(T*C -> agg_dim)
        self.temporal_agg = nn.Linear(in_channels * num_frames, agg_dim)
        # query = proj(concat(pos, agg))
        self.query_proj = nn.Linear(pos_dim + agg_dim, query_dim)
        self.query_norm = nn.LayerNorm(query_dim)

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
        print(f"[PerceiverSegPASTIS] in_channels={in_channels}, num_classes={num_classes}, "
              f"img_size={img_size}, T={num_frames}")
        print(f"[PerceiverSegPASTIS] encoder input_dim={input_dim} "
              f"(C={in_channels} + pos={pos_dim} + time={time_dim})")
        print(f"[PerceiverSegPASTIS] query: 1x1conv(T*C={in_channels*num_frames}->{agg_dim}) "
              f"+ pos({pos_dim}) -> {query_dim}")
        print(f"[PerceiverSegPASTIS] latents={num_latents}x{latent_dim}, depth={depth}")
        print(f"[PerceiverSegPASTIS] Parameters: {n_params:,}")

    def forward(self, image: torch.Tensor, doy: torch.Tensor = None,
                frame_valid: torch.Tensor = None):
        """
        Args:
            image:       [B, T, C, H, W]   multi-temporal (or [B, C, H, W] -> T=1)
            doy:         [B, T] long        day-of-year per frame
            frame_valid: [B, T] bool        True = real frame, False = dropped
                                            (dropped frames are masked in the
                                            encoder and zero-filled in the query
                                            conv). If None, all frames are valid.

        Returns:
            logits: [B, num_classes, H, W]
        """
        if image.dim() == 4:
            image = image.unsqueeze(1)
        B, T, C, H, W = image.shape
        assert T == self.num_frames, (
            f"PerceiverSegPASTIS built for T={self.num_frames} but got T={T}. "
            f"The query 1x1 conv is fixed-T; zero-fill dropped frames to "
            f"T={self.num_frames} and mark them in frame_valid.")
        device, dtype = image.device, image.dtype

        if frame_valid is None:
            frame_valid = torch.ones(B, T, dtype=torch.bool, device=device)

        # ── Position & time encodings ────────────────────────────────
        pos = self.pos_encoder(H, W, device=device, dtype=dtype)     # [H*W, pos_dim]

        if doy is not None and T > 1:
            time_per_frame = self.time_encoder(doy.to(device)).to(dtype)   # [B, T, time_dim]
        else:
            time_per_frame = self.no_time_vector.to(dtype).view(1, 1, -1).expand(B, T, -1)

        # ── ENCODER tokens: one per (t, h, w) ────────────────────────
        pixels = rearrange(image, 'b t c h w -> b (t h w) c')
        pos_for_tokens  = repeat(pos, '(h w) d -> b (t h w) d', b=B, t=T, h=H, w=W)
        time_for_tokens = repeat(time_per_frame, 'b t d -> b (t h w) d', h=H, w=W)
        tokens = torch.cat([pixels, pos_for_tokens, time_for_tokens], dim=-1)  # [B, T*H*W, input_dim]

        # encoder mask: drop tokens belonging to invalid (dropped) frames.
        # frame_valid [B, T] -> token mask [B, T*H*W] (True = keep / attend)
        tok_mask = repeat(frame_valid, 'b t -> b (t h w)', h=H, w=W)

        # ── QUERY: one per pixel (h, w), 1x1 conv over that pixel's T*C ──
        # slice the pixel's T temporal features directly from the image:
        # image [B, T, C, H, W] -> per-pixel [B, H*W, T, C]
        pix_tc = rearrange(image, 'b t c h w -> b (h w) t c')           # [B, H*W, T, C]
        # zero-fill dropped frames in the query conv (fixed-T rigidity)
        fv = rearrange(frame_valid, 'b t -> b 1 t 1').to(dtype)         # [B,1,T,1]
        pix_tc = pix_tc * fv
        pix_concat = rearrange(pix_tc, 'b n t c -> b n (t c)')          # [B, H*W, T*C]
        agg = self.temporal_agg(pix_concat)                            # [B, H*W, agg_dim]

        pos_q = repeat(pos, '(h w) d -> b (h w) d', b=B, h=H, w=W)      # [B, H*W, pos_dim]
        q = torch.cat([pos_q, agg], dim=-1)                            # [B, H*W, pos+agg]
        q = self.query_norm(self.query_proj(q))                        # [B, H*W, query_dim]

        # ── Perceiver-IO with encoder mask + per-pixel queries ───────
        logits = self.perceiver(tokens, queries=q, mask=tok_mask)      # [B, H*W, num_classes]

        return rearrange(logits, 'b (h w) c -> b c h w', h=H, w=W)
