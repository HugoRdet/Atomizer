"""
PerceiverDales
================

Wraps PerceiverIO for DALES LIDAR-only semantic segmentation.

Consumes the output of DalesPerceiverDataset:
    lidar_tokens [B, N_lidar, LIDAR_RAW_DIM=198]  Fourier(X,Y,Z) + echo(a,b) + intensity
    lidar_mask   [B, N_lidar]  bool, True=masked
    queries      [B, M, QUERY_DIM=195]  Fourier(X,Y,Z) query positions

DIFFERENCES FROM PerceiverFractal (documented explicitly, not left implicit):
  1. NO VHR modality -- DALES is LIDAR-only, so there's no second token
     stream to concatenate, and consequently NO PADDING needed to match
     another modality's width (FRACTAL's LIDAR_PAD_DIM doesn't exist
     here -- LIDAR's natural encoded width IS the input_dim).
  2. Intensity gets its OWN dedicated MLP (intensity_mlp), architecturally
     identical to echo_mlp (1/2 -> hidden -> out_dim, GELU, LayerNorm,
     zero-init final layer) -- matching Atomizer's DalesTokenProcessor,
     which encodes echo and intensity as two separate small MLPs rather
     than folding intensity into position or omitting it. This is a
     deliberate content addition since DALES has no VHR to compensate for
     omitting a real, informative per-point signal (see project history:
     confusion-matrix analysis suggested intensity-sensitive classes like
     power_lines/poles benefit from it, plausibly via metal reflectivity).

Processing pipeline
-------------------
1. Echo MLP:       lidar_tokens[..., 195:197]  -> [B, N_lidar, ECHO_MLP_OUT_DIM]
2. Intensity MLP:  lidar_tokens[..., 197:198]  -> [B, N_lidar, INTENSITY_MLP_OUT_DIM]
3. LIDAR full:      cat([Fourier(X,Y,Z), echo_out, intensity_out]) -> [B, N_lidar, INPUT_DIM]
4. Encode:          PerceiverEncoder(lidar_full, mask) -> latents [B, L, latent_dim]
5. Query proj:       queries [B, M, 195] -> [B, M, latent_dim]
6. Decode:          PerceiverDecoder(latents, query_proj) -> [B, M, num_classes]

Mask convention
---------------
DalesPerceiverDataset uses True=masked (padding), matching PyTorch's
MultiheadAttention key_padding_mask convention. PerceiverIO's encoder
forward() uses mask: True=valid. Flipped once on entry.
"""

import torch
import torch.nn as nn

from .perceiver_io import PerceiverIO

# -- Dimension constants (must match DalesPerceiverDataset) ----------------
LIDAR_FOURIER_DIM    = 195   # 3*65 -- Fourier(X,Y,Z) position
ECHO_SCALARS_DIM     = 2     # raw (a, b)
INTENSITY_SCALARS_DIM = 1    # raw normalized intensity
LIDAR_RAW_DIM = LIDAR_FOURIER_DIM + ECHO_SCALARS_DIM + INTENSITY_SCALARS_DIM  # 198

ECHO_MLP_OUT_DIM      = 49   # matches FRACTAL's echo_mlp out_dim, kept
                              # consistent across both PerceiverIO baselines
INTENSITY_MLP_OUT_DIM = 49   # symmetric with echo -- no strong reason for
                              # a different width, kept equal for simplicity

# NO padding needed -- unlike FRACTAL (which pads LIDAR to match VHR's
# width for concatenation), DALES has nothing to concatenate with.
INPUT_DIM = LIDAR_FOURIER_DIM + ECHO_MLP_OUT_DIM + INTENSITY_MLP_OUT_DIM  # 293
QUERY_DIM = 195   # 3*65 -- position-only queries


def _build_small_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Shared architecture for echo_mlp and intensity_mlp: in_dim -> hidden
    -> out_dim, GELU, LayerNorm, zero-init last layer -- matches Atomizer's
    echo_encoder/intensity_encoder convention exactly."""
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class PerceiverDales(nn.Module):
    """
    PerceiverIO for DALES LIDAR-only semantic segmentation.

    See module docstring for the full processing pipeline and explicit
    differences from PerceiverFractal.
    """

    def __init__(
        self,
        num_classes: int = 8,
        num_latents: int = 256,
        latent_dim: int = 256,
        depth: int = 6,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        self_per_cross_attn: int = 1,
        weight_tie_layers: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        echo_hidden_dim: int = 64,
        intensity_hidden_dim: int = 64,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.echo_mlp = _build_small_mlp(
            ECHO_SCALARS_DIM, echo_hidden_dim, ECHO_MLP_OUT_DIM
        )
        self.intensity_mlp = _build_small_mlp(
            INTENSITY_SCALARS_DIM, intensity_hidden_dim, INTENSITY_MLP_OUT_DIM
        )

        # No lidar_pad parameter -- see module docstring, nothing to pad
        # against without a second (VHR) modality.

        self.query_proj = nn.Linear(QUERY_DIM, latent_dim)

        self.perceiver = PerceiverIO(
            input_dim=INPUT_DIM,
            query_dim=latent_dim,
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
        print(f"[PerceiverDales] num_classes={num_classes}, "
              f"latents={num_latents}x{latent_dim}, depth={depth}")
        print(f"[PerceiverDales] input_dim={INPUT_DIM} "
              f"(position={LIDAR_FOURIER_DIM}, echo->{ECHO_MLP_OUT_DIM}, "
              f"intensity->{INTENSITY_MLP_OUT_DIM}) -- NO VHR, NO padding")
        print(f"[PerceiverDales] query_dim={QUERY_DIM} "
              f"-> projected to latent_dim={latent_dim}")
        print(f"[PerceiverDales] Parameters: {n_params:,}")

    def forward(
        self,
        batch: dict,
        training: bool = True,
        query_chunk_size: int = None,
    ) -> torch.Tensor:
        """
        Args:
            batch: dict from DalesPerceiverDataset DataLoader, with keys:
                lidar_tokens [B, N_lidar, 198]
                lidar_mask   [B, N_lidar]  bool  True=masked
                queries      [B, M, 195]
                queries_mask [B, M]  bool  True=masked (unused here directly
                                     -- padding queries just get decoded
                                     and their loss/metric contribution is
                                     masked out downstream by the trainer)
            training:         unused, kept for API parity with Atomizer/FRACTAL.
            query_chunk_size: if set, decode queries in chunks (avoids OOM
                              on large full-scene query sets). Only used
                              when training=False.

        Returns:
            logits: [B, M, num_classes]
        """
        lidar_tokens = batch["lidar_tokens"]  # [B, N_lidar, 198]
        lidar_mask   = batch["lidar_mask"]    # [B, N_lidar]
        queries      = batch["queries"]       # [B, M, 195]

        # -- 1. Split raw LIDAR token into position / echo / intensity ---
        lidar_pos   = lidar_tokens[..., :LIDAR_FOURIER_DIM]
        echo_ab     = lidar_tokens[..., LIDAR_FOURIER_DIM:LIDAR_FOURIER_DIM + ECHO_SCALARS_DIM]
        intensity   = lidar_tokens[..., LIDAR_FOURIER_DIM + ECHO_SCALARS_DIM:]

        echo_out      = self.echo_mlp(echo_ab)            # [B, N_lidar, 49]
        intensity_out = self.intensity_mlp(intensity)     # [B, N_lidar, 49]

        lidar_full = torch.cat(
            [lidar_pos, echo_out, intensity_out], dim=-1
        )  # [B, N_lidar, INPUT_DIM] -- no VHR, no padding, no concatenation
           # with another modality; this IS the full token stream.

        # -- 2. Build encoder mask (flip padding->valid convention) ------
        encoder_mask = ~lidar_mask  # [B, N_lidar]  True=valid

        # -- 3. Encode -----------------------------------------------------
        latents = self.perceiver.encode(lidar_full, mask=encoder_mask)

        # -- 4. Project queries ---------------------------------------------
        queries_proj = self.query_proj(queries)  # [B, M, latent_dim]

        # -- 5. Decode (optionally chunked) ---------------------------------
        if query_chunk_size is None or training:
            logits = self.perceiver.decode(latents, queries_proj)
        else:
            M = queries_proj.shape[1]
            chunks = []
            for start in range(0, M, query_chunk_size):
                end = min(start + query_chunk_size, M)
                chunk = self.perceiver.decode(latents, queries_proj[:, start:end, :])
                chunks.append(chunk)
            logits = torch.cat(chunks, dim=1)

        return logits
