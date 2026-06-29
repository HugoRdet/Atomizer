"""
PerceiverFractal
=================

Wraps PerceiverIO for FRACTAL LIDAR + VHR semantic segmentation.

Consumes the output of FractalPerceiverDataset:
    vhr_tokens   [B, N_vhr,   VHR_DIM=262]    Fourier-encoded VHR tokens
    vhr_mask     [B, N_vhr]   bool
    lidar_tokens [B, N_lidar, LIDAR_RAW_DIM=197]  Fourier(X,Y,Z) + echo (a,b)
    lidar_mask   [B, N_lidar] bool
    queries      [B, M,       QUERY_DIM=195]   Fourier(X,Y,Z) query tokens

Processing pipeline
-------------------
1. Echo MLP: lidar_tokens[..., 195:197] -> [B, N_lidar, 49]
2. LIDAR pad: learned [18] broadcast -> [B, N_lidar, 18]
3. LIDAR full: cat([Fourier(X,Y,Z), echo_out, pad]) -> [B, N_lidar, 262]
4. Tokens: cat([vhr_tokens, lidar_full], dim=1) -> [B, N_vhr+N_lidar, 262]
5. Mask:   cat([vhr_mask,   lidar_mask], dim=1) -> [B, N_vhr+N_lidar]
           PerceiverIO encoder expects True = VALID, so we flip the mask
           convention here (dataset: True=masked -> encoder: True=valid).
6. Encode: PerceiverEncoder(tokens, mask) -> latents [B, L, latent_dim]
7. Query proj: queries [B, M, 195] -> [B, M, latent_dim]
8. Decode: PerceiverDecoder(latents, query_proj) -> [B, M, num_classes]

Mask convention
---------------
FractalPerceiverDataset uses True=masked (padding), matching PyTorch's
MultiheadAttention key_padding_mask convention. PerceiverIO's encoder
forward() uses mask: True=valid (the Perceiver paper convention). We
flip once on entry so everything downstream is consistent.

Args (constructor)
------------------
    num_classes:        Number of output classes. Default 7.
    num_latents:        Number of latent vectors. Default 256.
    latent_dim:         Latent dimension. Default 256.
    depth:              Number of encoder cross+self-attn blocks. Default 6.
    cross_heads:        Heads for cross-attention. Default 1.
    latent_heads:       Heads for self-attention. Default 8.
    cross_dim_head:     Dim per head (cross). Default 64.
    latent_dim_head:    Dim per head (self). Default 64.
    self_per_cross_attn: Self-attn blocks per cross-attn. Default 1.
    weight_tie_layers:  Share weights across encoder blocks > 0. Default True.
    attn_dropout:       Attention dropout. Default 0.0.
    ff_dropout:         Feedforward dropout. Default 0.0.
    echo_hidden_dim:    Hidden dim of echo MLP. Default 64.
"""

import torch
import torch.nn as nn

from .perceiver_io import PerceiverIO

# ── Dimension constants (must match FractalPerceiverDataset) ──────────────
VHR_DIM          = 262    # 4*33 + 2*65
LIDAR_RAW_DIM    = 197    # 3*65 + 2 (echo scalars)
LIDAR_FOURIER_DIM = 195   # 3*65
ECHO_SCALARS_DIM  = 2     # (a, b)
ECHO_MLP_OUT_DIM  = 49    # must match time_encoder.out_dim in Atomizer
INPUT_DIM        = VHR_DIM                                     # 262
LIDAR_PAD_DIM    = INPUT_DIM - LIDAR_FOURIER_DIM - ECHO_MLP_OUT_DIM  # 18
QUERY_DIM        = 195    # 3*65


def _build_echo_mlp(hidden_dim: int, out_dim: int) -> nn.Sequential:
    """
    Small MLP for echo encoding: (a, b) -> out_dim features.
    Matches FractalTokenProcessor's echo_encoder architecture:
        2 -> hidden_dim -> out_dim, GELU, LayerNorm, zero-init last layer.
    """
    mlp = nn.Sequential(
        nn.Linear(ECHO_SCALARS_DIM, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )
    # Zero-init last linear so echo features start at zero at init,
    # matching Atomizer's FractalTokenProcessor behavior.
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class PerceiverFractal(nn.Module):
    """
    PerceiverIO for FRACTAL LIDAR + VHR semantic segmentation.

    See module docstring for the full processing pipeline.
    """

    def __init__(
        self,
        num_classes: int = 7,
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
    ):
        super().__init__()

        self.num_classes = num_classes

        # ── Echo MLP: (a, b) -> ECHO_MLP_OUT_DIM ────────────────────
        self.echo_mlp = _build_echo_mlp(
            hidden_dim=echo_hidden_dim,
            out_dim=ECHO_MLP_OUT_DIM,
        )

        # ── Learned LIDAR padding: broadcast to [B, N_lidar, 18] ────
        # Initialized to zeros so LIDAR tokens start identical to a
        # model that ignores the padding slot.
        self.lidar_pad = nn.Parameter(
            torch.zeros(LIDAR_PAD_DIM)
        )

        # ── Query projection: QUERY_DIM -> latent_dim ────────────────
        # PerceiverDecoder expects query_dim == latent_dim so its
        # cross-attention Q projection is square. We project here.
        self.query_proj = nn.Linear(QUERY_DIM, latent_dim)

        # ── Core PerceiverIO ─────────────────────────────────────────
        self.perceiver = PerceiverIO(
            input_dim=INPUT_DIM,
            query_dim=latent_dim,       # after query_proj
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
        print(f"[PerceiverFractal] num_classes={num_classes}, "
              f"latents={num_latents}x{latent_dim}, depth={depth}")
        print(f"[PerceiverFractal] input_dim={INPUT_DIM} "
              f"(VHR={VHR_DIM}, LIDAR raw={LIDAR_RAW_DIM} "
              f"-> {INPUT_DIM} after echo+pad={LIDAR_PAD_DIM})")
        print(f"[PerceiverFractal] query_dim={QUERY_DIM} "
              f"-> projected to latent_dim={latent_dim}")
        print(f"[PerceiverFractal] Parameters: {n_params:,}")

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        batch: dict,
        training: bool = True,
        query_chunk_size: int = None,
    ) -> torch.Tensor:
        """
        Args:
            batch: dict from FractalPerceiverDataset DataLoader, with keys:
                vhr_tokens   [B, N_vhr,   262]
                vhr_mask     [B, N_vhr]   bool  True=masked
                lidar_tokens [B, N_lidar, 197]
                lidar_mask   [B, N_lidar] bool  True=masked
                queries      [B, M,       195]
                queries_mask [B, M]       bool  True=masked
            training:         unused, kept for API parity with Atomizer.
            query_chunk_size: if set, decode queries in chunks of this size
                              to avoid OOM on large full-scene query sets
                              (e.g. 500k+ points). Only used when
                              training=False. Default None = no chunking.

        Returns:
            logits: [B, M, num_classes]
        """
        vhr_tokens   = batch["vhr_tokens"]    # [B, N_vhr,   262]
        vhr_mask     = batch["vhr_mask"]      # [B, N_vhr]
        lidar_tokens = batch["lidar_tokens"]  # [B, N_lidar, 197]
        lidar_mask   = batch["lidar_mask"]    # [B, N_lidar]
        queries      = batch["queries"]       # [B, M, 195]

        B = vhr_tokens.shape[0]

        # ── 1. Echo MLP + LIDAR padding ──────────────────────────────
        lidar_fourier = lidar_tokens[..., :LIDAR_FOURIER_DIM]   # [B, N_lidar, 195]
        echo_ab       = lidar_tokens[..., LIDAR_FOURIER_DIM:]   # [B, N_lidar, 2]

        echo_out = self.echo_mlp(echo_ab)   # [B, N_lidar, 49]

        # Broadcast learned padding to match batch and sequence dims
        pad = self.lidar_pad.view(1, 1, LIDAR_PAD_DIM).expand(
            B, lidar_fourier.shape[1], -1
        )   # [B, N_lidar, 18]

        lidar_full = torch.cat(
            [lidar_fourier, echo_out, pad], dim=-1
        )   # [B, N_lidar, 262]

        # ── 2. Concatenate VHR + LIDAR tokens ────────────────────────
        tokens = torch.cat([vhr_tokens, lidar_full], dim=1)
        # [B, N_vhr + N_lidar, 262]

        # ── 3. Build encoder mask ─────────────────────────────────────
        # Dataset convention: True = masked (padding).
        # PerceiverEncoder convention: True = valid token.
        # Flip here once so downstream is consistent.
        combined_mask = torch.cat([vhr_mask, lidar_mask], dim=1)
        # [B, N_vhr + N_lidar]   True=masked
        encoder_mask = ~combined_mask
        # [B, N_vhr + N_lidar]   True=valid  ✓

        # ── 4. Encode ─────────────────────────────────────────────────
        latents = self.perceiver.encode(tokens, mask=encoder_mask)
        # [B, L, latent_dim]

        # ── 5. Project queries ────────────────────────────────────────
        queries_proj = self.query_proj(queries)   # [B, M, latent_dim]

        # ── 6. Decode (optionally chunked) ────────────────────────────
        # During full-scene evaluation, M can be 500k+ points which OOMs
        # in a single decoder cross-attention. We chunk along the query
        # dimension and concatenate the results.
        # During training, query_chunk_size is None so we decode in one
        # pass as before (no overhead).
        if query_chunk_size is None or training:
            logits = self.perceiver.decode(latents, queries_proj)
        else:
            M = queries_proj.shape[1]
            chunks = []
            for start in range(0, M, query_chunk_size):
                end   = min(start + query_chunk_size, M)
                chunk = self.perceiver.decode(
                    latents, queries_proj[:, start:end, :]
                )   # [B, chunk_size, num_classes]
                chunks.append(chunk)
            logits = torch.cat(chunks, dim=1)   # [B, M, num_classes]

        return logits
