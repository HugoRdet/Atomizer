"""
Atomiser_Senflood_Skip + Temporal Transformer — Multi-Temporal Wrapper
======================================================================

Dataset contract (NEW — explicit time dimension):
    groups[res]["tokens"] : [B, T, N, 8]     same N for every timestamp
    groups[res]["mask"]   : [B, T, N]        False = VALID, True = PAD
    groups[res]["shape"]  : per-timestamp spatial shape (unchanged)
    queries / queries_mask: unchanged (time-invariant targets)
    batch["time_indices"] : [B, T] (optional) real acquisition time per
                             timestep (e.g. day-of-year), used as the
                             TemporalTransformer's positional signal instead
                             of ordinal position. Falls back to arange(T)
                             when absent, so datasets that don't supply it
                             keep working unchanged.

    SKIP cascade (if used): query_token_idx must index the TIME-FLATTENED
    pool, i.e. row = t * N + n, because pool_tokens is passed to the decoder
    as tokens.reshape(B, T*N, 8).

Pipeline:
    For each timestamp t:
        encode(groups[:, t]) → latents_t {res: [B, L_res, D]}
    Stack:            {res: [B, L_res, T, D]}
    TemporalTransformer (self-attn over T + CLS readout) → {res: [B, L_res, D]}
    reconstruct() with aggregated latents (skip cascade untouched).

    T=1 → aggregator skipped entirely.
    Global latents are dropped after encoding (decoder never uses them).
    MAE / adaptive / quadtree decode: out of scope for this wrapper.

Design invariants:
    * sample_config() is drawn ONCE per forward and shared by all timestamps,
      so every timestamp sees the same (tpl, cross_k) → same grid_configs →
      same L_spatial → identical, stackable latent grids and coords.
    * encode() is called directly (never forward(task="encoder")) to keep
      full control over grid_configs and cross_k.
    * Empty timestamps (all tokens padded for a sample) are excluded from
      temporal attention via src_key_padding_mask (softmax sees -inf).

Config:
    Atomiser:
        temporal_module: "transformer"
        transformer:
            n_head: 8
            n_layers: 2
            dim_feedforward: 1024
            dropout: 0.1
            max_T: 1000
            positional_encoding: true
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional

from .Atomiser_senflood_skip import Atomiser_Senflood_Skip
from training.utils.datasets.token_grouping import compute_grid_config


# =============================================================================
# Temporal Transformer — self-attention over T timestamps + CLS readout
# =============================================================================

from .RPE import SelfAttentionRoPE, PreNormRoPE


# =============================================================================
# Temporal Transformer block — reuses the codebase's SelfAttentionRoPE
# (LocalRoPE2D under the hood), rather than a bespoke RoPE implementation.
# Time is inherently 1D, but LocalRoPE2D is built for 2D; the clean way to
# reuse it unmodified is pos_x = time (continuous, e.g. day-of-year),
# pos_y = 0 for every token, so the y-half of the rotation is always the
# identity (cos=1, sin=0) and only the x-half carries the temporal signal.
# This also gets the learnable physical-compression scale (pos / (S+|pos|))
# for free, which is a better fit for irregular day gaps than raw RoPE
# frequencies on an unbounded day count.
# =============================================================================

class TemporalBlock(nn.Module):
    def __init__(self, dim, n_head, dim_head, dim_feedforward, dropout,
                 rope_compression_scale, rope_learnable_scale):
        super().__init__()
        self.attn = PreNormRoPE(dim, SelfAttentionRoPE(
            dim=dim, heads=n_head, dim_head=dim_head, dropout=dropout,
            use_rope=True,
            rope_compression_scale=rope_compression_scale,
            rope_learnable_scale=rope_learnable_scale,
        ))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, pos_x, pos_y, attn_mask=None):
        x = x + self.attn(x, pos_x=pos_x, pos_y=pos_y, attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TemporalTransformer(nn.Module):
    """
    Aggregates a temporal sequence of latent states into a single state,
    using RoPE (LocalRoPE2D, time as pos_x / pos_y=0) instead of additive
    absolute positional encoding.

    Interface (drop-in replacement for the previous sinusoidal-PE version —
    same call signature, but `time_indices` should now be CONTINUOUS real
    time values, e.g. day-of-year, not categorical IDs; see
    batch["time_positions"] vs batch["time_indices"] in the wrapper):
        forward(x [B*L, T, D], time_indices [B*L, T] or None,
                pad_mask [B*L, T] or None)  →  [B*L, D]

    A learned CLS token is prepended to the sequence at position 0 (angle 0
    under the compression, since compress(0) = 0). RoPE rotates Q/K at every
    layer based on each token's continuous time position, so relative timing
    is what the model can attend to — there is no fixed per-position vector
    to memorize, which is the intended regularizing effect versus the
    previous additive sinusoidal PE.
    """

    def __init__(
        self,
        in_channels: int,
        n_head: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        T: int = 1000,
        positional_encoding: bool = True,
        rope_compression_scale: float = 50.0,
        rope_learnable_scale: bool = True,
    ):
        super().__init__()
        assert in_channels % n_head == 0, "in_channels must be divisible by n_head"
        dim_head = in_channels // n_head
        assert dim_head % 4 == 0, (
            f"in_channels/n_head must be divisible by 4 for LocalRoPE2D "
            f"(got dim_head={dim_head}); adjust n_head."
        )

        self.in_channels         = in_channels
        self.positional_encoding = positional_encoding
        self.T_max               = T

        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        nn.init.trunc_normal_(self.cls_token, std=0.02, a=-2., b=2.)

        self.layers = nn.ModuleList([
            TemporalBlock(
                in_channels, n_head, dim_head, dim_feedforward, dropout,
                rope_compression_scale, rope_learnable_scale,
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(in_channels)

    def forward(
        self,
        x: torch.Tensor,                              # [B*L, T, D]
        time_indices: Optional[torch.Tensor] = None,    # [B*L, T] continuous, or None
        pad_mask: Optional[torch.Tensor] = None,        # [B*L, T] True = PAD
    ) -> torch.Tensor:                                 # [B*L, D]
        BL, T, D = x.shape

        if time_indices is None:
            # Fallback: ordinal positions (still fine for RoPE — it only
            # needs SOME continuous scalar per position, just less
            # informative than real elapsed time when acquisitions are
            # irregularly spaced).
            time_indices = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0).expand(BL, -1)
        time_indices = time_indices.float()

        cls = self.cls_token.expand(BL, 1, -1)
        x   = torch.cat([cls, x], dim=1)                # [B*L, 1+T, D]

        # CLS sits at position 0 (compress(0) = 0 -> identity rotation).
        cls_pos = torch.zeros(BL, 1, device=x.device, dtype=time_indices.dtype)
        pos_x = torch.cat([cls_pos, time_indices], dim=1)   # [B*L, 1+T]
        pos_y = torch.zeros_like(pos_x)                     # 1D time -> y-half unused

        if not self.positional_encoding:
            pos_x = torch.zeros_like(pos_x)  # collapses RoPE to identity everywhere

        attn_mask = None
        if pad_mask is not None:
            cls_pad = torch.zeros(BL, 1, dtype=torch.bool, device=x.device)
            key_pad = torch.cat([cls_pad, pad_mask], dim=1)     # [B*L, 1+T], True=PAD
            attn_mask = (~key_pad).unsqueeze(1).unsqueeze(1)    # [B*L, 1, 1, 1+T], True=keep

        for layer in self.layers:
            x = layer(x, pos_x, pos_y, attn_mask=attn_mask)

        return self.out_norm(x[:, 0])                    # CLS readout → [B*L, D]


# =============================================================================
# Wrapper
# =============================================================================

class AtomiserTemporal(pl.LightningModule):
    """
    Multi-temporal wrapper around Atomiser_Senflood_Skip.

    Flow (see module docstring):
        per-timestamp encode → stack → TemporalTransformer → reconstruct.
    """

    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=["lookup_table"])
        self.config = config

        self.atomiser = Atomiser_Senflood_Skip(config=config, lookup_table=lookup_table)

        t_cfg = config["Atomiser"].get("transformer", {})
        self.temporal = TemporalTransformer(
            in_channels         = self.atomiser.latent_dim,
            n_head              = t_cfg.get("n_head", 4),
            n_layers            = t_cfg.get("n_layers", 1),
            dim_feedforward     = t_cfg.get("dim_feedforward", 256),
            dropout             = t_cfg.get("dropout", 0.1),
            T                   = t_cfg.get("max_T", 1000),
            positional_encoding = t_cfg.get("positional_encoding", True),
        )

        n_temporal = sum(p.numel() for p in self.temporal.parameters() if p.requires_grad)
        n_atomiser = sum(p.numel() for p in self.atomiser.parameters() if p.requires_grad)
        print(f"[AtomiserTemporal] Temporal module: TemporalTransformer "
              f"(n_head={t_cfg.get('n_head', 8)}, n_layers={t_cfg.get('n_layers', 2)}, "
              f"PE={self.temporal.positional_encoding})")
        print(f"[AtomiserTemporal] Trainable: atomiser={n_atomiser/1e6:.1f}M, "
              f"temporal={n_temporal/1e6:.2f}M, "
              f"total={(n_atomiser + n_temporal)/1e6:.1f}M")

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, batch, training=True, **kwargs):
        groups       = batch["groups"]      # groups[res]["tokens"]: [B, T, N, 8]
        queries      = batch["queries"]
        queries_mask = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)

        # Continuous real acquisition time per timestep (day-of-year), used
        # by RoPE. NOT the same as the categorical time_indices used inside
        # tokens for phi_t — see pastis_temporal_dataset.py docstring.
        # Falls back to batch["time_indices"] (still better than nothing)
        # then to None (ordinal fallback inside TemporalTransformer) for
        # datasets that don't supply time_positions.
        time_positions = batch.get("time_positions", batch.get("time_indices", None))

        resolutions = sorted(groups.keys())
        first = groups[resolutions[0]]["tokens"]
        B, T = first.shape[0], first.shape[1]

        # ── SKIP cascade inputs: time-flattened pool ─────────────────────
        query_token_idx   = batch.get("query_token_idx", None)    # [B, M, A] rows into [B, T*N]
        query_token_valid = batch.get("query_token_valid", None)
        skip_pool_tokens = None
        skip_pool_mask   = None
        if self.atomiser.use_decoder_skip and query_token_idx is not None:
            skip_res = self.config["Atomiser"].get("skip_resolution", 10.0)
            if skip_res not in groups:
                skip_res = min(groups.keys())
            tk = groups[skip_res]["tokens"]                       # [B, T, N, 8]
            mk = groups[skip_res]["mask"]                         # [B, T, N]
            skip_pool_tokens = tk.reshape(B, -1, tk.shape[-1])    # [B, T*N, 8]
            skip_pool_mask   = mk.reshape(B, -1)                  # [B, T*N]

        # ── ONE sampling config shared by all timestamps ─────────────────
        # (a per-timestamp draw would desynchronize grid_configs → L_spatial)
        tpl, cross_k = self.atomiser.sample_config(training)
        geo_k_budget = cross_k * 2

        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                tokens_per_latent=tpl,
                total_tokens=groups[res]["tokens"].shape[2],      # N per timestamp
                sigma_factor=self.atomiser.sigma_factor,
                max_k=geo_k_budget,
            )
            for res in resolutions
        }

        # ── Encode each timestamp with the shared grid_configs ───────────
        latents_per_t = []
        coords_per_res = None

        for t in range(T):
            groups_t = {
                res: {
                    "tokens": groups[res]["tokens"][:, t],        # [B, N, 8]
                    "mask":   groups[res]["mask"][:, t],          # [B, N]
                    "shape":  groups[res]["shape"],
                }
                for res in resolutions
            }
            enc = self.atomiser.encode(
                groups=groups_t,
                grid_configs=grid_configs,
                training=training,
                cross_k=cross_k,
            )
            latents_per_t.append(enc.latents_per_res)
            if coords_per_res is None:
                coords_per_res = enc.coords_per_res   # identical for all t
            # global latents intentionally dropped (decoder never uses them)

        # ── Temporal aggregation per resolution ──────────────────────────
        if T == 1:
            aggregated_latents = latents_per_t[0]
        else:
            # A timestamp is padded for a sample when ALL its tokens are
            # padded across every resolution group.
            empty_t = torch.stack(
                [groups[res]["mask"].all(dim=-1) for res in resolutions],
                dim=0).all(dim=0)                                  # [B, T]

            aggregated_latents = {}
            for res in sorted(latents_per_t[0].keys(), key=str):
                stacked = torch.stack(
                    [latents_per_t[t][res] for t in range(T)], dim=2)  # [B, L, T, D]
                Bc, L, _, D = stacked.shape

                x = stacked.reshape(Bc * L, T, D)
                pad = empty_t.unsqueeze(1).expand(Bc, L, T).reshape(Bc * L, T)

                ti = None
                if time_positions is not None:
                    # [B, T] -> broadcast over L -> [B*L, T]
                    ti = time_positions.unsqueeze(1).expand(Bc, L, T).reshape(Bc * L, T)

                agg = self.temporal(x, time_indices=ti, pad_mask=pad)  # [B*L, D]
                aggregated_latents[res] = agg.reshape(Bc, L, D)

        # ── Decode (chunked), skip cascade passed through untouched ──────
        chunk_size = 10_000
        M = queries.shape[1]

        def _decode(q, qm, qti, qtv):
            return self.atomiser.reconstruct(
                aggregated_latents, coords_per_res,
                q, qm,
                target_resolution=target_resolution,
                training=training,
                query_token_idx=qti,
                query_token_valid=qtv,
                pool_tokens=skip_pool_tokens,
                pool_mask=skip_pool_mask,
            )

        if M > chunk_size:
            preds = []
            for i in range(0, M, chunk_size):
                preds.append(_decode(
                    queries[:, i:i + chunk_size],
                    queries_mask[:, i:i + chunk_size],
                    query_token_idx[:, i:i + chunk_size] if query_token_idx is not None else None,
                    query_token_valid[:, i:i + chunk_size] if query_token_valid is not None else None,
                ))
            output = torch.cat(preds, dim=1)
        else:
            output = _decode(queries, queries_mask, query_token_idx, query_token_valid)

        return {"predictions": output}

    # =========================================================================
    # Freeze / unfreeze
    # =========================================================================

    def freeze_encoder(self):    self.atomiser.freeze_encoder()
    def unfreeze_encoder(self):  self.atomiser.unfreeze_encoder()
    def freeze_decoder(self):    self.atomiser.freeze_decoder()
    def unfreeze_decoder(self):  self.atomiser.unfreeze_decoder()

    def freeze_temporal(self):
        for p in self.temporal.parameters():
            p.requires_grad = False

    def unfreeze_temporal(self):
        for p in self.temporal.parameters():
            p.requires_grad = True

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
