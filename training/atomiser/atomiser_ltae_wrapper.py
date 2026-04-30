"""
Atomiser + Temporal Aggregator — End-to-End Multi-Temporal Wrapper
===================================================================

Supports two temporal aggregators (selected by config["Atomiser"]["temporal_module"]):
    - "ltae":        master-query attention (Garnot & Landrieu 2020)
    - "transformer": self-attention over T timestamps + CLS readout

Both are drop-in replacements for each other with identical [B*L, T, D] → [B*L, D]
interface. Placement is identical (after full per-timestamp encoder pass) to keep
ablations clean.

Pipeline:
    For each timestamp t:
        tokens_t → encoder (cross-attn + self-attn) → latents_t [B, L, D]
    Stack:           [B, L, T, D]
    Temporal agg:    [B, L, T, D] → [B, L, D]
    Decoder:         k-nearest → predictions

When T=1 (single-timestamp), the temporal module is SKIPPED entirely.

Config:
    Atomiser:
        use_ltae: true                     # enable temporal module
        temporal_module: "ltae"            # "ltae" or "transformer"
        ltae:                              # LTAE-specific
            n_head: 16
            d_k: 4
            d_model: 768
            dropout: 0.2
            max_T: 1000
            positional_encoding: true
        transformer:                       # TemporalTransformer-specific
            n_head: 8
            n_layers: 2
            dim_feedforward: 1024
            dropout: 0.1
            max_T: 1000
            positional_encoding: true
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional

from training.atomiser import Atomiser_Senflood


# =============================================================================
# LTAE — Lightweight Temporal Attention Encoder
# =============================================================================

class LTAE(nn.Module):
    """
    Lightweight Temporal Self-Attention (Garnot & Landrieu 2020).

    Processes [B*L, T, D] temporal sequences via learned master query.
    O(T) per spatial position, not O(T²).
    """

    def __init__(
        self,
        in_channels: int,
        n_head: int = 16,
        d_k: int = 4,
        d_model: Optional[int] = None,
        dropout: float = 0.2,
        T: int = 1000,
        positional_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels         = in_channels
        self.n_head              = n_head
        self.d_k                 = d_k
        self.d_model             = d_model or in_channels
        self.positional_encoding = positional_encoding
        self.T_max               = T

        self.fc_in = nn.Linear(in_channels, self.d_model)

        self.master_query = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.master_query, std=0.02)

        self.fc_k = nn.Linear(self.d_model, n_head * d_k)

        if positional_encoding:
            self.register_buffer(
                "pe_table", self._build_pe_table(T, self.d_model))

        self.attn_dropout = nn.Dropout(dropout)
        self.fc_out       = nn.Linear(self.d_model, in_channels)

    @staticmethod
    def _build_pe_table(T: int, d: int) -> torch.Tensor:
        pe  = torch.zeros(T, d)
        pos = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                        * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        x: torch.Tensor,                   # [B*L, T, D]
        time_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                     # [B*L, D]
        BL, T, D = x.shape

        x = self.fc_in(x)

        if self.positional_encoding and time_indices is not None:
            time_indices = time_indices.clamp(0, self.T_max - 1).long()
            pe = self.pe_table[time_indices]
            x = x + pe

        k = self.fc_k(x).reshape(BL, T, self.n_head, self.d_k)
        k = k.permute(0, 2, 1, 3)                                # [BL, H, T, d_k]

        q = self.master_query.unsqueeze(0).unsqueeze(2)           # [1, H, 1, d_k]
        q = q.expand(BL, -1, -1, -1)                              # [BL, H, 1, d_k]

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_dropout(attn)

        d_head = self.d_model // self.n_head
        if self.d_model % self.n_head != 0:
            v = x.unsqueeze(2).expand(-1, -1, self.n_head, -1)
            v = v.permute(0, 2, 1, 3)
            out = torch.matmul(attn, v).squeeze(2)
            out = out.mean(dim=1)
        else:
            v = x.reshape(BL, T, self.n_head, d_head)
            v = v.permute(0, 2, 1, 3)
            out = torch.matmul(attn, v).squeeze(2)
            out = out.reshape(BL, self.d_model)

        return self.fc_out(out)


# =============================================================================
# Atomiser + Temporal Aggregator
# =============================================================================

class AtomiserLTAE(pl.LightningModule):
    """
    End-to-end Atomiser with temporal aggregation (LTAE or TemporalTransformer).

    Name kept as AtomiserLTAE for back-compat with trainer/checkpoints.
    Actual aggregator is selected by config["Atomiser"]["temporal_module"].

    Flow:
        1. Partition tokens by time_idx (column 7)
        2. For each timestamp t:
              run self.atomiser.encode() on timestamp-t tokens
              → latents_per_res_t, coords_per_res_t
        3. Stack latents across T → [B, L, T, D]
        4. Temporal aggregator → [B, L, D]    (SKIPPED if T=1)
        5. Pass merged latents_per_res to self.atomiser.reconstruct()
    """

    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=["lookup_table"])
        self.config = config

        self.atomiser = Atomiser_Senflood(config=config, lookup_table=lookup_table)

        # Select temporal module
        temporal_cfg = config["Atomiser"]
        module_type  = temporal_cfg.get("temporal_module", "ltae").lower()

        if module_type == "ltae":
            ltae_cfg = temporal_cfg.get("ltae", {})
            self.temporal = LTAE(
                in_channels         = self.atomiser.latent_dim,
                n_head              = ltae_cfg.get("n_head", 16),
                d_k                 = ltae_cfg.get("d_k", 4),
                d_model             = ltae_cfg.get("d_model", self.atomiser.latent_dim),
                dropout             = ltae_cfg.get("dropout", 0.2),
                T                   = ltae_cfg.get("max_T", 1000),
                positional_encoding = ltae_cfg.get("positional_encoding", True),
            )
            print(f"[AtomiserLTAE] Temporal module: LTAE "
                  f"(n_head={self.temporal.n_head}, d_k={self.temporal.d_k}, "
                  f"d_model={self.temporal.d_model}, "
                  f"PE={self.temporal.positional_encoding})")

        else:
            raise ValueError(
                f"Unknown temporal_module='{module_type}'. "
                f"Valid: 'ltae', 'transformer'"
            )

        n_temporal = sum(p.numel() for p in self.temporal.parameters()
                         if p.requires_grad)
        n_atomiser = sum(p.numel() for p in self.atomiser.parameters()
                         if p.requires_grad)
        print(f"[AtomiserLTAE] Trainable: atomiser={n_atomiser/1e6:.1f}M, "
              f"temporal={n_temporal/1e6:.2f}M, "
              f"total={(n_atomiser + n_temporal)/1e6:.1f}M")

    # =========================================================================
    # Per-timestamp splitting (by time_idx)
    # =========================================================================

    @staticmethod
    def _split_tokens_by_time_idx(
        tokens: torch.Tensor,   # [B, N, 8]
        mask:   torch.Tensor,   # [B, N]  False = VALID, True = PAD (project convention)
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Partition tokens by time_idx (column 7).

        Project convention:
            mask == False  → token is VALID
            mask == True   → token is PADDED (ignored by attention)
        """
        time_col = tokens[..., 7].long()

        valid = ~mask
        valid_times = time_col[valid]
        if valid_times.numel() == 0:
            return [mask], torch.tensor([0], device=tokens.device)

        unique_times = torch.unique(valid_times).sort().values

        masks_per_t = []
        for t in unique_times:
            keep   = (time_col == t) & valid
            t_mask = ~keep
            masks_per_t.append(t_mask)

        return masks_per_t, unique_times

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, batch, training=True, return_for_error=False, **kwargs):
        groups       = batch["groups"]
        queries      = batch["queries"]
        queries_mask = batch["queries_mask"]

        # ── 1. Partition tokens by time_idx (per resolution) ────────────
        masks_per_t_by_res = {}
        unique_times = None

        for res in sorted(groups.keys()):
            m_per_t, times = self._split_tokens_by_time_idx(
                groups[res]["tokens"], groups[res]["mask"])
            masks_per_t_by_res[res] = m_per_t
            if unique_times is None:
                unique_times = times

        T = len(unique_times)

        # One-time diagnostic
        if not getattr(AtomiserLTAE, "_debug_printed", False):
            AtomiserLTAE._debug_printed = True
            print(f"[AtomiserLTAE debug] T={T} unique timestamps")
            for res, m_list in masks_per_t_by_res.items():
                counts = [(~m).sum().item() for m in m_list]
                if counts:
                    print(f"[AtomiserLTAE debug] res={res}: tokens/t "
                          f"min={min(counts)} max={max(counts)} "
                          f"mean={sum(counts)/len(counts):.0f}")
            print(f"[AtomiserLTAE debug] unique_times={unique_times.tolist()[:10]}...")

        # ── 2. Encode each timestamp separately ─────────────────────────
        latents_per_t = []
        coords_per_t  = []

        for t_idx in range(T):
            groups_t = {}
            for res in sorted(groups.keys()):
                groups_t[res] = {
                    "tokens": groups[res]["tokens"],
                    "mask":   masks_per_t_by_res[res][t_idx],
                    "shape":  groups[res]["shape"],
                }
            batch_t = {
                "groups":       groups_t,
                "queries":      queries,
                "queries_mask": queries_mask,
            }
            if "target_resolution" in batch:
                batch_t["target_resolution"] = batch["target_resolution"]

            encoded = self.atomiser(
                batch_t, training=training, task="encoder")
            latents_per_t.append(encoded["latents_per_res"])
            coords_per_t.append(encoded["coords_per_res"])

        # ── 3. Aggregate per-resolution ─────────────────────────────────
        aggregated_latents = {}
        aggregated_coords  = {}

        for res in sorted(latents_per_t[0].keys(), key=str):
            # First (and only) timestamp's coords — time-invariant
            aggregated_coords[res] = coords_per_t[0][res]

            if T == 1:
                # Skip temporal aggregator entirely for T=1
                aggregated_latents[res] = latents_per_t[0][res]
                continue

            # Stack: T copies of [B, L, D] → [B, L, T, D]
            stacked = torch.stack(
                [latents_per_t[t][res] for t in range(T)], dim=2)
            B, L, _, D = stacked.shape

            x = stacked.reshape(B * L, T, D)
            time_idx = unique_times.to(x.device).unsqueeze(0).expand(B * L, -1)

            aggregated = self.temporal(x, time_indices=time_idx)        # [B*L, D]
            aggregated_latents[res] = aggregated.reshape(B, L, D)

        # ── 4. Decode with aggregated latents ───────────────────────────
        predicted_errors = None
        if self.atomiser.use_error_predictor:
            all_lat = torch.cat(
                [aggregated_latents[r]
                 for r in sorted(aggregated_latents.keys(), key=str)],
                dim=1)
            predicted_errors = self.atomiser.error_predictor(
                all_lat.detach()).squeeze(-1)

        chunk_size = 10_000
        N = queries.shape[1]
        need_topk = return_for_error and self.atomiser.use_error_predictor

        target_resolution = batch.get("target_resolution", None)

        if N > chunk_size:
            preds_list      = []
            topk_idx_list   = []
            topk_dists_list = []
            for i in range(0, N, chunk_size):
                res = self.atomiser.reconstruct(
                    aggregated_latents, aggregated_coords,
                    queries[:, i:i + chunk_size],
                    queries_mask[:, i:i + chunk_size],
                    target_resolution=target_resolution,
                    training=training,
                    return_topk=need_topk,
                )
                if need_topk:
                    preds_list.append(res[0])
                    topk_idx_list.append(res[1])
                    topk_dists_list.append(res[2])
                else:
                    preds_list.append(res)
            output = torch.cat(preds_list, dim=1)
            if need_topk:
                topk_indices  = torch.cat(topk_idx_list,   dim=1)
                topk_dists_sq = torch.cat(topk_dists_list, dim=1)
        else:
            res = self.atomiser.reconstruct(
                aggregated_latents, aggregated_coords,
                queries, queries_mask,
                target_resolution=target_resolution,
                training=training,
                return_topk=need_topk,
            )
            if need_topk:
                output, topk_indices, topk_dists_sq = res
            else:
                output = res

        # ── 5. Return ───────────────────────────────────────────────────
        if return_for_error and need_topk:
            all_coords = torch.cat(
                [aggregated_coords[r]
                 for r in sorted(aggregated_coords.keys(), key=str)],
                dim=1)
            all_lat_post = torch.cat(
                [aggregated_latents[r]
                 for r in sorted(aggregated_latents.keys(), key=str)],
                dim=1)
            return {
                "predictions":      output,
                "predicted_errors": predicted_errors,
                "topk_indices":     topk_indices,
                "topk_dists_sq":    topk_dists_sq,
                "num_latents":      all_lat_post.shape[1],
                "latent_coords":    all_coords,
            }

        return {"predictions": output, "predicted_errors": predicted_errors}

    # =========================================================================
    # Freeze / unfreeze
    # =========================================================================

    def freeze_encoder(self):
        self.atomiser.freeze_encoder()

    def unfreeze_encoder(self):
        self.atomiser.unfreeze_encoder()

    def freeze_decoder(self):
        self.atomiser.freeze_decoder()

    def unfreeze_decoder(self):
        self.atomiser.unfreeze_decoder()

    def freeze_temporal(self):
        for p in self.temporal.parameters():
            p.requires_grad = False

    def unfreeze_temporal(self):
        for p in self.temporal.parameters():
            p.requires_grad = True

    # Back-compat aliases
    def freeze_ltae(self):   self.freeze_temporal()
    def unfreeze_ltae(self): self.unfreeze_temporal()

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True