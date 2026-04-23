"""
Atomiser + LTAE — End-to-End Multi-Temporal Wrapper
=====================================================

LTAE with Fourier-DOY-concat time encoding.

Flow:
    For each timestamp t:
        full atomiser encode (cross-attn + all self-attn passes) → latents_t
    Stack [B, L, T, D] → LTAE → [B, L, D]
    Decoder: k-nearest → predictions

Time encoding in LTAE:
    Fourier features of actual DOY (day-of-year), concatenated to content
    features. DOYs resolved from time_indices via lookup_table at forward
    time — no cached buffer, so encoding is fully deterministic given DOY
    (checkpoint reload-safe).

When T=1 the temporal module is skipped entirely.

Config:
    Atomiser:
        use_ltae: true
        temporal_module: "ltae"            # only "ltae" supported now
        ltae:
            n_head: 16
            d_k: 4
            d_model: 768
            dropout: 0.2
            num_freq_bands: 24             # K, so time_dim = 2K = 48
            cycle_period: 365.0
            positional_encoding: true
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from typing import Dict, List, Tuple, Optional

from training.atomiser import Atomiser_Senflood


# =============================================================================
# Helpers: DOY resolution and Fourier features
# =============================================================================

def resolve_dois(
    time_indices: torch.Tensor,
    lookup_table,
) -> torch.Tensor:
    """
    Resolve time_idx values to DOY floats via lookup_table.table_time.
    Called at forward time — does not cache. Fully deterministic in DOY:
    whatever idx DOY=135 gets assigned, resolving it back produces 135.

    Returns a [T] float tensor on the same device as `time_indices`.
    """
    idx_to_doy = {}
    for time_key, idx in lookup_table.table_time.items():
        if isinstance(time_key, (int, float)):
            idx_to_doy[int(idx)] = float(time_key)
        elif isinstance(time_key, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(time_key)
                idx_to_doy[int(idx)] = float(dt.timetuple().tm_yday)
            except Exception:
                idx_to_doy[int(idx)] = 0.0
        elif isinstance(time_key, tuple) and len(time_key) >= 2:
            idx_to_doy[int(idx)] = float(time_key[1])
        else:
            idx_to_doy[int(idx)] = 0.0

    dois = torch.tensor(
        [idx_to_doy.get(int(t.item()), 0.0) for t in time_indices],
        dtype=torch.float32, device=time_indices.device,
    )
    return dois


def fourier_doy_features(
    dois: torch.Tensor,             # [..., T] float DOY values
    angular_freqs: torch.Tensor,    # [K] = 2π·k/cycle_period for k=1..K
) -> torch.Tensor:                  # [..., T, 2K]
    """
    Pure function: DOY → [sin(2πk·DOY/P), cos(2πk·DOY/P)] for k=1..K.
    No learnable parameters, no state.
    """
    angles  = dois.unsqueeze(-1) * angular_freqs.view(*([1] * dois.dim()), -1)
    sin_enc = torch.sin(angles)
    cos_enc = torch.cos(angles)
    features = torch.stack([sin_enc, cos_enc], dim=-1).reshape(
        *dois.shape, -1)            # [..., T, 2K]
    return features


# =============================================================================
# LTAE — Lightweight Temporal Attention Encoder with Fourier-DOY concat
# =============================================================================

class LTAE(nn.Module):
    """
    LTAE (Garnot & Landrieu 2020) with Fourier-DOY concat time encoding.

    Master-query attention over T timestamps. Time info enters as
    Fourier(DOY) features concatenated to content, then `fc_in` projects
    the concatenated vector back to d_model.

    Interface: [B*L, T, D] → [B*L, D]
    DOYs are passed as a [T] float tensor (broadcast over B*L internally).
    """

    def __init__(
        self,
        in_channels: int,
        n_head: int = 16,
        d_k: int = 4,
        d_model: Optional[int] = None,
        dropout: float = 0.2,
        num_freq_bands: int = 24,
        cycle_period: float = 365.0,
        positional_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels         = in_channels
        self.n_head              = n_head
        self.d_k                 = d_k
        self.d_model             = d_model or in_channels
        self.positional_encoding = positional_encoding

        self.num_freq_bands = num_freq_bands
        self.cycle_period   = float(cycle_period)
        self.time_dim       = 2 * num_freq_bands if positional_encoding else 0

        if positional_encoding:
            freqs = torch.arange(1, num_freq_bands + 1, dtype=torch.float32)
            angular_freqs = 2.0 * math.pi * freqs / self.cycle_period
            self.register_buffer("angular_freqs", angular_freqs)

        # fc_in absorbs both dim-matching (as in original LTAE) and the
        # concatenation of Fourier time features. When positional_encoding
        # is True, input dim = in_channels + 2K; otherwise = in_channels.
        self.fc_in = nn.Linear(in_channels + self.time_dim, self.d_model)

        self.master_query = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.master_query, std=0.02)

        self.fc_k = nn.Linear(self.d_model, n_head * d_k)

        self.attn_dropout = nn.Dropout(dropout)
        self.fc_out       = nn.Linear(self.d_model, in_channels)

    def forward(
        self,
        x: torch.Tensor,                            # [B*L, T, D]
        dois: Optional[torch.Tensor] = None,        # [T] float DOY values
    ) -> torch.Tensor:                              # [B*L, D]
        BL, T, D = x.shape

        if self.positional_encoding and dois is not None:
            # Broadcast DOYs to [B*L, T]
            if dois.dim() == 1:
                dois_full = dois.unsqueeze(0).expand(BL, T)
            else:
                dois_full = dois
            time_feats = fourier_doy_features(                      # [B*L, T, 2K]
                dois_full, self.angular_freqs.to(x.device))
            x_cat = torch.cat([x, time_feats.to(x.dtype)], dim=-1)  # [B*L, T, D+2K]
        else:
            x_cat = x

        # Project concatenated features to d_model
        x = self.fc_in(x_cat)                                       # [B*L, T, d_model]

        # Keys: [B*L, T, n_head * d_k] → [B*L, n_head, T, d_k]
        k = self.fc_k(x).reshape(BL, T, self.n_head, self.d_k).permute(0, 2, 1, 3)

        # Master query: [1, n_head, 1, d_k] → [B*L, n_head, 1, d_k]
        q = self.master_query.unsqueeze(0).unsqueeze(2).expand(BL, -1, -1, -1)

        # Attention scores: [B*L, n_head, 1, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_dropout(attn)

        # Values split channel-wise across heads (LTAE2d-style grouping)
        d_head = self.d_model // self.n_head
        if self.d_model % self.n_head != 0:
            v = x.unsqueeze(2).expand(-1, -1, self.n_head, -1)
            v = v.permute(0, 2, 1, 3)
            out = torch.matmul(attn, v).squeeze(2).mean(dim=1)
        else:
            v = x.reshape(BL, T, self.n_head, d_head).permute(0, 2, 1, 3)
            out = torch.matmul(attn, v).squeeze(2).reshape(BL, self.d_model)

        return self.fc_out(out)


# =============================================================================
# Atomiser + LTAE Wrapper
# =============================================================================

class AtomiserLTAE(pl.LightningModule):
    """
    End-to-end Atomiser with LTAE temporal aggregation (Fourier-DOY concat).

    Flow:
        1. Partition tokens by time_idx (column 7)
        2. For each timestamp t: full atomiser encode → latents_t [B, L, D]
        3. Stack across T → [B, L, T, D]
        4. Resolve DOYs via lookup table, pass to LTAE → [B, L, D]
        5. Decoder

    Class name kept as AtomiserLTAE for back-compat with trainer/checkpoints.
    """

    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=["lookup_table"])
        self.config       = config
        self.lookup_table = lookup_table

        self.atomiser = Atomiser_Senflood(config=config, lookup_table=lookup_table)

        ltae_cfg = config["Atomiser"].get("ltae", {})
        self.temporal = LTAE(
            in_channels         = self.atomiser.latent_dim,
            n_head              = ltae_cfg.get("n_head", 16),
            d_k                 = ltae_cfg.get("d_k", 4),
            d_model             = ltae_cfg.get("d_model", self.atomiser.latent_dim),
            dropout             = ltae_cfg.get("dropout", 0.2),
            num_freq_bands      = ltae_cfg.get("num_freq_bands", 24),
            cycle_period        = ltae_cfg.get("cycle_period", 365.0),
            positional_encoding = ltae_cfg.get("positional_encoding", True),
        )
        print(f"[AtomiserLTAE] Temporal module: LTAE "
              f"(n_head={self.temporal.n_head}, d_k={self.temporal.d_k}, "
              f"d_model={self.temporal.d_model}, "
              f"K={self.temporal.num_freq_bands}, "
              f"period={self.temporal.cycle_period}, "
              f"PE={self.temporal.positional_encoding})")

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
        mask:   torch.Tensor,   # [B, N]  False=VALID, True=PAD
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
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

        # ── 1. Partition tokens by time_idx ─────────────────────────────
        masks_per_t_by_res = {}
        unique_times = None

        for res in sorted(groups.keys()):
            m_per_t, times = self._split_tokens_by_time_idx(
                groups[res]["tokens"], groups[res]["mask"])
            masks_per_t_by_res[res] = m_per_t
            if unique_times is None:
                unique_times = times

        T = len(unique_times)

        # ── 2. Resolve DOYs (for LTAE's Fourier features) ───────────────
        dois = resolve_dois(unique_times, self.lookup_table)  # [T] floats

        # Debug print (once)
        if not getattr(AtomiserLTAE, "_debug_printed", False):
            AtomiserLTAE._debug_printed = True
            print(f"[AtomiserLTAE debug] T={T}, "
                  f"time_indices={unique_times.tolist()[:10]}")
            print(f"[AtomiserLTAE debug] resolved DOYs={dois.tolist()[:10]}")
            for res, m_list in masks_per_t_by_res.items():
                counts = [(~m).sum().item() for m in m_list]
                if counts:
                    print(f"[AtomiserLTAE debug] res={res}: tokens/t "
                          f"min={min(counts)} max={max(counts)} "
                          f"mean={sum(counts)/len(counts):.0f}")

        # ── 3. Encode each timestamp separately (full encode) ───────────
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

        # ── 4. Aggregate per-resolution via LTAE ────────────────────────
        aggregated_latents = {}
        aggregated_coords  = {}

        for res in sorted(latents_per_t[0].keys(), key=str):
            aggregated_coords[res] = coords_per_t[0][res]

            if T == 1:
                aggregated_latents[res] = latents_per_t[0][res]
                continue

            stacked = torch.stack(
                [latents_per_t[t][res] for t in range(T)], dim=2)  # [B, L, T, D]
            B, L, _, D = stacked.shape
            x = stacked.reshape(B * L, T, D)

            aggregated = self.temporal(x, dois=dois)                # [B*L, D]
            aggregated_latents[res] = aggregated.reshape(B, L, D)

        # ── 5. Decode via atomiser.reconstruct ──────────────────────────
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
            preds_list, topk_idx_list, topk_dists_list = [], [], []
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

        # ── 6. Return ───────────────────────────────────────────────────
        if return_for_error and need_topk:
            all_coords = torch.cat(
                [aggregated_coords[r]
                 for r in sorted(aggregated_coords.keys(), key=str)], dim=1)
            all_lat_post = torch.cat(
                [aggregated_latents[r]
                 for r in sorted(aggregated_latents.keys(), key=str)], dim=1)
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

    def freeze_encoder(self):   self.atomiser.freeze_encoder()
    def unfreeze_encoder(self): self.atomiser.unfreeze_encoder()
    def freeze_decoder(self):   self.atomiser.freeze_decoder()
    def unfreeze_decoder(self): self.atomiser.unfreeze_decoder()

    def freeze_temporal(self):
        for p in self.temporal.parameters(): p.requires_grad = False

    def unfreeze_temporal(self):
        for p in self.temporal.parameters(): p.requires_grad = True

    freeze_ltae   = freeze_temporal
    unfreeze_ltae = unfreeze_temporal

    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True