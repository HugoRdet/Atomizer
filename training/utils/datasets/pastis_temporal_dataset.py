"""
PastisHDTemporalDataset — S2 + S1 aligned onto a shared timestep grid.

Differs from the flat-token PastisHDDataset (SKIP variant) in exactly one
place: instead of concatenating ALL S2 frames then ALL S1 frames into one
flat pool, tokens are grouped PER TIMESTEP so the result is directly
consumable by AtomiserTemporal:

    groups[10.0]["tokens"] : [T, N, 8]   N = (C2 + C1) * H * W, constant over T
    groups[10.0]["mask"]   : [T, N]      all False — every t always has a match
    T := Ts2 (S2 drives the timestep grid; S1 is nearest-date matched to it)

Per-timestep block layout (this is what the SKIP index below assumes):
    [ S2 tokens (channel-major, c*HW + p) | S1 tokens (channel-major, c*HW + p) ]
  so token row within a timestep block:
    S2 channel c, pixel p  ->  c*HW + p
    S1 channel c, pixel p  ->  C2*HW + c*HW + p
  and the flat pool row (after AtomiserTemporal reshapes to [T*N, 8]) is:
    row = t*N + <above>

No SPOT handling — S2/S1 only, per current scope.

S1 matching is nearest ACTUAL CALENDAR DATE to each S2 frame (via
date.toordinal()), not day-of-year, to avoid year-wrap ambiguity. Each
token still carries its own true acquisition date through time_idx -> phi_t;
only the S2-vs-S1 *pairing* (which S1 frame sits in S2 frame t's block) uses
nearest-date matching. Because every S2 frame always has some nearest S1
frame, there is no padding case to handle: mask stays all-False and
query_token_valid stays all-True.
"""

import os
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from .utils_dataset_PASTIS import PastisHDDataset  # the SKIP-variant base class


class PastisHDTemporalDataset(PastisHDDataset):
    """
    Same data source and normalization as PastisHDDataset, but builds
    per-timestep [T, N, 8] token groups (S2 anchor, S1 nearest-date matched)
    instead of one flat concatenated pool. Meant for AtomiserTemporal.
    """

    def __init__(self, *args, **kwargs):
        # SPOT is out of scope for the temporal wrapper; force it off
        # regardless of what's passed, so has_spot never gets set downstream.
        kwargs["use_spot"] = False
        super().__init__(*args, **kwargs)
        if not self.use_s1:
            raise ValueError(
                "PastisHDTemporalDataset requires use_s1=True: the S1 "
                "nearest-date pairing is the whole point of this dataset."
            )

    # =========================================================================
    # Date helpers (ordinal, for nearest-match — separate from doy/phi_t)
    # =========================================================================

    @staticmethod
    def _dates_to_ordinal(dates: List) -> torch.Tensor:
        ords = []
        for date in dates:
            if isinstance(date, str):
                date = int(date)
            year, month, day = date // 10000, (date % 10000) // 100, date % 100
            ords.append(datetime(year, month, day).toordinal())
        return torch.tensor(ords, dtype=torch.long)

    @staticmethod
    def _nearest_match(anchor_ord: torch.Tensor, other_ord: torch.Tensor) -> torch.Tensor:
        """For each entry in anchor_ord, index of the closest entry in other_ord."""
        # [Ta, 1] - [1, Tb] -> [Ta, Tb]
        diffs = (anchor_ord.unsqueeze(1) - other_ord.unsqueeze(0)).abs()
        return diffs.argmin(dim=1)  # [Ta]

    # =========================================================================
    # >>> SKIP (temporal layout): closed-form gather index, per-timestep blocks
    # =========================================================================

    @staticmethod
    def _build_full_pixel_index_temporal(T, C2, C1, H, W):
        """
        Closed-form gather index for ALL pixels under the per-timestep
        [T, N, 8] -> flattened [T*N, 8] pool layout (N = (C2+C1)*HW).

        Returns [H*W, T*(C2+C1)] long: row p, col (t, sensor, c) -> flat
        row index t*N + c*HW + p        (S2 block)
                  t*N + C2*HW + c*HW + p (S1 block)
        """
        HW = H * W
        N = (C2 + C1) * HW
        p = torch.arange(HW)

        t = torch.arange(T).view(T, 1, 1)
        c2 = torch.arange(C2).view(1, C2, 1)
        s2 = (t * N + c2 * HW).reshape(-1, 1) + p.view(1, -1)              # [T*C2, HW]

        c1 = torch.arange(C1).view(1, C1, 1)
        s1 = (t * N + C2 * HW + c1 * HW).reshape(-1, 1) + p.view(1, -1)    # [T*C1, HW]

        return torch.cat([s2, s1], dim=0).t().contiguous()                 # [HW, T*(C2+C1)]

    def _build_query_token_index_temporal(self, T, C2, C1, H, W, kept_indices=None):
        full = self._build_full_pixel_index_temporal(T, C2, C1, H, W)  # [H*W, T*(C2+C1)]
        idx = full if kept_indices is None else full[kept_indices]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)             # always matched -> always valid
        return idx, valid

    # =========================================================================
    # Per-timestep token construction (S2 anchor, S1 nearest-date matched)
    # =========================================================================

    def _build_temporal_tokens_aligned(
        self, s2_data, s1_data, label, s2_time_indices, s1_time_indices, match_idx,
    ):
        """
        Returns:
            tokens: [T, N, 8]   T = Ts2, N = (C2+C1)*H*W
            mask:   [T, N]      all False
        """
        Ts2 = s2_data.shape[0]
        frames = []
        for t in range(Ts2):
            s2_tok = self.token_builder.build_tokens(
                image=s2_data[t], label=label,
                resolution=self.SAT_RESOLUTION,
                spectral_indices=self.s2_spectral_indices,
                resolution_idx=self.sat_resolution_idx,
                time_idx=s2_time_indices[t],
            )
            j = match_idx[t].item()
            s1_tok = self.token_builder.build_tokens(
                image=s1_data[j], label=label,
                resolution=self.SAT_RESOLUTION,
                spectral_indices=self.s1_spectral_indices,
                resolution_idx=self.sat_resolution_idx,
                time_idx=s1_time_indices[j],
            )
            frames.append(torch.cat([s2_tok, s1_tok], dim=0))  # [N, 8]

        tokens = torch.stack(frames, dim=0)                    # [T, N, 8]
        mask = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.bool)
        return tokens, mask

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __getitem__(self, index: int) -> Dict:
        patch_row = self.metadata.iloc[index]
        patch_id = patch_row["ID_PATCH"]

        s2_data, s2_dates = self._load_s2(patch_id, patch_row)
        s1_data, s1_dates = self._load_s1(patch_id, patch_row)
        label = self._load_label(patch_id)

        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)
        s1_data = torch.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)

        s2_data, s1_data = self._normalize_sat(s2_data, s1_data)
        s1_data = torch.clamp(s1_data, -10, 10)
        s2_data = torch.clamp(s2_data, -10, 10)

        # ── Temporal sampling (S2 drives T; S1 sampled independently, matched below) ──
        s2_data, s2_dates, _ = self._sample_temporal(s2_data, s2_dates, self.multi_temporal)
        s1_data, s1_dates, _ = self._sample_temporal(s1_data, s1_dates, self.multi_temporal)

        # ── Nearest-date S1 match for each S2 frame ──
        s2_ord = self._dates_to_ordinal(s2_dates)
        s1_ord = self._dates_to_ordinal(s1_dates)
        match_idx = self._nearest_match(s2_ord, s1_ord)   # [Ts2], indexes into s1_data/s1_dates

        # ── Convert dates to phi_t inputs (each token keeps its OWN true date) ──
        s2_doy = self._dates_to_doy(s2_dates)
        s2_time_indices = self._doy_to_time_indices(s2_doy)
        s1_doy = self._dates_to_doy(s1_dates)
        s1_time_indices = self._doy_to_time_indices(s1_doy)

        # ── D4 augmentation (applied identically to both sensors + label) ──
        if self.augment:
            import random
            d4_k = random.randint(0, 3)
            d4_flip = random.random() > 0.5
            if d4_k > 0:
                s2_data = torch.rot90(s2_data, d4_k, dims=(-2, -1))
                s1_data = torch.rot90(s1_data, d4_k, dims=(-2, -1))
                label = torch.rot90(label, d4_k, dims=(-2, -1))
            if d4_flip:
                s2_data = torch.flip(s2_data, dims=(-1,))
                s1_data = torch.flip(s1_data, dims=(-1,))
                label = torch.flip(label, dims=(-1,))

        _, _, H, W = s2_data.shape
        Ts2 = s2_data.shape[0]
        C2, C1 = self.NUM_S2_BANDS, self.NUM_S1_BANDS

        tokens, mask = self._build_temporal_tokens_aligned(
            s2_data, s1_data, label, s2_time_indices, s1_time_indices, match_idx,
        )  # tokens: [T, N, 8], mask: [T, N]

        groups = {
            self.SAT_RESOLUTION: {
                "tokens": tokens,
                "mask": mask,
                "shape": (H, W),
            },
        }

        # ── Queries (time-invariant; anchored on S2 frame 0's metadata as before) ──
        queries = self.token_builder.build_queries(
            label=label, resolution=self.SAT_RESOLUTION,
            first_spectral_idx=self.s2_spectral_indices[0],
            resolution_idx=self.sat_resolution_idx,
            time_idx=s2_time_indices[0],
        )
        queries, kept_indices = self.token_builder.subsample_queries(
            queries, max_queries=self.max_queries,
            ignore_index=self.IGNORE_INDEX, prioritize_valid=True,
            return_indices=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        # >>> SKIP: per-query gather index into the pixel's own T*(C2+C1) atoms.
        query_token_idx, query_token_valid = self._build_query_token_index_temporal(
            Ts2, C2, C1, H, W, kept_indices=kept_indices,
        )

        image = torch.cat([s2_data[0], s1_data[match_idx[0].item()]], dim=0)

        return {
            "groups": groups,
            "tasks": {self.TASK_NAME: {"queries": queries, "queries_mask": queries_mask}},
            "label": label,
            "target_resolution": self.SAT_RESOLUTION,
            "image": image,
            # >>> SKIP
            "query_token_idx": query_token_idx,      # [N_q, T*(C2+C1)]
            "query_token_valid": query_token_valid,  # [N_q] bool, all True
            # For AtomiserTemporal's TemporalTransformer positional encoding —
            # the S2 anchor date per timestep. Two DIFFERENT things, kept
            # separate on purpose:
            #   time_indices   = CATEGORICAL lookup ID (registration order,
            #                     no metric meaning) — used inside tokens for
            #                     phi_t, unrelated to temporal-transformer PE.
            #   time_positions = CONTINUOUS day-of-year (float) — what RoPE
            #                     rotates against, since it needs an actual
            #                     relative DISTANCE between timesteps, not a
            #                     categorical ID.
            "time_indices": s2_time_indices,          # [T] long, categorical
            "time_positions": s2_doy,                 # [T] float, continuous
        }


# =============================================================================
# Collate
# =============================================================================

def pastis_temporal_collate_fn(batch: List[Dict]) -> Dict:
    """
    Stacks a list of PastisHDTemporalDataset __getitem__ outputs into a batch.

    Assumes every sample in the batch has the SAME T (true whenever every
    patch has >= self.multi_temporal S2 acquisitions, which _sample_temporal
    guarantees by truncating to exactly multi_temporal — the only exception
    is a patch with FEWER raw S2 acquisitions than multi_temporal, which
    _sample_temporal returns as-is with a smaller T. If your data can hit
    that edge case, filter/pad those patches upstream before collating, since
    this function does not pad across T.)

    query_token_idx / query_token_valid are stacked as-is (same N_q across
    the batch, since subsample_queries pads/truncates to max_queries — verify
    against your token_builder.subsample_queries implementation if unsure).
    """
    resolutions = list(batch[0]["groups"].keys())

    groups = {}
    for res in resolutions:
        groups[res] = {
            "tokens": torch.stack([b["groups"][res]["tokens"] for b in batch], dim=0),  # [B,T,N,8]
            "mask": torch.stack([b["groups"][res]["mask"] for b in batch], dim=0),      # [B,T,N]
            "shape": batch[0]["groups"][res]["shape"],
        }

    task_name = list(batch[0]["tasks"].keys())[0]
    queries = torch.stack([b["tasks"][task_name]["queries"] for b in batch], dim=0)
    queries_mask = torch.stack([b["tasks"][task_name]["queries_mask"] for b in batch], dim=0)

    query_token_idx = torch.stack([b["query_token_idx"] for b in batch], dim=0)
    query_token_valid = torch.stack([b["query_token_valid"] for b in batch], dim=0)

    time_indices = torch.stack([b["time_indices"] for b in batch], dim=0)      # [B, T] long
    time_positions = torch.stack([b["time_positions"] for b in batch], dim=0)  # [B, T] float

    labels = torch.stack([b["label"] for b in batch], dim=0)
    target_resolution = batch[0]["target_resolution"]

    return {
        "groups": groups,
        "queries": queries,
        "queries_mask": queries_mask,
        "query_token_idx": query_token_idx,
        "query_token_valid": query_token_valid,
        "time_indices": time_indices,
        "time_positions": time_positions,
        "label": labels,
        "target_resolution": target_resolution,
    }
