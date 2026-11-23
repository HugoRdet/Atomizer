"""
PASTIS-HD Dataset — Multi-temporal, multi-resolution segmentation (SKIP variant)

Adds a per-query gather index `query_token_idx` / `query_token_valid` so a
decoder skip cross-attention can read each query-pixel's OWN atoms directly.
On PASTIS a pixel's atoms span BANDS x TIMESTEPS for BOTH sensors (not just
bands as on Sen1Floods11).

Everything tagged  # >>> SKIP  is new relative to the base PastisHDDataset.

POOL LAYOUT (verified against TokenBuilder.build_tokens flatten + the cats):
    sat_tokens = cat([ S2_f0(c h w), ..., S2_f(Ts2-1),  S1_f0(c h w), ..., S1_f(Ts1-1) ])
  - within a frame: channel-major (c h w)->row  => pixel p={p + c*HW} per sub-block
  - frames FRAME-MAJOR; S2 block precedes S1 block.
  pixel p atoms:
    S2: { t*C2*HW + c*HW + p : t<Ts2, c<C2 }
    S1: (Ts2*C2*HW) + { t*C1*HW + c*HW + p : t<Ts1, c<C1 }

  The index is RELATIVE TO sat_tokens (the 10m SAT group) ONLY. It does NOT
  include the optional SPOT (1m) group. The skip must gather from groups[10.0].

NOTE on splits: PASTIS subsamples queries on EVERY split (unlike Sen1Floods11
which only subsamples on train). We therefore capture kept_indices via
return_indices=True on ALL splits, so the gather index always matches the
(reordered) queries.
"""

import os
import json
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .token_builder import TokenBuilder

try:
    import pandas as pd
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("[Warning] geopandas not installed. Install with: pip install geopandas")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


class PastisHDDataset(Dataset):
    """
    PASTIS-HD Dataset — Multi-temporal, multi-resolution segmentation, SKIP variant.
    Flat temporal tokens only (one token per pixel × band × frame).
    """

    SAT_RESOLUTION  = 10.0
    SPOT_RESOLUTION = 1.0

    NUM_S2_BANDS   = 10
    NUM_S1_BANDS   = 3       # VV, VH, VV/VH ratio (matches PANGAEA convention)
    NUM_SPOT_BANDS = 3

    NUM_CLASSES  = 20
    IGNORE_INDEX = 255
    TIME_IDX_NA  = -1

    S2_BANDS_INFO = {
        "B02": {"central_wavelength": 490,  "bandwidth": 65,  "idx": 0},
        "B03": {"central_wavelength": 560,  "bandwidth": 35,  "idx": 1},
        "B04": {"central_wavelength": 665,  "bandwidth": 30,  "idx": 2},
        "B05": {"central_wavelength": 705,  "bandwidth": 15,  "idx": 3},
        "B06": {"central_wavelength": 740,  "bandwidth": 15,  "idx": 4},
        "B07": {"central_wavelength": 783,  "bandwidth": 20,  "idx": 5},
        "B08": {"central_wavelength": 842,  "bandwidth": 115, "idx": 6},
        "B8A": {"central_wavelength": 865,  "bandwidth": 20,  "idx": 7},
        "B11": {"central_wavelength": 1610, "bandwidth": 90,  "idx": 8},
        "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 9},
    }

    S1_BANDS_INFO = {
        "VV":    {"central_wavelength": -1, "bandwidth": -1, "idx": 0},
        "VH":    {"central_wavelength": -2, "bandwidth": -2, "idx": 1},
        "VV_VH": {"central_wavelength": -3, "bandwidth": -3, "idx": 2},
    }

    SPOT_BANDS_INFO = {
        "SPOT_R": {"central_wavelength": 660, "bandwidth": 120, "idx": 0},
        "SPOT_G": {"central_wavelength": 560, "bandwidth": 120, "idx": 1},
        "SPOT_B": {"central_wavelength": 490, "bandwidth": 140, "idx": 2},
    }

    TASK_NAME = "pastis_segmentation"

    def __init__(
        self,
        root_path: str = "./data/PASTIS-HD",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
        use_s1: bool = True,
        use_spot: bool = True,
    ):
        super().__init__()

        self.root_path    = root_path
        self.split        = mode
        self.look_up      = look_up
        self.config_model = config_model
        self.use_s1       = use_s1
        self.use_spot     = use_spot
        self.augment      = (mode == "train")

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens   = config_model["trainer"]["max_tokens"]
        self.max_queries = config_model["trainer"].get("max_tokens_reconstruction", 100_000)

        self.max_temporal_samples = config_model.get("dataset", {}).get("max_temporal_samples", 50)
        self.multi_temporal       = config_model.get("dataset", {}).get("multi_temporal", 10)
        self.reference_date       = datetime(2018, 9, 1)

        self.split_mapping = {
            "train":      "train",
            "validation": "val",
            "test":       "test",
        }

        self.spot_dir = os.path.join(
            root_path, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019")
        self.has_spot = (
            self.use_spot
            and os.path.isdir(self.spot_dir)
            and HAS_RASTERIO
        )

        self._load_metadata()
        self._setup_band_indices()

        self.sat_resolution_idx = self.look_up.get_resolution_idx(self.SAT_RESOLUTION)
        if self.has_spot:
            self.spot_resolution_idx = self.look_up.get_resolution_idx(self.SPOT_RESOLUTION)

        self.norm_stats = self._load_or_compute_normalization()

        print(f"[PASTIS-HD-SKIP] {len(self.patch_ids)} patches, split='{self.split}'")
        print(f"[PASTIS-HD-SKIP] S2: {self.NUM_S2_BANDS} bands @ {self.SAT_RESOLUTION}m")
        print(f"[PASTIS-HD-SKIP] S1: {'enabled' if self.use_s1 else 'DISABLED'} "
              f"({self.NUM_S1_BANDS} bands)")
        print(f"[PASTIS-HD-SKIP] SPOT: {'available' if self.has_spot else 'NOT found/disabled'}")
        if self.has_spot:
            print(f"[PASTIS-HD-SKIP] WARNING: SPOT group present. The skip gather index "
                  f"covers the 10m SAT group ONLY; ensure the model's _pixel_skip "
                  f"gathers from groups[{self.SAT_RESOLUTION}], not a merged pool.")
        print(f"[PASTIS-HD-SKIP] Temporal: {self.multi_temporal} frames "
              f"(evenly spaced — matches PANGAEA)")

    # =========================================================================
    # >>> SKIP: per-query gather index into own band×time atoms
    # =========================================================================

    @staticmethod
    def _build_full_pixel_index(Ts2, C2, Ts1, C1, H, W):
        """
        Closed-form gather index for ALL pixels, pixel order p = h*W + w.

        Ts2/Ts1 are the ACTUAL frame counts of the S2/S1 token blocks (derived
        from the built tensors, so S1!=S2 frame counts are handled correctly).
        C1=0 disables the S1 block.

        Returns [H*W, Ts2*C2 + Ts1*C1] long. Verified numerically against
        build_tokens' einops flatten + the dataset's frame/sensor cats.
        """
        HW = H * W
        p = torch.arange(HW)                                          # [HW]
        blocks = []

        # S2 sub-block: t*C2*HW + c*HW + p
        t2 = torch.arange(Ts2).view(Ts2, 1, 1)
        c2 = torch.arange(C2).view(1, C2, 1)
        s2 = (t2 * C2 * HW + c2 * HW).reshape(-1, 1) + p.view(1, -1)  # [Ts2*C2, HW]
        blocks.append(s2)

        # S1 sub-block (offset by full S2 block): off + t*C1*HW + c*HW + p
        if C1 > 0 and Ts1 > 0:
            off = Ts2 * C2 * HW
            t1 = torch.arange(Ts1).view(Ts1, 1, 1)
            c1 = torch.arange(C1).view(1, C1, 1)
            s1 = (off + t1 * C1 * HW + c1 * HW).reshape(-1, 1) + p.view(1, -1)  # [Ts1*C1, HW]
            blocks.append(s1)

        return torch.cat(blocks, dim=0).t().contiguous()             # [HW, Ts2*C2 + Ts1*C1]

    def _build_query_token_index(self, Ts2, C2, Ts1, C1, H, W, kept_indices=None):
        """
        Per-query gather index into the pixel's own atoms (bands x timesteps).

        kept_indices: [N_q] long or None. Row positions (into the full pixel
        grid) kept by subsample_queries, in the SAME order as returned queries.
        None -> full grid in pixel order.

        Returns:
            idx   : [N_q, Ts2*C2 + Ts1*C1] long  -- rows into sat_tokens
            valid : [N_q] bool                    -- all True (closed form)
        """
        full = self._build_full_pixel_index(Ts2, C2, Ts1, C1, H, W)  # [H*W, A]
        idx = full if kept_indices is None else full[kept_indices]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)
        return idx, valid

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> Dict:
        patch_row = self.metadata.iloc[index]
        patch_id  = patch_row["ID_PATCH"]

        s2_data, s2_dates = self._load_s2(patch_id, patch_row)
        label = self._load_label(patch_id)
        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_s1:
            s1_data, s1_dates = self._load_s1(patch_id, patch_row)
            s1_data = torch.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ────────────────────────────────────────────────
        if self.use_s1:
            s2_data, s1_data = self._normalize_sat(s2_data, s1_data)
            s1_data = torch.clamp(s1_data, -10, 10)
        else:
            s2_mean = self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_std  = self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_data = (s2_data - s2_mean) / s2_std.clamp(min=1e-6)
        s2_data = torch.clamp(s2_data, -10, 10)

        # ── Temporal sampling (evenly spaced — PANGAEA convention) ───
        s2_data, s2_dates, _ = self._sample_temporal(
            s2_data, s2_dates, self.multi_temporal)
        if self.use_s1:
            s1_data, s1_dates, _ = self._sample_temporal(
                s1_data, s1_dates, self.multi_temporal)

        # ── Convert dates ────────────────────────────────────────────
        s2_doy          = self._dates_to_doy(s2_dates)
        s2_time_indices = self._doy_to_time_indices(s2_doy)
        if self.use_s1:
            s1_doy          = self._dates_to_doy(s1_dates)
            s1_time_indices = self._doy_to_time_indices(s1_doy)

        # ── D4 augmentation ──────────────────────────────────────────
        if self.augment:
            d4_k    = random.randint(0, 3)
            d4_flip = random.random() > 0.5
            if d4_k > 0:
                s2_data = torch.rot90(s2_data, d4_k, dims=(-2, -1))
                label   = torch.rot90(label,   d4_k, dims=(-2, -1))
                if self.use_s1:
                    s1_data = torch.rot90(s1_data, d4_k, dims=(-2, -1))
            if d4_flip:
                s2_data = torch.flip(s2_data, dims=(-1,))
                label   = torch.flip(label,   dims=(-1,))
                if self.use_s1:
                    s1_data = torch.flip(s1_data, dims=(-1,))

        # ── Build tokens ─────────────────────────────────────────────
        _, _, H, W = s2_data.shape

        s2_tokens = self._build_temporal_tokens(
            s2_data, label, s2_time_indices,
            self.s2_spectral_indices, self.sat_resolution_idx)
        Ts2 = s2_data.shape[0]                       # >>> SKIP: actual S2 frame count

        if self.use_s1:
            s1_tokens = self._build_temporal_tokens(
                s1_data, label, s1_time_indices,
                self.s1_spectral_indices, self.sat_resolution_idx)
            Ts1 = s1_data.shape[0]                   # >>> SKIP: actual S1 frame count
            sat_tokens = torch.cat([s2_tokens, s1_tokens], dim=0)
        else:
            Ts1 = 0
            sat_tokens = s2_tokens

        sat_mask = torch.zeros(sat_tokens.shape[0], dtype=torch.bool)
        groups   = {
            self.SAT_RESOLUTION: {
                "tokens": sat_tokens,
                "mask":   sat_mask,
                "shape":  (H, W),
            },
        }

        # ── SPOT (optional) ──────────────────────────────────────────
        if self.has_spot:
            spot_data = self._load_spot(patch_id)
            if spot_data is not None:
                spot_data = torch.nan_to_num(spot_data)
                spot_data = torch.clamp(self._normalize_spot(spot_data), -10, 10)
                _, H_hr, W_hr = spot_data.shape
                label_hr = torch.nn.functional.interpolate(
                    label.float().unsqueeze(0).unsqueeze(0),
                    size=(H_hr, W_hr), mode="nearest",
                ).squeeze(0).squeeze(0).long()
                spot_tokens = self.token_builder.build_tokens(
                    image=spot_data, label=label_hr,
                    resolution=self.SPOT_RESOLUTION,
                    spectral_indices=self.spot_spectral_indices,
                    resolution_idx=self.spot_resolution_idx,
                    time_idx=self.TIME_IDX_NA,
                )
                groups[self.SPOT_RESOLUTION] = {
                    "tokens": spot_tokens,
                    "mask":   torch.zeros(spot_tokens.shape[0], dtype=torch.bool),
                    "shape":  (H_hr, W_hr),
                }

        # ── Queries ──────────────────────────────────────────────────
        queries = self.token_builder.build_queries(
            label=label, resolution=self.SAT_RESOLUTION,
            first_spectral_idx=self.s2_spectral_indices[0],
            resolution_idx=self.sat_resolution_idx,
            time_idx=s2_time_indices[0],
        )
        # >>> SKIP: capture kept_indices on EVERY split (PASTIS subsamples all splits)
        queries, kept_indices = self.token_builder.subsample_queries(
            queries, max_queries=self.max_queries,
            ignore_index=self.IGNORE_INDEX, prioritize_valid=True,
            return_indices=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        # >>> SKIP: per-query gather index into the pixel's band×time atoms.
        C1 = self.NUM_S1_BANDS if self.use_s1 else 0
        query_token_idx, query_token_valid = self._build_query_token_index(
            Ts2, self.NUM_S2_BANDS, Ts1, C1, H, W, kept_indices=kept_indices,
        )

        image = torch.cat([s2_data[0], s1_data[0]], dim=0) if self.use_s1 else s2_data[0]

        return {
            "groups": groups,
            "tasks":  {self.TASK_NAME: {"queries": queries, "queries_mask": queries_mask}},
            "label":  label,
            "target_resolution": self.SAT_RESOLUTION,
            "image":  image,
            # >>> SKIP
            "query_token_idx":   query_token_idx,    # [N_q, Ts2*C2 + Ts1*C1]
            "query_token_valid": query_token_valid,  # [N_q] bool
        }

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_temporal_tokens(self, data, label, time_indices,
                                spectral_indices, resolution_idx):
        """Build flat tokens for multi-temporal data. Returns [T*C*H*W, 8]."""
        frames = []
        for t in range(data.shape[0]):
            frames.append(self.token_builder.build_tokens(
                image=data[t], label=label,
                resolution=self.SAT_RESOLUTION,
                spectral_indices=spectral_indices,
                resolution_idx=resolution_idx,
                time_idx=time_indices[t],
            ))
        return torch.cat(frames, dim=0)

    # =========================================================================
    # DATE HANDLING
    # =========================================================================

    def _dates_to_doy(self, dates) -> torch.Tensor:
        doy_list = []
        for date in dates:
            if isinstance(date, str):
                date = int(date)
            year  = date // 10000
            month = (date % 10000) // 100
            day   = date % 100
            doy   = datetime(year, month, day).timetuple().tm_yday
            doy_list.append(doy)
        return torch.tensor(doy_list, dtype=torch.float32)

    def _doy_to_time_indices(self, doy: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [self.look_up.get_or_register_time_idx(int(d)) for d in doy],
            dtype=torch.long)

    # =========================================================================
    # TEMPORAL SAMPLING
    # =========================================================================

    def _sample_temporal(self, data, dates, n_samples):
        """Evenly spaced frames via torch.linspace — matches PANGAEA."""
        T = data.shape[0]
        if T <= n_samples:
            return data, dates, torch.arange(T)
        indices = torch.linspace(0, T - 1, n_samples, dtype=torch.long)
        return data[indices], [dates[i.item()] for i in indices], indices

    # =========================================================================
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        """Full-resolution sample for visualization — all pixels as queries."""
        patch_row = self.metadata.iloc[index]
        patch_id  = patch_row["ID_PATCH"]

        s2_data, s2_dates = self._load_s2(patch_id, patch_row)
        label = self._load_label(patch_id)
        s2_data = torch.nan_to_num(s2_data)

        if self.use_s1:
            s1_data, s1_dates = self._load_s1(patch_id, patch_row)
            s1_data = torch.nan_to_num(s1_data)
            s2_data, s1_data = self._normalize_sat(s2_data, s1_data)
            s1_data = torch.clamp(s1_data, -10, 10)
        else:
            s2_mean = self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_std  = self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_data = (s2_data - s2_mean) / s2_std.clamp(min=1e-6)
        s2_data = torch.clamp(s2_data, -10, 10)

        s2_data, s2_dates, _ = self._sample_temporal(s2_data, s2_dates, self.multi_temporal)
        if self.use_s1:
            s1_data, s1_dates, _ = self._sample_temporal(s1_data, s1_dates, self.multi_temporal)

        s2_doy          = self._dates_to_doy(s2_dates)
        s2_time_indices = self._doy_to_time_indices(s2_doy)
        if self.use_s1:
            s1_doy          = self._dates_to_doy(s1_dates)
            s1_time_indices = self._doy_to_time_indices(s1_doy)

        _, _, H, W = s2_data.shape

        s2_tokens = self._build_temporal_tokens(
            s2_data, label, s2_time_indices,
            self.s2_spectral_indices, self.sat_resolution_idx)
        Ts2 = s2_data.shape[0]

        if self.use_s1:
            s1_tokens = self._build_temporal_tokens(
                s1_data, label, s1_time_indices,
                self.s1_spectral_indices, self.sat_resolution_idx)
            Ts1 = s1_data.shape[0]
            sat_tokens = torch.cat([s2_tokens, s1_tokens], dim=0)
        else:
            Ts1 = 0
            sat_tokens = s2_tokens

        sat_mask = torch.zeros(sat_tokens.shape[0], dtype=torch.bool)
        groups   = {self.SAT_RESOLUTION: {"tokens": sat_tokens, "mask": sat_mask, "shape": (H, W)}}

        if self.has_spot:
            spot_data = self._load_spot(patch_id)
            if spot_data is not None:
                spot_data = torch.clamp(self._normalize_spot(
                    torch.nan_to_num(spot_data)), -10, 10)
                _, H_hr, W_hr = spot_data.shape
                label_hr = torch.nn.functional.interpolate(
                    label.float().unsqueeze(0).unsqueeze(0),
                    size=(H_hr, W_hr), mode="nearest").squeeze(0).squeeze(0).long()
                spot_tokens = self.token_builder.build_tokens(
                    image=spot_data, label=label_hr,
                    resolution=self.SPOT_RESOLUTION,
                    spectral_indices=self.spot_spectral_indices,
                    resolution_idx=self.spot_resolution_idx,
                    time_idx=self.TIME_IDX_NA)
                groups[self.SPOT_RESOLUTION] = {
                    "tokens": spot_tokens,
                    "mask":   torch.zeros(spot_tokens.shape[0], dtype=torch.bool),
                    "shape":  (H_hr, W_hr),
                }

        queries = self.token_builder.build_queries(
            label=label, resolution=self.SAT_RESOLUTION,
            first_spectral_idx=self.s2_spectral_indices[0],
            resolution_idx=self.sat_resolution_idx,
            time_idx=s2_time_indices[0])
        # >>> SKIP: viz uses full seg_queries in pixel order (no subsample) -> kept=None
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        C1 = self.NUM_S1_BANDS if self.use_s1 else 0
        query_token_idx, query_token_valid = self._build_query_token_index(
            Ts2, self.NUM_S2_BANDS, Ts1, C1, H, W, kept_indices=None)

        image = torch.cat([s2_data[0], s1_data[0]], dim=0) if self.use_s1 else s2_data[0]

        return {
            "groups": groups,
            "tasks":  {self.TASK_NAME: {"queries": queries, "queries_mask": queries_mask}},
            "label":  label,
            "target_resolution": self.SAT_RESOLUTION,
            "image":  image,
            "image_shape": (H, W),
            "n_queries": queries.shape[0],
            # >>> SKIP
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }

    # =========================================================================
    # METADATA
    # =========================================================================

    def _load_metadata(self):
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required for PASTIS-HD dataset.")

        metadata_path = os.path.join(self.root_path, "metadata.geojson")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.geojson not found at {metadata_path}")

        self.metadata = gpd.read_file(metadata_path)

        fold_mapping = {"train": [1, 2, 3], "val": [4], "test": [5]}
        mapped_split = self.split_mapping.get(self.split, self.split)
        folds = fold_mapping.get(mapped_split)
        if folds is None:
            raise ValueError(f"Invalid split '{self.split}'")

        self.metadata = pd.concat(
            [self.metadata[self.metadata["Fold"] == f] for f in folds]
        ).reset_index(drop=True)
        self.patch_ids = self.metadata["ID_PATCH"].tolist()

    def _setup_band_indices(self):
        self.s2_spectral_indices = []
        for name in sorted(self.S2_BANDS_INFO, key=lambda b: self.S2_BANDS_INFO[b]["idx"]):
            info = self.S2_BANDS_INFO[name]
            key  = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(f"S2 band {name} key={key} not in lookup.")
            self.s2_spectral_indices.append(self.look_up.table_wave[key])
        self.s2_spectral_indices = torch.tensor(self.s2_spectral_indices, dtype=torch.long)

        if self.use_s1:
            self.s1_spectral_indices = []
            for name in sorted(self.S1_BANDS_INFO, key=lambda b: self.S1_BANDS_INFO[b]["idx"]):
                info = self.S1_BANDS_INFO[name]
                key  = (info["bandwidth"], info["central_wavelength"])
                if key not in self.look_up.table_wave:
                    raise KeyError(f"S1 band {name} key={key} not in lookup.")
                self.s1_spectral_indices.append(self.look_up.table_wave[key])
            self.s1_spectral_indices = torch.tensor(self.s1_spectral_indices, dtype=torch.long)

        if self.has_spot:
            self.spot_spectral_indices = []
            for name in sorted(self.SPOT_BANDS_INFO, key=lambda b: self.SPOT_BANDS_INFO[b]["idx"]):
                info = self.SPOT_BANDS_INFO[name]
                key  = (info["bandwidth"], info["central_wavelength"])
                if key not in self.look_up.table_wave:
                    self.has_spot = False
                    return
                self.spot_spectral_indices.append(self.look_up.table_wave[key])
            self.spot_spectral_indices = torch.tensor(self.spot_spectral_indices, dtype=torch.long)

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_s2(self, patch_id, patch_row):
        path   = os.path.join(self.root_path, "DATA_S2", f"S2_{patch_id}.npy")
        s2     = torch.from_numpy(np.load(path).astype(np.float32))
        dates  = self._parse_dates(patch_row, "dates-S2", s2.shape[0])
        if len(dates) > self.max_temporal_samples:
            idx   = torch.linspace(0, len(dates) - 1, self.max_temporal_samples, dtype=torch.long)
            s2    = s2[idx]
            dates = [dates[i] for i in idx]
        return s2, dates

    def _load_s1(self, patch_id, patch_row):
        path   = os.path.join(self.root_path, "DATA_S1A", f"S1A_{patch_id}.npy")
        s1     = torch.from_numpy(np.load(path).astype(np.float32))
        s1     = s1[:, :self.NUM_S1_BANDS]
        dates  = self._parse_dates(patch_row, "dates-S1A", s1.shape[0])
        if len(dates) > self.max_temporal_samples:
            idx   = torch.linspace(0, len(dates) - 1, self.max_temporal_samples, dtype=torch.long)
            s1    = s1[idx]
            dates = [dates[i] for i in idx]
        return s1, dates

    def _parse_dates(self, patch_row, key, T_fallback):
        if key in patch_row:
            d = patch_row[key]
            if isinstance(d, str):
                d = json.loads(d)
            return pd.DataFrame.from_dict(d, orient="index")[0].tolist()
        return list(range(T_fallback))

    def _load_spot(self, patch_id):
        path = os.path.join(self.spot_dir, f"SPOT6_RVB_1M00_2019_{patch_id}.tif")
        if not os.path.exists(path):
            return None
        try:
            with rasterio.open(path) as src:
                return torch.from_numpy(src.read().astype(np.float32))
        except Exception as e:
            print(f"[PASTIS-HD] Warning: SPOT load failed {path}: {e}")
            return None

    def _load_label(self, patch_id):
        path  = os.path.join(self.root_path, "ANNOTATIONS", f"TARGET_{patch_id}.npy")
        label = torch.from_numpy(np.load(path)[0].astype(np.int64))
        label[label < 0]                  = self.IGNORE_INDEX
        label[label == 19]                = self.IGNORE_INDEX
        label[label >= self.NUM_CLASSES]  = self.IGNORE_INDEX
        return label

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)
            if "s1_mean" in stats and stats["s1_mean"].shape[0] != self.NUM_S1_BANDS:
                os.remove(norm_file)
            else:
                self._print_norm_stats(stats)
                return stats

        if self.split != "train":
            stats = {
                "s2_mean": torch.zeros(self.NUM_S2_BANDS),
                "s2_std":  torch.ones(self.NUM_S2_BANDS),
                "s1_mean": torch.zeros(self.NUM_S1_BANDS),
                "s1_std":  torch.ones(self.NUM_S1_BANDS),
            }
            if self.has_spot:
                stats["spot_mean"] = torch.zeros(self.NUM_SPOT_BANDS)
                stats["spot_std"]  = torch.ones(self.NUM_SPOT_BANDS)
            return stats

        print(f"[PASTIS-HD] Computing normalization from {len(self.patch_ids)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(self.NUM_S2_BANDS,   dtype=torch.float64)
        s2_sq  = torch.zeros(self.NUM_S2_BANDS,   dtype=torch.float64)
        s2_n   = torch.zeros(self.NUM_S2_BANDS,   dtype=torch.float64)
        s1_sum = torch.zeros(self.NUM_S1_BANDS,   dtype=torch.float64)
        s1_sq  = torch.zeros(self.NUM_S1_BANDS,   dtype=torch.float64)
        s1_n   = torch.zeros(self.NUM_S1_BANDS,   dtype=torch.float64)
        sp_sum = torch.zeros(self.NUM_SPOT_BANDS,  dtype=torch.float64)
        sp_sq  = torch.zeros(self.NUM_SPOT_BANDS,  dtype=torch.float64)
        sp_n   = torch.zeros(self.NUM_SPOT_BANDS,  dtype=torch.float64)

        for pid in tqdm(self.patch_ids, desc="Norm stats"):
            try:
                s2 = np.nan_to_num(np.load(
                    os.path.join(self.root_path, "DATA_S2", f"S2_{pid}.npy")
                ).astype(np.float64))
                for c in range(self.NUM_S2_BANDS):
                    v = s2[:, c].flatten(); v = v[v != 0]
                    if len(v):
                        s2_sum[c] += v.sum(); s2_sq[c] += (v**2).sum(); s2_n[c] += len(v)
            except Exception:
                pass
            try:
                s1 = np.nan_to_num(np.load(
                    os.path.join(self.root_path, "DATA_S1A", f"S1A_{pid}.npy")
                ).astype(np.float64))
                for c in range(self.NUM_S1_BANDS):
                    v = s1[:, c].flatten(); v = v[v != 0]
                    if len(v):
                        s1_sum[c] += v.sum(); s1_sq[c] += (v**2).sum(); s1_n[c] += len(v)
            except Exception:
                pass
            if self.has_spot:
                try:
                    sp_path = os.path.join(self.spot_dir, f"SPOT6_RVB_1M00_2019_{pid}.tif")
                    if os.path.exists(sp_path):
                        with rasterio.open(sp_path) as src:
                            sp = np.nan_to_num(src.read().astype(np.float64))
                        for c in range(self.NUM_SPOT_BANDS):
                            v = sp[c].flatten(); v = v[v != 0]
                            if len(v):
                                sp_sum[c] += v.sum(); sp_sq[c] += (v**2).sum(); sp_n[c] += len(v)
                except Exception:
                    pass

        def _stats(s, sq, n):
            mean = (s / n.clamp(min=1)).float()
            std  = ((sq / n.clamp(min=1) - mean.double()**2).clamp(min=0).sqrt()).float()
            return mean, std

        s2_mean, s2_std = _stats(s2_sum, s2_sq, s2_n)
        s1_mean, s1_std = _stats(s1_sum, s1_sq, s1_n)
        stats = {"s2_mean": s2_mean, "s2_std": s2_std,
                 "s1_mean": s1_mean, "s1_std": s1_std}
        if self.has_spot:
            sp_mean, sp_std = _stats(sp_sum, sp_sq, sp_n)
            stats["spot_mean"] = sp_mean
            stats["spot_std"]  = sp_std
        return stats

    def _print_norm_stats(self, stats):
        print(f"[PASTIS-HD] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[PASTIS-HD] S2 std:  {stats['s2_std'].numpy()}")
        if "s1_mean" in stats:
            print(f"[PASTIS-HD] S1 mean: {stats['s1_mean'].numpy()}")
            print(f"[PASTIS-HD] S1 std:  {stats['s1_std'].numpy()}")

    def _normalize_sat(self, s2, s1):
        s2 = (s2 - self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)) \
             / self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1).clamp(min=1e-6)
        s1 = (s1 - self.norm_stats["s1_mean"].view(1, self.NUM_S1_BANDS, 1, 1)) \
             / self.norm_stats["s1_std"].view(1, self.NUM_S1_BANDS, 1, 1).clamp(min=1e-6)
        return s2, s1

    def _normalize_spot(self, spot):
        if "spot_mean" not in self.norm_stats:
            return spot
        return (spot - self.norm_stats["spot_mean"].view(self.NUM_SPOT_BANDS, 1, 1)) \
               / self.norm_stats["spot_std"].view(self.NUM_SPOT_BANDS, 1, 1).clamp(min=1e-6)
