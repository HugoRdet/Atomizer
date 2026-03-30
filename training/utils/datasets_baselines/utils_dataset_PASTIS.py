"""
PASTIS-HD Dataset — Multi-temporal, multi-resolution segmentation

Modalities:
    - S2:   10 bands @ 10m, multi-temporal
    - S1A:  3 bands (VV, VH, VV-VH) @ 10m, multi-temporal
    - SPOT: 3 bands (R, G, B) @ 1m, single frame (optional)

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

All temporal frames are flattened into one token pool per resolution.
Temporal information is encoded in column 7 (time_idx).

Returns:
{
    "groups": {
        10.0: {
            "tokens": [N_sat, 8],     # all S2+S1 frames flat
            "mask":   [N_sat],        # bool, False=valid
            "shape":  (H, W),         # spatial dims for grid config
        },
        1.0: {                        # only if SPOT available
            "tokens": [N_spot, 8],
            "mask":   [N_spot],
            "shape":  (H_hr, W_hr),
        },
    },
    "queries":           [M, 8],
    "queries_mask":      [M],
    "label":             [H, W],
    "target_resolution": 10.0,
    "image":             [13, H, W],   # first S2(10) + first S1(3) frame
}

Directory structure:
./data/PASTIS-HD/
├── metadata.geojson
├── DATA_S2/
│   └── S2_{patch_id}.npy                           # [T, 10, H, W]
├── DATA_S1A/
│   └── S1A_{patch_id}.npy                          # [T, 3, H, W]
├── DATA_SPOT/
│   └── PASTIS_SPOT6_RVB_1M00_2019/
│       └── SPOT6_RVB_1M00_2019_{patch_id}.tif      # [3, H_hr, W_hr]
└── ANNOTATIONS/
    └── TARGET_{patch_id}.npy                       # [1, H, W]
"""

import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple

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
    PASTIS-HD Dataset — Multi-temporal, multi-resolution segmentation.

    Two resolution groups:
        10.0 m/px: S2 (10 bands × T_s2) + S1A (3 bands × T_s1) — all flat
         1.0 m/px: SPOT6 (3 RGB bands × 1 frame) — flat (optional)

    Temporal info is encoded per-token in column 7 (time_idx).
    No T dimension in output — everything is a flat token pool.

    Mask Convention:
        False (0) = valid token
        True (1)  = invalid/masked token
    """

    # Resolutions
    SAT_RESOLUTION = 10.0    # m/px for S2 + S1
    SPOT_RESOLUTION = 1.0    # m/px for SPOT6

    # Band counts
    NUM_S2_BANDS = 10
    NUM_S1_BANDS = 3         # VV, VH, VV-VH
    NUM_SPOT_BANDS = 3       # R, G, B

    NUM_CLASSES = 20
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1         # For SPOT (single frame, no temporal)

    # S2 band info (Sentinel-2 at 10m)
    S2_BANDS_INFO = {
        "B02": {"central_wavelength": 490, "bandwidth": 65, "idx": 0},
        "B03": {"central_wavelength": 560, "bandwidth": 35, "idx": 1},
        "B04": {"central_wavelength": 665, "bandwidth": 30, "idx": 2},
        "B05": {"central_wavelength": 705, "bandwidth": 15, "idx": 3},
        "B06": {"central_wavelength": 740, "bandwidth": 15, "idx": 4},
        "B07": {"central_wavelength": 783, "bandwidth": 20, "idx": 5},
        "B08": {"central_wavelength": 842, "bandwidth": 115, "idx": 6},
        "B8A": {"central_wavelength": 865, "bandwidth": 20, "idx": 7},
        "B11": {"central_wavelength": 1610, "bandwidth": 90, "idx": 8},
        "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 9},
    }

    # S1A band info (SAR — 3 channels: VV, VH, VV-VH)
    S1_BANDS_INFO = {
        "VV":    {"central_wavelength": -1, "bandwidth": -1, "idx": 0},
        "VH":    {"central_wavelength": -2, "bandwidth": -2, "idx": 1},
        "VV-VH": {"central_wavelength": -3, "bandwidth": -3, "idx": 2},
    }

    # SPOT6 band info (RGB at 1m — file order is R, G, B)
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
        temporal_last: bool = False,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model
        self.use_s1 = use_s1
        self.use_spot = use_spot
        self.temporal_last = temporal_last
        self.augment = (mode == "train")

        # Initialize TokenBuilder
        self.token_builder = TokenBuilder(look_up)

        # Config parameters
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_queries = config_model["trainer"].get("max_tokens_reconstruction", 100_000)

        # Temporal parameters
        self.max_temporal_samples = config_model.get("dataset", {}).get("max_temporal_samples", 50)
        self.multi_temporal = config_model.get("dataset", {}).get("multi_temporal", 10)
        self.reference_date = datetime(2018, 9, 1)

        # Split mapping (UnifiedDataModule passes "validation", PASTIS uses "val")
        self.split_mapping = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }

        # Check SPOT availability (filesystem check AND user toggle)
        self.spot_dir = os.path.join(
            root_path, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019"
        )
        self.has_spot = (
            self.use_spot
            and os.path.isdir(self.spot_dir)
            and HAS_RASTERIO
        )

        # Load metadata and setup splits
        self._load_metadata()

        # Setup band indices
        self._setup_band_indices()

        # Resolution indices
        self.sat_resolution_idx = self.look_up.get_resolution_idx(self.SAT_RESOLUTION)
        if self.has_spot:
            self.spot_resolution_idx = self.look_up.get_resolution_idx(self.SPOT_RESOLUTION)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[PASTIS-HD] Loaded {len(self.patch_ids)} patches for split '{self.split}'")
        print(f"[PASTIS-HD] S2: {self.NUM_S2_BANDS} bands @ {self.SAT_RESOLUTION}m")
        print(f"[PASTIS-HD] S1: {'enabled' if self.use_s1 else 'DISABLED'} "
              f"({self.NUM_S1_BANDS} bands @ {self.SAT_RESOLUTION}m)")
        print(f"[PASTIS-HD] SPOT: {'available' if self.has_spot else 'NOT found/disabled'} "
              f"({self.NUM_SPOT_BANDS} bands @ {self.SPOT_RESOLUTION}m)")
        print(f"[PASTIS-HD] Multi-temporal: {self.multi_temporal} frames"
              f" ({'last' if self.temporal_last else 'uniform'})")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> Dict:
        patch_row = self.metadata.iloc[index]
        patch_id = patch_row["ID_PATCH"]

        # ── Load ────────────────────────────────────────────
        s2_data, s2_dates = self._load_s2(patch_id, patch_row)  # [T_s2, 10, H, W]
        label = self._load_label(patch_id)                       # [H, W]

        # ── Clean ───────────────────────────────────────────
        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Load S1 (conditional) ───────────────────────────
        if self.use_s1:
            s1_data, s1_dates = self._load_s1(patch_id, patch_row)  # [T_s1, 2, H, W]
            s1_data = torch.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ───────────────────────────────────────
        if self.use_s1:
            s2_data, s1_data = self._normalize_sat(s2_data, s1_data)
            s1_data = torch.clamp(s1_data, -10, 10)
        else:
            s2_mean = self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_std  = self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_data = (s2_data - s2_mean) / s2_std.clamp(min=1e-6)
        s2_data = torch.clamp(s2_data, -10, 10)

        # ── Sample temporal dimension ───────────────────────
        s2_data, s2_dates, _ = self._sample_temporal(s2_data, s2_dates, self.multi_temporal)
        if self.use_s1:
            s1_data, s1_dates, _ = self._sample_temporal(s1_data, s1_dates, self.multi_temporal)

        # ── Convert dates to time indices ───────────────────
        s2_time_indices = self._dates_to_indices(s2_dates)
        if self.use_s1:
            s1_time_indices = self._dates_to_indices(s1_dates)

        # ── D4 augmentation ─────────────────────────────────
        if self.augment:
            d4_k = random.randint(0, 3)
            d4_flip = random.random() > 0.5

            if d4_k > 0:
                s2_data = torch.rot90(s2_data, d4_k, dims=(-2, -1))
                label = torch.rot90(label, d4_k, dims=(-2, -1))
                if self.use_s1:
                    s1_data = torch.rot90(s1_data, d4_k, dims=(-2, -1))

            if d4_flip:
                s2_data = torch.flip(s2_data, dims=(-1,))
                label = torch.flip(label, dims=(-1,))
                if self.use_s1:
                    s1_data = torch.flip(s1_data, dims=(-1,))

        # ── Build flat tokens ───────────────────────────────
        s2_tokens = self._build_temporal_tokens(
            s2_data, label, s2_time_indices,
            self.s2_spectral_indices, self.sat_resolution_idx,
        )

        if self.use_s1:
            s1_tokens = self._build_temporal_tokens(
                s1_data, label, s1_time_indices,
                self.s1_spectral_indices, self.sat_resolution_idx,
            )
            sat_tokens = torch.cat([s2_tokens, s1_tokens], dim=0)
        else:
            sat_tokens = s2_tokens

        sat_mask = torch.zeros(sat_tokens.shape[0], dtype=torch.bool)

        _, _, H, W = s2_data.shape

        groups = {
            self.SAT_RESOLUTION: {
                "tokens": sat_tokens,
                "mask": sat_mask,
                "shape": (H, W),
            },
        }

        # ── SPOT (optional) ────────────────────────────────
        if self.has_spot:
            spot_data = self._load_spot(patch_id)
            if spot_data is not None:
                spot_data = torch.nan_to_num(spot_data, nan=0.0, posinf=0.0, neginf=0.0)
                spot_data = self._normalize_spot(spot_data)
                spot_data = torch.clamp(spot_data, -10, 10)

                _, H_hr, W_hr = spot_data.shape

                label_hr = torch.nn.functional.interpolate(
                    label.float().unsqueeze(0).unsqueeze(0),
                    size=(H_hr, W_hr),
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()

                spot_tokens = self.token_builder.build_tokens(
                    image=spot_data,
                    label=label_hr,
                    resolution=self.SPOT_RESOLUTION,
                    spectral_indices=self.spot_spectral_indices,
                    resolution_idx=self.spot_resolution_idx,
                    time_idx=self.TIME_IDX_NA,
                )
                spot_mask = torch.zeros(spot_tokens.shape[0], dtype=torch.bool)

                groups[self.SPOT_RESOLUTION] = {
                    "tokens": spot_tokens,
                    "mask": spot_mask,
                    "shape": (H_hr, W_hr),
                }

        # ── Queries ─────────────────────────────────────────
        first_spectral_idx = self.s2_spectral_indices[0]
        first_time_idx = s2_time_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.SAT_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.sat_resolution_idx,
            time_idx=first_time_idx,
        )
        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries,
            ignore_index=self.IGNORE_INDEX,
            prioritize_valid=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        # ── Reference image (first S2 frame, optionally + first S1) ─
        if self.use_s1:
            image = torch.cat([s2_data[0], s1_data[0]], dim=0)  # [13, H, W]
        else:
            image = s2_data[0]  # [10, H, W]

        return {
            "groups": groups,
            "tasks": {
                self.TASK_NAME: {
                    "queries": queries,
                    "queries_mask": queries_mask,
                },
            },
            "label": label,
            "target_resolution": self.SAT_RESOLUTION,
            "image": image,
        }

    # =========================================================================
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        """
        Viz sample for segmentation — no subsampling on queries.
        Returns all pixels as queries so the callback can reshape predictions
        back to a full [H, W] map.
        """
        patch_row = self.metadata.iloc[index]
        patch_id = patch_row["ID_PATCH"]

        # ── Load ────────────────────────────────────────────
        s2_data, s2_dates = self._load_s2(patch_id, patch_row)
        label = self._load_label(patch_id)

        # ── Clean + normalize ───────────────────────────────
        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_s1:
            s1_data, s1_dates = self._load_s1(patch_id, patch_row)
            s1_data = torch.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)
            s2_data, s1_data = self._normalize_sat(s2_data, s1_data)
            s1_data = torch.clamp(s1_data, -10, 10)
        else:
            s2_mean = self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_std  = self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1)
            s2_data = (s2_data - s2_mean) / s2_std.clamp(min=1e-6)
        s2_data = torch.clamp(s2_data, -10, 10)

        # ── Sample temporal ─────────────────────────────────
        s2_data, s2_dates, _ = self._sample_temporal(s2_data, s2_dates, self.multi_temporal)
        if self.use_s1:
            s1_data, s1_dates, _ = self._sample_temporal(s1_data, s1_dates, self.multi_temporal)

        s2_time_indices = self._dates_to_indices(s2_dates)
        if self.use_s1:
            s1_time_indices = self._dates_to_indices(s1_dates)

        # ── Build flat tokens ───────────────────────────────
        s2_tokens = self._build_temporal_tokens(
            s2_data, label, s2_time_indices,
            self.s2_spectral_indices, self.sat_resolution_idx,
        )

        if self.use_s1:
            s1_tokens = self._build_temporal_tokens(
                s1_data, label, s1_time_indices,
                self.s1_spectral_indices, self.sat_resolution_idx,
            )
            sat_tokens = torch.cat([s2_tokens, s1_tokens], dim=0)
        else:
            sat_tokens = s2_tokens

        sat_mask = torch.zeros(sat_tokens.shape[0], dtype=torch.bool)

        _, _, H, W = s2_data.shape

        groups = {
            self.SAT_RESOLUTION: {
                "tokens": sat_tokens,
                "mask": sat_mask,
                "shape": (H, W),
            },
        }

        # ── SPOT (optional) ────────────────────────────────
        if self.has_spot:
            spot_data = self._load_spot(patch_id)
            if spot_data is not None:
                spot_data = torch.nan_to_num(spot_data, nan=0.0, posinf=0.0, neginf=0.0)
                spot_data = self._normalize_spot(spot_data)
                spot_data = torch.clamp(spot_data, -10, 10)

                _, H_hr, W_hr = spot_data.shape

                label_hr = torch.nn.functional.interpolate(
                    label.float().unsqueeze(0).unsqueeze(0),
                    size=(H_hr, W_hr),
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()

                spot_tokens = self.token_builder.build_tokens(
                    image=spot_data,
                    label=label_hr,
                    resolution=self.SPOT_RESOLUTION,
                    spectral_indices=self.spot_spectral_indices,
                    resolution_idx=self.spot_resolution_idx,
                    time_idx=self.TIME_IDX_NA,
                )
                spot_mask = torch.zeros(spot_tokens.shape[0], dtype=torch.bool)

                groups[self.SPOT_RESOLUTION] = {
                    "tokens": spot_tokens,
                    "mask": spot_mask,
                    "shape": (H_hr, W_hr),
                }

        # ── All pixels as queries (no subsampling) ──────────
        first_spectral_idx = self.s2_spectral_indices[0]
        first_time_idx = s2_time_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.SAT_RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.sat_resolution_idx,
            time_idx=first_time_idx,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        # ── Reference image ─────────────────────────────────
        if self.use_s1:
            image = torch.cat([s2_data[0], s1_data[0]], dim=0)  # [13, H, W]
        else:
            image = s2_data[0]  # [10, H, W]

        return {
            "groups": groups,
            "tasks": {
                self.TASK_NAME: {
                    "queries": queries,
                    "queries_mask": queries_mask,
                },
            },
            "label": label,
            "target_resolution": self.SAT_RESOLUTION,
            "image": image,
            "image_shape": (H, W),
            "n_queries": queries.shape[0],
        }

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_temporal_tokens(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        time_indices: torch.Tensor,
        spectral_indices: torch.Tensor,
        resolution_idx: int,
    ) -> torch.Tensor:
        """
        Build flat tokens for multi-temporal data.

        Each timestamp gets its own time_idx value in column 7.
        All frames concatenated into one flat tensor.

        Args:
            data: [T, C, H, W]
            label: [H, W]
            time_indices: [T]
            spectral_indices: [C]
            resolution_idx: int

        Returns:
            tokens: [T * C * H * W, 8]
        """
        T = data.shape[0]
        frames = []

        for t in range(T):
            frame_tokens = self.token_builder.build_tokens(
                image=data[t],
                label=label,
                resolution=self.SAT_RESOLUTION,
                spectral_indices=spectral_indices,
                resolution_idx=resolution_idx,
                time_idx=time_indices[t],
            )
            frames.append(frame_tokens)

        return torch.cat(frames, dim=0)

    # =========================================================================
    # METADATA
    # =========================================================================

    def _load_metadata(self):
        if not HAS_GEOPANDAS:
            raise ImportError(
                "geopandas is required for PASTIS-HD dataset.\n"
                "Install with: pip install geopandas --break-system-packages"
            )

        metadata_path = os.path.join(self.root_path, "metadata.geojson")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.geojson not found at {metadata_path}\n"
                f"Expected PASTIS-HD structure:\n"
                f"  {self.root_path}/\n"
                f"    ├── metadata.geojson\n"
                f"    ├── DATA_S2/\n"
                f"    ├── DATA_S1A/\n"
                f"    ├── DATA_SPOT/ (optional)\n"
                f"    └── ANNOTATIONS/"
            )

        print(f"[PASTIS-HD] Loading metadata from {metadata_path}")
        self.metadata = gpd.read_file(metadata_path)

        fold_mapping = {
            "train": [1, 2, 3],
            "val": [4],
            "test": [5],
        }

        mapped_split = self.split_mapping.get(self.split, self.split)
        folds = fold_mapping.get(mapped_split)
        if folds is None:
            raise ValueError(f"Invalid split '{self.split}'. Must be train/validation/test")

        self.metadata = pd.concat(
            [self.metadata[self.metadata["Fold"] == f] for f in folds]
        )
        self.metadata = self.metadata.reset_index(drop=True)

        self.patch_ids = self.metadata["ID_PATCH"].tolist()

    def _setup_band_indices(self):
        # S2
        self.s2_spectral_indices = []
        for band_name in sorted(self.S2_BANDS_INFO.keys(), key=lambda b: self.S2_BANDS_INFO[b]["idx"]):
            info = self.S2_BANDS_INFO[band_name]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"S2 band {band_name} key={key} not in lookup table.\n"
                    f"Available keys: {list(self.look_up.table_wave.keys())}"
                )
            self.s2_spectral_indices.append(self.look_up.table_wave[key])
        self.s2_spectral_indices = torch.tensor(self.s2_spectral_indices, dtype=torch.long)

        # S1 (only if enabled)
        if self.use_s1:
            self.s1_spectral_indices = []
            for band_name in sorted(self.S1_BANDS_INFO.keys(), key=lambda b: self.S1_BANDS_INFO[b]["idx"]):
                info = self.S1_BANDS_INFO[band_name]
                key = (info["bandwidth"], info["central_wavelength"])
                if key not in self.look_up.table_wave:
                    raise KeyError(
                        f"S1 band {band_name} key={key} not in lookup table.\n"
                        f"Available keys: {list(self.look_up.table_wave.keys())}"
                    )
                self.s1_spectral_indices.append(self.look_up.table_wave[key])
            self.s1_spectral_indices = torch.tensor(self.s1_spectral_indices, dtype=torch.long)

        # SPOT (only if enabled and available)
        if self.has_spot:
            self.spot_spectral_indices = []
            for band_name in sorted(self.SPOT_BANDS_INFO.keys(), key=lambda b: self.SPOT_BANDS_INFO[b]["idx"]):
                info = self.SPOT_BANDS_INFO[band_name]
                key = (info["bandwidth"], info["central_wavelength"])
                if key not in self.look_up.table_wave:
                    print(f"[PASTIS-HD] WARNING: SPOT band {band_name} key={key} "
                          f"not in lookup table. Disabling SPOT.")
                    self.has_spot = False
                    return
                self.spot_spectral_indices.append(self.look_up.table_wave[key])
            self.spot_spectral_indices = torch.tensor(self.spot_spectral_indices, dtype=torch.long)

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_s2(self, patch_id: int, patch_row) -> Tuple[torch.Tensor, List]:
        s2_path = os.path.join(self.root_path, "DATA_S2", f"S2_{patch_id}.npy")
        if not os.path.exists(s2_path):
            raise FileNotFoundError(f"S2 data not found: {s2_path}")

        s2_data = np.load(s2_path).astype(np.float32)
        s2_data = torch.from_numpy(s2_data)

        dates_key = "dates-S2"
        if dates_key in patch_row:
            dates_dict = patch_row[dates_key]
            if isinstance(dates_dict, str):
                dates_dict = json.loads(dates_dict)
            dates = pd.DataFrame.from_dict(dates_dict, orient="index")[0].tolist()
        else:
            T = s2_data.shape[0]
            dates = list(range(T))

        if len(dates) > self.max_temporal_samples:
            indices = torch.linspace(0, len(dates) - 1, self.max_temporal_samples, dtype=torch.long)
            s2_data = s2_data[indices]
            dates = [dates[i] for i in indices]

        return s2_data, dates

    def _load_s1(self, patch_id: int, patch_row) -> Tuple[torch.Tensor, List]:
        s1_path = os.path.join(self.root_path, "DATA_S1A", f"S1A_{patch_id}.npy")
        if not os.path.exists(s1_path):
            raise FileNotFoundError(f"S1 data not found: {s1_path}")

        s1_data = np.load(s1_path).astype(np.float32)
        s1_data = torch.from_numpy(s1_data)
        # All 3 channels: VV, VH, VV-VH

        dates_key = "dates-S1A"
        if dates_key in patch_row:
            dates_dict = patch_row[dates_key]
            if isinstance(dates_dict, str):
                dates_dict = json.loads(dates_dict)
            dates = pd.DataFrame.from_dict(dates_dict, orient="index")[0].tolist()
        else:
            T = s1_data.shape[0]
            dates = list(range(T))

        if len(dates) > self.max_temporal_samples:
            indices = torch.linspace(0, len(dates) - 1, self.max_temporal_samples, dtype=torch.long)
            s1_data = s1_data[indices]
            dates = [dates[i] for i in indices]

        return s1_data, dates

    def _load_spot(self, patch_id: int) -> torch.Tensor:
        spot_path = os.path.join(
            self.spot_dir, f"SPOT6_RVB_1M00_2019_{patch_id}.tif"
        )
        if not os.path.exists(spot_path):
            return None

        try:
            with rasterio.open(spot_path) as src:
                spot_data = src.read().astype(np.float32)
            return torch.from_numpy(spot_data)
        except Exception as e:
            print(f"[PASTIS-HD] Warning: Could not read SPOT {spot_path}: {e}")
            return None

    def _load_label(self, patch_id: int) -> torch.Tensor:
        label_path = os.path.join(self.root_path, "ANNOTATIONS", f"TARGET_{patch_id}.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")

        label = np.load(label_path)[0].astype(np.int64)
        label = torch.from_numpy(label)

        label[label < 0] = self.IGNORE_INDEX
        label[label == 19] = self.IGNORE_INDEX
        label[label >= self.NUM_CLASSES] = self.IGNORE_INDEX

        return label

    # =========================================================================
    # TEMPORAL SAMPLING
    # =========================================================================

    def _sample_temporal(
        self, data: torch.Tensor, dates: List, n_samples: int
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        T = data.shape[0]
        if T <= n_samples:
            return data, dates, torch.arange(T)

        if self.temporal_last:
            # Take the last n_samples timesteps
            indices = torch.arange(T - n_samples, T)
        else:
            # Uniformly spaced sampling across the full sequence
            indices = torch.linspace(0, T - 1, n_samples, dtype=torch.long)

        sampled_data = data[indices]
        sampled_dates = [dates[i] for i in indices]

        return sampled_data, sampled_dates, indices

    def _dates_to_indices(self, dates):
        time_indices = []
        for date in dates:
            if isinstance(date, str):
                date = int(date)
            year = date // 10000
            month = (date % 10000) // 100
            day = date % 100
            doy = datetime(year, month, day).timetuple().tm_yday
            idx = self.look_up.get_or_register_time_idx(doy)
            time_indices.append(idx)
        return torch.tensor(time_indices, dtype=torch.long)

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[PASTIS-HD] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            # Check for stale stats (e.g., old 2-band S1 vs new 3-band)
            if "s1_mean" in stats and stats["s1_mean"].shape[0] != self.NUM_S1_BANDS:
                print(f"[PASTIS-HD] WARNING: Stale normalization stats "
                      f"(S1: {stats['s1_mean'].shape[0]} bands, expected {self.NUM_S1_BANDS}). "
                      f"Recomputing...")
                os.remove(norm_file)
            else:
                self._print_norm_stats(stats)
                return stats

        if self.split != "train":
            print(f"[PASTIS-HD] WARNING: No normalization file at {norm_file}")
            stats = {
                "s2_mean": torch.zeros(self.NUM_S2_BANDS),
                "s2_std": torch.ones(self.NUM_S2_BANDS),
                "s1_mean": torch.zeros(self.NUM_S1_BANDS),
                "s1_std": torch.ones(self.NUM_S1_BANDS),
            }
            if self.has_spot:
                stats["spot_mean"] = torch.zeros(self.NUM_SPOT_BANDS)
                stats["spot_std"] = torch.ones(self.NUM_SPOT_BANDS)
            return stats

        print(f"[PASTIS-HD] Computing normalization from {len(self.patch_ids)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[PASTIS-HD] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s2_sq  = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s2_n   = torch.zeros(self.NUM_S2_BANDS, dtype=torch.float64)
        s1_sum = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        s1_sq  = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        s1_n   = torch.zeros(self.NUM_S1_BANDS, dtype=torch.float64)
        spot_sum = torch.zeros(self.NUM_SPOT_BANDS, dtype=torch.float64)
        spot_sq  = torch.zeros(self.NUM_SPOT_BANDS, dtype=torch.float64)
        spot_n   = torch.zeros(self.NUM_SPOT_BANDS, dtype=torch.float64)

        for idx in tqdm(range(len(self.patch_ids)), desc="Computing normalization"):
            patch_id = self.patch_ids[idx]

            try:
                s2_path = os.path.join(self.root_path, "DATA_S2", f"S2_{patch_id}.npy")
                s2 = np.load(s2_path).astype(np.float64)
                s2 = np.nan_to_num(s2)
                for c in range(self.NUM_S2_BANDS):
                    valid = s2[:, c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        s2_sum[c] += valid.sum()
                        s2_sq[c]  += (valid ** 2).sum()
                        s2_n[c]   += len(valid)
            except Exception:
                continue

            try:
                s1_path = os.path.join(self.root_path, "DATA_S1A", f"S1A_{patch_id}.npy")
                s1 = np.load(s1_path).astype(np.float64)
                s1 = np.nan_to_num(s1)
                for c in range(self.NUM_S1_BANDS):
                    valid = s1[:, c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        s1_sum[c] += valid.sum()
                        s1_sq[c]  += (valid ** 2).sum()
                        s1_n[c]   += len(valid)
            except Exception:
                continue

            if self.has_spot:
                try:
                    spot_path = os.path.join(
                        self.spot_dir, f"SPOT6_RVB_1M00_2019_{patch_id}.tif"
                    )
                    if os.path.exists(spot_path):
                        with rasterio.open(spot_path) as src:
                            spot = src.read().astype(np.float64)
                        spot = np.nan_to_num(spot)
                        for c in range(self.NUM_SPOT_BANDS):
                            valid = spot[c].flatten()
                            valid = valid[valid != 0]
                            if len(valid):
                                spot_sum[c] += valid.sum()
                                spot_sq[c]  += (valid ** 2).sum()
                                spot_n[c]   += len(valid)
                except Exception:
                    continue

        s2_mean = (s2_sum / s2_n.clamp(min=1)).float()
        s2_std  = ((s2_sq / s2_n.clamp(min=1) - s2_mean.double() ** 2).sqrt()).float()
        s1_mean = (s1_sum / s1_n.clamp(min=1)).float()
        s1_std  = ((s1_sq / s1_n.clamp(min=1) - s1_mean.double() ** 2).sqrt()).float()

        stats = {
            "s2_mean": s2_mean, "s2_std": s2_std,
            "s1_mean": s1_mean, "s1_std": s1_std,
        }

        if self.has_spot:
            spot_mean = (spot_sum / spot_n.clamp(min=1)).float()
            spot_std  = ((spot_sq / spot_n.clamp(min=1) - spot_mean.double() ** 2).sqrt()).float()
            stats["spot_mean"] = spot_mean
            stats["spot_std"] = spot_std

        return stats

    def _print_norm_stats(self, stats):
        print(f"[PASTIS-HD] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[PASTIS-HD] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[PASTIS-HD] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[PASTIS-HD] S1 std:  {stats['s1_std'].numpy()}")
        if "spot_mean" in stats:
            print(f"[PASTIS-HD] SPOT mean: {stats['spot_mean'].numpy()}")
            print(f"[PASTIS-HD] SPOT std:  {stats['spot_std'].numpy()}")

    def _normalize_sat(self, s2, s1):
        s2_mean = self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)
        s2_std  = self.norm_stats["s2_std"].view(1, self.NUM_S2_BANDS, 1, 1)
        s1_mean = self.norm_stats["s1_mean"].view(1, self.NUM_S1_BANDS, 1, 1)
        s1_std  = self.norm_stats["s1_std"].view(1, self.NUM_S1_BANDS, 1, 1)
        return (s2 - s2_mean) / s2_std.clamp(min=1e-6), (s1 - s1_mean) / s1_std.clamp(min=1e-6)

    def _normalize_spot(self, spot):
        if "spot_mean" not in self.norm_stats:
            return spot
        spot_mean = self.norm_stats["spot_mean"].view(self.NUM_SPOT_BANDS, 1, 1)
        spot_std  = self.norm_stats["spot_std"].view(self.NUM_SPOT_BANDS, 1, 1)
        return (spot - spot_mean) / spot_std.clamp(min=1e-6)