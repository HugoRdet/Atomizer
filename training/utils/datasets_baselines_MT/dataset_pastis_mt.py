"""
PASTIS-HD dataset for multi-task baseline training.

Sentinel-2 only. The single-task PASTIS dataset supports S1A as well, but
S2 and S1A are acquired at independent dates, so they cannot be combined
into a single 15-channel temporal stack without first resampling onto a
shared time grid. For multi-task we keep PASTIS S2-only and zero-fill the
canonical SAR slots (same convention as BurnScars).

Differences from the single-task PastisBaselineDataset:
    - S2-only (use_s1 dropped).
    - Fixed T=6 frames at the dataset level (no variable-T padding in the
      collate). Frames are sub-sampled uniformly along the native time axis
      (or last-N with --temporal_last). Samples with fewer native frames
      than `num_frames` are zero-padded along the time axis with date 0.
    - Canonicalizes 10 PASTIS S2 bands to the 13-channel S2 grid (B02-B8A
      and B11-B12 are exact matches; B09 and B10 are linearly interpolated
      from B8A and B11; B01 is OOR -> zero). Then concatenates 2 zero SAR
      channels -> 15-channel canonical per frame.
    - Pads spatial extent 128x128 -> 512x512 (top-left, all frames).
    - Returns the unified multi-task output shape with the temporal axis
      preserved; the trainer model's TimeMerge adapter collapses T at
      forward time.

Output format:
    {
        "image": {"input": [6, 15, 512, 512]},   # float32, normalized
        "dates": {"input": [6]},                  # long, day-of-year
        "target": [512, 512],                     # long; {0..18, 255}
        "valid_mask": [512, 512],                 # uint8; 1 = real pixel
        "original_size": [2],                     # long; (128, 128)
        "metadata": {...},
    }

Splits:
    train -> folds {1, 2, 3}
    val   -> fold  4
    test  -> fold  5

Class remapping:
    Source labels are in {0..19, plus negatives / out-of-range}. Following
    the single-task convention, source class 19 ("Shrub/Forest") and any
    out-of-range value are remapped to IGNORE_INDEX (255). The 19
    remaining classes (0..18) feed into the model unchanged.

Reference: https://github.com/VSainteuf/pastis-benchmark
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import geopandas as gpd
    import pandas as pd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from .multitask_utils import (
    CANONICAL_SIZE,
    IGNORE_INDEX,
    apply_interpolation_matrix,
    build_canonical_image,
    build_interpolation_matrix,
    pad_to_canonical,
)


class PastisMTDataset(Dataset):
    """PASTIS-HD dataset for multi-task baselines (S2 only)."""

    # PASTIS S2 bands: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12.
    S2_WAVELENGTHS_NM = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
    NUM_S2_BANDS = 10

    # Effective classes after remapping. Source class 19 mapped to IGNORE.
    NUM_CLASSES = 19            # active classes 0..18
    SOURCE_NUM_CLASSES = 20     # >=20 also mapped to IGNORE

    SPLIT_FOLDS = {
        "train":      [1, 2, 3],
        "validation": [4],
        "test":       [5],
    }

    # Cap on initial frames before further temporal sampling — same as the
    # single-task default. Keeps memory bounded for patches with very long
    # native time series.
    DEFAULT_MAX_TEMPORAL_SAMPLES = 50

    def __init__(
        self,
        root_path: str = "./data/PASTIS-HD",
        mode: str = "train",
        num_frames: int = 6,
        temporal_last: bool = False,
        max_temporal_samples: int = None,
        augment: bool = True,
    ):
        super().__init__()
        if not HAS_GEOPANDAS:
            raise ImportError(
                "geopandas is required for PASTIS metadata: "
                "`pip install geopandas`"
            )
        assert mode in self.SPLIT_FOLDS, f"Unknown split: {mode}"

        self.root_path = root_path
        self.split = mode
        self.num_frames = num_frames
        self.temporal_last = temporal_last
        self.max_temporal_samples = (
            max_temporal_samples
            if max_temporal_samples is not None
            else self.DEFAULT_MAX_TEMPORAL_SAMPLES
        )
        self.augment = augment and (mode == "train")

        # ── Load metadata ────────────────────────────────────
        self._load_metadata()

        # ── Native normalization stats ───────────────────────
        self.norm_stats = self._load_normalization()

        # ── Spectral interpolation matrix [13, 10] ───────────
        self.interp_matrix = build_interpolation_matrix(self.S2_WAVELENGTHS_NM)

        print(f"[PASTIS-MT] split={mode}, samples={len(self.patch_ids)}")
        sampling = "last" if self.temporal_last else "uniform"
        print(f"[PASTIS-MT] T={num_frames} ({sampling}) -> canonical 15ch")
        print(f"[PASTIS-MT] D4 augment: {'ON' if self.augment else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # METADATA / SPLITS
    # ─────────────────────────────────────────────────────────────────────

    def _load_metadata(self):
        metadata_path = os.path.join(self.root_path, "metadata.geojson")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"[PASTIS-MT] metadata.geojson not found at {metadata_path}"
            )

        self.metadata = gpd.read_file(metadata_path)
        folds = self.SPLIT_FOLDS[self.split]
        self.metadata = pd.concat(
            [self.metadata[self.metadata["Fold"] == f] for f in folds]
        ).reset_index(drop=True)
        self.patch_ids = self.metadata["ID_PATCH"].tolist()

    # ─────────────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────

    def _load_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")
        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)
            print(f"[PASTIS-MT] Loaded normalization stats from {norm_file}")
            return stats
        print(f"[PASTIS-MT] WARNING: No normalization file at {norm_file}, "
              f"using zero-mean unit-std.")
        return {
            "s2_mean": torch.zeros(self.NUM_S2_BANDS),
            "s2_std":  torch.ones(self.NUM_S2_BANDS),
        }

    def _normalize_s2(self, data: torch.Tensor) -> torch.Tensor:
        """Per-band z-score on [T, C, H, W] tensors."""
        mean = self.norm_stats["s2_mean"].view(1, self.NUM_S2_BANDS, 1, 1)
        std  = self.norm_stats["s2_std"].clamp(min=1e-6) \
                                        .view(1, self.NUM_S2_BANDS, 1, 1)
        return (data - mean) / std

    # ─────────────────────────────────────────────────────────────────────
    # FILE LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_s2(self, patch_id: int, patch_row):
        """Load S2 [T, 10, H, W] and dates list. Cap to max_temporal_samples."""
        path = os.path.join(self.root_path, "DATA_S2", f"S2_{patch_id}.npy")
        data = torch.from_numpy(np.load(path).astype(np.float32))

        if "dates-S2" in patch_row:
            dates_dict = patch_row["dates-S2"]
            if isinstance(dates_dict, str):
                dates_dict = json.loads(dates_dict)
            dates = pd.DataFrame.from_dict(dates_dict, orient="index")[0].tolist()
        else:
            dates = list(range(data.shape[0]))

        if len(dates) > self.max_temporal_samples:
            indices = torch.linspace(
                0, len(dates) - 1, self.max_temporal_samples, dtype=torch.long,
            )
            data = data[indices]
            dates = [dates[i] for i in indices]

        return data, dates

    def _load_label(self, patch_id: int) -> torch.Tensor:
        path = os.path.join(self.root_path, "ANNOTATIONS", f"TARGET_{patch_id}.npy")
        label = torch.from_numpy(np.load(path)[0].astype(np.int64))
        # Remap: anything outside {0..18} -> IGNORE_INDEX (255).
        label[label < 0] = IGNORE_INDEX
        label[label == 19] = IGNORE_INDEX
        label[label >= self.SOURCE_NUM_CLASSES] = IGNORE_INDEX
        return label

    # ─────────────────────────────────────────────────────────────────────
    # TEMPORAL SAMPLING
    # ─────────────────────────────────────────────────────────────────────

    def _sample_temporal(self, data, dates, n):
        """Sub-sample to n frames (uniform along T or last-n)."""
        T = data.shape[0]
        if T <= n:
            return data, dates
        if self.temporal_last:
            indices = torch.arange(T - n, T)
        else:
            indices = torch.linspace(0, T - 1, n, dtype=torch.long)
        return data[indices], [dates[i] for i in indices]

    @staticmethod
    def _pad_temporal(data, dates, n):
        """Zero-pad along T to reach n frames. Pad dates with 0."""
        T = data.shape[0]
        if T >= n:
            return data, dates
        pad_T = n - T
        pad_data = torch.zeros(pad_T, *data.shape[1:], dtype=data.dtype)
        data = torch.cat([data, pad_data], dim=0)
        dates = list(dates) + [0] * pad_T
        return data, dates

    @staticmethod
    def _dates_to_doy(dates) -> torch.Tensor:
        """Convert YYYYMMDD ints (or DOY ints) to day-of-year [1..366]."""
        doys = []
        for d in dates:
            if isinstance(d, str):
                d = int(d)
            if isinstance(d, (int, np.integer)) and d > 1000:
                year, month, day = d // 10000, (d % 10000) // 100, d % 100
                try:
                    doy = datetime(year, month, day).timetuple().tm_yday
                except ValueError:
                    doy = 0
            else:
                doy = int(d)
            doys.append(doy)
        return torch.tensor(doys, dtype=torch.long)

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        """D4 group on [T, C, H, W] image and [H, W] label (last 2 dims = H, W)."""
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=(-1,))
            label = torch.flip(label, dims=(-1,))
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=(-2, -1))
            label = torch.rot90(label, k, dims=(-2, -1))
        return image, label

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_row = self.metadata.iloc[index]
        patch_id = int(patch_row["ID_PATCH"])

        # ── Load ──────────────────────────────────────────────
        s2_data, s2_dates = self._load_s2(patch_id, patch_row)   # [T, 10, H, W]
        label = self._load_label(patch_id)                        # [H, W]

        # ── Clean ─────────────────────────────────────────────
        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Sub-sample to fixed T, zero-pad if shorter ────────
        s2_data, s2_dates = self._sample_temporal(
            s2_data, s2_dates, self.num_frames,
        )
        n_native = s2_data.shape[0]
        s2_data, s2_dates = self._pad_temporal(
            s2_data, s2_dates, self.num_frames,
        )

        # ── Native normalization (per-band z-score) ──────────
        s2_data = self._normalize_s2(s2_data)
        s2_data = torch.clamp(s2_data, -10, 10)

        # ── D4 augmentation (consistent across T) ────────────
        if self.augment:
            s2_data, label = self._d4_augment(s2_data, label)

        # ── Spectral canonicalization: per-frame [10] -> [13] ──
        optical_canonical = apply_interpolation_matrix(
            s2_data, self.interp_matrix,
        )                                                         # [T, 13, H, W]

        # ── Concatenate with zero SAR -> [T, 15, H, W] ───────
        canonical = build_canonical_image(optical_canonical, sar=None)

        # ── Spatial padding to 512x512 (per-frame) ───────────
        canonical, label, valid_mask, original_size = pad_to_canonical(
            canonical, label, size=CANONICAL_SIZE,
        )

        # ── Day-of-year vector [T] ───────────────────────────
        doy = self._dates_to_doy(s2_dates)

        return {
            "image": {"input": canonical},          # [T=6, 15, 512, 512]
            "dates": {"input": doy},                # [T=6]
            "target": label,                        # [512, 512]
            "valid_mask": valid_mask,               # [512, 512]
            "original_size": original_size,         # [2] -> (128, 128)
            "metadata": {
                "patch_id": patch_id,
                "n_frames_native": n_native,        # before temporal padding
                "native_bands": [
                    "B02", "B03", "B04", "B05", "B06", "B07",
                    "B08", "B8A", "B11", "B12",
                ],
            },
        }