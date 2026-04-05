"""
PASTIS-HD Baseline Dataset
============================

Fixed-format dataset for baseline models (UNet, ViT+UperNet, U-TAE).
Returns {"image": {sensor: tensor}, "target": [H, W], "metadata": {...}}

No tokenization, no metadata encoding — just normalized imagery.

Temporal handling (controlled via temporal_mode):
  - "stack":    [C*T, H, W]  — all frames concatenated as channels (UNet, ViT)
  - "sequence": [T, C, H, W] — temporal sequence preserved (U-TAE, TSViT)

Normalization:
  Per-band z-score using precomputed stats (normalization_stats.pt).
  Each band independently standardized, then clamped to [-10, 10].

Directory structure:
./data/PASTIS-HD/
├── metadata.geojson
├── DATA_S2/S2_{patch_id}.npy        # [T, 10, H, W]
├── DATA_S1A/S1A_{patch_id}.npy      # [T, 3, H, W]
├── normalization_stats.pt
└── ANNOTATIONS/TARGET_{patch_id}.npy # [1, H, W]
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 20
IGNORE_INDEX = 255

CLASS_NAMES = {
    0: "Background",      1: "Meadow",          2: "Soft Winter Wheat",
    3: "Corn",            4: "Winter Barley",    5: "Winter Rapeseed",
    6: "Spring Barley",   7: "Sunflower",        8: "Grapevine",
    9: "Beet",           10: "Soy",             11: "Sorghum",
    12: "Flax",          13: "Protein Crops",   14: "Other Cereals",
    15: "Fruits/Veg",    16: "Other Crops",     17: "Grassland",
    18: "Shrub/Forest",
}

NUM_S2_BANDS = 10
NUM_S1_BANDS = 3         # VV, VH, VV-VH

S2_WAVELENGTHS = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
S2_BANDWIDTHS = [65, 35, 30, 15, 15, 20, 115, 20, 90, 180]
S1_WAVELENGTHS = [-1, -2, -3]  # SAR placeholders
S1_BANDWIDTHS = [-1, -2, -3]


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

def augment_d4(image: torch.Tensor, label: torch.Tensor):
    """Random D4 symmetry group (4 rotations × 2 flips).
    Works for both [C, H, W] and [T, C, H, W]."""
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, dims=(-2, -1))
        label = torch.rot90(label, k, dims=(-2, -1))
    if random.random() > 0.5:
        image = torch.flip(image, dims=(-1,))
        label = torch.flip(label, dims=(-1,))
    return image, label


# ═══════════════════════════════════════════════════════════════════════
# BASELINE DATASET
# ═══════════════════════════════════════════════════════════════════════

class PastisBaselineDataset(Dataset):
    """
    PASTIS-HD dataset for baseline segmentation models.

    Returns:
        {
            "image": {
                "s2": [C*T, H, W] or [T, C, H, W],
                "s1": [C*T, H, W] or [T, C, H, W],  (if use_s1)
            },
            "target": [H, W],
            "metadata": {
                "sensors": list[str],
                "n_s2_bands": int,
                "n_s1_bands": int,
                "n_frames": int,
                "temporal_mode": str,
                "patch_id": int,
            },
        }

    Temporal modes:
        "stack":    concatenate all frames as channels → [C*T, H, W]
        "sequence": keep temporal dim → [T, C, H, W]
    """

    def __init__(
        self,
        root_path: str = "./data/PASTIS-HD",
        mode: str = "train",
        use_s1: bool = True,
        multi_temporal: int = 10,
        temporal_last: bool = False,
        temporal_mode: str = "stack",
        max_temporal_samples: int = 50,
        augment: bool = True,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.use_s1 = use_s1
        self.multi_temporal = multi_temporal
        self.temporal_last = temporal_last
        self.temporal_mode = temporal_mode
        self.max_temporal_samples = max_temporal_samples
        self.augment = augment and (mode == "train")

        self.split_mapping = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }

        # ── Load metadata ───────────────────────────────────────────
        self._load_metadata()

        # ── Normalization stats ─────────────────────────────────────
        self.norm_stats = self._load_normalization()

        # ── Summary ─────────────────────────────────────────────────
        sensors = ["S2"] + (["S1"] if self.use_s1 else [])
        temporal_str = f"{'last' if self.temporal_last else 'uniform'}"
        print(f"[PASTIS-BL] {len(self.patch_ids)} patches, split='{self.split}'")
        print(f"[PASTIS-BL] Sensors: {'+'.join(sensors)}")
        print(f"[PASTIS-BL] Temporal: {self.multi_temporal} frames ({temporal_str}), "
              f"mode={self.temporal_mode}")

    # ═════════════════════════════════════════════════════════════════
    # METADATA
    # ═════════════════════════════════════════════════════════════════

    def _load_metadata(self):
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas required: pip install geopandas")

        metadata_path = os.path.join(self.root_path, "metadata.geojson")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.geojson not found at {metadata_path}")

        self.metadata = gpd.read_file(metadata_path)

        fold_mapping = {
            "train": [1, 2, 3],
            "val": [4],
            "test": [5],
        }

        mapped_split = self.split_mapping.get(self.split, self.split)
        folds = fold_mapping.get(mapped_split)
        if folds is None:
            raise ValueError(f"Invalid split '{self.split}'")

        self.metadata = pd.concat(
            [self.metadata[self.metadata["Fold"] == f] for f in folds]
        ).reset_index(drop=True)

        self.patch_ids = self.metadata["ID_PATCH"].tolist()

    # ═════════════════════════════════════════════════════════════════
    # NORMALIZATION
    # ═════════════════════════════════════════════════════════════════

    def _load_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            stats = torch.load(norm_file, weights_only=True)
            print(f"[PASTIS-BL] Loaded normalization stats from {norm_file}")
            return stats

        print(f"[PASTIS-BL] WARNING: No normalization file, using identity")
        stats = {
            "s2_mean": torch.zeros(NUM_S2_BANDS),
            "s2_std": torch.ones(NUM_S2_BANDS),
            "s1_mean": torch.zeros(NUM_S1_BANDS),
            "s1_std": torch.ones(NUM_S1_BANDS),
        }
        return stats

    def _normalize_s2(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize S2: per-band z-score. data is [T, C, H, W] or [C, H, W]."""
        mean = self.norm_stats["s2_mean"]
        std = self.norm_stats["s2_std"].clamp(min=1e-6)

        if data.dim() == 4:
            mean = mean.view(1, NUM_S2_BANDS, 1, 1)
            std = std.view(1, NUM_S2_BANDS, 1, 1)
        else:
            mean = mean.view(NUM_S2_BANDS, 1, 1)
            std = std.view(NUM_S2_BANDS, 1, 1)

        return (data - mean) / std

    def _normalize_s1(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize S1: per-band z-score. data is [T, C, H, W] or [C, H, W]."""
        mean = self.norm_stats["s1_mean"]
        std = self.norm_stats["s1_std"].clamp(min=1e-6)

        if data.dim() == 4:
            mean = mean.view(1, NUM_S1_BANDS, 1, 1)
            std = std.view(1, NUM_S1_BANDS, 1, 1)
        else:
            mean = mean.view(NUM_S1_BANDS, 1, 1)
            std = std.view(NUM_S1_BANDS, 1, 1)

        return (data - mean) / std

    # ═════════════════════════════════════════════════════════════════
    # FILE LOADING
    # ═════════════════════════════════════════════════════════════════

    def _load_s2(self, patch_id: int, patch_row) -> Tuple[torch.Tensor, List]:
        path = os.path.join(self.root_path, "DATA_S2", f"S2_{patch_id}.npy")
        data = torch.from_numpy(np.load(path).astype(np.float32))

        dates_key = "dates-S2"
        if dates_key in patch_row:
            dates_dict = patch_row[dates_key]
            if isinstance(dates_dict, str):
                dates_dict = json.loads(dates_dict)
            dates = pd.DataFrame.from_dict(dates_dict, orient="index")[0].tolist()
        else:
            dates = list(range(data.shape[0]))

        if len(dates) > self.max_temporal_samples:
            indices = torch.linspace(0, len(dates) - 1,
                                     self.max_temporal_samples, dtype=torch.long)
            data = data[indices]
            dates = [dates[i] for i in indices]

        return data, dates

    def _load_s1(self, patch_id: int, patch_row) -> Tuple[torch.Tensor, List]:
        path = os.path.join(self.root_path, "DATA_S1A", f"S1A_{patch_id}.npy")
        data = torch.from_numpy(np.load(path).astype(np.float32))
        # All 3 channels: VV, VH, VV-VH

        dates_key = "dates-S1A"
        if dates_key in patch_row:
            dates_dict = patch_row[dates_key]
            if isinstance(dates_dict, str):
                dates_dict = json.loads(dates_dict)
            dates = pd.DataFrame.from_dict(dates_dict, orient="index")[0].tolist()
        else:
            dates = list(range(data.shape[0]))

        if len(dates) > self.max_temporal_samples:
            indices = torch.linspace(0, len(dates) - 1,
                                     self.max_temporal_samples, dtype=torch.long)
            data = data[indices]
            dates = [dates[i] for i in indices]

        return data, dates

    def _load_label(self, patch_id: int) -> torch.Tensor:
        path = os.path.join(self.root_path, "ANNOTATIONS", f"TARGET_{patch_id}.npy")
        label = torch.from_numpy(np.load(path)[0].astype(np.int64))
        label[label < 0] = IGNORE_INDEX
        label[label == 19] = IGNORE_INDEX
        label[label >= NUM_CLASSES] = IGNORE_INDEX
        return label

    # ═════════════════════════════════════════════════════════════════
    # TEMPORAL SAMPLING
    # ═════════════════════════════════════════════════════════════════

    def _sample_temporal(self, data, dates, n_samples):
        T = data.shape[0]
        if T <= n_samples:
            return data, dates

        if self.temporal_last:
            indices = torch.arange(T - n_samples, T)
        else:
            indices = torch.linspace(0, T - 1, n_samples, dtype=torch.long)

        return data[indices], [dates[i] for i in indices]

    # ═════════════════════════════════════════════════════════════════
    # DATE CONVERSION
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def _dates_to_doy(dates: List) -> torch.Tensor:
        """Convert YYYYMMDD int dates to day-of-year [0..365]. Returns [T]."""
        doys = []
        for date in dates:
            if isinstance(date, str):
                date = int(date)
            if isinstance(date, (int, np.integer)) and date > 1000:
                year = date // 10000
                month = (date % 10000) // 100
                day = date % 100
                try:
                    doy = datetime(year, month, day).timetuple().tm_yday
                except ValueError:
                    doy = 0
            else:
                doy = int(date)
            doys.append(doy)
        return torch.tensor(doys, dtype=torch.long)

    # ═════════════════════════════════════════════════════════════════
    # FORMAT OUTPUT
    # ═════════════════════════════════════════════════════════════════

    def _format_temporal(self, data: torch.Tensor) -> torch.Tensor:
        """
        Format temporal data according to temporal_mode.
        Input: [T, C, H, W]
        Output:
            "stack"    → [C*T, H, W]
            "sequence" → [T, C, H, W]
        """
        if self.temporal_mode == "stack":
            T, C, H, W = data.shape
            return data.reshape(T * C, H, W)
        else:
            return data

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> dict:
        patch_row = self.metadata.iloc[index]
        patch_id = patch_row["ID_PATCH"]

        # ── Load & clean ────────────────────────────────────────────
        try:
            s2_data, s2_dates = self._load_s2(patch_id, patch_row)
            label = self._load_label(patch_id)
        except Exception as e:
            print(f"[PASTIS-BL] Error loading patch {patch_id}: {e}")
            H = W = 128
            n_frames = self.multi_temporal
            if self.temporal_mode == "stack":
                dummy = torch.zeros(NUM_S2_BANDS * n_frames, H, W)
            else:
                dummy = torch.zeros(n_frames, NUM_S2_BANDS, H, W)
            image_dict = {"s2": dummy}
            dates_dict = {"s2": torch.zeros(n_frames, dtype=torch.long)}
            if self.use_s1:
                if self.temporal_mode == "stack":
                    image_dict["s1"] = torch.zeros(NUM_S1_BANDS * n_frames, H, W)
                else:
                    image_dict["s1"] = torch.zeros(n_frames, NUM_S1_BANDS, H, W)
                dates_dict["s1"] = torch.zeros(n_frames, dtype=torch.long)
            return {
                "image": image_dict,
                "dates": dates_dict,
                "target": torch.full((H, W), IGNORE_INDEX, dtype=torch.long),
                "metadata": {"sensors": [], "patch_id": patch_id},
            }

        s2_data = torch.nan_to_num(s2_data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Load S1 (conditional) ───────────────────────────────────
        if self.use_s1:
            s1_data, s1_dates = self._load_s1(patch_id, patch_row)
            s1_data = torch.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normalize ───────────────────────────────────────────────
        s2_data = self._normalize_s2(s2_data)
        s2_data = torch.clamp(s2_data, -10, 10)

        if self.use_s1:
            s1_data = self._normalize_s1(s1_data)
            s1_data = torch.clamp(s1_data, -10, 10)

        # ── Temporal sampling ───────────────────────────────────────
        s2_data, s2_dates = self._sample_temporal(
            s2_data, s2_dates, self.multi_temporal)

        if self.use_s1:
            s1_data, s1_dates = self._sample_temporal(
                s1_data, s1_dates, self.multi_temporal)

        # ── D4 augmentation (applied per-frame consistently) ────────
        if self.augment:
            k = random.randint(0, 3)
            do_flip = random.random() > 0.5

            if k > 0:
                s2_data = torch.rot90(s2_data, k, dims=(-2, -1))
                label = torch.rot90(label, k, dims=(-2, -1))
                if self.use_s1:
                    s1_data = torch.rot90(s1_data, k, dims=(-2, -1))

            if do_flip:
                s2_data = torch.flip(s2_data, dims=(-1,))
                label = torch.flip(label, dims=(-1,))
                if self.use_s1:
                    s1_data = torch.flip(s1_data, dims=(-1,))

        # ── Format temporal dimension ───────────────────────────────
        s2_out = self._format_temporal(s2_data)

        n_frames = s2_data.shape[0]
        sensors = ["s2"]
        image_dict = {"s2": s2_out}
        dates_dict = {"s2": self._dates_to_doy(s2_dates)}

        if self.use_s1:
            s1_out = self._format_temporal(s1_data)
            image_dict["s1"] = s1_out
            dates_dict["s1"] = self._dates_to_doy(s1_dates)
            sensors.append("s1")

        return {
            "image": image_dict,
            "dates": dates_dict,
            "target": label,
            "metadata": {
                "sensors": sensors,
                "n_s2_bands": NUM_S2_BANDS,
                "n_s1_bands": NUM_S1_BANDS if self.use_s1 else 0,
                "n_frames": n_frames,
                "temporal_mode": self.temporal_mode,
                "patch_id": int(patch_id),
                "wavelengths_s2": S2_WAVELENGTHS,
                "bandwidths_s2": S2_BANDWIDTHS,
            },
        }