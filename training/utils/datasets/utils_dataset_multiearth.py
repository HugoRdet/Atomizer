"""
MultiEarth Dataset — Atomizer Token Format
=============================================

Multi-temporal, cross-sensor deforestation segmentation.

Sensors:
    S2:  12 bands @ 10m (native 256×256)
    L8:   7 bands @ 30m (upsampled to 256×256 for this experiment)

All images upsampled to 256×256 to isolate spectral transfer effects.
Resolution set to 10m for both sensors.

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6            7

Temporal: 3 closest images BEFORE label date.
Time encoding: Δt (days before prediction) registered as time_idx.

Normalization: per-band z-score → 3σ clamp → [-1, 1]
Same normalization as baselines. Stats from multiearth_norm_stats.pt

Returns:
{
    "groups": {
        10.0: {
            "tokens": [N_tokens, 8],
            "mask":   [N_tokens],
            "shape":  (256, 256),
        },
    },
    "tasks": {
        "multiearth_deforestation": {
            "queries": [M, 8],
            "queries_mask": [M],
        },
    },
    "label":             [256, 256],
    "target_resolution": 10.0,
    "image":             [C, 256, 256],  # first frame for viz
}
"""

import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import xarray as xr

from .token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 2
IGNORE_INDEX = 255
IMG_SIZE = 256

CLASS_NAMES = {0: "Forest", 1: "Deforested"}

# ── Sentinel-2 L2A ──────────────────────────────────────────────────
S2_BANDS_INFO = {
    "B01": {"central_wavelength": 443,  "bandwidth": 20,  "idx": 0},
    "B02": {"central_wavelength": 490,  "bandwidth": 65,  "idx": 1},
    "B03": {"central_wavelength": 560,  "bandwidth": 35,  "idx": 2},
    "B04": {"central_wavelength": 665,  "bandwidth": 30,  "idx": 3},
    "B05": {"central_wavelength": 705,  "bandwidth": 15,  "idx": 4},
    "B06": {"central_wavelength": 740,  "bandwidth": 15,  "idx": 5},
    "B07": {"central_wavelength": 783,  "bandwidth": 20,  "idx": 6},
    "B08": {"central_wavelength": 842,  "bandwidth": 115, "idx": 7},
    "B8A": {"central_wavelength": 865,  "bandwidth": 20,  "idx": 8},
    "B09": {"central_wavelength": 945,  "bandwidth": 20,  "idx": 9},
    "B11": {"central_wavelength": 1610, "bandwidth": 90,  "idx": 10},
    "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 11},
}
S2_BAND_ORDER = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S2_GSD = 10.0
S2_SCALE = 10000.0
NUM_S2_BANDS = 12

# ── Landsat-8 Collection 2 SR ───────────────────────────────────────
L8_BANDS_INFO = {
    "SR_B1": {"central_wavelength": 443,  "bandwidth": 16,  "idx": 0},
    "SR_B2": {"central_wavelength": 482,  "bandwidth": 60,  "idx": 1},
    "SR_B3": {"central_wavelength": 562,  "bandwidth": 57,  "idx": 2},
    "SR_B4": {"central_wavelength": 655,  "bandwidth": 37,  "idx": 3},
    "SR_B5": {"central_wavelength": 865,  "bandwidth": 28,  "idx": 4},
    "SR_B6": {"central_wavelength": 1609, "bandwidth": 85,  "idx": 5},
    "SR_B7": {"central_wavelength": 2201, "bandwidth": 187, "idx": 6},
}
L8_BAND_ORDER = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
L8_GSD = 30.0
L8_SCALE_FACTOR = 0.0000275
L8_SCALE_OFFSET = -0.2
NUM_L8_BANDS = 7

RESOLUTION = 10.0  # All outputs at 10m for this experiment

TASK_NAME = "multiearth_deforestation"


# ═══════════════════════════════════════════════════════════════════════
# NC FILE MANAGER
# ═══════════════════════════════════════════════════════════════════════

class NCManager:
    """Opens NetCDF files lazily and caches handles."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._handles: Dict[str, xr.Dataset] = {}

    def get(self, filename: str) -> xr.Dataset:
        if filename not in self._handles:
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"NC file not found: {path}")
            self._handles[filename] = xr.open_dataset(path)
        return self._handles[filename]

    def close_all(self):
        for ds in self._handles.values():
            ds.close()
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

class MultiEarthDataset(Dataset):
    """
    MultiEarth Deforestation dataset — Atomizer token format.

    Loads multi-temporal satellite imagery, builds per-pixel-per-band tokens
    with spectral, spatial, resolution, and temporal metadata.

    Cross-sensor transfer: train on S2 tokens (12 bands × 3 timesteps),
    test on L8 tokens (7 bands × 3 timesteps). Same encoder, different
    spectral_idx values → spectral encoder handles the transfer.

    Args:
        data_dir: directory with NC files, CSV, and norm stats
        csv_path: precomputed split CSV
        split: "train", "val", "test"
        sensor: "s2" or "l8"
        n_timesteps: number of temporal frames
        config_model: model config dict
        look_up: lookup table for spectral/resolution/time indices
        augment: D4 augmentation (train only)
    """

    def __init__(
        self,
        data_dir: str = "./data/multi_earth",
        csv_path: str = "multiearth_split.csv",
        split: str = "train",
        sensor: str = "s2",
        n_timesteps: int = 1,
        config_model: dict = None,
        look_up=None,
        augment: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.sensor = sensor.lower()
        self.n_timesteps = n_timesteps
        self.look_up = look_up
        self.config_model = config_model or {}
        self.augment = augment and (split == "train")

        assert self.sensor in ("s2", "l8"), f"Unknown sensor: {sensor}"

        # Token builder
        self.token_builder = TokenBuilder(look_up)

        # Config
        self.max_queries = config_model.get("trainer", {}).get(
            "max_tokens_reconstruction", 100_000)

        # NC file manager
        self.nc = NCManager(data_dir)

        # Load samples from CSV
        self.samples = self._load_csv(
            os.path.join(data_dir, csv_path), split, self.sensor)

        # Sensor config
        if self.sensor == "s2":
            self.bands_info = S2_BANDS_INFO
            self.n_bands = NUM_S2_BANDS
        else:
            self.bands_info = L8_BANDS_INFO
            self.n_bands = NUM_L8_BANDS

        # Normalization stats
        self.norm_stats = self._load_norm_stats(data_dir)

        # NC index maps
        self._build_nc_index_map()

        # Register spectral indices in lookup table
        self._setup_spectral_indices()

        # Resolution index
        self.resolution_idx = self.look_up.get_resolution_idx(RESOLUTION)

        # Register Δt values as time indices
        self._register_time_indices()

        print(f"[MultiEarth] split={split}, sensor={sensor}, "
              f"samples={len(self.samples)}")
        print(f"[MultiEarth] {self.n_bands} bands, {n_timesteps} timesteps, "
              f"resolution={RESOLUTION}m")

    # ═════════════════════════════════════════════════════════════════
    # CSV LOADING
    # ═════════════════════════════════════════════════════════════════

    def _load_csv(self, csv_path: str, split: str, sensor: str) -> List[dict]:
        count_col = f"{sensor}_count"
        idx_col = f"{sensor}_indices"
        dt_col = f"{sensor}_delta_days"

        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                n_available = int(row[count_col])
                if n_available < self.n_timesteps:
                    continue

                indices = [int(x) for x in row[idx_col].split(";")][:self.n_timesteps]
                delta_days = [float(x) for x in row[dt_col].split(";")][:self.n_timesteps]

                samples.append({
                    "label_idx": int(row["label_idx"]),
                    "label_date": row["label_date"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "sensor_indices": indices,
                    "delta_days": delta_days,
                })
        return samples

    # ═════════════════════════════════════════════════════════════════
    # NC INDEX MAPPING
    # ═════════════════════════════════════════════════════════════════

    def _build_nc_index_map(self):
        ds_label = self.nc.get('deforestation_train.nc')
        self.label_idx_to_pos = {
            int(idx): pos
            for pos, idx in enumerate(ds_label.coords['index'].values)
        }

        if self.sensor == "s2":
            ds = self.nc.get('sent2_b9-b12_train.nc')
        else:
            ds = self.nc.get('landsat8_train.nc')
        self.sensor_idx_to_pos = {
            int(idx): pos
            for pos, idx in enumerate(ds.coords['index'].values)
        }

    # ═════════════════════════════════════════════════════════════════
    # SPECTRAL / TIME SETUP
    # ═════════════════════════════════════════════════════════════════

    def _setup_spectral_indices(self):
        """Register all bands in the lookup table and store indices."""
        self.spectral_indices = []
        for band_name in sorted(self.bands_info.keys(),
                                key=lambda b: self.bands_info[b]["idx"]):
            info = self.bands_info[band_name]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {band_name} key={key} not in lookup table.\n"
                    f"Register it in your config or lookup initialization."
                )
            self.spectral_indices.append(self.look_up.table_wave[key])
        self.spectral_indices = torch.tensor(self.spectral_indices, dtype=torch.long)

    def _register_time_indices(self):
        """Pre-register all unique Δt values in the lookup table."""
        all_dts = set()
        for sample in self.samples:
            for dt in sample["delta_days"]:
                all_dts.add(int(dt))

        for dt in sorted(all_dts):
            self.look_up.get_or_register_time_idx(dt)

        print(f"[MultiEarth] Registered {len(all_dts)} unique Δt values "
              f"(range: {min(all_dts)}-{max(all_dts)} days)")

    # ═════════════════════════════════════════════════════════════════
    # NORMALIZATION
    # ═════════════════════════════════════════════════════════════════

    def _load_norm_stats(self, data_dir: str) -> dict:
        path = os.path.join(data_dir, "multiearth_norm_stats.pt")
        if os.path.exists(path):
            stats = torch.load(path, weights_only=True)
            print(f"[MultiEarth] Loaded norm stats from {path}")
            return stats

        print(f"[MultiEarth] WARNING: {path} not found, using identity")
        return {
            "s2_mean": torch.zeros(NUM_S2_BANDS),
            "s2_std": torch.ones(NUM_S2_BANDS),
            "l8_mean": torch.zeros(NUM_L8_BANDS),
            "l8_std": torch.ones(NUM_L8_BANDS),
        }

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Per-band z-score → 3σ clamp → [-1, 1]."""
        mean = self.norm_stats[f"{self.sensor}_mean"].view(-1, 1, 1)
        std = self.norm_stats[f"{self.sensor}_std"].clamp(min=1e-6).view(-1, 1, 1)
        normalized = (image - mean) / std
        return torch.clamp(normalized, -3.0, 3.0) / 3.0

    # ═════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═════════════════════════════════════════════════════════════════

    def _load_label(self, label_idx: int) -> torch.Tensor:
        """Load binary deforestation mask. Returns [256, 256] long."""
        pos = self.label_idx_to_pos[label_idx]
        ds = self.nc.get('deforestation_train.nc')
        mask = ds.images[pos, 0].values.astype(np.int64)
        return torch.from_numpy(mask)

    def _load_s2_image_raw(self, nc_idx: int) -> torch.Tensor:
        """
        Load one S2 image from 3 NC files → [12, 256, 256] reflectance.

        File layout:
          sent2_b1-b4_train.nc:  B1, B2, B3, B4, QA60
          sent2_b5-b8_train.nc:  B5, B6, B7, B8, B8A, QA60
          sent2_b9-b12_train.nc: B11, B12, B9, QA60
        """
        pos = self.sensor_idx_to_pos[nc_idx]
        bands = []

        ds1 = self.nc.get('sent2_b1-b4_train.nc')
        ds2 = self.nc.get('sent2_b5-b8_train.nc')
        ds3 = self.nc.get('sent2_b9-b12_train.nc')

        all_bands_1 = list(ds1.coords['data_band'].values)
        all_bands_2 = list(ds2.coords['data_band'].values)
        all_bands_3 = list(ds3.coords['data_band'].values)

        band_file_map = {
            'B1': (ds1, all_bands_1), 'B2': (ds1, all_bands_1),
            'B3': (ds1, all_bands_1), 'B4': (ds1, all_bands_1),
            'B5': (ds2, all_bands_2), 'B6': (ds2, all_bands_2),
            'B7': (ds2, all_bands_2), 'B8': (ds2, all_bands_2),
            'B8A': (ds2, all_bands_2),
            'B9': (ds3, all_bands_3),
            'B11': (ds3, all_bands_3), 'B12': (ds3, all_bands_3),
        }

        for bname in S2_BAND_ORDER:
            ds, band_list = band_file_map[bname]
            bi = band_list.index(bname)
            bands.append(ds.images[pos, bi, 0].values.astype(np.float32))

        image = np.stack(bands, axis=0) / S2_SCALE
        return torch.from_numpy(np.clip(image, 0.0, 1.0))

    def _load_l8_image_raw(self, nc_idx: int) -> torch.Tensor:
        """
        Load one L8 image → [7, 256, 256] reflectance (upsampled from 85×85).
        """
        pos = self.sensor_idx_to_pos[nc_idx]
        ds = self.nc.get('landsat8_train.nc')
        all_bands = list(ds.coords['data_band'].values)

        bands = []
        for bname in L8_BAND_ORDER:
            bi = all_bands.index(bname)
            bands.append(ds.images[pos, bi, 0].values.astype(np.float32))

        image = np.stack(bands, axis=0)
        image = image * L8_SCALE_FACTOR + L8_SCALE_OFFSET
        image = np.clip(image, 0.0, 1.0)
        image = torch.from_numpy(image)

        # Upsample to 256×256
        image = F.interpolate(
            image.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        return image

    def _load_sensor_image(self, nc_idx: int) -> torch.Tensor:
        """Load one image → normalize → [C, 256, 256] in [-1, 1]."""
        if self.sensor == "s2":
            image = self._load_s2_image_raw(nc_idx)
        else:
            image = self._load_l8_image_raw(nc_idx)

        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        return self._normalize(image)

    # ═════════════════════════════════════════════════════════════════
    # TOKEN BUILDING
    # ═════════════════════════════════════════════════════════════════

    def _build_temporal_tokens(
        self,
        frames: List[torch.Tensor],
        label: torch.Tensor,
        delta_days: List[float],
    ) -> torch.Tensor:
        """
        Build flat tokens for multi-temporal data.

        Each timestep gets Δt registered as its time_idx.
        All frames concatenated into one flat tensor.

        Args:
            frames: list of [C, H, W] tensors (one per timestep)
            label: [H, W]
            delta_days: list of Δt values (days before prediction)

        Returns:
            tokens: [T * C * H * W, 8]
        """
        all_tokens = []

        for frame, dt in zip(frames, delta_days):
            time_idx = self.look_up.get_or_register_time_idx(int(dt))

            frame_tokens = self.token_builder.build_tokens(
                image=frame,
                label=label,
                resolution=RESOLUTION,
                spectral_indices=self.spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=time_idx,
            )
            all_tokens.append(frame_tokens)

        return torch.cat(all_tokens, dim=0)

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        # ── Load label (always 256×256) ─────────────────────────────
        label = self._load_label(sample["label_idx"])

        # ── Load T sensor images (normalized) ───────────────────────
        frames = []
        for nc_idx in sample["sensor_indices"]:
            try:
                img = self._load_sensor_image(nc_idx)
            except Exception as e:
                img = torch.zeros(self.n_bands, IMG_SIZE, IMG_SIZE)
            frames.append(img)

        # ── D4 augmentation ─────────────────────────────────────────
        if self.augment:
            d4_k = random.randint(0, 3)
            d4_flip = random.random() > 0.5

            if d4_k > 0:
                frames = [torch.rot90(f, d4_k, dims=(-2, -1)) for f in frames]
                label = torch.rot90(label, d4_k, dims=(-2, -1))
            if d4_flip:
                frames = [torch.flip(f, dims=(-1,)) for f in frames]
                label = torch.flip(label, dims=(-1,))

        # ── Build tokens ────────────────────────────────────────────
        tokens = self._build_temporal_tokens(
            frames, label, sample["delta_days"])

        token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        groups = {
            RESOLUTION: {
                "tokens": tokens,
                "mask": token_mask,
                "shape": (IMG_SIZE, IMG_SIZE),
            },
        }

        # ── Queries ─────────────────────────────────────────────────
        first_spectral_idx = self.spectral_indices[0]
        first_time_idx = self.look_up.get_or_register_time_idx(
            int(sample["delta_days"][0]))

        queries = self.token_builder.build_queries(
            label=label,
            resolution=RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=first_time_idx,
        )
        queries = self.token_builder.subsample_queries(
            queries,
            max_queries=self.max_queries,
            ignore_index=IGNORE_INDEX,
            prioritize_valid=True,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        # ── Reference image (first frame for viz) ───────────────────
        image = frames[0]  # [C, 256, 256]

        return {
            "groups": groups,
            "tasks": {
                TASK_NAME: {
                    "queries": queries,
                    "queries_mask": queries_mask,
                },
            },
            "label": label,
            "target_resolution": RESOLUTION,
            "image": image,
        }

    # ═════════════════════════════════════════════════════════════════
    # VIZ SAMPLE (no query subsampling)
    # ═════════════════════════════════════════════════════════════════

    def get_viz_sample(self, index: int) -> dict:
        """Full-resolution sample for visualization — all pixels as queries."""
        sample = self.samples[index]

        label = self._load_label(sample["label_idx"])

        frames = []
        for nc_idx in sample["sensor_indices"]:
            try:
                img = self._load_sensor_image(nc_idx)
            except Exception:
                img = torch.zeros(self.n_bands, IMG_SIZE, IMG_SIZE)
            frames.append(img)

        tokens = self._build_temporal_tokens(
            frames, label, sample["delta_days"])
        token_mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        groups = {
            RESOLUTION: {
                "tokens": tokens,
                "mask": token_mask,
                "shape": (IMG_SIZE, IMG_SIZE),
            },
        }

        # All pixels as queries (no subsampling)
        first_spectral_idx = self.spectral_indices[0]
        first_time_idx = self.look_up.get_or_register_time_idx(
            int(sample["delta_days"][0]))

        queries = self.token_builder.build_queries(
            label=label,
            resolution=RESOLUTION,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=first_time_idx,
        )
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        image = frames[0]

        return {
            "groups": groups,
            "tasks": {
                TASK_NAME: {
                    "queries": queries,
                    "queries_mask": queries_mask,
                },
            },
            "label": label,
            "target_resolution": RESOLUTION,
            "image": image,
            "image_shape": (IMG_SIZE, IMG_SIZE),
            "n_queries": queries.shape[0],
            "dataset_name": "MultiEarth"
        }