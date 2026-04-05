"""
MultiEarth Baseline Dataset — Deforestation Segmentation
==========================================================

Fixed-format dataset for baseline models (UNet, ViT+UperNet, U-TAE).

All outputs are 256×256 (L8 upsampled from 85×85 via bilinear).
Labels always 256×256 at native 10m resolution.

Normalization (same for Atomizer and baselines):
  Per-band z-score → clamp to [-3, 3] → rescale to [-1, 1]
  Stats precomputed per sensor and saved in multiearth_norm_stats.pt

Cross-sensor evaluation:
  Load data from source sensor, interpolate bands to target sensor's
  wavelength grid. Normalization applied BEFORE interpolation (in
  source sensor's band space).

Data:
  Samples loaded from precomputed multiearth_split.csv
  NC files opened lazily and cached.

Directory structure:
  ./data/multi_earth/
  ├── multiearth_split.csv
  ├── multiearth_norm_stats.pt
  ├── deforestation_train.nc
  ├── sent2_b1-b4_train.nc
  ├── sent2_b5-b8_train.nc
  ├── sent2_b9-b12_train.nc
  └── landsat8_train.nc
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


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 2
IGNORE_INDEX = 255
IMG_SIZE = 256

CLASS_NAMES = {0: "Forest", 1: "Deforested"}

# ── Sentinel-2 L2A ──────────────────────────────────────────────────
S2_BAND_ORDER  = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S2_WAVELENGTHS = torch.tensor([443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190], dtype=torch.float32)
S2_BANDWIDTHS  = torch.tensor([20,  65,  35,  30,  15,  15,  20, 115,  20,  20,   90,  180], dtype=torch.float32)
S2_GSD = 10.0
S2_SCALE = 10000.0
NUM_S2_BANDS = 12

# ── Landsat-8 Collection 2 SR ───────────────────────────────────────
L8_BAND_ORDER  = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
L8_WAVELENGTHS = torch.tensor([443, 482, 562, 655, 865, 1609, 2201], dtype=torch.float32)
L8_BANDWIDTHS  = torch.tensor([16,  60,  57,  37,  28,   85,  187], dtype=torch.float32)
L8_GSD = 30.0
L8_SCALE_FACTOR = 0.0000275
L8_SCALE_OFFSET = -0.2
NUM_L8_BANDS = 7

# ── Landsat-5 TM ────────────────────────────────────────────────────
L5_BAND_ORDER  = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
L5_WAVELENGTHS = torch.tensor([486, 571, 660, 835, 1676, 2223], dtype=torch.float32)
L5_BANDWIDTHS  = torch.tensor([66,  81,  61, 121,  200,  261], dtype=torch.float32)
L5_GSD = 30.0
L5_SCALE_FACTOR = 0.0000275
L5_SCALE_OFFSET = -0.2
NUM_L5_BANDS = 6


# ═══════════════════════════════════════════════════════════════════════
# SPECTRAL INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════

def spectral_interpolate(
    image: torch.Tensor,
    source_wavelengths: torch.Tensor,
    target_wavelengths: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate spectral bands from source to target wavelength grid.

    For each target wavelength, linearly interpolates between the two
    nearest source bands. Outside source range: nearest-neighbor.

    Args:
        image: [C_source, H, W]
        source_wavelengths: [C_source] sorted, in nm
        target_wavelengths: [C_target] sorted, in nm

    Returns:
        [C_target, H, W]
    """
    C_src = source_wavelengths.shape[0]

    idx = torch.searchsorted(source_wavelengths, target_wavelengths)
    idx_hi = idx.clamp(1, C_src - 1)
    idx_lo = idx_hi - 1

    wl_lo = source_wavelengths[idx_lo]
    wl_hi = source_wavelengths[idx_hi]

    denom = (wl_hi - wl_lo).clamp(min=1e-6)
    alpha = ((target_wavelengths - wl_lo) / denom).clamp(0.0, 1.0)

    val_lo = image[idx_lo]
    val_hi = image[idx_hi]

    return val_lo * (1.0 - alpha[:, None, None]) + val_hi * alpha[:, None, None]


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

def augment_d4(image: torch.Tensor, label: torch.Tensor):
    """Random D4 (4 rotations × 2 flips). Works for [C,H,W] and [T,C,H,W]."""
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, dims=(-2, -1))
        label = torch.rot90(label, k, dims=(-2, -1))
    if random.random() > 0.5:
        image = torch.flip(image, dims=(-1,))
        label = torch.flip(label, dims=(-1,))
    return image, label


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

class MultiEarthBaselineDataset(Dataset):
    """
    MultiEarth Deforestation dataset for baseline models.

    All outputs are 256×256. L8/L5 upsampled via bilinear.
    Labels always 256×256.

    Normalization: per-band z-score → clamp [-3, 3] → scale to [-1, 1].
    Same normalization for Atomizer and baselines.

    For cross-sensor evaluation:
      Load from `sensor`, normalize with source stats, then interpolate
      bands to `cross_sensor_target` wavelength grid. A model trained
      on the target sensor can then process this data.

    Args:
        data_dir: directory with NC files, CSV, and norm stats
        csv_path: precomputed split CSV filename
        split: "train", "val", "test"
        sensor: "s2" or "l8" — which sensor data to load
        cross_sensor_target: None (same-sensor) or "s2"/"l8"
        n_timesteps: temporal frames before label date
        temporal_mode: "stack" or "sequence"
        augment: D4 augmentation (train only)
    """

    def __init__(
        self,
        data_dir: str = "./data/multi_earth",
        csv_path: str = "multiearth_split.csv",
        split: str = "train",
        sensor: str = "s2",
        cross_sensor_target: Optional[str] = None,
        n_timesteps: int = 3,
        temporal_mode: str = "stack",
        augment: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.sensor = sensor.lower()
        self.cross_sensor_target = cross_sensor_target
        self.n_timesteps = n_timesteps
        self.temporal_mode = temporal_mode
        self.augment = augment and (split == "train")

        assert self.sensor in ("s2", "l8"), f"Unknown sensor: {sensor}"
        if cross_sensor_target:
            assert cross_sensor_target in ("s2", "l8"), \
                f"Unknown cross_sensor_target: {cross_sensor_target}"

        # NC file manager
        self.nc = NCManager(data_dir)

        # Load samples from CSV
        self.samples = self._load_csv(
            os.path.join(data_dir, csv_path), split, self.sensor)

        # Sensor configs
        self._src_cfg = self._sensor_config(self.sensor)
        if cross_sensor_target and cross_sensor_target != self.sensor:
            self._out_cfg = self._sensor_config(cross_sensor_target)
            self.output_sensor = cross_sensor_target
        else:
            self._out_cfg = self._src_cfg
            self.output_sensor = self.sensor

        # Normalization stats
        self.norm_stats = self._load_norm_stats(data_dir)

        # NC index maps
        self._build_nc_index_map()

        # Summary
        mode_str = (f"{self.sensor}→{self.output_sensor}"
                    if cross_sensor_target else self.sensor)
        print(f"[MultiEarth-BL] split={split}, mode={mode_str}, "
              f"samples={len(self.samples)}")
        print(f"[MultiEarth-BL] output: {self._out_cfg['n_bands']} bands × "
              f"{IMG_SIZE}×{IMG_SIZE}, temporal={temporal_mode}")
        print(f"[MultiEarth-BL] normalization: per-band z-score → 3σ → [-1,1]")

    @staticmethod
    def _sensor_config(sensor: str) -> dict:
        if sensor == "s2":
            return {
                "n_bands": NUM_S2_BANDS,
                "wavelengths": S2_WAVELENGTHS,
                "bandwidths": S2_BANDWIDTHS,
                "gsd": S2_GSD,
                "native_size": 256,
                "band_order": S2_BAND_ORDER,
            }
        elif sensor == "l8":
            return {
                "n_bands": NUM_L8_BANDS,
                "wavelengths": L8_WAVELENGTHS,
                "bandwidths": L8_BANDWIDTHS,
                "gsd": L8_GSD,
                "native_size": 85,
                "band_order": L8_BAND_ORDER,
            }
        raise ValueError(f"Unknown sensor: {sensor}")

    # ═════════════════════════════════════════════════════════════════
    # NORMALIZATION
    # ═════════════════════════════════════════════════════════════════

    def _load_norm_stats(self, data_dir: str) -> dict:
        """Load precomputed per-band normalization stats."""
        path = os.path.join(data_dir, "multiearth_norm_stats.pt")
        if os.path.exists(path):
            stats = torch.load(path, weights_only=True)
            print(f"[MultiEarth-BL] Loaded norm stats from {path}")
            return stats

        print(f"[MultiEarth-BL] WARNING: {path} not found, using identity")
        return {
            "s2_mean": torch.zeros(NUM_S2_BANDS),
            "s2_std": torch.ones(NUM_S2_BANDS),
            "l8_mean": torch.zeros(NUM_L8_BANDS),
            "l8_std": torch.ones(NUM_L8_BANDS),
        }

    def _normalize(self, image: torch.Tensor, sensor: str) -> torch.Tensor:
        """
        Per-band z-score → clamp [-3, 3] → rescale to [-1, 1].

        Args:
            image: [C, H, W] in physical reflectance units
            sensor: "s2" or "l8" (determines which stats to use)

        Returns:
            [C, H, W] in [-1, 1]
        """
        mean = self.norm_stats[f"{sensor}_mean"]
        std = self.norm_stats[f"{sensor}_std"].clamp(min=1e-6)

        # Reshape for broadcasting: [C] → [C, 1, 1]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

        # Z-score
        normalized = (image - mean) / std

        # 3-sigma clamp → [-1, 1]
        normalized = torch.clamp(normalized, -3.0, 3.0) / 3.0

        return normalized

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
        """Map nc_index → array position (indices may be non-contiguous)."""
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
        Band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12

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

        # Band → (ds, band_list) mapping
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

        image = np.stack(bands, axis=0) / S2_SCALE  # [12, 256, 256] reflectance
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

        image = np.stack(bands, axis=0)  # [7, 85, 85]
        image = image * L8_SCALE_FACTOR + L8_SCALE_OFFSET
        image = np.clip(image, 0.0, 1.0)
        image = torch.from_numpy(image)

        # Upsample to 256×256
        image = F.interpolate(
            image.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        return image  # [7, 256, 256]

    def _load_sensor_image_raw(self, nc_idx: int) -> torch.Tensor:
        """Load one image as raw reflectance [0, 1]. Always 256×256."""
        if self.sensor == "s2":
            return self._load_s2_image_raw(nc_idx)  # [12, 256, 256]
        else:
            return self._load_l8_image_raw(nc_idx)   # [7, 256, 256]

    def _load_sensor_image(self, nc_idx: int) -> torch.Tensor:
        """
        Load one image, apply cross-sensor interpolation if needed,
        then normalize with the OUTPUT sensor's stats.

        Flow for same-sensor:
          raw reflectance → normalize(source stats) → [-1, 1]

        Flow for cross-sensor:
          raw reflectance → interpolate to target grid → normalize(target stats) → [-1, 1]
        """
        image = self._load_sensor_image_raw(nc_idx)  # [C_src, 256, 256] reflectance

        if self.cross_sensor_target and self.cross_sensor_target != self.sensor:
            # Cross-sensor: interpolate in reflectance space, then normalize
            # with target sensor stats
            source_wl = self._src_cfg["wavelengths"]
            target_wl = self._out_cfg["wavelengths"]
            image = spectral_interpolate(image, source_wl, target_wl)  # [C_target, 256, 256]
            image = self._normalize(image, self.output_sensor)
        else:
            # Same-sensor: normalize with own stats
            image = self._normalize(image, self.sensor)

        return image

    # ═════════════════════════════════════════════════════════════════
    # TEMPORAL FORMATTING
    # ═════════════════════════════════════════════════════════════════

    def _format_temporal(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Input: [T, C, H, W]
        Output: "stack" → [C*T, H, W], "sequence" → [T, C, H, W]
        """
        if self.temporal_mode == "stack":
            T, C, H, W = frames.shape
            return frames.reshape(T * C, H, W)
        return frames

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
                img = self._load_sensor_image(nc_idx)  # [C_out, 256, 256] in [-1, 1]
            except Exception as e:
                n_bands = self._out_cfg["n_bands"]
                img = torch.zeros(n_bands, IMG_SIZE, IMG_SIZE)

            frames.append(img)

        # Stack: [T, C_out, 256, 256]
        frames = torch.stack(frames, dim=0)
        frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=-1.0)

        # ── D4 augmentation ─────────────────────────────────────────
        if self.augment:
            k = random.randint(0, 3)
            do_flip = random.random() > 0.5
            if k > 0:
                frames = torch.rot90(frames, k, dims=(-2, -1))
                label = torch.rot90(label, k, dims=(-2, -1))
            if do_flip:
                frames = torch.flip(frames, dims=(-1,))
                label = torch.flip(label, dims=(-1,))

        # ── Format temporal ─────────────────────────────────────────
        image = self._format_temporal(frames)

        # ── Delta days ──────────────────────────────────────────────
        delta_days = torch.tensor(sample["delta_days"], dtype=torch.float32)

        return {
            "image": {self.output_sensor: image},
            "dates": {self.output_sensor: delta_days},
            "target": label,
            "metadata": {
                "source_sensor": self.sensor,
                "output_sensor": self.output_sensor,
                "n_bands": self._out_cfg["n_bands"],
                "n_timesteps": self.n_timesteps,
                "temporal_mode": self.temporal_mode,
                "gsd": self._src_cfg["gsd"],
                "lat": sample["lat"],
                "lon": sample["lon"],
                "label_date": sample["label_date"],
                "delta_days": sample["delta_days"],
                "wavelengths": self._out_cfg["wavelengths"].tolist(),
                "bandwidths": self._out_cfg["bandwidths"].tolist(),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════

def build_multiearth_loaders(
    data_dir: str,
    sensor: str = "s2",
    cross_sensor_target: Optional[str] = None,
    n_timesteps: int = 3,
    temporal_mode: str = "stack",
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Build train/val/test dataloaders."""
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = MultiEarthBaselineDataset(
            data_dir=data_dir,
            split=split,
            sensor=sensor,
            cross_sensor_target=cross_sensor_target,
            n_timesteps=n_timesteps,
            temporal_mode=temporal_mode,
            augment=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


# ═══════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════
#
# 1. Same-sensor S2 (train + test on S2):
#    ds = MultiEarthBaselineDataset(sensor="s2")
#    → image["s2"]: [36, 256, 256] = 12 bands × 3 timesteps, values in [-1, 1]
#
# 2. Same-sensor L8 (train + test on L8):
#    ds = MultiEarthBaselineDataset(sensor="l8")
#    → image["l8"]: [21, 256, 256] = 7 bands × 3 timesteps (upsampled), [-1, 1]
#
# 3. Cross-sensor: UNet trained on S2, test on L8 data:
#    train = MultiEarthBaselineDataset(sensor="s2", split="train")
#    test  = MultiEarthBaselineDataset(sensor="l8", cross_sensor_target="s2", split="test")
#    → test outputs image["s2"]: [36, 256, 256] = 12 bands (interpolated from L8)
#
# 4. Cross-sensor: UNet trained on L8, test on S2 data:
#    train = MultiEarthBaselineDataset(sensor="l8", split="train")
#    test  = MultiEarthBaselineDataset(sensor="s2", cross_sensor_target="l8", split="test")
#    → test outputs image["l8"]: [21, 256, 256] = 7 bands (selected from S2)