"""
DalesPerceiverDataset
========================

DALES LIDAR-only semantic segmentation, PerceiverIO baseline. Mirrors
FractalPerceiverDataset's flat, fixed-size-padded tensor output (no
grouped-token format, no lookup table, default DataLoader collation) --
but with two DALES-specific differences from the FRACTAL baseline:

  1. NO VHR modality (DALES is LIDAR-only) -- so no vhr_tokens/vhr_mask,
     and no need to pad LIDAR tokens to match another modality's width.
  2. Intensity gets its OWN raw scalar in the LIDAR token, alongside
     echo (return_number/number_of_returns) -- the model's echo_mlp/
     intensity_mlp split (see perceiver_dales.py) encodes them
     separately, matching Atomizer's DalesTokenProcessor design
     (echo_encoder + intensity_encoder as two distinct small MLPs).

Position encoding: absolute Fourier(X, Y, Z) -- same convention as
FractalPerceiverDataset, NOT Atomizer's relative-RoPE-to-latents scheme.
This is an intentional, correct architectural difference between vanilla
PerceiverIO and Atomizer, not something to paper over.

Output per __getitem__ (all fixed-size, so default collation works):
    lidar_tokens  [max_lidar_points, LIDAR_RAW_DIM]  raw features (see below)
    lidar_mask    [max_lidar_points]  bool, True=padding
    queries       [max_queries, QUERY_DIM]  Fourier(X,Y,Z) query positions
    queries_mask  [max_queries]  bool, True=padding
    label         [max_queries]  long, ground-truth class (IGNORE_INDEX=255
                                  for padding)

LIDAR_RAW_DIM breakdown:
    LIDAR_FOURIER_DIM (3*65=195): Fourier(X,Y,Z) position
    ECHO_SCALARS_DIM  (2):        raw (a, b) echo values -- encoded by
                                   the model's echo_mlp, not here
    INTENSITY_SCALAR_DIM (1):     raw normalized intensity -- encoded by
                                   the model's intensity_mlp, not here

Reuses the SAME tiled .laz patches as Atomizer's DalesDataset (see
DALES_DATASET.md) -- no separate data preparation needed, just point at
the same root_path.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[Warning] laspy not installed -- required for DALES LAZ reading.")


# ============================================================================
# DALES code -> dense 0-7 label remap (same as DalesDataset / DALES_DATASET.md)
# ============================================================================

DALES_TO_ATOMIZER = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
}


def _build_remap_lut(mapping: dict, num_codes: int = 256,
                      ignore: int = 255) -> np.ndarray:
    lut = np.full(num_codes, ignore, dtype=np.int64)
    for raw_code, label in mapping.items():
        lut[raw_code] = label
    return lut


REMAP_LUT = _build_remap_lut(DALES_TO_ATOMIZER)


# ============================================================================
# Fourier positional encoding
# ============================================================================
# gamma(x; L, f_max) = [x, sin(pi*f_1*x), cos(pi*f_1*x), ..., sin(pi*f_L*x),
#                        cos(pi*f_L*x)]  ->  dim = 2L + 1
# L=32, f_max chosen so dim=65 per axis, matching FRACTAL's convention
# (same Fourier "resolution" across both baselines for a fair comparison).

FOURIER_NUM_BANDS = 32       # L -> dim = 2*32+1 = 65 per axis
FOURIER_MAX_FREQ  = 16.0     # matches Atomizer's pos_max_freq default


def fourier_encode_axis(x: np.ndarray, num_bands: int = FOURIER_NUM_BANDS,
                         max_freq: float = FOURIER_MAX_FREQ) -> np.ndarray:
    """x: [N] in [-1, 1] (or similar bounded range) -> [N, 2*num_bands+1]."""
    freqs = np.linspace(1.0, max_freq, num_bands)  # [num_bands]
    # [N, num_bands]
    angles = np.pi * x[:, None] * freqs[None, :]
    sin = np.sin(angles)
    cos = np.cos(angles)
    return np.concatenate([x[:, None], sin, cos], axis=-1).astype(np.float32)


LIDAR_FOURIER_DIM    = 2 * FOURIER_NUM_BANDS + 1   # 65 per axis * 3 = 195
ECHO_SCALARS_DIM     = 2
INTENSITY_SCALAR_DIM = 1
LIDAR_RAW_DIM = 3 * LIDAR_FOURIER_DIM + ECHO_SCALARS_DIM + INTENSITY_SCALAR_DIM  # 198
QUERY_DIM = 3 * LIDAR_FOURIER_DIM  # 195 -- position-only queries


def _normalize_intensity(intensity: np.ndarray, p_lo: float = 1.0,
                          p_hi: float = 99.0) -> np.ndarray:
    """Same per-patch robust-percentile scaling as Atomizer's DalesDataset
    -- see that module's docstring for the calibration caveat."""
    if intensity.size == 0:
        return intensity.astype(np.float32)
    lo = np.percentile(intensity, p_lo)
    hi = np.percentile(intensity, p_hi)
    if hi <= lo:
        return np.zeros_like(intensity, dtype=np.float32)
    norm = (intensity.astype(np.float32) - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


class DalesPerceiverDataset(Dataset):
    """
    DALES semantic segmentation, PerceiverIO baseline, LIDAR-only.

    Expects the SAME tiled .laz patches as Atomizer's DalesDataset:
        root_path/DALES/<split_dir>/*.laz
    (see DALES_DATASET.md -- no separate data prep needed)
    """

    PATCH_SIZE_M = 50.0   # MUST match how the .laz patches were tiled

    NUM_CLASSES  = 8
    IGNORE_INDEX = 255

    Z_GROUND_REL_LO    = -15.0
    Z_GROUND_REL_HI    = 30.0
    Z_GROUND_REL_SCALE = 15.0
    GROUND_MEDIAN_MIN_PTS = 50

    SPLIT_DIRS = {
        "train": "train", "val": "val", "validation": "val", "test": "test",
    }

    def __init__(
        self,
        root_path: str = "./data",
        mode: str = "train",
        max_lidar_points: int = 256_000,
        max_queries: int = 256_000,
        valid_patches_file: str = None,
        use_augmentation: bool = True,
        sigma_xy_m: float = 0.05,     # 5cm physical jitter, matching
                                       # Atomizer's ~0.25px @ 0.2m/px
        sigma_z_normed: float = 0.003,
        min_points: int = 1000,
    ):
        super().__init__()
        if not HAS_LASPY:
            raise ImportError("laspy required for DALES dataset")

        self.root_path = root_path
        self.split = mode
        self.max_lidar_points = max_lidar_points
        self.max_queries = max_queries
        self.min_points = min_points

        self.use_augmentation = bool(use_augmentation) and self.split == "train"
        self.sigma_xy_m = float(sigma_xy_m)
        self.sigma_z_normed = float(sigma_z_normed)

        self._collect_patches(valid_patches_file)
        print(f"[DalesPerceiver] Loaded {len(self.patch_rows)} patches, "
              f"split='{self.split}', augmentation="
              f"{'ON' if self.use_augmentation else 'OFF'}")

    def _collect_patches(self, valid_patches_file: str = None):
        split_dir = self.SPLIT_DIRS.get(self.split)
        if split_dir is None:
            raise ValueError(f"Unknown split: {self.split}")
        laz_root = Path(self.root_path) / "DALES" / split_dir
        if not laz_root.exists():
            raise FileNotFoundError(f"DALES tiled LAZ root not found: {laz_root}")

        valid_set = None
        if valid_patches_file is not None and os.path.exists(valid_patches_file):
            with open(valid_patches_file) as f:
                valid_data = json.load(f)
            split_key = {"train": "train", "val": "val",
                         "validation": "val", "test": "test"}[self.split]
            valid_set = set(valid_data.get(split_key, []))

        self.patch_rows = []
        for laz_path in sorted(laz_root.rglob("*.laz")):
            patch_id = laz_path.stem
            if valid_set is not None and patch_id not in valid_set:
                continue
            self.patch_rows.append({"patch_id": patch_id, "laz_path": str(laz_path)})

        if not self.patch_rows:
            raise RuntimeError(f"[DalesPerceiver] No patches found under {laz_root}.")

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]
        las = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]
        if n_points_raw < self.min_points:
            return self.__getitem__((index + 1) % len(self))

        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float32)
        classification = np.clip(np.asarray(las.classification, dtype=np.int64),
                                  0, REMAP_LUT.shape[0] - 1)
        labels = REMAP_LUT[classification]
        intensity_raw = np.asarray(las.intensity, dtype=np.float32)
        return_number = np.asarray(las.return_number, dtype=np.float32)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.float32)

        # -- Position: center x,y within the patch to [-1, 1] --------------
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        half_extent = self.PATCH_SIZE_M / 2.0
        x_norm = np.clip((x - x_center) / half_extent, -1.0, 1.0)
        y_norm = np.clip((y - y_center) / half_extent, -1.0, 1.0)

        # -- Elevation: SAME ground-relative normalization as Atomizer's
        # DalesDataset (a data-preprocessing choice, not an architecture
        # difference -- reusing it keeps the comparison fair). -------------
        ground_mask = (labels == 0)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z[ground_mask]))
        else:
            local_ground = float(np.percentile(z, 5.0))
        z_rel = z - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = (z_clip / self.Z_GROUND_REL_SCALE).astype(np.float32)
        # z_norm roughly in [-1, 2] -- clip to [-1,1] for Fourier encoding
        # stability (rare tall structures beyond +15m above local ground
        # get saturated, same trade-off Atomizer accepts).
        z_norm_clipped = np.clip(z_norm, -1.0, 1.0)

        intensity_norm = _normalize_intensity(intensity_raw)

        # -- Echo (a, b) -- same formula as Atomizer's echo encoding -------
        total_returns = np.clip(number_of_returns, 1.0, None)
        echo_a = (return_number - 1.0) / total_returns
        echo_b = (total_returns - return_number) / total_returns

        # -- Augmentation: jitter only (D4 flips would need consistent
        # relabeling of x/y here too -- simple jitter is sufficient for a
        # baseline and avoids re-deriving D4 for a flat, non-tiled-latent
        # architecture where it matters less than for Atomizer's fixed
        # hex grid). ---------------------------------------------------
        if self.use_augmentation:
            rng = np.random.default_rng()
            if self.sigma_xy_m > 0:
                x_norm = np.clip(
                    x_norm + rng.normal(0, self.sigma_xy_m / half_extent, size=x_norm.shape),
                    -1.0, 1.0)
                y_norm = np.clip(
                    y_norm + rng.normal(0, self.sigma_xy_m / half_extent, size=y_norm.shape),
                    -1.0, 1.0)
            if self.sigma_z_normed > 0:
                z_norm_clipped = np.clip(
                    z_norm_clipped + rng.normal(0, self.sigma_z_normed, size=z_norm_clipped.shape),
                    -1.0, 1.0)

        # -- Context (LIDAR) subsampling -------------------------------
        if n_points_raw > self.max_lidar_points:
            seed = None if self.split == "train" else hash(row["patch_id"]) & 0xFFFFFFFF
            rng_ctx = np.random.default_rng(seed=seed)
            sel = rng_ctx.choice(n_points_raw, size=self.max_lidar_points, replace=False)
        else:
            sel = None

        def _sub(arr):
            return arr if sel is None else arr[sel]

        ctx_x, ctx_y, ctx_z = _sub(x_norm), _sub(y_norm), _sub(z_norm_clipped)
        ctx_intensity = _sub(intensity_norm)
        ctx_echo_a, ctx_echo_b = _sub(echo_a), _sub(echo_b)

        pos_fourier = np.concatenate([
            fourier_encode_axis(ctx_x.astype(np.float32)),
            fourier_encode_axis(ctx_y.astype(np.float32)),
            fourier_encode_axis(ctx_z.astype(np.float32)),
        ], axis=-1)  # [n_ctx, 195]

        lidar_raw = np.concatenate([
            pos_fourier,
            ctx_echo_a[:, None].astype(np.float32),
            ctx_echo_b[:, None].astype(np.float32),
            ctx_intensity[:, None].astype(np.float32),
        ], axis=-1)  # [n_ctx, LIDAR_RAW_DIM]

        n_ctx = lidar_raw.shape[0]
        lidar_tokens = torch.zeros(self.max_lidar_points, LIDAR_RAW_DIM)
        lidar_mask   = torch.ones(self.max_lidar_points, dtype=torch.bool)
        lidar_tokens[:n_ctx] = torch.from_numpy(lidar_raw)
        lidar_mask[:n_ctx] = False

        # -- Queries: position only, ALL points (test/val) or a random
        # subsample (train) up to max_queries. ---------------------------
        q_pos_fourier = np.concatenate([
            fourier_encode_axis(x_norm.astype(np.float32)),
            fourier_encode_axis(y_norm.astype(np.float32)),
            fourier_encode_axis(z_norm_clipped.astype(np.float32)),
        ], axis=-1)  # [n_points_raw, 195]

        if n_points_raw > self.max_queries:
            if self.split == "train":
                q_sel = np.random.default_rng().choice(
                    n_points_raw, size=self.max_queries, replace=False)
            else:
                q_sel = np.random.default_rng(
                    seed=hash(row["patch_id"] + "_q") & 0xFFFFFFFF
                ).choice(n_points_raw, size=self.max_queries, replace=False)
            q_pos_fourier = q_pos_fourier[q_sel]
            q_labels = labels[q_sel]
        else:
            q_labels = labels

        n_q = q_pos_fourier.shape[0]
        queries = torch.zeros(self.max_queries, QUERY_DIM)
        queries_mask = torch.ones(self.max_queries, dtype=torch.bool)
        label = torch.full((self.max_queries,), self.IGNORE_INDEX, dtype=torch.long)
        queries[:n_q] = torch.from_numpy(q_pos_fourier)
        queries_mask[:n_q] = False
        label[:n_q] = torch.from_numpy(q_labels.astype(np.int64))

        return {
            "lidar_tokens": lidar_tokens,
            "lidar_mask":   lidar_mask,
            "queries":      queries,
            "queries_mask": queries_mask,
            "label":        label,
            "patch_id":     row["patch_id"],
        }
