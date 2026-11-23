"""
FRACTAL PTv3 Dataset
====================

Point-cloud version of FractalPerceiverDataset for Point Transformer V3.
Same patches, same LAS -> 7-class remap, same ground-relative Z, same
D4 + jitter, and the same per-split subsampling rule as the PerceiverIO
dataset, so both models are scored on the same kind of points.

Place this file in the SAME package as FractalPerceiverDataset (it
imports .augmentations).

One sample = one LiDAR patch, as points:

    coord       [Nv, 3] float32  (x, y, z) in metres, patch-local;
                                 z = ground-relative height (clipped)
    grid_coord  [Nv, 3] int32    floor((coord - min) / grid_size)
    feat        [Nv, 9] float32  x_n, y_n, z_n, echo a, echo b,
                                 NIR, R, G, B (ortho pixel under the point)
    eval_index  [Ne]    int64    for each evaluated LiDAR point, the row
                                 of coord/feat whose prediction it takes
    eval_labels [Ne]    int64    labels of the evaluated points (255 = ignore)

Why two point sets
------------------
PTv3's sparse convolutions need at most ONE point per voxel, so points
sharing a voxel are merged (representative: random in train, first in
val/test). Every LiDAR point is still evaluated: eval_index maps it to
its voxel's representative. Loss and metrics are computed on the eval
points, i.e. the same points PerceiverIO is scored on.

  standard (train/val/test)
      context     = up to max_points points (same rule as PerceiverIO:
                    random in train, fixed per-patch subset in val/test)
      eval points = the context points

  eval_full_scene=True (test only)
      eval points = ALL points of the patch
      full_scene_context="subsample": PTv3 sees the max_points subsample
          (like PerceiverIO's context); every point takes the prediction
          of its nearest context point (KD-tree, in metres).
      full_scene_context="all": PTv3 sees every point (more context than
          PerceiverIO gets; report it separately).

VHR
---
Early fusion: each point carries the 4 ortho bands of the pixel it falls
in. Unlike PerceiverIO, which also receives EVERY ortho pixel as a token,
PTv3 only sees the ortho at LiDAR point locations.
"""

import os
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .augmentations import D4Augmentation


# ============================================================================
# Constants (copied verbatim from FractalPerceiverDataset)
# ============================================================================

IGNORE_INDEX = 255
NUM_FEATURES = 9

LAS_TO_FRACTAL = {
    1:  0,   # unclassified      -> other
    2:  1,   # ground            -> ground
    3:  2,   # low vegetation    -> vegetation
    4:  2,   # medium vegetation -> vegetation
    5:  2,   # high vegetation   -> vegetation
    6:  3,   # building          -> building
    9:  4,   # water             -> water
    17: 5,   # bridge deck       -> bridge
    64: 6,   # permanent struct  -> permanent_structure
}


def _build_remap_lut(num_codes: int = 256) -> np.ndarray:
    lut = np.full(num_codes, IGNORE_INDEX, dtype=np.int64)
    for las_code, fractal_label in LAS_TO_FRACTAL.items():
        lut[las_code] = fractal_label
    return lut


REMAP_LUT = _build_remap_lut()


def _normalize_coords(coords_px: np.ndarray, patch_size_px: int) -> np.ndarray:
    """Map pixel coordinates [0, patch_size_px) to [-1, 1]."""
    return (coords_px / (patch_size_px - 1)) * 2.0 - 1.0


def _echo_ab(return_number: np.ndarray, number_of_returns: np.ndarray) -> np.ndarray:
    """a = (r-1)/t (pulse already returned), b = (t-r)/t (still to return)."""
    r = return_number.astype(np.float32)
    t = np.maximum(number_of_returns.astype(np.float32), 1.0)
    return np.stack([(r - 1.0) / t, (t - r) / t], axis=1)


def _patch_seed(patch_id: str) -> int:
    """Stable per-patch seed. (Python's hash() of a str changes between
    runs unless PYTHONHASHSEED is set, so it is not used here.)"""
    return zlib.crc32(patch_id.encode("utf-8")) & 0xFFFFFFFF


# ============================================================================
# Dataset
# ============================================================================

class FractalPTv3Dataset(Dataset):

    VHR_RESOLUTION = 0.2
    PATCH_SIZE_PX  = 250
    NUM_CLASSES    = 7
    MIN_POINTS     = 1000

    Z_GROUND_REL_LO       = -15.0
    Z_GROUND_REL_HI       =  30.0
    Z_GROUND_REL_SCALE    =  15.0
    GROUND_MEDIAN_MIN_PTS =  50

    SPLIT_DIRS = {
        "train":      "train/train",
        "val":        "val/val",
        "test":       "test/test",
        "validation": "val/val",
    }

    def __init__(
        self,
        root_path: str = "./data",
        mode: str = "train",
        max_points: int = 16_000,
        grid_size: float = 0.1,
        valid_patches_file: str = None,
        use_augmentation: bool = True,
        sigma_xy_pixels: float = 0.25,
        sigma_z_normed: float = 0.003,
        eval_full_scene: bool = False,
        full_scene_context: str = "subsample",
        use_vhr: bool = True,
    ):
        super().__init__()
        # use_vhr=False: LiDAR-only run. The ortho is not read at all and the
        # 4 colour features are dropped (feat = x_n, y_n, z_n, echo a, echo b).
        # Patches without an ortho are still skipped, so the patch set is the
        # same as in the VHR runs.
        self.use_vhr      = bool(use_vhr)
        self.num_features = NUM_FEATURES if self.use_vhr else NUM_FEATURES - 4
        for lib, ok in [("laspy", HAS_LASPY), ("rasterio", HAS_RASTERIO)]:
            if not ok:
                raise ImportError(f"{lib} required for FRACTAL dataset")
        assert full_scene_context in ("subsample", "all"), full_scene_context

        self.root_path          = root_path
        self.split              = mode
        self.max_points         = max_points
        self.grid_size          = float(grid_size)
        self.eval_full_scene    = bool(eval_full_scene) and mode == "test"
        self.full_scene_context = full_scene_context

        if eval_full_scene and mode != "test":
            print(f"[FractalPTv3] WARNING: eval_full_scene ignored for split='{mode}'.")
        if (self.eval_full_scene and full_scene_context == "subsample"
                and not HAS_SCIPY):
            raise ImportError("scipy required for full_scene_context='subsample'")

        self.augmenter = D4Augmentation(
            enabled=(use_augmentation and mode == "train"),
            p_flip_h=0.5,
            p_flip_v=0.5,
        )
        self.sigma_xy_pixels = float(sigma_xy_pixels)
        self.sigma_z_normed  = float(sigma_z_normed)

        self._collect_patches(valid_patches_file)

        print(f"[FractalPTv3] split='{mode}', {len(self.patch_rows)} patches, "
              f"max_points={max_points}, grid_size={self.grid_size} m, "
              f"augment={'ON' if self.augmenter.enabled else 'OFF'}, "
              f"VHR={'ON' if self.use_vhr else 'OFF'} "
              f"({self.num_features} features)")
        if self.eval_full_scene:
            print(f"[FractalPTv3] full-scene eval: all points scored, "
                  f"context='{full_scene_context}'")

    # =========================================================================
    # PATCH COLLECTION (same logic as FractalPerceiverDataset)
    # =========================================================================

    def _collect_patches(self, valid_patches_file: str = None):
        split_dir = self.SPLIT_DIRS.get(self.split)
        if split_dir is None:
            raise ValueError(f"Unknown split: {self.split}")

        laz_root  = Path(self.root_path) / "FRACTAL"      / "data" / split_dir
        irgb_root = Path(self.root_path) / "FRACTAL-IRGB" / "data" / split_dir
        if not laz_root.exists():
            raise FileNotFoundError(f"FRACTAL LAZ root not found: {laz_root}")
        if not irgb_root.exists():
            raise FileNotFoundError(f"FRACTAL IRGB root not found: {irgb_root}")

        ortho_index = {}
        for ext in ("*.tiff", "*.tif"):
            for op in irgb_root.rglob(ext):
                ortho_index[op.stem] = op

        valid_set = None
        if valid_patches_file is not None and os.path.exists(valid_patches_file):
            with open(valid_patches_file) as f:
                valid_data = json.load(f)
            split_key = {"train": "train", "val": "val",
                         "validation": "val", "test": "test"}[self.split]
            valid_set = set(valid_data.get(split_key, []))

        self.patch_rows = []
        missing_orthos = skipped_invalid = 0
        for laz_path in sorted(laz_root.rglob("*.laz")):
            patch_id = laz_path.stem
            if valid_set is not None and patch_id not in valid_set:
                skipped_invalid += 1
                continue
            ortho_path = ortho_index.get(patch_id)
            if ortho_path is None:
                missing_orthos += 1
                continue
            self.patch_rows.append({
                "patch_id":   patch_id,
                "laz_path":   str(laz_path),
                "ortho_path": str(ortho_path),
            })

        if missing_orthos:
            print(f"[FractalPTv3] WARNING: {missing_orthos} LAZ files had no ortho — skipped.")
        if skipped_invalid:
            print(f"[FractalPTv3] Filtered out {skipped_invalid} patches.")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]
        aug = self.augmenter.sample(index=index)

        # ── 1. LiDAR (identical preprocessing to PerceiverIO) ──────────────
        las   = laspy.read(row["laz_path"])
        n_raw = las.x.shape[0]
        if n_raw < self.MIN_POINTS:
            return self.__getitem__((index + 1) % len(self))

        x_min   = float(las.x.min())
        y_max   = float(las.y.max())
        lidar_x = (np.asarray(las.x) - x_min) / self.VHR_RESOLUTION
        lidar_y = (y_max - np.asarray(las.y)) / self.VHR_RESOLUTION
        lidar_x = np.clip(lidar_x, 0.0, self.PATCH_SIZE_PX - 1e-3)
        lidar_y = np.clip(lidar_y, 0.0, self.PATCH_SIZE_PX - 1e-3)

        return_number     = np.asarray(las.return_number,     dtype=np.int64)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

        las_cls = np.clip(np.asarray(las.classification, dtype=np.int64),
                          0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]

        z_raw       = np.asarray(las.z, dtype=np.float32)
        ground_mask = (labels == 1)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z_raw[ground_mask]))
        else:
            local_ground = float(np.percentile(z_raw, 5.0))
        z_clip = np.clip(z_raw - local_ground,
                         self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        xy = np.stack([lidar_x, lidar_y], axis=1).astype(np.float32)
        if not aug.is_identity:
            xy = self.augmenter.apply_to_xy(xy, aug, patch_size_px=self.PATCH_SIZE_PX)
        if self.augmenter.enabled and (self.sigma_xy_pixels > 0 or self.sigma_z_normed > 0):
            jitter_seed = (index * 2147483647) ^ 0x9E3779B9
            xy, z_norm = self.augmenter.apply_jitter(
                xy, z=z_norm,
                sigma_xy=self.sigma_xy_pixels,
                sigma_z=self.sigma_z_normed,
                seed=jitter_seed,
            )
        xy = np.clip(xy, 0.0, self.PATCH_SIZE_PX - 1e-3).astype(np.float32)
        lidar_x, lidar_y = xy[:, 0], xy[:, 1]

        # ── 2. Per-point LiDAR features (all points) ───────────────────────
        feat_parts = [
            _normalize_coords(lidar_x, self.PATCH_SIZE_PX)[:, None],
            _normalize_coords(lidar_y, self.PATCH_SIZE_PX)[:, None],
            np.clip(z_norm, -1.0, 1.0)[:, None],
            _echo_ab(return_number, number_of_returns),
        ]

        # ── 3. Ortho colours under each point (skipped if use_vhr=False) ───
        if self.use_vhr:
            with rasterio.open(row["ortho_path"]) as src:
                ortho = src.read().astype(np.float32)      # [4, 250, 250] NIR/R/G/B
            ortho = torch.from_numpy(ortho)
            ortho = (ortho / 127.5) - 1.0
            ortho = torch.clamp(ortho, -10.0, 10.0)
            ortho = torch.nan_to_num(ortho, nan=0.0, posinf=10.0, neginf=-10.0)
            if not aug.is_identity:
                ortho = self.augmenter.apply(ortho, aug)
            ortho = ortho.numpy()

            ix = np.clip(lidar_x.astype(np.int64), 0, self.PATCH_SIZE_PX - 1)
            iy = np.clip(lidar_y.astype(np.int64), 0, self.PATCH_SIZE_PX - 1)
            feat_parts.append(ortho[:, iy, ix].T)                     # [N, 4]

        feat_all = np.concatenate(feat_parts, axis=1).astype(np.float32)
        # [N, 9] with VHR, [N, 5] without

        # metric coordinates (all points)

        coord_all = np.stack([
            lidar_x * self.VHR_RESOLUTION,
            lidar_y * self.VHR_RESOLUTION,
            z_norm * self.Z_GROUND_REL_SCALE,
        ], axis=1).astype(np.float32)                                 # [N, 3] metres

        # ── 4. Context points (same subsampling rule as PerceiverIO) ───────
        if self.max_points is not None and n_raw > self.max_points:
            rng = np.random.default_rng(
                None if self.split == "train" else _patch_seed(row["patch_id"])
            )
            ctx = np.sort(rng.choice(n_raw, size=self.max_points, replace=False))
        else:
            ctx = np.arange(n_raw)
        if self.eval_full_scene and self.full_scene_context == "all":
            ctx = np.arange(n_raw)

        # ── 5. Voxelize: one representative point per voxel ────────────────
        c    = coord_all[ctx]
        g    = np.floor((c - c.min(0)) / self.grid_size).astype(np.int64)
        dims = g.max(0) + 1
        key  = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]

        order = (np.random.default_rng().permutation(len(ctx))
                 if self.split == "train" else np.arange(len(ctx)))
        _, first, inv_sorted = np.unique(key[order], return_index=True,
                                         return_inverse=True)
        keep = order[first]                          # ctx-local index of each voxel's point
        inv  = np.empty(len(ctx), dtype=np.int64)    # voxel of each ctx point
        inv[order] = inv_sorted.reshape(-1)

        # ── 6. Evaluated points and where their prediction comes from ──────
        if self.eval_full_scene and len(ctx) < n_raw:
            _, nn_idx   = cKDTree(c).query(coord_all, k=1)
            eval_index  = inv[nn_idx]
            eval_labels = labels
        else:
            eval_index  = inv
            eval_labels = labels[ctx]

        return {
            "coord":       torch.from_numpy(c[keep]),
            "grid_coord":  torch.from_numpy(g[keep].astype(np.int32)),
            "feat":        torch.from_numpy(feat_all[ctx][keep]),
            "eval_index":  torch.from_numpy(eval_index.astype(np.int64)),
            "eval_labels": torch.from_numpy(eval_labels.astype(np.int64)),
            "n_context":   int(len(ctx)),
            "patch_id":    row["patch_id"],
        }


# ============================================================================
# Collate: concatenate points, separate samples with offset
# ============================================================================

def fractal_ptv3_collate(batch):
    coord, grid, feat, eidx, elab = [], [], [], [], []
    offset, eval_offset = [], []
    n_vox = n_eval = n_ctx = 0
    for s in batch:
        coord.append(s["coord"])
        grid.append(s["grid_coord"])
        feat.append(s["feat"])
        eidx.append(s["eval_index"] + n_vox)     # shift into the packed batch
        elab.append(s["eval_labels"])
        n_vox  += s["coord"].shape[0]
        n_eval += s["eval_labels"].shape[0]
        n_ctx  += s["n_context"]
        offset.append(n_vox)
        eval_offset.append(n_eval)
    return {
        "coord":       torch.cat(coord),
        "grid_coord":  torch.cat(grid),
        "feat":        torch.cat(feat),
        "offset":      torch.tensor(offset, dtype=torch.long),
        "eval_index":  torch.cat(eidx),
        "eval_labels": torch.cat(elab),
        "eval_offset": torch.tensor(eval_offset, dtype=torch.long),
        "n_context":   n_ctx,
        "patch_id":    [s["patch_id"] for s in batch],
    }
