"""
FRACTAL perceiverIO Dataset
============================

Parallel to FractalDataset (Atomizer) but builds tokens suitable for
PerceiverIO, which expects a flat array of fixed-dimension input tokens.

Token design
------------
All tokens (VHR and LIDAR) are projected to the same dimension input_dim=262
so PerceiverIO's encoder can process them as a single flat sequence.

VHR tokens  [262-dim]:
    Fourier(NIR)  2*16+1 = 33
    Fourier(R)    2*16+1 = 33
    Fourier(G)    2*16+1 = 33
    Fourier(B)    2*16+1 = 33
    Fourier(X)    2*32+1 = 65
    Fourier(Y)    2*32+1 = 65
    ──────────────────────────
    total:             262

LIDAR tokens  [262-dim]:
    Fourier(X)    2*32+1 = 65
    Fourier(Y)    2*32+1 = 65
    Fourier(Z)    2*32+1 = 65
    echo MLP out:       49
    learned pad:        18
    ──────────────────────────
    total:             262

Query tokens  [195-dim, no padding needed]:
    Fourier(X)    2*32+1 = 65
    Fourier(Y)    2*32+1 = 65
    Fourier(Z)    2*32+1 = 65
    ──────────────────────────
    total:             195

The learned LIDAR padding (18-dim) is a single nn.Parameter that lives
in PerceiverFractal (the model), not the dataset. The dataset outputs raw
197-dim LIDAR features (195 Fourier + 2 echo scalars) so the model can
apply the echo MLP + padding consistently across all items.

Query format (Option B)
-----------------------
Queries and their labels are stored as separate tensors:
    "queries"       [M, 195]  Fourier(X, Y, Z)
    "query_labels"  [M]       int64, IGNORE_INDEX=255 for padding

Subsampling
-----------
Training: queries subsampled to MAX_QUERIES_TRAIN=40_000.
Val/test:  queries padded to max_lidar_points.

Augmentation
------------
Same D4 + LIDAR jitter as FractalDataset. Reuses D4Augmentation directly.
"""

import os
import json
from math import pi
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[Warning] laspy not installed — required for FRACTAL LAZ reading.")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed — required for FRACTAL ortho.")

from .augmentations import D4Augmentation


# ============================================================================
# Constants
# ============================================================================

IGNORE_INDEX       = 255
MAX_QUERIES_TRAIN  = 40_000

# Fourier encoding parameters
POS_BANDS = 32      # for X, Y, Z positional axes
POS_FMAX  = 32.0
VAL_BANDS = 16      # for NIR, R, G, B band values
VAL_FMAX  = 16.0

# Per-axis Fourier output dim: 2*bands + 1  (sin + cos + raw)
POS_DIM = 2 * POS_BANDS + 1    # 65
VAL_DIM = 2 * VAL_BANDS + 1    # 33

# VHR: 4 bands + 2 position axes
VHR_DIM   = 4 * VAL_DIM + 2 * POS_DIM     # 4*33 + 2*65 = 262
INPUT_DIM = VHR_DIM                        # 262  (target for both modalities)

# LIDAR raw (before echo MLP + padding): Fourier(X,Y,Z) + echo scalars (a,b)
LIDAR_FOURIER_DIM = 3 * POS_DIM            # 195
LIDAR_ECHO_DIM    = 2                      # raw (a, b) scalars
LIDAR_RAW_DIM     = LIDAR_FOURIER_DIM + LIDAR_ECHO_DIM   # 197

# Echo MLP output dim (matches time_encoder.out_dim in Atomizer)
ECHO_MLP_OUT_DIM  = 49
# Padding needed to bring LIDAR up to INPUT_DIM after echo MLP
# INPUT_DIM = LIDAR_FOURIER_DIM + ECHO_MLP_OUT_DIM + LIDAR_PAD_DIM
# 262        = 195               + 49               + 18
LIDAR_PAD_DIM     = INPUT_DIM - LIDAR_FOURIER_DIM - ECHO_MLP_OUT_DIM  # 18

QUERY_DIM = 3 * POS_DIM    # 195

# LAS classification → FRACTAL 7-class remap
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


# ============================================================================
# Fourier encoding helpers (pure functions, no nn.Module needed in dataset)
# ============================================================================

def _fourier_encode(x: torch.Tensor, num_bands: int, fmax: float) -> torch.Tensor:
    """
    Encode a scalar tensor with sinusoidal Fourier features.

    Matches the convention in fourier.py:
        [sin(f1*pi*x), cos(f1*pi*x), ..., sin(fK*pi*x), cos(fK*pi*x), x]

    Args:
        x:         [...] normalized scalar tensor (any shape).
        num_bands: Number of frequency bands K.
        fmax:      Maximum frequency.

    Returns:
        encoded: [..., 2*num_bands+1]
    """
    bands = torch.linspace(1.0, fmax / 2.0, num_bands,
                           device=x.device, dtype=x.dtype)   # [K]
    x_exp  = x.unsqueeze(-1)                                  # [..., 1]
    scaled = x_exp * bands * pi                               # [..., K]
    return torch.cat([scaled.sin(), scaled.cos(), x_exp], dim=-1)


def _encode_position(coord: torch.Tensor) -> torch.Tensor:
    """Encode a normalized position scalar → [POS_DIM=65]."""
    return _fourier_encode(coord, POS_BANDS, POS_FMAX)


def _encode_value(val: torch.Tensor) -> torch.Tensor:
    """Encode a normalized band value scalar → [VAL_DIM=33]."""
    return _fourier_encode(val, VAL_BANDS, VAL_FMAX)


def _normalize_coords(coords_px: np.ndarray,
                      patch_size_px: int) -> np.ndarray:
    """Map pixel coordinates [0, patch_size_px) to [-1, 1]."""
    return (coords_px / (patch_size_px - 1)) * 2.0 - 1.0


# ============================================================================
# Echo encoding helper
# ============================================================================

def _echo_ab(return_number: np.ndarray,
             number_of_returns: np.ndarray) -> np.ndarray:
    """
    Continuous (a, b) echo encoding.
        a = (r - 1) / t   proportion of pulse already returned (above)
        b = (t - r) / t   proportion of pulse yet to return (below)
    Both in [0, 1). Returns [N, 2].
    """
    r = return_number.astype(np.float32)
    t = np.maximum(number_of_returns.astype(np.float32), 1.0)
    a = (r - 1.0) / t
    b = (t - r)   / t
    return np.stack([a, b], axis=1)    # [N, 2]


# ============================================================================
# Dataset
# ============================================================================

class FractalPerceiverDataset(Dataset):
    """
    FRACTAL dataset for PerceiverIO baseline.

    Outputs flat Fourier-encoded token tensors instead of Atomizer's
    structured token groups. The echo MLP and LIDAR padding live in the
    model (PerceiverFractal), not here.

    __getitem__ returns
    -------------------
    vhr_tokens   : [N_vhr,   VHR_DIM=262]      Fourier-encoded VHR tokens
    vhr_mask     : [N_vhr]   bool              True = masked (padding)
    lidar_tokens : [N_lidar, LIDAR_RAW_DIM=197] Fourier(X,Y,Z) + echo (a,b)
    lidar_mask   : [N_lidar] bool              True = masked (padding)
    queries      : [M, QUERY_DIM=195]          Fourier(X,Y,Z) query tokens
    query_labels : [M]       int64             class label, 255=ignore
    queries_mask : [M]       bool              True = padding query
    patch_id     : str
    image        : [4, 250, 250]  raw ortho for visualization

    Args:
        root_path:          Root containing FRACTAL/ and FRACTAL-IRGB/.
        mode:               'train', 'val', or 'test'.
        max_lidar_points:   Max LIDAR context tokens. Default 16_000.
        max_queries:        Val/test query padding target. Default 32_000.
        valid_patches_file: Optional JSON with per-split patch allow-lists.
        use_augmentation:   D4 + jitter (auto-disabled for val/test).
        sigma_xy_pixels:    LIDAR XY jitter std in pixels. Default 0.25.
        sigma_z_normed:     LIDAR Z jitter std in norm units. Default 0.003.
        eval_full_scene:    Test-only: queries cover ALL LIDAR points.
                            REQUIRES batch_size=1.
    """

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
        max_lidar_points: int = 16_000,
        max_queries: int = 32_000,
        valid_patches_file: str = None,
        use_augmentation: bool = True,
        sigma_xy_pixels: float = 0.25,
        sigma_z_normed:  float = 0.003,
        eval_full_scene: bool = False,
    ):
        super().__init__()
        for lib, ok in [("laspy", HAS_LASPY), ("rasterio", HAS_RASTERIO)]:
            if not ok:
                raise ImportError(f"{lib} required for FRACTAL dataset")

        self.root_path        = root_path
        self.split            = mode
        self.max_lidar_points = max_lidar_points
        self.max_queries      = max_queries

        # ── Full-scene evaluation (test only) ──────────────────────
        self.eval_full_scene = bool(eval_full_scene)
        if self.eval_full_scene and self.split != "test":
            print(f"[FractalPerceiver] WARNING: eval_full_scene=True ignored "
                  f"for split='{self.split}'.")

        # ── Augmentation ────────────────────────────────────────────
        self.augmenter = D4Augmentation(
            enabled=(use_augmentation and self.split == "train"),
            p_flip_h=0.5,
            p_flip_v=0.5,
        )
        self.sigma_xy_pixels = float(sigma_xy_pixels)
        self.sigma_z_normed  = float(sigma_z_normed)

        if self.augmenter.enabled:
            print(f"[FractalPerceiver] D4 + jitter ENABLED "
                  f"(sigma_xy={self.sigma_xy_pixels}px, "
                  f"sigma_z={self.sigma_z_normed})")
        else:
            print(f"[FractalPerceiver] Augmentation DISABLED "
                  f"(split={self.split})")

        self._collect_patches(valid_patches_file)

        print(f"[FractalPerceiver] {len(self.patch_rows)} patches, "
              f"split='{self.split}'")
        print(f"[FractalPerceiver] Dimensions: "
              f"input_dim={INPUT_DIM} (VHR={VHR_DIM}, "
              f"LIDAR raw={LIDAR_RAW_DIM} -> {INPUT_DIM} after echo+pad), "
              f"query_dim={QUERY_DIM}, lidar_pad={LIDAR_PAD_DIM}")
        if self.eval_full_scene and self.split == "test":
            print(f"[FractalPerceiver] eval_full_scene=True: "
                  f"REQUIRES batch_size=1")

    # =========================================================================
    # PATCH COLLECTION (identical logic to FractalDataset)
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

        print(f"[FractalPerceiver] Indexing ortho files under {irgb_root}...")
        ortho_index = {}
        for ext in ("*.tiff", "*.tif"):
            for op in irgb_root.rglob(ext):
                ortho_index[op.stem] = op
        print(f"[FractalPerceiver]   indexed {len(ortho_index):,} ortho files")

        valid_set = None
        if valid_patches_file is not None and os.path.exists(valid_patches_file):
            with open(valid_patches_file) as f:
                valid_data = json.load(f)
            split_key = {"train": "train", "val": "val",
                         "validation": "val", "test": "test"}[self.split]
            valid_set = set(valid_data.get(split_key, []))
            print(f"[FractalPerceiver] Valid-patch filter: "
                  f"{len(valid_set)} patches for split={self.split}")

        self.patch_rows     = []
        missing_orthos      = 0
        skipped_invalid     = 0
        for laz_path in sorted(laz_root.rglob("*.laz")):
            patch_id   = laz_path.stem
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

        if missing_orthos > 0:
            print(f"[FractalPerceiver] WARNING: {missing_orthos} LAZ files "
                  f"had no matching ortho — skipped.")
        if skipped_invalid > 0:
            print(f"[FractalPerceiver] Filtered out {skipped_invalid} patches.")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]

        # ── Augmentation transform (identity for val/test) ─────────
        aug = self.augmenter.sample(index=index)
        full_scene_active = (self.eval_full_scene and self.split == "test")

        # ══════════════════════════════════════════════════════════════
        # 1. LOAD AND PREPROCESS LIDAR
        # ══════════════════════════════════════════════════════════════

        las          = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]
        if n_points_raw < self.MIN_POINTS:
            return self.__getitem__((index + 1) % len(self))

        # Patch-local pixel coordinates
        x_min   = float(las.x.min())
        y_max   = float(las.y.max())
        lidar_x = (np.asarray(las.x) - x_min) / self.VHR_RESOLUTION
        lidar_y = (y_max - np.asarray(las.y)) / self.VHR_RESOLUTION
        lidar_x = np.clip(lidar_x, 0.0, self.PATCH_SIZE_PX - 1e-3)
        lidar_y = np.clip(lidar_y, 0.0, self.PATCH_SIZE_PX - 1e-3)

        # Echo
        return_number     = np.asarray(las.return_number,     dtype=np.int64)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

        # Labels
        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]

        # Z normalization (ground-relative)
        z_raw        = np.asarray(las.z, dtype=np.float32)
        ground_mask  = (labels == 1)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z_raw[ground_mask]))
        else:
            local_ground = float(np.percentile(z_raw, 5.0))
        z_rel  = z_raw - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        # ── D4 augmentation on LIDAR XY ────────────────────────────
        xy_stacked = np.stack([lidar_x, lidar_y], axis=1).astype(np.float32)
        if not aug.is_identity:
            xy_stacked = self.augmenter.apply_to_xy(
                xy_stacked, aug, patch_size_px=self.PATCH_SIZE_PX
            )

        # ── LIDAR XY jitter ─────────────────────────────────────────
        if self.augmenter.enabled and (self.sigma_xy_pixels > 0
                                       or self.sigma_z_normed > 0):
            jitter_seed = (index * 2147483647) ^ 0x9E3779B9
            xy_stacked, z_norm = self.augmenter.apply_jitter(
                xy_stacked, z=z_norm,
                sigma_xy=self.sigma_xy_pixels,
                sigma_z=self.sigma_z_normed,
                seed=jitter_seed,
            )

        xy_stacked = np.clip(
            xy_stacked, 0.0, self.PATCH_SIZE_PX - 1e-3
        ).astype(np.float32)
        lidar_x = xy_stacked[:, 0]
        lidar_y = xy_stacked[:, 1]

        # ══════════════════════════════════════════════════════════════
        # 2. CONTEXT / QUERY SPLIT
        # ══════════════════════════════════════════════════════════════

        if full_scene_active:
            # Full arrays kept for queries
            full_lidar_x = lidar_x.copy()
            full_lidar_y = lidar_y.copy()
            full_z_norm  = z_norm.copy()
            full_labels  = labels.copy()
            # Subsample for context only
            if (self.max_lidar_points is not None
                    and n_points_raw > self.max_lidar_points):
                rng = np.random.default_rng(
                    seed=hash(row["patch_id"]) & 0xFFFFFFFF
                )
                sel = rng.choice(n_points_raw, size=self.max_lidar_points,
                                 replace=False)
                ctx_x  = lidar_x[sel];  ctx_y = lidar_y[sel]
                ctx_z  = z_norm[sel];   ctx_labels = labels[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]
            else:
                ctx_x = lidar_x; ctx_y = lidar_y
                ctx_z = z_norm;  ctx_labels = labels
        else:
            # Standard: subsample once, context == queries
            if (self.max_lidar_points is not None
                    and n_points_raw > self.max_lidar_points):
                rng = np.random.default_rng(
                    seed=hash(row["patch_id"]) & 0xFFFFFFFF
                    if self.split != "train" else None
                )
                sel = rng.choice(n_points_raw, size=self.max_lidar_points,
                                 replace=False)
                lidar_x = lidar_x[sel]; lidar_y = lidar_y[sel]
                z_norm  = z_norm[sel];  labels  = labels[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]

            ctx_x  = lidar_x; ctx_y = lidar_y
            ctx_z  = z_norm;  ctx_labels = labels
            full_lidar_x = lidar_x; full_lidar_y = lidar_y
            full_z_norm  = z_norm;  full_labels  = labels

        # ══════════════════════════════════════════════════════════════
        # 3. LOAD AND PREPROCESS VHR ORTHO
        # ══════════════════════════════════════════════════════════════

        with rasterio.open(row["ortho_path"]) as src:
            ortho = src.read().astype(np.float32)    # [4, 250, 250] NIR/R/G/B
        ortho = torch.from_numpy(ortho)
        ortho = (ortho / 127.5) - 1.0
        ortho = torch.clamp(ortho, -10.0, 10.0)
        ortho = torch.nan_to_num(ortho, nan=0.0, posinf=10.0, neginf=-10.0)

        # Apply same D4 transform to ortho
        if not aug.is_identity:
            ortho = self.augmenter.apply(ortho, aug)

        # ══════════════════════════════════════════════════════════════
        # 4. BUILD VHR TOKENS  [N_vhr, VHR_DIM=262]
        #    One token per pixel combining all 4 bands:
        #    [Fourier(NIR), Fourier(R), Fourier(G), Fourier(B),
        #     Fourier(Y), Fourier(X)]
        #    = 4*33 + 2*65 = 132 + 130 = 262
        #    N_vhr = H*W = 250*250 = 62_500
        # ══════════════════════════════════════════════════════════════

        H, W = self.PATCH_SIZE_PX, self.PATCH_SIZE_PX

        # Normalized pixel coordinates [-1, 1]
        ys = torch.linspace(-1.0, 1.0, H, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, W, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
        grid_y_flat = grid_y.reshape(-1)   # [H*W]
        grid_x_flat = grid_x.reshape(-1)   # [H*W]

        # Fourier-encode positions: [H*W, 65] each
        pos_y_enc = _encode_position(grid_y_flat)   # [H*W, 65]
        pos_x_enc = _encode_position(grid_x_flat)   # [H*W, 65]

        # Fourier-encode all 4 bands independently: [H*W, 33] each
        band_encs = []
        for b in range(4):   # NIR=0, R=1, G=2, B=3
            band_vals = ortho[b].reshape(-1)            # [H*W]
            band_encs.append(_encode_value(band_vals))  # [H*W, 33]

        # One token per pixel: all 4 band encodings + position
        # [Fourier(NIR), Fourier(R), Fourier(G), Fourier(B), Fourier(Y), Fourier(X)]
        # shapes: 33 + 33 + 33 + 33 + 65 + 65 = 262
        vhr_tokens = torch.cat(
            [*band_encs, pos_y_enc, pos_x_enc], dim=-1
        )   # [H*W, 262]

        N_vhr    = vhr_tokens.shape[0]   # 250*250 = 62_500
        vhr_mask = torch.zeros(N_vhr, dtype=torch.bool)

        # ══════════════════════════════════════════════════════════════
        # 5. BUILD LIDAR CONTEXT TOKENS  [N_lidar, LIDAR_RAW_DIM=197]
        #    Fourier(X, Y, Z) + raw echo scalars (a, b)
        #    Echo MLP + padding (to reach 262) live in the model.
        # ══════════════════════════════════════════════════════════════

        # Normalize LIDAR XY to [-1, 1]
        ctx_x_norm = _normalize_coords(ctx_x, self.PATCH_SIZE_PX)
        ctx_y_norm = _normalize_coords(ctx_y, self.PATCH_SIZE_PX)
        # Z normalized by Z_GROUND_REL_SCALE; clip to [-1, 1] for Fourier stability
        ctx_z_clipped = np.clip(ctx_z, -1.0, 1.0)

        ctx_x_t = torch.from_numpy(ctx_x_norm.astype(np.float32))
        ctx_y_t = torch.from_numpy(ctx_y_norm.astype(np.float32))
        ctx_z_t = torch.from_numpy(ctx_z_clipped.astype(np.float32))

        lidar_fourier = torch.cat([
            _encode_position(ctx_x_t),   # [N_lidar, 65]
            _encode_position(ctx_y_t),   # [N_lidar, 65]
            _encode_position(ctx_z_t),   # [N_lidar, 65]
        ], dim=-1)                        # [N_lidar, 195]

        echo_ab_np = _echo_ab(return_number, number_of_returns)
        echo_ab_t  = torch.from_numpy(echo_ab_np)   # [N_lidar, 2]

        lidar_tokens = torch.cat([lidar_fourier, echo_ab_t], dim=-1)
        # [N_lidar, 197]

        N_lidar    = lidar_tokens.shape[0]
        lidar_mask = torch.zeros(N_lidar, dtype=torch.bool)

        # ── Pad LIDAR context to fixed size ─────────────────────────
        if (self.max_lidar_points is not None
                and N_lidar < self.max_lidar_points):
            n_pad        = self.max_lidar_points - N_lidar
            pad          = torch.zeros(n_pad, LIDAR_RAW_DIM)
            lidar_tokens = torch.cat([lidar_tokens, pad], dim=0)
            lidar_mask   = torch.cat([
                lidar_mask,
                torch.ones(n_pad, dtype=torch.bool),
            ])

        # ══════════════════════════════════════════════════════════════
        # 6. BUILD QUERY TOKENS  [M, QUERY_DIM=195]  +  labels [M]
        # ══════════════════════════════════════════════════════════════

        q_x_norm = _normalize_coords(full_lidar_x, self.PATCH_SIZE_PX)
        q_y_norm = _normalize_coords(full_lidar_y, self.PATCH_SIZE_PX)
        q_z_clip = np.clip(full_z_norm, -1.0, 1.0)

        q_x_t = torch.from_numpy(q_x_norm.astype(np.float32))
        q_y_t = torch.from_numpy(q_y_norm.astype(np.float32))
        q_z_t = torch.from_numpy(q_z_clip.astype(np.float32))

        queries = torch.cat([
            _encode_position(q_x_t),   # [N_q, 65]
            _encode_position(q_y_t),   # [N_q, 65]
            _encode_position(q_z_t),   # [N_q, 65]
        ], dim=-1)                      # [N_q, 195]

        query_labels = torch.from_numpy(
            full_labels.astype(np.int64)
        )   # [N_q]

        # ── Subsample queries during training ────────────────────────
        if self.split == "train":
            N_q = queries.shape[0]
            if N_q > MAX_QUERIES_TRAIN:
                # Prioritize valid (non-ignored) labels
                valid_mask  = (query_labels != IGNORE_INDEX)
                valid_idx   = valid_mask.nonzero(as_tuple=False).squeeze(1)
                invalid_idx = (~valid_mask).nonzero(as_tuple=False).squeeze(1)
                n_valid     = valid_idx.shape[0]
                if n_valid >= MAX_QUERIES_TRAIN:
                    perm    = torch.randperm(n_valid)[:MAX_QUERIES_TRAIN]
                    sel_idx = valid_idx[perm]
                else:
                    n_fill  = MAX_QUERIES_TRAIN - n_valid
                    perm    = torch.randperm(invalid_idx.shape[0])[:n_fill]
                    sel_idx = torch.cat([valid_idx, invalid_idx[perm]])
                queries      = queries[sel_idx]
                query_labels = query_labels[sel_idx]

        # ── Pad queries to fixed size for val/test batching ──────────
        if full_scene_active:
            queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)
        else:
            target_n = (MAX_QUERIES_TRAIN if self.split == "train"
                        else (self.max_lidar_points
                              if self.max_lidar_points is not None
                              else queries.shape[0]))
            N_q = queries.shape[0]
            if N_q < target_n:
                n_pad        = target_n - N_q
                q_pad        = torch.zeros(n_pad, QUERY_DIM)
                lbl_pad      = torch.full((n_pad,), IGNORE_INDEX,
                                          dtype=torch.long)
                queries      = torch.cat([queries,      q_pad],  dim=0)
                query_labels = torch.cat([query_labels, lbl_pad], dim=0)
                queries_mask = torch.cat([
                    torch.zeros(N_q,   dtype=torch.bool),
                    torch.ones(n_pad,  dtype=torch.bool),
                ])
            elif N_q > target_n:
                queries      = queries[:target_n]
                query_labels = query_labels[:target_n]
                queries_mask = torch.zeros(target_n, dtype=torch.bool)
            else:
                queries_mask = torch.zeros(target_n, dtype=torch.bool)

        return {
            # ── Input tokens ─────────────────────────────────────
            "vhr_tokens":    vhr_tokens,      # [N_vhr,   VHR_DIM=262]
            "vhr_mask":      vhr_mask,         # [N_vhr]   bool
            "lidar_tokens":  lidar_tokens,     # [N_lidar, LIDAR_RAW_DIM=197]
            "lidar_mask":    lidar_mask,        # [N_lidar] bool

            # ── Query tokens ──────────────────────────────────────
            "queries":       queries,          # [M, QUERY_DIM=195]
            "query_labels":  query_labels,     # [M]       int64
            "queries_mask":  queries_mask,     # [M]       bool

            # ── Metadata ──────────────────────────────────────────
            "patch_id":      row["patch_id"],
            "image":         ortho,            # [4, 250, 250] for visualization
        }
