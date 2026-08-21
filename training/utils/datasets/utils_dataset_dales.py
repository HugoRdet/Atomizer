"""
DALES Atomizer Dataset (LIDAR-only, with jitter augmentation)
==================================================================

Adapted from utils_dataset_fractal.py. Same conventions (token layout,
echo encoding, z-normalization, TokenBuilder usage) but:

  1. NO VHR / ortho branch -- DALES ships LIDAR only. `groups` contains a
     single modality (LIDAR-as-elevation) instead of VHR+LIDAR concatenated.
  2. NO D4 spatial flip/rotation augmentation on an image raster, since
     there's no raster to keep aligned with -- but we KEEP D4 on the point
     cloud (x, y) itself, since square patches are still rotation/flip
     invariant. Only Gaussian XY/Z jitter changes semantics vs. FRACTAL:
     here it's applied unconditionally to all points (there's no VHR to
     leave un-jittered).
  3. Patches are expected to already be pre-tiled fixed-size .laz files
     (see tile_dales.py) -- DALES scenes are far too large to tile lazily
     inside __getitem__.
  4. Label remap covers DALES' 8 semantic classes instead of FRACTAL's 7.

If you don't need full-scene eval / vhr-drop-bands equivalents, those flags
are simply absent here (there's no VHR to drop bands from).

NOTE: class-balanced query sampling (sqrt-inverse-frequency weighted) has
been REMOVED -- train-split query selection now uses the generic
TokenBuilder.subsample_queries (uniform among valid queries), same as
val/test. The k-NN decoder-skip cascade upgrade has also been left OUT of
this version (query_token_idx stays the simple 1-atom identity/inverse
mapping) -- both were deliberately deferred, not lost; ask for either to
be re-added when ready.

==============================================================================
IMPORTANT -- verify against your actual DALES release before training:

DALES semantic segmentation classes (official release) are commonly:
    0 : Unknown / unclassified
    1 : Ground
    2 : Vegetation
    3 : Cars
    4 : Trucks
    5 : Power lines
    6 : Fences
    7 : Poles
    8 : Buildings

Some DALES distributions (e.g. the "DALES Objects" instance-seg release)
use different integer codes. Check `np.unique(las.classification)` on a
few of your actual files before trusting DALES_TO_ATOMIZER below -- update
the LUT if your codes differ.
==============================================================================
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

from .token_grouping import *
from .token_builder import TokenBuilder
from .augmentations import D4Augmentation, D4Transform


# ============================================================================
# Token column indices (must match TokenProcessor / TokenBuilder convention)
# ============================================================================

TOKEN_VALUE_IDX    = 0   # reflectance / z_norm value
TOKEN_SPECTRAL_IDX = 3   # spectral_idx (lookup into wavelength/bandwidth table)


# ============================================================================
# DALES code -> Atomizer 8-class label remap
# ============================================================================
# VERIFY against your files (see module docstring) before trusting this.

DALES_TO_ATOMIZER = {
    1: 0,   # ground          -> ground
    2: 1,   # vegetation      -> vegetation
    3: 2,   # cars            -> cars
    4: 3,   # trucks          -> trucks
    5: 4,   # power lines     -> power_lines
    6: 5,   # fences          -> fences
    7: 6,   # poles           -> poles
    8: 7,   # buildings       -> buildings
    # 0 (unknown/unclassified) intentionally omitted -> IGNORE_INDEX (255).
}


def _build_remap_lut(mapping: dict, num_codes: int = 256,
                      ignore: int = 255) -> np.ndarray:
    """Build a 1D LUT for fast raw-code -> Atomizer-label remap."""
    lut = np.full(num_codes, ignore, dtype=np.int64)
    for raw_code, label in mapping.items():
        lut[raw_code] = label
    return lut


REMAP_LUT = _build_remap_lut(DALES_TO_ATOMIZER)


# ============================================================================
# Helper: resolve the ELEVATION spectral_idx (unchanged from FRACTAL)
# ============================================================================

def _resolve_elevation_spectral_idx(lookup) -> int:
    if hasattr(lookup, "abstract_channel_indices"):
        idx = lookup.abstract_channel_indices.get("ELEVATION")
        if idx is not None:
            return int(idx)
    if hasattr(lookup, "get_abstract_channel_idx"):
        try:
            return int(lookup.get_abstract_channel_idx("ELEVATION"))
        except Exception:
            pass
    candidates_table_wave = [
        ("ELEVATION", "ELEVATION"),
        (-3, -3), (-4, -4), (-5, -5), (-6, -6),
    ]
    if hasattr(lookup, "table_wave"):
        for key in candidates_table_wave:
            if key in lookup.table_wave:
                return int(lookup.table_wave[key])
    if hasattr(lookup, "get_spectral_idx_by_name"):
        try:
            return int(lookup.get_spectral_idx_by_name("ELEVATION"))
        except Exception:
            pass
    raise RuntimeError(
        "[DALES] Could not resolve spectral_idx for 'ELEVATION'. "
        "Register it via lookup_table.register_abstract_channel('ELEVATION') "
        "before constructing the dataset."
    )


def _resolve_intensity_spectral_idx(lookup) -> int:
    """DEPRECATED / no longer called: intensity is no longer a separate
    LIDAR channel. It's now folded into column 6 (see DalesTokenProcessor
    and build_sparse_tokens' intensity_override param) rather than being
    its own spectral_idx-tagged channel. Left here only in case a future
    variant wants to go back to the 2-channel design; _setup_spectral_idx
    no longer calls this.
    """
    if hasattr(lookup, "abstract_channel_indices"):
        idx = lookup.abstract_channel_indices.get("INTENSITY")
        if idx is not None:
            return int(idx)
    if hasattr(lookup, "get_abstract_channel_idx"):
        try:
            return int(lookup.get_abstract_channel_idx("INTENSITY"))
        except Exception:
            pass
    candidates_table_wave = [
        ("INTENSITY", "INTENSITY"),
        (-7, -7), (-8, -8),
    ]
    if hasattr(lookup, "table_wave"):
        for key in candidates_table_wave:
            if key in lookup.table_wave:
                return int(lookup.table_wave[key])
    if hasattr(lookup, "get_spectral_idx_by_name"):
        try:
            return int(lookup.get_spectral_idx_by_name("INTENSITY"))
        except Exception:
            pass
    raise RuntimeError(
        "[DALES] Could not resolve spectral_idx for 'INTENSITY'. "
        "Register it via lookup_table.register_abstract_channel('INTENSITY') "
        "before constructing the dataset."
    )


def _normalize_intensity(intensity: "np.ndarray", p_lo: float = 1.0,
                          p_hi: float = 99.0) -> "np.ndarray":
    """Normalize raw LIDAR intensity to roughly [0, 1] via per-scene robust
    percentile scaling, then clip.

    WHY per-scene, not a fixed global constant: raw intensity is NOT
    radiometrically calibrated across flight lines/sensors (see prior
    discussion) -- a fixed divisor (e.g. /255 or /65535) would be wrong if
    DALES' effective range doesn't actually span that, and would make
    intensity incomparable across scenes with different flight
    conditions. Percentile normalization per scene is a reasonable
    default; VERIFY against your actual intensity value ranges (inspect
    a few files' `las.intensity.min()/.max()/.mean()`) before trusting
    this blindly -- if DALES' intensity turns out to already be
    well-behaved/consistent across files, a fixed normalization constant
    might be preferable for consistency with any pretrained expectations.
    """
    if intensity.size == 0:
        return intensity.astype(np.float32)
    lo = np.percentile(intensity, p_lo)
    hi = np.percentile(intensity, p_hi)
    if hi <= lo:
        return np.zeros_like(intensity, dtype=np.float32)
    norm = (intensity.astype(np.float32) - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


# ============================================================================
# DALES Dataset
# ============================================================================

class DalesDataset(Dataset):
    """
    DALES semantic segmentation, Atomizer format, LIDAR-only.

    Expects `root_path` to contain pre-tiled patches produced by
    tile_dales.py, laid out as:

        root_path/DALES/<split_dir>/*.laz

    Args:
        use_augmentation:  Master switch for D4 + jitter. Auto-disabled for
                            val/test regardless. Default True.
        sigma_xy_pixels:    Std dev of LIDAR XY jitter, in PIXEL units
                            (PIXEL_RESOLUTION m/px). Default 0.25.
        sigma_z_normed:     Std dev of LIDAR Z jitter, in NORMALIZED units.
                            Default 0.003.
        eval_full_scene:    If True (test mode only), queries cover ALL
                            LIDAR points in a patch so metrics reflect the
                            full patch. Context tokens stay subsampled.
                            REQUIRES batch_size=1.
    """

    # DALES is airborne. Patches are FIXED SIZE -- see tile_dales.py's
    # module docstring for why: Atomiser_Senflood_Skip computes ONE latent
    # grid PER BATCH (not per sample), so every patch must share
    # byte-identical geometry for that batch-shared assumption to hold.
    # PATCH_SIZE_M MUST match whatever tile_dales.py used to produce the
    # tiled patches, and PIXEL_RESOLUTION is purely for TokenBuilder
    # coordinate-frame compatibility (no raster exists at this
    # "resolution").
    PIXEL_RESOLUTION = 0.2      # meters/px equivalent, same convention as FRACTAL
    PATCH_SIZE_M     = 50.0     # MUST match tile_dales.py --patch_size_m
    PATCH_SIZE_PX    = int(PATCH_SIZE_M / PIXEL_RESOLUTION)   # 250

    NUM_CLASSES  = 8
    IGNORE_INDEX = 255
    TIME_IDX_NA  = -1

    MIN_POINTS = 1000

    # Ground-relative Z normalization, same convention as FRACTAL. DALES
    # scenes include taller structures (power line pylons, tall buildings)
    # so you may want to widen Z_GROUND_REL_HI -- check your data's z-range
    # relative to ground before trusting these defaults.
    Z_GROUND_REL_LO    = -15.0
    Z_GROUND_REL_HI    = 30.0
    Z_GROUND_REL_SCALE = 15.0
    GROUND_MEDIAN_MIN_PTS = 50

    DALES_CLASSES = [
        "ground", "vegetation", "cars", "trucks",
        "power_lines", "fences", "poles", "buildings",
    ]

    SPLIT_DIRS = {
        "train":      "train",
        "val":        "val",
        "test":       "test",
        "validation": "val",
    }

    def __init__(
        self,
        root_path: str = "./data",
        mode: str = "train",
        dataset_config=None,
        config_model=None,
        look_up=None,
        max_lidar_points: int = 256_000,
        max_queries: int = 256_000,
        valid_patches_file: str = None,
        use_augmentation: bool = True,
        sigma_xy_pixels: float = 0.25,
        sigma_z_normed:  float = 0.003,
        eval_full_scene: bool = False,
    ):
        # NOTE on defaults: DALES averages ~49 pts/m^2, so a 50x50m tiled
        # patch (PATCH_SIZE_M) contains ~122,500 points on average.
        # max_lidar_points=256_000 therefore means MOST patches see NO
        # subsampling at all -- only unusually dense patches (dense
        # vegetation/building clusters) get trimmed. This is a deliberate
        # choice for an H100 80GB with flexible batch size; if you hit
        # OOM, lower this before shrinking batch size further, since the
        # padded [max_lidar_points, 8] tensors dominate per-sample memory.
        super().__init__()
        if not HAS_LASPY:
            raise ImportError("laspy required for DALES dataset")

        self.root_path        = root_path
        self.split            = mode
        self.look_up          = look_up
        self.config_model     = config_model
        self.dataset_config   = dataset_config
        self.max_lidar_points = max_lidar_points
        self.max_queries      = max_queries

        self.eval_full_scene = bool(eval_full_scene)
        if self.eval_full_scene and self.split != "test":
            print(f"[DALES] WARNING: eval_full_scene=True with split="
                  f"'{self.split}' -- full-scene queries only take effect "
                  f"during test evaluation. Ignored for this split.")

        # -- Augmentation config -----------------------------------
        # D4 still applies to the point cloud (x, y) -- patches are square
        # and flip/rotation invariant even without a raster to align to.
        self.augmenter = D4Augmentation(
            enabled=(use_augmentation and self.split == "train"),
            p_flip_h=0.5,
            p_flip_v=0.5,
        )
        self.sigma_xy_pixels = float(sigma_xy_pixels)
        self.sigma_z_normed  = float(sigma_z_normed)
        if self.augmenter.enabled:
            print(f"[DALES] D4 + jitter ENABLED "
                  f"(sigma_xy={self.sigma_xy_pixels}px, "
                  f"sigma_z={self.sigma_z_normed} normed)")
        else:
            print(f"[DALES] Augmentation DISABLED "
                  f"(split={self.split}, use_augmentation={use_augmentation})")

        self.token_builder = TokenBuilder(look_up)
        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"].get(
            "max_tokens_reconstruction", max_queries
        )
        # Reuse the same "resolution" bucket convention as FRACTAL LIDAR
        # tokens -- this is a bookkeeping index into the lookup table, not a
        # physical pixel size claim.
        self.resolution_idx = look_up.get_resolution_idx(self.PIXEL_RESOLUTION)

        self._setup_spectral_idx()
        self._collect_patches(valid_patches_file)

        print(f"[DALES] Loaded {len(self.patch_rows)} patches, "
              f"split='{self.split}'")
        print(f"[DALES] Modalities: LIDAR-only "
              f"(elev@{self.PIXEL_RESOLUTION}m equiv, "
              f"<={self.max_lidar_points if self.max_lidar_points else 'inf'} pts)")
        if self.eval_full_scene and self.split == "test":
            print(f"[DALES] eval_full_scene=True: queries cover ALL points "
                  f"per patch (REQUIRES batch_size=1 in DataLoader)")

    # =========================================================================
    # INITIALIZATION HELPERS
    # =========================================================================

    def _setup_spectral_idx(self):
        self.lidar_spectral_idx = _resolve_elevation_spectral_idx(self.look_up)
        print(f"[DALES] LIDAR spectral_idx (ELEVATION): "
              f"{self.lidar_spectral_idx}")

    def _collect_patches(self, valid_patches_file: str = None):
        split_dir = self.SPLIT_DIRS.get(self.split)
        if split_dir is None:
            raise ValueError(f"Unknown split: {self.split}")
        laz_root = Path(self.root_path) / "DALES" / split_dir
        if not laz_root.exists():
            raise FileNotFoundError(
                f"DALES tiled LAZ root not found: {laz_root}\n"
                f"Did you run tile_dales.py to pre-tile the raw scenes?"
            )

        valid_set = None
        if valid_patches_file is not None and os.path.exists(valid_patches_file):
            with open(valid_patches_file) as f:
                valid_data = json.load(f)
            split_key = {"train": "train", "val": "val",
                         "validation": "val", "test": "test"}[self.split]
            valid_set = set(valid_data.get(split_key, []))
            print(f"[DALES] Loaded valid-patch filter: "
                  f"{len(valid_set)} patches for split={self.split}")

        self.patch_rows = []
        skipped_invalid = 0
        for laz_path in sorted(laz_root.rglob("*.laz")):
            patch_id = laz_path.stem
            if valid_set is not None and patch_id not in valid_set:
                skipped_invalid += 1
                continue
            self.patch_rows.append({
                "patch_id": patch_id,
                "laz_path": str(laz_path),
            })

        if skipped_invalid > 0:
            print(f"[DALES] Filtered out {skipped_invalid} patches via "
                  f"valid-patches list.")
        if not self.patch_rows:
            raise RuntimeError(
                f"[DALES] No patches found under {laz_root}. "
                f"Did tile_dales.py run successfully for this split?"
            )

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]

        aug = self.augmenter.sample(index=index)
        full_scene_active = (self.eval_full_scene and self.split == "test")

        # -- Load LIDAR -----------------------------------------------
        las = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]
        if n_points_raw < self.MIN_POINTS:
            return self.__getitem__((index + 1) % len(self))

        # Patch bounds -> patch-local pixel coords (same convention as
        # FRACTAL; the "pixel" frame here is purely for TokenBuilder
        # compatibility, there's no raster underneath it). Patches are
        # FIXED SIZE (self.PATCH_SIZE_PX, matching tile_dales.py's
        # --patch_size_m) -- origin is still each patch's own (x_min, y_max)
        # so points are correctly placed within the shared-size frame, but
        # the FRAME SIZE ITSELF is the constant, not derived from this
        # patch's actual point extent (which may be smaller than the full
        # tile, e.g. near a scene boundary -- that's fine, those points
        # just don't reach the frame's far edge).
        x_min = float(las.x.min())
        y_max = float(las.y.max())
        patch_size_px = self.PATCH_SIZE_PX

        lidar_x = (np.asarray(las.x) - x_min) / self.PIXEL_RESOLUTION
        lidar_y = (y_max - np.asarray(las.y)) / self.PIXEL_RESOLUTION

        lidar_x = np.clip(lidar_x, 0.0, patch_size_px - 1e-3)
        lidar_y = np.clip(lidar_y, 0.0, patch_size_px - 1e-3)

        # -- Echo info (return_number, number_of_returns) -------------
        return_number     = np.asarray(las.return_number,     dtype=np.int64)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

        # -- Intensity --------------------------------------------------
        intensity_raw = np.asarray(las.intensity, dtype=np.float32)

        # -- Labels (BEFORE z-norm since z-norm uses ground mask) -----
        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]

        # -- Z normalization (ground-relative) -------------------------
        z_raw = np.asarray(las.z, dtype=np.float32)
        ground_mask = (labels == 0)   # 0 == "ground" in DALES_TO_ATOMIZER
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z_raw[ground_mask]))
        else:
            local_ground = float(np.percentile(z_raw, 5.0))
        z_rel  = z_raw - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        # Normalized once per patch (percentile-based, see
        # _normalize_intensity docstring for caveats on cross-scene
        # comparability).
        intensity_norm = _normalize_intensity(intensity_raw)

        # -- Load precomputed token->latent assignment (offline, see
        # precompute_dales_latent_assignment.py) ------------------------
        # variant_idx encoding MUST match precompute's
        # variant_idx_to_params: n_rot*4 + int(flip_h)*2 + int(flip_v).
        variant_idx = aug.n_rot * 4 + int(aug.flip_h) * 2 + int(aug.flip_v)
        assign_path = Path(row["laz_path"]).parent / (
            Path(row["laz_path"]).stem + "_latent_assign.npz"
        )
        with np.load(assign_path) as npz:
            full_assignment = npz["assignment"][variant_idx]  # [n_points_raw]
        assert full_assignment.shape[0] == n_points_raw, (
            f"Precomputed assignment for {assign_path.name} has "
            f"{full_assignment.shape[0]} points, but this patch has "
            f"{n_points_raw} -- re-run precompute_dales_latent_assignment.py "
            f"(tiling/precompute out of sync, or points changed since "
            f"precompute ran)."
        )

        # -- Apply D4 to LIDAR (x, y) -----------------------------------
        xy_stacked = np.stack([lidar_x, lidar_y], axis=1).astype(np.float32)
        if not aug.is_identity:
            xy_stacked = self.augmenter.apply_to_xy(
                xy_stacked, aug, patch_size_px=patch_size_px
            )

        # -- Apply jitter (always on, no VHR to preserve exactness for) --
        if self.augmenter.enabled and (self.sigma_xy_pixels > 0
                                       or self.sigma_z_normed > 0):
            jitter_seed = (index * 2147483647) ^ 0x9E3779B9
            xy_stacked, z_norm = self.augmenter.apply_jitter(
                xy_stacked,
                z=z_norm,
                sigma_xy=self.sigma_xy_pixels,
                sigma_z=self.sigma_z_normed,
                seed=jitter_seed,
            )

        xy_stacked = np.clip(
            xy_stacked, 0.0, patch_size_px - 1e-3
        ).astype(np.float32)
        lidar_x = xy_stacked[:, 0]
        lidar_y = xy_stacked[:, 1]

        # ===================================================================
        # CONTEXT vs QUERY SPLIT (full-scene-eval logic, unchanged logic
        # from FRACTAL -- just no VHR alongside it)
        # ===================================================================
        if full_scene_active:
            full_lidar_x = lidar_x.copy()
            full_lidar_y = lidar_y.copy()
            full_z_norm  = z_norm.copy()
            full_labels  = labels.copy()

            if (self.max_lidar_points is not None
                    and n_points_raw > self.max_lidar_points):
                rng = np.random.default_rng(
                    seed=hash(row["patch_id"]) & 0xFFFFFFFF
                )
                sel = rng.choice(n_points_raw, size=self.max_lidar_points,
                                 replace=False)
                ctx_lidar_x = lidar_x[sel]
                ctx_lidar_y = lidar_y[sel]
                ctx_z_norm  = z_norm[sel]
                ctx_intensity = intensity_norm[sel]
                ctx_labels  = labels[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]
            else:
                sel = None
                ctx_lidar_x = lidar_x
                ctx_lidar_y = lidar_y
                ctx_z_norm  = z_norm
                ctx_intensity = intensity_norm
                ctx_labels  = labels

            n_real_lidar_ctx     = ctx_lidar_x.shape[0]
            n_real_lidar_queries = full_lidar_x.shape[0]
        else:
            if (self.max_lidar_points is not None
                    and n_points_raw > self.max_lidar_points):
                rng = np.random.default_rng(
                    seed=hash(row["patch_id"]) & 0xFFFFFFFF
                    if self.split != "train" else None
                )
                sel = rng.choice(n_points_raw, size=self.max_lidar_points,
                                 replace=False)
                lidar_x = lidar_x[sel]
                lidar_y = lidar_y[sel]
                z_norm  = z_norm[sel]
                intensity_norm = intensity_norm[sel]
                labels  = labels[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]
            else:
                sel = None

            ctx_lidar_x  = lidar_x
            ctx_lidar_y  = lidar_y
            ctx_z_norm   = z_norm
            ctx_intensity = intensity_norm
            ctx_labels   = labels
            full_lidar_x = lidar_x
            full_lidar_y = lidar_y
            full_z_norm  = z_norm
            full_labels  = labels
            n_real_lidar_ctx     = ctx_lidar_x.shape[0]
            n_real_lidar_queries = n_real_lidar_ctx

        # -- Gather precomputed assignment by the SAME sel indices used for
        # elevation/intensity/labels context subsampling -- sel=None means
        # every point was kept (no subsampling), so the assignment is used
        # as-is (already in the ctx point order, since ctx == full there).
        ctx_assignment = (full_assignment if sel is None
                          else full_assignment[sel])

        # -- Build context_map for the decoder-skip cascade ----------------
        # For each QUERY row (in the order build_sparse_queries will
        # produce -- i.e. same order as positions_query/full_lidar_x), the
        # index into the CONTEXT (ctx) token array where the SAME point
        # lives, or -1 if that point isn't part of the (possibly
        # subsampled) context. Single-atom identity/inverse mapping (the
        # k-NN upgrade has been deliberately deferred, see module docstring).
        if full_scene_active:
            context_map = np.full(n_points_raw, -1, dtype=np.int64)
            if sel is not None:
                context_map[sel] = np.arange(sel.shape[0], dtype=np.int64)
            else:
                context_map[:] = np.arange(n_points_raw, dtype=np.int64)
        else:
            context_map = np.arange(n_real_lidar_ctx, dtype=np.int64)

        positions_ctx_lidar = torch.from_numpy(
            np.stack([ctx_lidar_x, ctx_lidar_y], axis=1)).float()
        # Single channel now: col 0 = elevation only (z_norm). Intensity is
        # NOT a second channel -- it rides in column 6 (normally
        # resolution_idx, constant/uninformative for LIDAR) via
        # intensity_override, routed by DalesTokenProcessor.
        values_ctx_lidar = torch.from_numpy(ctx_z_norm).float()
        labels_ctx_lidar    = torch.from_numpy(ctx_labels.astype(np.int64))
        intensity_override_ctx = torch.from_numpy(ctx_intensity).float()

        positions_query = torch.from_numpy(
            np.stack([full_lidar_x, full_lidar_y], axis=1)).float()
        # Query target value stays elevation-only (segmentation label is
        # what's actually supervised; col0 here is only meaningful if/when
        # an elevation-reconstruction auxiliary task reads it, mirroring
        # FRACTAL's convention).
        values_query    = torch.from_numpy(full_z_norm).float()
        labels_query    = torch.from_numpy(full_labels.astype(np.int64))

        # -- Tokenize LIDAR (sparse) -- this is now the ONLY modality -----
        lidar_tokens = self.token_builder.build_sparse_tokens(
            values=values_ctx_lidar,
            positions=positions_ctx_lidar,
            labels=labels_ctx_lidar,
            resolution=self.PIXEL_RESOLUTION,
            spectral_indices=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=patch_size_px,
            time_idx=self.TIME_IDX_NA,
            return_number=return_number,
            number_of_returns=number_of_returns,
            intensity_override=intensity_override_ctx,
        )

        # -- Pad LIDAR context tokens to fixed size ------------------------
        n_lidar_tokens = lidar_tokens.shape[0]
        if (self.max_lidar_points is not None
                and n_lidar_tokens < self.max_lidar_points):
            n_pad = self.max_lidar_points - n_lidar_tokens
            pad = torch.zeros(n_pad, 8)
            pad[:, 4] = self.IGNORE_INDEX
            lidar_tokens = torch.cat([lidar_tokens, pad], dim=0)
            lidar_mask = torch.cat([
                torch.zeros(n_lidar_tokens, dtype=torch.bool),
                torch.ones(n_pad, dtype=torch.bool),
            ])
            # Pad token_latent_assignment in lockstep with lidar_tokens.
            # Padded positions get index 0 -- harmless: those rows are
            # already flagged masked=True above, and GeographicPruningDales
            # gathers the INPUT mask too (not just its own validity mask),
            # so a padded token landing in latent 0's cell just wastes a
            # slot there, it never contributes to the actual computation.
            assign_pad = np.zeros(n_pad, dtype=np.int64)
            token_latent_assignment = np.concatenate([ctx_assignment, assign_pad])
        else:
            lidar_mask = torch.zeros(lidar_tokens.shape[0], dtype=torch.bool)
            token_latent_assignment = ctx_assignment

        token_latent_assignment = torch.from_numpy(
            token_latent_assignment.astype(np.int64)
        )

        # No VHR to concatenate -- LIDAR tokens ARE the hires group.
        groups = {
            self.PIXEL_RESOLUTION: {
                "tokens": lidar_tokens,
                "mask":   lidar_mask,
                "shape":  (1, patch_size_px, patch_size_px),
            }
        }

        # -- Build queries from FULL positions/labels ----------------------
        queries = self.token_builder.build_sparse_queries(
            positions=positions_query,
            labels=labels_query,
            resolution=self.PIXEL_RESOLUTION,
            first_spectral_idx=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=patch_size_px,
            time_idx=self.TIME_IDX_NA,
        )

        assert queries.shape[0] == values_query.shape[0], (
            f"Query count ({queries.shape[0]}) doesn't match LIDAR point "
            f"count ({values_query.shape[0]})."
        )
        queries[:, 0] = values_query

        # query_token_idx_full aligned with queries' current row order
        # (== context_map, since build_sparse_queries doesn't reorder rows).
        query_token_idx_full = context_map.copy()

        if self.split == "train":
            # Class-balanced sampling REMOVED -- back to the generic
            # method (uniform random among valid queries, same as
            # val/test's implicit behavior). See module docstring.
            queries, kept_indices = self.token_builder.subsample_queries(
                queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
                return_indices=True,
            )
            kept_indices_np = kept_indices.cpu().numpy()
            query_token_idx_full = query_token_idx_full[kept_indices_np]

        # Split into (idx, valid) -- invalid (-1) entries get clamped to 0
        # (harmless dummy -- masked via query_token_valid=False, and
        # _pixel_skip's force-keep guard + "output discarded" handling
        # takes care of the rest).
        query_token_valid_full = (query_token_idx_full >= 0)
        query_token_idx_full   = np.clip(query_token_idx_full, 0, None)

        if full_scene_active:
            queries_mask  = torch.zeros(queries.shape[0], dtype=torch.bool)
            labels_padded = labels_query
            n_real_lidar_for_return = n_real_lidar_queries

            query_token_idx = torch.from_numpy(query_token_idx_full).long().unsqueeze(-1)
            query_token_valid = torch.from_numpy(query_token_valid_full).bool()
        else:
            target_n_queries = (self.max_tokens_reconstruction
                                if self.split == "train"
                                else (self.max_lidar_points
                                      if self.max_lidar_points is not None
                                      else queries.shape[0]))
            n_real_queries = queries.shape[0]
            if n_real_queries < target_n_queries:
                n_pad = target_n_queries - n_real_queries
                qpad = torch.zeros(n_pad, 8)
                qpad[:, 4] = self.IGNORE_INDEX
                queries = torch.cat([queries, qpad], dim=0)
                queries_mask = torch.cat([
                    torch.zeros(n_real_queries, dtype=torch.bool),
                    torch.ones(n_pad, dtype=torch.bool),
                ])
                query_token_idx_full = np.concatenate([
                    query_token_idx_full, np.zeros(n_pad, dtype=np.int64)
                ])
                query_token_valid_full = np.concatenate([
                    query_token_valid_full, np.zeros(n_pad, dtype=bool)
                ])
            elif n_real_queries > target_n_queries:
                queries      = queries[:target_n_queries]
                queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)
                query_token_idx_full   = query_token_idx_full[:target_n_queries]
                query_token_valid_full = query_token_valid_full[:target_n_queries]
            else:
                queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)

            query_token_idx = torch.from_numpy(query_token_idx_full).long().unsqueeze(-1)
            query_token_valid = torch.from_numpy(query_token_valid_full).bool()

            labels_padded = labels_ctx_lidar
            if (self.max_lidar_points is not None
                    and labels_ctx_lidar.shape[0] < self.max_lidar_points):
                n_pad = self.max_lidar_points - labels_ctx_lidar.shape[0]
                label_pad = torch.full((n_pad,), self.IGNORE_INDEX,
                                       dtype=torch.long)
                labels_padded = torch.cat([labels_ctx_lidar, label_pad], dim=0)
            n_real_lidar_for_return = n_real_lidar_ctx

        return {
            "groups":            groups,
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             labels_padded,
            "n_real_lidar":      torch.tensor(n_real_lidar_for_return,
                                              dtype=torch.long),
            "target_resolution": self.PIXEL_RESOLUTION,
            "token_latent_assignment": token_latent_assignment,
            "patch_id":          row["patch_id"],
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }
