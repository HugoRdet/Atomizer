"""
FRACTAL Atomizer Dataset (with D4 + LIDAR jitter augmentation)
==================================================================

Same as the base utils_dataset_fractal.py but with training-time augmentation:

  1. D4 dihedral group (8 rotations/flips) applied consistently to both
     VHR ortho and LIDAR (x, y) coordinates.
  2. Gaussian positional jitter applied to LIDAR (x, y, z_norm) before
     tokenization — simulates sensor noise (~5cm physical XY, ~4.5cm Z).

Augmentation is sampled ONCE per __getitem__ (reproducible via per-
(worker, index) seed) and applied to both modalities. Val/test always
get identity transform + no jitter.

Why augment in pixel space:
  LIDAR (x, y) is already converted to patch-local pixel coordinates in
  [0, 250) for tokenization, matching the VHR raster's coordinate frame.
  Applying D4 in this space means we can use D4Augmentation.apply() for
  the raster and D4Augmentation.apply_to_xy() for points — they share the
  same patch geometry by construction.

Why jitter on LIDAR but NOT VHR:
  - LIDAR sensors have real positional noise (~few cm); jitter simulates
    this and prevents the model from memorizing exact point positions.
  - VHR pixels are at fixed geographic positions — there's no physical
    noise to simulate. Color jitter on VHR (brightness/contrast) is a
    separate kind of augmentation we deliberately skip for now to keep
    spectral fidelity for vegetation/soil discrimination.

Default jitter parameters (tunable via constructor):
  - sigma_xy_pixels:  default 0.25 px @ 0.2m/px = 5cm physical XY noise
  - sigma_z_normed:   default 0.003 in normalized z units = ~4.5cm physical
                      (Z_GROUND_REL_SCALE=15m, 0.003 * 15 = 0.045m)

==============================================================================
NEW FLAGS (added for paper experiments):

  eval_full_scene (bool, default False):
    When True AND split=="test", queries cover ALL LIDAR points in a scene
    (not just the subsampled context subset). Used for end-of-training
    full-scene evaluation to match baselines that evaluate on all points.
    Context tokens are still subsampled to max_lidar_points for GPU memory.

    REQUIRES batch_size=1 at the dataloader level: per-scene query counts
    vary and cannot be batched together. The flag is silently ignored for
    train/val splits — those always pad to a fixed size for batching.

  vhr_drop_bands (list[int] | str | None, default None):
    Modality-dropout at INFERENCE time — mask out a subset of VHR bands
    without retraining. Tokens for dropped bands are still created (so
    the model sees the expected token count) but their reflectance value
    is set to 0 AND they are flagged in the attention mask so they
    contribute nothing to the latents.

    Options:
      None / []                -> no masking (default)              all 4 bands
      [0]                      -> drop NIR              keep [R, G, B] + LIDAR
      [1, 2, 3]                -> drop RGB              keep [NIR]    + LIDAR
      [0, 1, 2, 3]             -> drop all VHR          LIDAR only
      "no_nir" / "rgb_only"    -> [0]
      "no_rgb" / "nir_only"    -> [1, 2, 3]
      "lidar_only"             -> [0, 1, 2, 3]

    This matches the Sen1Floods11 modality-dropout protocol: "Dropped
    bands are replaced by padding tokens (Atomiser) or zeroed channels
    (baselines)." Allows running the same trained checkpoint under
    multiple test-time modality configurations.
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
    print("[Warning] laspy not installed — required for FRACTAL LAZ reading.")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed — required for FRACTAL ortho.")

from .token_grouping import *
from .token_builder import TokenBuilder
from .augmentations import D4Augmentation, D4Transform


# ============================================================================
# Token column indices (must match TokenBuilder / TokenProcessor convention)
# ============================================================================

TOKEN_VALUE_IDX    = 0   # reflectance / z_norm value
TOKEN_SPECTRAL_IDX = 3   # spectral_idx (lookup into wavelength/bandwidth table)


# ============================================================================
# LAS code → FRACTAL 7-class label remap (unchanged)
# ============================================================================

LAS_TO_FRACTAL = {
    1:  0,    # unclassified      -> other
    2:  1,    # ground            -> ground
    3:  2,    # low vegetation    -> vegetation
    4:  2,    # medium vegetation -> vegetation
    5:  2,    # high vegetation   -> vegetation
    6:  3,    # building          -> building
    9:  4,    # water             -> water
    17: 5,    # bridge deck       -> bridge
    64: 6,    # permanent struct  -> permanent_structure
    # 65, 66, 67 intentionally omitted -> IGNORE_INDEX (255) via the LUT.
}


def _build_remap_lut(num_codes: int = 256, ignore: int = 255) -> np.ndarray:
    """Build a 1D LUT for fast LAS → FRACTAL remap. Unmapped codes → ignore."""
    lut = np.full(num_codes, ignore, dtype=np.int64)
    for las_code, fractal_label in LAS_TO_FRACTAL.items():
        lut[las_code] = fractal_label
    return lut


REMAP_LUT = _build_remap_lut()


# ============================================================================
# Helper: resolve the ELEVATION spectral_idx (unchanged)
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
        "[FRACTAL] Could not resolve spectral_idx for 'ELEVATION'. "
        "Register it via lookup_table.register_abstract_channel('ELEVATION') "
        "before constructing the dataset."
    )


# ============================================================================
# Helper: resolve VHR drop-bands spec
# ============================================================================

# Maps name to channel indices into the original 4-band [NIR, R, G, B]
# ortho. These are the bands to DROP (mask) at inference, not the ones to
# keep. Empty list / None means no dropping.
_VHR_DROP_PRESETS = {
    None:           [],
    "none":         [],
    "no_nir":       [0],
    "rgb_only":     [0],            # alias: keep RGB only -> drop NIR
    "no_rgb":       [1, 2, 3],
    "nir_only":     [1, 2, 3],      # alias: keep NIR only -> drop R/G/B
    "lidar_only":   [0, 1, 2, 3],
    "drop_all_vhr": [0, 1, 2, 3],   # alias
}


def _resolve_vhr_drop_bands(spec):
    """Resolve a drop-bands spec to a list of channel indices in [0..3]."""
    if spec is None:
        return []
    if isinstance(spec, str):
        if spec not in _VHR_DROP_PRESETS:
            valid = sorted(k for k in _VHR_DROP_PRESETS.keys() if k is not None)
            raise ValueError(
                f"[FRACTAL] Unknown vhr_drop_bands={spec!r}. "
                f"Valid string options: {valid}, "
                f"or pass a list of indices in 0..3."
            )
        return list(_VHR_DROP_PRESETS[spec])
    # List/tuple of indices
    indices = list(spec)
    for i in indices:
        if not (0 <= int(i) < 4):
            raise ValueError(
                f"[FRACTAL] vhr_drop_bands index {i} out of range "
                f"(must be in 0..3)."
            )
    return [int(i) for i in indices]


# ============================================================================
# FRACTAL Dataset
# ============================================================================

class FractalDataset(Dataset):
    """
    FRACTAL semantic segmentation, Atomizer format, with D4 + jitter augs.

    Args (most unchanged from base version):
        use_augmentation:    Master switch for D4 + jitter. Auto-disabled for
                             val/test regardless. Default True.
        sigma_xy_pixels:     Std dev of LIDAR XY jitter, in PIXEL units
                             (0.2m/px). Default 0.25 ≈ 5cm physical noise.
        sigma_z_normed:      Std dev of LIDAR Z jitter, in NORMALIZED units
                             (z_norm scale = 15m). Default 0.003 ≈ 4.5cm.
        eval_full_scene:     If True (test mode only), queries cover ALL
                             LIDAR points so the metric reflects the full
                             scene. Context tokens stay subsampled.
                             REQUIRES batch_size=1.
        vhr_drop_bands:      Bands to MASK at inference (no retraining).
                             None / [] = no masking (default). See module
                             docstring for the full list of options.
    """

    VHR_RESOLUTION  = 0.2
    PATCH_SIZE_M    = 50.0
    PATCH_SIZE_PX   = 250
    NUM_VHR_BANDS   = 4

    NUM_CLASSES     = 7
    IGNORE_INDEX    = 255
    TIME_IDX_NA     = -1

    MIN_POINTS      = 1000

    Z_GROUND_REL_LO    = -15.0
    Z_GROUND_REL_HI    = 30.0
    Z_GROUND_REL_SCALE = 15.0
    GROUND_MEDIAN_MIN_PTS = 50

    FRACTAL_CLASSES = [
        "other", "ground", "vegetation", "building",
        "water", "bridge", "permanent_structure",
    ]
    VHR_BAND_NAMES = ["NIR", "R", "G", "B"]

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
        dataset_config=None,
        config_model=None,
        look_up=None,
        max_lidar_points: int = 16_000,
        max_queries: int = 32_000,
        valid_patches_file: str = None,
        use_augmentation: bool = True,
        sigma_xy_pixels: float = 0.25,
        sigma_z_normed:  float = 0.003,
        eval_full_scene: bool = False,
        vhr_drop_bands=None,
    ):
        super().__init__()
        for lib, ok in [("laspy", HAS_LASPY), ("rasterio", HAS_RASTERIO)]:
            if not ok:
                raise ImportError(f"{lib} required for FRACTAL dataset")

        self.root_path        = root_path
        self.split            = mode
        self.look_up          = look_up
        self.config_model     = config_model
        self.dataset_config   = dataset_config
        self.max_lidar_points = max_lidar_points
        self.max_queries      = max_queries

        # ── Full-scene evaluation flag ──────────────────────────
        # Only meaningful for test split. For train/val we always pad to
        # max_lidar_points so DataLoader batching works.
        self.eval_full_scene = bool(eval_full_scene)
        if self.eval_full_scene and self.split != "test":
            print(f"[FRACTAL] WARNING: eval_full_scene=True with split="
                  f"'{self.split}' — full-scene queries only take effect "
                  f"during test evaluation. Ignored for this split.")

        # ── VHR band-drop spec (modality dropout at inference) ──
        # List of band indices in 0..3 that should be masked. Default is
        # an empty list (no masking). Resolved here so the tensors built
        # in _setup_band_indices can be used to identify dropped tokens
        # in __getitem__.
        self.vhr_drop_bands_spec = vhr_drop_bands
        self.vhr_drop_bands = _resolve_vhr_drop_bands(vhr_drop_bands)
        if self.vhr_drop_bands and self.split == "train":
            print(f"[FRACTAL] WARNING: vhr_drop_bands={self.vhr_drop_bands} "
                  f"with split='train' — band masking will be applied "
                  f"during training too. This is usually only desired "
                  f"for test-time modality-dropout evaluation.")

        # ── Augmentation config ─────────────────────────────────
        self.augmenter = D4Augmentation(
            enabled=(use_augmentation and self.split == "train"),
            p_flip_h=0.5,
            p_flip_v=0.5,
        )
        self.sigma_xy_pixels = float(sigma_xy_pixels)
        self.sigma_z_normed  = float(sigma_z_normed)
        if self.augmenter.enabled:
            print(f"[FRACTAL] D4 + jitter ENABLED "
                  f"(sigma_xy={self.sigma_xy_pixels}px, "
                  f"sigma_z={self.sigma_z_normed} normed)")
        else:
            print(f"[FRACTAL] Augmentation DISABLED "
                  f"(split={self.split}, use_augmentation={use_augmentation})")

        self.token_builder = TokenBuilder(look_up)
        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"].get(
            "max_tokens_reconstruction", max_queries
        )
        self.resolution_idx = look_up.get_resolution_idx(self.VHR_RESOLUTION)

        self._setup_band_indices()
        self._collect_patches(valid_patches_file)

        # ── Build the set of spectral_idx values to mask ─────────
        # After _setup_band_indices, self.vhr_spectral_indices is a
        # length-4 tensor mapping band index (0..3) to its spectral_idx
        # in the lookup table. We translate vhr_drop_bands (band indices)
        # into the actual spectral_idx values that the tokens will carry.
        # This buffer is then consulted in __getitem__ to identify which
        # VHR tokens to mask.
        if self.vhr_drop_bands:
            dropped_spectral_idxs = [
                int(self.vhr_spectral_indices[bi].item())
                for bi in self.vhr_drop_bands
            ]
            self._dropped_spectral_set = set(dropped_spectral_idxs)
            dropped_names = [self.VHR_BAND_NAMES[i]
                             for i in self.vhr_drop_bands]
            print(f"[FRACTAL] Modality dropout at inference: "
                  f"masking bands {dropped_names} "
                  f"(spectral_idx={dropped_spectral_idxs})")
        else:
            self._dropped_spectral_set = set()

        print(f"[FRACTAL] Loaded {len(self.patch_rows)} patches, "
              f"split='{self.split}'")
        print(f"[FRACTAL] Modalities: "
              f"VHR({self.NUM_VHR_BANDS}ch@{self.VHR_RESOLUTION}m) + "
              f"LIDAR(elev@{self.VHR_RESOLUTION}m, "
              f"≤{self.max_lidar_points if self.max_lidar_points else '∞'} pts)")
        if self.eval_full_scene and self.split == "test":
            print(f"[FRACTAL] eval_full_scene=True: queries cover ALL points "
                  f"per scene (REQUIRES batch_size=1 in DataLoader)")

    # =========================================================================
    # INITIALIZATION HELPERS
    # =========================================================================

    def _setup_band_indices(self):
        if "bands_fractal_irgb_info" not in self.dataset_config:
            raise KeyError(
                "[FRACTAL] 'bands_fractal_irgb_info' missing from bands config."
            )
        all_bands = []
        for name, data in self.dataset_config["bands_fractal_irgb_info"].items():
            if all(k in data for k in ("bandwidth", "central_wavelength", "idx")):
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])
        indices = []
        for band in all_bands:
            key = (band["bandwidth"], band["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[FRACTAL] VHR band {band['name']} key={key} not in lookup."
                )
            indices.append(self.look_up.table_wave[key])
        self.vhr_spectral_indices = torch.tensor(indices, dtype=torch.long)
        print(f"[FRACTAL] VHR spectral indices ({len(indices)} bands): {indices}")
        self.lidar_spectral_idx = _resolve_elevation_spectral_idx(self.look_up)
        print(f"[FRACTAL] LIDAR spectral_idx (ELEVATION): "
              f"{self.lidar_spectral_idx}")

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
        print(f"[FRACTAL] Indexing ortho files under {irgb_root}...")
        ortho_index = {}
        for ext in ("*.tiff", "*.tif"):
            for op in irgb_root.rglob(ext):
                ortho_index[op.stem] = op
        print(f"[FRACTAL]   indexed {len(ortho_index):,} ortho files")
        valid_set = None
        if valid_patches_file is not None and os.path.exists(valid_patches_file):
            with open(valid_patches_file) as f:
                valid_data = json.load(f)
            split_key = {"train": "train", "val": "val",
                         "validation": "val", "test": "test"}[self.split]
            valid_set = set(valid_data.get(split_key, []))
            print(f"[FRACTAL] Loaded valid-patch filter: "
                  f"{len(valid_set)} patches for split={self.split}")
        self.patch_rows = []
        missing_orthos = 0
        skipped_invalid = 0
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
        if missing_orthos > 0:
            print(f"[FRACTAL] WARNING: {missing_orthos} LAZ files had no "
                  f"matching ortho — skipped.")
        if skipped_invalid > 0:
            print(f"[FRACTAL] Filtered out {skipped_invalid} patches via "
                  f"valid-patches list.")

    # =========================================================================
    # MASK DROPPED VHR BANDS
    # =========================================================================

    def _apply_vhr_band_dropout(self, vhr_tokens: torch.Tensor,
                                 vhr_mask: torch.Tensor) -> tuple:
        """
        Identify VHR tokens whose spectral_idx is in self._dropped_spectral_set,
        zero their reflectance value, and flag them in the attention mask.

        The model still sees the same number of VHR tokens — dropped tokens
        just contribute nothing through cross-attention (masked) and carry
        no spectral information (value=0).

        Args:
            vhr_tokens: [N_vhr, 8] VHR token tensor
            vhr_mask:   [N_vhr] bool mask (True = masked out of attention)

        Returns:
            (vhr_tokens, vhr_mask) with dropped-band entries zeroed/masked.
        """
        if not self._dropped_spectral_set:
            return vhr_tokens, vhr_mask

        spectral_idxs = vhr_tokens[:, TOKEN_SPECTRAL_IDX].long()
        # Build per-token "is this token's band dropped?" mask
        drop = torch.zeros_like(vhr_mask, dtype=torch.bool)
        for sidx in self._dropped_spectral_set:
            drop = drop | (spectral_idxs == sidx)

        # Zero the reflectance value for dropped tokens
        vhr_tokens = vhr_tokens.clone()
        vhr_tokens[drop, TOKEN_VALUE_IDX] = 0.0

        # Flag them in the attention mask (True = excluded)
        vhr_mask = vhr_mask | drop

        return vhr_tokens, vhr_mask

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]

        # ── Sample augmentation ONCE per item ─────────────────────
        # Same transform applied to both ortho and LIDAR (x, y) → spatial
        # alignment preserved across modalities by construction.
        # Identity for val/test (augmenter.enabled is False).
        aug = self.augmenter.sample(index=index)

        # ── Effective full-scene mode for this call ───────────────
        # Only the test split honors the eval_full_scene flag; train/val
        # always pad to a fixed query count so DataLoader batching works.
        full_scene_active = (self.eval_full_scene
                             and self.split == "test")

        # ── Load LIDAR ─────────────────────────────────────────────
        las = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]
        if n_points_raw < self.MIN_POINTS:
            return self.__getitem__((index + 1) % len(self))

        # Patch bounds (Lambert-93) → patch-local pixel coords
        x_min = float(las.x.min())
        y_max = float(las.y.max())
        lidar_x = (np.asarray(las.x) - x_min) / self.VHR_RESOLUTION
        lidar_y = (y_max - np.asarray(las.y)) / self.VHR_RESOLUTION

        lidar_x = np.clip(lidar_x, 0.0, self.PATCH_SIZE_PX - 1e-3)
        lidar_y = np.clip(lidar_y, 0.0, self.PATCH_SIZE_PX - 1e-3)

        # ── Echo info (return_number, number_of_returns) ───────────
        return_number     = np.asarray(las.return_number,     dtype=np.int64)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

        # ── Labels (BEFORE z-norm since z-norm uses ground mask) ───
        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]

        # ── Z normalization (ground-relative) ──────────────────────
        z_raw = np.asarray(las.z, dtype=np.float32)
        ground_mask = (labels == 1)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z_raw[ground_mask]))
        else:
            local_ground = float(np.percentile(z_raw, 5.0))
        z_rel  = z_raw - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        # ── Apply D4 to LIDAR (x, y) ───────────────────────────────
        xy_stacked = np.stack([lidar_x, lidar_y], axis=1).astype(np.float32)
        if not aug.is_identity:
            xy_stacked = self.augmenter.apply_to_xy(
                xy_stacked, aug, patch_size_px=self.PATCH_SIZE_PX
            )

        # ── Apply jitter (LIDAR only) ──────────────────────────────
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

        # ── Re-clip XY to valid range ──────────────────────────────
        xy_stacked = np.clip(
            xy_stacked, 0.0, self.PATCH_SIZE_PX - 1e-3
        ).astype(np.float32)
        lidar_x = xy_stacked[:, 0]
        lidar_y = xy_stacked[:, 1]

        # ═══════════════════════════════════════════════════════════════
        # CONTEXT vs QUERY SPLIT (full-scene-eval logic)
        # ═══════════════════════════════════════════════════════════════
        if full_scene_active:
            # Full arrays for queries (no subsampling on query side)
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
                ctx_lidar_x = lidar_x[sel]
                ctx_lidar_y = lidar_y[sel]
                ctx_z_norm  = z_norm[sel]
                ctx_labels  = labels[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]
            else:
                ctx_lidar_x = lidar_x
                ctx_lidar_y = lidar_y
                ctx_z_norm  = z_norm
                ctx_labels  = labels

            n_real_lidar_ctx     = ctx_lidar_x.shape[0]
            n_real_lidar_queries = full_lidar_x.shape[0]
        else:
            # Standard behavior: subsample once, context == queries.
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
                labels  = labels[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]

            ctx_lidar_x  = lidar_x
            ctx_lidar_y  = lidar_y
            ctx_z_norm   = z_norm
            ctx_labels   = labels
            full_lidar_x = lidar_x
            full_lidar_y = lidar_y
            full_z_norm  = z_norm
            full_labels  = labels
            n_real_lidar_ctx     = ctx_lidar_x.shape[0]
            n_real_lidar_queries = n_real_lidar_ctx

        # ── Tensors for context tokenization (subsampled set) ──────
        positions_ctx_lidar = torch.from_numpy(
            np.stack([ctx_lidar_x, ctx_lidar_y], axis=1)).float()
        values_ctx_lidar    = torch.from_numpy(ctx_z_norm).float()
        labels_ctx_lidar    = torch.from_numpy(ctx_labels.astype(np.int64))

        # ── Tensors for queries (full set in eval-full-scene mode) ─
        positions_query = torch.from_numpy(
            np.stack([full_lidar_x, full_lidar_y], axis=1)).float()
        values_query    = torch.from_numpy(full_z_norm).float()
        labels_query    = torch.from_numpy(full_labels.astype(np.int64))

        # ── Load ortho ──────────────────────────────────────────────
        with rasterio.open(row["ortho_path"]) as src:
            ortho = src.read().astype(np.float32)
        ortho = torch.from_numpy(ortho)
        ortho = (ortho / 127.5) - 1.0
        ortho = torch.clamp(ortho, -10, 10)
        ortho = torch.nan_to_num(ortho, nan=0.0, posinf=10.0, neginf=-10.0)

        # ── Apply D4 to ortho (same transform as LIDAR) ─────────────
        # apply() rotates the last 2 axes of [4, 250, 250].
        if not aug.is_identity:
            ortho = self.augmenter.apply(ortho, aug)

        # Dense label for VHR tokens: all IGNORE (we only supervise on LIDAR)
        dense_label = torch.full(
            (self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            self.IGNORE_INDEX, dtype=torch.long,
        )

        # ── Tokenize VHR ────────────────────────────────────────────
        # Tokenize all 4 bands as usual. If vhr_drop_bands is non-empty,
        # _apply_vhr_band_dropout below will zero the value and mask the
        # tokens belonging to dropped bands. The model still sees the
        # expected number of VHR tokens — only their contribution is
        # neutralized.
        vhr_tokens = self.token_builder.build_tokens(
            image=ortho,
            label=dense_label,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.vhr_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Tokenize LIDAR (sparse) ─────────────────────────────────
        # Echo info goes into column 7 (overriding time_idx, which is -1 for
        # FRACTAL anyway). build_sparse_tokens looks up the echo index via
        # self.look_up.get_echo_idx(r, t) and writes it per-point.
        # Uses CONTEXT positions/values (subsampled set in eval-full mode).
        lidar_tokens = self.token_builder.build_sparse_tokens(
            values=values_ctx_lidar,
            positions=positions_ctx_lidar,
            labels=labels_ctx_lidar,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
            return_number=return_number,
            number_of_returns=number_of_returns,
        )

        # ── Pad LIDAR context tokens to fixed size ──────────────────
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
        else:
            lidar_mask = torch.zeros(lidar_tokens.shape[0], dtype=torch.bool)

        vhr_mask = torch.zeros(vhr_tokens.shape[0], dtype=torch.bool)

        # ── Apply VHR band dropout (modality dropout at inference) ──
        # No-op when self.vhr_drop_bands is empty. When non-empty:
        # zeros the reflectance value AND flags the attention mask for
        # tokens whose spectral_idx is in the dropped set. The model
        # still receives every VHR token position; it just sees them as
        # masked-out padding-equivalents.
        if self._dropped_spectral_set:
            vhr_tokens, vhr_mask = self._apply_vhr_band_dropout(
                vhr_tokens, vhr_mask)

        hires_tokens = torch.cat([vhr_tokens, lidar_tokens], dim=0)
        hires_mask   = torch.cat([vhr_mask, lidar_mask], dim=0)

        groups = {
            self.VHR_RESOLUTION: {
                "tokens": hires_tokens,
                "mask":   hires_mask,
                "shape":  (self.NUM_VHR_BANDS,
                           self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            }
        }

        # ── Build queries from FULL positions/labels ────────────────
        queries = self.token_builder.build_sparse_queries(
            positions=positions_query,
            labels=labels_query,
            resolution=self.VHR_RESOLUTION,
            first_spectral_idx=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
        )

        assert queries.shape[0] == values_query.shape[0], (
            f"Query count ({queries.shape[0]}) doesn't match LIDAR point "
            f"count ({values_query.shape[0]})."
        )
        queries[:, 0] = values_query

        # ── Subsample queries during TRAINING only ──────────────────
        if self.split == "train":
            queries = self.token_builder.subsample_queries(
                queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
            )

        # ═══════════════════════════════════════════════════════════════
        # PADDING LOGIC: full-scene-eval (variable) vs standard (fixed)
        # ═══════════════════════════════════════════════════════════════
        if full_scene_active:
            queries_mask  = torch.zeros(queries.shape[0], dtype=torch.bool)
            labels_padded = labels_query
            n_real_lidar_for_return = n_real_lidar_queries
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
            elif n_real_queries > target_n_queries:
                queries      = queries[:target_n_queries]
                queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)
            else:
                queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)

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
            "target_resolution": self.VHR_RESOLUTION,
            "image":             ortho,
            "patch_id":          row["patch_id"],
        }
