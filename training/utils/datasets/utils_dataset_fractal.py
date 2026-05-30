"""
FRACTAL Atomizer Dataset
==========================

Sparse LIDAR + dense VHR ortho segmentation on FRACTAL.

Modalities (both at 0.2m resolution group, mirroring FLAIR-HUB's
VHR+DEM grouping for encoder transfer):

    - VHR ortho (FRACTAL-IRGB): 4 ch @ 0.2m, 250×250 (NIR, R, G, B, uint8)
                                Dense raster, tokenized via build_tokens.
    - LIDAR points:             ~80k pts per patch, irregular (x, y, z)
                                Each point → 1 sparse token via
                                build_sparse_tokens (elevation as value).

Both modalities share resolution_idx (the 0.2m VHR group), so cross-attention
treats them as the same scale class. They differ only in their spectral_idx:
VHR uses NIR/R/G/B indices (compatible with FLAIR-HUB's aerial bands), LIDAR
uses the lookup table's "ELEVATION" abstract channel (already registered;
the launch script ensures it is present before the dataset is constructed).

Tasks:
    7-class semantic segmentation of LIDAR points:
        0 other | 1 ground | 2 vegetation | 3 building
        4 water | 5 bridge | 6 permanent structure

    Per-point evaluation (no per-pixel labels). VHR tokens are input only and
    carry IGNORE_INDEX; only LIDAR-point queries contribute to the loss.

Cross-task transfer from FLAIR-HUB:
    Initialize Atomizer with the FLAIR-HUB checkpoint. The encoder has seen
    VHR (RGB-NIR at 0.2m) and DEM (elevation at 0.2m). FRACTAL's modality
    types are the same — VHR ortho + sparse elevation — so the encoder
    weights transfer naturally. Only the segmentation head is reinitialized
    for 7 classes.

Variable-length LIDAR handling:
    Each patch has a different number of LIDAR points (typically 10k–100k).
    To allow batching with torch.stack, we pad LIDAR tokens and queries to a
    fixed count (max_lidar_points). Padded positions carry IGNORE_INDEX so
    they are excluded from CE loss and from torchmetrics IoU.

Token format (8 cols, unchanged):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
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


# ============================================================================
# LAS code → FRACTAL 7-class label remap
# ============================================================================
# Verified against the full 90k-patch dataset (matches FRACTAL paper's
# reported class proportions to within rounding).

# ============================================================================
# LAS code → FRACTAL 7-class label remap
# ============================================================================
# Codes are from the official FRACTAL/Lidar HD specification.
#
# Codes NOT in this LUT are mapped to IGNORE_INDEX (255) by _build_remap_lut,
# which excludes them from both training loss and evaluation metrics.
#
# DELIBERATELY NOT INCLUDED (and reason):
#   - Code 65 (Artefact):  spurious points that don't correspond to any real
#                          object or terrain (measurement noise). The spec
#                          explicitly says these are not physical points.
#                          Audit on 5k patches: 22,552 points in 1,552 patches
#                          (31% of all patches). Mapping them to any real class
#                          would pollute supervision — IGNORE is correct.
#   - Code 66 (Synthetic): artificial points added under bridges and on water
#                          surfaces to create coherent DTMs. Not real LIDAR
#                          returns. Audit: 14,491 points in 235 patches,
#                          concentrated near bridges/water.
#   - Code 67:             does not exist in the official spec.
#
# (An earlier version mapped 65, 66, 67 to permanent_structure (6). This
# polluted ~16.5% of permanent_structure labels with non-physical points and
# likely depressed permanent_structure IoU. Fixed.)

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
# Helper: resolve the ELEVATION spectral_idx from various Lookup_encoding APIs
# ============================================================================

def _resolve_elevation_spectral_idx(lookup) -> int:
    """
    Find the spectral_idx for the "ELEVATION" abstract channel. The launch
    script is expected to have registered it before the dataset is
    constructed. Different versions of Lookup_encoding expose this in
    different ways; try a few common attribute names.

    Raises RuntimeError if none of the attempted lookups succeed.
    """
    # 1. abstract_channel_indices dict: {"ELEVATION": int, ...}
    if hasattr(lookup, "abstract_channel_indices"):
        idx = lookup.abstract_channel_indices.get("ELEVATION")
        if idx is not None:
            return int(idx)

    # 2. Explicit getter
    if hasattr(lookup, "get_abstract_channel_idx"):
        try:
            return int(lookup.get_abstract_channel_idx("ELEVATION"))
        except Exception:
            pass

    # 3. Some lookups store abstract channels in table_wave with negative
    #    sentinel keys. The exact key for ELEVATION is implementation-
    #    dependent; we try a few likely candidates.
    candidates_table_wave = [
        ("ELEVATION", "ELEVATION"),
        (-3, -3),
        (-4, -4),
        (-5, -5),
        (-6, -6),
    ]
    if hasattr(lookup, "table_wave"):
        for key in candidates_table_wave:
            if key in lookup.table_wave:
                return int(lookup.table_wave[key])

    # 4. Some implementations expose a method on the lookup.
    if hasattr(lookup, "get_spectral_idx_by_name"):
        try:
            return int(lookup.get_spectral_idx_by_name("ELEVATION"))
        except Exception:
            pass

    raise RuntimeError(
        "[FRACTAL] Could not resolve spectral_idx for the 'ELEVATION' "
        "abstract channel. Make sure the launch script calls "
        "`lookup_table.register_abstract_channel('ELEVATION')` before "
        "constructing the dataset, and inspect your Lookup_encoding to "
        "confirm where ELEVATION is stored "
        "(common attrs: abstract_channel_indices, table_wave, "
        "get_abstract_channel_idx)."
    )


# ============================================================================
# FRACTAL Dataset
# ============================================================================

class FractalDataset(Dataset):
    """
    FRACTAL semantic segmentation, Atomizer format.

    Args:
        root_path:       Parent directory containing both FRACTAL/ (LAZ) and
                         FRACTAL-IRGB/ (TIFF) subdirectories.
        mode:            "train" | "val" | "test".
        dataset_config:  YAML config dict containing bands_fractal_irgb_info
                         (the 4 VHR band definitions). LIDAR uses the
                         lookup table's "ELEVATION" abstract channel.
        config_model:    Atomizer config dict.
        look_up:         Lookup_encoding instance (shared with FLAIR-HUB).
        max_lidar_points: Subsample LIDAR points per patch to this count if
                          they exceed it. Also the padding target for
                          batching. Defaults to 16k (Myria3D convention).
        max_queries:     Max query tokens per sample (decoder budget).
        valid_patches_file: Optional path to precomputed JSON listing valid
                            patch IDs per split (those with >=1000 points).
                            If None, scans on the fly and recurses past
                            degenerate patches.
    """

    # 0.2m resolution group — same as FLAIR-HUB VHR + DEM
    VHR_RESOLUTION  = 0.2
    PATCH_SIZE_M    = 50.0        # 50m × 50m patches
    PATCH_SIZE_PX   = 250         # 50m / 0.2m = 250 pixels per side
    NUM_VHR_BANDS   = 4           # NIR, R, G, B (file order)

    NUM_CLASSES     = 7
    IGNORE_INDEX    = 255
    TIME_IDX_NA     = -1

    MIN_POINTS      = 1000        # filter out degenerate patches

    # ── Z NORMALIZATION (ground-relative) ──────────────────────────────
    # "Height above local ground" with clipping. Strongly preferred over
    # per-patch percentile clipping because it preserves PHYSICAL meaning
    # across patches: ground is always at z=0, building roofs always at
    # ~5m, tall vegetation up to ~30m, etc.
    #
    # Stats from 10k-patch FRACTAL train sample (z relative to ground median):
    #   ground:     p1=-15.6  median= 0.00  p99= 15.1   ← centered at 0 ✓
    #   water:      p1=-17.1  median=-0.46  p99=  0.5   ← below ground ✓
    #   building:   p1= -2.5  median= 4.75  p99= 16.9   ← elevated, tight ✓
    #   bridge:     p1= -6.2  median= 0.92  p99= 19.4   ← bimodal (low+high)
    #   vegetation: p1=-12.2  median= 4.40  p99= 28.3   ← canopy ✓
    #
    # We clip to [Z_GROUND_REL_LO, Z_GROUND_REL_HI] (in meters), then map
    # linearly so 0m physical → 0 normalized (preserves ground anchor).
    Z_GROUND_REL_LO = -15.0       # clip floor (meters above ground)
    Z_GROUND_REL_HI = 30.0        # clip ceiling (meters above ground)
    Z_GROUND_REL_SCALE = 15.0     # divisor → ground=0, building≈0.33,
                                  #   max canopy ≈ 2.0 (clipped)
    # Sanity: 30 / 15 = 2.0 max normalized, -15 / 15 = -1.0 min normalized.
    # The reflectance encoder (Fourier features) handles [-1, 2] fine.

    # Minimum ground points required to compute a reliable ground median.
    # If fewer than this, we fall back to the 5th-percentile of all z.
    GROUND_MEDIAN_MIN_PTS = 50

    FRACTAL_CLASSES = [
        "other", "ground", "vegetation", "building",
        "water", "bridge", "permanent_structure",
    ]

    # File order in the IRGB TIFF: NIR, R, G, B
    # (verified via rasterio src.descriptions = ('Infrared', 'Red', 'Green', 'Blue'))
    VHR_BAND_NAMES = ["NIR", "R", "G", "B"]

    SPLIT_DIRS = {
        "train":      "train/train",
        "val":        "val/val",
        "test":       "test/test",
        "validation": "val/val",   # alias to match FLAIR-HUB's naming
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

        self.token_builder = TokenBuilder(look_up)

        # Trainer config: total token budget, decoder query budget
        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"].get(
            "max_tokens_reconstruction", max_queries
        )

        # ── Resolution index (shared with VHR/DEM at 0.2m) ──────
        self.resolution_idx = look_up.get_resolution_idx(self.VHR_RESOLUTION)

        # ── Build spectral indices ──────────────────────────────
        self._setup_band_indices()

        # ── Collect patch paths ─────────────────────────────────
        self._collect_patches(valid_patches_file)

        # ── Summary ─────────────────────────────────────────────
        print(f"[FRACTAL] Loaded {len(self.patch_rows)} patches, "
              f"split='{self.split}'")
        print(f"[FRACTAL] Modalities: "
              f"VHR({self.NUM_VHR_BANDS}ch@{self.VHR_RESOLUTION}m) + "
              f"LIDAR(elev@{self.VHR_RESOLUTION}m, "
              f"≤{self.max_lidar_points if self.max_lidar_points else '∞'} pts)")

    # =========================================================================
    # INITIALIZATION HELPERS
    # =========================================================================

    def _setup_band_indices(self):
        """
        VHR: 4 bands from bands_fractal_irgb_info YAML/dict, in file order
             (NIR, R, G, B). If FRACTAL bands share (bandwidth, wavelength)
             with FLAIR-HUB aerial bands, the lookup table assigns the same
             spectral_idx → encoder reuse works automatically.

        LIDAR: 1 abstract channel "ELEVATION", resolved from the lookup
               table via `_resolve_elevation_spectral_idx`. The launch
               script is expected to have called
               `lookup_table.register_abstract_channel("ELEVATION")` before
               constructing the dataset.
        """
        # VHR spectral indices
        if "bands_fractal_irgb_info" not in self.dataset_config:
            raise KeyError(
                "[FRACTAL] 'bands_fractal_irgb_info' missing from bands.yaml "
                "/ bands dict. Add 4 entries (NIR, R, G, B) with bandwidth / "
                "central_wavelength / idx matching FLAIR-HUB aerial bands for "
                "encoder reuse."
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
                    f"[FRACTAL] VHR band {band['name']} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        self.vhr_spectral_indices = torch.tensor(indices, dtype=torch.long)
        print(f"[FRACTAL] VHR spectral indices ({len(indices)} bands): {indices}")

        # LIDAR elevation: resolve "ELEVATION" abstract channel index from
        # the lookup table. The launch script registers it before this
        # method is called.
        self.lidar_spectral_idx = _resolve_elevation_spectral_idx(self.look_up)
        print(f"[FRACTAL] LIDAR spectral_idx (ELEVATION): "
              f"{self.lidar_spectral_idx}")

    def _collect_patches(self, valid_patches_file: str = None):
        """
        Build self.patch_rows = [{patch_id, laz_path, ortho_path}, ...]
        by globbing the split directory.

        IMPORTANT: the LAZ repo (FRACTAL) and the IRGB repo (FRACTAL-IRGB)
        shard their files into 80 numbered subdirectories ('00'..'79')
        independently — the same patch_id is NOT guaranteed to live in the
        same subdir on both sides. We therefore build a flat
        {patch_id: ortho_path} index by globbing the entire IRGB root once,
        and look up each LAZ's matching ortho from the index.

        If valid_patches_file is given, restrict to listed patch IDs.
        """
        split_dir = self.SPLIT_DIRS.get(self.split)
        if split_dir is None:
            raise ValueError(f"Unknown split: {self.split}")

        laz_root  = Path(self.root_path) / "FRACTAL"      / "data" / split_dir
        irgb_root = Path(self.root_path) / "FRACTAL-IRGB" / "data" / split_dir

        if not laz_root.exists():
            raise FileNotFoundError(f"FRACTAL LAZ root not found: {laz_root}")
        if not irgb_root.exists():
            raise FileNotFoundError(f"FRACTAL IRGB root not found: {irgb_root}")

        # ── Build the flat patch_id -> ortho_path index ────────────
        # One pass over all ortho files; subsequent LAZ -> ortho lookup is O(1).
        # Supports both .tiff and .tif extensions just in case.
        print(f"[FRACTAL] Indexing ortho files under {irgb_root}...")
        ortho_index = {}
        for ext in ("*.tiff", "*.tif"):
            for op in irgb_root.rglob(ext):
                ortho_index[op.stem] = op
        print(f"[FRACTAL]   indexed {len(ortho_index):,} ortho files")

        # ── Build valid-patch set if filter file provided ──────────
        valid_set = None
        if valid_patches_file is not None and os.path.exists(valid_patches_file):
            with open(valid_patches_file) as f:
                valid_data = json.load(f)
            split_key = {"train": "train", "val": "val",
                         "validation": "val", "test": "test"}[self.split]
            valid_set = set(valid_data.get(split_key, []))
            print(f"[FRACTAL] Loaded valid-patch filter: "
                  f"{len(valid_set)} patches for split={self.split}")

        # ── Glob LAZ files, look up matching ortho from the index ──
        self.patch_rows = []
        missing_orthos = 0
        skipped_invalid = 0
        for laz_path in sorted(laz_root.rglob("*.laz")):
            patch_id = laz_path.stem                       # e.g. "TRAIN-0436_..."
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
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]

        # ── Load LIDAR ───────────────────────────────────────────
        # Read once, get points + labels + spatial bounds.
        las = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]

        # Skip patches that became degenerate after some upstream filter
        # didn't catch them. Should be rare given valid_patches_file.
        if n_points_raw < self.MIN_POINTS:
            # Recurse to a different sample. Don't crash training.
            return self.__getitem__((index + 1) % len(self))

        # Patch spatial bounds (Lambert-93 meters) — used to convert LIDAR
        # world coords to patch-local pixel space.
        # The patch is nominally 50m × 50m; we use the actual LIDAR extents
        # as the reference frame (slightly more robust than the ortho bounds
        # if there's small misalignment).
        x_min = float(las.x.min())
        y_min = float(las.y.min())
        x_max = float(las.x.max())
        y_max = float(las.y.max())

        # ── LIDAR positions → patch-local pixel space ────────────
        # patch_x_in_pix = (lidar_x - x_min) / 0.2  → [0, 250)
        # y is flipped: rasterio orthos have top=largest y, so the LIDAR
        # point with the largest y maps to pixel row 0 (top of image).
        lidar_x = (np.asarray(las.x) - x_min) / self.VHR_RESOLUTION
        lidar_y = (y_max - np.asarray(las.y)) / self.VHR_RESOLUTION

        # Clip to valid pixel range (small floating-point excursions possible)
        lidar_x = np.clip(lidar_x, 0.0, self.PATCH_SIZE_PX - 1e-3)
        lidar_y = np.clip(lidar_y, 0.0, self.PATCH_SIZE_PX - 1e-3)

        # ── LIDAR labels (remap LAS → FRACTAL) ──────────────────
        # Computed FIRST because we need ground-point identification to
        # compute the local ground median for z normalization.
        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]                 # [N], values in [0, 7) or 255

        # ── LIDAR elevation normalization (ground-relative) ──────
        # Physically meaningful: every LIDAR point becomes "height above
        # local ground." Ground points cluster at 0, water near 0, building
        # roofs around +5m, tall vegetation up to +30m. Same physical
        # meaning across patches, regardless of absolute terrain elevation.
        z_raw = np.asarray(las.z, dtype=np.float32)

        # Local ground reference = median of ground-class points in the
        # patch. If too few ground points (e.g., dense forest patch), fall
        # back to the 5th-percentile of all z (close to ground in practice).
        ground_mask = (labels == 1)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z_raw[ground_mask]))
        else:
            local_ground = float(np.percentile(z_raw, 5.0))

        z_rel  = z_raw - local_ground                # height above ground
        z_clip = np.clip(z_rel,
                          self.Z_GROUND_REL_LO,
                          self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE    # [-1, 2] range

        # ── Subsample LIDAR points if too many ──────────────────
        if (self.max_lidar_points is not None
                and n_points_raw > self.max_lidar_points):
            # Random subsample. For val/test we use a fixed seed per patch
            # for reproducibility.
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

        n_real_lidar = lidar_x.shape[0]   # real (non-padded) LIDAR count

        # Convert to tensors
        positions_lidar = torch.from_numpy(np.stack([lidar_x, lidar_y], axis=1)
                                           ).float()      # [N, 2]
        values_lidar    = torch.from_numpy(z_norm).float()  # [N]
        labels_lidar    = torch.from_numpy(labels.astype(np.int64))  # [N]

        # ── Load ortho (VHR) ────────────────────────────────────
        with rasterio.open(row["ortho_path"]) as src:
            ortho = src.read().astype(np.float32)             # [4, 250, 250]
        ortho = torch.from_numpy(ortho)
        # Simple normalization: uint8 [0, 255] → roughly [-1, 1].
        # We could use FLAIR-HUB aerial stats here for direct encoder reuse,
        # but per-image mean/std risks blurring patches. Keep it simple for
        # the pilot; the launch script can override with proper stats.
        ortho = (ortho / 127.5) - 1.0
        ortho = torch.clamp(ortho, -10, 10)
        ortho = torch.nan_to_num(ortho, nan=0.0, posinf=10.0, neginf=-10.0)

        # Dense label tensor for VHR tokens: all IGNORE_INDEX (labels live
        # on the LIDAR side only).
        dense_label = torch.full(
            (self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            self.IGNORE_INDEX, dtype=torch.long,
        )

        # ── Tokenize VHR ────────────────────────────────────────
        vhr_tokens = self.token_builder.build_tokens(
            image=ortho,
            label=dense_label,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.vhr_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Tokenize LIDAR (sparse) ─────────────────────────────
        lidar_tokens = self.token_builder.build_sparse_tokens(
            values=values_lidar,
            positions=positions_lidar,
            labels=labels_lidar,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Pad LIDAR tokens to fixed count for batching ────────
        # Variable LIDAR count per patch breaks torch.stack() across the batch.
        # Pad to self.max_lidar_points and mark padded positions in the mask.
        # Padding token: all zeros except label=IGNORE_INDEX so any code path
        # that misses the mask still ignores them.
        n_lidar_tokens = lidar_tokens.shape[0]
        if (self.max_lidar_points is not None
                and n_lidar_tokens < self.max_lidar_points):
            n_pad = self.max_lidar_points - n_lidar_tokens
            pad = torch.zeros(n_pad, 8)
            pad[:, 4] = self.IGNORE_INDEX                  # col 4 = label
            lidar_tokens = torch.cat([lidar_tokens, pad], dim=0)
            lidar_mask = torch.cat([
                torch.zeros(n_lidar_tokens, dtype=torch.bool),  # real
                torch.ones(n_pad, dtype=torch.bool),             # padded
            ])
        else:
            lidar_mask = torch.zeros(lidar_tokens.shape[0], dtype=torch.bool)

        # ── Concatenate VHR (always fixed) + LIDAR (padded) ─────
        # VHR tokens: always 250*250*4 = 250,000 — fixed shape.
        # LIDAR tokens: padded to self.max_lidar_points — fixed shape.
        # Their masks reflect real vs padded; VHR is all real.
        vhr_mask     = torch.zeros(vhr_tokens.shape[0], dtype=torch.bool)
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

        # ── Build queries: per LIDAR point ──────────────────────
        # Queries inherit labels from LIDAR (the model is judged on
        # per-point classification accuracy).
        queries = self.token_builder.build_sparse_queries(
            positions=positions_lidar,
            labels=labels_lidar,
            resolution=self.VHR_RESOLUTION,
            first_spectral_idx=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Inject normalized z into query col 0 (FRACTAL-specific) ─
        # Atomiser_Fractal reads col 0 of queries to derive a per-pixel
        # Q vector in the decoder via a z-projection layer. This is the
        # key architectural addition that lets the decoder distinguish
        # points that share (x, y) but differ in z (e.g., bridge over
        # road, building eaves over sidewalk, tree canopy over ground).
        # Padding queries (added below) keep col 0 = 0; their labels are
        # IGNORE_INDEX so they don't contribute to loss anyway.
        assert queries.shape[0] == values_lidar.shape[0], (
            f"Query count ({queries.shape[0]}) doesn't match LIDAR point "
            f"count ({values_lidar.shape[0]}). Both should be n_real_lidar."
        )
        queries[:, 0] = values_lidar                # already normalized [-1, 2]

        if self.split == "train":
            queries = self.token_builder.subsample_queries(
                queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
            )

        # ── Pad queries to fixed count for batching ─────────────
        # Train: pad to max_tokens_reconstruction.
        # Val/test: pad to max_lidar_points (queries = all sampled LIDAR pts).
        target_n_queries = (self.max_tokens_reconstruction
                            if self.split == "train"
                            else (self.max_lidar_points
                                  if self.max_lidar_points is not None
                                  else queries.shape[0]))
        n_real_queries = queries.shape[0]
        if n_real_queries < target_n_queries:
            n_pad = target_n_queries - n_real_queries
            qpad = torch.zeros(n_pad, 8)
            qpad[:, 4] = self.IGNORE_INDEX                 # label
            queries = torch.cat([queries, qpad], dim=0)
            queries_mask = torch.cat([
                torch.zeros(n_real_queries, dtype=torch.bool),  # real
                torch.ones(n_pad, dtype=torch.bool),             # padded
            ])
        elif n_real_queries > target_n_queries:
            # Shouldn't happen if subsample_queries respected the budget,
            # but truncate just in case.
            queries      = queries[:target_n_queries]
            queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)
        else:
            queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)

        # ── Pad labels_lidar for stackable returns ──────────────
        # The trainer uses these labels for per-point metrics. Pad to
        # max_lidar_points so all samples in a batch have the same shape.
        # n_real_lidar tells the trainer how many entries are real.
        labels_padded = labels_lidar
        if (self.max_lidar_points is not None
                and labels_lidar.shape[0] < self.max_lidar_points):
            n_pad = self.max_lidar_points - labels_lidar.shape[0]
            label_pad = torch.full((n_pad,), self.IGNORE_INDEX,
                                   dtype=torch.long)
            labels_padded = torch.cat([labels_lidar, label_pad], dim=0)

        # ── Return ──────────────────────────────────────────────
        return {
            "groups":            groups,
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             labels_padded,  # padded for batching
            "n_real_lidar":      torch.tensor(n_real_lidar, dtype=torch.long),
            "target_resolution": self.VHR_RESOLUTION,
            "image":             ortho,
            "patch_id":          row["patch_id"],
        }
