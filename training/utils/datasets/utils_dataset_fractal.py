"""
FRACTAL Atomizer Dataset (with D4 + LIDAR jitter augmentation)
==================================================================

Same as the base utils_dataset_fractal.py but with training-time
augmentation, PLUS two additions matching the proven DalesDataset pattern:

  1. token_latent_assignment (NEW):
     Loads the offline-precomputed per-token nearest-latent assignment
     (see precompute_fractal_latent_assignment.py) — the shared VHR
     assignment (loaded once at __init__ as a [16, N_vhr] array, one row
     per D4 variant — VHR is NOT D4-invariant despite the raster's fixed
     (row,col)->meters mapping never changing, since apply(ortho, aug)
     moves pixel content across that grid by the same transform LIDAR
     points get) plus a per-patch LIDAR sidecar covering ALL 16 D4 variants
     and the FULL (un-subsampled) point set, gathered by whatever context
     subsample `sel` this __getitem__ call happens to draw — and
     concatenates the CURRENT variant's rows in the SAME [vhr, lidar] order
     as `hires_tokens`. Passed to the model as a single
     batch["token_latent_assignment"] tensor (FRACTAL has only one
     resolution group, so no per-res dict is needed — matching
     DalesDataset's convention exactly).

     Because precompute covers every point (not just a subsample), context
     subsampling is RANDOM PER EPOCH again during training, same as the
     original dataset — an earlier draft of this pipeline mistakenly
     required deterministic (patch_id-seeded) subsampling for precompute
     validity; that constraint no longer applies (see
     precompute_fractal_latent_assignment.py's module docstring).

     variant_idx convention: `n_rot * 4 + int(flip_h) * 2 + int(flip_v)`,
     matching DalesDataset.__getitem__ exactly.

  2. query_token_idx / query_token_valid (NEW):
     For each query (LIDAR point being predicted), the indices of its
     OWN pixel's VHR band-tokens in the pool (`hires_tokens`), for the
     decoder pixel-skip cascade in Atomiser_Senflood_Skip._pixel_skip.
     Unlike DALES (which has no VHR and uses a 1-atom identity/inverse
     mapping into its own context array), FRACTAL's "own atoms" are the
     (up to 4) VHR band-tokens co-located at the query's raster pixel.

     ASSUMPTION: VHR token order is band-major, then row-major:
         index = band * (H*W) + row*W + col
     (see _vhr_pool_index below). VERIFY this against
     TokenBuilder.build_tokens before trusting query_token_idx.

     Query subsampling now uses the real
     token_builder.subsample_queries(..., return_indices=True) — an
     earlier draft of this file reimplemented subsample_queries inline
     because that kwarg wasn't known to exist; it does, so the
     reimplementation is gone.

Everything else (D4 dihedral group augmentation, Gaussian jitter, full-scene
eval, VHR band-drop modality dropout) is UNCHANGED from the previous version.
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
from .fractal_geometry import compute_patch_local_pixel_coords


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
    FRACTAL semantic segmentation, Atomizer format, with D4 + jitter augs,
    offline-precomputed Voronoi assignment, and pixel-skip query indices.

    NEW args (on top of the previous version):
        vhr_assignment_path:  Path to the shared VHR->latent assignment
                              (.npy, int32) produced by
                              precompute_fractal_latent_assignment.py.
                              Required if use_precomputed_assignment=True.
        use_precomputed_assignment: Master switch. If False, no
                              token_latent_assignment is returned and
                              GeographicPruning falls back to its
                              shared-batch/patch_id-fallback paths
                              (INCORRECT for the shared-batch path on the
                              LIDAR portion — only disable for smoke tests).
        build_pixel_skip_queries: If True, computes query_token_idx /
                              query_token_valid for the decoder pixel-skip
                              cascade. Requires config["Atomiser"]
                              ["use_decoder_skip"]=True on the model side.
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
        vhr_assignment_path: str = None,
        use_precomputed_assignment: bool = True,
        build_pixel_skip_queries: bool = True,
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

        self.eval_full_scene = bool(eval_full_scene)
        if self.eval_full_scene and self.split != "test":
            print(f"[FRACTAL] WARNING: eval_full_scene=True with split="
                  f"'{self.split}' — full-scene queries only take effect "
                  f"during test evaluation. Ignored for this split.")

        self.vhr_drop_bands_spec = vhr_drop_bands
        self.vhr_drop_bands = _resolve_vhr_drop_bands(vhr_drop_bands)
        if self.vhr_drop_bands and self.split == "train":
            print(f"[FRACTAL] WARNING: vhr_drop_bands={self.vhr_drop_bands} "
                  f"with split='train' — band masking will be applied "
                  f"during training too. This is usually only desired "
                  f"for test-time modality-dropout evaluation.")

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

        # ── Load precomputed Voronoi assignment ──────────────────────
        # LIDAR sidecars are looked up relative to EACH PATCH'S OWN
        # laz_path (row["laz_path"].parent), not a flat directory passed
        # in here — FRACTAL's actual layout nests patches under numbered
        # subdirectories (e.g. val/val/00/, val/val/01/, ...), and a flat
        # directory param can't account for that (this was a real bug in
        # an earlier version of this file: sidecars were written correctly
        # by precompute, next to each patch wherever it actually lives,
        # but looked up from a fixed flat directory that only matched
        # patches sitting directly in it). Matches DalesDataset's own
        # pattern exactly (Path(row["laz_path"]).parent / ...).
        self.use_precomputed_assignment = bool(use_precomputed_assignment)
        self._vhr_assignment = None
        if self.use_precomputed_assignment:
            if vhr_assignment_path is None:
                raise ValueError(
                    "[FRACTAL] use_precomputed_assignment=True requires "
                    "vhr_assignment_path. Run "
                    "precompute_fractal_latent_assignment.py first, or "
                    "pass use_precomputed_assignment=False for smoke tests."
                )
            self._vhr_assignment = torch.from_numpy(
                np.load(vhr_assignment_path)).long()
            assert self._vhr_assignment.dim() == 2 and self._vhr_assignment.shape[0] == 16, (
                f"[FRACTAL] vhr_assignment_path={vhr_assignment_path} has "
                f"shape {tuple(self._vhr_assignment.shape)}, expected "
                f"[16, N_vhr] — re-run precompute_fractal_latent_assignment.py "
                f"(older versions of that script saved a flat [N_vhr] array "
                f"assuming VHR was D4-invariant, which was incorrect)."
            )
            print(f"[FRACTAL] Loaded shared VHR assignment (per D4 variant): "
                  f"shape={tuple(self._vhr_assignment.shape)} "
                  f"from {vhr_assignment_path}")
            print(f"[FRACTAL] LIDAR assignment sidecars: looked up per-patch, "
                  f"relative to each patch's own .laz location.")
        else:
            print("[FRACTAL] WARNING: use_precomputed_assignment=False — "
                  "geo_pruning will use its shared-batch/patch_id-fallback "
                  "paths, which are INCORRECT for LIDAR's per-sample token "
                  "ordering. Only use this for quick smoke tests.")

        self.build_pixel_skip_queries = bool(build_pixel_skip_queries)

        self.token_builder = TokenBuilder(look_up)
        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"].get(
            "max_tokens_reconstruction", max_queries
        )
        self.resolution_idx = look_up.get_resolution_idx(self.VHR_RESOLUTION)

        self._setup_band_indices()
        self._collect_patches(valid_patches_file)

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
        if not self._dropped_spectral_set:
            return vhr_tokens, vhr_mask

        spectral_idxs = vhr_tokens[:, TOKEN_SPECTRAL_IDX].long()
        drop = torch.zeros_like(vhr_mask, dtype=torch.bool)
        for sidx in self._dropped_spectral_set:
            drop = drop | (spectral_idxs == sidx)

        vhr_tokens = vhr_tokens.clone()
        vhr_tokens[drop, TOKEN_VALUE_IDX] = 0.0
        vhr_mask = vhr_mask | drop

        return vhr_tokens, vhr_mask

    # =========================================================================
    # Own-pixel VHR pool indices for the decoder pixel-skip cascade
    # =========================================================================

    def _vhr_pool_index(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        """
        ASSUMPTION (see module docstring): VHR token order is band-major,
        then row-major: index = band * (H*W) + row*W + col. VERIFY against
        TokenBuilder.build_tokens before trusting this.

        Returns shape [N, NUM_VHR_BANDS] — one pool index per band, per
        query point.
        """
        n_pix = self.PATCH_SIZE_PX * self.PATCH_SIZE_PX
        band_offsets = np.arange(self.NUM_VHR_BANDS, dtype=np.int64)[None, :]      # [1,4]
        pix_flat = (row.astype(np.int64) * self.PATCH_SIZE_PX
                    + col.astype(np.int64))[:, None]                              # [N,1]
        return band_offsets * n_pix + pix_flat                                     # [N,4]

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]

        aug = self.augmenter.sample(index=index)
        # variant_idx convention matches DalesDataset exactly.
        variant_idx = aug.n_rot * 4 + int(aug.flip_h) * 2 + int(aug.flip_v)

        full_scene_active = (self.eval_full_scene
                             and self.split == "test")

        # ── Load LIDAR ─────────────────────────────────────────────
        las = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]
        if n_points_raw < self.MIN_POINTS:
            return self.__getitem__((index + 1) % len(self))

        x_min = float(las.x.min())
        y_max = float(las.y.max())
        lidar_x, lidar_y = compute_patch_local_pixel_coords(
            las.x, las.y, x_min, y_max, self.VHR_RESOLUTION, self.PATCH_SIZE_PX)

        return_number     = np.asarray(las.return_number,     dtype=np.int64)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]

        z_raw = np.asarray(las.z, dtype=np.float32)
        ground_mask = (labels == 1)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(z_raw[ground_mask]))
        else:
            local_ground = float(np.percentile(z_raw, 5.0))
        z_rel  = z_raw - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        # ── Load precomputed token->latent assignment (offline) ────────
        # Covers ALL n_points_raw points, ALL 16 D4 variants — see
        # precompute_fractal_latent_assignment.py's module docstring for
        # why this means subsampling below can stay random per epoch.
        # Sidecar lives NEXT TO the patch's own .laz file (wherever that
        # actually is — FRACTAL nests patches under numbered subdirs like
        # val/val/00/, val/val/01/, ...), not in a flat directory.
        full_assignment = None
        if self.use_precomputed_assignment:
            sidecar_path = (Path(row["laz_path"]).parent
                            / f"{row['patch_id']}_latent_assign.npz")
            with np.load(sidecar_path) as npz:
                full_assignment = npz["assignment"][variant_idx]  # [n_points_raw]
            assert full_assignment.shape[0] == n_points_raw, (
                f"[FRACTAL] Precomputed assignment for {sidecar_path.name} "
                f"has {full_assignment.shape[0]} points, but this patch has "
                f"{n_points_raw} — re-run precompute_fractal_latent_assignment.py "
                f"(tiling changed, or points changed since precompute ran)."
            )

        # ── Original per-split subsample (RESTORED to random-per-epoch
        # for train — no longer needs to be deterministic, since precompute
        # covers every point) ───────────────────────────────────────────
        if (self.max_lidar_points is not None
                and n_points_raw > self.max_lidar_points):
            rng = np.random.default_rng(
                seed=hash(row["patch_id"]) & 0xFFFFFFFF
                if self.split != "train" else None
            )
            sel = rng.choice(n_points_raw, size=self.max_lidar_points,
                             replace=False)
        else:
            sel = None

        # ── Full-scene-eval branch keeps ALL points as queries; context
        # gets the (possibly subsampled) `sel` selection ────────────────
        if full_scene_active:
            full_lidar_x = lidar_x.copy()
            full_lidar_y = lidar_y.copy()
            full_z_norm  = z_norm.copy()
            full_labels  = labels.copy()
            if sel is not None:
                ctx_lidar_x = lidar_x[sel]
                ctx_lidar_y = lidar_y[sel]
                ctx_z_norm  = z_norm[sel]
                ctx_labels  = labels[sel]
                ctx_return_number     = return_number[sel]
                ctx_number_of_returns = number_of_returns[sel]
            else:
                ctx_lidar_x = lidar_x
                ctx_lidar_y = lidar_y
                ctx_z_norm  = z_norm
                ctx_labels  = labels
                ctx_return_number     = return_number
                ctx_number_of_returns = number_of_returns
            n_real_lidar_ctx     = ctx_lidar_x.shape[0]
            n_real_lidar_queries = full_lidar_x.shape[0]
        else:
            if sel is not None:
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
            ctx_return_number     = return_number
            ctx_number_of_returns = number_of_returns
            full_lidar_x = lidar_x
            full_lidar_y = lidar_y
            full_z_norm  = z_norm
            full_labels  = labels
            n_real_lidar_ctx     = ctx_lidar_x.shape[0]
            n_real_lidar_queries = n_real_lidar_ctx

        # ── Gather assignment by the SAME sel used for context ──────────
        if self.use_precomputed_assignment:
            ctx_assignment = (full_assignment if sel is None
                              else full_assignment[sel])

        # ── Apply D4 to LIDAR (x, y) ───────────────────────────────
        ctx_xy = np.stack([ctx_lidar_x, ctx_lidar_y], axis=1).astype(np.float32)
        if not aug.is_identity:
            ctx_xy = self.augmenter.apply_to_xy(
                ctx_xy, aug, patch_size_px=self.PATCH_SIZE_PX)

        # ── Apply jitter (LIDAR only) ──────────────────────────────
        if self.augmenter.enabled and (self.sigma_xy_pixels > 0
                                       or self.sigma_z_normed > 0):
            jitter_seed = (index * 2147483647) ^ 0x9E3779B9
            ctx_xy, ctx_z_norm = self.augmenter.apply_jitter(
                ctx_xy,
                z=ctx_z_norm,
                sigma_xy=self.sigma_xy_pixels,
                sigma_z=self.sigma_z_normed,
                seed=jitter_seed,
            )

        ctx_xy = np.clip(
            ctx_xy, 0.0, self.PATCH_SIZE_PX - 1e-3
        ).astype(np.float32)
        ctx_lidar_x = ctx_xy[:, 0]
        ctx_lidar_y = ctx_xy[:, 1]

        # Query positions. Standard path: query set == context set (same
        # array throughout, matching the original single-set design) — just
        # reuse the already-D4'd, already-jittered ctx coordinates. Only
        # full-scene-eval mode has a genuinely separate (larger, unjittered)
        # query set that needs its own D4 application.
        if full_scene_active:
            full_xy = np.stack([full_lidar_x, full_lidar_y], axis=1).astype(np.float32)
            if not aug.is_identity:
                full_xy = self.augmenter.apply_to_xy(
                    full_xy, aug, patch_size_px=self.PATCH_SIZE_PX)
            full_xy = np.clip(full_xy, 0.0, self.PATCH_SIZE_PX - 1e-3).astype(np.float32)
            full_lidar_x = full_xy[:, 0]
            full_lidar_y = full_xy[:, 1]
        else:
            full_lidar_x = ctx_lidar_x
            full_lidar_y = ctx_lidar_y
            full_z_norm  = ctx_z_norm
            full_labels  = ctx_labels

        positions_ctx_lidar = torch.from_numpy(
            np.stack([ctx_lidar_x, ctx_lidar_y], axis=1)).float()
        values_ctx_lidar    = torch.from_numpy(ctx_z_norm).float()
        labels_ctx_lidar    = torch.from_numpy(ctx_labels.astype(np.int64))

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

        if not aug.is_identity:
            ortho = self.augmenter.apply(ortho, aug)

        dense_label = torch.full(
            (self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            self.IGNORE_INDEX, dtype=torch.long,
        )

        vhr_tokens = self.token_builder.build_tokens(
            image=ortho,
            label=dense_label,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.vhr_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        lidar_tokens = self.token_builder.build_sparse_tokens(
            values=values_ctx_lidar,
            positions=positions_ctx_lidar,
            labels=labels_ctx_lidar,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
            return_number=ctx_return_number,
            number_of_returns=ctx_number_of_returns,
        )

        # ── Pad LIDAR context tokens (+ assignment) to fixed size ───────
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
            if self.use_precomputed_assignment:
                assign_pad = np.zeros(n_pad, dtype=np.int64)
                ctx_assignment = np.concatenate([ctx_assignment, assign_pad])
        else:
            lidar_mask = torch.zeros(lidar_tokens.shape[0], dtype=torch.bool)

        vhr_mask = torch.zeros(vhr_tokens.shape[0], dtype=torch.bool)

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

        # ── Combine VHR (shared, precomputed per-variant) + LIDAR (gathered)
        # VHR is NOT D4-invariant despite the raster's fixed (row,col)->
        # meters mapping never changing — apply(ortho, aug) moves pixel
        # content across that fixed grid by the same D4 transform LIDAR
        # points get, so it needs the same per-variant indexing. ──────────
        token_latent_assignment = None
        if self.use_precomputed_assignment:
            lidar_assignment = torch.from_numpy(
                ctx_assignment.astype(np.int64))
            vhr_assignment_this_variant = self._vhr_assignment[variant_idx]
            token_latent_assignment = torch.cat(
                [vhr_assignment_this_variant, lidar_assignment], dim=0)

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

        # ── Own-pixel VHR pool indices, tracked through subsampling ─────
        if self.build_pixel_skip_queries:
            query_row = np.clip(np.round(full_lidar_y).astype(np.int64),
                                 0, self.PATCH_SIZE_PX - 1)
            query_col = np.clip(np.round(full_lidar_x).astype(np.int64),
                                 0, self.PATCH_SIZE_PX - 1)
            query_token_idx_pre = torch.from_numpy(
                self._vhr_pool_index(query_row, query_col)).long()  # [Npts, 4]
        else:
            query_token_idx_pre = None

        # ── Subsample queries during TRAINING only — real
        # subsample_queries(return_indices=True), tracking query_token_idx
        # in lockstep via the returned kept_indices. ─────────────────────
        if self.split == "train":
            if self.build_pixel_skip_queries:
                queries, kept_indices = self.token_builder.subsample_queries(
                    queries,
                    max_queries=self.max_tokens_reconstruction,
                    ignore_index=self.IGNORE_INDEX,
                    prioritize_valid=True,
                    return_indices=True,
                )
                query_token_idx = query_token_idx_pre[kept_indices]
            else:
                queries = self.token_builder.subsample_queries(
                    queries,
                    max_queries=self.max_tokens_reconstruction,
                    ignore_index=self.IGNORE_INDEX,
                    prioritize_valid=True,
                )
                query_token_idx = None
        else:
            query_token_idx = query_token_idx_pre

        # ═══════════════════════════════════════════════════════════════
        # PADDING LOGIC: full-scene-eval (variable) vs standard (fixed)
        # ═══════════════════════════════════════════════════════════════
        if full_scene_active:
            queries_mask  = torch.zeros(queries.shape[0], dtype=torch.bool)
            labels_padded = labels_query
            n_real_lidar_for_return = n_real_lidar_queries
            query_token_valid = ~queries_mask if query_token_idx is not None else None
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
                if query_token_idx is not None:
                    idx_pad = torch.zeros(n_pad, self.NUM_VHR_BANDS, dtype=torch.long)
                    query_token_idx = torch.cat([query_token_idx, idx_pad], dim=0)
            elif n_real_queries > target_n_queries:
                queries      = queries[:target_n_queries]
                queries_mask = torch.zeros(target_n_queries, dtype=torch.bool)
                if query_token_idx is not None:
                    query_token_idx = query_token_idx[:target_n_queries]
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
            query_token_valid = (~queries_mask
                                 if query_token_idx is not None else None)

        out = {
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
        if token_latent_assignment is not None:
            out["token_latent_assignment"] = token_latent_assignment
        if query_token_idx is not None:
            out["query_token_idx"]   = query_token_idx
            out["query_token_valid"] = query_token_valid
        return out
