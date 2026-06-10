"""
PureForest Classification Dataset for Atomizer
================================================

18-class (reduced to 13 in practice) pure forest species classification.
Each scene is a 50×50 m patch with:
  - RGB+NIR ortho at 0.2 m/px  → 250×250 px, 4 bands, uint8
  - LiDAR point cloud           → ~100k–180k pts, LAS format 8

Directory layout (after extraction):
    data/
        imagery-{Species}/imagery/{split}/{patch_id}.tiff
        lidar-{Species}/lidar/{split}/{patch_id}.laz
    metadata/
        PureForest-patches.csv   — patch_id, split, class_index, class_name, ...

Output format (identical contract to EuroSATDataset):
    {
        "groups": {
            0.2: {
                "tokens": [N_rgb, 8],
                "mask":   [N_rgb],
                "shape":  (4, 250, 250),
            },
            # LiDAR tokens are appended into the same 0.2 group
            # (same physical GSD as imagery) so the encoder sees
            # them in one unified group.
        },
        "queries":           [1, 8]   — single CLS query at patch center
        "queries_mask":      [1],
        "label":             scalar long (0..12)
        "task":              "classification",
        "target_resolution": 0.2,
        "image":             [4, 250, 250]  float32, normalised
        "patch_id":          str
    }

LiDAR tokenisation strategy
----------------------------
Each point becomes one token (mirroring FRACTAL's sparse token approach):
  - value    : z_norm  (ground-relative elevation, clipped to [-1, 2],
                        divided by Z_GROUND_REL_SCALE=15 m)
  - position : patch-local pixel coords (x, y) ∈ [0, 250)
  - spectral : ELEVATION channel (registered in lookup as for FRACTAL)
  - resolution_idx / time_idx : same as imagery

Augmentation (training only)
-----------------------------
D4 dihedral group (8 orientations) applied consistently to both
imagery and LiDAR (x, y) coordinates, exactly as in FRACTAL.
LiDAR z (height) is invariant under horizontal rotation — not touched.
No per-point jitter by default (pure forest task doesn't need it,
but the parameter is exposed for experimentation).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[Warning] laspy not installed — required for PureForest LAZ reading.")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed — required for PureForest ortho.")

from .token_builder import TokenBuilder
from .augmentations import D4Augmentation


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR attribute constants  (LAS format 8, same codes as FRACTAL)
# ─────────────────────────────────────────────────────────────────────────────

# Z normalisation: ground-relative, clipped and scaled
Z_GROUND_REL_LO    = -5.0    # trees don't go much below ground
Z_GROUND_REL_HI    =  40.0   # covers all species including Larch/Black pine
Z_GROUND_REL_SCALE =  20.0   # -> normalized range [-0.25, 2.0], centered usefully
GROUND_MEDIAN_MIN_PTS = 50   # min ground points to trust median; else use p5

# LAS classification codes used to identify ground returns
LAS_GROUND_CODE = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helper — resolve spectral index for ELEVATION in the lookup table
# (copy of the pattern used in utils_dataset_fractal.py)
# ─────────────────────────────────────────────────────────────────────────────

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
    candidates = [("ELEVATION", "ELEVATION"), (-3, -3), (-4, -4), (-5, -5)]
    if hasattr(lookup, "table_wave"):
        for key in candidates:
            if key in lookup.table_wave:
                return int(lookup.table_wave[key])
    raise RuntimeError(
        "[PureForest] Could not resolve spectral_idx for 'ELEVATION'. "
        "Register it via lookup_table.register_abstract_channel('ELEVATION') "
        "before constructing the dataset."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PureForestDataset
# ─────────────────────────────────────────────────────────────────────────────

class PureForestDataset(Dataset):
    """
    PureForest tree species classification — Atomizer format.

    Args:
        root_path:          Path to the PureForest root directory
                            (the one containing data/ and metadata/).
        mode:               'train' | 'val' | 'test'
        dataset_config:     Config dict; must contain 'bands_pureforest_irgb_info'
                            with per-band bandwidth / central_wavelength / idx.
        config_model:       Model config dict (trainer.max_tokens, etc.)
        look_up:            Lookup_encoding instance.
        max_lidar_points:   Hard cap on LiDAR points per scene (random
                            subsample). None = no cap.
        use_augmentation:   Master switch; auto-disabled for val/test.
        sigma_xy_pixels:    Std dev of optional XY jitter in pixel units
                            (0.2 m/px). 0 = disabled.
        sigma_z_normed:     Std dev of optional Z jitter in normalised units.
                            0 = disabled.
        modality:           'both' (RGB + LiDAR), 'rgb', or 'lidar'.
                            Default 'both'.
    """

    # ── Scene geometry ────────────────────────────────────────────────────
    VHR_RESOLUTION = 0.2       # m/px
    PATCH_SIZE_M   = 50.0      # physical extent
    PATCH_SIZE_PX  = 250       # 50 m / 0.2 m/px
    NUM_VHR_BANDS  = 4         # NIR, R, G, B  (uint8 → float32)

    # ── Task ──────────────────────────────────────────────────────────────
    NUM_CLASSES  = 13
    IGNORE_INDEX = 255
    TIME_IDX_NA  = -1
    TASK_NAME    = "classification"

    # Minimum valid points after subsample; skip scene if below this.
    MIN_POINTS = 500

    # ── Split → subfolder name as it appears in the CSV 'split' column ───
    # The CSV uses 'train' / 'val' / 'test' directly.
    SPLIT_MAPPING = {
        "train":      "train",
        "val":        "val",
        "validation": "val",
        "test":       "test",
    }

    CLASS_NAMES = [
        "Deciduous oak",        # 0
        "Evergreen oak",        # 1
        "Beech",                # 2
        "Chestnut",             # 3
        "Black locust",         # 4
        "Maritime pine",        # 5
        "Scotch pine",          # 6
        "Black pine",           # 7
        "Aleppo pine",          # 8
        "Fir",                  # 9
        "Spruce",               # 10
        "Larch",                # 11
        "Douglas",              # 12
    ]

    def __init__(
        self,
        root_path: str = "./data/PureForest",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        max_lidar_points: int = 16_000,
        use_augmentation: bool = True,
        sigma_xy_pixels: float = 0.0,
        sigma_z_normed: float = 0.0,
        modality: str = "both",
    ):
        super().__init__()

        for lib, ok in [("laspy", HAS_LASPY), ("rasterio", HAS_RASTERIO)]:
            if not ok:
                raise ImportError(f"{lib} is required for PureForestDataset.")

        assert modality in ("both", "rgb", "lidar"), \
            f"modality must be 'both', 'rgb', or 'lidar', got '{modality}'"

        self.root_path        = Path(root_path)
        self.split            = self.SPLIT_MAPPING[mode]
        self.look_up          = look_up
        self.config_model     = config_model
        self.dataset_config   = dataset_config
        self.max_lidar_points = max_lidar_points
        self.modality         = modality

        # ── Token / model config ──────────────────────────────────
        self.nb_tokens     = config_model["trainer"]["max_tokens"]
        self.token_builder = TokenBuilder(look_up)
        self.resolution_idx = look_up.get_resolution_idx(self.VHR_RESOLUTION)

        # ── Augmentation ──────────────────────────────────────────
        # D4 is applied to both ortho and LiDAR XY in lock-step.
        # Val/test: augmenter.enabled is always False.
        aug_on = use_augmentation and (self.split == "train")
        self.augmenter = D4Augmentation(
            enabled=aug_on,
            p_flip_h=0.5,
            p_flip_v=0.5,
        )
        self.sigma_xy_pixels = float(sigma_xy_pixels)
        self.sigma_z_normed  = float(sigma_z_normed)
        if aug_on:
            print(f"[PureForest] D4 augmentation ENABLED "
                  f"(sigma_xy={self.sigma_xy_pixels}px, "
                  f"sigma_z={self.sigma_z_normed})")
        else:
            print(f"[PureForest] Augmentation DISABLED "
                  f"(split={self.split})")

        # ── Band / spectral setup ─────────────────────────────────
        self._setup_band_indices()

        # ── Load patch index from CSV ─────────────────────────────
        self._collect_patches()

        # ── Class distribution summary ────────────────────────────
        from collections import Counter
        dist = Counter(r["class_index"] for r in self.patch_rows)
        print(f"[PureForest] split='{self.split}', "
              f"modality='{self.modality}', "
              f"scenes={len(self.patch_rows)}")
        print(f"[PureForest] class distribution: "
              f"{dict(sorted(dist.items()))}")

    # =========================================================================
    # INIT HELPERS
    # =========================================================================

    def _setup_band_indices(self):
        """Build spectral index tensors for VHR bands and LiDAR elevation."""
        key = "bands_pureforest_irgb_info"
        if key not in self.dataset_config:
            raise KeyError(
                f"[PureForest] '{key}' missing from dataset_config. "
                f"Add per-band bandwidth/central_wavelength/idx entries "
                f"matching the 4 IRGB bands (NIR≈833nm, R≈668nm, "
                f"G≈555nm, B≈490nm)."
            )
        all_bands = []
        for name, data in self.dataset_config[key].items():
            if all(k in data for k in ("bandwidth", "central_wavelength", "idx")):
                all_bands.append({
                    "idx":               data["idx"],
                    "bandwidth":         int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name":              name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        indices = []
        for b in all_bands:
            key_wave = (b["bandwidth"], b["central_wavelength"])
            if key_wave not in self.look_up.table_wave:
                raise KeyError(
                    f"[PureForest] Band '{b['name']}' key={key_wave} not in "
                    f"lookup table. Register the band before constructing the "
                    f"dataset."
                )
            indices.append(self.look_up.table_wave[key_wave])

        self.vhr_spectral_indices = torch.tensor(indices, dtype=torch.long)
        print(f"[PureForest] VHR spectral indices "
              f"({len(indices)} bands): {indices}")

        self.lidar_spectral_idx = _resolve_elevation_spectral_idx(self.look_up)
        print(f"[PureForest] LiDAR spectral_idx (ELEVATION): "
              f"{self.lidar_spectral_idx}")

    def _collect_patches(self):
        """
        Build self.patch_rows from PureForest-patches.csv.

        Each row: {patch_id, class_index, laz_path, ortho_path}

        File naming convention (from exploration output):
            lidar-{Species}/lidar/{split}/{patch_id}.laz
            imagery-{Species}/imagery/{split}/{patch_id}.tiff

        The CSV's 'name_latin' column gives the species (e.g. 'Pinus_sylvestris')
        which maps directly to the folder prefix.
        """
        csv_path = self.root_path / "metadata" / "PureForest-patches.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"[PureForest] Metadata CSV not found: {csv_path}"
            )

        df = pd.read_csv(csv_path, low_memory=False)

        # Keep only the requested split
        df = df[df["split"] == self.split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"[PureForest] No patches found for split='{self.split}' "
                f"in {csv_path}. Available splits: "
                f"{pd.read_csv(csv_path, low_memory=False)['split'].unique().tolist()}"
            )

        data_root = self.root_path / "data"

        self.patch_rows = []
        missing = 0

        for _, row in df.iterrows():
            patch_id    = str(row["patch_id"])
            class_index = int(row["class_index"])
            species     = str(row["name_latin"]).replace(" ", "_")  # CSV has spaces, folders use underscores

            # Filenames on disk have an uppercase split prefix:
            # e.g. patch_id="Pinus_halepensis-C8-3_1_244"
            #   -> "TRAIN-Pinus_halepensis-C8-3_1_244.laz"
            split_prefix = self.split.upper()
            laz_path  = (data_root
                         / f"lidar-{species}"
                         / "lidar"
                         / self.split
                         / f"{split_prefix}-{patch_id}.laz")
            tiff_path = (data_root
                         / f"imagery-{species}"
                         / "imagery"
                         / self.split
                         / f"{split_prefix}-{patch_id}.tiff")

            # Both files must exist
            if not laz_path.exists() or not tiff_path.exists():
                missing += 1
                continue

            self.patch_rows.append({
                "patch_id":    patch_id,
                "class_index": class_index,
                "laz_path":    str(laz_path),
                "ortho_path":  str(tiff_path),
            })

        if missing > 0:
            print(f"[PureForest] WARNING: {missing} CSV rows had missing "
                  f"laz or tiff file — skipped.")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index: int) -> dict:
        row = self.patch_rows[index]
        label = torch.tensor(row["class_index"], dtype=torch.long)

        # ── Sample D4 transform ONCE (shared by ortho + LiDAR) ───
        aug = self.augmenter.sample(index=index)

        # ─────────────────────────────────────────────────────────
        # 1.  LOAD & PROCESS LiDAR
        # ─────────────────────────────────────────────────────────
        lidar_tokens = None
        lidar_mask   = None

        if self.modality in ("both", "lidar"):
            las = laspy.read(row["laz_path"])
            n_pts_raw = int(las.x.shape[0])

            if n_pts_raw < self.MIN_POINTS:
                # Skip degenerate scene — recurse to next
                return self.__getitem__((index + 1) % len(self))

            # ── Patch-local pixel coordinates ────────────────────
            # Lambert-93 → pixel space, matching the ortho frame.
            x_min = float(las.x.min())
            y_max = float(las.y.max())
            lidar_x = (np.asarray(las.x, dtype=np.float32) - x_min) / self.VHR_RESOLUTION
            lidar_y = (y_max - np.asarray(las.y, dtype=np.float32)) / self.VHR_RESOLUTION
            lidar_x = np.clip(lidar_x, 0.0, self.PATCH_SIZE_PX - 1e-3)
            lidar_y = np.clip(lidar_y, 0.0, self.PATCH_SIZE_PX - 1e-3)

            # ── Z normalisation (ground-relative) ────────────────
            z_raw = np.asarray(las.z, dtype=np.float32)
            las_cls = np.asarray(las.classification, dtype=np.int64)
            ground_mask = (las_cls == LAS_GROUND_CODE)
            if ground_mask.sum() >= GROUND_MEDIAN_MIN_PTS:
                local_ground = float(np.median(z_raw[ground_mask]))
            else:
                local_ground = float(np.percentile(z_raw, 5.0))
            z_rel  = z_raw - local_ground
            z_clip = np.clip(z_rel, Z_GROUND_REL_LO, Z_GROUND_REL_HI)
            z_norm = z_clip / Z_GROUND_REL_SCALE   # ∈ [-1, 2]

            # ── Return number / number of returns ─────────────────
            return_number     = np.asarray(las.return_number,     dtype=np.int64)
            number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

            # ── Apply D4 to LiDAR XY ─────────────────────────────
            xy = np.stack([lidar_x, lidar_y], axis=1)
            if not aug.is_identity:
                xy = self.augmenter.apply_to_xy(
                    xy, aug, patch_size_px=self.PATCH_SIZE_PX
                )

            # ── Optional XY / Z jitter ────────────────────────────
            if self.augmenter.enabled and (
                self.sigma_xy_pixels > 0 or self.sigma_z_normed > 0
            ):
                jitter_seed = (index * 2147483647) ^ 0x9E3779B9
                xy, z_norm = self.augmenter.apply_jitter(
                    xy,
                    z=z_norm,
                    sigma_xy=self.sigma_xy_pixels,
                    sigma_z=self.sigma_z_normed,
                    seed=jitter_seed,
                )

            xy = np.clip(xy, 0.0, self.PATCH_SIZE_PX - 1e-3).astype(np.float32)

            # ── Subsample if over cap ─────────────────────────────
            n_pts = xy.shape[0]
            if self.max_lidar_points is not None and n_pts > self.max_lidar_points:
                rng = np.random.default_rng(
                    seed=None if self.split == "train"
                    else hash(row["patch_id"]) & 0xFFFFFFFF
                )
                sel = rng.choice(n_pts, size=self.max_lidar_points, replace=False)
                xy                = xy[sel]
                z_norm            = z_norm[sel]
                return_number     = return_number[sel]
                number_of_returns = number_of_returns[sel]

            n_lidar = xy.shape[0]

            positions_lidar = torch.from_numpy(xy).float()          # [N, 2]
            values_lidar    = torch.from_numpy(z_norm).float()      # [N]

            # For classification, per-point labels are all IGNORE
            # (the scene-level label is in the CLS query)
            labels_lidar = torch.full(
                (n_lidar,), self.IGNORE_INDEX, dtype=torch.long
            )

            # ── Tokenise LiDAR ────────────────────────────────────
            lidar_tokens = self.token_builder.build_sparse_tokens(
                values=values_lidar,
                positions=positions_lidar,
                labels=labels_lidar,
                resolution=self.VHR_RESOLUTION,
                spectral_indices=self.lidar_spectral_idx,
                resolution_idx=self.resolution_idx,
                patch_size_px=self.PATCH_SIZE_PX,
                time_idx=self.TIME_IDX_NA,
                return_number=return_number,
                number_of_returns=number_of_returns,
            )

            # ── Pad LiDAR tokens to max_lidar_points ─────────────
            n_tok = lidar_tokens.shape[0]
            if (self.max_lidar_points is not None
                    and n_tok < self.max_lidar_points):
                n_pad = self.max_lidar_points - n_tok
                pad   = torch.zeros(n_pad, 8)
                pad[:, 4] = self.IGNORE_INDEX
                lidar_tokens = torch.cat([lidar_tokens, pad], dim=0)
                lidar_mask = torch.cat([
                    torch.zeros(n_tok,  dtype=torch.bool),
                    torch.ones(n_pad,   dtype=torch.bool),
                ])
            else:
                lidar_mask = torch.zeros(lidar_tokens.shape[0], dtype=torch.bool)

        # ─────────────────────────────────────────────────────────
        # 2.  LOAD & PROCESS RGB+NIR ortho
        # ─────────────────────────────────────────────────────────
        vhr_tokens = None
        vhr_mask   = None
        ortho      = None

        if self.modality in ("both", "rgb"):
            with rasterio.open(row["ortho_path"]) as src:
                ortho_np = src.read().astype(np.float32)   # [4, 250, 250]

            ortho = torch.from_numpy(ortho_np)

            # Normalise uint8 → [-1, 1]
            ortho = (ortho / 127.5) - 1.0
            ortho = torch.clamp(ortho, -10.0, 10.0)
            ortho = torch.nan_to_num(ortho, nan=0.0, posinf=10.0, neginf=-10.0)

            # ── Apply same D4 as LiDAR ────────────────────────────
            if not aug.is_identity:
                ortho = self.augmenter.apply(ortho, aug)

            # Per-pixel label: all IGNORE (scene-level task)
            dummy_label = torch.full(
                (self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
                self.IGNORE_INDEX, dtype=torch.long,
            )

            vhr_tokens = self.token_builder.build_tokens(
                image=ortho,
                label=dummy_label,
                resolution=self.VHR_RESOLUTION,
                spectral_indices=self.vhr_spectral_indices,
                resolution_idx=self.resolution_idx,
                time_idx=self.TIME_IDX_NA,
            )
            vhr_mask = torch.zeros(vhr_tokens.shape[0], dtype=torch.bool)

        # ─────────────────────────────────────────────────────────
        # 3.  MERGE MODALITIES into a single resolution group
        # ─────────────────────────────────────────────────────────
        # Both modalities share the same physical GSD (0.2 m/px), so they
        # naturally live in the same group — the encoder sees one flat token
        # set regardless of origin (pixel band vs. LiDAR point).
        if self.modality == "both":
            all_tokens = torch.cat([vhr_tokens, lidar_tokens], dim=0)
            all_mask   = torch.cat([vhr_mask,   lidar_mask],   dim=0)
            shape = (self.NUM_VHR_BANDS, self.PATCH_SIZE_PX, self.PATCH_SIZE_PX)
        elif self.modality == "rgb":
            all_tokens = vhr_tokens
            all_mask   = vhr_mask
            shape = (self.NUM_VHR_BANDS, self.PATCH_SIZE_PX, self.PATCH_SIZE_PX)
        else:  # lidar only
            all_tokens = lidar_tokens
            all_mask   = lidar_mask
            shape = (1, self.PATCH_SIZE_PX, self.PATCH_SIZE_PX)

        # ── Optional global token cap (e.g. trainer.max_tokens) ──
        N = all_tokens.shape[0]
        if N > self.nb_tokens:
            # Random subsample preserving the mask alignment
            perm       = torch.randperm(N)[:self.nb_tokens]
            all_tokens = all_tokens[perm]
            all_mask   = all_mask[perm]

        # ─────────────────────────────────────────────────────────
        # 4.  CLS QUERY  (one per scene, at patch center)
        # ─────────────────────────────────────────────────────────
        cx = (self.PATCH_SIZE_PX - 1) / 2.0
        cy = (self.PATCH_SIZE_PX - 1) / 2.0
        first_spectral_idx = int(self.vhr_spectral_indices[0].item()) \
            if self.modality in ("both", "rgb") \
            else self.lidar_spectral_idx

        query = torch.tensor([[
            0.0,                      # value (unused for CLS query)
            cx,                       # x — patch centre
            cy,                       # y — patch centre
            float(first_spectral_idx),# spectral_idx
            float(label.item()),      # label  ← scene-level class
            0.0,                      # query_idx
            float(self.resolution_idx),
            float(self.TIME_IDX_NA),
        ]], dtype=torch.float32)      # [1, 8]

        queries_mask = torch.zeros(1, dtype=torch.bool)

        # ─────────────────────────────────────────────────────────
        # 5.  RETURN
        # ─────────────────────────────────────────────────────────
        return {
            "groups": {
                self.VHR_RESOLUTION: {
                    "tokens": all_tokens,   # [N_rgb + N_lidar, 8]
                    "mask":   all_mask,     # [N_rgb + N_lidar]
                    "shape":  shape,
                },
            },
            "queries":           query,         # [1, 8]
            "queries_mask":      queries_mask,  # [1]
            "label":             label,         # scalar long
            "task":              self.TASK_NAME,
            "target_resolution": self.VHR_RESOLUTION,
            "image":             ortho if ortho is not None
                                 else torch.zeros(
                                     self.NUM_VHR_BANDS,
                                     self.PATCH_SIZE_PX,
                                     self.PATCH_SIZE_PX,
                                 ),
            "patch_id":          row["patch_id"],
        }

    # =========================================================================
    # VIZ SAMPLE (no augmentation, no subsampling randomness)
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        """Return a deterministic sample (augmentation off) for visualisation."""
        # Temporarily disable augmentation
        was_enabled = self.augmenter.enabled
        self.augmenter.enabled = False
        sample = self.__getitem__(index)
        self.augmenter.enabled = was_enabled
        return sample
