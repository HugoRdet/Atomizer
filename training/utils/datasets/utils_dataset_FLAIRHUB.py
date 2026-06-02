"""
FLAIR-HUB Atomizer Dataset
============================

Multi-modal multi-resolution land cover segmentation on FLAIR-HUB.

Modalities:
    - AERIAL_RGBI (VHR):   4 ch @ 0.2m, 512×512  (RGB + NIR, uint8)
    - SPOT_RGBI:           4 ch @ 1.6m, 64×64    (RGB + NIR, uint16)
    - DEM_ELEV:            2 ch @ 0.2m, 512×512  (DSM + DTM, float32)
    - SENTINEL2_TS:       10 ch × T_s2 timesteps @ 10m, 10×10   (variable T)
    - SENTINEL1-ASC_TS:    2 ch × T_s1a timesteps @ 10m, 10×10  (variable T)
    - SENTINEL1-DESC_TS:   2 ch × T_s1d timesteps @ 10m, 10×10  (variable T)

Labels:
    - AERIAL_LABEL-COSIA: per-pixel @ 0.2m, 19 classes (uint8)

Resolution groups (Atomizer-native multi-resolution):
    0.2m: VHR + DEM (when both used)         → 4 + 2 = 6 channels
          When spot_as_vhr=True, SPOT joins this group (upsampled to 512×512)
    1.6m: SPOT (when spot_as_vhr=False)       → 4 channels
    10.0m: S2 + S1 ASC + S1 DESC time series → (10 + 2 + 2) channels × ~6 timesteps

Cross-sensor transfer (the headline experiment):
    Train with use_vhr=True,  use_spot=False
    Test  with use_vhr=False, use_spot=True, spot_as_vhr=True
    Same model checkpoint, different test-time modality flags.

    `spot_as_vhr=True` upsamples SPOT 64×64 → 512×512 and tags tokens with
    the VHR resolution_idx (0.2m), so the optical input maintains the same
    spatial density as during training. The spectral metadata stays SPOT-
    specific (different bandwidth/wavelength from VHR), so the model still
    knows it's looking at a different sensor — just at the same spatial
    scale.

D4 augmentation (training only):
    Each __getitem__ samples ONE D4 transform (rotation 0/90/180/270° +
    optional horizontal/vertical flips) and applies it consistently to
    every modality and the label. Cross-modal spatial alignment is
    preserved because all modalities share the same geographic patch
    geometry. Augmentation is sampled deterministically from index +
    worker_id (reproducible), and identity transform on val/test.

Token format (8 cols, same across all modalities):
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
"""

import os
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Warning] pandas not installed — required for FLAIR-HUB CSV reading.")

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("[Warning] geopandas not installed — required for FLAIR-HUB date metadata.")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed — required for FLAIR-HUB TIF reading.")

from .token_grouping import *
from .token_builder import TokenBuilder
from .augmentations import D4Augmentation, D4Transform


class FlairHubDataset(Dataset):
    """
    FLAIR-HUB multi-modal segmentation, Atomizer format.

    Args:
        root_path:       FLAIR-HUB data root (containing data/, extracted/,
                         and the three split CSVs + normalization_stats.json).
        mode:            "train" | "validation" | "test".
        dataset_config:  YAML dict; reads bands_aerial_info, bands_spot_info,
                         bands_sen2_info, bands_dem_info, plus existing
                         bands for S1 (VV/VH abstract channels via lookup).
        config_model:    Atomizer config dict.
        look_up:         Lookup_encoding instance (shared with other datasets).
        use_vhr:         Use AERIAL_RGBI (4 ch @ 0.2m).
        use_spot:        Use SPOT_RGBI (4 ch @ 1.6m). Default off — flip on
                         for cross-sensor transfer test.
        spot_as_vhr:     If True, upsample SPOT 64×64 → 512×512 (bilinear)
                         and tokenize at VHR resolution (0.2m, vhr_resolution_idx).
                         SPOT joins the 0.2m group alongside DEM (and VHR if
                         enabled). Spectral indices stay SPOT-specific (different
                         bandwidth/wavelength signature). Default False.
        use_dem:         Use DEM_ELEV (2 ch @ 0.2m, DSM + DTM).
        use_s2:          Use SENTINEL2_TS time series.
        use_s1:          Use SENTINEL1-ASC_TS + SENTINEL1-DESC_TS time series.
        multi_temporal:  Number of timesteps to sample evenly via linspace.
        max_queries:     Subsample queries per sample (memory).
        use_augmentation: Enable D4 augmentation in training split. Defaults
                         True; auto-disabled for val/test regardless.
    """

    # Resolutions (m/px). Single canonical values for resolution_idx lookup.
    VHR_RESOLUTION  = 0.2     # AERIAL_RGBI, DEM, AERIAL_LABEL
    SPOT_RESOLUTION = 1.6
    SAT_RESOLUTION  = 10.0    # S2 and S1 (raw is 10.24, rounded for lookup)

    # Patch sizes per modality (px)
    VHR_SIZE  = 512
    SPOT_SIZE = 64
    SAT_SIZE  = 10

    # Class count for AERIAL_LABEL-COSIA. FLAIR-HUB convention: 19 classes,
    # background+ignore mapped to IGNORE_INDEX. Adjust to your benchmark version.
    NUM_CLASSES  = 19
    IGNORE_INDEX = 255
    TIME_IDX_NA  = -1

    # Reference date for DOY computation (matches PASTIS-HD convention)
    REFERENCE_DATE = datetime(2018, 9, 1)

    # FLAIR-HUB COSIA class names (for logging; adapt to your benchmark)
    COSIA_CLASSES = [
        "building",            # 0
        "pervious_surface",    # 1
        "impervious_surface",  # 2
        "bare_soil",           # 3
        "water",               # 4
        "coniferous",          # 5
        "deciduous",           # 6
        "brushwood",           # 7
        "vineyard",            # 8
        "herbaceous",          # 9
        "agricultural",        # 10
        "plowed",              # 11
        "swimming_pool",       # 12
        "snow",                # 13
        "clearcut",            # 14
        "mixed",               # 15
        "ligneous",            # 16
        "greenhouse",          # 17
        "other",               # 18
    ]

    SPLIT_FILES = {
        "train":      "FLAIR-HUB_TRAIN.csv",
        "validation": "FLAIR-HUB_VALID.csv",
        "test":       "FLAIR-HUB_TEST.csv",
    }

    def __init__(
        self,
        root_path: str = "./data/FLAIR-HUB",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config=None,
        config_model=None,
        look_up=None,
        use_vhr: bool = True,
        use_spot: bool = False,
        spot_as_vhr: bool = False,
        spot_norm_as_vhr: bool = False,
        use_dem: bool = True,
        use_s2: bool = True,
        use_s1: bool = True,
        multi_temporal: int = 6,
        max_queries: int = 100_000,
        use_augmentation: bool = True,
    ):
        super().__init__()
        for lib, ok in [("rasterio", HAS_RASTERIO),
                        ("pandas", HAS_PANDAS),
                        ("geopandas", HAS_GEOPANDAS)]:
            if not ok:
                raise ImportError(f"{lib} required for FLAIR-HUB dataset")
        if not (use_vhr or use_spot):
            raise ValueError("At least one of use_vhr / use_spot must be True.")
        if spot_as_vhr and not use_spot:
            print("[FLAIR-HUB] WARNING: spot_as_vhr=True but use_spot=False, "
                  "the flag will have no effect.")
        if spot_norm_as_vhr and not use_spot:
            print("[FLAIR-HUB] WARNING: spot_norm_as_vhr=True but use_spot=False, "
                  "the flag will have no effect.")

        self.root_path      = root_path
        self.split          = mode
        self.look_up        = look_up
        self.config_model   = config_model
        self.dataset_config = dataset_config
        self.use_vhr        = use_vhr
        self.use_spot       = use_spot
        self.spot_as_vhr    = spot_as_vhr
        self.spot_norm_as_vhr = spot_norm_as_vhr
        self.use_dem        = use_dem
        self.use_s2         = use_s2
        self.use_s1         = use_s1
        self.multi_temporal = multi_temporal

        self.token_builder = TokenBuilder(look_up)

        # ── D4 augmentation ─────────────────────────────────
        # Only active in training split. One transform is sampled per
        # __getitem__ and applied identically to all modalities + label.
        # Val/test always get identity transform.
        self.augmenter = D4Augmentation(
            enabled=(use_augmentation and self.split == "train"),
            p_flip_h=0.5,
            p_flip_v=0.5,
        )
        if self.augmenter.enabled:
            print(f"[FLAIR-HUB] D4 augmentation ENABLED (training split)")
        else:
            print(f"[FLAIR-HUB] D4 augmentation DISABLED "
                  f"(split={self.split}, use_augmentation={use_augmentation})")

        # Token budgets from config
        self.nb_tokens                  = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction  = config_model["trainer"].get(
            "max_tokens_reconstruction", max_queries)

        # ── Resolution indices ──────────────────────────────
        # All used resolutions must be registered in the lookup table by the
        # launch script's register_all_resolutions() call.
        self.vhr_resolution_idx  = look_up.get_resolution_idx(self.VHR_RESOLUTION)
        self.spot_resolution_idx = look_up.get_resolution_idx(self.SPOT_RESOLUTION)
        self.sat_resolution_idx  = look_up.get_resolution_idx(self.SAT_RESOLUTION)

        # ── Read split CSV ──────────────────────────────────
        self._load_split_manifest()

        # ── Load global date metadata (once) ────────────────
        # GeoPackage tables, ~241k rows each. Read full and index by patch_id
        # for O(1) lookup at __getitem__ time.
        self._load_date_metadata()

        # ── Load normalization stats ────────────────────────
        self._load_normalization_stats()

        # ── Build per-modality spectral indices ─────────────
        # Each modality's (bw, wl) pairs map to spectral_idx in the lookup
        # table. Same path as PASTIS-HD/Sen1Floods11 — guarantees that bands
        # with identical (bw, wl) keys (e.g., S2 B02 across all datasets)
        # share the same spectral_idx.
        self._setup_band_indices()

        # ── Print summary ───────────────────────────────────
        print(f"[FLAIR-HUB] Loaded {len(self.patch_rows)} patches, "
              f"split='{self.split}'")
        modality_str = []
        if self.use_vhr:  modality_str.append(f"VHR({self.NUM_VHR_BANDS}ch@{self.VHR_RESOLUTION}m)")
        if self.use_spot:
            if self.spot_as_vhr:
                res_label = f"@{self.VHR_RESOLUTION}m [resized from {self.SPOT_RESOLUTION}m]"
            else:
                res_label = f"@{self.SPOT_RESOLUTION}m"
            norm_label = " [VHR-norm]" if self.spot_norm_as_vhr else ""
            modality_str.append(
                f"SPOT({self.NUM_SPOT_BANDS}ch{res_label}{norm_label})"
            )
        if self.use_dem:  modality_str.append(f"DEM({self.NUM_DEM_BANDS}ch@{self.VHR_RESOLUTION}m)")
        if self.use_s2:   modality_str.append(f"S2({self.NUM_S2_BANDS}ch×T@{self.SAT_RESOLUTION}m)")
        if self.use_s1:   modality_str.append(f"S1({self.NUM_S1_BANDS}ch×2T@{self.SAT_RESOLUTION}m)")
        print(f"[FLAIR-HUB] Modalities: {' + '.join(modality_str)}")
        print(f"[FLAIR-HUB] Temporal: {self.multi_temporal} timesteps "
              f"(linspace sampling)")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _load_split_manifest(self):
        """Read the per-split CSV listing patches and their modality paths."""
        split_csv = os.path.join(self.root_path, self.SPLIT_FILES[self.split])
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        # Semicolon-separated (FLAIR-HUB convention)
        self.split_df = pd.read_csv(split_csv, sep=";")
        # Convert to list of dicts for fast iteration
        self.patch_rows = self.split_df.to_dict("records")

    def _resolve_path(self, rel_path: str) -> str:
        """
        Resolve a CSV-relative path to an absolute filesystem path.

        CSV paths use `../<modality_dir>/<region>/<file>.tif`. The actual
        files live under `<root_path>/extracted/<modality_dir>/...`.
        Strip leading `../` and prepend the extracted root.
        """
        rel_clean = rel_path
        # Strip any number of leading "../" or "./"
        while rel_clean.startswith(("../", "./")):
            if rel_clean.startswith("../"):
                rel_clean = rel_clean[3:]
            else:
                rel_clean = rel_clean[2:]
        return os.path.join(self.root_path, "extracted", rel_clean)

    def _load_date_metadata(self):
        """
        Load acquisition dates per patch from GLOBAL_*_MTD_DATES.gpkg.

        Each gpkg has columns: patch_id, acquisition_dates (JSON string), geometry.
        We parse the JSON once and store as {patch_id: {"1": "20210114", ...}}.
        """
        mtd_root = os.path.join(self.root_path, "extracted", "GLOBAL_ALL_MTD")
        if not os.path.isdir(mtd_root):
            raise FileNotFoundError(
                f"GLOBAL_ALL_MTD/ not found at {mtd_root}. Extract GLOBAL_ALL_MTD.zip."
            )

        date_files = {
            "s2":     "GLOBAL_SENTINEL2_MTD_DATES.gpkg",
            "s1_asc": "GLOBAL_SENTINEL1-ASC_MTD_DATES.gpkg",
            "s1_desc": "GLOBAL_SENTINEL1-DESC_MTD_DATES.gpkg",
        }

        self.dates_per_modality = {}
        for mod, fname in date_files.items():
            path = os.path.join(mtd_root, fname)
            if not os.path.exists(path):
                print(f"[FLAIR-HUB] WARNING: {fname} not found, "
                      f"{mod} timesteps will use TIME_IDX_NA.")
                self.dates_per_modality[mod] = {}
                continue

            # Load geopackage (slow — ~minutes for 241k rows × 3-4 modalities,
            # but only at dataset construction).
            print(f"[FLAIR-HUB] Loading {fname}...")
            gdf = gpd.read_file(path)
            # Build {patch_id: {timestep_str: date_str}} dict
            patch_to_dates = {}
            for _, row in gdf.iterrows():
                pid = row["patch_id"]
                ad  = row["acquisition_dates"]
                if isinstance(ad, str):
                    try:
                        ad = json.loads(ad)
                    except Exception:
                        ad = {}
                patch_to_dates[pid] = ad
            self.dates_per_modality[mod] = patch_to_dates
            print(f"[FLAIR-HUB]   loaded dates for {len(patch_to_dates)} patches")

    def _load_normalization_stats(self):
        """Load per-modality normalization stats from JSON."""
        norm_path = os.path.join(self.root_path, "normalization_stats.json")
        if not os.path.exists(norm_path):
            print(f"[FLAIR-HUB] WARNING: normalization_stats.json not found. "
                  f"Using identity normalization.")
            self.norm_stats = {}
            return

        with open(norm_path) as f:
            self.norm_stats = json.load(f)
        print(f"[FLAIR-HUB] Norm stats loaded for: "
              f"{list(self.norm_stats.keys())}")

    def _setup_band_indices(self):
        """Build spectral indices per modality from the YAML config."""
        # VHR (AERIAL_RGBI): 4 channels (R, G, B, NIR)
        if self.use_vhr:
            self.vhr_spectral_indices = self._parse_band_yaml(
                "bands_aerial_info", expected_count=4)
            self.NUM_VHR_BANDS = len(self.vhr_spectral_indices)
        else:
            self.NUM_VHR_BANDS = 0

        # SPOT_RGBI: 4 channels
        if self.use_spot:
            self.spot_spectral_indices = self._parse_band_yaml(
                "bands_spot_info", expected_count=4)
            self.NUM_SPOT_BANDS = len(self.spot_spectral_indices)
        else:
            self.NUM_SPOT_BANDS = 0

        # DEM: 2 channels (DSM, DTM)
        if self.use_dem:
            self.dem_spectral_indices = self._parse_band_yaml(
                "bands_dem_info", expected_count=2)
            self.NUM_DEM_BANDS = len(self.dem_spectral_indices)
        else:
            self.NUM_DEM_BANDS = 0

        # Sentinel-2: 10 bands (FLAIR-HUB stores 10 bands per timestep,
        # not 13 — B01/B09/B10 typically excluded).
        if self.use_s2:
            self.s2_spectral_indices = self._parse_band_yaml(
                "bands_sen2_info", expected_count=None)
            # FLAIR-HUB stores B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (10).
            # Filter to those if YAML has all 13.
            self.s2_spectral_indices = self._filter_s2_to_flair_subset(
                self.s2_spectral_indices)
            self.NUM_S2_BANDS = len(self.s2_spectral_indices)
        else:
            self.NUM_S2_BANDS = 0

        # Sentinel-1: VV + VH (abstract channels). Same spectral_idx for ASC and DESC.
        # Register them once if not already in lookup.
        if self.use_s1:
            if hasattr(self.look_up, "register_abstract_channel"):
                self.look_up.register_abstract_channel("VV")
                self.look_up.register_abstract_channel("VH")
            # Look up directly from the table_wave (matches Sen1Floods11 convention)
            vv_key = (-1, -1)
            vh_key = (-2, -2)
            if vv_key in self.look_up.table_wave and vh_key in self.look_up.table_wave:
                self.s1_spectral_indices = torch.tensor(
                    [self.look_up.table_wave[vv_key],
                     self.look_up.table_wave[vh_key]],
                    dtype=torch.long,
                )
            else:
                raise KeyError(
                    f"[FLAIR-HUB] S1 VV/VH spectral_idx missing from lookup. "
                    f"Ensure register_abstract_channel('VV')/('VH') "
                    f"or matching bands_*_info entries exist."
                )
            self.NUM_S1_BANDS = 2
        else:
            self.NUM_S1_BANDS = 0

    def _parse_band_yaml(self, yaml_key: str, expected_count: int = None):
        """Same parse-and-lookup logic as Sen1Floods11._build_spectral_indices."""
        if yaml_key not in self.dataset_config:
            raise KeyError(
                f"[FLAIR-HUB] '{yaml_key}' missing from bands.yaml. "
                f"Available: {list(self.dataset_config.keys())}"
            )
        all_bands = []
        for name, data in self.dataset_config[yaml_key].items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        if expected_count is not None and len(all_bands) != expected_count:
            print(f"[FLAIR-HUB] WARNING: '{yaml_key}' has {len(all_bands)} "
                  f"bands, expected {expected_count}.")

        indices = []
        for band in all_bands:
            key = (band["bandwidth"], band["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[FLAIR-HUB] {yaml_key}/{band['name']} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])

        print(f"[FLAIR-HUB] {yaml_key}: {len(indices)} bands → "
              f"spectral_idx={indices}")
        return torch.tensor(indices, dtype=torch.long)

    def _filter_s2_to_flair_subset(self, full_s2_indices: torch.Tensor):
        """
        FLAIR-HUB stores 10 S2 bands per timestep (380 / 38 = 10).
        Filter the full 13-band YAML list down to the 10 used by FLAIR-HUB.

        Standard FLAIR-HUB band order: B02, B03, B04, B05, B06, B07, B08, B8A,
        B11, B12 (excludes B01, B09, B10).
        """
        FLAIR_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
                          "B08", "B8A", "B11", "B12"]
        bands_yaml = self.dataset_config.get("bands_sen2_info", {})

        # Build name → idx lookup from the full YAML
        name_to_yaml_idx = {name: data["idx"]
                            for name, data in bands_yaml.items()
                            if isinstance(data, dict) and "idx" in data}

        # Map FLAIR-HUB's band names to their position in the original
        # full_s2_indices tensor. full_s2_indices is sorted by yaml idx,
        # so position == yaml idx.
        filtered = []
        for band_name in FLAIR_S2_BANDS:
            if band_name not in name_to_yaml_idx:
                # Try matching by wavelength as fallback (B8A vs 8A naming)
                print(f"[FLAIR-HUB] WARNING: S2 band '{band_name}' "
                      f"not in YAML. Using B08 as proxy.")
                continue
            yaml_idx = name_to_yaml_idx[band_name]
            if yaml_idx < len(full_s2_indices):
                filtered.append(full_s2_indices[yaml_idx].item())

        return torch.tensor(filtered, dtype=torch.long)

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, index):
        row = self.patch_rows[index]
        patch_id = row["patch_id"]

        # ── Sample D4 augmentation ONCE per item ───────────────
        # Same transform passed to every modality below — guarantees
        # cross-modal spatial alignment. Val/test returns identity.
        aug = self.augmenter.sample(index=index)

        groups = {}

        # ── Load and augment label first ───────────────────────
        # Augmenting the label here means every downstream modality
        # tokenizer that uses `label` (VHR, DEM, SPOT@VHR) already
        # sees the augmented version — no need to re-augment.
        label = self._load_label(row)                          # [H, W] uint8 (numpy)
        label = self._remap_label(label)
        label = torch.from_numpy(label.astype(np.int64))
        label = self.augmenter.apply(label, aug)               # [H, W] augmented

        # ── 0.2m group: VHR + DEM (+ SPOT-resized if spot_as_vhr) ──
        hires_tokens_list = []
        hires_channels    = 0

        if self.use_vhr:
            vhr_tokens = self._load_and_tokenize_vhr(row, label, aug=aug)
            hires_tokens_list.append(vhr_tokens)
            hires_channels += self.NUM_VHR_BANDS

        if self.use_spot and self.spot_as_vhr:
            # SPOT resized to 512×512 and tokenized as a VHR-resolution sensor.
            # Joins the 0.2m group; its tokens share latent placement with VHR/DEM.
            spot_tokens, _ = self._load_and_tokenize_spot(row, label, aug=aug)
            hires_tokens_list.append(spot_tokens)
            hires_channels += self.NUM_SPOT_BANDS

        if self.use_dem:
            dem_tokens = self._load_and_tokenize_dem(row, label, aug=aug)
            hires_tokens_list.append(dem_tokens)
            hires_channels += self.NUM_DEM_BANDS

        if hires_tokens_list:
            hires_tokens = torch.cat(hires_tokens_list, dim=0)
            groups[self.VHR_RESOLUTION] = {
                "tokens": hires_tokens,
                "mask":   torch.zeros(hires_tokens.shape[0], dtype=torch.bool),
                "shape":  (hires_channels, self.VHR_SIZE, self.VHR_SIZE),
            }

        # ── 1.6m group: SPOT (native, only when spot_as_vhr=False) ──
        if self.use_spot and not self.spot_as_vhr:
            spot_tokens, _ = self._load_and_tokenize_spot(row, label, aug=aug)
            groups[self.SPOT_RESOLUTION] = {
                "tokens": spot_tokens,
                "mask":   torch.zeros(spot_tokens.shape[0], dtype=torch.bool),
                "shape":  (self.NUM_SPOT_BANDS, self.SPOT_SIZE, self.SPOT_SIZE),
            }

        # ── 10m group: S2 + S1 ASC + S1 DESC time series ────
        # All three get the SAME transform `aug`. Across timesteps within a
        # single stack, the transform is also identical (applied to the
        # whole [T, B, H, W] tensor). This keeps temporal coherence: a
        # north-south road stays N-S across all timesteps after rotation.
        sat_tokens_list = []
        if self.use_s2:
            s2_tokens = self._load_and_tokenize_s2_ts(row, patch_id, aug=aug)
            sat_tokens_list.append(s2_tokens)

        if self.use_s1:
            s1a_tokens = self._load_and_tokenize_s1_ts(row, patch_id, mode="asc",  aug=aug)
            s1d_tokens = self._load_and_tokenize_s1_ts(row, patch_id, mode="desc", aug=aug)
            sat_tokens_list.append(s1a_tokens)
            sat_tokens_list.append(s1d_tokens)

        if sat_tokens_list:
            sat_tokens = torch.cat(sat_tokens_list, dim=0)
            sat_total_bands = (self.NUM_S2_BANDS + 2 * self.NUM_S1_BANDS) * self.multi_temporal
            groups[self.SAT_RESOLUTION] = {
                "tokens": sat_tokens,
                "mask":   torch.zeros(sat_tokens.shape[0], dtype=torch.bool),
                "shape":  (sat_total_bands, self.SAT_SIZE, self.SAT_SIZE),
            }

        # ── Build queries from (already augmented) label at VHR resolution ──
        # Queries follow PASTIS convention: per-pixel at the highest available
        # spatial resolution. Subsample to max_tokens_reconstruction.
        # Pick the first spectral_idx from whatever 0.2m source is active.
        # The label is ALREADY augmented above, so queries will be
        # consistent with all the augmented modality tokens.
        if self.use_vhr:
            first_spectral = self.vhr_spectral_indices[0]
        elif self.use_spot and self.spot_as_vhr:
            first_spectral = self.spot_spectral_indices[0]
        elif self.use_dem:
            first_spectral = self.dem_spectral_indices[0]
        else:
            first_spectral = None

        queries = self.token_builder.build_queries(
            label=label,
            resolution=self.VHR_RESOLUTION,
            first_spectral_idx=first_spectral,
            resolution_idx=self.vhr_resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        if self.split == "train":
            queries = self.token_builder.subsample_queries(
                queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
            )
        # else: full per-pixel queries on val/test (decoder chunks internally)
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        # ── Image for visualization (also augmented for consistency) ──
        # If we returned the un-augmented image alongside augmented tokens,
        # plotting predictions over the image would look misaligned.
        if self.use_vhr:
            image = self._load_vhr_image(row)
            image = self.augmenter.apply(image, aug)
        elif self.use_spot:
            image, _ = self._load_spot_image(row)
            image = self.augmenter.apply(image, aug)
        else:
            image = torch.zeros(1, self.VHR_SIZE, self.VHR_SIZE)

        return {
            "groups":            groups,
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label,
            "target_resolution": self.VHR_RESOLUTION,
            "image":             image,
        }

    # =========================================================================
    # PER-MODALITY LOADERS
    # =========================================================================

    def _load_label(self, row) -> np.ndarray:
        """Load AERIAL_LABEL-COSIA TIF as numpy [H, W]."""
        path = self._resolve_path(row["AERIAL_LABEL-COSIA"])
        with rasterio.open(path) as src:
            return src.read(1).astype(np.int64)

    def _remap_label(self, label: np.ndarray) -> np.ndarray:
        """Remap any label outside [0, NUM_CLASSES) to IGNORE_INDEX."""
        valid = (label >= 0) & (label < self.NUM_CLASSES)
        label = np.where(valid, label, self.IGNORE_INDEX)
        return label

    # ── VHR ─────────────────────────────────────────────────

    def _load_vhr_image(self, row) -> torch.Tensor:
        """Load AERIAL_RGBI as [C, H, W] float32, normalized."""
        path = self._resolve_path(row["AERIAL_RGBI"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)                # [4, 512, 512]
        img = torch.from_numpy(img)
        img = self._normalize(img, "aerial")
        img = torch.clamp(img, -10, 10)
        return torch.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)

    def _load_and_tokenize_vhr(self, row, label, aug: D4Transform = None):
        """
        Load + tokenize VHR. Returns [C*H*W, 8] tokens.

        Augmentation is applied AFTER normalization and BEFORE tokenization.
        The `label` arg is expected to already be augmented (done at the top
        of __getitem__).
        """
        image = self._load_vhr_image(row)
        if aug is not None:
            image = self.augmenter.apply(image, aug)
        return self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.vhr_spectral_indices,
            resolution_idx=self.vhr_resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

    # ── SPOT ────────────────────────────────────────────────

    def _load_spot_image(self, row):
        """
        Load SPOT_RGBI. Returns (image [C, H, W], label [H, W]).

        Behavior depends on `self.spot_as_vhr`:
          False (default): native 64×64 image, label nearest-downsampled to
                           64×64 to match SPOT spatial size.
          True:            bilinearly upsampled 64×64 → 512×512 image,
                           full-resolution 512×512 label preserved.
                           Used for cross-sensor transfer where we want the
                           SPOT input to occupy the same token grid as VHR.

        Note: this method returns the RAW (un-augmented) image and label.
        Augmentation is applied by the caller (_load_and_tokenize_spot)
        AFTER resizing so the transform operates on the final geometry.
        """
        path = self._resolve_path(row["SPOT_RGBI"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)                # [4, 64, 64]
        img = torch.from_numpy(img)
        # Normalization: by default use SPOT's own stats. When
        # spot_norm_as_vhr=True, normalize using VHR (aerial) stats instead,
        # which is a diagnostic for whether cross-sensor degradation is
        # driven by pixel-value distribution shift rather than other
        # architectural factors.
        norm_key = "aerial" if self.spot_norm_as_vhr else "spot"
        img = self._normalize(img, norm_key)
        img = torch.clamp(img, -10, 10)
        img = torch.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)

        # Load label at full VHR resolution (always).
        label_path = self._resolve_path(row["AERIAL_LABEL-COSIA"])
        with rasterio.open(label_path) as src:
            label_full = src.read(1).astype(np.int64)
        label_full = self._remap_label(label_full)
        label_full_t = torch.from_numpy(label_full)

        if self.spot_as_vhr:
            # Bilinear upsample SPOT 64×64 → 512×512 for cross-sensor test.
            # Token density now matches VHR training.
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(self.VHR_SIZE, self.VHR_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)                                       # [4, 512, 512]
            # Full-resolution label, no downsampling.
            return img, label_full_t.long()

        # Default: native 64×64 SPOT with downsampled label.
        label_spot = torch.nn.functional.interpolate(
            label_full_t.float().unsqueeze(0).unsqueeze(0),
            size=(self.SPOT_SIZE, self.SPOT_SIZE),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        return img, label_spot

    def _load_and_tokenize_spot(self, row, label_full, aug: D4Transform = None):
        """
        Tokenize SPOT. Resolution_idx depends on `spot_as_vhr`:
          False: SPOT_RESOLUTION (1.6m), spot_resolution_idx
                 → tokens go in the 1.6m group.
          True:  VHR_RESOLUTION (0.2m), vhr_resolution_idx
                 → tokens go in the 0.2m group alongside VHR/DEM.
                 Spectral indices stay SPOT-specific (different
                 bandwidth/wavelength than VHR), so the model still
                 sees a sensor mismatch via spectral metadata.

        Augmentation: applied to both image and SPOT-resolution label AFTER
        all resizing is complete. When spot_as_vhr=True the label arrives
        already at VHR resolution and is replaced by `label_full` (which is
        already augmented).
        """
        image, label_spot = self._load_spot_image(row)

        # Apply augmentation to image at its final size (post-resize).
        # The label needs careful handling:
        #   - spot_as_vhr=True : label_spot here is the 512×512 label from
        #     _load_spot_image; but `label_full` passed in is ALREADY the
        #     augmented version at 512×512 — we use that for consistency.
        #   - spot_as_vhr=False: label_spot is downsampled to 64×64 from a
        #     RAW label (loaded inside _load_spot_image). We must augment
        #     it here ourselves since it's at SPOT resolution.
        if aug is not None:
            image = self.augmenter.apply(image, aug)

        if self.spot_as_vhr:
            # Use the already-augmented label passed in by __getitem__.
            # `label_full` here is the 512×512 augmented label.
            tokens = self.token_builder.build_tokens(
                image=image,
                label=label_full,
                resolution=self.VHR_RESOLUTION,                # 0.2m
                spectral_indices=self.spot_spectral_indices,   # SPOT bands
                resolution_idx=self.vhr_resolution_idx,        # VHR's idx
                time_idx=self.TIME_IDX_NA,
            )
            return tokens, label_full

        # Native SPOT: label_spot is 64×64 from RAW label inside
        # _load_spot_image. We must augment it ourselves to match the
        # augmented image at the same resolution.
        if aug is not None:
            label_spot = self.augmenter.apply(label_spot, aug)

        tokens = self.token_builder.build_tokens(
            image=image,
            label=label_spot,
            resolution=self.SPOT_RESOLUTION,
            spectral_indices=self.spot_spectral_indices,
            resolution_idx=self.spot_resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        return tokens, label_spot

    # ── DEM ─────────────────────────────────────────────────

    def _load_and_tokenize_dem(self, row, label, aug: D4Transform = None):
        """
        Load DEM (DSM band 1, DTM band 2) as [2, H, W] float32, normalize,
        tokenize. DEM is at VHR resolution (0.2m, 512×512).

        Augmentation applied AFTER normalization, before tokenization.
        The label argument is expected to already be augmented (done at the
        top of __getitem__).
        """
        path = self._resolve_path(row["DEM_ELEV"])
        with rasterio.open(path) as src:
            dem = src.read().astype(np.float32)                # [2, 512, 512]
        dem = torch.from_numpy(dem)

        # Normalize per channel using running stats from sample 0.
        # If norm_stats has 'dem', use it. Otherwise standardize per-image
        # (z-score with this sample's mean/std — coarse but bounded).
        dem = self._normalize_dem(dem)
        dem = torch.clamp(dem, -10, 10)
        dem = torch.nan_to_num(dem, nan=0.0, posinf=10.0, neginf=-10.0)

        if aug is not None:
            dem = self.augmenter.apply(dem, aug)

        return self.token_builder.build_tokens(
            image=dem,
            label=label,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.dem_spectral_indices,
            resolution_idx=self.vhr_resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

    def _normalize_dem(self, dem: torch.Tensor) -> torch.Tensor:
        """
        Z-score per-channel normalization. Per-image std is brittle
        (a flat patch has near-zero std → div by epsilon → huge values).
        We use a fixed std=50m as a reasonable elevation scale, and
        per-image mean (centers each patch around its own elevation).
        """
        if "dem" in self.norm_stats:
            stats = self.norm_stats["dem"]
            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)
            return (dem - mean) / std

        # Fallback: per-image mean, fixed std
        per_image_mean = dem.mean(dim=(1, 2), keepdim=True)
        return (dem - per_image_mean) / 50.0

    # ── Sentinel-2 ──────────────────────────────────────────

    def _load_and_tokenize_s2_ts(self, row, patch_id, aug: D4Transform = None):
        """
        Load S2 TS, sample multi_temporal timesteps via linspace, tokenize.
        Returns [num_bands * T * H * W, 8].

        Augmentation: applied to the FULL [T, B, H, W] stack AFTER
        normalization. The D4Augmentation handler operates on the last 2
        axes (H, W), so T and B are preserved and every timestep receives
        the same spatial transform. This keeps temporal coherence — a
        north-south road stays N-S across all timesteps after rotation.
        """
        path = self._resolve_path(row["SENTINEL2_TS"])
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32)              # [T*B, H, W]
        T_total = stack.shape[0] // self.NUM_S2_BANDS
        # Reshape: [T, B, H, W]
        stack = stack.reshape(T_total, self.NUM_S2_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)

        # Sample T timesteps and align dates
        sample_idx = self._linspace_sample(T_total, self.multi_temporal)
        stack = stack[sample_idx]                              # [T_sel, B, H, W]
        time_indices = self._get_time_indices(patch_id, "s2", sample_idx)

        # Convert and normalize
        stack = torch.from_numpy(stack)
        stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        stack = self._normalize_per_timestep(stack, "s2")
        stack = torch.clamp(stack, -10, 10)

        # Same spatial transform applied to every timestep simultaneously
        if aug is not None:
            stack = self.augmenter.apply(stack, aug)           # [T_sel, B, H, W]

        return self._tokenize_temporal(
            stack=stack,
            spectral_indices=self.s2_spectral_indices,
            time_indices=time_indices,
        )

    # ── Sentinel-1 ──────────────────────────────────────────

    def _load_and_tokenize_s1_ts(self, row, patch_id, mode: str, aug: D4Transform = None):
        """
        Load S1 ASC or DESC TS, sample timesteps, tokenize.

        Augmentation: applied to the FULL [T, B, H, W] stack. Same transform
        as VHR / SPOT / DEM / S2 / label — guarantees that radar features
        align with the optical and elevation features after rotation/flip.
        """
        if mode == "asc":
            col, dates_key = "SENTINEL1-ASC_TS", "s1_asc"
        elif mode == "desc":
            col, dates_key = "SENTINEL1-DESC_TS", "s1_desc"
        else:
            raise ValueError(f"Unknown S1 mode: {mode}")

        path = self._resolve_path(row[col])
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32)              # [T*2, H, W]
        T_total = stack.shape[0] // self.NUM_S1_BANDS
        stack = stack.reshape(T_total, self.NUM_S1_BANDS,
                              self.SAT_SIZE, self.SAT_SIZE)

        sample_idx = self._linspace_sample(T_total, self.multi_temporal)
        stack = stack[sample_idx]
        time_indices = self._get_time_indices(patch_id, dates_key, sample_idx)

        stack = torch.from_numpy(stack)
        stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        stack = self._normalize_per_timestep(stack, dates_key)
        stack = torch.clamp(stack, -10, 10)

        # Same spatial transform applied to every timestep simultaneously
        if aug is not None:
            stack = self.augmenter.apply(stack, aug)           # [T_sel, B, H, W]

        return self._tokenize_temporal(
            stack=stack,
            spectral_indices=self.s1_spectral_indices,
            time_indices=time_indices,
        )

    # =========================================================================
    # TOKEN HELPERS
    # =========================================================================

    def _tokenize_temporal(self, stack: torch.Tensor,
                           spectral_indices: torch.Tensor,
                           time_indices: torch.Tensor):
        """
        Tokenize a [T, B, H, W] stack into [T*B*H*W, 8] tokens. Each timestep
        gets its own time_idx column. Spectral indices repeat per timestep.

        We use a dummy ignore-label tensor: per-timestep tokens don't carry
        meaningful labels (labels are at VHR resolution, queries handle them).
        """
        T, B, H, W = stack.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        per_t_tokens = []
        for t in range(T):
            tokens_t = self.token_builder.build_tokens(
                image=stack[t],                                # [B, H, W]
                label=dummy_label,
                resolution=self.SAT_RESOLUTION,
                spectral_indices=spectral_indices,
                resolution_idx=self.sat_resolution_idx,
                time_idx=int(time_indices[t]),
            )
            per_t_tokens.append(tokens_t)
        return torch.cat(per_t_tokens, dim=0)

    def _linspace_sample(self, T_total: int, n: int) -> np.ndarray:
        """Same convention as PASTIS-HD._sample_temporal."""
        if T_total <= n:
            return np.arange(T_total)
        return np.linspace(0, T_total - 1, n, dtype=int)

    def _get_time_indices(self, patch_id, mod_key, sample_idx) -> torch.Tensor:
        """
        For the sampled timesteps, look up YYYYMMDD dates and convert to
        time_idx via the lookup table's get_or_register_time_idx.

        If dates aren't available for this patch, return TIME_IDX_NA for all.
        """
        dates_dict = self.dates_per_modality.get(mod_key, {})
        patch_dates = dates_dict.get(patch_id, {})
        if not patch_dates:
            return torch.full((len(sample_idx),), self.TIME_IDX_NA,
                              dtype=torch.long)

        time_idxs = []
        for ti in sample_idx:
            # Dates are 1-indexed in the JSON ("1": "20210114")
            date_str = patch_dates.get(str(int(ti) + 1)) or patch_dates.get(int(ti) + 1)
            if date_str is None:
                time_idxs.append(self.TIME_IDX_NA)
                continue
            doy = self._date_to_doy(date_str)
            time_idxs.append(self.look_up.get_or_register_time_idx(int(doy)))
        return torch.tensor(time_idxs, dtype=torch.long)

    def _date_to_doy(self, date_str: str) -> int:
        """YYYYMMDD → day-of-year (1..366)."""
        d = datetime.strptime(date_str, "%Y%m%d")
        return d.timetuple().tm_yday

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _normalize(self, img: torch.Tensor, mod_key: str) -> torch.Tensor:
        """Generic per-channel z-score using norm_stats[mod_key]."""
        if mod_key not in self.norm_stats:
            return img
        stats = self.norm_stats[mod_key]
        C = img.shape[0]
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)[:C]
        std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1)[:C].clamp(min=1e-6)
        return (img - mean) / std

    def _normalize_per_timestep(self, stack: torch.Tensor, mod_key: str) -> torch.Tensor:
        """Apply per-channel z-score to a [T, B, H, W] stack."""
        if mod_key not in self.norm_stats:
            return stack
        stats = self.norm_stats[mod_key]
        B = stack.shape[1]
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)[:, :B]
        std  = torch.tensor(stats["std"],  dtype=torch.float32).view(1, -1, 1, 1)[:, :B].clamp(min=1e-6)
        return (stack - mean) / std