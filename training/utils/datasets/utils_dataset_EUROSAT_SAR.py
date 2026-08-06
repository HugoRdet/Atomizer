"""
EuroSAT-SAR (+ paired EuroSAT_MS) Classification Dataset for Atomiser
=======================================================================

10-class land cover classification, fusing Sentinel-2 optical (13 bands)
and Sentinel-1 SAR (VV, VH) patches, paired by identical filename across
two class-folder trees.

Source layout (both trees share the same class/filename structure):
    {root_ms}/{ClassName}/{ClassName}_{id}.tif   — 13 bands, uint16, [64, 64]
    {root_sar}/{ClassName}/{ClassName}_{id}.tif  — 2 bands (VV, VH), float32, [64, 64]

Classes (EuroSAT standard, 0-indexed, alphabetical == label order):
    0=AnnualCrop, 1=Forest, 2=HerbaceousVegetation, 3=Highway, 4=Industrial,
    5=Pasture, 6=PermanentCrop, 7=Residential, 8=River, 9=SeaLake

No official split ships with the raw class-folder release, so a
stratified train/val/test split is generated once (seeded) and cached
to `split_cache.json`. Per-band normalization stats are computed once
from the train split only and cached to `normalization_stats.pt`
(mirrors the Sen1Floods11Skip pattern).

Band selection mirrors Sen1Floods11Skip: `config_model["trainer"]["bands"]`
may specify `keep` and/or `drop` band names from ALL_BAND_NAMES (15 total:
13 optical + VV + VH). Dropped bands are zeroed + masked, not removed from
the grid (so token count / shapes stay stable across configs).

Band-dropout augmentation (train only): mirrors Sen1Floods11SkipDataset's
identical mechanism — a stochastic, per-sample version of the same
zero-value + mask=1.0 drop used by the static bands.drop config above,
so training exposes Atomiser to the same kind of missingness it's tested
under at the eval-time modality-drop ablation. Configured via
config_model["trainer"]["band_dropout_augmentation"] (enabled,
p_dropout_applied, p_whole_modality, p_band_drop), same config-dict
convention as `bands`. Unlike Sen1Floods11SkipDataset, there's no
pixel-skip cascade here (classification, single CLS query per sample —
no query_token_idx/query_token_valid), so the augmentation only needs to
touch image_tokens/attention_mask, nothing decoder-side.

Output format (compatible with Model_ForestNet — classification):
    {
        "groups": {
            10.0: {
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (C, 64, 64),
            },
        },
        "queries":           [1, 8]   (single CLS query at center)
        "queries_mask":      [1],
        "label":             scalar long (0..9),
        "task":              "classification",
        "target_resolution": 10.0,
        "image":             [C, 64, 64],
    }

Augmentations (training only): D4 group on image (label is scalar), plus
band-dropout on the token pool (see above).
"""

import glob
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .token_grouping import *
from .token_builder import TokenBuilder


class EuroSATSARDataset(Dataset):
    """EuroSAT-SAR (+ paired EuroSAT_MS) 10-class classification dataset for Atomiser."""

    OPTICAL_RESOLUTION = 10.0
    NUM_MS_BANDS = 13
    NUM_SAR_BANDS = 2
    NUM_CLASSES = 10
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE = 64
    TASK_NAME = "classification"

    CLASS_NAMES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    ]

    # Raw .tif channel order for EuroSAT_MS (standard Sentinel-2 "all bands" order)
    RAW_MS_BAND_ORDER = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B08A", "B09", "B10", "B11", "B12",
    ]

    # Raw .tif channel order for EuroSAT-SAR
    RAW_SAR_BAND_ORDER = ["VV", "VH"]

    # bands_eurosat config-key name -> raw S2 band code
    NAME_TO_S2CODE = {
        "Blue": "B02", "Green": "B03", "Red": "B04", "NIR": "B08",
        "RedEdge1": "B05", "RedEdge2": "B06", "RedEdge3": "B07", "RedEdge4": "B08A",
        "SWIR1": "B11", "SWIR2": "B12",
        "CoastalAerosol": "B01", "WaterVapour": "B09", "Cirrus": "B10",
    }

    # Full band-selection namespace (15), in bands_eurosat idx order (0-12) + VV, VH (13, 14)
    ALL_BAND_NAMES = [
        "Blue", "Green", "Red", "NIR", "RedEdge1", "RedEdge2", "RedEdge3", "RedEdge4",
        "SWIR1", "SWIR2", "CoastalAerosol", "WaterVapour", "Cirrus", "VV", "VH",
    ]

    # >>> AUGMENTATION: band names belonging to the SAR modality, used to
    # split spectral indices into S1/S2 pools for whole-modality dropout.
    S1_BAND_NAMES = {"VV", "VH"}

    SPLIT_SEED = 42
    SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}

    # Sibling subfolder names under `root_path` (matches ./data/EuroSAT_MS,
    # ./data/EuroSAT-SAR layout on disk).
    MS_SUBDIR  = "EuroSAT_MS"
    SAR_SUBDIR = "EuroSAT-SAR"

    def __init__(
        self,
        root_path: str = "./data",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        root_ms: str = None,
        root_sar: str = None,
    ):
        """
        Args:
            root_path: parent directory containing both `EuroSAT_MS/` and
                `EuroSAT-SAR/` subfolders (this is what UnifiedDataModule's
                `_create_grouped_dataset` passes via `root_path=self.path`).
            root_ms / root_sar: optional explicit overrides, if the two
                trees don't live as siblings under a common root_path.
        """
        super().__init__()
        assert mode in ("train", "valid", "validation", "test"), f"Unknown split: {mode}"

        self.root_ms  = root_ms  if root_ms  is not None else os.path.join(root_path, self.MS_SUBDIR)
        self.root_sar = root_sar if root_sar is not None else os.path.join(root_path, self.SAR_SUBDIR)
        # UnifiedDataModule calls with mode="validation" (see
        # _setup_grouped_datasets); split_cache.json / sample filtering use
        # "valid" internally (matches SPLIT_RATIOS keys), so normalize here.
        self.split    = "valid" if mode == "validation" else mode
        self.look_up   = look_up
        self.config_model  = config_model
        self.dataset_config = dataset_config

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]

        # ── Band selection (Sen1Floods11-style) ─────────────────────────
        bands_cfg = config_model["trainer"].get("bands", {}) or {}
        keep_names = bands_cfg.get("keep", None)
        drop_names = bands_cfg.get("drop", None)

        if keep_names is None:
            sc = config_model["trainer"].get("single_channel", -1)
            if isinstance(sc, list):
                keep_names = [self.ALL_BAND_NAMES[i] for i in sorted(sc)]
            elif isinstance(sc, int) and sc >= 0:
                keep_names = [self.ALL_BAND_NAMES[sc]]

        self.selected_channels = self._resolve_band_names(keep_names)
        self.drop_band_names   = set(drop_names) if drop_names else set()

        # ── Pair MS + SAR samples by filename, build split ──────────────
        self.samples = self._build_sample_list()          # [(class_idx, filename), ...]
        self.split_assignment = self._load_or_build_split()  # filename -> split
        self.sample_list = [
            (cls, fn) for (cls, fn) in self.samples
            if self.split_assignment[fn] == self.split
        ]

        # ── Band metadata: combine bands_eurosat (0-12) + VV/VH from bands_senflood (13-14) ──
        self.bands_info = self._build_combined_bands_info(dataset_config)
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()

        if self.selected_channels is not None:
            self.bandwidths  = self.bandwidths[self.selected_channels]
            self.wavelengths = self.wavelengths[self.selected_channels]
            self.band_names  = [self.band_names[i] for i in self.selected_channels]

        self.spectral_indices = self._build_spectral_indices()
        self.dropped_spectral_indices = self._resolve_drop_indices()

        # >>> AUGMENTATION: split the (post band-selection) spectral index
        # pool into S1/S2 subsets, for whole-modality dropout sampling.
        # Computed once here rather than per-sample.
        s1_mask = torch.tensor(
            [name in self.S1_BAND_NAMES for name in self.band_names], dtype=torch.bool
        )
        self.s1_spectral_indices = self.spectral_indices[s1_mask]
        self.s2_spectral_indices = self.spectral_indices[~s1_mask]

        aug_cfg = config_model["trainer"].get("band_dropout_augmentation", {}) or {}
        self.band_dropout_enabled = bool(aug_cfg.get("enabled", True)) and (self.split == "train")
        self.p_dropout_applied = float(aug_cfg.get("p_dropout_applied", 0.5))
        self.p_whole_modality  = float(aug_cfg.get("p_whole_modality", 0.5))
        self.p_band_drop       = float(aug_cfg.get("p_band_drop", 0.15))
        # >>> END AUGMENTATION

        self._print_band_selection()

        # ── MS channel reorder: raw tif order -> bands_eurosat idx order ──
        self.ms_reorder_idx = torch.tensor(
            [self.RAW_MS_BAND_ORDER.index(self.NAME_TO_S2CODE[name])
             for name in self.ALL_BAND_NAMES[:self.NUM_MS_BANDS]],
            dtype=torch.long,
        )

        self.resolution_idx = self.look_up.get_resolution_idx(self.OPTICAL_RESOLUTION)

        # ── Normalization (train-split-only, cached) ─────────────────────
        self.norm_stats = self._load_or_compute_normalization()

        label_counts = Counter(cls for cls, _ in self.sample_list)
        print(f"[EuroSATSAR] task={self.TASK_NAME}, split={mode} → "
              f"{len(self.sample_list)} samples")
        print(f"[EuroSATSAR] patch size: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        print(f"[EuroSATSAR] num_classes: {self.NUM_CLASSES}")
        print(f"[EuroSATSAR] D4 augment: {'ON' if self.split == 'train' else 'OFF'}")
        if self.band_dropout_enabled:
            print(f"[EuroSATSAR] Band-dropout augmentation: ON "
                  f"(p_applied={self.p_dropout_applied}, "
                  f"p_whole_modality={self.p_whole_modality}, "
                  f"p_band_drop={self.p_band_drop})")
        else:
            print(f"[EuroSATSAR] Band-dropout augmentation: OFF")
        print(f"[EuroSATSAR] class distribution: "
              f"{ {self.CLASS_NAMES[k]: v for k, v in sorted(label_counts.items())} }")

    # =========================================================================
    # BAND SELECTION HELPERS (mirrors Sen1Floods11Skip)
    # =========================================================================

    def _resolve_band_names(self, names):
        if names is None:
            return None
        invalid = set(names) - set(self.ALL_BAND_NAMES)
        if invalid:
            raise ValueError(f"Unknown band names: {invalid}. Valid: {self.ALL_BAND_NAMES}")
        return [self.ALL_BAND_NAMES.index(n) for n in names]

    def _resolve_drop_indices(self):
        if not self.drop_band_names:
            return set()
        kept = set(self.band_names)
        unknown = self.drop_band_names - set(self.ALL_BAND_NAMES)
        if unknown:
            raise ValueError(f"bands.drop contains unknown names: {unknown}")
        not_kept = self.drop_band_names - kept
        if not_kept:
            raise ValueError(
                f"bands.drop {not_kept} are not in bands.keep {kept}. "
                f"You can only drop bands that were kept."
            )
        dropped = set()
        for name in self.drop_band_names:
            data = self.bands_info[name]
            key = (int(data["bandwidth"]), int(data["central_wavelength"]))
            if key in self.look_up.table_wave:
                dropped.add(self.look_up.table_wave[key])
            else:
                raise KeyError(f"Band '{name}' key={key} not found in lookup table.")
        return dropped

    def _print_band_selection(self):
        if self.selected_channels is None:
            kept_str = "ALL"
        else:
            kept_str = str([self.ALL_BAND_NAMES[i] for i in self.selected_channels])
        drop_str = str(sorted(self.drop_band_names)) if self.drop_band_names else "none"
        print(f"[EuroSATSAR] Bands kept    : {kept_str}")
        print(f"[EuroSATSAR] Bands dropped : {drop_str} (padding tokens, grid unchanged)")

    def _zero_and_mask_by_spectral_indices(self, tokens: torch.Tensor,
                                            mask: torch.Tensor,
                                            spectral_indices_to_drop):
        """
        Shared primitive: zero the token value and set mask=1.0 for every
        token whose spectral index (col 3) is in `spectral_indices_to_drop`.
        Used both by the static eval-time drop config (_apply_drop_mask)
        and the stochastic training-time augmentation
        (_sample_band_dropout_indices), so both go through identical
        semantics — same convention as Sen1Floods11SkipDataset.
        """
        if not spectral_indices_to_drop:
            return tokens, mask
        tokens = tokens.clone()
        mask   = mask.clone().float()
        spec_idx = tokens[:, 3]
        drop = torch.zeros(tokens.shape[0], dtype=torch.bool)
        for sid in spectral_indices_to_drop:
            drop |= (spec_idx == sid)
        tokens[drop, 0] = 0.0
        mask[drop]      = 1.0
        return tokens, mask

    def _apply_drop_mask(self, tokens: torch.Tensor, mask: torch.Tensor):
        """Static, config-driven band drop (bands.drop) — applied at every split."""
        return self._zero_and_mask_by_spectral_indices(tokens, mask, self.dropped_spectral_indices)

    # =========================================================================
    # >>> AUGMENTATION: stochastic per-sample band dropout (train only)
    # =========================================================================

    def _sample_band_dropout_indices(self):
        """
        Per-sample stochastic augmentation, mirroring
        Sen1Floods11SkipDataset's identical mixture:
          - with prob (1 - p_dropout_applied): no-op, all bands kept
          - else with prob p_whole_modality: drop ALL S1 or ALL S2 spectral
            indices (mirrors the "S2 only"/"S1 only" eval ablations)
          - else: drop each currently-kept spectral index independently
            with probability p_band_drop

        Returns a set of spectral indices to drop this sample (possibly
        empty). Layered ON TOP of the static eval-config drop, applied
        separately in __getitem__.
        """
        if torch.rand(1).item() >= self.p_dropout_applied:
            return set()

        if torch.rand(1).item() < self.p_whole_modality:
            pool = (self.s1_spectral_indices if torch.rand(1).item() < 0.5
                    else self.s2_spectral_indices)
            return set(pool.tolist())
        else:
            keep_mask = torch.rand(self.spectral_indices.shape[0]) < self.p_band_drop
            return set(self.spectral_indices[keep_mask].tolist())

    def _select_channels(self, image: torch.Tensor) -> torch.Tensor:
        if self.selected_channels is None:
            return image
        return image[self.selected_channels]

    # =========================================================================
    # D4 AUGMENTATION
    # =========================================================================

    @staticmethod
    def _d4_augment(image: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
        return image

    # =========================================================================
    # SAMPLE LIST / SPLIT
    # =========================================================================

    def _build_sample_list(self):
        """Pair MS + SAR by identical filename within each class folder."""
        samples = []
        for cls_idx, cls_name in enumerate(self.CLASS_NAMES):
            ms_files  = {os.path.basename(p) for p in
                         glob.glob(os.path.join(self.root_ms, cls_name, "*.tif"))}
            sar_files = {os.path.basename(p) for p in
                         glob.glob(os.path.join(self.root_sar, cls_name, "*.tif"))}
            paired = sorted(ms_files & sar_files)
            missing_ms  = sar_files - ms_files
            missing_sar = ms_files - sar_files
            if missing_ms or missing_sar:
                print(f"[EuroSATSAR] WARN class={cls_name}: "
                      f"{len(missing_ms)} SAR-only, {len(missing_sar)} MS-only filenames skipped")
            for fn in paired:
                samples.append((cls_idx, fn))
        if not samples:
            raise RuntimeError(
                f"[EuroSATSAR] No paired MS/SAR samples found under "
                f"{self.root_ms} / {self.root_sar}"
            )
        print(f"[EuroSATSAR] Paired {len(samples)} MS/SAR samples across "
              f"{len(self.CLASS_NAMES)} classes")
        return samples

    def _load_or_build_split(self):
        cache_path = os.path.join(self.root_ms, "split_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"[EuroSATSAR] Loaded split cache from {cache_path} "
                  f"({len(cached)} entries)")
            return cached

        print(f"[EuroSATSAR] Building stratified {self.SPLIT_RATIOS} split "
              f"(seed={self.SPLIT_SEED})...")
        by_class = defaultdict(list)
        for cls_idx, fn in self.samples:
            by_class[cls_idx].append(fn)

        rng = random.Random(self.SPLIT_SEED)
        assignment = {}
        for cls_idx, filenames in by_class.items():
            filenames = sorted(filenames)
            rng.shuffle(filenames)
            n = len(filenames)
            n_train = int(round(n * self.SPLIT_RATIOS["train"]))
            n_valid = int(round(n * self.SPLIT_RATIOS["valid"]))
            for fn in filenames[:n_train]:
                assignment[fn] = "train"
            for fn in filenames[n_train:n_train + n_valid]:
                assignment[fn] = "valid"
            for fn in filenames[n_train + n_valid:]:
                assignment[fn] = "test"

        try:
            with open(cache_path, "w") as f:
                json.dump(assignment, f)
            print(f"[EuroSATSAR] Saved split cache to {cache_path}")
        except Exception as e:
            print(f"[EuroSATSAR] WARN: could not save split cache: {e}")

        counts = Counter(assignment.values())
        print(f"[EuroSATSAR] Split sizes: {dict(counts)}")
        return assignment

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.sample_list)

    def _load_pair(self, cls_idx, filename):
        cls_name = self.CLASS_NAMES[cls_idx]
        ms_path  = os.path.join(self.root_ms,  cls_name, filename)
        sar_path = os.path.join(self.root_sar, cls_name, filename)

        with rasterio.open(ms_path) as src:
            ms = src.read().astype(np.float32)       # [13, 64, 64], raw tif order
        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)       # [2, 64, 64], [VV, VH]

        ms  = np.nan_to_num(ms, nan=0.0, posinf=0.0, neginf=0.0)
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

        ms  = torch.from_numpy(ms)
        sar = torch.from_numpy(sar)

        # Reorder MS channels: raw tif order -> bands_eurosat idx order
        ms = ms[self.ms_reorder_idx]

        return ms, sar

    def __getitem__(self, index):
        cls_idx, filename = self.sample_list[index]
        ms, sar = self._load_pair(cls_idx, filename)

        ms  = (ms  - self.norm_stats["ms_mean"].view(-1, 1, 1))  / self.norm_stats["ms_std"].view(-1, 1, 1)
        sar = (sar - self.norm_stats["sar_mean"].view(-1, 1, 1)) / self.norm_stats["sar_std"].view(-1, 1, 1)

        image_full = torch.cat([ms, sar], dim=0)   # [15, 64, 64], ALL_BAND_NAMES order
        image      = self._select_channels(image_full)
        label      = torch.tensor(cls_idx, dtype=torch.long)

        if self.split == "train":
            image = self._d4_augment(image)

        C, H, W = image.shape
        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)

        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=dummy_label,
            resolution=self.OPTICAL_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        first_spectral_idx = int(self.spectral_indices[0].item())
        query = torch.tensor([[
            0.0,                   # value (unused for query)
            (W - 1) / 2.0,         # x — center
            (H - 1) / 2.0,         # y — center
            first_spectral_idx,    # spectral_idx (placeholder)
            int(label.item()),     # label (scalar class)
            0,                     # query_idx
            self.resolution_idx,   # resolution_idx
            self.TIME_IDX_NA,      # time_idx
        ]], dtype=torch.float32)

        N = image_tokens.shape[0]
        if N > self.nb_tokens:
            perm = torch.randperm(N)[:self.nb_tokens]
            image_tokens = image_tokens[perm]

        attention_mask = torch.zeros(image_tokens.shape[0])
        image_tokens, attention_mask = self._apply_drop_mask(image_tokens, attention_mask)

        # >>> AUGMENTATION: stochastic per-sample band dropout (train only).
        # Layered on top of the static eval-config drop above, via the
        # same zero-value + mask=1.0 primitive — see module docstring.
        if self.band_dropout_enabled:
            aug_drop_indices = self._sample_band_dropout_indices()
            image_tokens, attention_mask = self._zero_and_mask_by_spectral_indices(
                image_tokens, attention_mask, aug_drop_indices
            )
        # >>> END AUGMENTATION

        queries_mask = torch.zeros(query.shape[0])

        return {
            "groups": {
                self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           query,
            "queries_mask":      queries_mask,
            "label":             label,
            "task":              self.TASK_NAME,
            "target_resolution": self.OPTICAL_RESOLUTION,
            "image":             image,
        }

    # =========================================================================
    # VIZ SAMPLE (no augmentation)
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        cls_idx, filename = self.sample_list[index]
        ms, sar = self._load_pair(cls_idx, filename)

        ms  = (ms  - self.norm_stats["ms_mean"].view(-1, 1, 1))  / self.norm_stats["ms_std"].view(-1, 1, 1)
        sar = (sar - self.norm_stats["sar_mean"].view(-1, 1, 1)) / self.norm_stats["sar_std"].view(-1, 1, 1)

        image_full = torch.cat([ms, sar], dim=0)
        image      = self._select_channels(image_full)
        label      = torch.tensor(cls_idx, dtype=torch.long)
        C, H, W    = image.shape

        # NOTE: get_viz_sample deliberately does NOT apply the training-time
        # band-dropout augmentation (only the static eval-config drop, via
        # _apply_drop_mask, same as before) — viz should show the model's
        # real deployed behavior, not augmentation noise. Same choice as
        # Sen1Floods11SkipDataset.

        dummy_label = torch.full((H, W), self.IGNORE_INDEX, dtype=torch.long)
        image_tokens = self.token_builder.build_tokens(
            image=image, label=dummy_label,
            resolution=self.OPTICAL_RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = int(self.spectral_indices[0].item())
        query = torch.tensor([[
            0.0, (W - 1) / 2.0, (H - 1) / 2.0,
            first_spectral_idx, int(label.item()), 0,
            self.resolution_idx, self.TIME_IDX_NA,
        ]], dtype=torch.float32)

        attention_mask = torch.zeros(image_tokens.shape[0])
        image_tokens, attention_mask = self._apply_drop_mask(image_tokens, attention_mask)
        queries_mask = torch.zeros(query.shape[0])

        return {
            "groups": {
                self.OPTICAL_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                },
            },
            "queries":           query,
            "queries_mask":      queries_mask,
            "label":             label,
            "task":              self.TASK_NAME,
            "target_resolution": self.OPTICAL_RESOLUTION,
            "image":             image,
        }

    # =========================================================================
    # NORMALIZATION (train-split-only, cached)
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_ms, "normalization_stats.pt")
        if os.path.exists(norm_file):
            print(f"[EuroSATSAR] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[EuroSATSAR] WARNING: No normalization file at {norm_file}; "
                  f"using identity stats (compute a train split first).")
            return {
                "ms_mean":  torch.zeros(self.NUM_MS_BANDS),  "ms_std":  torch.ones(self.NUM_MS_BANDS),
                "sar_mean": torch.zeros(self.NUM_SAR_BANDS), "sar_std": torch.ones(self.NUM_SAR_BANDS),
            }

        print(f"[EuroSATSAR] Computing normalization from "
              f"{len(self.sample_list)} train samples...")
        stats = self._compute_normalization_stats()
        try:
            torch.save(stats, norm_file)
            print(f"[EuroSATSAR] Saved normalization stats to {norm_file}")
        except Exception as e:
            print(f"[EuroSATSAR] WARN: could not save normalization stats: {e}")
        self._print_norm_stats(stats)
        return stats

    # Tunable: thread pool size (I/O-bound rasterio opens) and how many
    # samples to stack together per vectorized reduction step. Keep
    # NORM_NUM_WORKERS moderate on shared Lustre filesystems (Jean-Zay) —
    # too many concurrent opens can hammer the metadata server.
    NORM_BATCH_SIZE  = 256
    NORM_NUM_WORKERS = 8

    def _load_pair_raw_np(self, item):
        """Worker fn: load one (ms, sar) pair as raw (unnormalized) numpy
        float64 arrays, or None on failure. Used by the thread pool below."""
        cls_idx, filename = item
        try:
            ms, sar = self._load_pair(cls_idx, filename)
            return ms.numpy().astype(np.float64), sar.numpy().astype(np.float64)
        except Exception as e:
            print(f"[EuroSATSAR] WARN: failed to load {self.CLASS_NAMES[item[0]]}/{item[1]}: {e}")
            return None

    def _compute_normalization_stats(self, batch_size: int = None, num_workers: int = None):
        """Parallel I/O (thread pool) + vectorized per-batch reduction.

        Instead of one Python loop over samples with a nested loop over
        channels, we: (1) fetch NORM_BATCH_SIZE samples concurrently via a
        thread pool (rasterio/GDAL releases the GIL during actual disk
        reads, so this overlaps I/O latency), (2) stack each batch into a
        single [B, C, H, W] array and reduce over (B, H, W) in one
        vectorized numpy call per channel-group instead of per-sample.
        """
        from concurrent.futures import ThreadPoolExecutor

        batch_size  = batch_size  or self.NORM_BATCH_SIZE
        num_workers = num_workers or self.NORM_NUM_WORKERS

        ms_sum = np.zeros(self.NUM_MS_BANDS, dtype=np.float64)
        ms_sq  = np.zeros(self.NUM_MS_BANDS, dtype=np.float64)
        ms_n   = np.zeros(self.NUM_MS_BANDS, dtype=np.float64)
        sar_sum = np.zeros(self.NUM_SAR_BANDS, dtype=np.float64)
        sar_sq  = np.zeros(self.NUM_SAR_BANDS, dtype=np.float64)
        sar_n   = np.zeros(self.NUM_SAR_BANDS, dtype=np.float64)

        samples = self.sample_list
        n_total = len(samples)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for start in tqdm(range(0, n_total, batch_size),
                               desc=f"Computing normalization "
                                    f"(batch={batch_size}, workers={num_workers})"):
                chunk   = samples[start:start + batch_size]
                results = list(executor.map(self._load_pair_raw_np, chunk))
                results = [r for r in results if r is not None]
                if not results:
                    continue

                ms_batch  = np.stack([r[0] for r in results], axis=0)   # [B, 13, H, W]
                sar_batch = np.stack([r[1] for r in results], axis=0)   # [B, 2,  H, W]

                # MS: valid = value > 0 (matches original per-sample logic)
                ms_valid = ms_batch > 0
                ms_sum += np.where(ms_valid, ms_batch, 0.0).sum(axis=(0, 2, 3))
                ms_sq  += np.where(ms_valid, ms_batch ** 2, 0.0).sum(axis=(0, 2, 3))
                ms_n   += ms_valid.sum(axis=(0, 2, 3))

                # SAR: valid = finite and != 0
                sar_valid = np.isfinite(sar_batch) & (sar_batch != 0)
                sar_sum += np.where(sar_valid, sar_batch, 0.0).sum(axis=(0, 2, 3))
                sar_sq  += np.where(sar_valid, sar_batch ** 2, 0.0).sum(axis=(0, 2, 3))
                sar_n   += sar_valid.sum(axis=(0, 2, 3))

        ms_sum, ms_sq, ms_n    = (torch.from_numpy(x) for x in (ms_sum, ms_sq, ms_n))
        sar_sum, sar_sq, sar_n = (torch.from_numpy(x) for x in (sar_sum, sar_sq, sar_n))

        ms_mean  = (ms_sum / ms_n.clamp(min=1)).float()
        ms_std   = ((ms_sq / ms_n.clamp(min=1) - ms_mean.double() ** 2).clamp(min=0).sqrt()).float()
        sar_mean = (sar_sum / sar_n.clamp(min=1)).float()
        sar_std  = ((sar_sq / sar_n.clamp(min=1) - sar_mean.double() ** 2).clamp(min=0).sqrt()).float()

        return {"ms_mean": ms_mean, "ms_std": ms_std.clamp(min=1e-6),
                "sar_mean": sar_mean, "sar_std": sar_std.clamp(min=1e-6)}

    def _print_norm_stats(self, stats):
        print(f"[EuroSATSAR] MS  mean: {stats['ms_mean'].numpy()}")
        print(f"[EuroSATSAR] MS  std:  {stats['ms_std'].numpy()}")
        print(f"[EuroSATSAR] SAR mean: {stats['sar_mean'].numpy()}")
        print(f"[EuroSATSAR] SAR std:  {stats['sar_std'].numpy()}")

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _build_combined_bands_info(self, dataset_config):
        """13 optical bands from bands_eurosat (idx 0-12) + VV/VH from
        bands_senflood (idx 13-14), so SAR tokens share spectral_idx with
        Sen1Floods11's VV/VH."""
        bands_eurosat  = dataset_config["bands_eurosat"]
        bands_senflood = dataset_config["bands_senflood"]

        combined = dict(bands_eurosat)  # idx 0-12, unchanged
        for name in ("VV", "VH"):
            if name not in bands_senflood:
                raise KeyError(f"'{name}' not found in dataset_config['bands_senflood']")
            combined[name] = bands_senflood[name]  # idx 13, 14 already set there
        return combined

    def _parse_bands_info(self):
        all_bands = []
        for name, data in self.bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": int(data["bandwidth"]),
                    "central_wavelength": int(data["central_wavelength"]),
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        bw    = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl    = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[EuroSATSAR] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:16s} → "
                  f"bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[EuroSATSAR] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)
