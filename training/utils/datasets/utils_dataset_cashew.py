"""
Cashew (m-cashew-plant, geo-bench) Dataset for Atomiser (Segmentation, SKIP)
=============================================================================

Single-temporal segmentation dataset in the grouped-token format, SKIP
variant — matches Sen1Floods11SkipDataset's contract so it can drive
Atomiser_Senflood_Skip directly: emits `query_token_idx` / `query_token_valid`
alongside the usual fields, letting the decoder's pixel-skip cascade read
each query-pixel's own band-tokens without cross-attention.

Modeled on ForestNetDataset (HDF5-per-sample loading) and
Sen1Floods11SkipDataset (segmentation D4-augment + skip gather index).

query_token_idx construction:
    Same closed-form scheme as Sen1Floods11SkipDataset. TokenBuilder.
    build_tokens flattens channel-major, `(c h w) -> row`, so pixel p's C
    band-tokens sit at rows {p + c*H*W : c in 0..C-1}, strided by H*W (not
    contiguous). Since Cashew never randomly subsamples image_tokens before
    building this index (see NOTE in __getitem__ below — unlike Sen1Floods11,
    the token pool is NOT randomly cropped down to nb_tokens here, because
    doing so would silently invalidate the closed-form row index), the same
    full-pixel-grid formula applies unmodified.

Source files (geo-bench-1.0/segmentation_v1.0/m-cashew-plant/):
    band_stats.json         — per-band mean/std (13 keys incl. Cloud
                               Probability and 'label' — 'label' is NOT a
                               reflectance band and must not be normalized
                               as one)
    default_partition.json  — {"train": [...], "valid": [...], "test": [...]}
                               sample names match HDF5 filenames directly,
                               e.g. "sample_0" -> "sample_0.hdf5"
    {sample_name}.hdf5       — 13 date-suffixed band datasets, [256, 256]
                               float32 each, PLUS a "label" dataset,
                               [256, 256] int64, holding the per-pixel
                               segmentation mask directly (no separate
                               label_map.json / label file needed).

HDF5 keys are date-suffixed (e.g. "02 - Blue_2019-11-15"), same convention
as ForestNet. We match by prefix at load time. The "label" key has no date
suffix and no band-stats entry we want to normalize with (see below).

Cloud Probability is EXCLUDED from the model input by default: it is not a
reflectance channel and has no physical wavelength, so it doesn't fit the
phi_lambda spectral encoding (which needs central_wavelength/bandwidth).
Set `include_cloud_probability=True` to keep it in as a non-optical band
routed through a learned embedding — NOT wired up in this version; the
lookup table would need a corresponding entry. Left off unless requested.

Band <-> Sentinel-2 code mapping (cashew uses GeoBench's descriptive band
names; Sen1Floods11 uses Sentinel-2 codes B01..B12 for the SAME physical
bands). We reuse dataset_config["bands_senflood"] for wavelength/bandwidth
metadata rather than duplicating it, by mapping cashew's HDF5 key prefixes
to their B0X equivalents. This assumes bands_senflood's B01..B12 wavelength/
bandwidth values are standard Sentinel-2 (they should be, since Sen1Floods11
is also Sentinel-2) — if bands_senflood ever changes, this mapping tracks it
automatically rather than drifting out of sync with a hardcoded duplicate.

    Cashew HDF5 prefix                  -> Sentinel-2 code (bands_senflood key)
    "01 - Coastal aerosol"              -> "B01"
    "02 - Blue"                         -> "B02"
    "03 - Green"                        -> "B03"
    "04 - Red"                          -> "B04"
    "05 - Vegetation Red Edge"          -> "B05"
    "06 - Vegetation Red Edge"          -> "B06"
    "07 - Vegetation Red Edge"          -> "B07"
    "08 - NIR"                          -> "B08"
    "08A - Vegetation Red Edge"         -> "B08A"
    "09 - Water vapour"                 -> "B09"
    "11 - SWIR"                         -> "B11"
    "12 - SWIR"                         -> "B12"

    NOTE: there is no B10 (cirrus) in the cashew band set, same as
    Sen1Floods11 — both skip B10. If your bands_senflood dict happens not
    to have a "B10" entry either, this mapping just never references it.

Output format:
    {
        "groups": {
            10.0: {                          # Sentinel-2 resolution
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (12, H, W),
            },
        },
        "queries":           [N_q, 8],
        "queries_mask":      [N_q],
        "label":             [H, W] long tensor (7 classes, IGNORE_INDEX=255
                              for any invalid pixels if present),
        "task":              "segmentation",
        "target_resolution": 10.0,
        "image":             [12, H, W],
    }

Augmentations (training only):
    D4 group: 4 rotations x 2 flips, applied identically to image AND
    label (unlike ForestNet's image-only D4 — segmentation labels are
    spatial and must rotate/flip in sync with the image).
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .token_grouping import *
from .token_builder import TokenBuilder


class CashewSkipDataset(Dataset):
    """Cashew (geo-bench m-cashew-plant) segmentation dataset for Atomiser,
    SKIP variant — emits query_token_idx/query_token_valid for
    Atomiser_Senflood_Skip's decoder pixel-skip cascade."""

    SENTINEL2_RESOLUTION = 10.0
    NUM_BANDS = 12
    NUM_CLASSES = 7
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1
    PATCH_SIZE_NATIVE = 256
    TASK_NAME = "segmentation"

    # HDF5 key prefix -> Sentinel-2 band code (key into bands_senflood).
    # Order here defines channel order in the output image tensor.
    BAND_PREFIX_TO_S2_CODE = {
        "01 - Coastal aerosol":       "B01",
        "02 - Blue":                  "B02",
        "03 - Green":                 "B03",
        "04 - Red":                   "B04",
        "05 - Vegetation Red Edge":   "B05",
        "06 - Vegetation Red Edge":   "B06",
        "07 - Vegetation Red Edge":   "B07",
        "08 - NIR":                   "B08",
        "08A - Vegetation Red Edge":  "B08A",
        "09 - Water vapour":          "B09",
        "11 - SWIR":                  "B11",
        "12 - SWIR":                  "B12",
    }
    BAND_PREFIXES = list(BAND_PREFIX_TO_S2_CODE.keys())

    CLOUD_PROB_PREFIX = "Cloud Probability"
    LABEL_KEY = "label"

    SPLIT_MAPPING = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    def __init__(
        self,
        root_path: str = "./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        crop_size: int = 256,
        include_cloud_probability: bool = False,
    ):
        super().__init__()
        assert mode in self.SPLIT_MAPPING, f"Unknown split: {mode}"
        assert crop_size <= self.PATCH_SIZE_NATIVE
        if include_cloud_probability:
            raise NotImplementedError(
                "[Cashew] include_cloud_probability=True requires a "
                "non-optical learned-embedding lookup entry for "
                "'Cloud Probability' that isn't wired up yet. Leave "
                "this False, or add that lookup entry first."
            )

        self.root_path      = root_path
        self.split           = mode
        self.crop_size       = crop_size
        self.look_up          = look_up
        self.config_model     = config_model
        self.dataset_config   = dataset_config

        self.token_builder = TokenBuilder(look_up)
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"].get(
            "max_tokens_reconstruction", self.nb_tokens
        )
        self.reconstruction = config_model["trainer"].get("mode", "segmentation") == "reconstruction"

        # ── Load JSON metadata ──────────────────────────────
        with open(os.path.join(root_path, "default_partition.json")) as f:
            default_partition = json.load(f)
        with open(os.path.join(root_path, "band_stats.json")) as f:
            self.band_stats = json.load(f)

        split_key = self.SPLIT_MAPPING[mode]
        self.sample_names = list(default_partition[split_key])

        # ── Normalization tensors ───────────────────────────
        # NOTE: band_stats.json's keys are the cashew descriptive prefixes
        # (same as BAND_PREFIXES), NOT the B0X codes — those are only used
        # to look wavelength/bandwidth up in bands_senflood. Don't confuse
        # the two key spaces.
        means, stds = [], []
        for prefix in self.BAND_PREFIXES:
            if prefix not in self.band_stats:
                raise KeyError(
                    f"[Cashew] Band '{prefix}' not in band_stats.json. "
                    f"Available: {list(self.band_stats.keys())}"
                )
            means.append(self.band_stats[prefix]["mean"])
            stds.append(self.band_stats[prefix]["std"])
        self.norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std  = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)

        # ── Band metadata + spectral indices ────────────────
        # Reuse bands_senflood (Sentinel-2 wavelength/bandwidth by B0X code)
        # rather than duplicating values, since cashew's optical bands are
        # the same physical Sentinel-2 bands under different names.
        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        self.resolution_idx = self.look_up.get_resolution_idx(self.SENTINEL2_RESOLUTION)

        print(f"[Cashew] task={self.TASK_NAME}, split={mode} -> "
              f"{len(self.sample_names)} samples")
        print(f"[Cashew] bands ({self.NUM_BANDS}): "
              f"{[self.BAND_PREFIX_TO_S2_CODE[p] for p in self.BAND_PREFIXES]}")
        print(f"[Cashew] center crop: {crop_size}x{crop_size}")
        print(f"[Cashew] resolution idx: {self.resolution_idx}")
        print(f"[Cashew] D4 augment (image+label): {'ON' if mode == 'train' else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    # AUGMENTATION — image AND label must transform together (segmentation)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _d4_augment(image: torch.Tensor, label: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label

    @staticmethod
    def _center_crop(image: torch.Tensor, label: torch.Tensor, size: int):
        C, H, W = image.shape
        if H == size and W == size:
            return image, label
        top  = (H - size) // 2
        left = (W - size) // 2
        return (
            image[:, top:top + size, left:left + size],
            label[top:top + size, left:left + size],
        )

    # ─────────────────────────────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_sample(self, name):
        path = os.path.join(self.root_path, f"{name}.hdf5")
        bands = []
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for prefix in self.BAND_PREFIXES:
                matches = [k for k in keys if k.startswith(prefix)]
                if not matches:
                    raise KeyError(f"[Cashew] No key with prefix '{prefix}' in {path}")
                bands.append(np.asarray(f[matches[0]], dtype=np.float32))
            if self.LABEL_KEY not in f:
                raise KeyError(f"[Cashew] No '{self.LABEL_KEY}' dataset in {path}")
            label = np.asarray(f[self.LABEL_KEY], dtype=np.int64)

        image = torch.from_numpy(np.stack(bands, axis=0))
        label = torch.from_numpy(label)

        image, label = self._center_crop(image, label, self.crop_size)

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.norm_mean) / self.norm_std

        return image, label

    # ─────────────────────────────────────────────────────────────────────
    # SKIP: per-query gather index into own band-tokens
    # ─────────────────────────────────────────────────────────────────────
    # Identical closed-form scheme to Sen1Floods11SkipDataset. Requires
    # image_tokens to be the FULL, un-subsampled pixel grid — see the
    # NOTE in __getitem__ for why no random token-pool subsampling happens
    # here (it would desync this index from image_tokens' actual rows).

    def _build_full_pixel_index(self, C, H, W):
        """
        Closed-form gather index for ALL pixels, in pixel order p = h*W + w.
        TokenBuilder.build_tokens flattens channel-major, `(c h w) -> row`,
        so pixel p's band-tokens live at rows {p + c*H*W : c in 0..C-1},
        strided by H*W (NOT contiguous).
        Returns [H*W, C] long.
        """
        HW = H * W
        p = torch.arange(HW)
        c = torch.arange(C)
        return p.unsqueeze(1) + c.unsqueeze(0) * HW

    def _build_query_token_index(self, C, H, W, kept_indices=None):
        """
        idx[i] = the C row indices (into this sample's image_tokens) of the
        band-tokens for query i's pixel.

        kept_indices: [N_q] long or None.
            None   -> queries are the full pixel grid in order (val/test,
                      and viz).
            tensor -> row positions (into the full pixel grid) that
                      subsample_queries kept, in the SAME order as the
                      returned queries (training).
        """
        full = self._build_full_pixel_index(C, H, W)          # [H*W, C]
        if kept_indices is None:
            idx = full
        else:
            idx = full[kept_indices]                          # [N_q, C]
        valid = torch.ones(idx.shape[0], dtype=torch.bool)
        return idx, valid

    # ─────────────────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        image, label = self._load_sample(name)

        if self.split == "train":
            image, label = self._d4_augment(image, label)

        resolution = self.SENTINEL2_RESOLUTION
        image_tokens, seg_queries = self._build_tokens(image, label, resolution)

        # NOTE: unlike ForestNet, we do NOT randomly subsample image_tokens
        # down to nb_tokens here. The SKIP query_token_idx below is a
        # closed-form gather into image_tokens' rows (pixel p's bands live
        # at fixed strided positions); subsampling image_tokens would shift
        # those rows and desync the index. Sen1Floods11SkipDataset has the
        # same constraint. If token-budget capping is needed for Cashew's
        # 256x256x12 = 786432 tokens, it must happen via band/query
        # subsampling instead (which this dataset already does via
        # subsample_queries below), not via dropping rows from image_tokens.
        attention_mask = torch.zeros(image_tokens.shape[0])

        C_img, H_img, W_img = image.shape

        if self.split == "train":
            queries, kept_indices = self.token_builder.subsample_queries(
                seg_queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
                return_indices=True,
            )
        else:
            queries = seg_queries
            kept_indices = None  # full pixel grid in order

        queries_mask = torch.zeros(queries.shape[0])

        query_token_idx, query_token_valid = self._build_query_token_index(
            C_img, H_img, W_img, kept_indices=kept_indices
        )

        return {
            "groups": {
                self.SENTINEL2_RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label,
            "task":              self.TASK_NAME,
            "target_resolution": self.SENTINEL2_RESOLUTION,
            "image":             image,
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }

    # ─────────────────────────────────────────────────────────────────────
    # VIZ SAMPLE
    # ─────────────────────────────────────────────────────────────────────

    def get_viz_sample(self, index: int) -> dict:
        name = self.sample_names[index]
        image, label = self._load_sample(name)

        resolution = self.SENTINEL2_RESOLUTION
        image_tokens, queries = self._build_tokens(image, label, resolution)
        attention_mask = torch.zeros(image_tokens.shape[0])
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.bool)

        C, H, W = image.shape
        query_token_idx, query_token_valid = self._build_query_token_index(
            C, H, W, kept_indices=None
        )

        return {
            "groups": {
                resolution: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  tuple(image.shape),
                },
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             label,
            "task":              self.TASK_NAME,
            "target_resolution": resolution,
            "image":             image,
            "query_token_idx":   query_token_idx,
            "query_token_valid": query_token_valid,
        }

    # ─────────────────────────────────────────────────────────────────────
    # TOKEN BUILDING
    # ─────────────────────────────────────────────────────────────────────

    def _build_tokens(self, image, label, resolution):
        image_tokens = self.token_builder.build_tokens(
            image=image,
            label=label,
            resolution=resolution,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        first_spectral_idx = self.spectral_indices[0]
        queries = self.token_builder.build_queries(
            label=label,
            resolution=resolution,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        return image_tokens, queries

    # ─────────────────────────────────────────────────────────────────────
    # BAND METADATA
    # ─────────────────────────────────────────────────────────────────────

    def _parse_bands_info(self):
        """
        Pull wavelength/bandwidth from dataset_config["bands_senflood"] for
        each cashew band's Sentinel-2 code equivalent, in BAND_PREFIXES
        order (which fixes the channel order of the output image tensor).
        """
        bw_list, wl_list, names = [], [], []
        for prefix in self.BAND_PREFIXES:
            code = self.BAND_PREFIX_TO_S2_CODE[prefix]
            if code not in self.bands_info:
                raise KeyError(
                    f"[Cashew] Sentinel-2 code '{code}' (for cashew band "
                    f"'{prefix}') not found in dataset_config['bands_senflood']. "
                    f"Available: {list(self.bands_info.keys())}"
                )
            data = self.bands_info[code]
            if not ("bandwidth" in data and "central_wavelength" in data):
                raise KeyError(
                    f"[Cashew] bands_senflood['{code}'] missing "
                    f"bandwidth/central_wavelength: {data}"
                )
            bw_list.append(int(data["bandwidth"]))
            wl_list.append(int(data["central_wavelength"]))
            names.append(code)

        bw = torch.tensor(bw_list, dtype=torch.float32)
        wl = torch.tensor(wl_list, dtype=torch.float32)

        print(f"[Cashew] Band order:")
        for prefix, code, b, w in zip(self.BAND_PREFIXES, names, bw_list, wl_list):
            print(f"  {prefix:28s} -> {code:5s}  bw={b:4d}, wl={w:4d}")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[Cashew] Band {self.band_names[i]} key={key} not in "
                    f"lookup. Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)
