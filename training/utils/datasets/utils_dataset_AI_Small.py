"""
AI4SmallFarms Dataset — grouped token format (8 columns).

Sentinel-2 optical imagery @ 10m, single temporal.
Binary segmentation: Background (0) vs Crop Field (1).

Bands: B2, B3, B4, B8 (4 bands, confirmed from PANGAEA config)
Image size: 496×496 after focus crop (matches PANGAEA img_size)
Ignore index: -1 (no pixels ignored — all participate in loss)
Distribution: ~72.7% background, ~27.3% crop field

Normalization: fixed values from PANGAEA config
  mean: [750.21, 1032.73, 1165.13, 2416.04]
  std:  [283.38,  332.73,  518.90,  702.38]

Directory structure:
    ./data/Small/sentinel-2-asia/
    ├── train/images/*.tif  ├── train/masks/*.tif
    ├── val/images/*.tif    ├── val/masks/*.tif
    └── test/images/*.tif   └── test/masks/*.tif

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
"""

import os
import random
from glob import glob

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .token_builder import TokenBuilder


class AI4SmallFarmsDataset(Dataset):

    NUM_CLASSES  = 2
    IGNORE_INDEX = -1     # PANGAEA uses -1 — no pixels are ignored
    RESOLUTION   = 10.0
    TIME_IDX_NA  = -1
    CROP_SIZE    = 496    # matches PANGAEA img_size

    CLASS_NAMES  = ["Background", "Crop Field"]

    # Confirmed 4 bands from PANGAEA config: B2, B3, B4, B8
    S2_BANDS_INFO = {
        "B02": {"central_wavelength": 490, "bandwidth": 65,  "idx": 0},
        "B03": {"central_wavelength": 560, "bandwidth": 35,  "idx": 1},
        "B04": {"central_wavelength": 665, "bandwidth": 30,  "idx": 2},
        "B08": {"central_wavelength": 842, "bandwidth": 115, "idx": 3},
    }

    # Fixed normalization from PANGAEA config (more reliable than recomputing)
    PANGAEA_MEAN = torch.tensor([750.2136, 1032.7277, 1165.1279, 2416.0448])
    PANGAEA_STD  = torch.tensor([283.3842,  332.7280,  518.9025,  702.3791])

    def __init__(
        self,
        root_path: str = "./data/Small",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.root_path    = root_path
        self.look_up      = look_up
        self.config_model = config_model

        self.token_builder = TokenBuilder(look_up)

        self.nb_tokens                 = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]

        self.split_mapping = {
            "train":      "train",
            "validation": "val",
            "test":       "test",
        }
        self.split = self.split_mapping[mode]

        # File discovery
        base_dir        = os.path.join(root_path, "sentinel-2-asia", self.split)
        self.image_list = sorted(glob(os.path.join(base_dir, "images", "*.tif")))
        self.mask_list  = sorted(glob(os.path.join(base_dir, "masks",  "*.tif")))

        assert len(self.image_list) == len(self.mask_list), (
            f"[AI4SmallFarms] Mismatch: {len(self.image_list)} images "
            f"vs {len(self.mask_list)} masks in {base_dir}"
        )
        if len(self.image_list) == 0:
            raise FileNotFoundError(
                f"[AI4SmallFarms] No tif files found in {base_dir}")

        # Auto-detect band count from first image
        with rasterio.open(self.image_list[0]) as src:
            self.num_bands = src.count
        print(f"[AI4SmallFarms] Detected {self.num_bands} bands")

        self._setup_band_indices()
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        print(f"[AI4SmallFarms] {len(self.image_list)} samples, split={self.split}")
        print(f"[AI4SmallFarms] Bands: {[b for b in self.S2_BANDS_INFO]}")
        print(f"[AI4SmallFarms] Crop size: {self.CROP_SIZE}×{self.CROP_SIZE}")
        print(f"[AI4SmallFarms] Ignore index: {self.IGNORE_INDEX} (all pixels valid)")
        print(f"[AI4SmallFarms] Normalization: PANGAEA fixed values")

    # =========================================================================
    # BAND SETUP
    # =========================================================================

    def _setup_band_indices(self):
        all_bands = sorted(self.S2_BANDS_INFO.items(),
                           key=lambda x: x[1]["idx"])[:self.num_bands]
        indices = []
        for name, info in all_bands:
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"[AI4SmallFarms] Band {name} key={key} not in lookup.")
            indices.append(self.look_up.table_wave[key])
        self.spectral_indices = torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # ── Load ─────────────────────────────────────────────────────
        with rasterio.open(self.image_list[index]) as src:
            image = src.read().astype(np.float32)   # [C, H, W]
        with rasterio.open(self.mask_list[index]) as src:
            target = src.read(1).astype(np.int64)   # [H, W], values 0 or 255

        image  = torch.from_numpy(image)
        target = (torch.from_numpy(target) / 255).long()  # 0/255 → 0/1

        # ── Clean ────────────────────────────────────────────────────
        image = torch.nan_to_num(image, nan=0.0)

        # ── Normalize (PANGAEA fixed values) ─────────────────────────
        image = self._normalize(image)
        image = torch.clamp(image, -10, 10)

        # ── Pad if smaller than crop size ─────────────────────────────
        image, target = self._pad_if_needed(image, target)

        # ── Focus crop to 496×496 ─────────────────────────────────────
        image, target = self._focus_crop(image, target)

        # ── D4 augmentation (training only) ──────────────────────────
        if self.split == "train":
            image, target = self._d4_augment(image, target)

        # ── Build tokens ─────────────────────────────────────────────
        C, H, W = image.shape
        image_tokens = self.token_builder.build_tokens(
            image=image, label=target,
            resolution=self.RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        attention_mask = torch.zeros(image_tokens.shape[0])

        # ── Queries ──────────────────────────────────────────────────
        queries = self.token_builder.build_queries(
            label=target,
            resolution=self.RESOLUTION,
            first_spectral_idx=self.spectral_indices[0],
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        if self.split == "train":
            queries = self.token_builder.subsample_queries(
                queries,
                max_queries=self.max_tokens_reconstruction,
                ignore_index=self.IGNORE_INDEX,
                prioritize_valid=True,
            )
        queries_mask = torch.zeros(queries.shape[0])

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   attention_mask,
                    "shape":  (C, H, W),
                }
            },
            "queries":           queries,
            "queries_mask":      queries_mask,
            "label":             target,
            "target_resolution": self.RESOLUTION,
            "image":             image,
        }

    # =========================================================================
    # VIZ SAMPLES
    # =========================================================================

    def get_viz_sample(self, index: int) -> dict:
        with rasterio.open(self.image_list[index]) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(self.mask_list[index]) as src:
            target = src.read(1).astype(np.int64)

        image  = torch.from_numpy(image)
        target = (torch.from_numpy(target) / 255).long()
        image  = torch.clamp(
            self._normalize(torch.nan_to_num(image)), -10, 10)

        image, target = self._pad_if_needed(image, target)
        image, target = self._focus_crop(image, target)

        C, H, W = image.shape
        image_tokens = self.token_builder.build_tokens(
            image=image, label=target,
            resolution=self.RESOLUTION,
            spectral_indices=self.spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )
        queries = self.token_builder.build_queries(
            label=target,
            resolution=self.RESOLUTION,
            first_spectral_idx=self.spectral_indices[0],
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        return {
            "groups": {
                self.RESOLUTION: {
                    "tokens": image_tokens,
                    "mask":   torch.zeros(image_tokens.shape[0]),
                    "shape":  (C, H, W),
                }
            },
            "queries":           queries,
            "queries_mask":      torch.zeros(queries.shape[0], dtype=torch.bool),
            "label":             target,
            "target_resolution": self.RESOLUTION,
            "image":             image,
        }

    # =========================================================================
    # FOCUS CROP + PADDING
    # =========================================================================

    def _pad_if_needed(self, image: torch.Tensor,
                       label: torch.Tensor) -> tuple:
        """Pad to CROP_SIZE if image is smaller. Pad label with 0 (background)."""
        C, H, W = image.shape
        pad_h   = max(0, self.CROP_SIZE - H)
        pad_w   = max(0, self.CROP_SIZE - W)
        if pad_h == 0 and pad_w == 0:
            return image, label

        image = F.pad(image, (0, pad_w, 0, pad_h),
                      mode="constant", value=0.0)
        # Pad label with 0 (background) — no ignore_index since IGNORE_INDEX=-1
        label = F.pad(label.unsqueeze(0).float(),
                      (0, pad_w, 0, pad_h),
                      mode="constant", value=0.0,
                      ).squeeze(0).long()
        return image, label

    def _focus_crop(self, image: torch.Tensor,
                    label: torch.Tensor) -> tuple:
        """Crop CROP_SIZE×CROP_SIZE centered on a random valid pixel."""
        C, H, W = image.shape
        th = tw  = self.CROP_SIZE

        if H == th and W == tw:
            return image, label

        # Find crop field pixels (class 1) to anchor the crop
        valid_yx = torch.where(label == 1)
        if len(valid_yx[0]) > 0:
            anchor   = torch.randint(0, len(valid_yx[0]), (1,)).item()
            anchor_y = valid_yx[0][anchor].item()
            anchor_x = valid_yx[1][anchor].item()
            i = random.randint(max(0, anchor_y - th + 1),
                               min(anchor_y, H - th))
            j = random.randint(max(0, anchor_x - tw + 1),
                               min(anchor_x, W - tw))
        else:
            i = torch.randint(0, H - th + 1, (1,)).item()
            j = torch.randint(0, W - tw + 1, (1,)).item()

        return image[:, i:i+th, j:j+tw], label[i:i+th, j:j+tw]

    # =========================================================================
    # AUGMENTATION
    # =========================================================================

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

    # =========================================================================
    # NORMALIZATION (PANGAEA fixed values)
    # =========================================================================

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        C    = image.shape[0]
        mean = self.PANGAEA_MEAN[:C].view(C, 1, 1)
        std  = self.PANGAEA_STD[:C].view(C, 1, 1).clamp(min=1e-6)
        return (image - mean) / std