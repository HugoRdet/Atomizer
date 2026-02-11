import os
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import einops
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .token_grouping import *


class HLSBurnScarsDataset(Dataset):
    """
    HLS Burn Scars Dataset — grouped token format (8 columns).
    
    6-band HLS imagery at 30m, 512×512. Single resolution → single group.
    
    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2       3          4        5            6            7
    
    For HLS Burn Scars:
        - resolution_idx: looked up from GSD (30.0 m/px) via lookup_table (≥0)
        - time_idx: -1 (no temporal info, zeroed out by encoder)
    
    Convention:
        -1 = "not applicable" → encoder outputs zero vector
        ≥0 = valid index      → encoder outputs real features
    
    Returns:
    {
        "groups": {
            30.0: {
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (6, 512, 512),
            },
        },
        "queries":           [M, 8],
        "queries_mask":      [M],
        "label":             [512, 512],
        "target_resolution": 30.0,
        "image":             [6, 512, 512],
    }
    
    Directory structure:
    ./data/hls_burn_scars/
    ├── training/
    │   ├── subsetted_512x512_HLS.S30.T10SEH.2018280.v1.4_merged.tif
    │   ├── subsetted_512x512_HLS.S30.T10SEH.2018280.v1.4.mask.tif
    │   └── ...
    ├── validation/
    │   └── ...
    └── normalization_stats.pt  (auto-generated)
    
    Bands (6):
        B02 (Blue, 490nm), B03 (Green, 560nm), B04 (Red, 665nm),
        B8A (NIR, 865nm), B11 (SWIR1, 1610nm), B12 (SWIR2, 2190nm)
    
    Classes: 0 = not burned, 1 = burn scar, 255 = ignore
    """

    NUM_BANDS = 6
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    RESOLUTION = 30.0   # meters per pixel (HLS common grid)
    IMG_SIZE = 512
    TIME_IDX_NA = -1    # No temporal info → -1 → zeroed by encoder

    def __init__(
        self,
        root_path: str = "./data/hls_burn_scars",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = mode
        self.look_up = look_up
        self.config_model = config_model

        # Config parameters
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]

        # Split mapping (matches PANGAEA convention):
        #   train + val come from training/ (90/10 split, random_state=23)
        #   test uses validation/
        self.split_dir_mapping = {
            "train": "training",
            "validation": "training",
            "test": "validation",
        }

        # Load file lists
        self.scene_list, self.mask_list = self._load_file_lists()

        # Band metadata
        self.bands_info = dataset_config["bands_hls_info"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        # Resolution index (same for all tokens — all optical, single resolution)
        self.resolution_idx = self.look_up.get_resolution_idx(self.RESOLUTION)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[HLSBurnScars] Loaded {len(self.scene_list)} samples, "
              f"{len(self.bandwidths)} bands, split={self.split}")
        print(f"[HLSBurnScars] Resolution idx: {self.resolution_idx} "
              f"(GSD={self.RESOLUTION} m/px)")
        print(f"[HLSBurnScars] Time idx: -1 (no temporal info, zeroed by encoder)")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index):
        # ── Load ────────────────────────────────────────────
        with rasterio.open(self.scene_list[index]) as src:
            image = src.read().astype(np.float32)       # [6, 512, 512]
        with rasterio.open(self.mask_list[index]) as src:
            label = src.read(1).astype(np.int64)        # [512, 512]

        # ── Clean ───────────────────────────────────────────
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image[image == 9999] = 0.0                      # HLS nodata value
        label[label == -1] = self.IGNORE_INDEX

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # ── Normalize ───────────────────────────────────────
        image = self.normalize_image(image)
        image = torch.clamp(image, -10, 10)

        # ── Build tokens [N, 8] ─────────────────────────────
        resolution = self.RESOLUTION
        image_tokens, queries = self._build_tokens(image, label, resolution)

        # ── Attention mask ──────────────────────────────────
        attention_mask = torch.zeros(image_tokens.shape[0])

        # ── Subsample queries ───────────────────────────────
        queries_mask = torch.zeros(queries.shape[0])
        queries, queries_mask = self._shuffle_arrays([queries, queries_mask])
        nb_queries = self.max_tokens_reconstruction
        queries = queries[:nb_queries]
        queries_mask = queries_mask[:nb_queries]

        # ── Return grouped format ───────────────────────────
        return {
            "groups": {
                resolution: {
                    "tokens": image_tokens,           # [C*H*W, 8]
                    "mask": attention_mask,            # [C*H*W]
                    "shape": tuple(image.shape),       # (6, 512, 512)
                },
            },
            "queries": queries,                        # [M, 8]
            "queries_mask": queries_mask,               # [M]
            "label": label,                            # [H, W]
            "target_resolution": resolution,
            "image": image,                            # [6, 512, 512]
        }

    # =========================================================================
    # TOKEN BUILDING
    # =========================================================================

    def _build_tokens(self, image, label, resolution):
        """
        Build [N, 8] tokens from image + label.
        
        Token format:
            [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
             col 0  1  2       3          4        5            6            7
        
        All bands are optical → real resolution_idx (≥0).
        No temporal info → time_idx = -1 (zeroed by encoder).
        
        Args:
            image:      [C, H, W] normalized values
            label:      [H, W] class labels
            resolution: meters per pixel
        
        Returns:
            image_tokens: [C*H*W, 8]
            queries:      [H*W, 8]   (single band for segmentation)
        """
        C, H, W = image.shape

        # Spectral indices: [C*H*W]
        spectral_coords = self.spectral_indices.repeat_interleave(H * W)

        # Position coordinates: [C, H, W, 1]
        x_indices, y_indices = self._get_position_coordinates(image.shape, resolution)

        # Query position indices: [C, H, W, 1]
        query_indices = self._get_query_coordinates(image.shape, resolution)

        # Expand label: [H, W] → [C, H, W]
        label_expanded = label.unsqueeze(0).expand(C, -1, -1)

        # Resolution: real GSD index for all bands (all optical)
        resolution_col = torch.full((C, H, W, 1), self.resolution_idx, dtype=torch.float32)
        
        # Time: -1 for all tokens (no temporal info)
        time_col = torch.full((C, H, W, 1), self.TIME_IDX_NA, dtype=torch.float32)

        # Stack: [C, H, W, 8]
        image_tokens = torch.cat([
            image.unsqueeze(-1),                                           # col 0: value
            x_indices.float(),                                             # col 1: x
            y_indices.float(),                                             # col 2: y
            spectral_coords.view(C, H, W, 1).float(),                     # col 3: spectral_idx
            label_expanded.unsqueeze(-1).float(),                          # col 4: label
            query_indices.float(),                                         # col 5: query_idx
            resolution_col,                                                # col 6: resolution_idx
            time_col,                                                      # col 7: time_idx
        ], dim=-1)

        # Queries: first band only (for segmentation)
        queries = image_tokens[0].unsqueeze(0)

        # Flatten
        image_tokens = einops.rearrange(image_tokens, "c h w f -> (c h w) f")
        queries = einops.rearrange(queries, "c h w f -> (c h w) f")

        return image_tokens, queries

    def _get_position_coordinates(self, image_shape, resolution):
        """Pixel position indices via lookup table → [C, H, W, 1]."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        res_key = int(resolution * 1000)
        global_offset = self.look_up.table[(res_key, H)]

        y_coords = torch.arange(H)
        x_coords = torch.arange(W)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")

        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset

        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)

        return x_indices, y_indices

    def _get_query_coordinates(self, image_shape, resolution):
        """Query position indices via lookup table → [C, H, W, 1]."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        resolution_latents = 10  # m
        res_key = int(resolution_latents * 1000)

        if (res_key, H) in self.look_up.table_queries:
            global_offset = self.look_up.table_queries[(res_key, H)]
        else:
            res_key = int(resolution * 1000)
            global_offset = self.look_up.table_queries.get((res_key, H), 0)

        return torch.full((C, H, W, 1), global_offset, dtype=torch.float32)

    # =========================================================================
    # FILE LOADING
    # =========================================================================

    def _load_file_lists(self):
        """Discover scene/mask pairs from directory, with train/val split."""
        split_dir = os.path.join(
            self.root_path, self.split_dir_mapping[self.split]
        )

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Contents of {self.root_path}: {os.listdir(self.root_path)}"
            )

        all_scenes = sorted(glob.glob(os.path.join(split_dir, "*_merged.tif")))
        all_masks = sorted(glob.glob(os.path.join(split_dir, "*.mask.tif")))

        scenes, masks = [], []
        for scene_path in all_scenes:
            mask_path = scene_path.replace("_merged.tif", ".mask.tif")
            if mask_path in all_masks:
                scenes.append(scene_path)
                masks.append(mask_path)

        if len(scenes) == 0:
            raise RuntimeError(
                f"No valid scene/mask pairs found in {split_dir}. "
                f"Expected files like *_merged.tif and *.mask.tif"
            )

        # train/val split for training directory
        # (matches PANGAEA: 90% train, 10% val, random_state=23)
        if self.split in ("train", "validation"):
            train_idxs, val_idxs = train_test_split(
                np.arange(len(scenes)),
                test_size=0.1,
                random_state=23,
            )
            indices = train_idxs if self.split == "train" else val_idxs
            scenes = [scenes[i] for i in indices]
            masks = [masks[i] for i in indices]

        print(f"[HLSBurnScars] Found {len(scenes)} samples in "
              f"{split_dir} (split={self.split})")

        return scenes, masks

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[HLSBurnScars] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[HLSBurnScars] WARNING: No normalization file at {norm_file}")
            return {
                "mean": torch.zeros(self.NUM_BANDS),
                "std": torch.ones(self.NUM_BANDS),
            }

        print(f"[HLSBurnScars] Computing normalization from "
              f"{len(self.scene_list)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[HLSBurnScars] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        ch_sum = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        ch_sum_sq = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        ch_count = torch.zeros(self.NUM_BANDS, dtype=torch.float64)

        for scene_path in tqdm(self.scene_list, desc="Computing normalization"):
            try:
                with rasterio.open(scene_path) as src:
                    data = src.read().astype(np.float64)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                for c in range(self.NUM_BANDS):
                    channel = data[c].flatten()
                    valid = channel[(channel > 0) & (channel != 9999)]
                    if len(valid) > 0:
                        ch_sum[c] += valid.sum()
                        ch_sum_sq[c] += (valid ** 2).sum()
                        ch_count[c] += len(valid)
            except Exception as e:
                print(f"[Warning] Could not read {scene_path}: {e}")
                continue

        mean = ch_sum / ch_count.clamp(min=1)
        var = (ch_sum_sq / ch_count.clamp(min=1)) - (mean ** 2)
        std = torch.sqrt(var.clamp(min=1e-8))

        return {"mean": mean.float(), "std": std.float()}

    def _print_norm_stats(self, stats):
        band_names = ["B02(Blue)", "B03(Green)", "B04(Red)",
                      "B8A(NIR)", "B11(SWIR1)", "B12(SWIR2)"]
        print(f"[HLSBurnScars] Normalization stats:")
        for i, name in enumerate(band_names):
            print(f"  {name}: mean={stats['mean'][i]:.4f}, std={stats['std'][i]:.4f}")

    def normalize_image(self, image):
        mean = self.norm_stats["mean"].view(self.NUM_BANDS, 1, 1)
        std = self.norm_stats["std"].view(self.NUM_BANDS, 1, 1)
        return (image - mean) / std

    # =========================================================================
    # BAND METADATA
    # =========================================================================

    def _parse_bands_info(self):
        all_bands = []
        for name, data in self.bands_info.items():
            if "bandwidth" in data and "central_wavelength" in data and "idx" in data:
                all_bands.append({
                    "idx": data["idx"],
                    "bandwidth": data["bandwidth"],
                    "central_wavelength": data["central_wavelength"],
                    "name": name,
                })
        all_bands.sort(key=lambda b: b["idx"])

        bw = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wl = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        names = [b["name"] for b in all_bands]

        print(f"[HLSBurnScars] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:4s} → "
                  f"bw={b['bandwidth']:4d}nm, wl={b['central_wavelength']:4d}nm")

        return bw, wl, names

    def _build_spectral_indices(self):
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {self.band_names[i]} key={key} not in lookup. "
                    f"Available: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _shuffle_arrays(arrays: list):
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]