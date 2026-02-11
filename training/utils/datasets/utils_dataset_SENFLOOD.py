import os
import csv
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import einops
from tqdm import tqdm

from .token_grouping import *


class Sen1Floods11Dataset(Dataset):
    """
    Sen1Floods11 Dataset — grouped token format (8 columns).
    
    S2 (13 optical bands) + S1 (2 SAR bands), all at 10m, 512×512.
    Both modalities share the same resolution → single group.
    
    Token format:
        [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
         col 0  1  2       3          4        5            6            7
    
    For Sen1Floods11:
        - resolution_idx: looked up from GSD (10.0 m/px), same for all bands
        - time_idx: -1 (no temporal info, zeroed out by encoder)
    
    Convention:
        -1 = "not applicable" → encoder outputs zero vector
        ≥0 = valid index      → encoder outputs real features
    
    Returns:
    {
        "groups": {
            10.0: {
                "tokens": [N, 8],
                "mask":   [N],
                "shape":  (15, 512, 512),
            },
        },
        "queries":           [M, 8],
        "queries_mask":      [M],
        "label":             [512, 512],
        "target_resolution": 10.0,
        "image":             [15, 512, 512],
    }
    
    Directory structure:
    ./data/SENFLOOD/
    ├── data/flood_events/HandLabeled/{S1Hand,S2Hand,LabelHand}/
    ├── splits/flood_handlabeled/{flood_train_data,flood_valid_data,flood_test_data}.csv
    └── normalization_stats.pt  (auto-generated)
    """

    OPTICAL_RESOLUTION = 10.0   # m/px (all bands at 10m)
    NUM_S2_BANDS = 13
    NUM_S1_BANDS = 2
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    TIME_IDX_NA = -1            # No temporal info → -1 → zeroed by encoder

    def __init__(
        self,
        root_path: str = "./data/SENFLOOD",
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

        # Split mapping
        self.split_mapping = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

        # Paths
        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled",
            f"flood_{self.split_mapping[mode]}_data.csv",
        )

        # Load & filter file lists
        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()

        # Band metadata
        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        self.spectral_indices = self._build_spectral_indices()

        # Resolution index: same for all bands (S2 + S1 both at 10m)
        self.resolution_idx = self.look_up.get_resolution_idx(self.OPTICAL_RESOLUTION)

        # Normalization
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[Sen1Floods11] Loaded {len(self.bandwidths)} bands")
        print(f"[Sen1Floods11] Resolution idx: {self.resolution_idx} "
              f"(GSD={self.OPTICAL_RESOLUTION} m/px, all bands)")
        print(f"[Sen1Floods11] Time idx: -1 (no temporal info, zeroed by encoder)")

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.s1_image_list)

    def __getitem__(self, index):
        # ── Load ────────────────────────────────────────────
        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)    # [13, H, W]
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)    # [2, H, W]
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)        # [H, W]

        # ── Clean ───────────────────────────────────────────
        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        label[label == -1] = self.IGNORE_INDEX

        image_s2 = torch.from_numpy(image_s2)
        image_s1 = torch.from_numpy(image_s1)
        label = torch.from_numpy(label)

        # ── Normalize ───────────────────────────────────────
        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)
        image_s2 = torch.clamp(image_s2, -10, 10)
        image_s1 = torch.clamp(image_s1, -10, 10)

        # ── Merge (same resolution → single image) ─────────
        image = torch.cat([image_s2, image_s1], dim=0)  # [15, H, W]

        # ── Build tokens [N, 8] ─────────────────────────────
        resolution = self.OPTICAL_RESOLUTION
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
                    "shape": tuple(image.shape),       # (15, H, W)
                },
            },
            "queries": queries,                        # [M, 8]
            "queries_mask": queries_mask,               # [M]
            "label": label,                            # [H, W]
            "target_resolution": resolution,
            "image": image,                            # [15, H, W]
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
        
        resolution_idx is the same for all bands (S2 + S1 both at 10m).
        time_idx is -1 for all tokens (zeroed out by TimeEncoder).
        
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

        # Resolution: same for all bands (all at 10m)
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
        """Load file lists from split CSV."""
        s1_images, s2_images, labels = [], [], []

        print(f"[Sen1Floods11] Loading split file: {self.split_file}")

        with open(self.split_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue

                s1_filename = row[0].replace("S1Hand/", "")
                label_filename = row[1].replace("LabelHand/", "")
                s2_filename = s1_filename.replace("_S1Hand", "_S2Hand")

                s1_images.append(os.path.join(self.data_root, "S1Hand", s1_filename))
                s2_images.append(os.path.join(self.data_root, "S2Hand", s2_filename))
                labels.append(os.path.join(self.data_root, "LabelHand", label_filename))

        return s1_images, s2_images, labels

    def _filter_invalid_samples(self):
        """Remove samples with no valid labels (all 255)."""
        valid_s1, valid_s2, valid_labels = [], [], []
        skipped = 0

        print(f"[Sen1Floods11] Filtering invalid samples...")
        for i in tqdm(range(len(self.label_list)), desc="Checking labels"):
            try:
                with rasterio.open(self.label_list[i]) as src:
                    lbl = src.read(1)
                lbl[lbl == -1] = 255
                if (lbl != 255).sum() > 100:
                    valid_s1.append(self.s1_image_list[i])
                    valid_s2.append(self.s2_image_list[i])
                    valid_labels.append(self.label_list[i])
                else:
                    skipped += 1
            except Exception as e:
                print(f"[Warning] Could not read {self.label_list[i]}: {e}")
                skipped += 1

        print(f"[Sen1Floods11] Skipped {skipped} invalid samples")
        self.s1_image_list = valid_s1
        self.s2_image_list = valid_s2
        self.label_list = valid_labels

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[Sen1Floods11] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        if self.split != "train":
            print(f"[Sen1Floods11] WARNING: No normalization file at {norm_file}")
            return {
                "s2_mean": torch.zeros(13), "s2_std": torch.ones(13),
                "s1_mean": torch.zeros(2),  "s1_std": torch.ones(2),
            }

        print(f"[Sen1Floods11] Computing normalization from {len(self.s1_image_list)} samples...")
        stats = self._compute_normalization_stats()
        torch.save(stats, norm_file)
        print(f"[Sen1Floods11] Saved normalization stats to {norm_file}")
        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        s2_sum = torch.zeros(13, dtype=torch.float64)
        s2_sq  = torch.zeros(13, dtype=torch.float64)
        s2_n   = torch.zeros(13, dtype=torch.float64)
        s1_sum = torch.zeros(2, dtype=torch.float64)
        s1_sq  = torch.zeros(2, dtype=torch.float64)
        s1_n   = torch.zeros(2, dtype=torch.float64)

        for idx in tqdm(range(len(self.s2_image_list)), desc="Computing normalization"):
            try:
                with rasterio.open(self.s2_image_list[idx]) as src:
                    s2 = src.read().astype(np.float64)
                s2 = np.nan_to_num(s2)
                for c in range(13):
                    valid = s2[c].flatten()
                    valid = valid[valid > 0]
                    if len(valid):
                        s2_sum[c] += valid.sum()
                        s2_sq[c]  += (valid ** 2).sum()
                        s2_n[c]   += len(valid)
            except Exception:
                continue

            try:
                with rasterio.open(self.s1_image_list[idx]) as src:
                    s1 = src.read().astype(np.float64)
                s1 = np.nan_to_num(s1)
                for c in range(2):
                    valid = s1[c].flatten()
                    valid = valid[valid != 0]
                    if len(valid):
                        s1_sum[c] += valid.sum()
                        s1_sq[c]  += (valid ** 2).sum()
                        s1_n[c]   += len(valid)
            except Exception:
                continue

        s2_mean = (s2_sum / s2_n.clamp(min=1)).float()
        s2_std  = ((s2_sq / s2_n.clamp(min=1) - s2_mean.double() ** 2).sqrt()).float()
        s1_mean = (s1_sum / s1_n.clamp(min=1)).float()
        s1_std  = ((s1_sq / s1_n.clamp(min=1) - s1_mean.double() ** 2).sqrt()).float()

        return {
            "s2_mean": s2_mean, "s2_std": s2_std,
            "s1_mean": s1_mean, "s1_std": s1_std,
        }

    def _print_norm_stats(self, stats):
        print(f"[Sen1Floods11] S2 mean: {stats['s2_mean'].numpy()}")
        print(f"[Sen1Floods11] S2 std:  {stats['s2_std'].numpy()}")
        print(f"[Sen1Floods11] S1 mean: {stats['s1_mean'].numpy()}")
        print(f"[Sen1Floods11] S1 std:  {stats['s1_std'].numpy()}")

    def normalize_image(self, s2, s1):
        s2_mean = self.norm_stats["s2_mean"].view(13, 1, 1)
        s2_std  = self.norm_stats["s2_std"].view(13, 1, 1)
        s1_mean = self.norm_stats["s1_mean"].view(2, 1, 1)
        s1_std  = self.norm_stats["s1_std"].view(2, 1, 1)
        return (s2 - s2_mean) / s2_std, (s1 - s1_mean) / s1_std

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

        print(f"[Sen1Floods11] Band order:")
        for b in all_bands:
            tag = " (SAR)" if b["bandwidth"] < 0 or b["central_wavelength"] < 0 else ""
            print(f"  idx={b['idx']:2d}: {b['name']:4s} → bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}{tag}")

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