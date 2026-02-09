import os
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import einops
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class HLSBurnScarsDataset(Dataset):
    """
    HLS Burn Scars Dataset with per-channel normalization.
    
    Directory structure expected:
    ./data/hls_burn_scars/
    ├── training/
    │   ├── subsetted_512x512_HLS.S30.T10SEH.2018280.v1.4_merged.tif
    │   ├── subsetted_512x512_HLS.S30.T10SEH.2018280.v1.4.mask.tif
    │   └── ...
    ├── validation/
    │   ├── subsetted_512x512_HLS.S30.T10SEH.2019305.v1.4_merged.tif
    │   ├── subsetted_512x512_HLS.S30.T10SEH.2019305.v1.4.mask.tif
    │   └── ...
    └── normalization_stats.pt  (auto-generated on first run)
    
    Each scene is a 6-band GeoTIFF (512x512, float32, reflectance in [0,1]):
        Ch 0: B02 (Blue, 490nm)
        Ch 1: B03 (Green, 560nm)
        Ch 2: B04 (Red, 665nm)
        Ch 3: B8A (NIR, 865nm)
        Ch 4: B11 (SWIR1, 1610nm)
        Ch 5: B12 (SWIR2, 2190nm)
    
    Masks are single-band int16:
        -1 = missing data (mapped to ignore_index=255)
         0 = not burned
         1 = burn scar
    
    Bands info format (flat dict, accessed via dataset_config["hls"]):
        B02: {bandwidth: 65, central_wavelength: 490, idx: 0, resolution: 30, ...}
        B03: {bandwidth: 35, central_wavelength: 560, idx: 1, resolution: 30, ...}
        ...
    """

    NUM_BANDS = 6
    NUM_CLASSES = 2  # 0: not burned, 1: burn scar
    IGNORE_INDEX = 255
    RESOLUTION = 30  # meters per pixel (HLS common grid)
    IMG_SIZE = 512

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

        # Parse band info
        self.bands_info = dataset_config["bands_hls"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()

        # Pre-compute spectral indices ONCE
        self.spectral_indices = self._build_spectral_indices()

        # Normalization: load or compute per-channel mean/std
        self.norm_stats = self._load_or_compute_normalization()

        print(f"[HLSBurnScars] Loaded {len(self.scene_list)} samples, "
              f"{len(self.bandwidths)} bands, split={self.split}")

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

        # Find all scenes and masks
        all_scenes = sorted(glob.glob(os.path.join(split_dir, "*_merged.tif")))
        all_masks = sorted(glob.glob(os.path.join(split_dir, "*.mask.tif")))

        # Verify pairing
        scenes = []
        masks = []
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

        # Apply train/val split for training directory
        # (matches PANGAEA: 90% train, 10% val, random_state=23)
        if self.split in ("train", "validation"):
            train_idxs, val_idxs = train_test_split(
                np.arange(len(scenes)),
                test_size=0.1,
                random_state=23,
            )
            if self.split == "train":
                indices = train_idxs
            else:
                indices = val_idxs
            scenes = [scenes[i] for i in indices]
            masks = [masks[i] for i in indices]

        print(f"[HLSBurnScars] Found {len(scenes)} samples in "
              f"{split_dir} (split={self.split})")

        return scenes, masks

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _load_or_compute_normalization(self):
        """Load normalization stats from file, or compute them if not found."""
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")

        if os.path.exists(norm_file):
            print(f"[HLSBurnScars] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats

        # Only compute from training split
        if self.split != "train":
            print(f"[HLSBurnScars] WARNING: No normalization file found at {norm_file}")
            print(f"[HLSBurnScars] Run training split first to compute stats")
            return {
                "mean": torch.zeros(self.NUM_BANDS),
                "std": torch.ones(self.NUM_BANDS),
            }

        print(f"[HLSBurnScars] Computing normalization stats from "
              f"{len(self.scene_list)} samples...")
        stats = self._compute_normalization_stats()

        torch.save(stats, norm_file)
        print(f"[HLSBurnScars] Saved normalization stats to {norm_file}")

        self._print_norm_stats(stats)
        return stats

    def _compute_normalization_stats(self):
        """Compute per-channel mean and std from training scenes."""
        ch_sum = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        ch_sum_sq = torch.zeros(self.NUM_BANDS, dtype=torch.float64)
        ch_count = torch.zeros(self.NUM_BANDS, dtype=torch.float64)

        for scene_path in tqdm(self.scene_list, desc="Computing normalization"):
            try:
                with rasterio.open(scene_path) as src:
                    data = src.read().astype(np.float64)  # [6, 512, 512]

                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                for c in range(self.NUM_BANDS):
                    channel = data[c].flatten()
                    # Exclude nodata (9999) and zero pixels
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

        return {
            "mean": mean.float(),
            "std": std.float(),
        }

    def _print_norm_stats(self, stats):
        """Print normalization statistics."""
        band_names = ["B02(Blue)", "B03(Green)", "B04(Red)",
                      "B8A(NIR)", "B11(SWIR1)", "B12(SWIR2)"]
        print(f"[HLSBurnScars] Normalization stats:")
        print(f"  mean: {stats['mean'].numpy()}")
        print(f"  std:  {stats['std'].numpy()}")
        for i, name in enumerate(band_names):
            print(f"    {name}: mean={stats['mean'][i]:.4f}, std={stats['std'][i]:.4f}")

    def normalize_image(self, image):
        """Apply per-channel z-score normalization: (x - mean) / std."""
        mean = self.norm_stats["mean"].view(self.NUM_BANDS, 1, 1)
        std = self.norm_stats["std"].view(self.NUM_BANDS, 1, 1)
        return (image - mean) / std

    # =========================================================================
    # BAND INFO PARSING
    # =========================================================================

    def _parse_bands_info(self):
        """
        Parse bands_info dict, sorted by idx field.

        Returns:
            bandwidths: Tensor [num_bands]
            wavelengths: Tensor [num_bands]
            band_names: List of band names in order
        """
        all_bands = []

        for band_name, band_data in self.bands_info.items():
            if "bandwidth" not in band_data or "central_wavelength" not in band_data:
                continue
            if "idx" not in band_data:
                raise ValueError(f"Band {band_name} missing 'idx' field")

            all_bands.append({
                "idx": band_data["idx"],
                "bandwidth": band_data["bandwidth"],
                "central_wavelength": band_data["central_wavelength"],
                "name": band_name,
            })

        all_bands = sorted(all_bands, key=lambda x: x["idx"])

        bandwidths = torch.tensor(
            [b["bandwidth"] for b in all_bands], dtype=torch.float32
        )
        wavelengths = torch.tensor(
            [b["central_wavelength"] for b in all_bands], dtype=torch.float32
        )
        band_names = [b["name"] for b in all_bands]

        print(f"[HLSBurnScars] Band order:")
        for b in all_bands:
            print(f"  idx={b['idx']:2d}: {b['name']:4s} -> "
                  f"bw={b['bandwidth']:4d}nm, wl={b['central_wavelength']:4d}nm")

        return bandwidths, wavelengths, band_names

    def _build_spectral_indices(self):
        """Map local band order to global spectral indices (called once)."""
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {self.band_names[i]} with key {key} not found in "
                    f"lookup table. Available keys: "
                    f"{list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # COORDINATE HELPERS
    # =========================================================================

    def get_wavelengths_coordinates(self, image_shape):
        """Expand pre-computed spectral indices to all pixels."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]
        return self.spectral_indices.repeat_interleave(H * W)

    def get_position_coordinates(self, image_shape, resolution, table):
        """Get global (x, y) position indices for each pixel."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        res_key = int(resolution * 1000)
        global_offset = table[(res_key, H)]

        y_coords = torch.arange(H)
        x_coords = torch.arange(W)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')

        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset

        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)

        return x_indices, y_indices

    def get_position_coordinates_queries(self, image_shape, resolution, table):
        """Get query position indices."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        resolution_latents = 10  # m
        res_key = int(resolution_latents * 1000)

        if (res_key, H) in table:
            global_offset = table[(res_key, H)]
        else:
            res_key = int(resolution * 1000)
            global_offset = table.get((res_key, H), 0)

        indices = torch.full((C, H, W, 1), global_offset, dtype=torch.float32)
        return indices

    def shuffle_arrays(self, arrays: list):
        """Shuffle multiple arrays with the same random permutation."""
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]

    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index):
        # Load scene (6 bands, 512x512, float32 reflectance in [0,1])
        with rasterio.open(self.scene_list[index]) as src:
            image = src.read().astype(np.float32)  # [6, 512, 512]

        # Load mask
        with rasterio.open(self.mask_list[index]) as src:
            label = src.read(1).astype(np.int64)  # [512, 512]

        # Handle NaN/Inf
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle nodata value (9999 used by HLS for invalid pixels)
        invalid_mask = image == 9999
        image[invalid_mask] = 0.0

        # Map missing data (-1) to ignore index
        label[label == -1] = self.IGNORE_INDEX

        # Convert to tensors
        image = torch.from_numpy(image)   # [6, 512, 512]
        label = torch.from_numpy(label)   # [512, 512]

        # Normalize
        image = self.normalize_image(image)

        # Clamp outliers
        image = torch.clamp(image, -10, 10)

        # =================================================================
        # Build token coordinates
        # =================================================================
        resolution = self.RESOLUTION  # 30 m/px

        # Spectral indices: [C * H * W]
        spectral_coords = self.get_wavelengths_coordinates(image.shape)

        # Position indices: [C, H, W, 1]
        x_indices, y_indices = self.get_position_coordinates(
            image.shape, resolution, table=self.look_up.table
        )

        # Query position indices: [C, H, W, 1]
        query_indices = self.get_position_coordinates_queries(
            image.shape, resolution, table=self.look_up.table_queries
        )

        # Expand label: [H, W] -> [C, H, W]
        label_expanded = label.unsqueeze(0).expand(image.shape[0], -1, -1)

        # =================================================================
        # Build tokens: [C, H, W, 6] -> [C*H*W, 6]
        # Format: [value, x, y, spectral_idx, label, query_idx]
        # =================================================================
        image_tokens = torch.cat([
            image.unsqueeze(-1),            # [C, H, W, 1] - values
            x_indices.float(),              # [C, H, W, 1] - x position
            y_indices.float(),              # [C, H, W, 1] - y position
            spectral_coords.view(
                image.shape[0], image.shape[1], image.shape[2], 1
            ).float(),                      # [C, H, W, 1] - spectral idx
            label_expanded.unsqueeze(-1).float(),  # [C, H, W, 1] - label
            query_indices.float(),          # [C, H, W, 1] - query idx
        ], dim=-1)

        # Queries: single band for segmentation
        queries = image_tokens[0].unsqueeze(0)

        # Flatten: [C, H, W, 6] -> [C*H*W, 6]
        image_tokens = einops.rearrange(image_tokens, "c h w f -> (c h w) f")
        queries = einops.rearrange(queries, "c h w f -> (c h w) f")

        # =================================================================
        # Attention mask (1.0 = ignore, 0.0 = valid)
        # =================================================================
        attention_mask = torch.zeros(image_tokens.shape[0])

        # =================================================================
        # Prepare queries (shuffled subset for reconstruction)
        # =================================================================
        queries_mask = torch.zeros(queries.shape[0])
        queries, queries_mask = self.shuffle_arrays([queries, queries_mask])

        nb_queries = self.max_tokens_reconstruction
        queries = queries[:nb_queries]
        queries_mask = queries_mask[:nb_queries]

        # Placeholder for latent positions
        latent_pos = torch.zeros(1)

        return image_tokens, attention_mask, queries, queries_mask, label, latent_pos, image