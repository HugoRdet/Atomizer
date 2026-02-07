import os
import csv
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import einops
from tqdm import tqdm


class Sen1Floods11Dataset(Dataset):
    """
    Sen1Floods11 Dataset with per-channel normalization.
    
    Directory structure expected:
    ./data/SENFLOOD/
    ├── data/
    │   └── flood_events/HandLabeled/
    │       ├── S1Hand/
    │       ├── S2Hand/
    │       └── LabelHand/
    ├── splits/
    │   └── flood_handlabeled/
    │       ├── flood_train_data.csv
    │       ├── flood_valid_data.csv
    │       └── flood_test_data.csv
    └── normalization_stats.pt  (auto-generated on first run)
    
    Bands info format (flat dict, accessed via dataset_config["senflood"]):
        B01: {bandwidth: 20, central_wavelength: 443, idx: 0, ...}
        B02: {bandwidth: 65, central_wavelength: 490, idx: 1, ...}
        ...
        VV:  {bandwidth: -1, central_wavelength: -1, idx: 13, ...}
        VH:  {bandwidth: -2, central_wavelength: -2, idx: 14, ...}
    """

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
            "validation": "validation",  # CSV uses "valid" not "validation"
            "test": "test"
        }
        
        # Paths
        self.data_root = os.path.join(root_path, "data", "flood_events", "HandLabeled")
        self.split_file = os.path.join(
            root_path, "splits", "flood_handlabeled", 
            f"flood_{self.split_mapping[mode]}_data.csv"
        )
        
        # Load file lists
        self.s1_image_list, self.s2_image_list, self.label_list = self._load_file_lists()
        self._filter_invalid_samples()
 
        # Parse band info (expects flat dict with idx field)
        self.bands_info = dataset_config["bands_senflood"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()
        
        # Pre-compute spectral indices ONCE
        self.spectral_indices = self._build_spectral_indices()
        
        # =====================================================================
        # NORMALIZATION: Load or compute per-channel mean/std
        # =====================================================================
        self.norm_stats = self._load_or_compute_normalization()
        
        print(f"[Sen1Floods11] Loaded {len(self.bandwidths)} bands")

    def _load_file_lists(self):
        """Load file lists from split CSV."""
        s1_images = []
        s2_images = []
        labels = []

        print(f"[Sen1Floods11] Loading split file: {self.split_file}")
      
        with open(self.split_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                    
                # row[0]: S1Hand/region_filename_S1Hand.tif or just filename
                # row[1]: LabelHand/region_filename_LabelHand.tif or just filename
                s1_filename = row[0].replace("S1Hand/", "")
                label_filename = row[1].replace("LabelHand/", "")
                
                # S2 filename: replace S1Hand with S2Hand in the filename itself
                s2_filename = s1_filename.replace("_S1Hand", "_S2Hand")
                
                s1_path = os.path.join(self.data_root, "S1Hand", s1_filename)
                s2_path = os.path.join(self.data_root, "S2Hand", s2_filename)
                label_path = os.path.join(self.data_root, "LabelHand", label_filename)
                
                s1_images.append(s1_path)
                s2_images.append(s2_path)
                labels.append(label_path)
        
        return s1_images, s2_images, labels

    # =========================================================================
    # NORMALIZATION
    # =========================================================================
    def _filter_invalid_samples(self):
        """Remove samples with no valid labels (all 255)."""
        valid_s1, valid_s2, valid_labels = [], [], []
        skipped = 0
        
        print(f"[Sen1Floods11] Filtering invalid samples...")
        for i in tqdm(range(len(self.label_list)), desc="Checking labels"):
            try:
                with rasterio.open(self.label_list[i]) as src:
                    label = src.read(1)
                label[label == -1] = 255
                
                if (label != 255).sum() > 100:  # At least 100 valid pixels
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
    
    def _load_or_compute_normalization(self):
        """Load normalization stats from file, or compute them if not found."""
        norm_file = os.path.join(self.root_path, "normalization_stats.pt")
        
        if os.path.exists(norm_file):
            print(f"[Sen1Floods11] Loading normalization stats from {norm_file}")
            stats = torch.load(norm_file, weights_only=True)
            self._print_norm_stats(stats)
            return stats
        
        # Only compute from training split
        if self.split != "train":
            print(f"[Sen1Floods11] WARNING: No normalization file found at {norm_file}")
            print(f"[Sen1Floods11] Run training split first to compute normalization stats")
            # Return dummy stats (no normalization) - will likely cause NaN!
            return {
                "s2_mean": torch.zeros(13),
                "s2_std": torch.ones(13),
                "s1_mean": torch.zeros(2),
                "s1_std": torch.ones(2),
            }
        
        # Compute normalization stats from training data
        print(f"[Sen1Floods11] Computing normalization stats from {len(self.s1_image_list)} samples...")
        stats = self._compute_normalization_stats()
        
        # Save for future use
        torch.save(stats, norm_file)
        print(f"[Sen1Floods11] Saved normalization stats to {norm_file}")
        
        self._print_norm_stats(stats)
        return stats
    
    def _compute_normalization_stats(self):
        """Compute per-channel mean and std."""
        # Accumulators: S2 has 13 channels, S1 has 2 channels
        s2_sum = torch.zeros(13, dtype=torch.float64)
        s2_sum_sq = torch.zeros(13, dtype=torch.float64)
        s2_count = torch.zeros(13, dtype=torch.float64)
        
        s1_sum = torch.zeros(2, dtype=torch.float64)
        s1_sum_sq = torch.zeros(2, dtype=torch.float64)
        s1_count = torch.zeros(2, dtype=torch.float64)
        
        for idx in tqdm(range(len(self.s2_image_list)), desc="Computing normalization"):
            # Load S2
            try:
                with rasterio.open(self.s2_image_list[idx]) as src:
                    s2 = src.read().astype(np.float64)  # [13, H, W]
                
                s2 = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
                
                for c in range(13):
                    channel = s2[c].flatten()
                    # Only count valid pixels (> 0 for optical)
                    valid_mask = channel > 0
                    valid = channel[valid_mask]
                    if len(valid) > 0:
                        s2_sum[c] += valid.sum()
                        s2_sum_sq[c] += (valid ** 2).sum()
                        s2_count[c] += len(valid)
            except Exception as e:
                print(f"[Warning] Could not read S2 {self.s2_image_list[idx]}: {e}")
                continue
            
            # Load S1
            try:
                with rasterio.open(self.s1_image_list[idx]) as src:
                    s1 = src.read().astype(np.float64)  # [2, H, W]
                
                s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
                
                for c in range(2):
                    channel = s1[c].flatten()
                    # SAR can have any value, just exclude zeros (no data)
                    valid_mask = channel != 0
                    valid = channel[valid_mask]
                    if len(valid) > 0:
                        s1_sum[c] += valid.sum()
                        s1_sum_sq[c] += (valid ** 2).sum()
                        s1_count[c] += len(valid)
            except Exception as e:
                print(f"[Warning] Could not read S1 {self.s1_image_list[idx]}: {e}")
                continue
        
        # Compute mean and std per channel
        s2_mean = s2_sum / s2_count.clamp(min=1)
        s2_var = (s2_sum_sq / s2_count.clamp(min=1)) - (s2_mean ** 2)
        s2_std = torch.sqrt(s2_var.clamp(min=1e-8))
        
        s1_mean = s1_sum / s1_count.clamp(min=1)
        s1_var = (s1_sum_sq / s1_count.clamp(min=1)) - (s1_mean ** 2)
        s1_std = torch.sqrt(s1_var.clamp(min=1e-8))
        
        return {
            "s2_mean": s2_mean.float(),
            "s2_std": s2_std.float(),
            "s1_mean": s1_mean.float(),
            "s1_std": s1_std.float(),
        }
    
    def _print_norm_stats(self, stats):
        """Print normalization statistics."""
        print(f"[Sen1Floods11] Normalization stats:")
        print(f"  S2 mean: {stats['s2_mean'].numpy()}")
        print(f"  S2 std:  {stats['s2_std'].numpy()}")
        print(f"  S1 mean: {stats['s1_mean'].numpy()}")
        print(f"  S1 std:  {stats['s1_std'].numpy()}")
    
    def normalize_image(self, s2, s1):
        """Apply per-channel z-score normalization: (x - mean) / std"""
        # S2: [13, H, W]
        s2_mean = self.norm_stats["s2_mean"].view(13, 1, 1)
        s2_std = self.norm_stats["s2_std"].view(13, 1, 1)
        s2_norm = (s2 - s2_mean) / s2_std
        
        # S1: [2, H, W]
        s1_mean = self.norm_stats["s1_mean"].view(2, 1, 1)
        s1_std = self.norm_stats["s1_std"].view(2, 1, 1)
        s1_norm = (s1 - s1_mean) / s1_std
        
        return s2_norm, s1_norm

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
        
        # Sort by idx to match tensor order
        all_bands = sorted(all_bands, key=lambda x: x["idx"])
        
        # Extract as tensors
        bandwidths = torch.tensor(
            [b["bandwidth"] for b in all_bands], dtype=torch.float32
        )
        wavelengths = torch.tensor(
            [b["central_wavelength"] for b in all_bands], dtype=torch.float32
        )
        band_names = [b["name"] for b in all_bands]
        
        # Debug print
        print(f"[Sen1Floods11] Band order:")
        for b in all_bands:
            marker = " (abstract)" if b["bandwidth"] < 0 or b["central_wavelength"] < 0 else ""
            print(f"  idx={b['idx']:2d}: {b['name']:4s} -> bw={b['bandwidth']:4d}, wl={b['central_wavelength']:4d}{marker}")
        
        return bandwidths, wavelengths, band_names

    def _build_spectral_indices(self):
        """Map local band order to global spectral indices (called once)."""
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {self.band_names[i]} with key {key} not found in lookup table. "
                    f"Available keys: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # COORDINATE HELPERS
    # =========================================================================

    def get_wavelengths_coordinates(self, image_shape):
        """Efficient: expand pre-computed spectral indices to all pixels."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]
        # [C] -> [C * H * W]: each band's index repeated for all its pixels
        return self.spectral_indices.repeat_interleave(H * W)

    def get_position_coordinates(self, image_shape, resolution, table):
        """Get global (x, y) position indices for each pixel."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]
        
        res_key = int(resolution * 1000)
        global_offset = table[(res_key, H)]  # Assuming square images
        
        # Create meshgrid
        y_coords = torch.arange(H)
        x_coords = torch.arange(W)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Add global offset
        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset

        # Expand for all channels: [H, W] -> [C, H, W, 1]
        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)
        
        return x_indices, y_indices

    def get_position_coordinates_queries(self, image_shape, resolution, table):
        """Get query position indices."""
        C, H, W = image_shape[0], image_shape[-2], image_shape[-1]

        # Use fixed resolution for latents
        resolution_latents = 0.2  # m
        res_key = int(resolution_latents * 1000)
        
        # Check if key exists, fallback to actual resolution if not
        if (res_key, H) in table:
            global_offset = table[(res_key, H)]
        else:
            res_key = int(resolution * 1000)
            global_offset = table.get((res_key, H), 0)
        
        # Create constant offset tensor
        indices = torch.full((C, H, W, 1), global_offset, dtype=torch.float32)
        
        return indices

    def shuffle_arrays(self, arrays: list):
        """Shuffle multiple arrays with the same random permutation."""
        perm = torch.randperm(arrays[0].shape[0])
        return [arr[perm] for arr in arrays]

    def __len__(self):
        return len(self.s1_image_list)

    def __getitem__(self, index):
        # index=1  ← REMOVED!
        
        # Load S2 (Optical) - 13 bands
        with rasterio.open(self.s2_image_list[index]) as src:
            image_s2 = src.read().astype(np.float32)
        
        # Load S1 (SAR) - 2 bands
        with rasterio.open(self.s1_image_list[index]) as src:
            image_s1 = src.read().astype(np.float32)
        
        # Handle NaN/Inf
        image_s1 = np.nan_to_num(image_s1, nan=0.0, posinf=0.0, neginf=0.0)
        image_s2 = np.nan_to_num(image_s2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Load Label
        with rasterio.open(self.label_list[index]) as src:
            label = src.read(1).astype(np.int64)
        
        label[label == -1] = 255
        
        # Convert to tensors
        image_s1 = torch.from_numpy(image_s1)
        image_s2 = torch.from_numpy(image_s2)
        label = torch.from_numpy(label)

        # Normalize
        image_s2, image_s1 = self.normalize_image(image_s2, image_s1)
        
        # Clamp outliers
        image_s2 = torch.clamp(image_s2, -10, 10)
        image_s1 = torch.clamp(image_s1, -10, 10)
        
        # Clamp to avoid extreme outliers that could cause NaN



        # =====================================================================
        # Concatenate S2 + S1 to match band idx order (0-12: S2, 13-14: S1)
        # =====================================================================
        image = torch.cat([image_s2, image_s1], dim=0)  # [15, H, W]
        
        # =====================================================================
        # Build token coordinates
        # =====================================================================
        resolution = 10  # meters per pixel
        
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
        
        
        # Expand label to match image shape: [H, W] -> [C, H, W]
        label_expanded = label.unsqueeze(0).expand(image.shape[0], -1, -1)
        
        # =====================================================================
        # Build tokens: [C, H, W, 6] -> [C*H*W, 6]
        # Format: [value, x, y, spectral_idx, label, query_idx]
        # =====================================================================
        image_tokens = torch.cat([
            image.unsqueeze(-1),                                    # [C, H, W, 1] - values
            x_indices.float(),                                      # [C, H, W, 1] - x position
            y_indices.float(),                                      # [C, H, W, 1] - y position
            spectral_coords.view(image.shape[0], image.shape[1], image.shape[2], 1).float(),  # [C, H, W, 1] - spectral idx
            label_expanded.unsqueeze(-1).float(),                   # [C, H, W, 1] - label
            query_indices.float(),                                  # [C, H, W, 1] - query idx
        ], dim=-1)


        queries = image_tokens[0].unsqueeze(0)  # only one band for segmentation

        # Flatten: [C, H, W, 6] -> [C*H*W, 6]
        image_tokens = einops.rearrange(image_tokens, "c h w f -> (c h w) f")
        queries = einops.rearrange(queries, "c h w f -> (c h w) f")
        
        # =====================================================================
        # Attention mask (1.0 = ignore, 0.0 = valid)
        # =====================================================================
        attention_mask = torch.zeros(image_tokens.shape[0])
        #attention_mask[image_tokens[:, 4] == 255] = 1.0
        
        # =====================================================================
        # Prepare queries (shuffled subset for reconstruction)
        # =====================================================================
        queries_mask = torch.zeros(queries.shape[0])
        queries, queries_mask = self.shuffle_arrays([queries, queries_mask])
        
        # Subsample queries
        nb_queries = self.max_tokens_reconstruction
        queries = queries[:nb_queries]
        queries_mask = queries_mask[:nb_queries]

        # Placeholder for latent positions
        latent_pos = torch.zeros(1)

        if ((label==255).sum()==label.shape[0]) or ((queries[:,4]==255).sum()==queries.shape[0]):
            print((label==255).sum(),(queries[:,4]==255).sum())

        return image_tokens, attention_mask, queries, queries_mask, label, latent_pos, image