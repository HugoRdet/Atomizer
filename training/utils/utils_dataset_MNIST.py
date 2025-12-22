"""
MNIST Sparse Canvas Dataset

Creates 512x512 black images with a single randomly-placed, randomly-sized MNIST digit.
Designed to test whether Perceiver latents can "move" to find sparse signal.

Compatible with FLAIR_MAE dataset interface.
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
import einops
import numpy as np


class MNISTSparseCanvas(Dataset):
    """
    Dataset that places a single MNIST digit on a large black canvas.
    
    The digit is:
    - Randomly resized (within min_digit_size to max_digit_size)
    - Randomly positioned (anywhere on canvas, fully contained)
    
    This creates extreme sparsity to test if latents can learn to "find" signal.
    """
    
    def __init__(
        self,
        canvas_size: int = 512,
        min_digit_size: int = 64,
        max_digit_size: int = 256,
        num_bands: int = 1,  # Grayscale = 1, can fake RGB = 3
        mode: str = "train",
        config_model: dict = None,
        look_up=None,
        num_samples: int = None,  # If None, use full MNIST
        fixed_position: bool = False,  # If True, always center (for ablation)
        fixed_size: int = None,  # If set, use this size instead of random
        **kwargs
    ):
        super().__init__()
        
        self.canvas_size = canvas_size
        self.min_digit_size = min_digit_size
        self.max_digit_size = max_digit_size
        self.num_bands = num_bands
        self.mode = mode
        self.fixed_position = fixed_position
        self.fixed_size = fixed_size
        self.look_up = look_up
        
        # Config for token counts
        if config_model is not None:
            self.nb_tokens = config_model["trainer"]["max_tokens"]
            self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]
            self.num_spatial_latents = config_model["Atomiser"]["spatial_latents"] ** 2
        else:
            # Defaults for standalone testing
            self.nb_tokens = 262144  # 512*512
            self.max_tokens_reconstruction = 60000
            self.num_spatial_latents = 16  # 4x4 latents
        

        print(self.max_tokens_reconstruction,self.num_spatial_latents)
        # Load MNIST
        is_train = (mode == "train")
        self.mnist = torchvision.datasets.MNIST(
            root='./data',
            train=is_train,
            download=True
        )
        
        # Limit samples if requested
        if num_samples is not None:
            self.num_samples = min(num_samples, len(self.mnist))
        else:
            self.num_samples = len(self.mnist)
        
        # Fake band info for single grayscale "band"
        # Using roughly visible spectrum center (550nm green)
        self.wavelengths = torch.tensor([550.0] * num_bands)
        self.bandwidths = torch.tensor([100.0] * num_bands)
        
        self.gsd = 0.2  # meters per pixel (matching FLAIR)

        self.num_latents = self.num_spatial_latents
        
        print(f"MNISTSparseCanvas initialized:")
        print(f"  Canvas: {canvas_size}x{canvas_size}")
        print(f"  Digit size range: {min_digit_size}-{max_digit_size}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Fixed position: {fixed_position}")
        print(f"  Num latents: {self.num_spatial_latents}")
        print(f"  *** ORACLE MODE: Latents placed ON digit! ***")

    def __len__(self):
        return self.num_samples
    
    def _place_digit_on_canvas(self, digit_img: torch.Tensor, idx: int):
        """
        Place a digit on a black canvas with random size and position.
        
        Args:
            digit_img: [28, 28] MNIST digit tensor
            idx: Sample index (used for reproducible randomness)
            
        Returns:
            canvas: [1, canvas_size, canvas_size] tensor
            bbox: dict with 'x', 'y', 'w', 'h' of digit location
        """
        # Use index-based seed for reproducibility
        rng = torch.Generator()
        rng.manual_seed(idx + 42)
        
        # Random size
        if self.fixed_size is not None:
            new_size = self.fixed_size
        else:
            new_size = int(torch.randint(
                self.min_digit_size, 
                self.max_digit_size + 1, 
                (1,), 
                generator=rng
            ).item())
        
        # Resize digit
        digit_resized = TF.resize(
            digit_img.unsqueeze(0),  # [1, 28, 28]
            [new_size, new_size],
            antialias=True
        ).squeeze(0)  # [new_size, new_size]
        
        # Random position (ensure digit fits)
        if self.fixed_position:
            # Center the digit
            x = (self.canvas_size - new_size) // 2
            y = (self.canvas_size - new_size) // 2
        else:
            max_x = self.canvas_size - new_size
            max_y = self.canvas_size - new_size
            x = int(torch.randint(0, max_x + 1, (1,), generator=rng).item())
            y = int(torch.randint(0, max_y + 1, (1,), generator=rng).item())
        
        # Create black canvas and place digit
        canvas = torch.zeros(self.canvas_size, self.canvas_size)
        canvas[y:y+new_size, x:x+new_size] = digit_resized
        
        # Expand to num_bands (repeat grayscale across bands)
        canvas = canvas.unsqueeze(0).repeat(self.num_bands, 1, 1)  # [B, H, W]
        
        bbox = {'x': x, 'y': y, 'w': new_size, 'h': new_size}
        
        return canvas, bbox
    
    def _compute_random_positions(self, idx: int) -> torch.Tensor:
        """
        Compute random latent positions for this sample.
        Uses index-based seed for reproducibility across epochs.
        
        Positions are in PIXEL COORDINATES (0 to canvas_size-1) to match
        the token x/y indices used in the dataset.
        
        Args:
            idx: Sample index for reproducible randomness
            
        Returns:
            positions: [num_latents, 2] tensor of (x, y) positions in pixels
        """
        rng = torch.Generator()
        rng.manual_seed(idx + 1337)  # Different seed from digit placement (which uses idx + 42)
        
        # Random positions in pixels [0, canvas_size)
        # Use continuous values (not integers) for smoother distribution
        px = torch.rand(self.num_latents, generator=rng) * self.canvas_size
        py = torch.rand(self.num_latents, generator=rng) * self.canvas_size
        
        # Return as pixel coordinates (matching token x/y indices)
        # Do NOT convert to meters - the Atomiser geometry handles that internally
        positions = torch.stack([px, py], dim=-1)
        
        return positions
    
    def _compute_oracle_positions(self, canvas: torch.Tensor, bbox: dict):
        """
        Compute oracle latent positions based on digit location.
        
        For this experiment, we place latents in a grid centered on the digit,
        with some spread to cover the digit area.
        
        Positions are in PIXEL COORDINATES to match token x/y indices.
        
        Args:
            canvas: [B, H, W] image tensor
            bbox: dict with digit bounding box
            
        Returns:
            positions: [num_latents, 2] tensor of (x, y) positions in pixels
        """
        # Digit center in pixels
        cx = bbox['x'] + bbox['w'] / 2
        cy = bbox['y'] + bbox['h'] / 2
        
        # Create a small grid of latents around the digit
        grid_size = int(np.sqrt(self.num_latents))
        
        # Spread latents across the digit area (with some margin)
        spread = max(bbox['w'], bbox['h']) * 1.5
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Normalized grid position [-0.5, 0.5]
                ni = (i / (grid_size - 1) - 0.5) if grid_size > 1 else 0
                nj = (j / (grid_size - 1) - 0.5) if grid_size > 1 else 0
                
                # Position in pixels
                px = cx + ni * spread
                py = cy + nj * spread
                
                # Clamp to canvas
                px = max(0, min(self.canvas_size - 1, px))
                py = max(0, min(self.canvas_size - 1, py))
                
                # Keep in pixel coordinates (don't convert to meters)
                positions.append([px, py])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def get_position_coordinates(self, image_shape):
        """
        Generate position coordinates for all pixels.
        Simplified version without look_up table.
        """
        image_size = image_shape[-1]
        channels_size = image_shape[0]
        
        # Create meshgrid
        y_coords = torch.arange(image_size, dtype=torch.float32)
        x_coords = torch.arange(image_size, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # If we have look_up table, use it for global offset
        if self.look_up is not None:
            tmp_resolution = int(self.gsd * 1000)  # 200 for 0.2m
            if hasattr(self.look_up, 'table') and (tmp_resolution, image_size) in self.look_up.table:
                global_offset = self.look_up.table[(tmp_resolution, image_size)]
                x_grid = x_grid + global_offset
                y_grid = y_grid + global_offset
        
        # Expand for all bands
        x_indices = einops.repeat(x_grid, "h w -> b h w 1", b=channels_size)
        y_indices = einops.repeat(y_grid, "h w -> b h w 1", b=channels_size)
        
        return x_indices, y_indices
    
    def get_wavelength_coordinates(self, image_shape):
        """
        Generate wavelength/bandwidth indices for all pixels.
        Simplified version - returns band index as placeholder.
        """
        image_size = image_shape[-1]
        channels_size = image_shape[0]
        
        if self.look_up is not None and hasattr(self.look_up, 'table_wave'):
            # Use look_up table
            idxs = []
            for b in range(channels_size):
                bw = int(self.bandwidths[b].item())
                wl = int(self.wavelengths[b].item())
                if (bw, wl) in self.look_up.table_wave:
                    idxs.append(self.look_up.table_wave[(bw, wl)])
                else:
                    idxs.append(b)  # Fallback to band index
            idxs = torch.tensor(idxs, dtype=torch.float32)
        else:
            # Simple band index
            idxs = torch.arange(channels_size, dtype=torch.float32)
        
        # Expand to full image size [B, H, W, 1]
        idxs = idxs.view(-1, 1, 1, 1).expand(channels_size, image_size, image_size, 1)
        
        return idxs
    
    def get_query_indices(self, image_shape):
        """Generate query position indices."""
        image_size = image_shape[-1]
        channels_size = image_shape[0]
        
        # Simple version: just use 0 as global offset
        indices = torch.zeros(channels_size, image_size, image_size, 1)
        
        return indices

    def __getitem__(self, idx):
        # Get MNIST digit
        digit_img, digit_label = self.mnist[idx % len(self.mnist)]
        digit_tensor = torch.tensor(np.array(digit_img), dtype=torch.float32) / 255.0
        
        # Place on canvas
        canvas, bbox = self._place_digit_on_canvas(digit_tensor, idx)  # [B, H, W]
        
        # Use ORACLE positions: place latents directly on the digit!
        # This tests if the architecture can learn when latents are in the right place
        latent_pos = self._compute_oracle_positions(canvas, bbox)
        
        # Create segmentation-style label (0 = background, 1 = digit)
        label = (canvas[0] > 0.1).float()  # [H, W] binary mask
        label_segment = label.unsqueeze(0).repeat(self.num_bands, 1, 1)  # [B, H, W]
        
        # Get coordinate encodings
        x_indices, y_indices = self.get_position_coordinates(canvas.shape)
        wavelength_indices = self.get_wavelength_coordinates(canvas.shape)
        query_indices = self.get_query_indices(canvas.shape)
        
        # Build token tensor [B, H, W, features]
        # Features: [value, x, y, wavelength_idx, label, query_idx]
        image = torch.cat([
            canvas.unsqueeze(-1),           # Band values [B, H, W, 1]
            x_indices,                       # X coordinates [B, H, W, 1]
            y_indices,                       # Y coordinates [B, H, W, 1]
            wavelength_indices,              # Wavelength index [B, H, W, 1]
            label_segment.unsqueeze(-1),     # Segmentation label [B, H, W, 1]
            query_indices,                   # Query indices [B, H, W, 1]
        ], dim=-1)
        
        queries = image.clone()
        
        # Flatten to tokens [B*H*W, features]
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries = einops.rearrange(queries, "b h w c -> (b h w) c")
        
        # Attention mask (all valid = 0)
        attention_mask = torch.zeros(image.shape[0], dtype=torch.float32)
        
        # =====================================================================
        # BALANCED QUERY SAMPLING: 50% white (digit) + 50% black (background)
        # This forces the model to learn spatial discrimination!
        # =====================================================================
        reflectance = queries[:, 0]  # First column is the pixel value
        
        num_white = self.max_tokens_reconstruction // 2
        num_black = self.max_tokens_reconstruction - num_white
        
        # Get indices of white pixels (digit) and black pixels (background)
        white_mask = reflectance > 0.05
        black_mask = reflectance <= 0.05
        
        white_indices = white_mask.nonzero().squeeze(-1)
        black_indices = black_mask.nonzero().squeeze(-1)
        
        # Sample white pixels (with replacement if not enough)
        if len(white_indices) >= num_white:
            perm = torch.randperm(len(white_indices))[:num_white]
            selected_white = white_indices[perm]
        else:
            # Not enough white pixels - sample with replacement
            selected_white = white_indices[torch.randint(0, len(white_indices), (num_white,))]
        
        # Sample black pixels (always have plenty)
        perm = torch.randperm(len(black_indices))[:num_black]
        selected_black = black_indices[perm]
        
        # Combine and shuffle so model can't exploit order
        query_indices_combined = torch.cat([selected_white, selected_black])
        shuffle_perm = torch.randperm(len(query_indices_combined))
        query_indices_combined = query_indices_combined[shuffle_perm]
        
        # Select queries
        queries = queries[query_indices_combined]
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.float32)
        
        # Classification label (digit class 0-9)
        label_class = torch.tensor(digit_label, dtype=torch.long)
        
        return image, attention_mask, queries, queries_mask, label_class, latent_pos
    
    def get_samples_to_viz(self, idx):
        """
        Get sample with additional visualization info.
        Returns raw image for plotting.
        """
        # Get MNIST digit
        digit_img, digit_label = self.mnist[idx % len(self.mnist)]
        digit_tensor = torch.tensor(np.array(digit_img), dtype=torch.float32) / 255.0
        
        # Place on canvas
        canvas, bbox = self._place_digit_on_canvas(digit_tensor, idx)
        
        # Use ORACLE positions (same as __getitem__) for consistency
        latent_pos = self._compute_oracle_positions(canvas, bbox)
        
        # Raw image for visualization [H, W, C]
        image_to_return = einops.rearrange(canvas, "c h w -> h w c")
        
        # Create label mask
        label = (canvas[0] > 0.1).float()
        label_segment = label.unsqueeze(0).repeat(self.num_bands, 1, 1)
        
        # Build tokens
        x_indices, y_indices = self.get_position_coordinates(canvas.shape)
        wavelength_indices = self.get_wavelength_coordinates(canvas.shape)
        query_indices = self.get_query_indices(canvas.shape)
        
        image = torch.cat([
            canvas.unsqueeze(-1),
            x_indices,
            y_indices,
            wavelength_indices,
            label_segment.unsqueeze(-1),
            query_indices,
        ], dim=-1)
        
        queries = image.clone()
        
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries = einops.rearrange(queries, "b h w c -> (b h w) c")
        attention_mask = torch.zeros(image.shape[0], dtype=torch.float32)
        
        # Use same balanced sampling as __getitem__
        reflectance = queries[:, 0]
        
        num_white = self.max_tokens_reconstruction // 2
        num_black = self.max_tokens_reconstruction - num_white
        
        white_mask = reflectance > 0.05
        black_mask = reflectance <= 0.05
        
        white_indices = white_mask.nonzero().squeeze(-1)
        black_indices = black_mask.nonzero().squeeze(-1)
        
        if len(white_indices) >= num_white:
            perm = torch.randperm(len(white_indices))[:num_white]
            selected_white = white_indices[perm]
        else:
            selected_white = white_indices[torch.randint(0, len(white_indices), (num_white,))]
        
        perm = torch.randperm(len(black_indices))[:num_black]
        selected_black = black_indices[perm]
        
        query_indices_combined = torch.cat([selected_white, selected_black])
        shuffle_perm = torch.randperm(len(query_indices_combined))
        query_indices_combined = query_indices_combined[shuffle_perm]
        
        queries = queries[query_indices_combined]
        queries_mask = torch.zeros(queries.shape[0], dtype=torch.float32)
        
        label_class = torch.tensor(digit_label, dtype=torch.long)
        
        return image_to_return, image, attention_mask, queries, queries_mask, label_class, latent_pos, bbox