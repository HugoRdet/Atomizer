import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import numpy as np
import einops
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import numpy as np
import einops
import matplotlib.pyplot as plt
import os

class MNISTSparseCanvas(Dataset):
    def __init__(
        self,
        canvas_size: int = 28,  # Changed default to 28
        num_bands: int = 1,
        mode: str = "train",
        config_model: dict = None,
        look_up=None,
        num_samples: int = None,
        shuffle_grid: bool = True, 
        **kwargs
    ):
        super().__init__()
        self.canvas_size = 28 # Original MNIST size
        self.num_bands = num_bands
        self.mode = mode
        self.look_up = look_up
        self.gsd = 0.2 
        self.shuffle_grid = False#shuffle_grid

        if self.shuffle_grid:
            os.makedirs("./figures", exist_ok=True)

        self.half_extent = (self.canvas_size * self.gsd) / 2.0  
        
        self.offset = 0.0
        if self.look_up is not None:
            tmp_resolution = int(self.gsd * 1000)
            if (tmp_resolution, self.canvas_size) in self.look_up.table:
                self.offset = self.look_up.table[(tmp_resolution, self.canvas_size)]
                print(f"[Dataset] Applying Global Offset: {self.offset}")

        if config_model is not None:
            # For 28x28, total tokens is 784. Sampling 512-784 is reasonable.
            self.max_tokens_reconstruction = config_model["trainer"].get("max_tokens_reconstruction", 784)
            self.num_spatial_latents = config_model["Atomiser"]["spatial_latents"] ** 2
        else:
            self.max_tokens_reconstruction = 784
            self.num_spatial_latents = 16 

        self.mnist = torchvision.datasets.MNIST(root='./data', train=(mode == "train"), download=True)
        self.num_samples = min(num_samples, len(self.mnist)) if num_samples else len(self.mnist)
        self.num_latents = self.num_spatial_latents

    def __len__(self):
        return self.num_samples

    def _apply_jigsaw(self, canvas: torch.Tensor, label: int):
        """
        Forces 28x28 image into a 4x4 grid of patches (each 7x7 pixels).
        Shuffles them to break global spatial coherence.
        """
        grid_num = 4 # 4x4 grid = 16 patches
        c, h, w = canvas.shape
        p_h, p_w = h // grid_num, w // grid_num # 28 / 4 = 7 pixels per patch

        # 1. Slice into 16 patches
        patches = einops.rearrange(
            canvas, 'c (gh ph) (gw pw) -> (gh gw) c ph pw', 
            gh=grid_num, gw=grid_num, ph=p_h, pw=p_w
        )

        # 2. Shuffle
        idx = torch.randperm(grid_num**2)
        shuffled_patches = patches[idx]

        # 3. Reassemble
        shuffled_canvas = einops.rearrange(
            shuffled_patches, '(gh gw) c ph pw -> c (gh ph) (gw pw)', 
            gh=grid_num, gw=grid_num
        )
        return shuffled_canvas
    
    def _place_digit_on_canvas(self, digit_img: torch.Tensor, idx: int, label: int):
        # Resize to 28x28 (though it already is)
        digit_4d = digit_img.unsqueeze(0).unsqueeze(0)
        resized_digit = F.interpolate(
            digit_4d, 
            size=(self.canvas_size, self.canvas_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        canvas = resized_digit.unsqueeze(0).repeat(self.num_bands, 1, 1)

        if self.shuffle_grid:
            canvas = self._apply_jigsaw(canvas, label)

        bbox = {'x': 0, 'y': 0, 'w': self.canvas_size, 'h': self.canvas_size}
        return canvas, bbox
    
    def _compute_oracle_positions(self, canvas, bbox):
        cx_m = (bbox['x'] + bbox['w'] / 2 - self.canvas_size / 2) * self.gsd + self.offset
        cy_m = (bbox['y'] + bbox['h'] / 2 - self.canvas_size / 2) * self.gsd + self.offset
        
        grid_size = int(np.sqrt(self.num_latents))
        spread_m = (self.canvas_size * self.gsd) * 0.7
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                ni = (i / (grid_size - 1) - 0.5) if grid_size > 1 else 0
                nj = (j / (grid_size - 1) - 0.5) if grid_size > 1 else 0
                positions.append([cx_m + ni * spread_m, cy_m + nj * spread_m])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def get_position_coordinates(self, image_shape, new_resolution, table=None):
        image_size = image_shape[-1]
        channels_size = 1
        tmp_resolution = int(new_resolution * 1000)
        global_offset = table[(tmp_resolution, image_size)]
        y_coords = torch.arange(image_size)
        x_coords = torch.arange(image_size)
        x_indices, y_indices = torch.meshgrid(x_coords, y_coords, indexing='xy')
        x_indices = x_indices + global_offset
        y_indices = y_indices + global_offset
        x_indices = einops.repeat(x_indices, "h w -> h w r", r=channels_size).unsqueeze(0)
        y_indices = einops.repeat(y_indices, "h w -> h w r", r=channels_size).unsqueeze(0)
        return x_indices, y_indices

    def get_wavelength_coordinates(self, image_shape):
        image_size = image_shape[-1]
        channels_size = image_shape[0]
        idxs = torch.arange(channels_size, dtype=torch.float32)
        return idxs.view(-1, 1, 1, 1).expand(channels_size, image_size, image_size, 1)
    
    def get_query_indices(self, image_shape):
        return torch.zeros(image_shape[0], image_shape[-1], image_shape[-1], 1)

    def __getitem__(self, idx):
        digit_img, digit_label = self.mnist[idx % len(self.mnist)]
        digit_tensor = torch.tensor(np.array(digit_img), dtype=torch.float32) / 255.0
        
        canvas, bbox = self._place_digit_on_canvas(digit_tensor, idx, digit_label)
        latent_pos = self._compute_oracle_positions(canvas, bbox)
        
        # CHANGED: 28x28 coordinates
        x_idx, y_idx = self.get_position_coordinates(
            image_shape=(self.canvas_size, self.canvas_size),
            new_resolution=0.2,
            table=self.look_up.table
        )
        wave_idx = self.get_wavelength_coordinates(canvas.shape)
        q_idx = self.get_query_indices(canvas.shape)
        label_seg = (canvas[0] > 0.1).float().unsqueeze(0).repeat(self.num_bands, 1, 1)
        
        image_tokens = torch.cat([
            canvas.unsqueeze(-1), 
            x_idx, 
            y_idx, 
            wave_idx, 
            label_seg.unsqueeze(-1),
            q_idx,
        ], dim=-1)
        
        image_tokens = einops.rearrange(image_tokens, "b h w c -> (b h w) c")
        attention_mask = torch.zeros(image_tokens.shape[0], dtype=torch.float32)

        reflectance = image_tokens[:, 0]
        white_idx = (reflectance > 0.05).nonzero().squeeze(-1)
        black_idx = (reflectance <= 0.05).nonzero().squeeze(-1)
        
        # Sampling logic for smaller image
        n_w = min(self.max_tokens_reconstruction // 2, len(white_idx)) if len(white_idx) > 0 else 0
        n_b = self.max_tokens_reconstruction - n_w
        
        sel_w = white_idx[torch.randint(0, len(white_idx), (n_w,))] if n_w > 0 else torch.tensor([], dtype=torch.long)
        sel_b = black_idx[torch.randint(0, len(black_idx), (n_b,))]
        
        queries = image_tokens[torch.cat([sel_w, sel_b])]
        q_mask = torch.zeros(queries.shape[0], dtype=torch.float32)
        
        return image_tokens, attention_mask, queries, q_mask, torch.tensor(digit_label, dtype=torch.long), latent_pos

    # ... get_samples_to_viz updated with (self.canvas_size, self.canvas_size) ...
    

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import numpy as np
import einops
from typing import Optional, Tuple, Dict

class EMNISTSparseCanvas(Dataset):
    def __init__(
        self,
        canvas_size: int = 28,
        num_bands: int = 1,
        mode: str = "train",
        config_model: dict = None,
        look_up=None,
        num_samples: int = None,
        **kwargs
    ):
        super().__init__()
        self.canvas_size = 28
        self.num_bands = num_bands
        self.mode = mode
        self.look_up = look_up
        self.gsd = 0.2 

        self.half_extent = (self.canvas_size * self.gsd) / 2.0  # 6.4 meters
        
        # Calculate the global offset from lookup table
        self.offset = 0.0
        if self.look_up is not None:
            tmp_resolution = int(self.gsd * 1000)
            if (tmp_resolution, self.canvas_size) in self.look_up.table:
                self.offset = self.look_up.table[(tmp_resolution, self.canvas_size)]
                print(f"[Dataset] Applying Global Offset: {self.offset}")

        # Model-specific config extraction
        if config_model is not None:
            self.max_tokens_reconstruction = config_model["trainer"].get("max_tokens_reconstruction", 1024)
            self.num_spatial_latents = config_model["Atomiser"]["spatial_latents"] ** 2
        else:
            self.max_tokens_reconstruction = 1024
            self.num_spatial_latents = 16 

        # EMNIST 'letters' split: 26 classes (a-z)
        # Note: EMNIST raw data is transposed (W, H) relative to MNIST
        self.emnist = torchvision.datasets.EMNIST(
            root='./data', 
            split='letters', 
            train=(mode == "train"), 
            download=True
        )
        
        self.num_samples = min(num_samples, len(self.emnist)) if num_samples else len(self.emnist)
        self.num_latents = self.num_spatial_latents

    def __len__(self):
        return self.num_samples
    
    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _place_digit_on_canvas(self, digit_img: torch.Tensor, idx: int):
        """Resizes character to full canvas size and places centered."""
        digit_4d = digit_img.unsqueeze(0).unsqueeze(0)
        resized_digit = F.interpolate(
            digit_4d, 
            size=(self.canvas_size, self.canvas_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        canvas = resized_digit.unsqueeze(0).repeat(self.num_bands, 1, 1)
        bbox = {'x': 0, 'y': 0, 'w': self.canvas_size, 'h': self.canvas_size}
        return canvas, bbox
    
    def _compute_oracle_positions(self, canvas, bbox):
        """Positions centered in METERS + Global Offset."""
        cx_m = (bbox['x'] + bbox['w'] / 2 - self.canvas_size / 2) * self.gsd + self.offset
        cy_m = (bbox['y'] + bbox['h'] / 2 - self.canvas_size / 2) * self.gsd + self.offset
        
        grid_size = int(np.sqrt(self.num_latents))
        spread_m = (self.canvas_size * self.gsd) * 0.7
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                ni = (i / (grid_size - 1) - 0.5) if grid_size > 1 else 0
                nj = (j / (grid_size - 1) - 0.5) if grid_size > 1 else 0
                positions.append([cx_m + ni * spread_m, cy_m + nj * spread_m])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    # =========================================================
    # COORDINATE GENERATORS
    # =========================================================

    def get_position_coordinates(self, image_shape, new_resolution, table=None):
        image_size = image_shape[-1]
        channels_size = 1
        tmp_resolution = int(new_resolution * 1000)
        global_offset = table[(tmp_resolution, image_size)]
        
        y_coords = torch.arange(image_size)
        x_coords = torch.arange(image_size)
        x_indices, y_indices = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        x_indices = x_indices + global_offset
        y_indices = y_indices + global_offset

        x_indices = einops.repeat(x_indices, "h w -> h w r", r=channels_size).unsqueeze(0)
        y_indices = einops.repeat(y_indices, "h w -> h w r", r=channels_size).unsqueeze(0)
        return x_indices, y_indices

    def get_wavelength_coordinates(self, image_shape):
        image_size = image_shape[-1]
        channels_size = image_shape[0]
        idxs = torch.arange(channels_size, dtype=torch.float32)
        return idxs.view(-1, 1, 1, 1).expand(channels_size, image_size, image_size, 1)
    
    def get_query_indices(self, image_shape):
        return torch.zeros(image_shape[0], image_shape[-1], image_shape[-1], 1)

    # =========================================================
    # MAIN DATA RETRIEVAL
    # =========================================================

    def __getitem__(self, idx):
        # EMNIST letters: label 1=a, 2=b, etc.
        digit_img, digit_label = self.emnist[idx % len(self.emnist)]
        
        # EMNIST correction: stored as (W, H), needs transpose to (H, W)
        digit_tensor = torch.tensor(np.array(digit_img), dtype=torch.float32).transpose(0, 1) / 255.0
        
        canvas, bbox = self._place_digit_on_canvas(digit_tensor, idx)
        latent_pos = self._compute_oracle_positions(canvas, bbox)
        
        x_indices, y_indices = self.get_position_coordinates(
            image_shape=(64,64), 
            new_resolution=0.2, 
            table=self.look_up.table
        )
        wavelength_indices = self.get_wavelength_coordinates(canvas.shape)
        query_indices = self.get_query_indices(canvas.shape)
        label_segment = (canvas[0] > 0.1).float().unsqueeze(0).repeat(self.num_bands, 1, 1)
        
        # [Channels, H, W, Features]
        image_tokens = torch.cat([
            canvas.unsqueeze(-1), x_indices, y_indices, 
            wavelength_indices, label_segment.unsqueeze(-1), query_indices,
        ], dim=-1)
        
        image_tokens = einops.rearrange(image_tokens, "b h w c -> (b h w) c")
        attention_mask = torch.zeros(image_tokens.shape[0], dtype=torch.float32)

        # Balanced Sampling for MAE/Reconstruction
        reflectance = image_tokens[:, 0]
        white_idx = (reflectance > 0.05).nonzero().squeeze(-1)
        black_idx = (reflectance <= 0.05).nonzero().squeeze(-1)
        
        n_w = self.max_tokens_reconstruction // 2
        n_b = self.max_tokens_reconstruction - n_w
        
        sel_w = white_idx[torch.randint(0, len(white_idx), (n_w,))] if len(white_idx) > 0 else torch.randint(0, len(reflectance), (n_w,))
        sel_b = black_idx[torch.randint(0, len(black_idx), (n_b,))]
        
        queries = image_tokens[torch.cat([sel_w, sel_b])]
        q_mask = torch.zeros(queries.shape[0], dtype=torch.float32)
        
        # Return label - 1 to make it 0-indexed (0-25)
        return image_tokens, attention_mask, queries, q_mask, torch.tensor(digit_label - 1, dtype=torch.long), latent_pos
    
    def get_samples_to_viz(self, idx):
        """Viz helper ensuring consistency with __getitem__."""
        digit_img, digit_label = self.emnist[idx % len(self.emnist)]
        
        # EMNIST transpose fix
        digit_tensor = torch.tensor(np.array(digit_img), dtype=torch.float32).transpose(0, 1) / 255.0
        
        canvas, bbox = self._place_digit_on_canvas(digit_tensor, idx)
        latent_pos = self._compute_oracle_positions(canvas, bbox)
        
        x_idx, y_idx = self.get_position_coordinates(
            image_shape=(64,64), 
            new_resolution=0.2, 
            table=self.look_up.table
        )
        
        wave_idx = self.get_wavelength_coordinates(canvas.shape)
        q_idx = self.get_query_indices(canvas.shape)
        label_seg = (canvas[0] > 0.1).float().unsqueeze(0).repeat(self.num_bands, 1, 1)
        
        image_tokens = torch.cat([
            canvas.unsqueeze(-1), 
            x_idx, 
            y_idx, 
            wave_idx, 
            label_seg.unsqueeze(-1), 
            q_idx
        ], dim=-1)
        
        image_to_return = einops.rearrange(canvas, "c h w -> h w c")
        flat_tokens = einops.rearrange(image_tokens, "b h w c -> (b h w) c")
        
        n_w = self.max_tokens_reconstruction // 2
        n_b = self.max_tokens_reconstruction - n_w
        refl = flat_tokens[:, 0]
        
        w_idx = (refl > 0.05).nonzero().squeeze(-1)
        b_idx = (refl <= 0.05).nonzero().squeeze(-1)
        
        if len(w_idx) > 0:
            s_w = w_idx[torch.randint(0, len(w_idx), (n_w,))]
        else:
            s_w = torch.randint(0, len(flat_tokens), (n_w,))
            
        s_b = b_idx[torch.randint(0, len(b_idx), (n_b,))]
        queries = flat_tokens[torch.cat([s_w, s_b])]
        
        return (image_to_return, flat_tokens, torch.zeros(flat_tokens.shape[0]), 
                queries, torch.zeros(queries.shape[0]), digit_label - 1, latent_pos, bbox)