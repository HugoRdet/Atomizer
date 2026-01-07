import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset,DataLoader,Sampler
import h5py
from tqdm import tqdm
from .image_utils import*
from .utils_dataset import*
import random
from training.utils.FLAIR_2 import*
from datetime import datetime, timezone
import torch.distributed as dist
import time
from training.utils.lookup_positional import*
from .mask_generator import *
import os


"""
Visualize oracle (complexity-based) latent placement on an image.
PyTorch implementation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

os.makedirs("./figures", exist_ok=True)


"""
Visualize oracle (complexity-based) latent placement on an image.
PyTorch implementation with gradient emphasis options.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

os.makedirs("./figures", exist_ok=True)



def del_file(path):
    if os.path.exists(path):
        os.remove(path)


import random
import torch
from torchvision.transforms.functional import rotate, hflip, vflip


class FLAIR_MAE_err(Dataset):

    def __init__(self, file_path, 
                 transform,

                 model="None",
                 mode="train",
                 modality_mode=None,
                 fixed_size=None,
                 fixed_resolution=None,
                 dataset_config=None,
                 config_model=None,
                 look_up=None):
        
        self.file_path = file_path
        self.num_samples = None
        self.mode = mode
        self.shapes = []
        self._initialize_file()
        self.transform = transform
        self.model = model
        self.original_mode = mode
        self.fixed_size = fixed_size
        self.fixed_resolution = fixed_resolution
        self.bands_info = dataset_config
        self.bandwidths = torch.zeros(5)
        self.wavelengths = torch.zeros(5)
        self.config_model = config_model
        self.nb_tokens = self.config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = self.config_model["trainer"]["max_tokens_reconstruction"]
        self.reconstruction_viz_idx = self.config_model["debug"]["idxs_to_viz"]

        self.look_up = look_up
        self.mask_gen = IJEPAStyleMaskGenerator(
            input_size=(512,512),
            mask_ratio_range=self.config_model["masking_MAE"]["mask_ratio_range"],       
            patch_ratio_range=self.config_model["masking_MAE"]["patch_ratio_range"],    
            aspect_ratio_range=self.config_model["masking_MAE"]["aspect_ratio_range"]    
        )

        self.prepare_band_infos()
        
        if modality_mode == None:
            self.modality_mode = mode
            self.original_mode = mode
        else:
            self.modality_mode = modality_mode
            self.original_mode = modality_mode

        # Initialize h5 to None - will be opened when needed
        self.h5 = None
        
        # Oracle latent positions config
        self.num_spatial_latents = config_model["Atomiser"]["spatial_latents"] ** 2  # e.g., 35*35
        self.gsd = 0.2  # meters per pixel
        self.gradient_power = 2.0
        self.min_threshold = 0.1
        
        # Precompute oracle positions for all images
        self.oracle_positions = self._precompute_oracle_positions()

    def _initialize_file(self):
        """Initialize file and get number of samples."""
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 8  # Number of samples

    def _precompute_oracle_positions(self):
     
        """Precompute oracle latent positions for all images, or load from cache."""
        import os
        
        # Cache file path
        cache_dir = os.path.dirname(self.file_path)
        cache_file = os.path.join(cache_dir, f"precomputed_oracle_{self.mode}.pt")
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"Loading precomputed oracle positions from {cache_file}")
            positions = torch.load(cache_file)
            print(f"Loaded {len(positions)} oracle positions")
            return positions
        
        # Compute if cache doesn't exist
        print(f"Precomputing oracle latent positions for {self.num_samples} images...")
        positions = {}
        
        with h5py.File(self.file_path, 'r') as f:
            for idx in tqdm(range(self.num_samples), desc="Oracle positions"):
                # Load image
                im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5, 512, 512]
                
                # Convert to RGB [H, W, 3]
                rgb_image = im_aerial[[2, 1, 0]].permute(1, 2, 0)
                normalized = normalize(rgb_image)
                
                # Compute oracle positions
                pos = compute_oracle_latent_positions_meters(
                    normalized,
                    num_latents=self.num_spatial_latents,
                    gsd=self.gsd,
                    gradient_power=self.gradient_power,
                    min_threshold=self.min_threshold,
                    sigma=None,
                )
                positions[idx] = pos  # [L, 2]
        
        # Save to cache
        torch.save(positions, cache_file)
        
        # Memory usage
        total_bytes = self.num_samples * self.num_spatial_latents * 2 * 4
        print(f"Precomputed {len(positions)} oracle positions ({total_bytes / 1e6:.2f} MB)")
        print(f"Saved to {cache_file}")
        
        return positions

    def _ensure_h5_open(self):
        """Ensure HDF5 file is open. Open it if it's None."""
        if self.h5 is None:
            self.h5 = h5py.File(self.file_path, 'r')
        return self.h5

    def __len__(self):
        return self.num_samples
    
    def set_modality_mode(self, mode):
        self.modality_mode = mode

    def reset_modality_mode(self):
        self.modality_mode = self.original_mode

    def prepare_band_infos(self):
        res_band=[]
        res_wave=[]
        for idx, band in enumerate(self.bands_info["bands_FLAIR_info"]):
            band_data = self.bands_info["bands_FLAIR_info"][band]
            res_band.append(band_data["bandwidth"])
            res_wave.append(band_data["central_wavelength"])
        self.bandwidths=torch.Tensor(res_band)
        self.wavelengths=torch.Tensor(res_wave)

    def get_position_coordinates(self, image_shape, new_resolution, table=None):
        image_size = image_shape[-1]
        channels_size = image_shape[0]
        
        tmp_resolution = int(new_resolution * 1000)
        global_offset = table[(tmp_resolution, image_size)]
        
        # Create meshgrid - clearer and more standard
        y_coords = torch.arange(image_size)  # 0 to image_size-1
        x_coords = torch.arange(image_size)  # 0 to image_size-1
        
        # meshgrid returns (X, Y) grid
        x_indices, y_indices = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Add global offset
        x_indices = x_indices + global_offset
        y_indices = y_indices + global_offset

        # Expand for all bands
        x_indices = einops.repeat(x_indices, "h w -> r h w", r=channels_size).unsqueeze(-1)
        y_indices = einops.repeat(y_indices, "h w -> r h w", r=channels_size).unsqueeze(-1)
        
        return x_indices, y_indices
    
    def get_position_coordinates_queries(self, image_shape, new_resolution, table=None):
        image_size = image_shape[-1]
        channels_size = image_shape[0]

        resolution_latents = 0.2  # m
        tmp_resolution = int(resolution_latents * 1000)
        global_offset = table[(tmp_resolution, image_size)]
        
        # Create LOCAL pixel indices (0 to image_size-1)
        indices = torch.full((image_size, image_size), global_offset)
        
        # Expand for all bands
        indices = einops.repeat(indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)

        return indices
    
    def get_wavelengths_coordinates(self, image_shape):
        image_size = image_shape[-1]
        channels_size = image_shape[0]

        idxs_bandwidths = []
        
        for idx_b in range(self.bandwidths.shape[0]):
            idxs_bandwidths.append(self.look_up.table_wave[(int(self.bandwidths[idx_b].item()), int(self.wavelengths[idx_b].item()))])
            
        idxs_bandwidths = torch.tensor(idxs_bandwidths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        idxs_bandwidths = einops.repeat(idxs_bandwidths, "b h w c -> b (h h1) (w w1) c", h1=image_size, w1=image_size)
        
        return idxs_bandwidths
    
    def shuffle_arrays(self, arrays: list):
        tmp_rand = torch.randperm(arrays[0].shape[0])
        res = []
        for tmp_array in arrays:
            tmp_array = tmp_array[tmp_rand]
            res.append(tmp_array)
        return res
    
    def padding_mae(self, mae_tokens):
        mae_tokens_mask = torch.zeros(mae_tokens.shape[0], dtype=torch.float32)
        
        # Padding mae tokens
        if mae_tokens.shape[0] < self.max_tokens_reconstruction:
            padding_size = self.max_tokens_reconstruction - mae_tokens.shape[0]
            
            # Create padding with same dtype as mae_tokens
            padding_mae = torch.zeros((padding_size, mae_tokens.shape[1]), dtype=mae_tokens.dtype)
            mae_tokens = torch.cat([mae_tokens, padding_mae], dim=0)
            
            # Create padding mask with same dtype as mae_tokens_mask (float32, not bool)
            padding_mae_mask = torch.ones(padding_size, dtype=torch.float32)  # 1.0 for padded tokens
            mae_tokens_mask = torch.cat([mae_tokens_mask, padding_mae_mask], dim=0)
            
        return mae_tokens, mae_tokens_mask
    
    def padding_image(self, image):
        # Create attention mask for input tokens
        attention_mask = torch.zeros(image.shape[0], dtype=torch.float32)
        
        # Handle input token padding
        current_len = image.shape[0]
        target_len = self.nb_tokens

        if current_len < target_len:
            # Repeat full image as many times as needed
            repeat_factor = target_len // current_len
            remainder = target_len % current_len

            repeated_image = image.repeat((repeat_factor, 1))
            if remainder > 0:
                remainder_image = image[:remainder]
                image = torch.cat([repeated_image, remainder_image], dim=0)
            else:
                image = repeated_image

            # Repeat the attention mask the same way
            repeated_mask = attention_mask.repeat(repeat_factor)
            if remainder > 0:
                remainder_mask = attention_mask[:remainder]
                attention_mask = torch.cat([repeated_mask, remainder_mask], dim=0)
            else:
                attention_mask = repeated_mask
        
        return image, attention_mask
    
    def process_mask(self, mask):
        mask = mask.float()
        mask[mask > 13] = 13
        mask = mask - 1
        return mask

    def __getitem__(self, idx):
        label = None
        id_img = None

        # Ensure HDF5 file is open
        f = self._ensure_h5_open()

        im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5,512,512]
        
        # Get precomputed oracle positions (fast lookup!)
        latent_pos =self.oracle_positions[idx].clone()  # [L, 2]

        label = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]
        label = self.process_mask(label)
        attention_mask = torch.zeros(im_aerial.shape)

        new_resolution = 0.2  # m/px
        label_segment = label.clone()
        label_segment = label_segment.repeat(im_aerial.shape[0], 1, 1)
        
        idxs_bandwidths = self.get_wavelengths_coordinates(im_aerial.shape)
        x_indices, y_indices = self.get_position_coordinates(im_aerial.shape, new_resolution, table=self.look_up.table)
        indices_queries = self.get_position_coordinates_queries(im_aerial.shape, new_resolution, table=self.look_up.table_queries)
        
        # Concatenate all token data
        image = torch.cat([
            im_aerial.unsqueeze(-1),      # Band values
            x_indices.float(),            # Global X indices
            y_indices.float(),            # Global Y indices  
            idxs_bandwidths.float(),      # Bandwidth indices
            label_segment.float().unsqueeze(-1),
            indices_queries.float()
        ], dim=-1)
        
        queries = image.clone()
        image_err=image.clone()
        
        # Reshape and sample tokens
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries = einops.rearrange(queries, "b h w c -> (b h w) c")
        attention_mask = einops.rearrange(attention_mask, "c h w -> (c h w)")
        image = image[attention_mask == 0.0]

        queries = self.shuffle_arrays([queries])[0]
      
        nb_queries = self.config_model["trainer"]["max_tokens_reconstruction"]
        queries = queries[:nb_queries]
        queries_mask = torch.zeros(queries.shape[0])

        latent_pos=self.shuffle_arrays([latent_pos])[0]
        latent_pos=latent_pos[:625]


        return image, attention_mask, queries, queries_mask, label, latent_pos, image_err
 
    def get_samples_to_viz(self, idx):
        label = None
        id_img = None

        # Ensure HDF5 file is open
        f = self._ensure_h5_open()

        im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5,512,512]

        # Get precomputed oracle positions (fast lookup!)
        latent_pos = self.oracle_positions[idx].clone()  # [L, 2]

        image_to_return = im_aerial.clone()
        image_to_return = einops.rearrange(image_to_return, "c h w -> h w c")

        label = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]
        label = self.process_mask(label)
        attention_mask = torch.zeros(im_aerial.shape)

        new_resolution = 0.2  # m/px
        label_segment = label.clone()
        label_segment = label_segment.repeat(im_aerial.shape[0], 1, 1)
        
        idxs_bandwidths = self.get_wavelengths_coordinates(im_aerial.shape)
        x_indices, y_indices = self.get_position_coordinates(im_aerial.shape, new_resolution, table=self.look_up.table)
        indices_queries = self.get_position_coordinates_queries(im_aerial.shape, new_resolution, table=self.look_up.table_queries)
        
        # Concatenate all token data
        image = torch.cat([
            im_aerial.unsqueeze(-1),      # Band values
            x_indices.float(),            # Global X indices
            y_indices.float(),            # Global Y indices  
            idxs_bandwidths.float(),      # Bandwidth indices
            label_segment.float().unsqueeze(-1),
            indices_queries.float()
        ], dim=-1)
        
        queries = image.clone()
        image_err=image.clone()

        # Reshape and sample tokens
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries = einops.rearrange(queries, "b h w c -> (b h w) c")
        attention_mask = einops.rearrange(attention_mask, "c h w -> (c h w)")

        image = image[attention_mask == 0.0]

        queries_mask = torch.zeros(queries.shape[0])

        #latent_pos=self.shuffle_arrays([latent_pos])[0]
        latent_pos=latent_pos[:625]

        return image_to_return, image, attention_mask, queries, queries_mask, label, latent_pos,image_err

