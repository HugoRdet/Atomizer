import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import EuroSAT
from einops import rearrange, repeat
import random

class EuroSAT_Atomizer(Dataset):
    def __init__(self, 
                 root,
                 transform=None,
                 download=False,
                 model="None",
                 mode="train",
                 modality_mode=None,
                 fixed_size=None,
                 fixed_resolution=None,
                 dataset_config=None,
                 config_model=None,
                 look_up=None):
        
        # Initialize the base EuroSAT dataset
        self.eurosat = EuroSAT(
            root=root,
            transform=None,  # We'll handle transforms ourselves
            download=download
        )
        
        self.transform = transform
        self.model = model
        self.mode = mode
        self.original_mode = mode
        self.fixed_size = fixed_size
        self.fixed_resolution = fixed_resolution
        self.bands_info = dataset_config
        self.config_model = config_model
        self.look_up = look_up
        
        # EuroSAT has 13 bands, adjust if needed
        self.bandwidths = torch.zeros(13)
        self.wavelengths = torch.zeros(13)
        
        if config_model:
            self.nb_tokens = self.config_model["trainer"]["max_tokens"]
        else:
            self.nb_tokens = 1000  # default value
        
        if modality_mode is None:
            self.modality_mode = mode
            self.original_mode = mode
        else:
            self.modality_mode = modality_mode
            self.original_mode = modality_mode
            
        self.prepare_band_infos()

    def __len__(self):
        return len(self.eurosat)
    
    def set_modality_mode(self, mode):
        self.modality_mode = mode

    def reset_modality_mode(self):
        self.modality_mode = self.original_mode

    def prepare_band_infos(self):
        """Prepare band information for Sentinel-2 data (EuroSAT uses Sentinel-2)"""
        if self.bands_info and "bands_sen2_info" in self.bands_info:
            for idx, band in enumerate(self.bands_info["bands_sen2_info"]):
                if idx < len(self.bandwidths):
                    band_data = self.bands_info["bands_sen2_info"][band]
                    self.bandwidths[idx] = band_data["bandwidth"]
                    self.wavelengths[idx] = band_data["central_wavelength"]
        else:
            # Default Sentinel-2 band information for EuroSAT
            # EuroSAT uses 13 bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
            default_wavelengths = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
            default_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 20, 90, 180]
            
            for i in range(min(13, len(self.wavelengths))):
                self.wavelengths[i] = default_wavelengths[i]
                self.bandwidths[i] = default_bandwidths[i]

    def random_rotate_flip(self, image):
        """Apply random rotation and flipping (adapted from your original code)"""
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            
        # Random vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[1])
            
        return image

    def __getitem__(self, idx):
        # Get the original sample from EuroSAT
        image, label = self.eurosat[idx]
        
        # Convert PIL image to tensor if needed
        if not isinstance(image, torch.Tensor):
            # EuroSAT images are PIL Images, convert to tensor
            image = T.ToTensor()(image)
        
        # Get image dimensions
        channels, height, width = image.shape
        
        # Apply random transformations
        image = self.random_rotate_flip(image)
        
        # Create attention mask (no masking for now)
        attention_mask = torch.zeros(image.shape)
        
        # Get a dummy id (since EuroSAT doesn't have specific IDs)
        id_img = idx
        
        
        new_resolution = 10.0  # Default resolution in meters/pixel for EuroSAT
        new_size = image.shape[1]
        
        if self.model == "Atomiser":
            # Your Atomiser-specific processing
            image_size = image.shape[-1]
            channels_size = image.shape[0]

            tmp_resolution = int(10.0/new_resolution*1000)
            resolution_tmp = 10.0/new_resolution
            
            # Get global offset for this modality
            if self.look_up:
                global_offset = self.look_up.table.get((tmp_resolution, image_size), 0)
            else:
                global_offset = 0
            
            # Create LOCAL pixel indices (0 to image_size-1)
            y_indices = torch.arange(image_size).unsqueeze(1).expand(image_size, image_size)
            x_indices = torch.arange(image_size).unsqueeze(0).expand(image_size, image_size)
            
            # Convert to GLOBAL indices by adding offset
            x_indices = x_indices + global_offset
            y_indices = y_indices + global_offset
            
            # Expand for all bands
            x_indices = repeat(x_indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)
            y_indices = repeat(y_indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)
            
            # Prepare bandwidth indices
            idxs_bandwidths = []
            for idx_b in range(min(self.bandwidths.shape[0], channels_size)):
                if self.look_up and hasattr(self.look_up, 'table_wave'):
                    bandwidth_key = (int(self.bandwidths[idx_b].item()), int(self.wavelengths[idx_b].item()))
                    idxs_bandwidths.append(self.look_up.table_wave.get(bandwidth_key, idx_b))
                else:
                    idxs_bandwidths.append(idx_b)  # Default to band index
                    
            idxs_bandwidths = torch.tensor(idxs_bandwidths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idxs_bandwidths = repeat(idxs_bandwidths, "b h w c -> b (h h1) (w w1) c", h1=image_size, w1=image_size)
            
            # Concatenate all token data
            image = torch.cat([
                image.unsqueeze(-1),      # Band values
                x_indices.float(),        # Global X indices
                y_indices.float(),        # Global Y indices  
                idxs_bandwidths.float()   # Bandwidth indices
            ], dim=-1)
            
            # Reshape and sample tokens
            image = rearrange(image, "b h w c -> (b h w) c")
            attention_mask = rearrange(attention_mask, "c h w -> (c h w)")
            
            # Filter valid tokens
            image = image[attention_mask == 0.0]
            
            # Shuffle and sample tokens
            tmp_rand = torch.randperm(image.shape[0])
            image = image[tmp_rand[:self.nb_tokens]]
            attention_mask = torch.zeros(image.shape[0])
            
            # Handle padding/repetition if needed
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

            # Ensure exact length
            image = image[:target_len]
            attention_mask = attention_mask[:target_len]
            
            return image, attention_mask, new_resolution, new_size, label, id_img
        
        return image, attention_mask, new_resolution, new_size, label, id_img

