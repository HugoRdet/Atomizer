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
from .FLAIR_2 import*
from datetime import datetime, timezone
import torch.distributed as dist
import time
from .lookup_positional import*
from .mask_generator import *


class Tiny_BigEarthNet(Dataset):
    def __init__(self, file_path, 
                 transform,
                 transform_tokens=None,
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
        self.mode=mode
        self.shapes=[]
        self._initialize_file()
        self.transform=transform
        self.model=model
        self.transform_tokens=transform_tokens
        self.original_mode=mode
        self.fixed_size=fixed_size
        self.fixed_resolution=fixed_resolution
        self.bands_info=dataset_config
        self.bandwidths=torch.zeros(12)
        self.wavelengths=torch.zeros(12)
        self.config_model=config_model
        self.nb_tokens=self.config_model["trainer"]["max_tokens"]
        self.look_up=look_up
        
        
        
        

        self.prepare_band_infos()
        

        if modality_mode==None:
            self.modality_mode=mode
            self.original_mode=mode
        else:
            self.modality_mode=modality_mode
            self.original_mode=modality_mode

        self.h5=None




    def _initialize_file(self):
     
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 6  # Nombre d'échantillons

            #for idx in range(self.num_samples):
            #    shape_key = int(f[f'shape_{self.mode}_{idx}'][()])
            #    self.shapes.append(shape_key)


  

    def __len__(self):
        return self.num_samples
    

    def set_modality_mode(self,mode):
        self.modality_mode=mode

    def reset_modality_mode(self):
        self.modality_mode=self.original_mode

    def prepare_band_infos(self):
        
        for idx,band in enumerate(self.bands_info["bands_sen2_info"]):
            band_data=self.bands_info["bands_sen2_info"][band]
            self.bandwidths[idx]=band_data["bandwidth"]
            self.wavelengths[idx]=band_data["central_wavelength"]
            
    


            

    def __getitem__(self, idx):


        image=None
        label=None
        id_img=None



        f = self.h5


        image = torch.tensor(f[f'image_{idx}'][:])[2:,:,:] #14;120;120
        image =random_rotate_flip(image)
        attention_mask=torch.zeros(image.shape)
        label = torch.tensor(f[f'label_{idx}'][:])
        id_img = int(f[f'id_{idx}'][()])



        image,attention_mask,new_resolution=self.transform.apply_transformations(image,attention_mask,id_img,mode=self.mode,modality_mode=self.modality_mode,f_s=self.fixed_size,f_r=self.fixed_resolution)
        
        
        new_size=image.shape[1]
        
        

        
        

        if self.model == "Atomiser":
            #12:size:size
            image_size = image.shape[-1]
            channels_size=image.shape[0]

            tmp_resolution = int(10.0/new_resolution*1000)
            resolution_tmp=10.0/new_resolution
            #print(f"Resolution: {tmp_resolution}, Size: {new_size}, Channels: {channels_size} new resolution: {new_resolution}")
            
            
            
            
            # Get global offset for this modality
            global_offset = self.look_up.table[(tmp_resolution, image_size)]
            
            #p_x=torch.linspace(-image_size/2.0*resolution_tmp,image_size/2.0*resolution_tmp,image_size)
            #p_x=p_x/1200
            
            #p_y=torch.linspace(-image_size/2.0*resolution_tmp,image_size/2.0*resolution_tmp,image_size)
            #p_y=p_y/1200
            
            # Create LOCAL pixel indices (0 to image_size-1)
            y_indices = torch.arange(image_size).unsqueeze(1).expand(image_size, image_size)
            x_indices = torch.arange(image_size).unsqueeze(0).expand(image_size, image_size)
            
            # Create LOCAL pixel indices (0 to image_size-1)
            
            
            # Convert to GLOBAL indices by adding offset
            x_indices = x_indices + global_offset
            y_indices = y_indices + global_offset
            
            # Expand for all bands
            x_indices = repeat(x_indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)
            y_indices = repeat(y_indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)
            
            
            idxs_bandwidths = []
            
            for idx_b in range(self.bandwidths.shape[0]):
                idxs_bandwidths.append(self.look_up.table_wave[(int(self.bandwidths[idx_b].item()), int(self.wavelengths[idx_b].item()))])
                
            idxs_bandwidths = torch.tensor(idxs_bandwidths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idxs_bandwidths = repeat(idxs_bandwidths, "b h w c -> b (h h1) (w w1) c", h1=image_size, w1=image_size)
            
            
            
            # Concatenate all token data
            image = torch.cat([
                image.unsqueeze(-1),      # Band values
                x_indices.float(),        # Global X indices
                y_indices.float(),        # Global Y indices  
                idxs_bandwidths.float()  # Bandwidth indices
            ], dim=-1)
            
            
            # Reshape and sample tokens
            image = rearrange(image, "b h w c -> (b h w) c")
            attention_mask= rearrange(attention_mask, "c h w -> (c h w)")
            image=image[attention_mask==0.0]
            tmp_rand = torch.randperm(image.shape[0])
            image = image[tmp_rand[:self.nb_tokens]]
            attention_mask=torch.zeros(image.shape[0])
       
            
            
            
            
            
            current_len = image.shape[0]
            target_len = self.nb_tokens

            if current_len < target_len:
                # Repeat full image as many times as needed
                repeat_factor = target_len // current_len
                remainder = target_len % current_len

                repeated_image = image.repeat((repeat_factor, 1))  # [repeat_factor * d, 4]
                if remainder > 0:
                    remainder_image = image[:remainder]            # [remainder, 4]
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

            # If needed, truncate in case of over-padding (not likely but safe)
            image = image[:target_len]
            attention_mask = attention_mask[:target_len]

                
            
            
            return image, attention_mask, new_resolution, new_size, label, id_img
        
        return image, attention_mask, new_resolution, new_size, label, id_img
    
  
    

class Tiny_BigEarthNet_MAE(Dataset):
    def __init__(self, file_path, 
                 transform,
                 transform_tokens=None,
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
        self.transform_tokens = transform_tokens
        self.original_mode = mode
        self.fixed_size = fixed_size
        self.fixed_resolution = fixed_resolution
        self.bands_info = dataset_config
        self.bandwidths = torch.zeros(12)
        self.wavelengths = torch.zeros(12)
        self.config_model = config_model
        self.nb_tokens = self.config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = self.config_model["trainer"]["max_tokens_reconstruction"]
        self.reconstruction_viz_idx = self.config_model["debug"]["idxs_to_viz"]

        self.look_up = look_up
        self.mask_gen = IJEPAStyleMaskGenerator(
            input_size=(120,120),
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

    def _initialize_file(self):
        """Initialize file and get number of samples."""
        print("self file_path:", self.file_path)
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 6  # Number of samples

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
        for idx, band in enumerate(self.bands_info["bands_sen2_info"]):
            band_data = self.bands_info["bands_sen2_info"][band]
            self.bandwidths[idx] = band_data["bandwidth"]
            self.wavelengths[idx] = band_data["central_wavelength"]

    def get_position_coordinates(self, image_shape, new_resolution):
        image_size = image_shape[-1]
        channels_size = image_shape[0]

        tmp_resolution = int(new_resolution*1000)
        global_offset = self.look_up.table[(tmp_resolution, image_size)]
        
        
        # Create LOCAL pixel indices (0 to image_size-1)
        y_indices = torch.arange(image_size).unsqueeze(1).expand(image_size, image_size)
        x_indices = torch.arange(image_size).unsqueeze(0).expand(image_size, image_size)
        
        x_indices = x_indices + global_offset
        y_indices = y_indices + global_offset
        
        # Expand for all bands
        x_indices = repeat(x_indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)
        y_indices = repeat(y_indices.unsqueeze(0), "u h w -> (u r) h w", r=channels_size).unsqueeze(-1)

        return x_indices, y_indices
    
    def get_wavelengths_coordinates(self, image_shape):
        image_size = image_shape[-1]
        channels_size = image_shape[0]

        idxs_bandwidths = []
        
        for idx_b in range(self.bandwidths.shape[0]):
            idxs_bandwidths.append(self.look_up.table_wave[(int(self.bandwidths[idx_b].item()), int(self.wavelengths[idx_b].item()))])
            
        idxs_bandwidths = torch.tensor(idxs_bandwidths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        idxs_bandwidths = repeat(idxs_bandwidths, "b h w c -> b (h h1) (w w1) c", h1=image_size, w1=image_size)
        
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

            repeated_image = image.repeat((repeat_factor, 1))  # [repeat_factor * d, 4]
            if remainder > 0:
                remainder_image = image[:remainder]            # [remainder, 4]
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

    def __getitem__(self, idx):
        image = None
        label = None
        id_img = None
        idx=0
        

        # Ensure HDF5 file is open
        f = self._ensure_h5_open()

        image = torch.tensor(f[f'image_{idx}'][:])[2:,:,:]  # Fixed: use idx, not 0
        #image = random_rotate_flip(image)
        attention_mask = torch.zeros(image.shape)
        label = torch.tensor(f[f'label_{idx}'][:])
        id_img = int(f[f'id_{idx}'][()])
        
        mask_MAE = None

        image, attention_mask, new_resolution_factor = self.transform.apply_transformations(
            image, attention_mask, id_img, mode=self.mode, modality_mode=self.modality_mode, 
            f_s=self.fixed_size, f_r=self.fixed_resolution
        )
        
        new_resolution=10.0/new_resolution_factor #in m/px
        
        
        
        self.mask_gen.H, self.mask_gen.W = image.shape[1], image.shape[2]
        mask_MAE = self.mask_gen.generate_mask()
        mask_MAE = mask_MAE.repeat(image.shape[0], 1, 1)  # Repeat for all bands
        
        idxs_bandwidths = self.get_wavelengths_coordinates(image.shape)
        x_indices, y_indices = self.get_position_coordinates(image.shape, new_resolution)
        
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
        MAE_mask = rearrange(mask_MAE, "c h w -> (c h w)")
        
        # Filter valid tokens
        image = image[attention_mask==0.0]          # image get resized and invalid bands removed
        mask_MAE = MAE_mask[attention_mask==0.0]    # same for mask
        
        # Shuffle tokens
        image, mask_MAE = self.shuffle_arrays([image, mask_MAE])
        
        # Split into input and target tokens
        input_tokens = image[mask_MAE==0.0].clone()
        mae_tokens = image[mask_MAE==1.0].clone()
        
        # Take required number of tokens
        image = input_tokens[:self.nb_tokens]
        mae_tokens = mae_tokens[:self.max_tokens_reconstruction]
        
        mae_tokens, mae_tokens_mask = self.padding_mae(mae_tokens)
        image, attention_mask = self.padding_image(image)

        return image, attention_mask, mae_tokens, mae_tokens_mask, label
 
    def get_samples_to_viz(self, idx):
        """Fixed version that properly opens HDF5 file."""
        # Ensure HDF5 file is open
        f = self._ensure_h5_open()
        idx=0
        
        
        
        
        image = torch.tensor(f[f'image_{idx}'][:])[2:,:,:]  
        #image= random_rotate_flip(image)
        attention_mask = torch.zeros(image.shape)
        
        mask_MAE = None
        new_resolution = 10.0 #in m/px
        
        self.mask_gen.H, self.mask_gen.W = image.shape[1], image.shape[2]
        mask_MAE = self.mask_gen.generate_mask()
        mask_MAE_res = mask_MAE.clone()
        mask_MAE = mask_MAE.repeat(image.shape[0], 1, 1)  # Repeat for all bands
        
        idxs_bandwidths = self.get_wavelengths_coordinates(image.shape)
        x_indices, y_indices = self.get_position_coordinates(image.shape, new_resolution)
        
        
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
        MAE_mask = rearrange(mask_MAE, "c h w -> (c h w)")
        mae_tokens = image.clone()
        image = image[attention_mask==0.0]          # image get resized and invalid bands removed
        
        
        mask_MAE = MAE_mask[attention_mask==0.0]    # same for mask
        
        input_tokens = image[mask_MAE==0.0].clone()
        attention_mask=torch.zeros(input_tokens.shape[0])
        
        return input_tokens, attention_mask, mae_tokens, mask_MAE_res

    def close(self):
        """Close HDF5 file if it's open."""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __del__(self):
        """Ensure HDF5 file is closed when dataset is deleted."""
        self.close()




