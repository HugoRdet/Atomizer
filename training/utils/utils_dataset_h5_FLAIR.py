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
import os

def del_file(path):
    if os.path.exists(path):
        os.remove(path)

def sample_query_tokens_optimal(
        image: torch.Tensor,  # [C, H, W, 6]
        patch_size: int = 16,
        percent_patch: float = 0.7,
        nb_max_tokens: int = 10000,
        shuffle: bool = True,
    ) -> torch.Tensor:
        """
        Optimal query sampling for BOTH high-frequency AND spectral learning.
        
        - Spatial patches: Strong gradient signal for edges
        - All channels per location: Forces spectral differentiation
        - Random fill: Coverage diversity
        
        Args:
            image: [C, H, W, 6]
            patch_size: Patch size in pixels
            percent_patch: Fraction from patches (rest is random)
            nb_max_tokens: Maximum tokens to return
            shuffle: Shuffle final order
        
        Returns:
            tokens: [nb_tokens, 6]
        """
        C, H, W, D = image.shape
        device = image.device
        dtype = image.dtype
        
        nb_patch_tokens_target = int(nb_max_tokens * percent_patch)
        tokens_per_spatial_patch = patch_size * patch_size * C
        
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_spatial_patches = num_patches_h * num_patches_w
        
        # Fallback for tiny images
        if total_spatial_patches == 0:
            all_tokens = einops.rearrange(image, 'c h w d -> (c h w) d')
            indices = torch.randperm(all_tokens.shape[0], device=device)[:nb_max_tokens]
            return all_tokens[indices]
        
        # How many spatial patches needed?
        num_patches_needed = (nb_patch_tokens_target + tokens_per_spatial_patch - 1) // tokens_per_spatial_patch
        num_patches_to_select = min(num_patches_needed, total_spatial_patches)
        
        # Reshape into patches: [num_patches_h, num_patches_w, C, patch_size, patch_size, D]
        image_patched = image.view(C, num_patches_h, patch_size, num_patches_w, patch_size, D)
        image_patched = image_patched.permute(1, 3, 0, 2, 4, 5)  # [nph, npw, C, ps, ps, D]
        image_patched = image_patched.reshape(total_spatial_patches, C, patch_size, patch_size, D)
        
        # Randomly select patches
        patch_perm = torch.randperm(total_spatial_patches, device=device)[:num_patches_to_select]
        selected_patches = image_patched[patch_perm]  # [num_selected, C, ps, ps, D]
        
        # Flatten: [num_selected, C, ps, ps, D] -> [num_selected * C * ps * ps, D]
        patch_tokens = einops.rearrange(selected_patches, 'n c h w d -> (n c h w) d')
        
        # Trim to target
        if patch_tokens.shape[0] > nb_patch_tokens_target:
            patch_tokens = patch_tokens[:nb_patch_tokens_target]
        
        actual_patch_tokens = patch_tokens.shape[0]
        nb_random_needed = nb_max_tokens - actual_patch_tokens
        
        # Random sampling from non-patch regions (all channels at each pixel)
        if nb_random_needed > 0:
            # Mask of selected patches
            selected_ph = patch_perm // num_patches_w
            selected_pw = patch_perm % num_patches_w
            
            # Create pixel-level mask
            patch_mask = torch.zeros(num_patches_h, num_patches_w, device=device, dtype=torch.bool)
            patch_mask[selected_ph, selected_pw] = True
            
            # Expand to pixel level
            pixel_mask = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
            
            # Full image mask (handle edges outside patch grid)
            full_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
            full_mask[:num_patches_h * patch_size, :num_patches_w * patch_size] = pixel_mask
            
            # Available pixels
            available_h, available_w = torch.where(~full_mask)
            num_available = len(available_h)
            
            if num_available > 0:
                # Sample spatial positions (all channels at each)
                pixels_needed = (nb_random_needed + C - 1) // C
                num_to_sample = min(pixels_needed, num_available)
                
                perm = torch.randperm(num_available, device=device)[:num_to_sample]
                sampled_h = available_h[perm]
                sampled_w = available_w[perm]
                
                # Gather all channels: [C, num_sampled, D] -> [num_sampled, C, D]
                random_tokens = image[:, sampled_h, sampled_w, :].permute(1, 0, 2)
                random_tokens = einops.rearrange(random_tokens, 'n c d -> (n c) d')
                
                if random_tokens.shape[0] > nb_random_needed:
                    random_tokens = random_tokens[:nb_random_needed]
            else:
                random_tokens = torch.empty(0, D, device=device, dtype=dtype)
        else:
            random_tokens = torch.empty(0, D, device=device, dtype=dtype)
        
        # Combine
        final_tokens = torch.cat([patch_tokens, random_tokens], dim=0)
        
        if final_tokens.shape[0] > nb_max_tokens:
            final_tokens = final_tokens[:nb_max_tokens]
        
        if shuffle:
            perm = torch.randperm(final_tokens.shape[0], device=device)
            final_tokens = final_tokens[perm]
        
        return final_tokens

def filter_dates(mask, clouds:bool=2, area_threshold:float=0.5, proba_threshold:int=60):
    """ Mask : array T*2*H*W
        Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
        Area_threshold : threshold on the surface covered by the clouds / snow 
        Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)
        Return array of indexes to keep
    """
    dates_to_keep = []
    
    for t in range(mask.shape[0]):
        if clouds != 2:
            cover = np.count_nonzero(mask[t, clouds, :,:]>=proba_threshold)
        else:
            cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
        cover /= mask.shape[2]*mask.shape[3]
        if cover < area_threshold:
            dates_to_keep.append(t)

    return dates_to_keep


def monthly_image(sp_patch, sp_raw_dates):
    average_patch, average_dates = [], []
    month_range = pd.period_range(
        start=sp_raw_dates[0].strftime('%Y-%m-%d'),
        end=sp_raw_dates[-1].strftime('%Y-%m-%d'),
        freq='M'
    )

    for m in month_range:
        month_dates = [i for i, date in enumerate(sp_raw_dates)
                       if date.month == m.month and date.year == m.year]

        if month_dates:
            average_patch.append(np.mean(sp_patch[month_dates], axis=0))
            # use the datetime CLASS you imported above
            average_dates.append(datetime(m.year, m.month, 1))

    return np.array(average_patch), average_dates


def create_dataset_flair(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtd, name="tiny", mode="train", stats=None,max_samples=-1):
    """
    Creates an HDF5 dataset using the given sample indices (dico_idxs) from ds.
    If stats (per-channel mean/std) is None, computes it on-the-fly in a streaming fashion.
    Then applies normalization to each image: (image - mean) / std

    Args:
        dico_idxs (dict): Mapping from some key to a list of sample indices
        ds: A dataset that supports ds[idx] -> (image, label)
            where image is shape (12, 120, 120).
        name (str): HDF5 file prefix
        mode (str): e.g. "train" or "test"
        stats (torch.Tensor or None): shape (12,2) with [:,0] as mean, [:,1] as std

    Returns:
        stats (torch.Tensor): The per-channel mean/std used for normalization
    """
    
    if stats is None:
        stats =compute_channel_mean_std_FLAIR(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtd)
        
    

    # 1) Clean up any existing file
    h5_path = f'./data/custom_flair/{name}_{mode}.h5'
    del_file(h5_path)
  
    # 2) If stats is not given, compute it in a streaming fashion
    

        


    # 3) Create a new HDF5 file
    db = h5py.File(h5_path, 'w')
    cpt_sample=0


    # 4) Iterate through your dictionary of IDs, fetch images, and store them
    for idx_img in tqdm(range(len(images))):
        
        im_aer,mask,sen_spatch,img_dates,sen_mask,aerial_date=get_sample(idx_img,images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtd, palette=lut_colors)
        
        

        # Convert to float (if needed) before normalization
        im_aer = im_aer.astype(float)
        sen_spatch=sen_spatch.astype(float)
        mask=mask.astype(int)


        # Apply per-channel normalization
        # normalized_value = (value - mean[channel]) / std[channel]
        
        im_aer = (im_aer - stats["im_mean"][:, None, None]) / stats["im_std"][:, None, None]
        sen_spatch = (sen_spatch - stats["sen_mean"][:, None, None]) / stats["sen_std"][:, None, None]


        to_keep = filter_dates(sen_mask, clouds=2, area_threshold=0.5, proba_threshold=60)
         
        sen_spatch = sen_spatch[to_keep]
        img_dates=img_dates[to_keep]


        sen_spatch, img_dates =monthly_image(sen_spatch, img_dates)

        days=[]
        months=[]
        years=[]
        for tmp_date in img_dates:
            tmp_day=tmp_date.day 
            tmp_month=tmp_date.month
            tmp_year=tmp_date.year
            days.append(tmp_day)
            months.append(tmp_month)
            years.append(tmp_year)


        


        im_aer = im_aer.astype(np.float32)
        sen_spatch = sen_spatch.astype(np.float32)
        days = np.array(days, dtype=np.float32)
        months = np.array(months, dtype=np.float32)
        years = np.array(years, dtype=np.float32)
        mask = mask.astype(np.float32)
        mask[mask>13]=13
        
        sen_mask = sen_mask.astype(np.float32)


 


        
        
        # Convert back to numpy to store in HDF5
        db.create_dataset(f'img_aerial_{idx_img}', data=im_aer)
        db.create_dataset(f'img_sen_{idx_img}', data=sen_spatch)
        db.create_dataset(f'days_{idx_img}', data=days)
        db.create_dataset(f'months_{idx_img}', data=months)
        db.create_dataset(f'years_{idx_img}', data=years)
        db.create_dataset(f'mask_{idx_img}',data=mask)
        db.create_dataset(f'sen_mask_{idx_img}',data=sen_mask)
        db.create_dataset(f'aerial_mtd_{idx_img}',data=aerial_date)
        
        cpt_sample+=1
        if max_samples!=-1 and cpt_sample>max_samples:
            db.close()
            return stats
  


    db.close()
    return stats


    
            








import random
import torch
from torchvision.transforms.functional import rotate, hflip, vflip


class FLAIR_MAE(Dataset):

#class FLAIR_SEG(Dataset):
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

    def _initialize_file(self):
        """Initialize file and get number of samples."""
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 8  # Number of samples
            

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
    
    def get_position_coordinates_queries(self, image_shape, new_resolution,table=None):
        image_size = image_shape[-1]
        channels_size = image_shape[0]

        resolution_latents=0.2 #m
        tmp_resolution = int(resolution_latents*1000)
        global_offset = table[(tmp_resolution, image_size)]
        
        # Create LOCAL pixel indices (0 to image_size-1)
        indices = torch.full((image_size, image_size),global_offset)
        
        
        
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
        
        
        #im_sen = torch.tensor(f[f'img_sen_{idx}'][:], dtype=torch.float32)  # [12,10,40,40]
        #days = torch.tensor(f[f'days_{idx}'][:], dtype=torch.float32)
        #months = torch.tensor(f[f'months_{idx}'][:], dtype=torch.float32)
        #years = torch.tensor(f[f'years_{idx}'][:], dtype=torch.float32)
        label = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]

        #im_aerial,label=random_augment_image_and_label(im_aerial,label)

        label = self.process_mask(label)
        attention_mask=torch.zeros(im_aerial.shape)

        new_resolution=0.2 #m/px
        label_segment=label.clone()
        
        label_segment=label_segment.repeat(im_aerial.shape[0],1,1)
        
        
        idxs_bandwidths = self.get_wavelengths_coordinates(im_aerial.shape)
        x_indices, y_indices = self.get_position_coordinates(im_aerial.shape, new_resolution,table=self.look_up.table)
        indices_queries = self.get_position_coordinates_queries(im_aerial.shape, new_resolution,table=self.look_up.table_queries)
        
        
        # Concatenate all token data
        image = torch.cat([
            im_aerial.unsqueeze(-1),      # Band values
            x_indices.float(),        # Global X indices
            y_indices.float(),        # Global Y indices  
            idxs_bandwidths.float(),   # Bandwidth indices
            label_segment.float().unsqueeze(-1),
            indices_queries.float()
        ], dim=-1)
        
        #at this point image shape is [5,512,512,6]
        # 5 -> channels
        # 512 512 -> H W
        # 5 -> meta data
        
        
        
        queries=image.clone()
        
        
        # Reshape and sample tokens
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries= einops.rearrange(queries,"b h w c -> (b h w) c")
        attention_mask = einops.rearrange(attention_mask, "c h w -> (c h w)")
        image= image[attention_mask==0.0]          # image get resized and invalid bands removed
        #queries= queries[attention_mask==0.0]    # same for mask

        #image,attention_mask = self.shuffle_arrays([image,attention_mask])
        queries= self.shuffle_arrays([queries])[0]
      
        nb_queries=self.config_model["trainer"]["max_tokens_reconstruction"]
        queries=queries[:nb_queries]
        queries_mask=torch.zeros(queries.shape[0])

        

      
        return image, attention_mask, queries,queries_mask,label
 
    def get_samples_to_viz(self, idx):
        label = None
        id_img = None
        


        # Ensure HDF5 file is open
        f = self._ensure_h5_open()

        

        im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5,512,512]
        image_to_return=im_aerial.clone()
        image_to_return=einops.rearrange(image_to_return,"c h w -> h w c")
        #im_sen = torch.tensor(f[f'img_sen_{idx}'][:], dtype=torch.float32)  # [12,10,40,40]
        #days = torch.tensor(f[f'days_{idx}'][:], dtype=torch.float32)
        #months = torch.tensor(f[f'months_{idx}'][:], dtype=torch.float32)
        #years = torch.tensor(f[f'years_{idx}'][:], dtype=torch.float32)
        label = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]
        label = self.process_mask(label)
        attention_mask=torch.zeros(im_aerial.shape)
        
        
        
        
        #sen_mask = f[f'sen_mask_{idx}'][:]
        #aerial_date = f[f'aerial_mtd_{idx}'].asstr()[()]





        

        #image, attention_mask, new_resolution = self.transform.apply_transformations(
        #    image, attention_mask, id_img, mode=self.mode, modality_mode=self.modality_mode, 
        #    f_s=self.fixed_size, f_r=self.fixed_resolution
        #)
        new_resolution=0.2 #m/px
        label_segment=label.clone()
        
        label_segment=label_segment.repeat(im_aerial.shape[0],1,1)
        
        
        idxs_bandwidths = self.get_wavelengths_coordinates(im_aerial.shape)
        x_indices, y_indices = self.get_position_coordinates(im_aerial.shape, new_resolution,table=self.look_up.table)
        indices_queries = self.get_position_coordinates_queries(im_aerial.shape, new_resolution,table=self.look_up.table_queries)
        
        # Concatenate all token data
        image = torch.cat([
            im_aerial.unsqueeze(-1),      # Band values
            x_indices.float(),        # Global X indices
            y_indices.float(),        # Global Y indices  
            idxs_bandwidths.float(),   # Bandwidth indices
            label_segment.float().unsqueeze(-1),
            indices_queries.float()
        ], dim=-1)
        
        #at this point image shape is [5,512,512,5]
        # 5 -> channels
        # 512 512 -> H W
        # 5 -> meta data
        
        
        queries=image.clone()

        
        # Reshape and sample tokens
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries= einops.rearrange(queries,"b h w c -> (b h w) c")
        attention_mask = einops.rearrange(attention_mask, "c h w -> (c h w)")

        image= image[attention_mask==0.0]          # image get resized and invalid bands removed


        
        
        
        
        
        
        
        # Shuffle tokens
        #image = self.shuffle_arrays([image])[0]
        
     
        queries_mask=torch.zeros(queries.shape[0])

        
        return image_to_return,image, attention_mask, queries,queries_mask,label

    def close(self):
        """Close HDF5 file if it's open."""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __del__(self):
        """Ensure HDF5 file is closed when dataset is deleted."""
        self.close()

class FLAIR_SEG(Dataset):
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

    def _initialize_file(self):
        """Initialize file and get number of samples."""
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 8  # Number of samples
            

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
    
    def get_position_coordinates_queries(self, image_shape, new_resolution,table=None):
        image_size = image_shape[-1]
        channels_size = image_shape[0]

        resolution_latents=0.2 #m
        tmp_resolution = int(resolution_latents*1000)
        global_offset = table[(tmp_resolution, image_size)]
        
        # Create LOCAL pixel indices (0 to image_size-1)
        indices = torch.full((image_size, image_size),global_offset)
        
        
        
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
    
    def process_mask(self, mask):
        mask = mask.float()
        mask[mask > 13] = 13
        mask = mask - 1
        return mask

    def save_img_aerial_to_pdf(im_aerial, filename="aerial.pdf", tmpdir="/tmp"):
        img = im_aerial[:3].cpu().numpy()
        img = img[[2,1,0], :, :]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1,2,0))

        tmp_path = os.path.join(tmpdir, "aerial_tmp.png")
        plt.imsave(tmp_path, img)

        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = [Image(tmp_path, width=500, height=500)]
        doc.build(story)

        os.remove(tmp_path)

    def __getitem__(self, idx):
        label = None
        id_img = None
        


        # Ensure HDF5 file is open
        f = self._ensure_h5_open()

        

        im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5,512,512]
        
        
        #im_sen = torch.tensor(f[f'img_sen_{idx}'][:], dtype=torch.float32)  # [12,10,40,40]
        #days = torch.tensor(f[f'days_{idx}'][:], dtype=torch.float32)
        #months = torch.tensor(f[f'months_{idx}'][:], dtype=torch.float32)
        #years = torch.tensor(f[f'years_{idx}'][:], dtype=torch.float32)
        label = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]

        #im_aerial,label=random_augment_image_and_label(im_aerial,label)

        label = self.process_mask(label)
        attention_mask=torch.zeros(im_aerial.shape)
        



        
        
        
        
        
        #sen_mask = f[f'sen_mask_{idx}'][:]
        #aerial_date = f[f'aerial_mtd_{idx}'].asstr()[()]





        

        #image, attention_mask, new_resolution = self.transform.apply_transformations(
        #    image, attention_mask, id_img, mode=self.mode, modality_mode=self.modality_mode, 
        #    f_s=self.fixed_size, f_r=self.fixed_resolution
        #)
        new_resolution=0.2 #m/px
        label_segment=label.clone()
        
        label_segment=label_segment.repeat(im_aerial.shape[0],1,1)
        
        
        idxs_bandwidths = self.get_wavelengths_coordinates(im_aerial.shape)
        x_indices, y_indices = self.get_position_coordinates(im_aerial.shape, new_resolution,table=self.look_up.table)
        indices_queries = self.get_position_coordinates_queries(im_aerial.shape, new_resolution,table=self.look_up.table_queries)
        
        
        # Concatenate all token data
        image = torch.cat([
            im_aerial.unsqueeze(-1),      # Band values
            x_indices.float(),        # Global X indices
            y_indices.float(),        # Global Y indices  
            idxs_bandwidths.float(),   # Bandwidth indices
            label_segment.float().unsqueeze(-1),
            indices_queries.float()
        ], dim=-1)
        
        #at this point image shape is [5,512,512,5]
        # 5 -> channels
        # 512 512 -> H W
        # 5 -> meta data
        
        
        queries=image[0].clone().unsqueeze(0)

        nb_queries=self.config_model["trainer"]["max_tokens_reconstruction"]
        queries=self.sample_query_tokens_optimal(
            queries,
            16,
            percent_patch = 0.7,
            nb_max_tokens=nb_queries,
        )
        

        print(queries)
        
        # Reshape and sample tokens
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        #queries= einops.rearrange(queries,"b h w c -> (b h w) c")
        attention_mask = einops.rearrange(attention_mask, "c h w -> (c h w)")
        image= image[attention_mask==0.0]          # image get resized and invalid bands removed
        #queries= queries[attention_mask==0.0]    # same for mask


        
        
        
        
        
        
        
        
        # Shuffle tokens
        image,attention_mask = self.shuffle_arrays([image,attention_mask])
        queries= self.shuffle_arrays([queries])[0]
        
    
        
        queries_mask=torch.zeros(queries.shape[0])
        return image, attention_mask, queries,queries_mask,label
 
    def get_samples_to_viz(self, idx):
        label = None
        id_img = None
        


        # Ensure HDF5 file is open
        f = self._ensure_h5_open()

        

        im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5,512,512]
        image_to_return=im_aerial.clone()
        image_to_return=einops.rearrange(image_to_return,"c h w -> h w c")
        #im_sen = torch.tensor(f[f'img_sen_{idx}'][:], dtype=torch.float32)  # [12,10,40,40]
        #days = torch.tensor(f[f'days_{idx}'][:], dtype=torch.float32)
        #months = torch.tensor(f[f'months_{idx}'][:], dtype=torch.float32)
        #years = torch.tensor(f[f'years_{idx}'][:], dtype=torch.float32)
        label = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]
        label = self.process_mask(label)
        attention_mask=torch.zeros(im_aerial.shape)
        
        
        
        
        #sen_mask = f[f'sen_mask_{idx}'][:]
        #aerial_date = f[f'aerial_mtd_{idx}'].asstr()[()]





        

        #image, attention_mask, new_resolution = self.transform.apply_transformations(
        #    image, attention_mask, id_img, mode=self.mode, modality_mode=self.modality_mode, 
        #    f_s=self.fixed_size, f_r=self.fixed_resolution
        #)
        new_resolution=0.2 #m/px
        label_segment=label.clone()
        
        label_segment=label_segment.repeat(im_aerial.shape[0],1,1)
        
        
        idxs_bandwidths = self.get_wavelengths_coordinates(im_aerial.shape)
        x_indices, y_indices = self.get_position_coordinates(im_aerial.shape, new_resolution,table=self.look_up.table)
        indices_queries = self.get_position_coordinates_queries(im_aerial.shape, new_resolution,table=self.look_up.table_queries)
        
        # Concatenate all token data
        image = torch.cat([
            im_aerial.unsqueeze(-1),      # Band values
            x_indices.float(),        # Global X indices
            y_indices.float(),        # Global Y indices  
            idxs_bandwidths.float(),   # Bandwidth indices
            label_segment.float().unsqueeze(-1),
            indices_queries.float()
        ], dim=-1)
        
        #at this point image shape is [5,512,512,5]
        # 5 -> channels
        # 512 512 -> H W
        # 5 -> meta data
        
        
        queries=image.clone()
        
        
        # Reshape and sample tokens
        image = einops.rearrange(image, "b h w c -> (b h w) c")
        queries= einops.rearrange(queries,"b h w c -> (b h w) c")
        attention_mask = einops.rearrange(attention_mask, "c h w -> (c h w)")

        image= image[attention_mask==0.0]          # image get resized and invalid bands removed


        
        
        
        
        
        
        
        # Shuffle tokens
        #image = self.shuffle_arrays([image])[0]
        
    
        
        queries_mask=torch.zeros(queries.shape[0])
        #
        return image_to_return,image, attention_mask, queries,queries_mask,label

    def close(self):
        """Close HDF5 file if it's open."""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __del__(self):
        """Ensure HDF5 file is closed when dataset is deleted."""
        self.close()