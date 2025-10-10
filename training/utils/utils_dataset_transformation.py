import torch
from .utils_dataset import read_yaml,save_yaml
from .image_utils import *
from .files_utils import*
from math import pi
import einops 
import datetime
import numpy as np
import datetime
from torchvision.transforms.functional import rotate, hflip, vflip
import random
import torch.nn as nn
import time
import math
from tqdm import tqdm
from torch.profiler import record_function



def fourier_encode(x, max_freq, num_bands = 4,low_pass=-1):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    if num_bands==-1:
        scales = torch.linspace(1., max_freq , max_freq, device = device, dtype = dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)

    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]

    x = x * scales * pi
    sin_x=x.sin()
    cos_x=x.cos()
    low_pass=-1
    if low_pass!=-1:
        sin_x[:,low_pass:]=0.0
        cos_x[:,low_pass:]=0.0
        
        
        
    x = torch.cat([sin_x, cos_x], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class transformations_config(nn.Module):

    

    def __init__(self,bands_yaml,config,lookup_table):
        super().__init__()
      
        self.bands_yaml=read_yaml(bands_yaml)
        self.bands_sen2_infos=self.bands_yaml["bands_sen2_info"]
        self.s2_waves=self.get_wavelengths_infos(self.bands_sen2_infos)
        self.s2_res_tmp=self.get_resolutions_infos(self.bands_sen2_infos)
        self.register_buffer("positional_encoding_s2", None)
        self.register_buffer('wavelength_processing_s2', None)
        #self.register_buffer("positional_encoding_fourrier", None)
        self.resolutions_x_sizes_cached={}
        
        self.config=config
        
        
        self.elevation_ = nn.Parameter(torch.empty(self._get_encoding_dim("wavelength") ))
        nn.init.trunc_normal_(self.elevation_, std=0.02, a=-2., b=2.)
        
        
        
        


        
        self.lookup_table=lookup_table
        

        self.register_buffer(
            "s2_res",
            torch.tensor(self.s2_res_tmp, dtype=torch.float32),
            persistent=True
        )
  

  
        self.nb_tokens_limit=config["trainer"]["max_tokens"]

        self.gaussian_means=[]
        self.gaussian_stds=[]

        if "wavelengths_encoding" in self.config:
            for gaussian_idx in self.config["wavelengths_encoding"]:
                self.gaussian_means.append(self.config["wavelengths_encoding"][gaussian_idx]["mean"])
                self.gaussian_stds.append(self.config["wavelengths_encoding"][gaussian_idx]["std"])
        
        self.gaussian_means=torch.Tensor(np.array(self.gaussian_means)).to(torch.float32).view(1, -1)
        self.gaussian_stds=torch.Tensor(np.array(self.gaussian_stds)).to(torch.float32).view(1, -1)
        
        
        
    def _get_encoding_dim(self, attribute):
        """Get encoding dimension for a specific attribute"""
        encoding_type = self.config["Atomiser"][f"{attribute}_encoding"]
        
        if encoding_type == "NOPE":
            return 0
        elif encoding_type == "NATURAL":
            return 1
        elif encoding_type == "FF":
            num_bands = self.config["Atomiser"][f"{attribute}_num_freq_bands"]
            max_freq = self.config["Atomiser"][f"{attribute}_max_freq"]
            if num_bands == -1:
                return int(max_freq) * 2 + 1
            else:
                return int(num_bands) * 2 + 1
        elif encoding_type == "GAUSSIANS":
            return len(self.config["wavelengths_encoding"])
        elif encoding_type == "GAUSSIANS_POS":
            return 420
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        

    def get_shape_attributes_config(self,attribute):
        if self.config["Atomiser"][attribute+"_encoding"]=="NOPE":
            return 0
        if self.config["Atomiser"][attribute+"_encoding"]=="NATURAL":
            return 1
        if self.config["Atomiser"][attribute+"_encoding"]=="FF":
            if self.config["Atomiser"][attribute+"_num_freq_bands"]==-1:
                return int(self.config["Atomiser"][attribute+"_max_freq"])*2+1
            else:
                return int(self.config["Atomiser"][attribute+"_num_freq_bands"])*2+1
        
        if self.config["Atomiser"][attribute+"_encoding"]=="GAUSSIANS":
            return int(len(self.config["wavelengths_encoding"].keys()))
        

    def get_wavelengths_infos(self,bands_info):
        bandwidth=[]
        central_w=[]
        for band_name in bands_info:
            band=bands_info[band_name]
            bandwidth.append(band["bandwidth"])
            central_w.append(band["central_wavelength"])

        return np.array(bandwidth),np.array(central_w)
    
    def get_resolutions_infos(self,bands_info):
        res=[]
        for band_name in bands_info:
            band=bands_info[band_name]
            res.append(band["resolution"])


        return torch.from_numpy(np.array([10 for _ in range(12)]))
   


        

        




    def get_band_identifier(self,bands,channel_idx):
        for band_key in bands:
            band=bands[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None

    
    def get_band_infos(self,bands,band_identifier):
        return bands[band_identifier]



    
    def pos_encoding(self,size,positional_scaling=None,max_freq=4,num_bands = 4,device="cpu",low_pass=None):


        axis = torch.linspace(-positional_scaling/2.0, positional_scaling/2.0, steps=size,device=device)
        pos=fourier_encode(axis,max_freq=max_freq,num_bands = num_bands,low_pass=low_pass)
        
        return pos
    
    def compute_adaptive_cutoff(self,resolution, image_size, base_norm=105.0, total_bands=256):
        """
        Formule pratique pour estimer le cutoff optimal
        """
        # Échelle physique de l'image
        physical_scale = resolution * image_size
        
        # Facteur d'échelle par rapport à la normalisation de base
        scale_factor = physical_scale / base_norm
        
        # Cutoff basé sur une décroissance en racine carrée
        # Empiriquement bon compromis entre Nyquist et considérations pratiques
        
        if scale_factor <= 1.0:
            cutoff_ratio = 1.0
        else:
            cutoff_ratio = 1.0 / np.sqrt(scale_factor)
        
        # Assurer un minimum de fréquences (au moins 10% pour capturer structure de base)
        cutoff_ratio = max(0.1, cutoff_ratio)
        
        return int(total_bands * cutoff_ratio)



    def get_positional_encoding_fourrier(
        self,
        size,             # e.g. (B_size, T_size, H, W, C)
        resolution: float,      # shape [C], base resolution per band
        device,
        sampling=None
        ):
        """
        Returns: Tensor [B_size, T_size, H, W, C, D]
        resolution: [C]
        resolution_factor: [B_size]
        """

        # spatial size (assume H == W)

        if sampling is None:
            sampling=size
        # -- 1) compute per-sample, per-band new resolution: [B, C]
        #    new_res[i, b] = resolution[b] / resolution_factor[i]
        

        # -- 2) compute positional scaling per band: [B, C]
        
        pos_scalings = (size * resolution) / 100
        
  
        max_freq  = self.config["Atomiser"]["pos_max_freq"]
        num_bands = self.config["Atomiser"]["pos_num_freq_bands"]
        #cutoff = self.compute_adaptive_cutoff(resolution=resolution,image_size=size,base_norm=100 ,total_bands= num_bands)      # ≈ 20 bands (63%)
        cutoff = -1
        
        
        
        
        

        
        
        raw = self.pos_encoding(
            size=sampling,
            positional_scaling=pos_scalings,
            max_freq=max_freq,
            num_bands=num_bands,
            device=device,
            low_pass=cutoff
        )
        
        
  
        
        
    
        
        return raw

    def get_gaussian_encoding(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        num_gaussians: int,        
        sigma: float,
        device=None,
        extremums=None
    ):
        """
        Compute Gaussian encoding for tokens with explicit position information.
        
        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens, 2 * num_gaussians]
        """
        device = device or token_data.device
        batch, tokens, _ = token_data.shape
        
        # Create cache key based on encoding parameters only
        cache_key = f"positional_encoding_{num_gaussians}_{sigma}_{device}"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_gaussian_encodings(num_gaussians, sigma, device, cache_key,extremums=extremums)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_x_indices = token_data[..., 1].long()  # [batch, tokens]
        global_y_indices = token_data[..., 2].long()  # [batch, tokens]
        
        # Direct lookup using global indices
        encoding_x = cached_encoding[global_x_indices]  # [batch, tokens, num_gaussians]
        encoding_y = cached_encoding[global_y_indices]  # [batch, tokens, num_gaussians]
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_x, encoding_y], dim=-1)  # [batch, tokens, 2*num_gaussians]
        
        return result
    
    def get_gaussian_encoding_latents(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        num_gaussians: int,        
        sigma: float,
        device=None,
        extremums=None
    ):
        """
        Compute Gaussian encoding for tokens with explicit position information.
        
        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens, 2 * num_gaussians]
        """
        device = device or token_data.device
        batch, tokens, _ = token_data.shape
        
        # Create cache key based on encoding parameters only
        cache_key = f"positional_encoding_{num_gaussians}_{sigma}_{device}_latents"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_gaussian_encodings(num_gaussians, sigma, device, cache_key,extremums=extremums,latents=True)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_indices = token_data[:,0, 5].long()  # [batch, tokens]
        
        
        
        global_indices=global_indices.unsqueeze(1) + torch.arange(0,self.lookup_table.nb_tokens_queries,device=device)
        encoding=cached_encoding[global_indices]
        
        B, C, E = encoding.shape
        idx = torch.arange(C, device=device)
        i, j = torch.meshgrid(idx, idx, indexing="ij")  # each [C, C]
        i = i.reshape(-1)  # [C*C]
        j = j.reshape(-1)  # [C*C]
        encoding_i = encoding[:, i, :]  # [B, C*C, E]  -> first index of each pair
        encoding_j = encoding[:, j, :]  # [B, C*C, E]  -> second index of each pair
        
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_i,encoding_j], dim=-1)  # [batch, tokens, 2*num_gaussians]
        
        return result


    
        
    def get_wavelength_encoding(self, token_data: torch.Tensor, device=None):
        cache_key = f"wavelength_encoding_gaussian"
        if not hasattr(self, cache_key):
            self._precompute_global_wavelength_encodings(device)

        cached_encoding = getattr(self, cache_key)              # [num_ids, 19] (no grad)
        global_x_indices = token_data.long()                    # [B, T]
        encoding_wavelengths = cached_encoding[global_x_indices]# [B, T, 19]

        elevation_key = (-1, -1)
        if elevation_key in self.lookup_table.table_wave:
            elevation_idx = self.lookup_table.table_wave[elevation_key]
            elevation_mask = (global_x_indices == elevation_idx)        # [B, T] bool
            
            elev_vec = self.elevation_.view(1, 1, -1).expand_as(encoding_wavelengths)
            encoding_wavelengths = torch.where(elevation_mask.unsqueeze(-1), elev_vec, encoding_wavelengths)

        return encoding_wavelengths
        
    def get_fourrier_encoding(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        device=None
    ):
        """
        Compute Fourier encoding for tokens with explicit position information.

        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens, 2 * num_gaussians]
        """
        
        # Create cache key based on encoding parameters only
        cache_key = f"positional_encoding_fourrier"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_fourrier_encodings(device)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_x_indices = token_data[..., 1].long()  # [batch, tokens]
        global_y_indices = token_data[..., 2].long()  # [batch, tokens]
        
        # Direct lookup using global indices
        encoding_x = cached_encoding[global_x_indices]  # [batch, tokens, num_gaussians]
        encoding_y = cached_encoding[global_y_indices]  # [batch, tokens, num_gaussians]
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_x, encoding_y], dim=-1)  # [batch, tokens, 2*num_gaussians]
        
        
        return result
    
    def get_bias_tokens_encoding(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        device=None
    ):
        """
        Compute bias encoding for tokens with explicit position information.

        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens,2,min_coodinate,max_coordinate]
            #there is coordinates for each axis
            #the function get_bias_latents_encoding return the parameters of the gaussians
        """
        
        # Create cache key based on encoding parameters only
        cache_key = f"bias_tokens_encoding"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_bias_tokens_encodings(device)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_x_indices = token_data[..., 1].long()  # [batch, tokens]
        global_y_indices = token_data[..., 2].long()  # [batch, tokens]
        
        # Direct lookup using global indices
        encoding_x = cached_encoding[global_x_indices].unsqueeze(-2)  # [batch, tokens, num_gaussians]
        encoding_y = cached_encoding[global_y_indices].unsqueeze(-2)  # [batch, tokens, num_gaussians]

        
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_x, encoding_y], dim=-2)  # [batch, tokens, 2*num_gaussians]
  
        return result
    

    def get_bias_latents_encoding(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        device=None
    ):
        """
        Compute bias encoding for tokens with explicit position information.

        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens,2,min_coodinate,max_coordinate]
            #there is coordinates for each axis
            #the function get_bias_latents_encoding return the parameters of the gaussians
        """
        
        # Create cache key based on encoding parameters only
        cache_key = f"bias_latents_encoding"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_bias_latents_encodings(device)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        global_indices = token_data[:,0, 5].long()  # [batch, tokens]
        
        
        global_indices=global_indices.unsqueeze(1) + torch.arange(0,self.lookup_table.nb_tokens_queries,device=device)
        encoding=cached_encoding[global_indices]
        
        B, C, E = encoding.shape
        idx = torch.arange(C, device=device)
        i, j = torch.meshgrid(idx, idx, indexing="ij")  # each [C, C]
        i = i.reshape(-1)  # [C*C]
        j = j.reshape(-1)  # [C*C]
        encoding_i = encoding[:, i, :].unsqueeze(-2)  # [B, C*C, E]  -> first index of each pair
        encoding_j = encoding[:, j, :].unsqueeze(-2)  # [B, C*C, E]  -> second index of each pair
        
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_i,encoding_j], dim=-2)  # [batch, tokens, 2*num_gaussians]
        
        return result
    
    def get_fourrier_encoding_queries(
        self,
        token_data: torch.Tensor,  # [batch, tokens, data] 
        device=None
    ):
        """
        Compute Fourier encoding for tokens with explicit position information.

        Args:
            token_data: [batch, tokens, data] where:
                - data[..., 1]: global x-axis pixel index 
                - data[..., 2]: global y-axis pixel index
                
        Returns:
            responses: [batch, tokens, 2 * num_gaussians]
        """
        
        # Create cache key based on encoding parameters only
        cache_key = f"positional_encoding_fourrier_queries"
        
        # Check if we have cached encodings
        if not hasattr(self, cache_key):
            self._precompute_global_fourrier_encodings(device,queries=True)
        
        cached_encoding = getattr(self, cache_key)  # [total_positions, num_gaussians]
        
        # Extract global pixel indices from token_data
        global_indices = token_data[:,0, 5].long()  # [batch, tokens]
        
        
        
        global_indices=global_indices.unsqueeze(1) + torch.arange(0,self.lookup_table.nb_tokens_queries,device=device)
        encoding=cached_encoding[global_indices]
        
        B, C, E = encoding.shape
        idx = torch.arange(C, device=device)
        i, j = torch.meshgrid(idx, idx, indexing="ij")  # each [C, C]
        i = i.reshape(-1)  # [C*C]
        j = j.reshape(-1)  # [C*C]
        encoding_i = encoding[:, i, :]  # [B, C*C, E]  -> first index of each pair
        encoding_j = encoding[:, j, :]  # [B, C*C, E]  -> second index of each pair
        
        
        # Concatenate x and y encodings
        result = torch.cat([encoding_i,encoding_j], dim=-1)  # [batch, tokens, 2*num_gaussians]
        
        return result
    
    
    def _precompute_global_gaussian_encodings(self, num_gaussians, sigma, device, cache_key,extremums=1200,latents=False):
        """
        Precompute Gaussian encodings for ALL possible pixel positions across all modalities.
        This creates a single global lookup table.
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())

        if latents:
            max_global_index = len(self.lookup_table.table_queries.keys())*self.lookup_table.nb_tokens_queries
        
        # Create Gaussian centers (same for all positions)
        
        centers = torch.linspace(-extremums, extremums, num_gaussians, device=device)
        
        # Initialize global encoding tensor
        global_encoding = torch.zeros(max_global_index, num_gaussians, device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            # Get global offset for this modality
            # self.lookup_table.nb_tokens_queries
            
            
            
            modality_key = (int(1000 * resolution), image_size)

            global_offset = self.lookup_table.table[modality_key]
            if latents:
                global_offset = self.lookup_table.table_queries[modality_key]
                image_size=self.lookup_table.nb_tokens_queries
                physical_coords = torch.linspace(
                    (-extremums/2.) * resolution, 
                    (extremums/2.) * resolution, 
                    steps=image_size, 
                    device=device
                )
            else:

            # Create physical coordinates for this modality's pixels
                physical_coords = torch.linspace(
                    (-image_size/2.) * resolution, 
                    (image_size/2.) * resolution, 
                    steps=image_size, 
                    device=device
                )
            
            # Compute Gaussian encoding for this modality
            modality_encoding = self._compute_1d_gaussian_encoding_vectorized(
                physical_coords, resolution/2.0, centers, sigma, device
            )  # [image_size, num_gaussians]
            
            # Place encodings at correct global indices
            global_encoding[global_offset:global_offset + image_size] = modality_encoding
        
        # Store the global encoding
        setattr(self, cache_key, global_encoding)

        
        
    def _precompute_global_fourrier_encodings(self, device,queries=False):
        """
        Precompute fourrier encodings for ALL possible pixel positions across all modalities.
        This creates a single global lookup table.
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())

        if queries:
            max_global_index = len(self.lookup_table.table_queries.keys())*self.lookup_table.nb_tokens_queries
        
        # Initialize global encoding tensor
        num_bands = self.config["Atomiser"]["pos_num_freq_bands"]* 2 + 1
        global_encoding = torch.zeros(max_global_index, num_bands, device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            # Get global offset for this modality
            if queries:
                pos=self.get_positional_encoding_fourrier(image_size,resolution,device,sampling=self.lookup_table.nb_tokens_queries)
            else:
                pos=self.get_positional_encoding_fourrier(image_size,resolution,device)
            
            
            
            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            if queries:
                global_offset = self.lookup_table.table_queries[modality_key]

            
            
            if queries:
                global_encoding[global_offset:global_offset + self.lookup_table.nb_tokens_queries] = pos
            else:
            # Place encodings at correct global indices
                global_encoding[global_offset:global_offset + image_size] = pos
        
        # Store the global encoding
        cache_key = f"positional_encoding_fourrier"
        if queries:
            cache_key = f"positional_encoding_fourrier_queries"
        setattr(self, cache_key, global_encoding)

    def _precompute_global_bias_tokens_encodings(self, device):
        """
        return min_max
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        global_encoding = torch.zeros(max_global_index, 2 , device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            pos_scalings = (image_size * resolution)
            axis = torch.linspace(-pos_scalings/2.0, pos_scalings/2.0, steps=image_size,device=device)
            
            axis_left=axis.clone()
            axis_right=axis.clone()
            axis_left=(axis_left-resolution/2.0).unsqueeze(-1)
            axis_right=(axis_right+resolution/2.0).unsqueeze(-1)

            axis=torch.cat([axis_left,axis_right],dim=-1)
            
            

            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            
            global_encoding[global_offset:global_offset + image_size] = axis
            
        
        # Store the global encoding
        cache_key = f"bias_tokens_encoding"
        
        setattr(self, cache_key, global_encoding)

    def _precompute_global_bias_latents_encodings(self, device):
        """
        return parameters for gaussians
        """
        
        # Get total number of positions from lookup table
        max_global_index = sum(20 for _ in self.lookup_table.table_queries.keys())
        global_encoding = torch.zeros(max_global_index, 2 , device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Gaussian encodings"):
            resolution, image_size = modality
            
            pos_scalings = (image_size * resolution)
            #centers of gaussians
            axis = torch.linspace(-pos_scalings/2.0, pos_scalings/2.0, steps=20,device=device).unsqueeze(-1)
            centers = torch.full((20,1),resolution*20,device=device)
            
            axis=torch.cat([axis,centers],dim=-1)

            

            modality_key = (int(1000 * resolution), image_size)

            
            global_offset = self.lookup_table.table_queries[modality_key]
     
            global_encoding[global_offset:global_offset + 20] = axis
            
        
        # Store the global encoding
        cache_key = f"bias_latents_encoding"
        
        setattr(self, cache_key, global_encoding)
        
    def _precompute_global_wavelength_encodings(self, device):
        """
        Precompute fourrier encodings for ALL possible pixel positions across all modalities.
        This creates a single global lookup table.
        """
        
        # Get total number of positions from lookup table
        max_global_index =len(self.lookup_table.table_wave.keys())
        global_encoding = torch.zeros(max_global_index, 19, device=device)
        
        # Compute encodings for each modality and place them at correct global indices
        for modality in tqdm(self.lookup_table.table_wave.keys(), desc="Precomputing Wavelength encodings"):
            bandwidth, central_wavelength = modality
            id_modality= self.lookup_table.table_wave[(bandwidth, central_wavelength)]
            
            #defined in the bands info config file
            encoded=None
            if bandwidth==-1 and central_wavelength==-1:
                continue
                
            
            if encoded==None:
                encoded=self.compute_gaussian_band_max_encoding([central_wavelength],[bandwidth], num_points=150).squeeze(0).squeeze(0)
            
            global_encoding[id_modality] = encoded
        
        # Store the global encoding
        cache_key = f"wavelength_encoding_gaussian"
        setattr(self, cache_key, global_encoding)
        
        

    def _compute_1d_gaussian_encoding_vectorized(self, positions, half_res, centers, sigma, device):
        """
        Vectorized computation of 1D Gaussian encoding.
        
        Args:
            positions: [size] - physical positions for each pixel
            half_res: float - half resolution
            centers: [num_gaussians] - Gaussian centers
            sigma: float - Gaussian sigma
            
        Returns:
            encoding: [size, num_gaussians]
        """
        sqrt_2 = math.sqrt(2)
        norm_factor = sigma * math.sqrt(math.pi / 2)
        inv_sqrt2_sigma = 1.0 / (sqrt_2 * sigma)
        
        # Expand dimensions for vectorized computation
        positions_exp = positions.unsqueeze(-1)  # [size, 1]
        centers_exp = centers.unsqueeze(0)       # [1, num_gaussians]
        
        # Compute pixel bounds
        lower_bound = positions_exp - half_res   # [size, 1]
        upper_bound = positions_exp + half_res   # [size, 1]
        
        # Vectorized computation of error function bounds
        lower_erf_input = (lower_bound - centers_exp) * inv_sqrt2_sigma  # [size, num_gaussians]
        upper_erf_input = (upper_bound - centers_exp) * inv_sqrt2_sigma  # [size, num_gaussians]
        
        # Compute integral using error function
        encoding = norm_factor * (torch.erf(upper_erf_input) - torch.erf(lower_erf_input))
        
        # L2 normalize
        encoding = encoding / (encoding.norm(dim=-1, keepdim=True) + 1e-8)
        
        return encoding  # [size, num_gaussians]

    def fourier_encode_scalar(self,scalar,size,max_freq,num_bands):
        
        tmp_encoding=fourier_encode(torch.Tensor(scalar), max_freq=max_freq, num_bands = num_bands) # B T C
        
        tmp_encoding = tmp_encoding.unsqueeze(2).unsqueeze(2)  # [B, T, 1, 1, C]
        tmp_encoding = tmp_encoding.expand(-1,    # keep B
                                   -1,    # keep T
                                    size, # H = size
                                    size, # W = size
                                   -1)    # keep C
  
        return tmp_encoding
    

    def compute_gaussian_band_max_encoding(self, lambda_centers, bandwidths, num_points=50,modality="S2"):

        
        # Ensure inputs are PyTorch tensors and on the same device as self.gaussian_means.
        device = self.gaussian_means.device

        
        lambda_centers = torch.as_tensor(lambda_centers, dtype=torch.float32, device=device)
        bandwidths = torch.as_tensor(bandwidths, dtype=torch.float32, device=device)

        
        lambda_min = lambda_centers - bandwidths / 2 
        lambda_max = lambda_centers + bandwidths / 2 
        
        
        t = torch.linspace(0, 1, num_points, device=device)  
        
        # Compute the sampled wavelengths for each spectral band using broadcasting.
        # Each spectral band gets its own set of sample points.
        # sampled_lambdas shape: [s, num_points]

        sampled_lambdas = lambda_min.unsqueeze(1) + (lambda_max - lambda_min).unsqueeze(1) * t.unsqueeze(0)

        gaussians = torch.exp(
            -0.5 * (
                ((sampled_lambdas.unsqueeze(2) - self.gaussian_means.unsqueeze(0).unsqueeze(0)) / 
                self.gaussian_stds.unsqueeze(0).unsqueeze(0)) ** 2
            )
        )
        
        # For each spectral band and each Gaussian, find the maximum activation across the sampled points.
        # The max is taken along dim=1 (the num_points axis), returning a tensor of shape [s, num_gaussians].
        
        encoding = gaussians.max(dim=-2).values
       

        
        return encoding


    
    def get_bvalue_processing(self,img):
        if self.config["Atomiser"]["bandvalue_encoding"]=="NATURAL":
                return img.unsqueeze(-1)
        
        elif self.config["Atomiser"]["bandvalue_encoding"]=="FF":
            num_bands=self.config["Atomiser"]["bandvalue_num_freq_bands"]
            max_freq=self.config["Atomiser"]["bandvalue_max_freq"]
            fourier_encoded_values=fourier_encode(img, max_freq, num_bands)
            
            return fourier_encoded_values

    def wavelength_processing(self,device,wavelength,bandwidth,modality="s2",tokens=None):

        id_cache="wavelength_processing_s2"
        encoded = getattr(self, id_cache)

        if encoded is None:
            encoded=self.compute_gaussian_band_max_encoding([wavelength], [bandwidth], num_points=50)
            
            tmp_wavelength=torch.from_numpy(wavelength).clone().unsqueeze(-1)
            tmp_bandwidth =torch.from_numpy(bandwidth).clone().unsqueeze(-1)
            encoded=encoded.squeeze(0)

            indexing_array=torch.cat([tmp_bandwidth,tmp_wavelength,encoded],dim=-1)

            
            encoded=indexing_array.to(device)
            id_cache="wavelength_processing_s2"
            setattr(self,id_cache, encoded)
        
      

        # tokens: [B, N, 5]
        # encoded: [M, D] (here D=21)
        B, N, _ = tokens.shape
        M = encoded.shape[0]
        D = encoded.shape[1] - 2

        res = torch.zeros(B, N, D, device=tokens.device)

        # Extract wavelength and bandwidth from tokens
        token_wl = tokens[:, :, 4].unsqueeze(-1)  # [B, N, 1]
        token_bw = tokens[:, :, 3].unsqueeze(-1)  # [B, N, 1]

        # Compare with encoded wavelength and bandwidth
        encoded_wl = encoded[:, 0]  # [M]
        encoded_bw = encoded[:, 1]  # [M]

        # Build a matching mask [B, N, M]
        mask = (token_wl == encoded_wl) & (token_bw == encoded_bw)

        # For each token, get the index of the matching row
        idx = mask.float().argmax(dim=-1)  # [B, N] (index of the match)

        # Gather the encoding values
        res = encoded[idx, 2:]  # Will broadcast to [B, N, D]
        

        
            
        
        return res
    


    def apply_transformations_optique(self, im_sen, mask_sen, mode,query=False):
        
        if query:
            #value_processed = self.get_bvalue_processing(im_sen[:,:,0])
            central_wavelength_processing = self.get_wavelength_encoding(im_sen[:,:,3],device=im_sen.device)
            p_x=self.get_fourrier_encoding(im_sen,device=im_sen.device)
            #band_post_proc_2=self.get_gaussian_encoding(im_sen,32,10.0, im_sen.device,extremums=52.5)
            #band_post_proc_3=self.get_gaussian_encoding(im_sen,64,3.0, im_sen.device,extremums=52.5)
            #band_post_proc_4=self.get_gaussian_encoding(im_sen,73,1.0, im_sen.device)
            #band_post_proc_4=self.get_gaussian_encoding(im_sen,300,0.05, im_sen.device,extremums=52.5)
            #band_post_proc_0=self.get_gaussian_encoding(im_sen,8,100, im_sen.device,extremums=600)
            #band_post_proc_1=self.get_gaussian_encoding(im_sen,16,40.0, im_sen.device,extremums=600)
            #band_post_proc_2=self.get_gaussian_encoding(im_sen,32,10.0, im_sen.device,extremums=300)
            #band_post_proc_3=self.get_gaussian_encoding(im_sen,64,3.0, im_sen.device,extremums=150)
            #band_post_proc_4=self.get_gaussian_encoding(im_sen,300,0.2, im_sen.device,extremums=150)

     
            tokens = torch.cat([
                #value_processed,
                central_wavelength_processing,
                p_x,
                #band_post_proc_2,
                #band_post_proc_3,
                #band_post_proc_4,
            ], dim=-1)
            
            return tokens, mask_sen
        
        # 2) Wavelength encoding
        central_wavelength_processing = self.get_wavelength_encoding(im_sen[:,:,3],device=im_sen.device)
        # 3) Band‑value encoding
        value_processed = self.get_bvalue_processing(im_sen[:,:,0])
        tokens_bias=self.get_bias_tokens_encoding(im_sen,device=im_sen.device)
        latents_bias=self.get_bias_latents_encoding(im_sen,device=im_sen.device)
        p_x=self.get_fourrier_encoding(im_sen,device=im_sen.device)
        p_latents=self.get_fourrier_encoding_queries(im_sen,device=im_sen.device)

        #p_latents_2=self.get_gaussian_encoding_latents(im_sen,32,10.0, im_sen.device,extremums=52.5)
        #p_latents_3=self.get_gaussian_encoding_latents(im_sen,64,3.0, im_sen.device,extremums=52.5)
        #p_latents_4=self.get_gaussian_encoding_latents(im_sen,300,0.05, im_sen.device,extremums=52.5)

        #p_latents=torch.cat([p_latents_2,p_latents_3,p_latents_4],dim=-1)
        
        
        #band_post_proc_0=self.get_gaussian_encoding(im_sen,8,100, im_sen.device,extremums=600)
        #band_post_proc_1=self.get_gaussian_encoding(im_sen,16,40.0, im_sen.device,extremums=600)
        #band_post_proc_2=self.get_gaussian_encoding(im_sen,32,10.0, im_sen.device,extremums=52.5)
        #band_post_proc_3=self.get_gaussian_encoding(im_sen,64,3.0, im_sen.device,extremums=52.5)
        #band_post_proc_4=self.get_gaussian_encoding(im_sen,73,1.0, im_sen.device)
        #band_post_proc_4=self.get_gaussian_encoding(im_sen,300,0.05, im_sen.device,extremums=52.5)
        

        #with record_function("Atomizer/process_data/get_tokens/cat"):
        
        tokens = torch.cat([
            value_processed,
            central_wavelength_processing,
            p_x
            #band_post_proc_0,
            #band_post_proc_1,
            #band_post_proc_2,
            #band_post_proc_3,
            #band_post_proc_4,
        ], dim=-1)

        


        return tokens, mask_sen, p_latents,(tokens_bias,latents_bias)
    
    
    def get_tokens(self,img,mask,mode="optique",modality="s2",wave_encoding=None,query=False):
        
  

        if mode=="optique":
            return self.apply_transformations_optique(img,mask,modality,query=query)
        

    def process_data(self,img,mask,query=False):
        
        L_tokens=[]
        L_masks=[]

        
        
        if self.config["dataset"]["S2"]:
            #with record_function("Atomizer/process_data/apply_temporal_spatial_transforms"):
            #    tmp_img,tmp_mask=self.apply_temporal_spatial_transforms(img, mask)
            
            #with record_function("Atomizer/process_data/get_tokens"):
            if query:
                tokens_s2,tokens_mask_s2=self.get_tokens(img,mask,mode="optique",modality="s2",query=query)
                return tokens_s2,tokens_mask_s2
            else:
                tokens_s2,tokens_mask_s2,latents,bias=self.get_tokens(img,mask,mode="optique",modality="s2",query=query)
                return tokens_s2,tokens_mask_s2,latents,bias



  



