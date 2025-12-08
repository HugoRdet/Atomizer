import torch
from .utils_dataset import read_yaml, save_yaml
from .image_utils import *
from .files_utils import *
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


def fourier_encode(x, max_freq, num_bands=4, low_pass=-1):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    if num_bands == -1:
        scales = torch.linspace(1., max_freq, max_freq, device=device, dtype=dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)

    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    sin_x = x.sin()
    cos_x = x.cos()
    low_pass = -1
    if low_pass != -1:
        sin_x[:, low_pass:] = 0.0
        cos_x[:, low_pass:] = 0.0

    x = torch.cat([sin_x, cos_x], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class transformations_config(nn.Module):

    def __init__(self, bands_yaml, config, lookup_table):
        super().__init__()

        self.bands_yaml = read_yaml(bands_yaml)
        self.bands_sen2_infos = self.bands_yaml["bands_sen2_info"]
        self.s2_waves = self.get_wavelengths_infos(self.bands_sen2_infos)
        self.s2_res_tmp = self.get_resolutions_infos(self.bands_sen2_infos)
        self.register_buffer("positional_encoding_s2", None)
        self.register_buffer('wavelength_processing_s2', None)
        self.resolutions_x_sizes_cached = {}

        self.config = config

        self.elevation_ = nn.Parameter(torch.empty(self._get_encoding_dim("wavelength")))
        nn.init.trunc_normal_(self.elevation_, std=0.02, a=-2., b=2.)

        self.lookup_table = lookup_table

        self.register_buffer(
            "s2_res",
            torch.tensor(self.s2_res_tmp, dtype=torch.float32),
            persistent=True
        )

        self.nb_tokens_limit = config["trainer"]["max_tokens"]

        self.gaussian_means = []
        self.gaussian_stds = []

        self.spatial_latents_per_row = config["Atomiser"]["spatial_latents"]  # e.g., 20

        if "wavelengths_encoding" in self.config:
            for gaussian_idx in self.config["wavelengths_encoding"]:
                self.gaussian_means.append(self.config["wavelengths_encoding"][gaussian_idx]["mean"])
                self.gaussian_stds.append(self.config["wavelengths_encoding"][gaussian_idx]["std"])

        self.gaussian_means = torch.Tensor(np.array(self.gaussian_means)).to(torch.float32).view(1, -1)
        self.gaussian_stds = torch.Tensor(np.array(self.gaussian_stds)).to(torch.float32).view(1, -1)

        # =====================================================================
        # POLAR POSITIONAL ENCODING PARAMETERS
        # =====================================================================
        self.polar_num_r_bands = config["Atomiser"].get("polar_num_r_bands", 8)
        self.polar_num_theta_bands = config["Atomiser"].get("polar_num_theta_bands", 8)
        self.G_ref = config["Atomiser"].get("G_ref", 50.0)  # Reference GSD in meters

        # Precompute frequencies (constant, tiny memory)
        self.register_buffer(
            "polar_r_frequencies",
            2.0 ** torch.arange(self.polar_num_r_bands, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "polar_theta_frequencies",
            torch.arange(1, self.polar_num_theta_bands + 1, dtype=torch.float32),
            persistent=False
        )

        # Number of latents (assuming square grid)
        # Number of latents (square grid: 20×20 = 400)
        self.sqrt_num_latents = lookup_table.nb_tokens_queries  # 20
        self.num_latents = self.sqrt_num_latents ** 2  # 400

    # =========================================================================
    # EXISTING HELPER METHODS (unchanged)
    # =========================================================================

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

    def get_shape_attributes_config(self, attribute):
        if self.config["Atomiser"][attribute + "_encoding"] == "NOPE":
            return 0
        if self.config["Atomiser"][attribute + "_encoding"] == "NATURAL":
            return 1
        if self.config["Atomiser"][attribute + "_encoding"] == "FF":
            if self.config["Atomiser"][attribute + "_num_freq_bands"] == -1:
                return int(self.config["Atomiser"][attribute + "_max_freq"]) * 2 + 1
            else:
                return int(self.config["Atomiser"][attribute + "_num_freq_bands"]) * 2 + 1

        if self.config["Atomiser"][attribute + "_encoding"] == "GAUSSIANS":
            return int(len(self.config["wavelengths_encoding"].keys()))

    def get_wavelengths_infos(self, bands_info):
        bandwidth = []
        central_w = []
        for band_name in bands_info:
            band = bands_info[band_name]
            bandwidth.append(band["bandwidth"])
            central_w.append(band["central_wavelength"])

        return np.array(bandwidth), np.array(central_w)

    def get_resolutions_infos(self, bands_info):
        res = []
        for band_name in bands_info:
            band = bands_info[band_name]
            res.append(band["resolution"])

        return torch.from_numpy(np.array([10 for _ in range(12)]))

    def get_band_identifier(self, bands, channel_idx):
        for band_key in bands:
            band = bands[band_key]

            if band["idx"] == channel_idx:
                return band_key
        return None

    def get_band_infos(self, bands, band_identifier):
        return bands[band_identifier]

    def pos_encoding(self, size, positional_scaling=None, max_freq=4, num_bands=4, device="cpu", low_pass=None):

        axis = torch.linspace(-positional_scaling / 2.0, positional_scaling / 2.0, steps=size, device=device)
        pos = fourier_encode(axis, max_freq=max_freq, num_bands=num_bands, low_pass=low_pass)

        return pos

    def compute_adaptive_cutoff(self, resolution, image_size, base_norm=105.0, total_bands=256):
        physical_scale = resolution * image_size
        scale_factor = physical_scale / base_norm

        if scale_factor <= 1.0:
            cutoff_ratio = 1.0
        else:
            cutoff_ratio = 1.0 / np.sqrt(scale_factor)

        cutoff_ratio = max(0.1, cutoff_ratio)

        return int(total_bands * cutoff_ratio)

    def get_positional_encoding_fourrier(self, size, resolution: float, device, sampling=None):

        if sampling is None:
            sampling = size

        pos_scalings = (size * resolution) / 100

        max_freq = self.config["Atomiser"]["pos_max_freq"]
        num_bands = self.config["Atomiser"]["pos_num_freq_bands"]
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

    # =========================================================================
    # POLAR POSITIONAL ENCODING - NEW METHODS
    # =========================================================================

    def _precompute_token_physical_centers(self, device):
        """
        Precompute physical CENTER coordinate for each position index.
        
        Similar structure to _precompute_global_fourrier_encodings.
        
        Lookup: position_index → physical_center_coordinate
        """
        cache_key = "token_physical_centers"
        if hasattr(self, cache_key) and getattr(self, cache_key) is not None:
            cached = getattr(self, cache_key)
            if cached.device == device:
                return cached
            else:
                return cached.to(device)

        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        global_centers = torch.zeros(max_global_index, device=device)

        # Compute centers for each modality
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing token physical centers"):
            resolution, image_size = modality

            # Physical extent of the image
            pos_scaling = image_size * resolution

            # Center coordinates for each pixel position
            # First pixel center: -pos_scaling/2 + resolution/2
            # Last pixel center: pos_scaling/2 - resolution/2
            centers = torch.linspace(
                -pos_scaling / 2.0 + resolution / 2.0,
                pos_scaling / 2.0 - resolution / 2.0,
                steps=image_size,
                device=device
            )

            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            global_centers[global_offset:global_offset + image_size] = centers

        setattr(self, cache_key, global_centers)
        return global_centers
    
    def _get_gsd_for_tokens(
        self,
        token_data: torch.Tensor,  # [B, L, m, 6] or [B, N, 6]
        device=None
    ) -> torch.Tensor:
        """
        Get GSD for each token via lookup table.
        
        Uses the same position index as physical centers (column 1 or 2).
        All positions within a modality share the same GSD.
        
        Args:
            token_data: Token tensor with position indices
            
        Returns:
            gsd: GSD in meters, same shape as token_data without last dim
        """
        device = device or token_data.device
        
        # Precompute lookup table
        gsd_lookup = self._precompute_token_gsd(device)
        
        # Use x position index (column 1) to look up GSD
        # (column 1 and 2 will give same GSD since it's per-modality)
        x_indices = token_data[..., 1].long()  # [B, L, m] or [B, N]
        
        gsd = gsd_lookup[x_indices]  # Same shape as x_indices
        
        return gsd

    def _precompute_token_gsd(self, device):
        """
        Precompute GSD (ground sampling distance) for each position index.
        
        Similar structure to _precompute_token_physical_centers.
        
        Lookup: position_index → GSD in meters
        """
        cache_key = "token_gsd_lookup"
        if hasattr(self, cache_key) and getattr(self, cache_key) is not None:
            cached = getattr(self, cache_key)
            if cached.device == device:
                return cached
            else:
                return cached.to(device)

        # Get total number of positions from lookup table
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        global_gsd = torch.zeros(max_global_index, device=device)

        # Compute GSD for each modality
        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing token GSD"):
            resolution, image_size = modality

            # All positions in this modality have the same GSD
            gsd_values = torch.full((image_size,), resolution, device=device, dtype=torch.float32)

            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            global_gsd[global_offset:global_offset + image_size] = gsd_values

        setattr(self, cache_key, global_gsd)
        return global_gsd

    def _precompute_latent_physical_positions(self, device):
        """
        Precompute physical positions of latent centers for each modality.
        
        Latents are arranged on a sqrt_L × sqrt_L grid covering the image.
        
        Lookup: [modality_index * L + latent_index] → (x_center, y_center)
        """
        cache_key = "latent_physical_positions"
        if hasattr(self, cache_key) and getattr(self, cache_key) is not None:
            cached = getattr(self, cache_key)
            if cached.device == device:
                return cached
            else:
                return cached.to(device)

        L = self.num_latents  # 400
        sqrt_L = self.sqrt_num_latents  # 20
        num_modalities = len(self.lookup_table.modalities)

        # [num_modalities * L, 2] where 2 = (x, y)
        global_positions = torch.zeros(num_modalities * L, 2, device=device)

        for mod_idx, modality in enumerate(tqdm(self.lookup_table.modalities, desc="Precomputing latent positions")):
            resolution, image_size = modality

            # Physical extent
            pos_scaling = image_size * resolution
            half_extent = pos_scaling / 2.0

            # Latent grid: evenly spaced within the image
            latent_spacing = pos_scaling / sqrt_L
            
            # Centers of the sqrt_L × sqrt_L grid
            latent_coords_1d = torch.linspace(
                -half_extent + latent_spacing / 2.0,
                half_extent - latent_spacing / 2.0,
                steps=sqrt_L,  # This is now 20, creating 20 positions per axis
                device=device
            )

            # Create 2D grid (row-major order)
            grid_y, grid_x = torch.meshgrid(latent_coords_1d, latent_coords_1d, indexing='ij')
            
            # Flatten to [L] = [400] in row-major order
            latent_x = grid_x.flatten()  # [400]
            latent_y = grid_y.flatten()  # [400]

            # Store at correct offset
            start_idx = mod_idx * L
            global_positions[start_idx:start_idx + L, 0] = latent_x
            global_positions[start_idx:start_idx + L, 1] = latent_y

        setattr(self, cache_key, global_positions)
        return global_positions

    def _precompute_modality_mapping(self, device):
        """
        Precompute mapping from query offset to modality index.
        
        This allows fast lookup of which modality a batch item belongs to.
        """
        cache_key = "_modality_mapping_precomputed"
        if hasattr(self, cache_key) and getattr(self, cache_key):
            return

        # Build sorted list of query offsets and corresponding modality indices
        query_offsets = []
        modality_indices = []
        
        for mod_idx, modality in enumerate(self.lookup_table.modalities):
            resolution, image_size = modality
            modality_key = (int(1000 * resolution), image_size)
            query_offset = self.lookup_table.table_queries[modality_key]
            query_offsets.append(query_offset)
            modality_indices.append(mod_idx)

        # Sort by query offset
        sorted_pairs = sorted(zip(query_offsets, modality_indices))
        query_offsets = [p[0] for p in sorted_pairs]
        modality_indices = [p[1] for p in sorted_pairs]

        self.register_buffer(
            "_query_offsets",
            torch.tensor(query_offsets, dtype=torch.long, device=device),
            persistent=False
        )
        self.register_buffer(
            "_modality_for_offset",
            torch.tensor(modality_indices, dtype=torch.long, device=device),
            persistent=False
        )

        setattr(self, cache_key, True)


    def _get_physical_scale_for_modality(self, modality_indices: torch.Tensor, device) -> torch.Tensor:
        """
        Get the physical scale (latent spacing) for each modality.
        
        This normalizes distances so r ≈ 1 when delta = one latent spacing.
        
        Args:
            modality_indices: [B] - modality index for each batch item
            
        Returns:
            physical_scale: [B] - latent spacing in meters for each modality
        """
        # Precompute if not cached
        cache_key = "modality_physical_scales"
        if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
            num_modalities = len(self.lookup_table.modalities)
            scales = torch.zeros(num_modalities, device=device)
            
            for mod_idx, modality in enumerate(self.lookup_table.modalities):
                resolution, image_size = modality
                physical_extent = image_size * resolution  # Total meters
                latent_spacing = physical_extent / self.sqrt_num_latents  # Meters per latent
                scales[mod_idx] = latent_spacing
            
            self.register_buffer(cache_key, scales, persistent=False)
        
        cached_scales = getattr(self, cache_key).to(device)
        return cached_scales[modality_indices]
    
    def _get_gsd_for_modality(self, modality_indices: torch.Tensor, device) -> torch.Tensor:
        """
        Get the GSD (ground sampling distance) for each modality.
        
        Args:
            modality_indices: [B] - modality index for each batch item
            
        Returns:
            gsd: [B] - GSD in meters for each modality
        """
        cache_key = "modality_gsds"
        if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
            num_modalities = len(self.lookup_table.modalities)
            gsds = torch.zeros(num_modalities, device=device)
            
            for mod_idx, modality in enumerate(self.lookup_table.modalities):
                resolution, image_size = modality
                gsds[mod_idx] = resolution
            
            self.register_buffer(cache_key, gsds, persistent=False)
        
        cached_gsds = getattr(self, cache_key).to(device)
        return cached_gsds[modality_indices]
    

    def _compute_polar_encoding(
        self,
        delta_x: torch.Tensor,         # [...] physical displacement in meters
        delta_y: torch.Tensor,         # [...] physical displacement in meters
        physical_scale: torch.Tensor,  # Broadcastable to delta shape
        gsd: torch.Tensor,             # Same shape as delta_x/delta_y
        device=None
    ) -> torch.Tensor:
        """
        Unified polar encoding computation for both encoder and decoder.
        
        Encodes:
        1. Spatial position (r, θ) - normalized by physical_scale, NOT by GSD
        2. Resolution context (gsd_ratio) - encoded separately, PER TOKEN
        
        Args:
            delta_x: Physical displacement in meters [...shape]
            delta_y: Physical displacement in meters [...shape]
            physical_scale: Latent spacing in meters (broadcastable)
            gsd: Ground sampling distance in meters, SAME SHAPE as delta_x/delta_y
            
        Returns:
            encoding: [..., D_pe]
        """
        device = device or delta_x.device
        
        # =========================================================================
        # 1. SPATIAL ENCODING: Normalize by physical_scale (NOT by GSD!)
        # =========================================================================
        delta_x_norm = delta_x / physical_scale
        delta_y_norm = delta_y / physical_scale
        
        # Polar coordinates
        r = torch.sqrt(delta_x_norm**2 + delta_y_norm**2 + 1e-8)
        theta = torch.atan2(delta_y_norm, delta_x_norm)  # [-π, π]
        
        # Angular fadeout when very close (r < 0.1 latent spacings)
        r_min = 0.1
        theta_weight = torch.tanh(r / r_min)
        
        # =========================================================================
        # 2. FOURIER ENCODING for spatial position
        # =========================================================================
        r_expanded = r.unsqueeze(-1)              # [..., 1]
        theta_expanded = theta.unsqueeze(-1)      # [..., 1]
        theta_weight_expanded = theta_weight.unsqueeze(-1)  # [..., 1]
        
        freq_r = self.polar_r_frequencies.to(device)      # [K_r]
        freq_theta = self.polar_theta_frequencies.to(device)  # [K_theta]
        
        # Reshape frequencies for broadcasting: [1, 1, ..., K]
        for _ in range(r_expanded.dim() - 1):
            freq_r = freq_r.unsqueeze(0)
            freq_theta = freq_theta.unsqueeze(0)
        
        # Radial Fourier
        r_sin = torch.sin(freq_r * r_expanded)    # [..., K_r]
        r_cos = torch.cos(freq_r * r_expanded)    # [..., K_r]
        
        # Angular Fourier
        theta_sin = torch.sin(freq_theta * theta_expanded) * theta_weight_expanded  # [..., K_theta]
        theta_cos = torch.cos(freq_theta * theta_expanded) * theta_weight_expanded  # [..., K_theta]
        
        # =========================================================================
        # 3. RESOLUTION ENCODING: GSD ratio PER TOKEN (separate from spatial!)
        # =========================================================================
        G_ref = 10.0  # Reference GSD in meters
        gsd_ratio = gsd / G_ref  # Same shape as delta_x: [...]
        
        # Add feature dimension
        gsd_ratio_expanded = gsd_ratio.unsqueeze(-1)  # [..., 1]
        
        # Log-scale encoding for wide GSD range
        log_gsd = torch.log(gsd_ratio + 1e-8)  # [...]
        log_gsd_expanded = log_gsd.unsqueeze(-1)  # [..., 1]
        
        # Fourier encoding of log(gsd_ratio)
        gsd_sin = torch.sin(freq_r * log_gsd_expanded)  # [..., K_r]
        gsd_cos = torch.cos(freq_r * log_gsd_expanded)  # [..., K_r]
        
        # =========================================================================
        # 4. Concatenate all components
        # =========================================================================
        encoding = torch.cat([
            # Spatial encoding
            r.unsqueeze(-1),       # [..., 1]
            r_sin,                 # [..., K_r]
            r_cos,                 # [..., K_r]
            theta_sin,             # [..., K_theta]
            theta_cos,             # [..., K_theta]
            # Resolution encoding (per token!)
            gsd_ratio_expanded,    # [..., 1]
            gsd_sin,               # [..., K_r]
            gsd_cos,               # [..., K_r]
        ], dim=-1)
        
        return encoding

    def _get_modality_indices_from_tokens(self, token_data: torch.Tensor, device) -> torch.Tensor:
        """
        Get modality index for each batch item from token data.
        
        Args:
            token_data: [B, L, m, 6] or [B, N, 6]
            
        Returns:
            modality_indices: [B] - index of modality (0, 1, 2, ...)
        """
        self._precompute_modality_mapping(device)

        # Get query index from first token
        if token_data.dim() == 4:
            # [B, L, m, 6]
            query_base = token_data[:, 0, 0, 5].long()  # [B]
        else:
            # [B, N, 6]
            query_base = token_data[:, 0, 5].long()  # [B]

        # Move buffers to correct device if needed
        query_offsets = self._query_offsets.to(device)
        modality_for_offset = self._modality_for_offset.to(device)

        # Find which modality each query_base belongs to
        # Use searchsorted: find insertion point, then go back one
        insertion_points = torch.searchsorted(query_offsets, query_base, right=True) - 1
        insertion_points = insertion_points.clamp(min=0)  # Safety
        modality_indices = modality_for_offset[insertion_points]

        return modality_indices

    
    def get_polar_positional_encoding(
        self,
        token_data: torch.Tensor,  # [B, L, m, 6]
        device=None
    ) -> torch.Tensor:
        """
        Compute polar relative positional encoding for encoder.
        """
        device = device or token_data.device
        B, L, m, _ = token_data.shape

        # 1. Get token physical positions
        token_centers = self._precompute_token_physical_centers(device)
        x_indices = token_data[..., 1].long()  # [B, L, m]
        y_indices = token_data[..., 2].long()  # [B, L, m]
        
        token_x = token_centers[x_indices]  # [B, L, m]
        token_y = token_centers[y_indices]  # [B, L, m]

        # 2. Get latent physical positions
        latent_positions = self._precompute_latent_physical_positions(device)
        modality_indices = self._get_modality_indices_from_tokens(token_data, device)  # [B]

        latent_start_indices = modality_indices * L  # [B]
        latent_idx = latent_start_indices.unsqueeze(1) + torch.arange(L, device=device).unsqueeze(0)  # [B, L]
        batch_latent_pos = latent_positions[latent_idx]  # [B, L, 2]

        latent_x = batch_latent_pos[:, :, 0].unsqueeze(-1)  # [B, L, 1]
        latent_y = batch_latent_pos[:, :, 1].unsqueeze(-1)  # [B, L, 1]

        # 3. Compute relative displacement
        delta_x = token_x - latent_x  # [B, L, m]
        delta_y = token_y - latent_y  # [B, L, m]

        # 4. Get physical scale (per batch) and GSD (per token!)
        physical_scale = self._get_physical_scale_for_modality(modality_indices, device)  # [B]
        physical_scale = physical_scale.view(B, 1, 1)  # [B, 1, 1]
        
        # GSD per token via lookup!
        gsd = self._get_gsd_for_tokens(token_data, device)  # [B, L, m]

        # 5. Compute unified polar encoding
        encoding = self._compute_polar_encoding(
            delta_x, delta_y, physical_scale, gsd, device
        )  # [B, L, m, D_pe]

        return encoding

    def get_polar_encoding_dimension(self) -> int:
        """
        Return the dimension of the polar + GSD encoding.
        
        D_pe = 1 + 2*K_r + 2*K_theta + 1 + 2*K_r
            = 2 + 4*K_r + 2*K_theta
        """
        K_r = self.polar_num_r_bands
        K_theta = self.polar_num_theta_bands
        return 2 + 4 * K_r + 2 * K_theta  # e.g., 2 + 32 + 16 = 50 for K_r=8, K_theta=8


    # =========================================================================
    # PROCESS DATA FOR ENCODER (NEW METHOD FOR [B, L, m, 6] SHAPED INPUT)
    # =========================================================================

    def _get_gsd_for_tokens(
        self,
        token_data: torch.Tensor,  # [B, ...] with GSD info in one of the columns
        device=None
    ) -> torch.Tensor:
        """
        Get GSD for each individual token.
        
        In Sentinel-2:
        - Bands 2,3,4,8 are 10m
        - Bands 5,6,7,8A,11,12 are 20m  
        - Bands 1,9,10 are 60m
        
        Args:
            token_data: Token tensor with band/resolution info
            
        Returns:
            gsd: Same shape as token_data[..., 0] (without last dim)
        """
        device = device or token_data.device
        
        # Option 1: If GSD is stored directly in a column (e.g., column 4)
        # Check your data format - adjust column index as needed
        gsd = token_data[..., 4].float()  # [B, L, m] or [B, N]
        
        # Option 2: If column 4 contains band index, lookup GSD from band
        # This requires a mapping from band index to GSD
        # band_to_gsd = torch.tensor([60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20], device=device)
        # band_idx = token_data[..., 4].long()
        # gsd = band_to_gsd[band_idx]
        
        return gsd

    def process_data_for_encoder(
        self,
        token_data: torch.Tensor,  # [B, L, m, 6]
        mask: torch.Tensor,        # [B, L, m]
        device=None
    ) -> torch.Tensor:
        """
        Process tokens grouped by latent for the encoder.
        
        Uses unified polar relative positional encoding.
        
        Args:
            token_data: [B, L, m, 6] - tokens grouped by geographic pruning
            mask: [B, L, m] - attention mask
            
        Returns:
            processed_tokens: [B, L, m, D_total]
        """
        device = device or token_data.device
        B, L, m, _ = token_data.shape

        # 1. Polar positional encoding (unified formula)
        polar_encoding = self.get_polar_positional_encoding(token_data, device)
        # [B, L, m, D_pe]

        # 2. Wavelength encoding
        wavelength_indices = token_data[..., 3].reshape(B, L * m)
        wavelength_encoding = self.get_wavelength_encoding(wavelength_indices, device=device)
        wavelength_encoding = wavelength_encoding.view(B, L, m, -1)
        # [B, L, m, D_wavelength]

        # 3. Reflectance encoding
        reflectance = token_data[..., 0]  # [B, L, m]
        reflectance_encoding = self.get_bvalue_processing(reflectance)
        # [B, L, m, D_refl]

        # 4. Concatenate all encodings
        processed = torch.cat([
            polar_encoding,       # [B, L, m, D_pe]
            wavelength_encoding,  # [B, L, m, D_wavelength]
            reflectance_encoding  # [B, L, m, D_refl]
        ], dim=-1)

        return processed
    

    def get_topk_latents_for_decoder(
        self,
        query_tokens: torch.Tensor,  # [B, N, 6] - tokens to reconstruct
        k: int = 16,                  # Number of latents per query
        device=None
    ) -> tuple:
        """
        For each query token, select the k nearest spatial latents.
        
        Args:
            query_tokens: [B, N, 6] - query tokens with position info
            k: number of latents to select per query
            
        Returns:
            selected_indices: [B, N, k] - indices of selected latents
            selected_distances: [B, N, k] - distances (for optional soft weighting)
        """
        device = device or query_tokens.device
        B, N, _ = query_tokens.shape
        L = self.num_latents  # 400
        
        # Get token physical positions
        token_centers = self._precompute_token_physical_centers(device)
        x_indices = query_tokens[..., 1].long()  # [B, N]
        y_indices = query_tokens[..., 2].long()  # [B, N]
        
        token_x = token_centers[x_indices]  # [B, N]
        token_y = token_centers[y_indices]  # [B, N]
        
        # Get latent positions
        latent_positions = self._precompute_latent_physical_positions(device)
        modality_indices = self._get_modality_indices_from_tokens(
            query_tokens.unsqueeze(2), device  # Add dummy dim to match expected shape
        )  # [B]
        
        latent_start_indices = modality_indices * L  # [B]
        latent_idx = latent_start_indices.unsqueeze(1) + torch.arange(L, device=device)  # [B, L]
        batch_latent_pos = latent_positions[latent_idx]  # [B, L, 2]
        
        latent_x = batch_latent_pos[:, :, 0]  # [B, L]
        latent_y = batch_latent_pos[:, :, 1]  # [B, L]
        
        # Compute distances: [B, N, L]
        # token positions: [B, N] -> [B, N, 1]
        # latent positions: [B, L] -> [B, 1, L]
        delta_x = token_x.unsqueeze(-1) - latent_x.unsqueeze(1)  # [B, N, L]
        delta_y = token_y.unsqueeze(-1) - latent_y.unsqueeze(1)  # [B, N, L]
        
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)  # [B, N, L]
        
        # Select k nearest latents (smallest distances)
        # Note: we want smallest, so negate for topk or use torch.topk with largest=False
        topk_distances, topk_indices = torch.topk(
            distances, k=k, dim=-1, largest=False
        )  # [B, N, k] each
        
        return topk_indices, topk_distances
    
    def get_decoder_relative_pe(
        self,
        query_tokens: torch.Tensor,  # [B, N, 6]
        latent_indices: torch.Tensor,  # [B, N, k]
        device=None
    ) -> torch.Tensor:
        """
        Compute relative polar PE from each query token to its selected latents.
        """
        device = device or query_tokens.device
        B, N, _ = query_tokens.shape
        k = latent_indices.shape[-1]
        L = self.num_latents

        # 1. Get token physical positions
        token_centers = self._precompute_token_physical_centers(device)
        token_x = token_centers[query_tokens[..., 1].long()]  # [B, N]
        token_y = token_centers[query_tokens[..., 2].long()]  # [B, N]

        # 2. Get latent physical positions for selected latents
        latent_positions = self._precompute_latent_physical_positions(device)
        modality_indices = self._get_modality_indices_from_tokens(
            query_tokens.unsqueeze(2), device
        )  # [B]

        latent_start_indices = modality_indices * L  # [B]
        global_latent_indices = latent_start_indices.unsqueeze(1).unsqueeze(2) + latent_indices  # [B, N, k]
        
        selected_latent_pos = latent_positions[global_latent_indices]  # [B, N, k, 2]
        latent_x = selected_latent_pos[..., 0]  # [B, N, k]
        latent_y = selected_latent_pos[..., 1]  # [B, N, k]

        # 3. Compute relative displacement
        delta_x = latent_x - token_x.unsqueeze(-1)  # [B, N, k]
        delta_y = latent_y - token_y.unsqueeze(-1)  # [B, N, k]

        # 4. Get physical scale and GSD
        physical_scale = self._get_physical_scale_for_modality(modality_indices, device)  # [B]
        physical_scale = physical_scale.view(B, 1, 1)  # [B, 1, 1]
        
        # GSD per token via lookup! [B, N]
        gsd = self._get_gsd_for_tokens(query_tokens, device)  # [B, N]
        # Expand to match delta shape [B, N, k]
        gsd = gsd.unsqueeze(-1).expand(-1, -1, k)  # [B, N, k]

        # 5. Compute unified polar encoding
        encoding = self._compute_polar_encoding(
            delta_x, delta_y, physical_scale, gsd, device
        )  # [B, N, k, D_pe]

        return encoding

    def get_gaussian_encoding(
        self,
        token_data: torch.Tensor,
        num_gaussians: int,
        sigma: float,
        device=None,
        extremums=None
    ):
        """Existing method - unchanged"""
        device = device or token_data.device
        batch, tokens, _ = token_data.shape

        cache_key = f"positional_encoding_{num_gaussians}_{sigma}_{device}"

        if not hasattr(self, cache_key):
            self._precompute_global_gaussian_encodings(num_gaussians, sigma, device, cache_key, extremums=extremums)

        cached_encoding = getattr(self, cache_key)

        global_x_indices = token_data[..., 1].long()
        global_y_indices = token_data[..., 2].long()

        encoding_x = cached_encoding[global_x_indices]
        encoding_y = cached_encoding[global_y_indices]

        result = torch.cat([encoding_x, encoding_y], dim=-1)

        return result

    def get_wavelength_encoding(self, token_data: torch.Tensor, device=None):
        """Existing method - unchanged"""
        cache_key = f"wavelength_encoding_gaussian"
        if not hasattr(self, cache_key):
            self._precompute_global_wavelength_encodings(device)

        cached_encoding = getattr(self, cache_key)
        global_x_indices = token_data.long()
        encoding_wavelengths = cached_encoding[global_x_indices]

        elevation_key = (-1, -1)
        if elevation_key in self.lookup_table.table_wave:
            elevation_idx = self.lookup_table.table_wave[elevation_key]
            elevation_mask = (global_x_indices == elevation_idx)

            elev_vec = self.elevation_.view(1, 1, -1).expand_as(encoding_wavelengths)
            encoding_wavelengths = torch.where(elevation_mask.unsqueeze(-1), elev_vec, encoding_wavelengths)

        return encoding_wavelengths

    def get_fourrier_encoding(self, token_data: torch.Tensor, device=None):
        """Existing method - unchanged (used for decoder)"""
        cache_key = f"positional_encoding_fourrier"

        if not hasattr(self, cache_key):
            self._precompute_global_fourrier_encodings(device)

        cached_encoding = getattr(self, cache_key)

        global_x_indices = token_data[..., 1].long()
        global_y_indices = token_data[..., 2].long()

        encoding_x = cached_encoding[global_x_indices]
        encoding_y = cached_encoding[global_y_indices]

        result = torch.cat([encoding_x, encoding_y], dim=-1)

        return result

    def get_bias_tokens_encoding(self, token_data: torch.Tensor, device=None):
        """Existing method - unchanged"""
        cache_key = f"bias_tokens_encoding"

        if not hasattr(self, cache_key):
            self._precompute_global_bias_tokens_encodings(device)

        cached_encoding = getattr(self, cache_key)

        global_x_indices = token_data[..., 1].long()
        global_y_indices = token_data[..., 2].long()

        encoding_x = cached_encoding[global_x_indices].unsqueeze(-2)
        encoding_y = cached_encoding[global_y_indices].unsqueeze(-2)

        result = torch.cat([encoding_x, encoding_y], dim=-2)

        return result

    def get_bias_latents_encoding(self, token_data: torch.Tensor, device=None):
        """Existing method - unchanged"""
        cache_key = f"bias_latents_encoding"

        if not hasattr(self, cache_key):
            self._precompute_global_bias_latents_encodings(device)

        cached_encoding = getattr(self, cache_key)

        global_indices = token_data[:, 0, 5].long()

        global_indices = global_indices.unsqueeze(1) + torch.arange(0, self.lookup_table.nb_tokens_queries,
                                                                     device=device)
        encoding = cached_encoding[global_indices]

        B, C, E = encoding.shape
        idx = torch.arange(C, device=device)
        i, j = torch.meshgrid(idx, idx, indexing="ij")
        i = i.reshape(-1)
        j = j.reshape(-1)
        encoding_i = encoding[:, i, :].unsqueeze(-2)
        encoding_j = encoding[:, j, :].unsqueeze(-2)

        result = torch.cat([encoding_i, encoding_j], dim=-2)

        return result

    def get_bvalue_processing(self, img):
        """Existing method - unchanged"""
        if self.config["Atomiser"]["bandvalue_encoding"] == "NATURAL":
            return img.unsqueeze(-1)

        elif self.config["Atomiser"]["bandvalue_encoding"] == "FF":
            num_bands = self.config["Atomiser"]["bandvalue_num_freq_bands"]
            max_freq = self.config["Atomiser"]["bandvalue_max_freq"]
            fourier_encoded_values = fourier_encode(img, max_freq, num_bands)

            return fourier_encoded_values

    # =========================================================================
    # PRECOMPUTATION METHODS (existing ones unchanged)
    # =========================================================================

    def _precompute_global_fourrier_encodings(self, device, queries=False):
        """Existing method - unchanged"""
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())

        if queries:
            max_global_index = len(self.lookup_table.table_queries.keys()) * self.lookup_table.nb_tokens_queries

        num_bands = self.config["Atomiser"]["pos_num_freq_bands"] * 2 + 1
        global_encoding = torch.zeros(max_global_index, num_bands, device=device)

        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing Fourier encodings"):
            resolution, image_size = modality

            if queries:
                pos = self.get_positional_encoding_fourrier(image_size, resolution, device,
                                                             sampling=self.lookup_table.nb_tokens_queries)
            else:
                pos = self.get_positional_encoding_fourrier(image_size, resolution, device)

            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]
            if queries:
                global_offset = self.lookup_table.table_queries[modality_key]

            if queries:
                global_encoding[global_offset:global_offset + self.lookup_table.nb_tokens_queries] = pos
            else:
                global_encoding[global_offset:global_offset + image_size] = pos

        cache_key = f"positional_encoding_fourrier"
        if queries:
            cache_key = f"positional_encoding_fourrier_queries"
        setattr(self, cache_key, global_encoding)

    def _precompute_global_bias_tokens_encodings(self, device):
        """Existing method - unchanged"""
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        global_encoding = torch.zeros(max_global_index, 2, device=device)

        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing bias token encodings"):
            resolution, image_size = modality

            pos_scalings = (image_size * resolution)
            axis = torch.linspace(-pos_scalings / 2.0, pos_scalings / 2.0, steps=image_size, device=device)

            axis_left = axis.clone()
            axis_right = axis.clone()
            axis_left = (axis_left - resolution / 2.0).unsqueeze(-1)
            axis_right = (axis_right + resolution / 2.0).unsqueeze(-1)

            axis = torch.cat([axis_left, axis_right], dim=-1)
            modality_key = (int(1000 * resolution), image_size)
            global_offset = self.lookup_table.table[modality_key]

            global_encoding[global_offset:global_offset + image_size] = axis

        cache_key = f"bias_tokens_encoding"
        setattr(self, cache_key, global_encoding)

    def _precompute_global_bias_latents_encodings(self, device):
        """Existing method - unchanged"""
        max_global_index = sum(self.spatial_latents_per_row  for _ in self.lookup_table.table_queries.keys())
        global_encoding = torch.zeros(max_global_index, 2, device=device)

        for modality in tqdm(self.lookup_table.modalities, desc="Precomputing bias latent encodings"):
            resolution, image_size = modality

            pos_scalings = (image_size * resolution)
            axis = torch.linspace(-pos_scalings / 2.0, pos_scalings / 2.0, steps=self.spatial_latents_per_row , device=device).unsqueeze(-1)
            centers = torch.full((self.spatial_latents_per_row , 1), resolution, device=device)
            
            axis = torch.cat([axis, centers], dim=-1)

            modality_key = (int(1000 * resolution), image_size)

            global_offset = self.lookup_table.table_queries[modality_key]

            global_encoding[global_offset:global_offset + self.spatial_latents_per_row ] = axis

        cache_key = f"bias_latents_encoding"
        setattr(self, cache_key, global_encoding)

    def _precompute_global_wavelength_encodings(self, device):
        """Existing method - unchanged"""
        max_global_index = len(self.lookup_table.table_wave.keys())
        global_encoding = torch.zeros(max_global_index, 19, device=device)

        for modality in tqdm(self.lookup_table.table_wave.keys(), desc="Precomputing Wavelength encodings"):
            bandwidth, central_wavelength = modality
            id_modality = self.lookup_table.table_wave[(bandwidth, central_wavelength)]

            encoded = None
            if bandwidth == -1 and central_wavelength == -1:
                continue

            if encoded == None:
                encoded = self.compute_gaussian_band_max_encoding([central_wavelength], [bandwidth],
                                                                   num_points=150).squeeze(0).squeeze(0)

            global_encoding[id_modality] = encoded

        cache_key = f"wavelength_encoding_gaussian"
        setattr(self, cache_key, global_encoding)

    def compute_gaussian_band_max_encoding(self, lambda_centers, bandwidths, num_points=50, modality="S2"):
        """Existing method - unchanged"""
        device = self.gaussian_means.device

        lambda_centers = torch.as_tensor(lambda_centers, dtype=torch.float32, device=device)
        bandwidths = torch.as_tensor(bandwidths, dtype=torch.float32, device=device)

        lambda_min = lambda_centers - bandwidths / 2
        lambda_max = lambda_centers + bandwidths / 2

        t = torch.linspace(0, 1, num_points, device=device)

        sampled_lambdas = lambda_min.unsqueeze(1) + (lambda_max - lambda_min).unsqueeze(1) * t.unsqueeze(0)

        gaussians = torch.exp(
            -0.5 * (
                    ((sampled_lambdas.unsqueeze(2) - self.gaussian_means.unsqueeze(0).unsqueeze(0)) /
                     self.gaussian_stds.unsqueeze(0).unsqueeze(0)) ** 2
            )
        )

        encoding = gaussians.max(dim=-2).values

        return encoding

    # =========================================================================
    # EXISTING APPLY TRANSFORMATIONS (for decoder - unchanged)
    # =========================================================================

    def apply_transformations_optique(self, im_sen, mask_sen, mode, query=False):
        """Existing method - used for decoder queries"""
        if query:
            central_wavelength_processing = self.get_wavelength_encoding(im_sen[:, :, 3], device=im_sen.device)
            #p_x = self.get_fourrier_encoding(im_sen, device=im_sen.device)

            tokens = torch.cat([
                central_wavelength_processing
            ], dim=-1)

            tokens_bias = self.get_bias_tokens_encoding(im_sen, device=im_sen.device)
            latents_bias = self.get_bias_latents_encoding(im_sen, device=im_sen.device)

            return tokens, mask_sen, (tokens_bias, latents_bias)

        # For encoder - this path is now deprecated in favor of process_data_for_encoder
        tokens=self.process_data_for_encoder(tokens,mask_sen,device=tokens.device)
        p_latents = self.get_fourrier_encoding_queries(im_sen, device=im_sen.device)

        

        return tokens, mask_sen, p_latents, (tokens_bias, latents_bias)

    def get_tokens(self, img, mask, mode="optique", modality="s2", wave_encoding=None, query=False):
        """Existing method - unchanged"""
        if mode == "optique":
            return self.apply_transformations_optique(img, mask, modality, query=query)

    def get_bias_data(self, img):
        """Existing method - unchanged"""
        tokens_bias = self.get_bias_tokens_encoding(img, device=img.device)
        latents_bias = self.get_bias_latents_encoding(img, device=img.device)

        return (tokens_bias, latents_bias)

    def process_data(self, img, mask, query=False):
        """Existing method - unchanged (used for decoder)"""
        if self.config["dataset"]["S2"]:
            if query:
                tokens_s2, tokens_mask_s2, bias = self.get_tokens(img, mask, mode="optique", modality="s2", query=query)
                return tokens_s2, tokens_mask_s2, bias
            else:
                tokens_s2, tokens_mask_s2, latents, bias = self.get_tokens(img, mask, mode="optique", modality="s2",
                                                                            query=query)
                return tokens_s2, tokens_mask_s2, latents, bias