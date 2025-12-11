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
        self.G_ref = config["Atomiser"].get("G_ref", 0.2)  # Reference GSD in meters

        self.cartesian_max_freq = config["Atomiser"].get("cartesian_max_freq", 32)
        self.cartesian_num_bands = config["Atomiser"].get("cartesian_num_bands", 32)

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

        self.spatial_latents_per_row = config["Atomiser"]["spatial_latents"]  # e.g., 35
        self.num_spatial_latents = self.spatial_latents_per_row ** 2          # e.g., 1225$

        self.latent_surface = config["Atomiser"].get("latent_surface", 103)  # meters
        self.physical_scale = self.latent_surface / (self.spatial_latents_per_row - 1)
        self.gsd = config["Atomiser"].get("gsd", 0.2)  # meters (constant for now)

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

    def _precompute_latent_physical_positions(self, device=None):
        """
        Latent grid spanning [-latent_surface/2, +latent_surface/2].
        """
        cache_key = "_latent_positions_cache"
        if hasattr(self, cache_key):
            return getattr(self, cache_key).to(device)
        
        half_extent = self.latent_surface / 2.0
        
        coords = torch.linspace(-half_extent, half_extent, self.spatial_latents_per_row)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        
        positions = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(),
        ], dim=-1)  # [L, 2]
        
        setattr(self, cache_key, positions)
        return positions.to(device)

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
    
    def _compute_cartesian_encoding(self, delta_x, delta_y, device=None):
        """
        Encode relative position in Cartesian coordinates with Fourier features.
        Uses fourier_encode for consistency.
        """
        device = device or delta_x.device
        
        # Normalize by physical_scale
        delta_x_norm = delta_x / self.physical_scale
        delta_y_norm = delta_y / self.physical_scale
        
        # Fourier encode x and y positions
        # fourier_encode returns [sin, cos, orig] with shape [..., num_bands*2 + 1]
        x_encoded = fourier_encode(
            delta_x_norm, 
            max_freq=self.cartesian_max_freq,
            num_bands=self.cartesian_num_bands
        )  # [..., num_bands*2 + 1]
        
        y_encoded = fourier_encode(
            delta_y_norm,
            max_freq=self.cartesian_max_freq,
            num_bands=self.cartesian_num_bands
        )  # [..., num_bands*2 + 1]
        
        # GSD encoding (constant for now)
        G_ref = 0.2
        gsd_ratio = self.gsd / G_ref
        log_gsd = np.log(gsd_ratio + 1e-8)
        
        # Create tensor matching delta shape
        log_gsd_tensor = torch.full_like(delta_x_norm, log_gsd)
        
        gsd_encoded = fourier_encode(
            log_gsd_tensor,
            max_freq=self.cartesian_max_freq,
            num_bands=self.cartesian_num_bands
        )  # [..., num_bands*2 + 1]
        
        # Concatenate all encodings
        encoding = torch.cat([
            x_encoded,    # [..., num_bands*2 + 1]
            y_encoded,    # [..., num_bands*2 + 1]
            gsd_encoded,  # [..., num_bands*2 + 1]
        ], dim=-1)
        
        return encoding


    def get_cartesian_encoding_dimension(self):
        """Return the dimension of cartesian positional encoding."""
        # fourier_encode returns: num_bands*2 (sin+cos) + 1 (original)
        per_component = self.cartesian_num_bands * 2 + 1
        # x, y, gsd
        return per_component * 3

    def _compute_polar_encoding(self, delta_x, delta_y, device=None):
        """
        Encode relative position in polar coordinates with Fourier features.
        Uses fourier_encode for consistency.
        """
        device = device or delta_x.device
        
        # Normalize by physical_scale
        delta_x_norm = delta_x / self.physical_scale
        delta_y_norm = delta_y / self.physical_scale
        
        # Polar coordinates
        r = torch.sqrt(delta_x_norm**2 + delta_y_norm**2 + 1e-8)
        theta = torch.atan2(delta_y_norm, delta_x_norm)  # [-π, π]
        
        # Normalize theta to [-1, 1] for fourier_encode (which multiplies by π)
        theta_norm = theta / torch.pi  # [-1, 1]
        
        # Fourier encode r (radial distance)
        r_encoded = fourier_encode(
            r,
            max_freq=self.cartesian_max_freq,
            num_bands=self.cartesian_num_bands
        )  # [..., num_bands*2 + 1]
        
        # Fourier encode theta (angle)
        theta_encoded = fourier_encode(
            theta_norm,
            max_freq=self.cartesian_max_freq,
            num_bands=self.cartesian_num_bands
        )  # [..., num_bands*2 + 1]
        
        # GSD encoding (constant for now)
        G_ref = 0.2
        gsd_ratio = self.gsd / G_ref
        log_gsd = np.log(gsd_ratio + 1e-8)
        
        log_gsd_tensor = torch.full_like(r, log_gsd)
        
        gsd_encoded = fourier_encode(
            log_gsd_tensor,
            max_freq=self.cartesian_max_freq,
            num_bands=self.cartesian_num_bands
        )  # [..., num_bands*2 + 1]
        
        # Concatenate all encodings
        encoding = torch.cat([
            r_encoded,      # [..., num_bands*2 + 1]
            theta_encoded,  # [..., num_bands*2 + 1]
            gsd_encoded,    # [..., num_bands*2 + 1]
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
        Simplified: latent positions are fixed, no modality lookup needed.
        """
        device = device or token_data.device
        B, L, m, _ = token_data.shape
        
        # 1. Get token physical positions
        token_centers = self._precompute_token_physical_centers(device)
        x_indices = token_data[..., 1].long()  # [B, L, m]
        y_indices = token_data[..., 2].long()  # [B, L, m]
        token_x = token_centers[x_indices]  # [B, L, m]
        token_y = token_centers[y_indices]  # [B, L, m]
        
        # 2. Get latent physical positions (FIXED - same for all batches!)
        latent_positions = self._precompute_latent_physical_positions(device)  # [L, 2]
        latent_x = latent_positions[:, 0].view(1, L, 1)  # [1, L, 1]
        latent_y = latent_positions[:, 1].view(1, L, 1)  # [1, L, 1]
        
        # 3. Compute relative displacement
        delta_x = token_x - latent_x  # [B, L, m]
        delta_y = token_y - latent_y  # [B, L, m]
        
        # 4. Compute polar encoding (physical_scale and gsd are constants now)
        #encoding = self._compute_cartesian_encoding(delta_x, delta_y, device) #
        encoding = self._compute_polar_encoding(delta_x, delta_y, device)  # [B, L, m, D_pe]
        return encoding

    def get_polar_encoding_dimension(self):
        """Return the dimension of polar positional encoding."""
        # fourier_encode returns: num_bands*2 (sin+cos) + 1 (original)
        r_dim = self.cartesian_num_bands * 2 + 1
        theta_dim = self.cartesian_num_bands * 2 + 1
        gsd_dim = self.cartesian_num_bands * 2 + 1
        return r_dim + theta_dim + gsd_dim


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
        Currently all tokens have 0.2m GSD.
        """
        device = device or token_data.device
        
        # Return constant GSD of 0.2m for all tokens
        # Shape: same as token_data without the last dimension
        gsd = torch.full(token_data.shape[:-1], 0.2, device=device, dtype=token_data.dtype)
        
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
        polar_encoding = self.get_polar_positional_encoding(token_data, device)#self.get_polar_positional_encoding(token_data, device)
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
        query_tokens: torch.Tensor,  # [B, N, 6]
        k: int = 16,
        device=None
    ) -> tuple:
        """
        For each query token, select the k nearest spatial latents.
        """
        device = device or query_tokens.device
        B, N, _ = query_tokens.shape
        L = self.num_spatial_latents  # e.g., 1225
        
        # Safety check
        k = min(k, L)
        
        # 1. Get token physical positions
        token_centers = self._precompute_token_physical_centers(device)
        x_indices = query_tokens[..., 1].long()  # [B, N]
        y_indices = query_tokens[..., 2].long()  # [B, N]
        token_x = token_centers[x_indices]  # [B, N]
        token_y = token_centers[y_indices]  # [B, N]
        
        # 2. Get latent positions (FIXED - same for all batches!)
        latent_positions = self._precompute_latent_physical_positions(device)  # [L, 2]
        latent_x = latent_positions[:, 0]  # [L]
        latent_y = latent_positions[:, 1]  # [L]
        
        # 3. Compute distances: [B, N, L] #we're using euclidean distance
        delta_x = token_x.unsqueeze(-1) - latent_x.view(1, 1, L)  # [B, N, L]
        delta_y = token_y.unsqueeze(-1) - latent_y.view(1, 1, L)  # [B, N, L]
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)  # [B, N, L]
        
        # 4. Select k nearest latents
        topk_distances, topk_indices = torch.topk(
            distances, k=k, dim=-1, largest=False
        )  # [B, N, k] each
        
        return topk_indices, topk_distances
    
    def get_decoder_relative_pe(
        self,
        query_tokens: torch.Tensor,  # [B, N, 6]
        latent_indices: torch.Tensor,  # [B, L , k]
        device=None
    ) -> torch.Tensor:
        """
        Compute relative polar PE from each query token to its selected latents.
        """
        device = device or query_tokens.device
        B, N, _ = query_tokens.shape
        k = latent_indices.shape[-1]
        
        # 1. Get token physical positions
        token_centers = self._precompute_token_physical_centers(device)
        token_x = token_centers[query_tokens[..., 1].long()]  # [B, N]
        token_y = token_centers[query_tokens[..., 2].long()]  # [B, N]
        
        # 2. Get latent physical positions (FIXED - direct indexing!)
        latent_positions = self._precompute_latent_physical_positions(device)  # [L, 2]
        selected_latent_pos = latent_positions[latent_indices]  # [B, N, k, 2]
        latent_x = selected_latent_pos[..., 0]  # [B, N, k]
        latent_y = selected_latent_pos[..., 1]  # [B, N, k]
        
        # 3. Compute relative displacement
        delta_x = latent_x - token_x.unsqueeze(-1)  # [B, N, k]
        delta_y = latent_y - token_y.unsqueeze(-1)  # [B, N, k]
        
        # 4. Compute polar encoding (physical_scale and gsd are constants)
        encoding = self._compute_cartesian_encoding(delta_x, delta_y, device)#self._compute_polar_encoding(delta_x, delta_y, device)  # [B, N, k, D_pe]
        
        return encoding

    def get_gaussian_encoding(
        self,
        token_data: torch.Tensor,
        num_gaussians: int,
        sigma: float,
        device=None,
        extremums=None
    ):
        
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
        
        central_wavelength_processing = self.get_wavelength_encoding(im_sen[:, :, 3], device=im_sen.device)
        #p_x = self.get_fourrier_encoding(im_sen, device=im_sen.device)

        tokens = torch.cat([
            central_wavelength_processing
        ], dim=-1)

        tokens_bias = self.get_bias_tokens_encoding(im_sen, device=im_sen.device)
        latents_bias = self.get_bias_latents_encoding(im_sen, device=im_sen.device)

        return tokens, mask_sen, (tokens_bias, latents_bias)

        

    def get_tokens(self, img, mask, mode="optique", modality="s2", wave_encoding=None, query=False):
        
        if mode == "optique":
            return self.apply_transformations_optique(img, mask, modality, query=query)

    def get_bias_data(self, img):
        
        tokens_bias = self.get_bias_tokens_encoding(img, device=img.device)
        latents_bias = self.get_bias_latents_encoding(img, device=img.device)

        return (tokens_bias, latents_bias)

    def process_data(self, img, mask, query=False):
        """Existing method - unchanged (used for decoder)"""
        if self.config["dataset"]["S2"]:
            if query:
                tokens_s2, tokens_mask_s2, bias = self.get_tokens(img, mask, mode="optique", modality="s2", query=query)
                return tokens_s2, tokens_mask_s2, bias
            