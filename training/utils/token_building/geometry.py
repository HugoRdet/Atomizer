import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from tqdm import tqdm
import math


class SensorGeometry(nn.Module):
    """
    The Physics Engine (The Map).
    
    Responsibilities:
    1. Manage physical constants (latent spacing, GSD).
    2. Convert token indices to physical coordinates (meters).
    3. Generate the default latent grid.
    4. Provide geometry data for Decoder Biases.
    
    PERFORMANCE: All constants are pre-computed in __init__ to avoid
    repeated tensor creation during forward pass.
    """
    
    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config = config
        self.lookup_table = lookup_table
        
        # --- Constants from Config ---
        self.spatial_latents_per_row = config["Atomiser"]["spatial_latents"] 
        self.num_spatial_latents = self.spatial_latents_per_row ** 2
        self.latent_surface = config["Atomiser"].get("latent_surface", 103.0)
        
        # Distance between adjacent latents
        if self.spatial_latents_per_row > 1:
            self.default_spacing = self.latent_surface / (self.spatial_latents_per_row - 1)
        else:
            self.default_spacing = self.latent_surface
            
        # Reference GSD
        self.default_gsd = config["Atomiser"].get("gsd", 0.2)
        
        # --- Pre-compute Constants (PERFORMANCE) ---
        # These are used in _precompute_integral_lut - compute once, not every call
        self.register_buffer("_sqrt_2", torch.tensor(math.sqrt(2.0)))
        
        # --- Initialize Lookup Tables ---
        self._init_token_geometry_buffers()
        self._init_modality_buffers()
        self._init_latent_grid()

    def _init_token_geometry_buffers(self):
        """Create lookup tables for token index -> physical coordinates."""
        max_global_index = sum(size for _, size in self.lookup_table.table.keys())
        
        centers = torch.zeros(max_global_index, dtype=torch.float32)
        gsds = torch.zeros(max_global_index, dtype=torch.float32)
        
        # Track token width (assume uniform within modality, use first)
        first_token_width = None

        for modality in tqdm(self.lookup_table.modalities, desc="Initializing Geometry"):
            resolution, image_size = modality
            pos_scaling = image_size * resolution 
            
            modality_centers = torch.linspace(
                -pos_scaling / 2.0 + resolution / 2.0,
                pos_scaling / 2.0 - resolution / 2.0,
                steps=image_size
            )
            
            modality_key = (int(1000 * resolution), image_size)
            start_idx = self.lookup_table.table[modality_key]
            
            centers[start_idx:start_idx + image_size] = modality_centers
            gsds[start_idx:start_idx + image_size] = resolution
            
            # Capture token width from first modality with >1 pixel
            if first_token_width is None and image_size > 1:
                first_token_width = (modality_centers[1] - modality_centers[0]).abs().item()

        self.register_buffer("token_centers_lookup", centers)
        self.register_buffer("token_gsd_lookup", gsds)
        
        # Pre-compute token width and half-width (PERFORMANCE)
        if first_token_width is None:
            first_token_width = 1.0
        self.register_buffer("_token_width", torch.tensor(first_token_width))
        self.register_buffer("_half_token_width", torch.tensor(first_token_width / 2.0))

    def _init_modality_buffers(self):
        """Create modality ID -> physical properties mapping."""
        query_offsets = []
        modality_indices = []
        physical_scales = []
        
        for mod_idx, modality in enumerate(self.lookup_table.modalities):
            resolution, image_size = modality
            modality_key = (int(1000 * resolution), image_size)
            query_offset = self.lookup_table.table_queries[modality_key]
            
            query_offsets.append(query_offset)
            modality_indices.append(mod_idx)
            
            physical_extent = image_size * resolution
            if self.spatial_latents_per_row > 1:
                scale = physical_extent / (self.spatial_latents_per_row - 1)
            else:
                scale = physical_extent
            physical_scales.append(scale)

        sorted_pairs = sorted(zip(query_offsets, modality_indices, physical_scales))
        
        self.register_buffer(
            "query_offsets", 
            torch.tensor([p[0] for p in sorted_pairs], dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "modality_indices", 
            torch.tensor([p[1] for p in sorted_pairs], dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "modality_scales", 
            torch.tensor([physical_scales[p[1]] for p in sorted_pairs], dtype=torch.float32),
            persistent=False
        )

    def _init_latent_grid(self):
        """Pre-compute the default latent grid (PERFORMANCE - done once)."""
        half_extent = self.latent_surface / 2.0
        coords = torch.linspace(-half_extent, half_extent, self.spatial_latents_per_row)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        # Register as buffer - will auto-move to GPU with model
        self.register_buffer("latent_grid", grid)

    # =========================================================================
    # PUBLIC API (Optimized)
    # =========================================================================

    def get_token_centers(self, token_data: torch.Tensor) -> torch.Tensor:
        """Convert token indices to (x, y) meters. No device transfers."""
        x_idx = token_data[..., 1].long()
        y_idx = token_data[..., 2].long()
        
        # Direct indexing - buffer is already on correct device
        x_meters = self.token_centers_lookup[x_idx]
        y_meters = self.token_centers_lookup[y_idx]
        
        return torch.stack([x_meters, y_meters], dim=-1)

    def get_token_gsd(self, token_data: torch.Tensor) -> torch.Tensor:
        """Get GSD for tokens."""
        x_idx = token_data[..., 1].long()
        return self.token_gsd_lookup[x_idx]

    def get_default_latent_grid(self, device=None) -> torch.Tensor:
        """
        Get the pre-computed latent grid.
        
        PERFORMANCE: Grid is pre-computed in __init__, just return it.
        Device parameter ignored - buffer auto-moves with model.
        """
        return self.latent_grid

    def get_physical_scale(self, token_data: torch.Tensor) -> torch.Tensor:
        """Get normalization scale for input batch."""
        if token_data.dim() == 4:
            query_base = token_data[:, 0, 0, 5].long()
        else:
            query_base = token_data[:, 0, 5].long()

        idx = torch.searchsorted(self.query_offsets, query_base, right=True) - 1
        idx = idx.clamp(min=0)
        
        scales = self.modality_scales[idx]
        return scales.view(-1, 1, 1)

    def get_integral_constants(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get pre-computed constants for Gaussian integral computation.
        
        Returns:
            sqrt_2: scalar tensor
            half_width: scalar tensor (half token width in meters)
            token_centers: [N] tensor of all token centers
        """
        return self._sqrt_2, self._half_token_width, self.token_centers_lookup

    def get_decoder_bias(self, query_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get geometry data for decoder attention (centers only)."""
        token_coords = self.get_token_centers(query_tokens)
        B = query_tokens.shape[0]
        latent_coords = self.latent_grid.unsqueeze(0).expand(B, -1, -1)
        return token_coords, latent_coords

    def get_decoder_bias_legacy(
        self, 
        query_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy bias format: edge bounds."""
        B, N, _ = query_tokens.shape
        device = query_tokens.device
        
        x_idx = query_tokens[..., 1].long()
        y_idx = query_tokens[..., 2].long()
        
        x_center = self.token_centers_lookup[x_idx]
        y_center = self.token_centers_lookup[y_idx]
        gsd = self.token_gsd_lookup[x_idx]
        
        half_gsd = gsd / 2.0
        x_edges = torch.stack([x_center - half_gsd, x_center + half_gsd], dim=-1)
        y_edges = torch.stack([y_center - half_gsd, y_center + half_gsd], dim=-1)
        token_bias = torch.stack([x_edges, y_edges], dim=-2)
        
        half_extent = self.latent_surface / 2.0
        latent_1d = torch.linspace(-half_extent, half_extent, 
                                   self.spatial_latents_per_row, device=device)
        gsd_col = torch.full((self.spatial_latents_per_row,), self.default_gsd, device=device)
        latent_bias_1d = torch.stack([latent_1d, gsd_col], dim=-1)
        latent_bias = latent_bias_1d.unsqueeze(0).expand(B, -1, -1)
        
        return token_bias, latent_bias

    def get_encoder_bias(
        self, 
        token_data: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get geometry data for encoder cross-attention."""
        B, L, m, _ = token_data.shape
        device = token_data.device
        
        x_idx = token_data[..., 1].long()
        y_idx = token_data[..., 2].long()
        
        x_center = self.token_centers_lookup[x_idx]
        y_center = self.token_centers_lookup[y_idx]
        gsd = self.token_gsd_lookup[x_idx]
        
        half_gsd = gsd / 2.0
        x_edges = torch.stack([x_center - half_gsd, x_center + half_gsd], dim=-1)
        y_edges = torch.stack([y_center - half_gsd, y_center + half_gsd], dim=-1)
        token_bias = torch.stack([x_edges, y_edges], dim=-2)
        
        if latent_positions is not None:
            latent_bias = latent_positions
        else:
            latent_bias = self.latent_grid.unsqueeze(0).expand(B, -1, -1)
        
        return token_bias, latent_bias