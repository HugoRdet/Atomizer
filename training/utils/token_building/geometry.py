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


    def get_query_pixel_coords(
        self, 
        query_tokens: torch.Tensor, 
        image_size: int = 512
    ) -> torch.Tensor:
        """
        Convert query token metadata into global pixel (x, y) coordinates.
        
        Args:
            query_tokens: [B, N, 6] or [B, N, D] tokens where index 1,2 are 
                          spatial indices mapping to meters.
            image_size: The target image resolution (default 512).
            
        Returns:
            pixel_coords: [B, N, 2] tensor of long integers (x, y) 
                          clamped to [0, image_size-1].
        """
        # 1. Convert the token metadata indices into meters (x, y)
        # Uses the pre-registered 'token_centers_lookup' buffer
        meter_coords = self.get_token_centers(query_tokens) # [B, N, 2]
        
        # 2. Project meters into pixel space
        # We reuse your existing meters_to_pixels logic
        pixel_coords = self.meters_to_pixels(
            meter_coords, 
            image_size=image_size, 
            gsd=None  # Uses self.default_gsd
        )
        
        return pixel_coords

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

    # =========================================================================
    # COORDINATE CONVERSION (for error-guided displacement)
    # =========================================================================

    def meters_to_pixels(
        self,
        coords_meters: torch.Tensor,
        image_size: int = 512,
        gsd: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert coordinates from meters to pixel indices.
        
        Assumes image is centered at origin, spanning [-extent/2, extent/2] meters
        where extent = image_size * gsd.
        
        Args:
            coords_meters: [..., 2] coordinates in meters (x, y)
                Can be any shape as long as last dimension is 2.
                Examples: [B, L, 2], [B, depth, L, 2], [B, depth, L, num_samples, 2]
            image_size: image dimension in pixels (default 512)
            gsd: ground sample distance in meters/pixel (default: self.default_gsd)
        
        Returns:
            coords_pixels: [..., 2] coordinates in pixels (long integers)
                Same shape as input, clamped to [0, image_size-1]
        
        Example:
            >>> geometry = SensorGeometry(config, lookup_table)
            >>> # Latent at origin (0, 0) meters → center of image (256, 256) pixels
            >>> coords_m = torch.tensor([[[0.0, 0.0]]])  # [1, 1, 2]
            >>> coords_px = geometry.meters_to_pixels(coords_m)
            >>> # coords_px ≈ [[[256, 256]]]
            
            >>> # Latent at (-51.2, -51.2) meters → corner (0, 0) pixels
            >>> coords_m = torch.tensor([[[-51.2, -51.2]]])
            >>> coords_px = geometry.meters_to_pixels(coords_m)
            >>> # coords_px = [[[0, 0]]]
        """
        if gsd is None:
            gsd = self.default_gsd
        
        # Physical extent of the image
        extent = image_size * gsd  # 512 * 0.2 = 102.4 meters
        half_extent = extent / 2.0  # 51.2 meters
        
        # Convert: meters → pixels
        # Origin (0,0) in meters → center of image (image_size/2) in pixels
        # -half_extent in meters → 0 in pixels
        # +half_extent in meters → image_size in pixels
        coords_pixels = (coords_meters + half_extent) / gsd
        
        # Round to integers and clamp to valid range
        coords_pixels = coords_pixels.round().long()
        coords_pixels = coords_pixels.clamp(0, image_size - 1)
        
        return coords_pixels

    def pixels_to_meters(
        self,
        coords_pixels: torch.Tensor,
        image_size: int = 512,
        gsd: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert coordinates from pixel indices to meters.
        
        Inverse of meters_to_pixels().
        
        Args:
            coords_pixels: [..., 2] coordinates in pixels (x, y)
            image_size: image dimension in pixels (default 512)
            gsd: ground sample distance in meters/pixel (default: self.default_gsd)
        
        Returns:
            coords_meters: [..., 2] coordinates in meters
        """
        if gsd is None:
            gsd = self.default_gsd
        
        extent = image_size * gsd
        half_extent = extent / 2.0
        
        # Convert: pixels → meters
        coords_meters = coords_pixels.float() * gsd - half_extent
        
        return coords_meters

    def sample_grid_around_positions(
        self,
        coords_pixels: torch.Tensor,
        grid_size: int = 3,
        spacing: int = 2,
        image_size: int = 512,
    ) -> torch.Tensor:
        """
        Sample a grid of points around each position.
        
        Creates a grid_size × grid_size grid of sample points centered 
        at each input position.
        
        Args:
            coords_pixels: [..., 2] center coordinates in pixels
                Examples: [B, L, 2], [B, depth, L, 2]
            grid_size: number of points per dimension (e.g., 3 → 3×3 = 9 points)
            spacing: distance between grid points in pixels
            image_size: image dimension for clamping (default 512)
        
        Returns:
            sample_coords: [..., grid_size², 2] sample coordinates in pixels
                Example: [B, L, 2] → [B, L, 9, 2] for grid_size=3
        
        Example:
            >>> coords = torch.tensor([[[256, 256]]])  # [1, 1, 2]
            >>> samples = geometry.sample_grid_around_positions(coords, grid_size=3, spacing=2)
            >>> # samples[0, 0] contains 9 points:
            >>> # (254,254), (254,256), (254,258),
            >>> # (256,254), (256,256), (256,258),
            >>> # (258,254), (258,256), (258,258)
        """
        device = coords_pixels.device
        original_shape = coords_pixels.shape[:-1]  # [...] without the 2
        
        # Create grid offsets: [-spacing, 0, spacing] for grid_size=3
        half_grid = (grid_size - 1) // 2
        offsets_1d = torch.arange(-half_grid, half_grid + 1, device=device) * spacing
        
        # Create 2D grid of offsets: [grid_size², 2]
        grid_y, grid_x = torch.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [grid_size², 2]
        num_samples = offsets.shape[0]
        
        # Expand coords to [..., 1, 2] and offsets to [1, ..., grid_size², 2]
        coords_expanded = coords_pixels.unsqueeze(-2)  # [..., 1, 2]
        
        # Broadcast addition: [..., 1, 2] + [grid_size², 2] → [..., grid_size², 2]
        sample_coords = coords_expanded + offsets
        
        # Clamp to valid image range
        sample_coords = sample_coords.clamp(0, image_size - 1)
        
        return sample_coords

    def extract_query_tokens_from_image(
        self,
        image_err: torch.Tensor,
        sample_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract query tokens from image at given pixel positions.
        
        For each position, extracts tokens for ALL channels (bands).
        
        Args:
            image_err: [B, C, H, W, 6] full image with metadata
            sample_coords: [B, ..., 2] pixel coordinates (x, y)
                Can be any shape, e.g., [B, L, 2] or [B, depth, L, num_samples, 2]
        
        Returns:
            query_tokens: [B, num_queries, 6] where num_queries = prod(sample_shape) * C
            ground_truth: [B, num_queries] reflectance values at those positions
            original_shape: tuple of the shape between B and 2 in sample_coords
                (useful for reshaping error back to per-latent format)
        
        Example:
            >>> sample_coords = torch.tensor([[[256, 256], [100, 100]]])  # [1, 2, 2]
            >>> tokens, gt, shape = geometry.extract_query_tokens_from_image(image_err, sample_coords)
            >>> # tokens.shape = [1, 2*5, 6] = [1, 10, 6] (2 positions × 5 channels)
            >>> # shape = (2,)
        """
        B, C, H, W, metadata_dim = image_err.shape
        device = image_err.device
        
        # Flatten sample coordinates: [B, ..., 2] → [B, N, 2]
        original_shape = sample_coords.shape[1:-1]  # everything between B and 2
        N = sample_coords[..., 0].numel() // B  # number of positions per batch
        sample_coords_flat = sample_coords.view(B, N, 2)
        
        # Get pixel indices
        px_x = sample_coords_flat[..., 0].long()  # [B, N]
        px_y = sample_coords_flat[..., 1].long()  # [B, N]
        
        # Clamp to valid range
        px_x = px_x.clamp(0, W - 1)
        px_y = px_y.clamp(0, H - 1)
        
        # Extract tokens for all channels at each position
        # We need: image_err[b, c, y, x, :] for all b, all c, all (x,y) pairs
        
        # Expand indices for all channels: [B, N] → [B, N, C]
        px_x_exp = px_x.unsqueeze(-1).expand(-1, -1, C)  # [B, N, C]
        px_y_exp = px_y.unsqueeze(-1).expand(-1, -1, C)  # [B, N, C]
        
        # Create batch and channel indices
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, C)
        channel_idx = torch.arange(C, device=device).view(1, 1, C).expand(B, N, -1)
        
        # Index into image_err: [B, C, H, W, 6]
        # Result: [B, N, C, 6]
        tokens = image_err[batch_idx, channel_idx, px_y_exp, px_x_exp, :]
        
        # Reshape to [B, N*C, 6]
        query_tokens = tokens.view(B, N * C, metadata_dim)
        
        # Ground truth is the reflectance (index 0)
        ground_truth = query_tokens[..., 0]  # [B, N*C]
        
        return query_tokens, ground_truth, original_shape