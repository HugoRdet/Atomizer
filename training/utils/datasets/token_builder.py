"""
Centralized token building utilities for remote sensing datasets.

Uses a reference grid system where all crops at a given resolution
extract coordinates from a shared reference grid (e.g., 512×512).
This ensures consistent coordinate spaces across different crop sizes.
"""

import torch
import einops


class TokenBuilder:
    """
    Centralized token builder for remote sensing datasets.
    
    Uses reference grid indexing: instead of creating position encodings
    for each crop size, maintains reference grids (e.g., 512×512) and
    extracts windows from them.
    
    Benefits:
    - All crops share the same coordinate space
    - No dynamic modality registration for different crop sizes
    - Consistent positioning between training and validation
    - Handles edge crops naturally
    - Auto-registers unseen resolutions (e.g., from resolution augmentation)
    
    Args:
        lookup_table: Lookup_encoding instance for position/spectral/query offsets
    
    Example:
        >>> lookup = Lookup_encoding(config)
        >>> builder = TokenBuilder(lookup)
        >>> # 240×240 crop extracts indices [136:376] from 512 reference
        >>> x_idx, y_idx = builder.get_position_coordinates((12, 240, 240), 10.0)
        >>> tokens = builder.build_tokens(image, label, resolution=10.0, ...)
    """
    
    # Reference grid sizes per resolution (large enough for any expected crop).
    # Unseen resolutions are auto-registered with default ref_size=512.
    REFERENCE_SIZES = {
        10.0: 512,  # Sentinel-2/Sentinel-1/EnMAP at 10m
    }
    
    # Default reference grid size for auto-registered resolutions
    DEFAULT_REF_SIZE = 512
    
    def __init__(self, lookup_table):
        """
        Initialize TokenBuilder with reference grid system.
        
        Args:
            lookup_table: Lookup_encoding instance
        """
        self.lookup = lookup_table
        
        # Pre-register reference grids
        self._register_reference_grids()
    
    def _register_reference_grids(self):
        """Pre-register reference grid sizes for all known resolutions."""
        print("[TokenBuilder] Registering reference grids:")
        for resolution, ref_size in sorted(self.REFERENCE_SIZES.items()):
            offset = self.lookup.get_or_register_modality(resolution, ref_size)
            print(f"  ({resolution}m, {ref_size}px) → offset {offset}")
    
    def _ensure_resolution_registered(self, resolution):
        """
        Ensure a resolution is registered in the reference grid system.
        
        Auto-registers unseen resolutions with DEFAULT_REF_SIZE.
        Called by get_position_coordinates and get_query_coordinates
        before accessing REFERENCE_SIZES.
        
        Args:
            resolution: float - GSD in meters/pixel
            
        Returns:
            ref_size: int - reference grid size for this resolution
        """
        if resolution not in self.REFERENCE_SIZES:
            ref_size = self.DEFAULT_REF_SIZE
            self.REFERENCE_SIZES[resolution] = ref_size
            offset = self.lookup.get_or_register_modality(resolution, ref_size)
            print(f"[TokenBuilder] Auto-registered resolution {resolution}m "
                  f"→ ref_size={ref_size}, offset={offset}")
        
        return self.REFERENCE_SIZES[resolution]
    
    # =========================================================================
    # COORDINATE GENERATION (Reference Grid Indexing)
    # =========================================================================
    
    def get_position_coordinates(self, image_shape, resolution):
        """
        Generate position coordinates by extracting from reference grid.
        
        All crops at a given resolution extract their coordinates from
        a shared reference grid, ensuring consistent coordinate space.
        
        Args:
            image_shape: (C, H, W) or (H, W) - actual crop dimensions
            resolution: float - GSD in meters/pixel
        
        Returns:
            x_indices: [C, H, W, 1] - x position indices from reference grid
            y_indices: [C, H, W, 1] - y position indices from reference grid
        
        Example:
            Reference grid: 512×512 (indices 0-511, center at 256)
            
            Crop 240×240:
            - Center: 256, Half: 120
            - Extract: [136:376, 136:376]
            - Indices: [136, 137, ..., 255, 256, 257, ..., 375]
            
            Crop 160×240 (edge crop):
            - Y: [176:336] (80 pixels each side of 256)
            - X: [136:376] (120 pixels each side of 256)
            - Both use indices from SAME reference grid
        """
        # Handle both (C, H, W) and (H, W)
        if len(image_shape) == 3:
            C, H, W = image_shape
        elif len(image_shape) == 2:
            H, W = image_shape
            C = 1
        else:
            raise ValueError(f"image_shape must be (C,H,W) or (H,W), got {image_shape}")
        
        # Auto-register if needed
        ref_size = self._ensure_resolution_registered(resolution)
        
        # Validate crop fits in reference
        if H > ref_size or W > ref_size:
            raise ValueError(
                f"Crop size ({H}×{W}) exceeds reference size ({ref_size}×{ref_size}) "
                f"at resolution {resolution}m. Increase REFERENCE_SIZES[{resolution}]."
            )
        
        # Get global offset for reference grid
        global_offset = self.lookup.get_or_register_modality(resolution, ref_size)
        
        # Calculate extraction window (centered in reference grid)
        ref_center = ref_size // 2
        half_h = H // 2
        half_w = W // 2
        
        # Extract indices from reference grid
        # For 240×240 from 512: [256-120:256+120] = [136:376]
        y_start = ref_center - half_h
        y_end = y_start + H  # Ensures exactly H elements
        x_start = ref_center - half_w
        x_end = x_start + W  # Ensures exactly W elements
        
        y_coords = torch.arange(y_start, y_end, dtype=torch.float32)
        x_coords = torch.arange(x_start, x_end, dtype=torch.float32)
        
        # Validate extracted window size
        assert len(y_coords) == H, f"y_coords length {len(y_coords)} != H {H}"
        assert len(x_coords) == W, f"x_coords length {len(x_coords)} != W {W}"
        
        # Create grids and add global offset
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")
        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset
        
        # Expand to [C, H, W, 1] format
        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)
        
        return x_indices, y_indices
    
    def get_query_coordinates(self, image_shape, resolution):
        """
        Generate query coordinate indices using reference grid.
        
        All crops at the same resolution share the same query offset
        (from the reference grid registration).
        
        Args:
            image_shape: (C, H, W) or (H, W) - image/crop dimensions
            resolution: float - GSD in meters/pixel
        
        Returns:
            query_indices: [C, H, W, 1] - query offset for all pixels
        
        Note:
            All pixels in a given resolution share the same query offset,
            regardless of crop size. This is a modality-level identifier.
        """
        # Handle both (C, H, W) and (H, W)
        if len(image_shape) == 3:
            C, H, W = image_shape
        elif len(image_shape) == 2:
            H, W = image_shape
            C = 1
        else:
            raise ValueError(f"image_shape must be (C,H,W) or (H,W), got {image_shape}")
        
        # Auto-register if needed
        ref_size = self._ensure_resolution_registered(resolution)
        
        global_offset = self.lookup.get_query_offset(resolution, ref_size)
        
        # All pixels get the same query offset
        return torch.full((C, H, W, 1), global_offset, dtype=torch.float32)
    
    # =========================================================================
    # HIGH-LEVEL TOKEN CONSTRUCTION
    # =========================================================================
    
    def build_tokens(
        self,
        image,
        label,
        resolution,
        spectral_indices,
        resolution_idx,
        time_idx=-1,
    ):
        """
        Build complete token array for image data.
        
        Token format: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
                       col 0    1  2       3          4        5            6            7
        
        Args:
            image: [C, H, W] - image data (reflectance/radiance values)
            label: [H, W] - per-pixel labels
            resolution: float - GSD in meters/pixel
            spectral_indices: [C] - spectral lookup indices for each band
            resolution_idx: int - resolution group index
            time_idx: int - temporal index (-1 for N/A)
        
        Returns:
            tokens: [C*H*W, 8] - flattened token array
        
        Example:
            >>> image = torch.randn(12, 240, 240)  # 12 bands
            >>> label = torch.randint(0, 15, (240, 240))
            >>> spectral_indices = torch.tensor([...])  # 12 wavelength indices
            >>> tokens = builder.build_tokens(
            ...     image, label, resolution=10.0,
            ...     spectral_indices=spectral_indices,
            ...     resolution_idx=0, time_idx=-1
            ... )
            >>> tokens.shape  # [12*240*240, 8] = [691200, 8]
        """
        C, H, W = image.shape
        
        # Get position and query coordinates (from reference grid)
        x_indices, y_indices = self.get_position_coordinates((C, H, W), resolution)
        query_indices = self.get_query_coordinates((C, H, W), resolution)
        
        # Expand spectral indices to [C*H*W]
        spectral_coords = spectral_indices.repeat_interleave(H * W)
        
        # Expand label to [C, H, W]
        label_expanded = label.unsqueeze(0).expand(C, -1, -1)
        
        # Create resolution and time columns
        resolution_col = torch.full((C, H, W, 1), resolution_idx, dtype=torch.float32)
        time_col = torch.full((C, H, W, 1), time_idx, dtype=torch.float32)
        
        # Concatenate all features
        # [value, x, y, spectral, label, query, resolution, time]
        tokens = torch.cat([
            image.unsqueeze(-1),                           # [C, H, W, 1] - col 0
            x_indices.float(),                              # [C, H, W, 1] - col 1
            y_indices.float(),                              # [C, H, W, 1] - col 2
            spectral_coords.view(C, H, W, 1).float(),      # [C, H, W, 1] - col 3
            label_expanded.unsqueeze(-1).float(),          # [C, H, W, 1] - col 4
            query_indices.float(),                          # [C, H, W, 1] - col 5
            resolution_col,                                 # [C, H, W, 1] - col 6
            time_col,                                       # [C, H, W, 1] - col 7
        ], dim=-1)
        
        # Flatten to [C*H*W, 8]
        return einops.rearrange(tokens, "c h w f -> (c h w) f")
    
    def build_queries(
        self,
        label,
        resolution,
        first_spectral_idx,
        resolution_idx,
        time_idx=-1,
    ):
        """
        Build query token array.
        
        Query tokens are used for reconstruction/prediction tasks. They have
        the same format as image tokens but with value=0 (to be predicted).
        
        Token format: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
                       col 0    1  2       3          4        5            6            7
        
        Args:
            label: [H, W] - per-pixel labels
            resolution: float - GSD in meters/pixel
            first_spectral_idx: int - spectral index to use (typically first band)
            resolution_idx: int - resolution group index
            time_idx: int - temporal index (-1 for N/A)
        
        Returns:
            queries: [H*W, 8] - query token array
        
        Example:
            >>> label = torch.randint(0, 15, (240, 240))
            >>> queries = builder.build_queries(
            ...     label, resolution=10.0,
            ...     first_spectral_idx=0,
            ...     resolution_idx=0, time_idx=-1
            ... )
            >>> queries.shape  # [240*240, 8] = [57600, 8]
        """
        H, W = label.shape
        
        # Get position and query coordinates (from reference grid)
        x_indices, y_indices = self.get_position_coordinates((1, H, W), resolution)
        query_indices = self.get_query_coordinates((1, H, W), resolution)
        
        # Build query tokens (value=0, single channel)
        queries = torch.cat([
            torch.zeros(1, H, W, 1),                                      # col 0 - value (to predict)
            x_indices.float(),                                            # col 1 - x position
            y_indices.float(),                                            # col 2 - y position
            torch.full((1, H, W, 1), first_spectral_idx, dtype=torch.float),  # col 3 - spectral
            label.unsqueeze(0).unsqueeze(-1).float(),                     # col 4 - label
            query_indices.float(),                                        # col 5 - query offset
            torch.full((1, H, W, 1), resolution_idx, dtype=torch.float),  # col 6 - resolution
            torch.full((1, H, W, 1), time_idx, dtype=torch.float),        # col 7 - time
        ], dim=-1)
        
        # Flatten to [H*W, 8]
        return einops.rearrange(queries, "c h w f -> (c h w) f")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def subsample_queries(queries, max_queries, ignore_index=255, prioritize_valid=True):
        """
        Subsample query tokens with optional prioritization of valid labels.
        
        Args:
            queries: [N, 8] - query token array
            max_queries: int - maximum number of queries to return
            ignore_index: int - label value to consider invalid (default: 255)
            prioritize_valid: bool - if True, prioritize valid labels (default: True)
        
        Returns:
            subsampled_queries: [max_queries, 8] - subsampled queries
        
        Strategy:
            If prioritize_valid=True:
            1. Select all valid labels if count <= max_queries
            2. Otherwise, randomly sample max_queries valid labels
            3. Fill remaining slots with invalid labels if needed
            
            If prioritize_valid=False:
            - Uniform random sampling
        """
        if queries.shape[0] <= max_queries:
            return queries
        
        if not prioritize_valid:
            perm = torch.randperm(queries.shape[0])[:max_queries]
            return queries[perm]
        
        # Prioritize valid labels
        query_labels = queries[:, 4]  # col 4 is label
        valid_mask = (query_labels != ignore_index)
        valid_indices = torch.where(valid_mask)[0]
        invalid_indices = torch.where(~valid_mask)[0]
        
        if len(valid_indices) >= max_queries:
            # Enough valid labels - sample from them
            perm = torch.randperm(len(valid_indices))[:max_queries]
            selected = valid_indices[perm]
        else:
            # Not enough valid - take all valid + some invalid
            n_invalid_needed = max_queries - len(valid_indices)
            n_invalid_needed = min(n_invalid_needed, len(invalid_indices))
            
            if n_invalid_needed > 0:
                invalid_perm = torch.randperm(len(invalid_indices))[:n_invalid_needed]
                selected = torch.cat([valid_indices, invalid_indices[invalid_perm]])
                # Shuffle combined set
                selected = selected[torch.randperm(len(selected))]
            else:
                selected = valid_indices
        
        return queries[selected]
    

    