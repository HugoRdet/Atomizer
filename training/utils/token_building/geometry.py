import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from tqdm import tqdm
import math


class SensorGeometry(nn.Module):
    """
    The Physics Engine (The Map).

    Responsibilities:
    1. Manage physical constants (GSD).
    2. Convert token indices to physical coordinates (meters).
    3. Provide geometry data for encoder biases.

    PERFORMANCE: All constants are pre-computed in __init__ to avoid
    repeated tensor creation during forward pass.

    NOTE: Latent grid generation lives in Atomiser._compute_latent_grid()
    which derives everything from runtime grid_config.
    """

    def __init__(self, config: Dict[str, Any], lookup_table: Any):
        super().__init__()
        self.config       = config
        self.lookup_table = lookup_table

        # Reference GSD (read by TokenProcessor for constant-GSD mode)
        self.default_gsd = config["Atomiser"].get("gsd", 10.0)

        # Pre-compute √2 constant
        self.register_buffer("_sqrt_2", torch.tensor(math.sqrt(2.0)))

        # Build lookup tables
        self._init_token_geometry_buffers()
        self._init_modality_buffers()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _init_token_geometry_buffers(self):
        """
        Create lookup tables for token index → physical coordinates.

        Buffer size is derived from lookup_table.next_position_offset,
        which tracks the exact number of slots allocated (one per pixel
        per registered modality). This avoids the previous bug where
        table key values (e.g. res_key=10000) were summed instead of
        table sizes (e.g. image_size=512).
        """
        max_global_index = self.lookup_table.next_position_offset  # exact slot count

        centers = torch.zeros(max_global_index, dtype=torch.float32)
        gsds    = torch.zeros(max_global_index, dtype=torch.float32)

        first_token_width = None

        for modality in tqdm(self.lookup_table.modalities, desc="Initializing Geometry"):
            resolution, image_size = modality
            pos_scaling = image_size * resolution

            modality_centers = torch.linspace(
                -pos_scaling / 2.0 + resolution / 2.0,
                 pos_scaling / 2.0 - resolution / 2.0,
                steps=image_size,
            )

            modality_key = (int(1000 * resolution), image_size)
            start_idx    = self.lookup_table.table[modality_key]

            centers[start_idx:start_idx + image_size] = modality_centers
            gsds   [start_idx:start_idx + image_size] = resolution

            if first_token_width is None and image_size > 1:
                first_token_width = (modality_centers[1] - modality_centers[0]).abs().item()

        self.register_buffer("token_centers_lookup", centers)
        self.register_buffer("token_gsd_lookup",     gsds)

        if first_token_width is None:
            first_token_width = 1.0
        self.register_buffer("_token_width",      torch.tensor(first_token_width))
        self.register_buffer("_half_token_width", torch.tensor(first_token_width / 2.0))

    def _init_modality_buffers(self):
        """Create modality ID → physical properties mapping."""
        query_offsets  = []
        modality_indices = []
        physical_scales  = []

        for mod_idx, modality in enumerate(self.lookup_table.modalities):
            resolution, image_size = modality
            modality_key  = (int(1000 * resolution), image_size)
            query_offset  = self.lookup_table.table_queries[modality_key]

            query_offsets.append(query_offset)
            modality_indices.append(mod_idx)

            physical_extent = image_size * resolution
            physical_scales.append(physical_extent)

        sorted_pairs = sorted(zip(query_offsets, modality_indices, physical_scales))

        self.register_buffer(
            "query_offsets",
            torch.tensor([p[0] for p in sorted_pairs], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "modality_indices",
            torch.tensor([p[1] for p in sorted_pairs], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "modality_scales",
            torch.tensor([physical_scales[p[1]] for p in sorted_pairs], dtype=torch.float32),
            persistent=False,
        )

    # =========================================================================
    # PUBLIC API — Token Coordinates
    # =========================================================================

    def get_token_centers(self, token_data: torch.Tensor) -> torch.Tensor:
        """Convert token x/y indices (cols 1, 2) to (x, y) in meters."""
        x_idx = token_data[..., 1].long()
        y_idx = token_data[..., 2].long()
        x_meters = self.token_centers_lookup[x_idx]
        y_meters = self.token_centers_lookup[y_idx]
        return torch.stack([x_meters, y_meters], dim=-1)

    def get_token_gsd(self, token_data: torch.Tensor) -> torch.Tensor:
        """Get GSD (m/px) for each token from col 1 (x index)."""
        x_idx = token_data[..., 1].long()
        return self.token_gsd_lookup[x_idx]

    def get_physical_scale(self, token_data: torch.Tensor) -> torch.Tensor:
        """Get physical normalization scale for an input batch."""
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
        Pre-computed constants for Gaussian integral computation.

        Returns:
            sqrt_2:        scalar tensor
            half_width:    scalar tensor (half token width in meters)
            token_centers: [N] tensor of all token center coordinates
        """
        return self._sqrt_2, self._half_token_width, self.token_centers_lookup

    def get_query_pixel_coords(
        self,
        query_tokens: torch.Tensor,
        image_size: int = 512,
    ) -> torch.Tensor:
        """Convert query token metadata into global pixel (x, y) coordinates."""
        meter_coords = self.get_token_centers(query_tokens)
        pixel_coords = self.meters_to_pixels(meter_coords, image_size=image_size, gsd=None)
        return pixel_coords

    # =========================================================================
    # PUBLIC API — Encoder Bias
    # =========================================================================

    def get_encoder_bias(
        self,
        token_data: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geometry data for encoder cross-attention.

        Returns:
            token_bias:  [B, L, m, 2, 2] — per-token pixel edge bounds (x, y)
            latent_bias: [B, L, 2]        — latent positions (must be provided)
        """
        x_idx = token_data[..., 1].long()
        y_idx = token_data[..., 2].long()

        x_center = self.token_centers_lookup[x_idx]
        y_center = self.token_centers_lookup[y_idx]
        gsd      = self.token_gsd_lookup[x_idx]

        half_gsd = gsd / 2.0
        x_edges  = torch.stack([x_center - half_gsd, x_center + half_gsd], dim=-1)
        y_edges  = torch.stack([y_center - half_gsd, y_center + half_gsd], dim=-1)
        token_bias = torch.stack([x_edges, y_edges], dim=-2)

        if latent_positions is None:
            raise ValueError(
                "latent_positions must be provided. "
                "Latent grid is computed at runtime by Atomiser._compute_latent_grid()."
            )

        return token_bias, latent_positions

    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================

    def meters_to_pixels(
        self,
        coords_meters: torch.Tensor,
        image_size: int = 512,
        gsd: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert coordinates from meters to pixel indices.
        Image is centered at origin, spanning [-extent/2, +extent/2] meters.
        """
        if gsd is None:
            gsd = self.default_gsd

        extent      = image_size * gsd
        half_extent = extent / 2.0

        coords_pixels = (coords_meters + half_extent) / gsd
        coords_pixels = coords_pixels.round().long()
        coords_pixels = coords_pixels.clamp(0, image_size - 1)
        return coords_pixels

    def pixels_to_meters(
        self,
        coords_pixels: torch.Tensor,
        image_size: int = 512,
        gsd: Optional[float] = None,
    ) -> torch.Tensor:
        """Convert pixel indices to meters."""
        if gsd is None:
            gsd = self.default_gsd

        extent      = image_size * gsd
        half_extent = extent / 2.0

        return coords_pixels.float() * gsd - half_extent

    def sample_grid_around_positions(
        self,
        coords_pixels: torch.Tensor,
        grid_size: int = 3,
        spacing: int = 2,
        image_size: int = 512,
    ) -> torch.Tensor:
        """Sample a regular grid of points around each position."""
        device    = coords_pixels.device
        half_grid = (grid_size - 1) // 2
        offsets_1d = torch.arange(-half_grid, half_grid + 1, device=device) * spacing

        grid_y, grid_x = torch.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        sample_coords = coords_pixels.unsqueeze(-2) + offsets
        sample_coords = sample_coords.clamp(0, image_size - 1)
        return sample_coords

    def extract_query_tokens_from_image(
        self,
        image_err: torch.Tensor,
        sample_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Extract query tokens from image tensor at given pixel positions."""
        B, C, H, W, metadata_dim = image_err.shape
        device = image_err.device

        original_shape      = sample_coords.shape[1:-1]
        N                   = sample_coords[..., 0].numel() // B
        sample_coords_flat  = sample_coords.view(B, N, 2)

        px_x = sample_coords_flat[..., 0].long().clamp(0, W - 1)
        px_y = sample_coords_flat[..., 1].long().clamp(0, H - 1)

        px_x_exp  = px_x.unsqueeze(-1).expand(-1, -1, C)
        px_y_exp  = px_y.unsqueeze(-1).expand(-1, -1, C)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, C)
        chan_idx  = torch.arange(C, device=device).view(1, 1, C).expand(B, N, -1)

        tokens       = image_err[batch_idx, chan_idx, px_y_exp, px_x_exp, :]
        query_tokens = tokens.view(B, N * C, metadata_dim)
        ground_truth = query_tokens[..., 0]

        return query_tokens, ground_truth, original_shape