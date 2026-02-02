import torch
import torch.nn as nn
import math
import gc
from typing import Tuple, Optional


class GeographicPruning(nn.Module):
    """
    Geographic pruning using Static Square Patch Sampling.
    Memory-efficient version that processes latents in chunks.
    
    Memory optimization:
    - Original: Creates [1, L, N, 2] tensor = 6.5 GB for L=625, N=1.3M
    - Optimized: Processes chunks of latents, peak ~500 MB
    """
    
    def __init__(
        self,
        geometry,
        num_spatial_latents: int,
        geo_k: int = 1500,
        default_sigma: float = 0.5,
        tokens_per_pixel: int = 5,
        chunk_size: int = 25,  # Process 25 latents at a time
    ):
        super().__init__()
        self.geometry = geometry
        self.num_spatial_latents = num_spatial_latents
        self.geo_k = geo_k
        self.default_sigma = default_sigma
        self.tokens_per_pixel = tokens_per_pixel
        self.chunk_size = chunk_size
        
        # Calculate the physical half-width of the square patch
        pixels_needed = geo_k / tokens_per_pixel
        self.patch_side_pixels = math.ceil(math.sqrt(pixels_needed))
        
        self._precomputed_modalities = set()

    def _get_static_square_patch_chunked(
        self, 
        tokens: torch.Tensor, 
        latent_coords: torch.Tensor, 
        id_modality: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient square patch computation.
        
        Processes latents in chunks to avoid creating massive intermediate tensors.
        Uses L-infinity distance (max of |Δx|, |Δy|) for square patches.
        
        Memory usage per chunk:
        - abs_dx, abs_dy: [N, chunk_size] each = 2 * N * chunk_size * 4 bytes
        - For N=1.3M, chunk_size=25: ~260 MB total (vs 6.5 GB original)
        """
        B, N, _ = tokens.shape
        L = self.num_spatial_latents
        k = self.geo_k
        
        buffer_indices = f"patch_idx_{id_modality}"
        buffer_dist = f"patch_dist_sq_{id_modality}"

        if id_modality not in self._precomputed_modalities:
            with torch.no_grad():
                device = tokens.device
                dtype = tokens.dtype
                
                # =============================================================
                # GET TOKEN AND LATENT COORDINATES
                # =============================================================
                # Token centers: [N, 2]
                pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)
                token_x = pixel_coords[:, 0]  # [N]
                token_y = pixel_coords[:, 1]  # [N]
                
                # Latent coords: [L, 2]
                lat_coords = latent_coords[0:1].squeeze(0)
                lat_x = lat_coords[:, 0]  # [L]
                lat_y = lat_coords[:, 1]  # [L]
                
                # =============================================================
                # PRE-ALLOCATE OUTPUT TENSORS
                # =============================================================
                all_indices = torch.empty(1, L, k, dtype=torch.long, device=device)
                all_dist_sq = torch.empty(1, L, k, dtype=dtype, device=device)
                
                # =============================================================
                # PROCESS LATENTS IN CHUNKS
                # =============================================================
                for lat_start in range(0, L, self.chunk_size):
                    lat_end = min(lat_start + self.chunk_size, L)
                    
                    # Chunk of latent coordinates: [chunk_L]
                    lat_x_chunk = lat_x[lat_start:lat_end]
                    lat_y_chunk = lat_y[lat_start:lat_end]
                    
                    # ---------------------------------------------------------
                    # Compute |Δx| and |Δy| separately (memory efficient!)
                    # [N, 1] - [1, chunk_L] -> [N, chunk_L]
                    # ---------------------------------------------------------
                    abs_dx = torch.abs(token_x.unsqueeze(1) - lat_x_chunk.unsqueeze(0))
                    abs_dy = torch.abs(token_y.unsqueeze(1) - lat_y_chunk.unsqueeze(0))
                    
                    # ---------------------------------------------------------
                    # L-infinity distance: max(|Δx|, |Δy|) defines square patch
                    # ---------------------------------------------------------
                    max_offset = torch.maximum(abs_dx, abs_dy)  # [N, chunk_L]
                    
                    # Transpose for per-latent top-k selection: [chunk_L, N]
                    max_offset_t = max_offset.t()
                    
                    # Top-k closest tokens for each latent in chunk
                    _, topk_idx = torch.topk(max_offset_t, k=k, dim=-1, largest=False)
                    
                    # ---------------------------------------------------------
                    # L2 distance for Gaussian bias: Δx² + Δy²
                    # ---------------------------------------------------------
                    dist_sq = abs_dx.pow(2) + abs_dy.pow(2)  # [N, chunk_L]
                    dist_sq_t = dist_sq.t()  # [chunk_L, N]
                    
                    # Gather distances for selected tokens
                    selected_dist = torch.gather(dist_sq_t, dim=-1, index=topk_idx)
                    
                    # ---------------------------------------------------------
                    # Store results
                    # ---------------------------------------------------------
                    all_indices[0, lat_start:lat_end] = topk_idx
                    all_dist_sq[0, lat_start:lat_end] = selected_dist
                    
                    # Free chunk memory immediately
                    del abs_dx, abs_dy, max_offset, max_offset_t
                    del dist_sq, dist_sq_t, topk_idx, selected_dist
                
                # =============================================================
                # CLEANUP AND REGISTER BUFFERS
                # =============================================================
                # Free coordinate tensors
                del pixel_coords, token_x, token_y, lat_coords, lat_x, lat_y
                
                # Register as buffers (persistent but not parameters)
                self.register_buffer(buffer_indices, all_indices)
                self.register_buffer(buffer_dist, all_dist_sq)
                self._precomputed_modalities.add(id_modality)
                
                # Force memory cleanup
                gc.collect()
                torch.cuda.empty_cache()

        return getattr(self, buffer_indices), getattr(self, buffer_dist)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latent_coords: torch.Tensor,
        sigma: Optional[float] = None,
        id_modality: str = "default"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: retrieve precomputed patches and gather tokens.
        
        Args:
            tokens: [B, N, D] input tokens
            mask: [B, N] boolean mask (True = invalid/masked)
            latent_coords: [B, L, 2] latent positions
            sigma: Optional sigma for Gaussian bias
            id_modality: Modality identifier for caching
            
        Returns:
            tokens_per_latent: [B, L, k, D] gathered tokens
            masks_per_latent: [B, L, k] gathered masks
            bias: [B, L, k] Gaussian spatial bias
        """
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        k = self.geo_k
        eff_sigma = sigma if sigma is not None else self.default_sigma
        
        # 1. Retrieve the fixed SQUARE patch indices [1, L, k]
        patch_indices, patch_dist_sq = self._get_static_square_patch_chunked(
            tokens, latent_coords, id_modality
        )

        # 2. Gaussian bias for spatial priority (central tokens get higher scores)
        selected_bias = -patch_dist_sq / (2 * (eff_sigma ** 2))

        # 3. Gather actual data
        tokens_per_latent = self._gather_tokens(tokens, patch_indices.expand(B, -1, -1))
        masks_per_latent = self._gather_masks(mask, patch_indices.expand(B, -1, -1))
        
        return tokens_per_latent, masks_per_latent, selected_bias.expand(B, -1, -1).to(tokens.dtype)

    def _gather_tokens(
        self, 
        tokens: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather tokens for each latent.
        
        Args:
            tokens: [B, N, D]
            indices: [B, L, k]
            
        Returns:
            [B, L, k, D]
        """
        B, L, k = indices.shape
        D = tokens.shape[-1]
        
        # Flatten indices for gather: [B, L*k]
        flat_indices = indices.view(B, L * k)
        # Expand for feature dimension: [B, L*k, D]
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        # Gather and reshape
        gathered = torch.gather(tokens, 1, flat_indices_exp)
        
        return gathered.view(B, L, k, D)

    def _gather_masks(
        self, 
        mask: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather masks for each latent.
        
        Args:
            mask: [B, N]
            indices: [B, L, k]
            
        Returns:
            [B, L, k]
        """
        B, L, k = indices.shape
        
        # Flatten indices and gather
        flat_indices = indices.view(B, L * k)
        gathered = torch.gather(mask, 1, flat_indices)
        
        return gathered.view(B, L, k).bool()

    def clear_cache(self, id_modality: Optional[str] = None):
        """
        Clear precomputed buffers to free memory.
        
        Args:
            id_modality: Specific modality to clear, or None to clear all.
        """
        if id_modality is not None:
            # Clear specific modality
            buffer_indices = f"patch_idx_{id_modality}"
            buffer_dist = f"patch_dist_sq_{id_modality}"
            
            if hasattr(self, buffer_indices):
                delattr(self, buffer_indices)
            if hasattr(self, buffer_dist):
                delattr(self, buffer_dist)
            
            self._precomputed_modalities.discard(id_modality)
        else:
            # Clear all modalities
            for mod in list(self._precomputed_modalities):
                buffer_indices = f"patch_idx_{mod}"
                buffer_dist = f"patch_dist_sq_{mod}"
                
                if hasattr(self, buffer_indices):
                    delattr(self, buffer_indices)
                if hasattr(self, buffer_dist):
                    delattr(self, buffer_dist)
            
            self._precomputed_modalities.clear()
        
        gc.collect()
        torch.cuda.empty_cache()

    def extra_repr(self) -> str:
        return (
            f"num_spatial_latents={self.num_spatial_latents}, "
            f"geo_k={self.geo_k}, "
            f"chunk_size={self.chunk_size}, "
            f"default_sigma={self.default_sigma:.2f}"
        )


def create_geographic_pruning(config: dict, geometry) -> GeographicPruning:
    """
    Factory function to create GeographicPruning from config.
    
    Config keys used:
        Atomiser.spatial_latents: Number of latents per row
        Atomiser.latent_surface: Physical span of latent grid (meters)
        Atomiser.geo_k: Number of tokens to keep per latent
        Atomiser.tokens_per_pixel: Expected tokens per pixel
        Atomiser.geo_pruning_chunk_size: Chunk size for memory-efficient computation
    """
    atom_cfg = config["Atomiser"]
    latents_per_row = atom_cfg["spatial_latents"]
    
    # Auto-calculate sigma based on latent spacing
    span = atom_cfg.get("latent_surface", 102.4)
    spacing = span / (latents_per_row - 1)
    
    return GeographicPruning(
        geometry=geometry,
        num_spatial_latents=latents_per_row ** 2,
        geo_k=atom_cfg.get("geo_k", 1500),
        default_sigma=spacing,
        tokens_per_pixel=atom_cfg.get("tokens_per_pixel", 5),
        chunk_size=atom_cfg.get("geo_pruning_chunk_size", 25),
    )