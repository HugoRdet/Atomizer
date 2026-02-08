"""
Geographic Pruning with Voronoi Cells and Padded Tensors

Memory-efficient, fully tensorized implementation.
"""

import torch
import torch.nn as nn
import gc
from typing import Tuple, Optional


class GeographicPruning(nn.Module):
    """
    Voronoi-based geographic pruning with padded tensor storage.
    
    Each token is assigned to its nearest latent (Voronoi cell).
    Tokens are stored in padded tensors for fully tensorized sampling.
    
    Precomputed buffers:
        cell_indices_padded: [L, max_cell_size] - token indices per cell
        cell_valid_mask: [L, max_cell_size] - True for valid positions
        cell_distances: [L, max_cell_size] - squared distance to latent center
    """
    
    def __init__(
        self,
        geometry,
        num_spatial_latents: int,
        geo_k: int = 1500,
        default_sigma: float = 0.5,
        chunk_size: int = 50,
    ):
        super().__init__()
        self.geometry = geometry
        self.num_spatial_latents = num_spatial_latents
        self.geo_k = geo_k
        self.default_sigma = default_sigma
        self.chunk_size = chunk_size
        
        self._precomputed_modalities = set()
    
    def _compute_nearest_latent_chunked(
        self,
        pixel_coords: torch.Tensor,
        latent_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Most stable: float64 + direct computation + double chunking.
        """
        N = pixel_coords.shape[0]
        L = latent_coords.shape[0]
        device = pixel_coords.device
        original_dtype = pixel_coords.dtype
        
        # Convert to float64
        pixel_f64 = pixel_coords.double()
        latent_f64 = latent_coords.double()
        
        nearest_latent = torch.empty(N, dtype=torch.long, device=device)
        min_dist_sq = torch.full((N,), float('inf'), dtype=torch.float64, device=device)
        
        tok_chunk_size = 50000
        lat_chunk_size = self.chunk_size
        
        for tok_start in range(0, N, tok_chunk_size):
            tok_end = min(tok_start + tok_chunk_size, N)
            tok_chunk = pixel_f64[tok_start:tok_end]
            n_tok = tok_end - tok_start
            
            tok_nearest = torch.empty(n_tok, dtype=torch.long, device=device)
            tok_min_dist = torch.full((n_tok,), float('inf'), dtype=torch.float64, device=device)
            
            for lat_start in range(0, L, lat_chunk_size):
                lat_end = min(lat_start + lat_chunk_size, L)
                lat_chunk = latent_f64[lat_start:lat_end]
                
                # Direct computation with float64
                diff = tok_chunk.unsqueeze(1) - lat_chunk.unsqueeze(0)
                dist_sq = (diff ** 2).sum(dim=-1)
                
                lat_min_dist, lat_min_idx = dist_sq.min(dim=-1)
                
                update_mask = lat_min_dist < tok_min_dist
                tok_nearest[update_mask] = lat_min_idx[update_mask] + lat_start
                tok_min_dist[update_mask] = lat_min_dist[update_mask]
                
                del diff, dist_sq
            
            nearest_latent[tok_start:tok_end] = tok_nearest
            min_dist_sq[tok_start:tok_end] = tok_min_dist
        
        return nearest_latent, min_dist_sq.to(original_dtype)
    
    def _precompute_voronoi_cells(
        self,
        tokens: torch.Tensor,
        latent_coords: torch.Tensor,
        id_modality: str,
    ):
        """
        Precompute Voronoi cell membership and build padded tensors.
        
        Steps:
        1. Assign each token to nearest latent (Voronoi)
        2. Group tokens by cell
        3. Shuffle tokens within each cell (uniform coverage)
        4. Pad to max_cell_size for tensorized operations
        """
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        dtype = tokens.dtype
        
        print(f"[VoronoiPruning] Precomputing cells for {id_modality}...")
        print(f"  Tokens: {N}, Latents: {L}, geo_k: {self.geo_k}")
        
        with torch.no_grad():
            # =================================================================
            # STEP 1: Get coordinates
            # =================================================================
            pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)  # [N, 2]
            lat_coords = latent_coords[0]  # [L, 2]
            
            # =================================================================
            # STEP 2: Compute Voronoi assignment
            # =================================================================
            cell_membership, dist_to_nearest = self._compute_nearest_latent_chunked(
                pixel_coords, lat_coords
            )  # [N], [N]
            
            # =================================================================
            # STEP 3: Count tokens per cell
            # =================================================================
            cell_counts = torch.bincount(cell_membership, minlength=L)  # [L]
            max_cell_size = cell_counts.max().item()
            
            print(f"  Cell sizes: min={cell_counts.min().item()}, "
                f"max={max_cell_size}, mean={cell_counts.float().mean().item():.0f}")
            
            # =================================================================
            # STEP 4: Build padded tensors with shuffled tokens per cell
            # =================================================================
            cell_indices_padded = torch.zeros(L, max_cell_size, dtype=torch.long, device=device)
            cell_valid_mask = torch.zeros(L, max_cell_size, dtype=torch.bool, device=device)
            cell_distances = torch.zeros(L, max_cell_size, dtype=dtype, device=device)
            
            for l in range(L):
                # Find tokens belonging to this cell
                in_cell_mask = (cell_membership == l)
                in_cell_indices = torch.where(in_cell_mask)[0]
                in_cell_distances = dist_to_nearest[in_cell_mask]
                
                n_in_cell = in_cell_indices.shape[0]
                
                if n_in_cell > 0:
                    # =========================================================
                    # SHUFFLE tokens within each cell for uniform coverage
                    # =========================================================
                    perm = torch.randperm(n_in_cell, device=device)
                    shuffled_indices = in_cell_indices[perm]
                    shuffled_distances = in_cell_distances[perm]
                    
                    # Fill padded tensors
                    cell_indices_padded[l, :n_in_cell] = shuffled_indices
                    cell_valid_mask[l, :n_in_cell] = True
                    cell_distances[l, :n_in_cell] = shuffled_distances
            
            # =================================================================
            # STEP 5: Register buffers
            # =================================================================
            self.register_buffer(f"cell_indices_{id_modality}", cell_indices_padded)
            self.register_buffer(f"cell_valid_{id_modality}", cell_valid_mask)
            self.register_buffer(f"cell_distances_{id_modality}", cell_distances)
            self.register_buffer(f"cell_counts_{id_modality}", cell_counts)
            
            self._precomputed_modalities.add(id_modality)
            
            # Cleanup
            del pixel_coords, lat_coords, cell_membership, dist_to_nearest
            gc.collect()
            torch.cuda.empty_cache()
            
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latent_coords: torch.Tensor,
        sigma: Optional[float] = None,
        id_modality: str = "default",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with tensorized random sampling.
        
        Args:
            tokens: [B, N, D] input tokens
            mask: [B, N] boolean mask (True = invalid)
            latent_coords: [B, L, 2] latent positions
            sigma: Gaussian bias sigma
            id_modality: Modality identifier for caching
            
        Returns:
            tokens_per_latent: [B, L, k, D]
            masks_per_latent: [B, L, k]
            bias: [B, L, k]
        """
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        k = self.geo_k
        device = tokens.device
        dtype = tokens.dtype
        
        # Precompute if needed
        if id_modality not in self._precomputed_modalities:
            self._precompute_voronoi_cells(tokens, latent_coords, id_modality)
        
        # Retrieve cached tensors
        cell_indices = getattr(self, f"cell_indices_{id_modality}")    # [L, max_cell]
        cell_valid = getattr(self, f"cell_valid_{id_modality}")        # [L, max_cell]
        cell_distances = getattr(self, f"cell_distances_{id_modality}")  # [L, max_cell]
        cell_counts = getattr(self, f"cell_counts_{id_modality}")      # [L]
        
        max_cell_size = cell_indices.shape[1]
        
        # =====================================================================
        # TENSORIZED RANDOM SAMPLING
        # =====================================================================
        
        if self.training:
            # -----------------------------------------------------------------
            # Step 1: Generate random scores [B, L, max_cell_size]
            # -----------------------------------------------------------------
            rand_scores = torch.rand(B, L, max_cell_size, device=device, dtype=dtype)
            
            # -----------------------------------------------------------------
            # Step 2: Mask invalid positions with -inf
            # -----------------------------------------------------------------
            # Expand cell_valid: [L, max_cell] -> [B, L, max_cell]
            valid_expanded = cell_valid.unsqueeze(0).expand(B, -1, -1)
            
            rand_scores = torch.where(
                valid_expanded,
                rand_scores,
                torch.tensor(float('-inf'), device=device, dtype=dtype)
            )
            
            # -----------------------------------------------------------------
            # Step 3: Sort descending to get random permutation
            # -----------------------------------------------------------------
            _, perm = torch.sort(rand_scores, dim=-1, descending=True)
            
            # -----------------------------------------------------------------
            # Step 4: Take first k positions
            # -----------------------------------------------------------------
            perm_k = perm[:, :, :k]  # [B, L, k]
            
            # -----------------------------------------------------------------
            # Step 5: Gather token indices and distances
            # -----------------------------------------------------------------
            # Expand cell_indices: [L, max_cell] -> [B, L, max_cell]
            indices_expanded = cell_indices.unsqueeze(0).expand(B, -1, -1)
            distances_expanded = cell_distances.unsqueeze(0).expand(B, -1, -1)
            
            selected_indices = torch.gather(indices_expanded, dim=2, index=perm_k)  # [B, L, k]
            selected_dist_sq = torch.gather(distances_expanded, dim=2, index=perm_k)  # [B, L, k]
            
        else:
            # -----------------------------------------------------------------
            # Deterministic: take first k tokens from each cell
            # -----------------------------------------------------------------
            selected_indices = cell_indices[:, :k].unsqueeze(0).expand(B, -1, -1).clone()
            selected_dist_sq = cell_distances[:, :k].unsqueeze(0).expand(B, -1, -1).clone()
        
        # =====================================================================
        # Handle cells with fewer than k tokens
        # =====================================================================
        # Check if any cell has fewer than k tokens
        min_cell_count = cell_counts.min().item()
        
        if min_cell_count < k:
            # Create mask for valid selections
            # Position j in cell l is valid if j < cell_counts[l]
            position_indices = torch.arange(k, device=device).unsqueeze(0)  # [1, k]
            cell_counts_expanded = cell_counts.unsqueeze(1)  # [L, 1]
            selection_valid = position_indices < cell_counts_expanded  # [L, k]
            selection_valid = selection_valid.unsqueeze(0).expand(B, -1, -1)  # [B, L, k]
            
            # For invalid selections, use index 0 (will be masked anyway)
            selected_indices = torch.where(selection_valid, selected_indices, 
                                          torch.zeros_like(selected_indices))
        
        # =====================================================================
        # Compute Gaussian bias
        # =====================================================================
        eff_sigma = sigma if sigma is not None else self.default_sigma
        bias = -selected_dist_sq / (2 * (eff_sigma ** 2))
        
        # =====================================================================
        # Gather tokens and masks
        # =====================================================================
        tokens_per_latent = self._gather_tokens(tokens, selected_indices)
        masks_per_latent = self._gather_masks(mask, selected_indices)
        
        # Mark selections from underfilled cells as masked
        if min_cell_count < k:
            masks_per_latent = masks_per_latent | (~selection_valid)
        
        return tokens_per_latent, masks_per_latent, bias.to(dtype)
    
    def _gather_tokens(
        self, 
        tokens: torch.Tensor,   # [B, N, D]
        indices: torch.Tensor,  # [B, L, k]
    ) -> torch.Tensor:
        """Gather tokens for each latent."""
        B, L, k = indices.shape
        D = tokens.shape[-1]
        
        # Flatten: [B, L, k] -> [B, L*k]
        flat_indices = indices.reshape(B, L * k)
        
        # Expand for feature dimension: [B, L*k, D]
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        
        # Gather and reshape
        gathered = torch.gather(tokens, dim=1, index=flat_indices_expanded)
        
        return gathered.reshape(B, L, k, D)
    
    def _gather_masks(
        self, 
        mask: torch.Tensor,     # [B, N]
        indices: torch.Tensor,  # [B, L, k]
    ) -> torch.Tensor:
        """Gather masks for each latent."""
        B, L, k = indices.shape
        
        # Flatten and gather
        flat_indices = indices.reshape(B, L * k)
        gathered = torch.gather(mask, dim=1, index=flat_indices)
        
        return gathered.reshape(B, L, k).bool()
    
    def clear_cache(self, id_modality: Optional[str] = None):
        """Clear precomputed buffers."""
        modalities_to_clear = [id_modality] if id_modality else list(self._precomputed_modalities)
        
        for mod in modalities_to_clear:
            for suffix in ["cell_indices", "cell_valid", "cell_distances", "cell_counts"]:
                buffer_name = f"{suffix}_{mod}"
                if hasattr(self, buffer_name):
                    delattr(self, buffer_name)
            self._precomputed_modalities.discard(mod)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def extra_repr(self) -> str:
        return (
            f"num_spatial_latents={self.num_spatial_latents}, "
            f"geo_k={self.geo_k}, "
            f"default_sigma={self.default_sigma:.2f}"
        )


# =============================================================================
# Factory function
# =============================================================================

def create_geographic_pruning(config: dict, geometry) -> GeographicPruning:
    """
    Factory function to create GeographicPruningVoronoi from config.
    """
    atomiser_cfg = config["Atomiser"]
    latents_per_row = atomiser_cfg["spatial_latents"]
    
    # Auto-calculate sigma based on latent spacing
    span = atomiser_cfg.get("latent_surface", 102.4)
    if latents_per_row > 1:
        spacing = span / (latents_per_row - 1)
    else:
        spacing = span
    
    return GeographicPruning(
        geometry=geometry,
        num_spatial_latents=latents_per_row ** 2,
        geo_k=atomiser_cfg.get("geo_k", 1500),
        default_sigma=atomiser_cfg.get("geo_sigma", spacing),
        chunk_size=atomiser_cfg.get("geo_pruning_chunk_size", 50),
    )