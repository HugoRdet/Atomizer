"""
Geographic Pruning with Voronoi Cells and Padded Tensors

Memory-efficient, fully tensorized implementation.

All runtime parameters (geo_k, sigma, L_spatial) are passed as forward() args.
Precomputed cells are cached by token shape (C, H, W).
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
    
    Precomputed buffers (cached per shape):
        cell_indices_padded: [L, max_cell_size] - token indices per cell
        cell_valid_mask: [L, max_cell_size] - True for valid positions
        cell_distances: [L, max_cell_size] - squared distance to latent center
    """
    
    def __init__(
        self,
        geometry,
        chunk_size: int = 50,
    ):
        super().__init__()
        self.geometry = geometry
        self.chunk_size = chunk_size
        
        # Cache keyed by shape tuple
        self._precomputed_keys = set()
    
    # =========================================================================
    # Voronoi assignment
    # =========================================================================
    
    def _compute_nearest_latent_chunked(
        self,
        pixel_coords: torch.Tensor,
        latent_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign each token to its nearest latent via chunked distance computation.
        
        Returns:
            nearest_latent: [N] index of nearest latent per token
            min_dist_sq:    [N] squared distance to nearest latent
        """
        N = pixel_coords.shape[0]
        L = latent_coords.shape[0]
        device = pixel_coords.device
        original_dtype = pixel_coords.dtype
        
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
    
    # =========================================================================
    # Precomputation
    # =========================================================================
    
    def _precompute_voronoi_cells(
        self,
        tokens: torch.Tensor,
        latent_coords: torch.Tensor,
        L_spatial: int,
        cache_key: str,
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
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype
        
        print(f"[VoronoiPruning] Precomputing cells for {cache_key}...")
        print(f"  Tokens: {N}, Latents: {L}")
        
        with torch.no_grad():
            # Get coordinates
            pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)  # [N, 2]
            lat_coords = latent_coords[0]  # [L, 2]
            
            # Voronoi assignment
            cell_membership, dist_to_nearest = self._compute_nearest_latent_chunked(
                pixel_coords, lat_coords
            )
            
            # Count tokens per cell
            cell_counts = torch.bincount(cell_membership, minlength=L)
            max_cell_size = cell_counts.max().item()
            
            print(f"  Cell sizes: min={cell_counts.min().item()}, "
                  f"max={max_cell_size}, mean={cell_counts.float().mean().item():.0f}")
            
            # Build padded tensors with shuffled tokens per cell
            cell_indices_padded = torch.zeros(L, max_cell_size, dtype=torch.long, device=device)
            cell_valid_mask = torch.zeros(L, max_cell_size, dtype=torch.bool, device=device)
            cell_distances = torch.zeros(L, max_cell_size, dtype=dtype, device=device)
            
            for l in range(L):
                in_cell_mask = (cell_membership == l)
                in_cell_indices = torch.where(in_cell_mask)[0]
                in_cell_distances = dist_to_nearest[in_cell_mask]
                
                n_in_cell = in_cell_indices.shape[0]
                
                if n_in_cell > 0:
                    perm = torch.randperm(n_in_cell, device=device)
                    cell_indices_padded[l, :n_in_cell] = in_cell_indices[perm]
                    cell_valid_mask[l, :n_in_cell] = True
                    cell_distances[l, :n_in_cell] = in_cell_distances[perm]
            
            # Register buffers
            self.register_buffer(f"cell_indices_{cache_key}", cell_indices_padded)
            self.register_buffer(f"cell_valid_{cache_key}", cell_valid_mask)
            self.register_buffer(f"cell_distances_{cache_key}", cell_distances)
            self.register_buffer(f"cell_counts_{cache_key}", cell_counts)
            
            self._precomputed_keys.add(cache_key)
            
            del pixel_coords, lat_coords, cell_membership, dist_to_nearest
            gc.collect()
            torch.cuda.empty_cache()
    
    # =========================================================================
    # Forward
    # =========================================================================
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latent_coords: torch.Tensor,
        geo_k: int,
        sigma: float,
        L_spatial: int,
        hexagonal: bool = False,  # ← ADD THIS
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, N, D = tokens.shape
        k = geo_k
        device = tokens.device
        dtype = tokens.dtype
        
        # =========================================================================
        # Cache key includes hexagonal flag
        # =========================================================================
        grid_type = "hex" if hexagonal else "sq"
        cache_key = f"{N}_{L_spatial}_{grid_type}"
        
        # Precompute if needed
        if cache_key not in self._precomputed_keys:
            self._precompute_voronoi_cells(tokens, latent_coords, L_spatial, cache_key)
    
        
        # Retrieve cached tensors
        cell_indices = getattr(self, f"cell_indices_{cache_key}")      # [L, max_cell]
        cell_valid = getattr(self, f"cell_valid_{cache_key}")          # [L, max_cell]
        cell_distances = getattr(self, f"cell_distances_{cache_key}")  # [L, max_cell]
        cell_counts = getattr(self, f"cell_counts_{cache_key}")        # [L]
        
        max_cell_size = cell_indices.shape[1]
        
        # =====================================================================
        # TENSORIZED RANDOM SAMPLING
        # =====================================================================
        
        # Clamp k to max_cell_size (if fewer tokens exist, take all)
        k = min(k, max_cell_size)
        
        if self.training:
            # Random scores, invalid positions get -inf
            rand_scores = torch.rand(B, L_spatial, max_cell_size, device=device, dtype=dtype)
            valid_expanded = cell_valid.unsqueeze(0).expand(B, -1, -1)
            rand_scores = torch.where(
                valid_expanded, rand_scores,
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )
            
            # Sort descending → random permutation, take first k
            _, perm = torch.sort(rand_scores, dim=-1, descending=True)
            perm_k = perm[:, :, :k]
            
            # Gather token indices and distances
            indices_expanded = cell_indices.unsqueeze(0).expand(B, -1, -1)
            distances_expanded = cell_distances.unsqueeze(0).expand(B, -1, -1)
            
            selected_indices = torch.gather(indices_expanded, dim=2, index=perm_k)
            selected_dist_sq = torch.gather(distances_expanded, dim=2, index=perm_k)
        else:
            # Deterministic: first k tokens per cell
            selected_indices = cell_indices[:, :k].unsqueeze(0).expand(B, -1, -1).clone()
            selected_dist_sq = cell_distances[:, :k].unsqueeze(0).expand(B, -1, -1).clone()
        
        # =====================================================================
        # Handle cells with fewer than k tokens
        # =====================================================================
        min_cell_count = cell_counts.min().item()
        
        if min_cell_count < k:
            position_indices = torch.arange(k, device=device).unsqueeze(0)
            cell_counts_expanded = cell_counts.unsqueeze(1)
            selection_valid = (position_indices < cell_counts_expanded).unsqueeze(0).expand(B, -1, -1)
            
            selected_indices = torch.where(
                selection_valid, selected_indices, torch.zeros_like(selected_indices)
            )
        
        # =====================================================================
        # Gaussian bias
        # =====================================================================
        bias = -selected_dist_sq / (2 * (sigma ** 2))
        
        # =====================================================================
        # Gather tokens and masks
        # =====================================================================
        tokens_per_latent = self._gather_tokens(tokens, selected_indices)
        masks_per_latent = self._gather_masks(mask, selected_indices)
        
        if min_cell_count < k:
            masks_per_latent = masks_per_latent | (~selection_valid)
        
        return tokens_per_latent, masks_per_latent, bias.to(dtype)
    
    # =========================================================================
    # Gather helpers
    # =========================================================================
    
    def _gather_tokens(
        self,
        tokens: torch.Tensor,   # [B, N, D]
        indices: torch.Tensor,  # [B, L, k]
    ) -> torch.Tensor:
        B, L, k = indices.shape
        D = tokens.shape[-1]
        flat_indices = indices.reshape(B, L * k)
        flat_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tokens, dim=1, index=flat_exp).reshape(B, L, k, D)
    
    def _gather_masks(
        self,
        mask: torch.Tensor,     # [B, N]
        indices: torch.Tensor,  # [B, L, k]
    ) -> torch.Tensor:
        B, L, k = indices.shape
        flat_indices = indices.reshape(B, L * k)
        return torch.gather(mask, dim=1, index=flat_indices).reshape(B, L, k).bool()
    
    # =========================================================================
    # Cache management
    # =========================================================================
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear precomputed buffers."""
        keys_to_clear = [cache_key] if cache_key else list(self._precomputed_keys)
        
        for key in keys_to_clear:
            for prefix in ["cell_indices", "cell_valid", "cell_distances", "cell_counts"]:
                buffer_name = f"{prefix}_{key}"
                if hasattr(self, buffer_name):
                    delattr(self, buffer_name)
            self._precomputed_keys.discard(key)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def extra_repr(self) -> str:
        cached = ", ".join(self._precomputed_keys) if self._precomputed_keys else "none"
        return f"chunk_size={self.chunk_size}, cached=[{cached}]"