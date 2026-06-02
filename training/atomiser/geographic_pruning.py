"""
Geographic Pruning with Voronoi Cells and Padded Tensors

Memory-efficient, fully tensorized implementation.

All runtime parameters (geo_k, sigma, L_spatial) are passed as forward() args.
Precomputed cells are cached by token shape + latent positions hash.

For large token counts (>50k), an on-the-fly chunked top-k path is used
instead of precomputation to avoid combinatorial cache explosion from
variable-length inputs (e.g., FLAIR-HUB with temporal dropout).

Note: the bias returned by this module is a binary attention mask
(0 for valid positions, -inf for invalid). The `sigma` parameter is
retained in the forward signature for API compatibility but is no
longer used to weight in-cell positions.
"""

import torch
import torch.nn as nn
import gc
from typing import Tuple, Optional
import zlib  

class GeographicPruning(nn.Module):
    """
    Voronoi-based geographic pruning with padded tensor storage.

    Each token is assigned to its nearest latent (Voronoi cell).
    Tokens are stored in padded tensors for fully tensorized sampling.

    Two paths:
        1. Cached Voronoi (N <= ON_THE_FLY_THRESHOLD):
           Precomputes cell membership, caches padded tensors per (N, L, grid) key.
           Fast for repeated identical shapes (e.g., MMEarth).

        2. On-the-fly top-k (N > ON_THE_FLY_THRESHOLD):
           Chunked distance computation + top-k per latent.
           No caching, handles variable token counts efficiently.

    Safety guarantees:
        - Empty Voronoi cells produce index 0 + mask=True (masked out)
        - All gather indices clamped to [0, N-1]
        - Bias set to -inf for invalid positions, 0 for valid positions
        - Cache key includes latent position hash to prevent collisions
    """

    MIN_CELL_SIZE = 1
    ON_THE_FLY_THRESHOLD = 100_000_000  # tokens above this skip precomputation

    def __init__(
        self,
        geometry,
        chunk_size: int = 10,
    ):
        super().__init__()
        self.geometry = geometry
        self.chunk_size = chunk_size

        # Cache keyed by shape + latent position hash
        self._precomputed_keys = set()

    # =========================================================================
    # Cache key computation
    # =========================================================================

    def _make_cache_key(
        self,
        N: int,
        L_spatial: int,
        hexagonal: bool,
        latent_coords: torch.Tensor,
    ) -> str:
        """
        Build a cache key that includes latent positions.

        Prevents cache collisions when the same N/L/grid_type is used
        with different latent coordinates (e.g., callbacks vs training,
        or different grid configs).
        """
        grid_type = "hex" if hexagonal else "sq"

        # Hash latent positions — round to 0.1m to be robust to float noise
        coords_flat = latent_coords[0].detach().float().cpu()
        coords_rounded = coords_flat.long()
        pos_hash = zlib.crc32(coords_rounded.numpy().tobytes()) % (10**8)

        return f"{N}_{L_spatial}_{grid_type}_{pos_hash}"

    # =========================================================================
    # Voronoi assignment (for cached path)
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
    # Precomputation (cached path, small N only)
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

        Empty cells are safe: indices default to 0, valid mask is False,
        and downstream code masks them out with -inf bias.
        """
        B, N, D = tokens.shape
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype

        print(f"[VoronoiPruning] Precomputing cells for {cache_key}...")
        print(f"  Tokens: {N}, Latents: {L}")

        with torch.no_grad():
            # Get coordinates
            pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)
            lat_coords = latent_coords[0]

            # Voronoi assignment
            cell_membership, dist_to_nearest = self._compute_nearest_latent_chunked(
                pixel_coords, lat_coords
            )

            # Count tokens per cell
            cell_counts = torch.bincount(cell_membership, minlength=L)

            empty_cells = (cell_counts == 0).sum().item()
            if empty_cells > 0:
                print(f"  WARNING: {empty_cells}/{L} latents have EMPTY Voronoi cells!")
                print(f"  These will be masked out during attention (bias=-inf).")

            # max_cell_size must be >= 1 for safe tensor ops
            max_cell_size = max(cell_counts.max().item(), self.MIN_CELL_SIZE)

            print(f"  Cell sizes: min={cell_counts.min().item()}, "
                  f"max={max_cell_size}, mean={cell_counts.float().mean().item():.0f}")

            # Build padded tensors
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
                # Empty cells: indices=0, valid=False, distances=0
                # Index 0 is a safe fallback — the token will be masked out

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
    # On-the-fly path (large N, no caching)
    # =========================================================================

    def _forward_on_the_fly(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latent_coords: torch.Tensor,
        geo_k: int,
        sigma: float,
        L_spatial: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        On-the-fly geographic pruning for large token counts.

        Uses chunked top-k per latent instead of full Voronoi precomputation.
        No caching — handles variable token counts without memory explosion.

        Chunks over latents to keep peak memory bounded:
            peak ≈ chunk_size × N × sizeof(float)
        """
        B, N, D = tokens.shape
        k = min(geo_k, N)
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype

        print(f"[GeoPruning] On-the-fly: N={N}, L={L}, k={k} ...")

        with torch.no_grad():
            pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)  # [N, 2]
            lat_coords = latent_coords[0]  # [L, 2]

            selected_indices = torch.zeros(L, k, dtype=torch.long, device=device)
            selected_dist_sq = torch.zeros(L, k, dtype=dtype, device=device)
            selection_valid = torch.zeros(L, k, dtype=torch.bool, device=device)

            # Chunk over latents to bound memory at chunk_size × N
            chunk = max(self.chunk_size, 1)
            for l_start in range(0, L, chunk):
                l_end = min(l_start + chunk, L)
                lat_chunk = lat_coords[l_start:l_end]  # [c, 2]

                # [c, N] distance matrix
                diff = pixel_coords.unsqueeze(0) - lat_chunk.unsqueeze(1)  # [c, N, 2]
                dist_sq = (diff ** 2).sum(dim=-1)  # [c, N]
                del diff

                actual_k = min(k, N)

                # Top-k nearest tokens per latent in this chunk
                topk_dist, topk_idx = torch.topk(
                    dist_sq, actual_k, dim=-1, largest=False
                )
                del dist_sq

                selected_indices[l_start:l_end, :actual_k] = topk_idx
                selected_dist_sq[l_start:l_end, :actual_k] = topk_dist
                selection_valid[l_start:l_end, :actual_k] = True

        # Expand to batch dimension
        selected_indices = selected_indices.unsqueeze(0).expand(B, -1, -1).contiguous()
        selected_dist_sq = selected_dist_sq.unsqueeze(0).expand(B, -1, -1).contiguous()
        selection_valid = selection_valid.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Training: shuffle the k selected tokens per latent for stochastic sampling
        if self.training:
            rand_perm = torch.rand(B, L, k, device=device, dtype=dtype)
            rand_perm = torch.where(
                selection_valid, rand_perm,
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )
            _, perm = torch.sort(rand_perm, dim=-1, descending=True)

            selected_indices = torch.gather(selected_indices, dim=2, index=perm)
            selected_dist_sq = torch.gather(selected_dist_sq, dim=2, index=perm)
            selection_valid = torch.gather(selection_valid, dim=2, index=perm)

        # Safety: clamp indices
        selected_indices = selected_indices.clamp(0, N - 1)
        selected_indices = torch.where(
            selection_valid, selected_indices, torch.zeros_like(selected_indices)
        )

        # Binary attention mask: 0 for valid positions, -inf for invalid.
        # The `sigma` argument is kept in the signature for API compatibility
        # but is no longer used; in-cell positions are not distance-weighted.
        bias = torch.zeros_like(selected_dist_sq).masked_fill(
            ~selection_valid, float('-inf')
        )

        # Gather tokens and masks
        tokens_per_latent = self._gather_tokens(tokens, selected_indices, N)
        masks_per_latent = self._gather_masks(mask, selected_indices, N)

        # Force-mask invalid positions
        masks_per_latent = masks_per_latent | (~selection_valid)

        print(f"[GeoPruning] On-the-fly done.")

        return tokens_per_latent, masks_per_latent, bias.to(dtype)

    # =========================================================================
    # Forward (routes to cached or on-the-fly path)
    # =========================================================================

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latent_coords: torch.Tensor,
        geo_k: int,
        sigma: float,
        L_spatial: int,
        hexagonal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N, D = tokens.shape
        k = geo_k
        device = tokens.device
        dtype = tokens.dtype

        # =================================================================
        # Route: on-the-fly for large inputs, cached for small
        # =================================================================
        if N > self.ON_THE_FLY_THRESHOLD:
            return self._forward_on_the_fly(
                tokens, mask, latent_coords, geo_k, sigma, L_spatial
            )

        # =================================================================
        # Cached Voronoi path (original logic, for small N)
        # =================================================================

        
        
        cache_key = self._make_cache_key(N, L_spatial, hexagonal, latent_coords)

        if cache_key not in self._precomputed_keys:
            self._precompute_voronoi_cells(tokens, latent_coords, L_spatial, cache_key)

        # Retrieve cached tensors
        cell_indices = getattr(self, f"cell_indices_{cache_key}")
        cell_valid = getattr(self, f"cell_valid_{cache_key}")
        cell_distances = getattr(self, f"cell_distances_{cache_key}")
        cell_counts = getattr(self, f"cell_counts_{cache_key}")

        max_cell_size = cell_indices.shape[1]

        # =====================================================================
        # Clamp k to available tokens
        # =====================================================================
        k = min(k, max_cell_size)

        # =====================================================================
        # Build selection validity mask (ALWAYS, not just when min < k)
        # This is the key fix: always track which positions are valid
        # =====================================================================
        position_indices = torch.arange(k, device=device).unsqueeze(0)      # [1, k]
        cell_counts_clamped = cell_counts.unsqueeze(1)                       # [L, 1]
        selection_valid = (position_indices < cell_counts_clamped)            # [L, k]
        selection_valid = selection_valid.unsqueeze(0).expand(B, -1, -1)     # [B, L, k]

        # =====================================================================
        # TENSORIZED SAMPLING
        # =====================================================================

        if self.training:
            rand_scores = torch.rand(B, L_spatial, max_cell_size, device=device, dtype=dtype)
            valid_expanded = cell_valid.unsqueeze(0).expand(B, -1, -1)
            rand_scores = torch.where(
                valid_expanded, rand_scores,
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )

            _, perm = torch.sort(rand_scores, dim=-1, descending=True)
            perm_k = perm[:, :, :k]

            # Clamp to valid buffer range
            perm_k = perm_k.clamp(0, max(max_cell_size - 1, 0))

            indices_expanded = cell_indices.unsqueeze(0).expand(B, -1, -1)
            distances_expanded = cell_distances.unsqueeze(0).expand(B, -1, -1)

            selected_indices = torch.gather(indices_expanded, dim=2, index=perm_k)
            selected_dist_sq = torch.gather(distances_expanded, dim=2, index=perm_k)
        else:
            selected_indices = cell_indices[:, :k].unsqueeze(0).expand(B, -1, -1).clone()
            selected_dist_sq = cell_distances[:, :k].unsqueeze(0).expand(B, -1, -1).clone()

        # =====================================================================
        # Clamp ALL indices to valid token range [0, N-1]
        # This is the critical safety net for empty cells
        # =====================================================================
        selected_indices = selected_indices.clamp(0, N - 1)

        # Zero out invalid positions (redundant with clamp but explicit)
        selected_indices = torch.where(
            selection_valid, selected_indices, torch.zeros_like(selected_indices)
        )

        # =====================================================================
        # Binary attention mask: 0 for valid positions, -inf for invalid.
        # The `sigma` argument is kept in the signature for API compatibility
        # but is no longer used; in-cell positions are not distance-weighted.
        # =====================================================================
        bias = torch.zeros_like(selected_dist_sq).masked_fill(
            ~selection_valid, float('-inf')
        )

        # =====================================================================
        # Gather tokens and masks
        # =====================================================================
        tokens_per_latent = self._gather_tokens(tokens, selected_indices, N)
        masks_per_latent = self._gather_masks(mask, selected_indices, N)

        # Force-mask invalid positions
        masks_per_latent = masks_per_latent | (~selection_valid)

        return tokens_per_latent, masks_per_latent, bias.to(dtype)

    # =========================================================================
    # Gather helpers (with bounds checking)
    # =========================================================================

    def _gather_tokens(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        B, L, k = indices.shape
        D = tokens.shape[-1]

        indices = indices.clamp(0, N - 1)

        flat_indices = indices.reshape(B, L * k)
        flat_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tokens, dim=1, index=flat_exp).reshape(B, L, k, D)

    def _gather_masks(
        self,
        mask: torch.Tensor,
        indices: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        B, L, k = indices.shape

        indices = indices.clamp(0, N - 1)

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
        return (
            f"chunk_size={self.chunk_size}, "
            f"on_the_fly_threshold={self.ON_THE_FLY_THRESHOLD}, "
            f"cached=[{cached}]"
        )
