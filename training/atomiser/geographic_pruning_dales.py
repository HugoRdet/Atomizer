"""
Geographic Pruning with Voronoi Cells and Padded Tensors
(DALES variant: PER-SAMPLE Voronoi assignment, LOADED from offline
precomputation rather than computed on GPU)

==============================================================================
WHY THIS DIFFERS FROM THE ORIGINAL (raster-oriented) GeographicPruning
==============================================================================
The original implementation computes Voronoi cell membership from a SINGLE
sample (`tokens[0:1]`) and applies the resulting index list to every sample
in the batch. This is correct for fixed-grid rasters (FRACTAL's VHR, or any
dense H×W tensor), because token index `i` always refers to the SAME pixel
position across every sample of a given crop size — the raster's row-major
flatten order is identical for every sample.

It is NOT correct for point clouds. Token index `i` in one DALES patch's
padded token array and token index `i` in another patch's array do not
correspond to the same spatial location — points are read from different
.laz files in whatever order laspy returns them, then (possibly) randomly
subsampled. Applying one sample's Voronoi assignment to another sample's
token array would silently gather the WRONG (spatially mismatched) points.

==============================================================================
AUTHORITATIVE APPROACH: OFFLINE PRECOMPUTE, LOADED HERE (not computed)
==============================================================================
Per-token nearest-latent assignment is computed OFFLINE by
precompute_dales_latent_assignment.py — once per (patch, D4 grid variant)
combination — and stored as `.npz` sidecars next to each tiled patch. This
module NO LONGER computes Voronoi assignment via GPU distance calculations
for DALES; it only CONSUMES an already-computed per-token assignment array
passed in via `forward(..., token_latent_assignment=...)`.

Why not compute-and-cache on GPU instead (an earlier draft of this file):
that approach cached by `patch_id` alone, but a patch's assignment
depends on WHICH D4 augmentation was applied that call — caching by
`patch_id` alone would silently reuse a stale assignment computed under a
DIFFERENT rotation/flip on a later epoch. The offline precompute avoids
this by producing one assignment per (patch, D4 variant) pair up front,
and the caller (DalesDataset / collate) is responsible for selecting the
correct variant's assignment and passing it in per-sample, per-call —
no caching ambiguity possible since there's no runtime cache left to go
stale.

`forward(..., patch_ids=...)` (compute-and-cache-by-patch_id) is KEPT as a
fallback path for cases where a precomputed assignment isn't available
(e.g. exploratory notebooks, tests) — but the recommended /
production path for DALES training is `token_latent_assignment`.

DDP NOTE (unchanged from original): no persistent per-rank caching happens
in the new path at all (nothing to desync across ranks); the fallback
compute-and-cache path still stores plain dict entries (NOT buffers), same
reasoning as before.
"""

import torch
import torch.nn as nn
import gc
from typing import Tuple, Optional, List
from collections import OrderedDict
import zlib
import torch.distributed as dist


class GeographicPruning(nn.Module):
    """
    Voronoi-based geographic pruning with padded tensor storage.

    Three paths:
        1. Cached Voronoi (N <= ON_THE_FLY_THRESHOLD): shared-batch, for
           raster-uniform modalities (FRACTAL). Unchanged from original.
        2. On-the-fly top-k (N > ON_THE_FLY_THRESHOLD): shared-batch,
           unchanged from original.
        3. Precomputed per-sample assignment (NEW, AUTHORITATIVE for DALES):
           used when `token_latent_assignment` is passed to forward() —
           builds the padded per-latent cell structure directly from a
           given [B, N] nearest-latent-index tensor, no distance
           computation at all.
        4. Fallback per-sample compute-and-cache (kept for cases without a
           precomputed assignment): used when `patch_ids` is passed
           instead. NOTE the staleness caveat above — prefer path 3.

    Safety guarantees (all paths):
        - Empty Voronoi cells produce index 0 + mask=True (masked out)
        - All gather indices clamped to [0, N-1]
        - Bias set to -inf for invalid positions, 0 for valid positions
        - Precomputed tensors (fallback path only) stored as plain
          attributes/dict entries (NOT buffers) so DDP does not try to
          synchronize them across ranks.
    """

    MIN_CELL_SIZE = 1
    ON_THE_FLY_THRESHOLD = 100_000_000  # tokens above this skip precomputation

    def __init__(
        self,
        geometry,
        chunk_size: int = 10,
        max_cached_patches: int = 64,
    ):
        super().__init__()
        self.geometry = geometry
        self.chunk_size = chunk_size

        # ── Original shared-batch cache (unchanged) ─────────────────────
        self._precomputed_keys = set()

        # ── Fallback per-sample (patch_id-keyed) LRU cache ──────────────
        # Only used when the caller doesn't supply a precomputed
        # token_latent_assignment. See module docstring for the staleness
        # caveat that makes this a fallback, not the recommended path.
        self.max_cached_patches = max_cached_patches
        self._patch_cache: "OrderedDict[str, dict]" = OrderedDict()

    # =========================================================================
    # NEW (AUTHORITATIVE): build padded cell structure from a GIVEN assignment
    # =========================================================================

    def _build_cell_from_assignment(
        self,
        assignment: torch.Tensor,   # [N] long — nearest-latent index per token,
                                     # already selected for the correct D4
                                     # variant and gathered to match this
                                     # sample's current token order.
        L_spatial: int,
        device: torch.device,
    ) -> dict:
        """
        Build the padded per-latent cell structure (cell_indices, cell_valid,
        cell_counts) directly from a precomputed assignment array — no
        distance computation, no geometry lookup. This is the O(N) "build
        the Voronoi partition given the answer" step; the expensive part
        (nearest-latent search) already happened offline.
        """
        N = assignment.shape[0]
        L = L_spatial

        cell_counts = torch.bincount(assignment, minlength=L)
        max_cell_size = max(cell_counts.max().item(), self.MIN_CELL_SIZE)

        cell_indices_padded = torch.zeros(L, max_cell_size, dtype=torch.long, device=device)
        cell_valid_mask = torch.zeros(L, max_cell_size, dtype=torch.bool, device=device)

        for l in range(L):
            in_cell_indices = torch.where(assignment == l)[0]
            n_in_cell = in_cell_indices.shape[0]
            if n_in_cell > 0:
                perm = torch.randperm(n_in_cell, device=device)
                cell_indices_padded[l, :n_in_cell] = in_cell_indices[perm]
                cell_valid_mask[l, :n_in_cell] = True

        return {
            "cell_indices": cell_indices_padded,  # [L, max_cell_size]
            "cell_valid":   cell_valid_mask,       # [L, max_cell_size]
            "cell_counts":  cell_counts,             # [L]
        }

    def _forward_from_precomputed(
        self,
        tokens: torch.Tensor,                 # [B, N, D]
        mask: torch.Tensor,                   # [B, N]
        geo_k: int,
        L_spatial: int,
        token_latent_assignment: torch.Tensor,  # [B, N] long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype

        assert token_latent_assignment.shape == (B, N), (
            f"token_latent_assignment must be [B, N] = [{B}, {N}], "
            f"got {tuple(token_latent_assignment.shape)}"
        )

        per_sample_cells = [
            self._build_cell_from_assignment(
                token_latent_assignment[b].long(), L, device
            )
            for b in range(B)
        ]

        max_cell_size = max(c["cell_indices"].shape[1] for c in per_sample_cells)

        cell_indices_b = torch.zeros(B, L, max_cell_size, dtype=torch.long, device=device)
        cell_valid_b   = torch.zeros(B, L, max_cell_size, dtype=torch.bool, device=device)
        cell_counts_b  = torch.zeros(B, L, dtype=torch.long, device=device)

        for b, c in enumerate(per_sample_cells):
            m = c["cell_indices"].shape[1]
            cell_indices_b[b, :, :m] = c["cell_indices"]
            cell_valid_b[b, :, :m]   = c["cell_valid"]
            cell_counts_b[b]         = c["cell_counts"]

        k = min(geo_k, max_cell_size)
        position_indices = torch.arange(k, device=device).view(1, 1, k)
        selection_valid = (position_indices < cell_counts_b.unsqueeze(-1))  # [B,L,k]

        if self.training:
            rand_scores = torch.rand(B, L, max_cell_size, device=device, dtype=dtype)
            rand_scores = torch.where(
                cell_valid_b, rand_scores,
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )
            _, perm = torch.sort(rand_scores, dim=-1, descending=True)
            perm_k = perm[:, :, :k].clamp(0, max(max_cell_size - 1, 0))
            selected_indices = torch.gather(cell_indices_b, dim=2, index=perm_k)
        else:
            selected_indices = cell_indices_b[:, :, :k].clone()

        selected_indices = selected_indices.clamp(0, N - 1)
        selected_indices = torch.where(
            selection_valid, selected_indices, torch.zeros_like(selected_indices)
        )

        # Binary attention mask (0 valid, -inf invalid) — no distance-based
        # weighting, same convention as the rest of this module.
        bias = torch.zeros(B, L, k, device=device, dtype=dtype).masked_fill(
            ~selection_valid, float('-inf')
        )

        tokens_per_latent = self._gather_tokens(tokens, selected_indices, N)
        masks_per_latent = self._gather_masks(mask, selected_indices, N)
        masks_per_latent = masks_per_latent | (~selection_valid)

        return tokens_per_latent, masks_per_latent, bias.to(dtype)

    # =========================================================================
    # Cache key computation (shared-batch path, unchanged from original)
    # =========================================================================

    def _make_cache_key(
        self,
        N: int,
        L_spatial: int,
        hexagonal: bool,
        latent_coords: torch.Tensor,
    ) -> str:
        grid_type = "hex" if hexagonal else "sq"
        coords_flat = latent_coords[0].detach().float().cpu()
        coords_rounded = coords_flat.long()
        pos_hash = zlib.crc32(coords_rounded.numpy().tobytes()) % (10**8)
        return f"{N}_{L_spatial}_{grid_type}_{pos_hash}"

    # =========================================================================
    # Voronoi assignment (shared helper, used by both shared-batch and
    # per-sample paths)
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
    # FALLBACK per-sample precomputation (one scene, keyed by patch_id) —
    # only used when no precomputed token_latent_assignment is supplied.
    # See module docstring for the staleness caveat vs the offline path.
    # =========================================================================

    def _compute_one_sample_cell(
        self,
        tokens_b: torch.Tensor,   # [N, D] — single sample's tokens
        latent_coords_b: torch.Tensor,  # [L, 2] — this sample's latent positions
        L_spatial: int,
    ) -> dict:
        N = tokens_b.shape[0]
        L = L_spatial
        device = tokens_b.device
        dtype = tokens_b.dtype

        with torch.no_grad():
            pixel_coords = self.geometry.get_token_centers(
                tokens_b.unsqueeze(0)
            ).squeeze(0)  # [N, 2]

            cell_membership, dist_to_nearest = self._compute_nearest_latent_chunked(
                pixel_coords, latent_coords_b
            )

            cell_counts = torch.bincount(cell_membership, minlength=L)
            max_cell_size = max(cell_counts.max().item(), self.MIN_CELL_SIZE)

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

            del pixel_coords, cell_membership, dist_to_nearest

        return {
            "cell_indices":   cell_indices_padded,   # [L, max_cell_size]
            "cell_valid":     cell_valid_mask,        # [L, max_cell_size]
            "cell_distances": cell_distances,          # [L, max_cell_size]
            "cell_counts":    cell_counts,              # [L]
        }

    def _get_or_compute_patch_cell(
        self,
        patch_id: str,
        tokens_b: torch.Tensor,
        latent_coords_b: torch.Tensor,
        L_spatial: int,
    ) -> dict:
        """
        FALLBACK ONLY. LRU-cached per-scene Voronoi lookup keyed by
        patch_id alone — STALE across differing D4 augmentations of the
        same patch (see module docstring). Prefer
        forward(..., token_latent_assignment=...) instead.
        """
        if patch_id in self._patch_cache:
            self._patch_cache.move_to_end(patch_id)
            return self._patch_cache[patch_id]

        cell_data = self._compute_one_sample_cell(tokens_b, latent_coords_b, L_spatial)
        self._patch_cache[patch_id] = cell_data

        if len(self._patch_cache) > self.max_cached_patches:
            oldest_id, oldest_data = self._patch_cache.popitem(last=False)
            del oldest_data
            gc.collect()
            torch.cuda.empty_cache()

        return cell_data

    # =========================================================================
    # FALLBACK per-sample forward path (compute-and-cache by patch_id only)
    # =========================================================================

    def _forward_per_sample(
        self,
        tokens: torch.Tensor,       # [B, N, D]
        mask: torch.Tensor,         # [B, N]
        latent_coords: torch.Tensor,  # [B, L, 2]
        geo_k: int,
        sigma: float,
        L_spatial: int,
        patch_ids: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype
        assert len(patch_ids) == B, (
            f"patch_ids must have length B={B}, got {len(patch_ids)}"
        )

        per_sample_cells = []
        for b in range(B):
            cell_data = self._get_or_compute_patch_cell(
                patch_ids[b], tokens[b], latent_coords[b], L_spatial
            )
            per_sample_cells.append(cell_data)

        max_cell_size = max(c["cell_indices"].shape[1] for c in per_sample_cells)

        cell_indices_b = torch.zeros(B, L, max_cell_size, dtype=torch.long, device=device)
        cell_valid_b   = torch.zeros(B, L, max_cell_size, dtype=torch.bool, device=device)
        cell_distances_b = torch.zeros(B, L, max_cell_size, dtype=dtype, device=device)
        cell_counts_b  = torch.zeros(B, L, dtype=torch.long, device=device)

        for b, c in enumerate(per_sample_cells):
            m = c["cell_indices"].shape[1]
            cell_indices_b[b, :, :m]   = c["cell_indices"]
            cell_valid_b[b, :, :m]     = c["cell_valid"]
            cell_distances_b[b, :, :m] = c["cell_distances"].to(dtype)
            cell_counts_b[b]           = c["cell_counts"]

        k = min(geo_k, max_cell_size)
        position_indices = torch.arange(k, device=device).view(1, 1, k)
        selection_valid = (position_indices < cell_counts_b.unsqueeze(-1))

        if self.training:
            rand_scores = torch.rand(B, L, max_cell_size, device=device, dtype=dtype)
            rand_scores = torch.where(
                cell_valid_b, rand_scores,
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )
            _, perm = torch.sort(rand_scores, dim=-1, descending=True)
            perm_k = perm[:, :, :k].clamp(0, max(max_cell_size - 1, 0))

            selected_indices = torch.gather(cell_indices_b, dim=2, index=perm_k)
            selected_dist_sq = torch.gather(cell_distances_b, dim=2, index=perm_k)
        else:
            selected_indices = cell_indices_b[:, :, :k].clone()
            selected_dist_sq = cell_distances_b[:, :, :k].clone()

        selected_indices = selected_indices.clamp(0, N - 1)
        selected_indices = torch.where(
            selection_valid, selected_indices, torch.zeros_like(selected_indices)
        )

        bias = torch.zeros_like(selected_dist_sq).masked_fill(
            ~selection_valid, float('-inf')
        )

        tokens_per_latent = self._gather_tokens(tokens, selected_indices, N)
        masks_per_latent = self._gather_masks(mask, selected_indices, N)
        masks_per_latent = masks_per_latent | (~selection_valid)

        return tokens_per_latent, masks_per_latent, bias.to(dtype)

    # =========================================================================
    # ORIGINAL: shared-batch precomputation (unchanged, kept for FRACTAL /
    # raster-uniform modalities where sample-0 geometry IS valid for the
    # whole batch)
    # =========================================================================

    def _precompute_voronoi_cells(
        self,
        tokens: torch.Tensor,
        latent_coords: torch.Tensor,
        L_spatial: int,
        cache_key: str,
    ):
        B, N, D = tokens.shape
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype

        print(f"[VoronoiPruning] Precomputing cells for {cache_key}...")
        print(f"  Tokens: {N}, Latents: {L}")

        with torch.no_grad():
            pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)
            lat_coords = latent_coords[0]

            cell_membership, dist_to_nearest = self._compute_nearest_latent_chunked(
                pixel_coords, lat_coords
            )

            cell_counts = torch.bincount(cell_membership, minlength=L)

            empty_cells = (cell_counts == 0).sum().item()
            if empty_cells > 0:
                print(f"  WARNING: {empty_cells}/{L} latents have EMPTY Voronoi cells!")
                print(f"  These will be masked out during attention (bias=-inf).")

            max_cell_size = max(cell_counts.max().item(), self.MIN_CELL_SIZE)

            print(f"  Cell sizes: min={cell_counts.min().item()}, "
                  f"max={max_cell_size}, mean={cell_counts.float().mean().item():.0f}")

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

            setattr(self, f"cell_indices_{cache_key}",   cell_indices_padded)
            setattr(self, f"cell_valid_{cache_key}",     cell_valid_mask)
            setattr(self, f"cell_distances_{cache_key}", cell_distances)
            setattr(self, f"cell_counts_{cache_key}",    cell_counts)

            self._precomputed_keys.add(cache_key)

            del pixel_coords, lat_coords, cell_membership, dist_to_nearest
            gc.collect()
            torch.cuda.empty_cache()

    # =========================================================================
    # ORIGINAL: on-the-fly path (large N, shared-batch, unchanged)
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
        B, N, D = tokens.shape
        k = min(geo_k, N)
        L = L_spatial
        device = tokens.device
        dtype = tokens.dtype

        print(f"[GeoPruning] On-the-fly: N={N}, L={L}, k={k} ...")

        with torch.no_grad():
            pixel_coords = self.geometry.get_token_centers(tokens[0:1]).squeeze(0)
            lat_coords = latent_coords[0]

            selected_indices = torch.zeros(L, k, dtype=torch.long, device=device)
            selected_dist_sq = torch.zeros(L, k, dtype=dtype, device=device)
            selection_valid = torch.zeros(L, k, dtype=torch.bool, device=device)

            chunk = max(self.chunk_size, 1)
            for l_start in range(0, L, chunk):
                l_end = min(l_start + chunk, L)
                lat_chunk = lat_coords[l_start:l_end]

                diff = pixel_coords.unsqueeze(0) - lat_chunk.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1)
                del diff

                actual_k = min(k, N)

                topk_dist, topk_idx = torch.topk(
                    dist_sq, actual_k, dim=-1, largest=False
                )
                del dist_sq

                selected_indices[l_start:l_end, :actual_k] = topk_idx
                selected_dist_sq[l_start:l_end, :actual_k] = topk_dist
                selection_valid[l_start:l_end, :actual_k] = True

        selected_indices = selected_indices.unsqueeze(0).expand(B, -1, -1).contiguous()
        selected_dist_sq = selected_dist_sq.unsqueeze(0).expand(B, -1, -1).contiguous()
        selection_valid = selection_valid.unsqueeze(0).expand(B, -1, -1).contiguous()

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

        selected_indices = selected_indices.clamp(0, N - 1)
        selected_indices = torch.where(
            selection_valid, selected_indices, torch.zeros_like(selected_indices)
        )

        bias = torch.zeros_like(selected_dist_sq).masked_fill(
            ~selection_valid, float('-inf')
        )

        tokens_per_latent = self._gather_tokens(tokens, selected_indices, N)
        masks_per_latent = self._gather_masks(mask, selected_indices, N)
        masks_per_latent = masks_per_latent | (~selection_valid)

        print(f"[GeoPruning] On-the-fly done.")

        return tokens_per_latent, masks_per_latent, bias.to(dtype)

    # =========================================================================
    # Forward (routes to precomputed / fallback per-sample / shared-batch)
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
        patch_ids: Optional[List[str]] = None,
        token_latent_assignment: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N, D = tokens.shape

        # ── AUTHORITATIVE (DALES): precomputed per-token assignment ─────
        if token_latent_assignment is not None:
            return self._forward_from_precomputed(
                tokens, mask, geo_k, L_spatial, token_latent_assignment
            )

        # ── FALLBACK (DALES without precompute available): per-sample,
        # compute-and-cache by patch_id only (staleness caveat applies) ──
        if patch_ids is not None:
            return self._forward_per_sample(
                tokens, mask, latent_coords, geo_k, sigma, L_spatial, patch_ids
            )

        # ── ORIGINAL: shared-batch paths (raster-uniform modalities) ────
        if N > self.ON_THE_FLY_THRESHOLD:
            return self._forward_on_the_fly(
                tokens, mask, latent_coords, geo_k, sigma, L_spatial
            )

        k = geo_k
        device = tokens.device
        dtype = tokens.dtype

        cache_key = self._make_cache_key(N, L_spatial, hexagonal, latent_coords)

        if cache_key not in self._precomputed_keys:
            self._precompute_voronoi_cells(tokens, latent_coords, L_spatial, cache_key)

        cell_indices = getattr(self, f"cell_indices_{cache_key}")
        cell_valid = getattr(self, f"cell_valid_{cache_key}")
        cell_distances = getattr(self, f"cell_distances_{cache_key}")
        cell_counts = getattr(self, f"cell_counts_{cache_key}")

        max_cell_size = cell_indices.shape[1]
        k = min(k, max_cell_size)

        position_indices = torch.arange(k, device=device).unsqueeze(0)
        cell_counts_clamped = cell_counts.unsqueeze(1)
        selection_valid = (position_indices < cell_counts_clamped)
        selection_valid = selection_valid.unsqueeze(0).expand(B, -1, -1)

        if self.training:
            rand_scores = torch.rand(B, L_spatial, max_cell_size, device=device, dtype=dtype)
            valid_expanded = cell_valid.unsqueeze(0).expand(B, -1, -1)
            rand_scores = torch.where(
                valid_expanded, rand_scores,
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )

            _, perm = torch.sort(rand_scores, dim=-1, descending=True)
            perm_k = perm[:, :, :k]
            perm_k = perm_k.clamp(0, max(max_cell_size - 1, 0))

            indices_expanded = cell_indices.unsqueeze(0).expand(B, -1, -1)
            distances_expanded = cell_distances.unsqueeze(0).expand(B, -1, -1)

            selected_indices = torch.gather(indices_expanded, dim=2, index=perm_k)
            selected_dist_sq = torch.gather(distances_expanded, dim=2, index=perm_k)
        else:
            selected_indices = cell_indices[:, :k].unsqueeze(0).expand(B, -1, -1).clone()
            selected_dist_sq = cell_distances[:, :k].unsqueeze(0).expand(B, -1, -1).clone()

        selected_indices = selected_indices.clamp(0, N - 1)
        selected_indices = torch.where(
            selection_valid, selected_indices, torch.zeros_like(selected_indices)
        )

        bias = torch.zeros_like(selected_dist_sq).masked_fill(
            ~selection_valid, float('-inf')
        )

        tokens_per_latent = self._gather_tokens(tokens, selected_indices, N)
        masks_per_latent = self._gather_masks(mask, selected_indices, N)
        masks_per_latent = masks_per_latent | (~selection_valid)

        return tokens_per_latent, masks_per_latent, bias.to(dtype)

    # =========================================================================
    # Gather helpers (with bounds checking) — unchanged
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
        """Clear the ORIGINAL shared-batch cache (unchanged behavior)."""
        keys_to_clear = [cache_key] if cache_key else list(self._precomputed_keys)

        for key in keys_to_clear:
            for prefix in ["cell_indices", "cell_valid", "cell_distances", "cell_counts"]:
                attr_name = f"{prefix}_{key}"
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
            self._precomputed_keys.discard(key)

        gc.collect()
        torch.cuda.empty_cache()

    def clear_patch_cache(self, patch_id: Optional[str] = None):
        """Clear the FALLBACK per-sample (patch_id-keyed) cache."""
        if patch_id is not None:
            self._patch_cache.pop(patch_id, None)
        else:
            self._patch_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def extra_repr(self) -> str:
        cached = ", ".join(self._precomputed_keys) if self._precomputed_keys else "none"
        return (
            f"chunk_size={self.chunk_size}, "
            f"on_the_fly_threshold={self.ON_THE_FLY_THRESHOLD}, "
            f"shared_batch_cached=[{cached}], "
            f"fallback_per_sample_cache_size={len(self._patch_cache)}/{self.max_cached_patches}"
        )
