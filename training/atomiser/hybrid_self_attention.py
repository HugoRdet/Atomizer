"""
Hybrid Self-Attention Module with k-NN + RPE + Gaussian Bias

Combines:
1. k-NN selection (bounds memory to O(Lk))
2. RPE from TokenProcessor's pos_encoder (same as cross-attention for consistency)
3. Gaussian bias (learnable per-head sharpness)
4. Bidirectional Cross-Attention between spatial and global latents

This gives you:
- Memory efficiency of k-NN
- Directional awareness of RPE (consistent with cross-attention)
- Multi-scale attention via learnable σ per head

Gaussian bias formula (same as full attention):
    bias = -dist² / (2σ²)
    
Only difference: applied to k neighbors instead of all L latents.

Usage:
    from training.utils.token_building.processor import TokenProcessor
    
    token_processor = TokenProcessor(config, lookup_table)
    
    hybrid_attn = HybridSelfAttention(
        dim=512,
        pos_encoder=token_processor.pos_encoder,  # Use same encoder!
        k=64,
        sigma_init=3.0,
        ...
    )
    
    # In encoder loop:
    cache = hybrid_attn.compute_cache(spatial_coords)
    latents = hybrid_attn(latents, cache, num_spatial=L_spatial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from einops import rearrange


# =============================================================================
# LOCAL ATTENTION CACHE (computes k-NN, distances, and RPE)
# =============================================================================

class LocalAttentionCache(nn.Module):
    """
    Computes everything needed for local self-attention:
    - k-nearest neighbor indices
    - Distances to neighbors (for Gaussian bias)
    - Relative position encoding (using TokenProcessor's pos_encoder)
    
    Uses the same PolarRelativeEncoder as cross-attention for consistency.
    Computed once per encoder layer, reused across all self-attention blocks.
    """
    
    def __init__(
        self,
        pos_encoder: nn.Module,
        latent_spacing: float = 3.0,
    ):
        """
        Args:
            pos_encoder: PolarRelativeEncoder or CartesianRelativeEncoder from TokenProcessor
            latent_spacing: Spacing between latents in meters (for adaptive physical_scale)
        """
        super().__init__()
        self.pos_encoder = pos_encoder
        self.latent_spacing = latent_spacing
        
        # Get output dim from the encoder (without GSD)
        self.pe_dim = pos_encoder.get_output_dim(include_gsd=False)
    
    def forward(
        self, 
        positions: torch.Tensor, 
        k: int,
        physical_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute k-nearest neighbors, RPE, and distances.
        
        Args:
            positions: [B, L, 2] latent positions in meters
            k: number of neighbors
            physical_scale: normalization for RPE (default: spacing * sqrt(k/π))
            
        Returns:
            cache dict with:
                - topk_indices: [B, L, k]
                - rpe: [B, L, k, pe_dim]
                - distances: [B, L, k] for Gaussian bias
        """
        B, L, _ = positions.shape
        device = positions.device
        
        k = min(k, L - 1)
        
        # Adaptive scale for RPE normalization
        # k-th neighbor at distance ≈ spacing * sqrt(k/π)
        if physical_scale is None:
            physical_scale = self.latent_spacing * math.sqrt(k / math.pi)
        
        # =====================================================================
        # 1. Find k-nearest neighbors (no gradient needed for indices)
        # =====================================================================
        with torch.no_grad():
            # diff_all[b, i, j] = positions[b, i] - positions[b, j]
            diff_all = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, L, L, 2]
            dist_sq_all = (diff_all ** 2).sum(dim=-1)  # [B, L, L]
            
            # Exclude self (set diagonal to inf)
            #mask = torch.eye(L, dtype=torch.bool, device=device)
            #dist_sq_all = dist_sq_all.masked_fill(mask.unsqueeze(0), float('inf'))
            
            # Get k nearest neighbors
            _, topk_indices = torch.topk(dist_sq_all, k=k, dim=-1, largest=False)
        
        # =====================================================================
        # 2. Gather neighbor positions and compute delta
        # =====================================================================
        # We want: neighbor_pos[b, i, n, :] = positions[b, topk_indices[b, i, n], :]
        # 
        # positions: [B, L, 2]
        # topk_indices: [B, L, k]
        #
        # CORRECT approach:
        # - positions.unsqueeze(1): [B, 1, L, 2]
        # - expand to [B, L, L, 2] where positions_exp[b, i, j, :] = positions[b, j, :]
        # - gather along dim=2 using topk_indices
        
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)  # [B, L, k, 2]
        positions_exp = positions.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, 2]
        neighbor_pos = torch.gather(positions_exp, dim=2, index=idx_exp)  # [B, L, k, 2]
        
        # Relative position: neighbor - self
        # self position: [B, L, 2] -> [B, L, 1, 2]
        delta = neighbor_pos - positions.unsqueeze(2)  # [B, L, k, 2]
        delta_x = delta[..., 0]  # [B, L, k]
        delta_y = delta[..., 1]  # [B, L, k]
        
        # =====================================================================
        # 3. Compute distances for Gaussian bias
        # =====================================================================
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)  # [B, L, k]
        
        # =====================================================================
        # 4. Compute RPE using the same encoder as cross-attention
        # =====================================================================
        # pos_encoder expects: delta_x, delta_y, physical_scale, gsd=None
        rpe = self.pos_encoder(delta_x, delta_y, physical_scale, gsd=None)  # [B, L, k, pe_dim]
        
        return {
            'topk_indices': topk_indices,
            'rpe': rpe,
            'distances': distances,
        }
    
    def get_output_dim(self) -> int:
        return self.pe_dim


# =============================================================================
# LOCAL SELF-ATTENTION WITH k-NN + RPE + GAUSSIAN BIAS
# =============================================================================

class LocalSelfAttentionWithGaussianBias(nn.Module):
    """
    Local self-attention combining:
    - k-NN selection (memory efficient)
    - RPE concatenation (directional information)
    - Gaussian bias (learnable per-head sharpness)
    
    This is the best of all worlds:
    - Memory: O(Lk) instead of O(L²)
    - Expressiveness: Full polar RPE with direction
    - Multi-scale: Learnable σ per head for adaptive receptive fields
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        
        # Q from latent only, K/V from latent + RPE
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim + pe_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim + pe_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Learnable sigma per head (in log space for numerical stability)
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma_init)))
        else:
            self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma_init)))
    
    @property
    def sigma(self):
        """Get sigma values (exponentiated from log space)."""
        return self.log_sigma.exp()
    
    def forward(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        rpe: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] spatial latent features
            topk_indices: [B, L, k] indices of k-nearest neighbors
            rpe: [B, L, k, pe_dim] relative positional encoding
            distances: [B, L, k] distances to neighbors (for Gaussian bias)
            
        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape
        k = topk_indices.shape[-1]
        H = self.heads
        
        # =====================================================================
        # Gather neighbor latents
        # =====================================================================
        # We want: neighbors[b, i, n, :] = x[b, topk_indices[b, i, n], :]
        #
        # x: [B, L, D]
        # topk_indices: [B, L, k]
        #
        # CORRECT approach:
        # - x.unsqueeze(1): [B, 1, L, D]
        # - expand to [B, L, L, D] where x_exp[b, i, j, :] = x[b, j, :]
        # - gather along dim=2 using topk_indices
        
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, L, k, D]
        x_exp = x.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, D]
        neighbors = torch.gather(x_exp, dim=2, index=idx_exp)  # [B, L, k, D]
        
        # Concatenate with RPE
        context = torch.cat([neighbors, rpe], dim=-1)  # [B, L, k, D + pe_dim]
        
        # Project
        Q = self.to_q(x)         # [B, L, inner_dim]
        K = self.to_k(context)   # [B, L, k, inner_dim]
        V = self.to_v(context)   # [B, L, k, inner_dim]
        
        # Multi-head reshape
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=H)
        K = rearrange(K, 'b l k (h d) -> b h l k d', h=H)
        V = rearrange(V, 'b l k (h d) -> b h l k d', h=H)
        
        # Content-based attention scores
        attn = torch.einsum('b h l d, b h l k d -> b h l k', Q, K) * self.scale
        
        # =====================================================================
        # Add Gaussian distance bias (learnable per-head sharpness)
        # =====================================================================
        # Gaussian bias: -dist² / (2σ²)
        # distances: [B, L, k] -> [B, 1, L, k]
        # sigma: [H] -> [1, H, 1, 1]
        sigma_sq = (self.sigma ** 2).view(1, H, 1, 1)  # [1, H, 1, 1]
        dist_sq = (distances ** 2).unsqueeze(1)  # [B, 1, L, k]
        
        gaussian_bias = -dist_sq / (2 * sigma_sq)  # [B, H, L, k]
        attn = attn + gaussian_bias
        
        # Softmax and aggregate
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h l k, b h l k d -> b h l d', attn, V)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)


# =============================================================================
# CROSS-ATTENTION (for spatial <-> global communication)
# =============================================================================

class CrossAttention(nn.Module):
    """Standard cross-attention for bidirectional spatial <-> global communication."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, D = query.shape
        H = self.heads
        
        Q = self.to_q(query)
        K = self.to_k(context)
        V = self.to_v(context)
        
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=H)
        K = rearrange(K, 'b m (h d) -> b h m d', h=H)
        V = rearrange(V, 'b m (h d) -> b h m d', h=H)
        
        attn = torch.einsum('b h n d, b h m d -> b h n m', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, V)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# =============================================================================
# FEEDFORWARD
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# HYBRID SELF-ATTENTION BLOCK
# =============================================================================

class HybridSelfAttentionBlock(nn.Module):
    """
    Complete self-attention block combining:
    1. Local Self-Attention for spatial latents (k-NN + RPE + Gaussian bias)
    2. Spatial → Global cross-attention
    3. Global → Spatial cross-attention
    4. FeedForward for both
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        has_global: bool = True,
    ):
        super().__init__()
        self.has_global = has_global
        
        # 1. Local self-attention for spatial latents (k-NN + RPE + Gaussian)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_local_attn = LocalSelfAttentionWithGaussianBias(
            dim=dim,
            pe_dim=pe_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma,
        )
        
        # 2. Spatial → Global cross-attention
        if has_global:
            self.spatial_cross_norm = nn.LayerNorm(dim)
            self.spatial_to_global = CrossAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )
            
            # 3. Global → Spatial cross-attention
            self.global_cross_norm = nn.LayerNorm(dim)
            self.global_to_spatial = CrossAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )
            
            # 4. FeedForward for global
            self.global_ff_norm = nn.LayerNorm(dim)
            self.global_ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        
        # 5. FeedForward for spatial
        self.spatial_ff_norm = nn.LayerNorm(dim)
        self.spatial_ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
    
    def forward(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, L_total, D] all latents (spatial + global)
            cache: dict with 'topk_indices', 'rpe', 'distances' from LocalAttentionCache
            num_spatial: number of spatial latents
            
        Returns:
            latents: [B, L_total, D] updated latents
        """
        L_spatial = num_spatial
        
        # Split spatial and global
        spatial = latents[:, :L_spatial]
        global_ = latents[:, L_spatial:] if latents.shape[1] > L_spatial else None
        
        # =====================================================================
        # 1. Local self-attention for spatial latents (k-NN + RPE + Gaussian)
        # =====================================================================
        spatial = spatial + self.spatial_local_attn(
            self.spatial_norm(spatial),
            cache['topk_indices'],
            cache['rpe'],
            cache['distances'],
        )
        
        # =====================================================================
        # 2 & 3. Bidirectional cross-attention (if global latents exist)
        # =====================================================================
        if self.has_global and global_ is not None and global_.shape[1] > 0:
            # Spatial reads from Global
            spatial = spatial + self.spatial_to_global(
                self.spatial_cross_norm(spatial),
                global_,
            )
            
            # Global reads from Spatial
            global_ = global_ + self.global_to_spatial(
                self.global_cross_norm(global_),
                spatial,
            )
            
            # FeedForward for global
            global_ = global_ + self.global_ff(self.global_ff_norm(global_))
        
        # =====================================================================
        # 4. FeedForward for spatial
        # =====================================================================
        spatial = spatial + self.spatial_ff(self.spatial_ff_norm(spatial))
        
        # =====================================================================
        # Recombine
        # =====================================================================
        if global_ is not None and global_.shape[1] > 0:
            return torch.cat([spatial, global_], dim=1)
        return spatial


# =============================================================================
# MAIN MODULE: HybridSelfAttention
# =============================================================================

class HybridSelfAttention(nn.Module):
    """
    Complete hybrid self-attention module.
    
    Features:
    - k-NN selection (memory efficient: O(Lk) instead of O(L²))
    - RPE from TokenProcessor's pos_encoder (consistent with cross-attention)
    - Gaussian bias (learnable per-head sharpness for multi-scale)
    - Bidirectional cross-attention with global latents
    
    Usage:
        from training.utils.token_building.processor import TokenProcessor
        
        token_processor = TokenProcessor(config, lookup_table)
        
        hybrid = HybridSelfAttention(
            dim=512,
            pos_encoder=token_processor.pos_encoder,
            k=64,
            sigma_init=3.0,
            ...
        )
        
        # In encoder layer:
        cache = hybrid.compute_cache(spatial_coords)
        latents = hybrid(latents, cache, num_spatial=L_spatial)
    """
    
    def __init__(
        self,
        dim: int,
        pos_encoder: nn.Module,
        k: int = 64,
        latent_spacing: float = 3.0,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        num_blocks: int = 4,
        has_global: bool = True,
        share_weights: bool = False,
    ):
        """
        Args:
            dim: Latent dimension
            pos_encoder: PolarRelativeEncoder from TokenProcessor (for consistency)
            k: Number of nearest neighbors for local attention
            latent_spacing: Spacing between latents in meters (for RPE normalization)
            heads: Number of attention heads
            dim_head: Dimension per head
            ff_mult: FeedForward hidden dimension multiplier
            dropout: Dropout rate
            sigma_init: Initial sigma for Gaussian bias (meters)
            learnable_sigma: Whether sigma is learnable
            num_blocks: Number of self-attention blocks (self_per_cross_attn)
            has_global: Whether global latents exist
            share_weights: Whether to share weights across blocks
        """
        super().__init__()
        
        self.k = k
        
        # Local attention cache using TokenProcessor's pos_encoder
        self.local_cache = LocalAttentionCache(
            pos_encoder=pos_encoder,
            latent_spacing=latent_spacing,
        )
        self.pe_dim = self.local_cache.get_output_dim()
        
        # Attention blocks
        if share_weights:
            block = HybridSelfAttentionBlock(
                dim=dim,
                pe_dim=self.pe_dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                sigma_init=sigma_init,
                learnable_sigma=learnable_sigma,
                has_global=has_global,
            )
            self.blocks = nn.ModuleList([block] * num_blocks)
        else:
            self.blocks = nn.ModuleList([
                HybridSelfAttentionBlock(
                    dim=dim,
                    pe_dim=self.pe_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    sigma_init=sigma_init,
                    learnable_sigma=learnable_sigma,
                    has_global=has_global,
                )
                for _ in range(num_blocks)
            ])
    
    def compute_cache(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None,
        physical_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RPE cache. Call once per encoder layer, reuse for all blocks.
        
        Args:
            positions: [B, L_spatial, 2] spatial latent positions in meters
            k: Override default k (optional)
            physical_scale: Override RPE normalization (optional)
            
        Returns:
            cache dict with topk_indices, rpe, distances
        """
        k = k or self.k
        return self.local_cache(positions, k, physical_scale)
    
    def forward(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through one or all blocks.
        
        Args:
            latents: [B, L_total, D] all latents (spatial + global)
            cache: from compute_cache()
            num_spatial: number of spatial latents
            block_idx: if provided, run only that block; else run all
            
        Returns:
            latents: [B, L_total, D] updated latents
        """
        if block_idx is not None:
            return self.blocks[block_idx](latents, cache, num_spatial)
        
        for block in self.blocks:
            latents = block(latents, cache, num_spatial)
        
        return latents
    
    def get_sigma_stats(self) -> Dict[str, Any]:
        """Get statistics about learned sigma values across all blocks."""
        all_sigmas = []
        for i, block in enumerate(self.blocks):
            sigma = block.spatial_local_attn.sigma.detach().cpu()
            all_sigmas.append(sigma)
        
        all_sigmas = torch.stack(all_sigmas)  # [num_blocks, heads]
        
        return {
            'per_block': all_sigmas.tolist(),
            'mean': all_sigmas.mean().item(),
            'min': all_sigmas.min().item(),
            'max': all_sigmas.max().item(),
            'std': all_sigmas.std().item(),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_hybrid_self_attention(
    config: dict,
    pos_encoder: nn.Module,
) -> HybridSelfAttention:
    """
    Factory function to create HybridSelfAttention from config.
    
    Args:
        config: dict with Atomiser config
        pos_encoder: TokenProcessor.pos_encoder
            
    Returns:
        HybridSelfAttention module
    """
    atomiser_config = config.get("Atomiser", config)
    
    return HybridSelfAttention(
        dim=atomiser_config.get("latent_dim", 512),
        pos_encoder=pos_encoder,
        k=atomiser_config.get("self_attn_k", 64),
        latent_spacing=atomiser_config.get("latent_spacing", 3.0),
        heads=atomiser_config.get("latent_heads", 8),
        dim_head=atomiser_config.get("latent_dim_head", 64),
        ff_mult=atomiser_config.get("ff_mult", 4),
        dropout=atomiser_config.get("attn_dropout", 0.0),
        sigma_init=atomiser_config.get("sigma_init", 3.0),
        learnable_sigma=atomiser_config.get("learnable_sigma", True),
        num_blocks=atomiser_config.get("self_per_cross_attn", 4),
        has_global=atomiser_config.get("global_latents", 0) > 0,
        share_weights=atomiser_config.get("weight_tie_layers", False),
    )


