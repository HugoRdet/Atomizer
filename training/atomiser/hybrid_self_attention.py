"""
Hybrid Self-Attention Module with k-NN + Global Context + Optional Gaussian Bias

Architecture matches original SelfAttentionWithGaussianBias:
- Spatial attention: Q=spatial, K/V=[self, k-NN spatial, ALL global]
- Global attention: Q=global, K/V=[ALL spatial, ALL global]
- Both use SINGLE softmax for proper competition between spatial and global

Key insight: In the original full L×L attention, all latents compete in the same
softmax. We preserve this for global latents (they see everything) and approximate
it for spatial latents (they see k-NN + all global).

Memory: O(L_spatial × (k + 1 + G) + G × (L_spatial + G)) instead of O((L_s + G)²)
For L_s=1225, k=64, G=128: ~280K pairs vs ~1.83M pairs (6.5x reduction)

Usage:
    hybrid_attn = HybridSelfAttention(
        dim=512,
        pos_encoder=token_processor.pos_encoder,
        k=64,
        use_gaussian_bias=True,
        sigma_init=3.0,
        has_global=True,
        ...
    )
    
    cache = hybrid_attn.compute_cache(spatial_coords)
    latents = hybrid_attn(latents, cache, num_spatial=L_spatial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from einops import rearrange

# Import existing components
from .nn_comp import FeedForward, CrossAttention


# =============================================================================
# LOCAL ATTENTION CACHE
# =============================================================================

class LocalAttentionCache(nn.Module):
    """
    Computes k-NN, distances, and RPE for local self-attention.
    Computed once per encoder layer, reused across all self-attention blocks.
    """
    
    def __init__(
        self,
        pos_encoder: nn.Module,
        latent_spacing: float = 3.0,
    ):
        super().__init__()
        self.pos_encoder = pos_encoder
        self.latent_spacing = latent_spacing
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
            k: number of neighbors (excluding self)
            physical_scale: normalization for RPE
            
        Returns:
            cache dict with topk_indices, rpe, self_rpe, distances
        """
        B, L, _ = positions.shape
        device = positions.device
        dtype = positions.dtype
        
        k = min(k, L - 1)
        
        if physical_scale is None:
            physical_scale = self.latent_spacing * math.sqrt(k / math.pi)
        
        # Find k-nearest neighbors (excluding self)
        with torch.no_grad():
            diff_all = positions.unsqueeze(2) - positions.unsqueeze(1)
            dist_sq_all = (diff_all ** 2).sum(dim=-1)
            
            mask = torch.eye(L, dtype=torch.bool, device=device)
            dist_sq_all = dist_sq_all.masked_fill(mask.unsqueeze(0), float('inf'))
            
            _, topk_indices = torch.topk(dist_sq_all, k=k, dim=-1, largest=False)
        
        # Gather neighbor positions and compute delta
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        positions_exp = positions.unsqueeze(1).expand(-1, L, -1, -1)
        neighbor_pos = torch.gather(positions_exp, dim=2, index=idx_exp)
        
        delta = neighbor_pos - positions.unsqueeze(2)
        delta_x = delta[..., 0]
        delta_y = delta[..., 1]
        
        # Distances for Gaussian bias
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        
        # RPE for neighbors
        rpe = self.pos_encoder(delta_x, delta_y, physical_scale, gsd=None)
        
        # Self-RPE (delta = 0, 0)
        self_delta_x = torch.zeros(B, L, 1, device=device, dtype=dtype)
        self_delta_y = torch.zeros(B, L, 1, device=device, dtype=dtype)
        self_rpe = self.pos_encoder(self_delta_x, self_delta_y, physical_scale, gsd=None)
        
        return {
            'topk_indices': topk_indices,
            'rpe': rpe,
            'self_rpe': self_rpe,
            'distances': distances,
        }
    
    def get_output_dim(self) -> int:
        return self.pe_dim


# =============================================================================
# SPATIAL LOCAL SELF-ATTENTION (Q=spatial, K/V=[self, k-NN, global])
# =============================================================================

class SpatialLocalAttention(nn.Module):
    """
    Local self-attention for spatial latents.
    
    Context: [self, k-NN spatial neighbors, ALL global latents]
    This matches the full L×L behavior where spatial sees global in same softmax.
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rpe: bool = False,
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rpe = use_rpe
        self.use_gaussian_bias = use_gaussian_bias
        self.pe_dim = pe_dim
        
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        
        kv_input_dim = dim + pe_dim if use_rpe else dim
        self.to_k = nn.Linear(kv_input_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_input_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        if use_gaussian_bias:
            if learnable_sigma:
                self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma_init)))
            else:
                self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma_init)))
            self.global_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.log_sigma = None
            self.global_bias = None
    
    @property
    def sigma(self):
        if self.log_sigma is None:
            return None
        return self.log_sigma.exp()
    
    def forward(
        self,
        spatial: torch.Tensor,
        topk_indices: torch.Tensor,
        rpe: Optional[torch.Tensor],
        self_rpe: Optional[torch.Tensor],
        distances: torch.Tensor,
        global_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial: [B, L_s, D]
            topk_indices: [B, L_s, k]
            rpe: [B, L_s, k, pe_dim] or None
            self_rpe: [B, L_s, 1, pe_dim] or None
            distances: [B, L_s, k]
            global_latents: [B, G, D] or None
        """
        B, L_s, D = spatial.shape
        k = topk_indices.shape[-1]
        H = self.heads
        device = spatial.device
        dtype = spatial.dtype
        
        G = global_latents.shape[1] if global_latents is not None else 0
        
        # Build context: [self, k neighbors, global]
        # 1. Gather k-NN neighbors
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        spatial_exp = spatial.unsqueeze(1).expand(-1, L_s, -1, -1)
        neighbors = torch.gather(spatial_exp, dim=2, index=idx_exp)  # [B, L_s, k, D]
        
        # 2. Add self
        self_feat = spatial.unsqueeze(2)  # [B, L_s, 1, D]
        context = torch.cat([self_feat, neighbors], dim=2)  # [B, L_s, 1+k, D]
        
        # 3. Add global latents
        if G > 0:
            global_exp = global_latents.unsqueeze(1).expand(-1, L_s, -1, -1)
            context = torch.cat([context, global_exp], dim=2)  # [B, L_s, 1+k+G, D]
        
        # Build distances: [self=0, neighbors, global=marker]
        self_dist = torch.zeros(B, L_s, 1, device=device, dtype=dtype)
        dist_cat = torch.cat([self_dist, distances], dim=2)  # [B, L_s, 1+k]
        if G > 0:
            global_dist = torch.full((B, L_s, G), float('inf'), device=device, dtype=dtype)
            dist_cat = torch.cat([dist_cat, global_dist], dim=2)
        
        # Optional: Add RPE to context
        if self.use_rpe and rpe is not None:
            rpe_cat = torch.cat([self_rpe, rpe], dim=2)  # [B, L_s, 1+k, pe_dim]
            if G > 0:
                global_rpe = torch.zeros(B, L_s, G, self.pe_dim, device=device, dtype=dtype)
                rpe_cat = torch.cat([rpe_cat, global_rpe], dim=2)
            context = torch.cat([context, rpe_cat], dim=-1)
        
        # Attention
        ctx_size = 1 + k + G
        
        Q = self.to_q(spatial)
        K = self.to_k(context)
        V = self.to_v(context)
        
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=H)
        K = rearrange(K, 'b l c (h d) -> b h l c d', h=H)
        V = rearrange(V, 'b l c (h d) -> b h l c d', h=H)
        
        attn = torch.einsum('b h l d, b h l c d -> b h l c', Q, K) * self.scale
        
        # Gaussian bias
        if self.use_gaussian_bias and self.sigma is not None:
            sigma_sq = (self.sigma ** 2).view(1, H, 1, 1)
            
            # Spatial part (self + k neighbors)
            spatial_dist = dist_cat[:, :, :1+k]
            dist_sq = (spatial_dist ** 2).unsqueeze(1)
            gaussian_bias = -dist_sq / (2 * sigma_sq)
            attn[:, :, :, :1+k] = attn[:, :, :, :1+k] + gaussian_bias
            
            # Global part
            if G > 0 and self.global_bias is not None:
                attn[:, :, :, 1+k:] = attn[:, :, :, 1+k:] + self.global_bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h l c, b h l c d -> b h l d', attn, V)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)


# =============================================================================
# GLOBAL FULL ATTENTION (Q=global, K/V=[ALL spatial, ALL global])
# =============================================================================

class GlobalFullAttention(nn.Module):
    """
    Full attention for global latents.
    
    Q: global [B, G, D]
    K/V: [ALL spatial, ALL global] = [B, L_s + G, D]
    
    This matches the original SelfAttentionWithGaussianBias where global
    latents see everything in a single softmax.
    
    Bias structure (matching original):
    - global → spatial: global_bias
    - global → global: global_bias
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_gaussian_bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_gaussian_bias = use_gaussian_bias
        
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Global latents get constant bias for everything they attend to
        # (matching original SelfAttentionWithGaussianBias)
        if use_gaussian_bias:
            self.global_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.global_bias = None
    
    def forward(
        self,
        global_latents: torch.Tensor,
        spatial_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_latents: [B, G, D]
            spatial_latents: [B, L_s, D]
            
        Returns:
            out: [B, G, D]
        """
        B, G, D = global_latents.shape
        L_s = spatial_latents.shape[1]
        H = self.heads
        
        # Context: [spatial, global]
        context = torch.cat([spatial_latents, global_latents], dim=1)  # [B, L_s + G, D]
        
        Q = self.to_q(global_latents)  # [B, G, inner]
        K = self.to_k(context)          # [B, L_s + G, inner]
        V = self.to_v(context)          # [B, L_s + G, inner]
        
        Q = rearrange(Q, 'b g (h d) -> b h g d', h=H)
        K = rearrange(K, 'b c (h d) -> b h c d', h=H)
        V = rearrange(V, 'b c (h d) -> b h c d', h=H)
        
        # Attention: [B, H, G, L_s + G]
        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        
        # Add global bias (constant for all positions, matching original)
        if self.use_gaussian_bias and self.global_bias is not None:
            attn = attn + self.global_bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        
        out = rearrange(out, 'b h g d -> b g (h d)')
        return self.to_out(out)


# =============================================================================
# HYBRID SELF-ATTENTION BLOCK (MERGED VERSION)
# =============================================================================

class HybridSelfAttentionBlock(nn.Module):
    """
    Complete self-attention block matching original SelfAttentionWithGaussianBias.
    
    Architecture:
    1. Spatial local attention: Q=spatial, K/V=[self, k-NN, ALL global]
    2. Global full attention: Q=global, K/V=[ALL spatial, ALL global]  <- MERGED!
    3. FeedForward for both
    
    Key insight: Both spatial and global use SINGLE softmax, allowing proper
    competition between attending to spatial vs global latents.
    
    Previous (wrong):
    - Global self-attn (softmax over G) + Global→Spatial cross-attn (softmax over L_s)
    - Two separate softmaxes = forced 100% attention to each
    
    Now (correct):
    - Global attention (softmax over L_s + G)
    - Single softmax = free to choose ratio
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rpe: bool = False,
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        has_global: bool = True,
    ):
        super().__init__()
        self.has_global = has_global
        
        # 1. Spatial local attention (with global in context)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = SpatialLocalAttention(
            dim=dim,
            pe_dim=pe_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            use_rpe=use_rpe,
            use_gaussian_bias=use_gaussian_bias,
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma,
        )
        self.spatial_ff_norm = nn.LayerNorm(dim)
        self.spatial_ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        
        if has_global:
            # 2. Global full attention (sees ALL spatial + ALL global in one softmax)
            self.global_norm = nn.LayerNorm(dim)
            self.global_attn = GlobalFullAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                use_gaussian_bias=use_gaussian_bias,
            )
            self.global_ff_norm = nn.LayerNorm(dim)
            self.global_ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
    
    def forward(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, L_total, D] = [spatial; global]
            cache: dict with topk_indices, rpe, self_rpe, distances
            num_spatial: L_s
            
        Returns:
            latents: [B, L_total, D]
        """
        L_s = num_spatial
        
        spatial = latents[:, :L_s]
        global_ = latents[:, L_s:] if latents.shape[1] > L_s else None
        
        # 1. Spatial attention (with global in context)
        spatial = spatial + self.spatial_attn(
            self.spatial_norm(spatial),
            cache['topk_indices'],
            cache.get('rpe'),
            cache.get('self_rpe'),
            cache['distances'],
            global_latents=global_,
        )
        spatial = spatial + self.spatial_ff(self.spatial_ff_norm(spatial))
        
        # 2. Global attention (sees ALL spatial + ALL global in ONE softmax)
        if self.has_global and global_ is not None and global_.shape[1] > 0:
            global_ = global_ + self.global_attn(
                self.global_norm(global_),
                spatial,  # Updated spatial from step 1
            )
            global_ = global_ + self.global_ff(self.global_ff_norm(global_))
        
        # Recombine
        if global_ is not None and global_.shape[1] > 0:
            return torch.cat([spatial, global_], dim=1)
        return spatial
    
    def forward_with_diagnostics(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with diagnostics."""
        L_s = num_spatial
        
        spatial = latents[:, :L_s]
        global_ = latents[:, L_s:] if latents.shape[1] > L_s else None
        
        spatial_input = spatial.clone()
        global_input = global_.clone() if global_ is not None else None
        
        # Spatial attention
        spatial = spatial + self.spatial_attn(
            self.spatial_norm(spatial),
            cache['topk_indices'],
            cache.get('rpe'),
            cache.get('self_rpe'),
            cache['distances'],
            global_latents=global_,
        )
        spatial = spatial + self.spatial_ff(self.spatial_ff_norm(spatial))
        
        spatial_change = (spatial - spatial_input).abs().mean().item()
        
        # Global attention
        global_change = 0.0
        if self.has_global and global_ is not None and global_.shape[1] > 0:
            global_ = global_ + self.global_attn(
                self.global_norm(global_),
                spatial,
            )
            global_ = global_ + self.global_ff(self.global_ff_norm(global_))
            global_change = (global_ - global_input).abs().mean().item()
        
        diagnostics = {
            'spatial_change': spatial_change,
            'global_change': global_change,
            'num_spatial': L_s,
            'num_global': global_.shape[1] if global_ is not None else 0,
        }
        
        if global_ is not None and global_.shape[1] > 0:
            return torch.cat([spatial, global_], dim=1), diagnostics
        return spatial, diagnostics


# =============================================================================
# MAIN MODULE: HybridSelfAttention
# =============================================================================

class HybridSelfAttention(nn.Module):
    """
    Hybrid self-attention matching original SelfAttentionWithGaussianBias.
    
    Architecture per block:
    1. Spatial: Q=spatial, K/V=[self, k-NN spatial, ALL global] - single softmax
    2. Global: Q=global, K/V=[ALL spatial, ALL global] - single softmax
    
    This is a TRUE approximation of full L×L attention:
    - Only approximation: spatial→spatial uses k-NN instead of all
    - Everything else is exact: global sees everything, spatial sees all global
    
    Memory comparison (L_s=1225, k=64, G=128):
    - Original: O((L_s + G)²) = 1.83M pairs
    - Hybrid: O(L_s × (k+1+G) + G × (L_s+G)) = 280K pairs
    - Reduction: 6.5x
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
        use_rpe: bool = False,
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        num_blocks: int = 4,
        has_global: bool = True,
        share_weights: bool = False,
    ):
        super().__init__()

        use_rpe=False
        
        self.k = k
        self.use_rpe = use_rpe
        self.use_gaussian_bias = use_gaussian_bias
        self.num_blocks = num_blocks
        
        # Cache for k-NN computation
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
                use_rpe=use_rpe,
                use_gaussian_bias=use_gaussian_bias,
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
                    use_rpe=use_rpe,
                    use_gaussian_bias=use_gaussian_bias,
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
        """Compute k-NN cache. Call once per encoder layer."""
        k = k or self.k
        return self.local_cache(positions, k, physical_scale)
    
    def forward(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward through one or all blocks."""
        if block_idx is not None:
            return self.blocks[block_idx](latents, cache, num_spatial)
        
        for block in self.blocks:
            latents = block(latents, cache, num_spatial)
        
        return latents
    
    def forward_with_diagnostics(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with diagnostics."""
        all_diag = {
            'per_block': [],
            'input_mean': latents.abs().mean().item(),
        }
        
        for i, block in enumerate(self.blocks):
            latents, block_diag = block.forward_with_diagnostics(latents, cache, num_spatial)
            block_diag['block_idx'] = i
            all_diag['per_block'].append(block_diag)
        
        all_diag['output_mean'] = latents.abs().mean().item()
        
        return latents, all_diag
    
    def get_sigma_stats(self) -> Optional[Dict[str, Any]]:
        """Get sigma statistics."""
        if not self.use_gaussian_bias:
            return None
        
        all_sigmas = []
        for block in self.blocks:
            sigma = block.spatial_attn.sigma
            if sigma is not None:
                all_sigmas.append(sigma.detach().cpu())
        
        if not all_sigmas:
            return None
        
        all_sigmas = torch.stack(all_sigmas)
        
        return {
            'per_block': all_sigmas.tolist(),
            'mean': all_sigmas.mean().item(),
            'min': all_sigmas.min().item(),
            'max': all_sigmas.max().item(),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_hybrid_self_attention(
    config: dict,
    pos_encoder: nn.Module,
) -> HybridSelfAttention:
    """Factory function to create HybridSelfAttention from config."""
    cfg = config.get("Atomiser", config)
    
    return HybridSelfAttention(
        dim=cfg.get("latent_dim", 512),
        pos_encoder=pos_encoder,
        k=cfg.get("self_attn_k", 64),
        latent_spacing=cfg.get("latent_spacing", 3.0),
        heads=cfg.get("latent_heads", 8),
        dim_head=cfg.get("latent_dim_head", 64),
        ff_mult=cfg.get("ff_mult", 4),
        dropout=cfg.get("attn_dropout", 0.0),
        use_rpe=cfg.get("use_rpe", False),
        use_gaussian_bias=cfg.get("use_gaussian_bias", True),
        sigma_init=cfg.get("sigma_init", 3.0),
        learnable_sigma=cfg.get("learnable_sigma", True),
        num_blocks=cfg.get("self_per_cross_attn", 4),
        has_global=cfg.get("global_latents", 0) > 0,
        share_weights=cfg.get("weight_tie_layers", False),
    )