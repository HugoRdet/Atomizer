"""
Hybrid Self-Attention Module with k-NN + Global Context + Log-Distance RPE

FIXES from previous version:
1. Bug fix: Check both self_rpe and rpe for None before concatenating
2. New RPE: Log-distance encoding that handles all scales without aliasing
3. Self-attention: Learnable embedding instead of degenerate (0,0) encoding

Architecture:
- Spatial attention: Q=spatial, K/V=[self, k-NN spatial, ALL global]
- Global attention: Q=global, K/V=[ALL spatial, ALL global]
- Both use SINGLE softmax for proper competition

Log-Distance RPE:
- distance: log(d_ij + eps) → spreads all scales evenly
- direction: atan2(dy, dx) / pi → [-1, 1]
- No aliasing at small distances, no saturation at large distances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from einops import rearrange

from .nn_comp import FeedForward


# =============================================================================
# LOG-DISTANCE POSITION ENCODER
# =============================================================================

def fourier_encode(x: torch.Tensor, max_freq: int, num_bands: int) -> torch.Tensor:
    """
    Fourier feature encoding.
    
    Args:
        x: [...] input values (should be roughly in [-1, 1] or [0, 1] for best results)
        max_freq: maximum frequency
        num_bands: number of frequency bands
        
    Returns:
        encoded: [..., num_bands * 2 + 1] with [x, sin(f1*x), cos(f1*x), ...]
    """
    freqs = torch.linspace(1, max_freq, num_bands, device=x.device, dtype=x.dtype)
    x_expanded = x.unsqueeze(-1)  # [..., 1]
    angles = x_expanded * freqs * math.pi  # [..., num_bands]
    
    encoded = torch.cat([
        x.unsqueeze(-1),  # Original value
        torch.sin(angles),
        torch.cos(angles),
    ], dim=-1)
    
    return encoded


class LogDistanceRPE(nn.Module):
    """
    Log-Distance Relative Position Encoding.
    
    Encodes pairwise relationships as:
    - log_distance: log(d_ij + eps) normalized to reasonable range
    - direction: atan2(dy, dx) / pi in [-1, 1]
    
    Advantages over linear-scale RPE:
    - No aliasing at small distances (0.1m and 0.2m are distinguishable)
    - No saturation at large distances (4m and 8m are distinguishable)
    - Log-space naturally spreads all scales evenly
    
    For self-attention (i=j), uses a learnable embedding instead of
    degenerate (0, 0) which would give log(0) = -inf.
    """
    
    def __init__(
        self,
        num_bands: int = 32,
        max_freq: int = 32,
        log_scale: float = 1.0,  # Normalization: log(d) / log_scale
        min_dist: float = 0.01,  # Floor to avoid log(0)
    ):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.log_scale = log_scale
        self.min_dist = min_dist
        
        # Output dim: (log_dist + direction) each with (1 + 2*num_bands)
        self.per_component_dim = num_bands * 2 + 1
        self.out_dim = self.per_component_dim * 2
        
        # Learnable embedding for self-attention (i=j)
        # This replaces the degenerate (0, 0) case
        self.self_token_embedding = nn.Parameter(torch.randn(self.out_dim) * 0.02)
    
    def forward(
        self,
        delta_x: torch.Tensor,
        delta_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode relative positions.
        
        Args:
            delta_x: [...] x component of (pos_j - pos_i)
            delta_y: [...] y component of (pos_j - pos_i)
            
        Returns:
            encoding: [..., out_dim]
        """
        # Distance (with floor to handle d=0)
        dist = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        dist_floored = torch.clamp(dist, min=self.min_dist)
        
        # Log-distance, normalized
        log_dist = torch.log(dist_floored) / self.log_scale  # Roughly in [-5, 5] for typical ranges
        log_dist_normalized = log_dist / 5.0  # Roughly in [-1, 1]
        
        # Direction
        theta = torch.atan2(delta_y, delta_x)  # [-pi, pi]
        theta_normalized = theta / math.pi  # [-1, 1]
        
        # Fourier encode both
        dist_enc = fourier_encode(log_dist_normalized, self.max_freq, self.num_bands)
        theta_enc = fourier_encode(theta_normalized, self.max_freq, self.num_bands)
        
        return torch.cat([dist_enc, theta_enc], dim=-1)
    
    def get_self_embedding(self, batch_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Get self-attention embedding expanded to batch shape.
        
        Args:
            batch_shape: (B, L) or similar
            device: target device
            
        Returns:
            self_emb: [*batch_shape, 1, out_dim]
        """
        return self.self_token_embedding.view(1, 1, 1, -1).expand(*batch_shape, 1, -1).to(device)
    
    def get_output_dim(self) -> int:
        return self.out_dim


# =============================================================================
# LOCAL ATTENTION CACHE (UPDATED)
# =============================================================================

class LocalAttentionCache(nn.Module):
    """
    Computes k-NN, distances, and RPE for local self-attention.
    
    UPDATED: Uses LogDistanceRPE for better multi-scale handling.
    """
    
    def __init__(
        self,
        num_bands: int = 32,
        max_freq: int = 32,
        log_scale: float = 1.0,
    ):
        super().__init__()
        self.rpe_encoder = LogDistanceRPE(
            num_bands=num_bands,
            max_freq=max_freq,
            log_scale=log_scale,
        )
        self.pe_dim = self.rpe_encoder.get_output_dim()
    
    def forward(
        self, 
        positions: torch.Tensor, 
        k: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute k-nearest neighbors and RPE.
        
        Args:
            positions: [B, L, 2] latent positions in meters
            k: number of neighbors (excluding self)
            
        Returns:
            cache dict with topk_indices, rpe, self_rpe, distances
        """
        B, L, _ = positions.shape
        device = positions.device
        
        k = min(k, L - 1)
        
        # Find k-nearest neighbors (excluding self)
        with torch.no_grad():
            diff_all = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, L, L, 2]
            dist_sq_all = (diff_all ** 2).sum(dim=-1)  # [B, L, L]
            
            mask = torch.eye(L, dtype=torch.bool, device=device)
            dist_sq_all = dist_sq_all.masked_fill(mask.unsqueeze(0), float('inf'))
            
            _, topk_indices = torch.topk(dist_sq_all, k=k, dim=-1, largest=False)
        
        # Gather neighbor positions and compute delta
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        positions_exp = positions.unsqueeze(1).expand(-1, L, -1, -1)
        neighbor_pos = torch.gather(positions_exp, dim=2, index=idx_exp)  # [B, L, k, 2]
        
        delta = neighbor_pos - positions.unsqueeze(2)  # [B, L, k, 2]
        delta_x = delta[..., 0]  # [B, L, k]
        delta_y = delta[..., 1]  # [B, L, k]
        
        # Distances for Gaussian bias
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)  # [B, L, k]
        
        # RPE for neighbors using log-distance encoding
        rpe = self.rpe_encoder(delta_x, delta_y)  # [B, L, k, pe_dim]
        
        # Self-RPE: learnable embedding (not computed from (0, 0))
        self_rpe = self.rpe_encoder.get_self_embedding((B, L), device)  # [B, L, 1, pe_dim]
        
        return {
            'topk_indices': topk_indices,
            'rpe': rpe,
            'self_rpe': self_rpe,
            'distances': distances,
        }
    
    def get_output_dim(self) -> int:
        return self.pe_dim


# =============================================================================
# SPATIAL LOCAL SELF-ATTENTION (FIXED)
# =============================================================================

class SpatialLocalAttention(nn.Module):
    """
    Local self-attention for spatial latents.
    
    Context: [self, k-NN spatial neighbors, ALL global latents]
    
    FIXED: Proper None checks for RPE tensors.
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rpe: bool = True,
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
        
        # Learnable RPE for global latents (they don't have spatial positions)
        if use_rpe:
            self.global_rpe = nn.Parameter(torch.randn(pe_dim) * 0.02)
        
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
        
        # =====================================================================
        # Build context: [self, k neighbors, global]
        # =====================================================================
        
        # 1. Gather k-NN neighbors
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        spatial_exp = spatial.unsqueeze(1).expand(-1, L_s, -1, -1)
        neighbors = torch.gather(spatial_exp, dim=2, index=idx_exp)  # [B, L_s, k, D]
        
        # 2. Add self
        self_feat = spatial.unsqueeze(2)  # [B, L_s, 1, D]
        context = torch.cat([self_feat, neighbors], dim=2)  # [B, L_s, 1+k, D]
        
        # 3. Add global latents
        if G > 0:
            global_exp = global_latents.unsqueeze(1).expand(-1, L_s, -1, -1)  # [B, L_s, G, D]
            context = torch.cat([context, global_exp], dim=2)  # [B, L_s, 1+k+G, D]
        
        # =====================================================================
        # Build distances for Gaussian bias
        # =====================================================================
        self_dist = torch.zeros(B, L_s, 1, device=device, dtype=dtype)
        dist_cat = torch.cat([self_dist, distances], dim=2)  # [B, L_s, 1+k]
        if G > 0:
            global_dist = torch.full((B, L_s, G), float('inf'), device=device, dtype=dtype)
            dist_cat = torch.cat([dist_cat, global_dist], dim=2)  # [B, L_s, 1+k+G]
        
        # =====================================================================
        # Add RPE to context (FIXED: proper None checks)
        # =====================================================================
        if self.use_rpe:
            # Check that we have both RPE tensors
            if rpe is not None and self_rpe is not None:
                rpe_cat = torch.cat([self_rpe, rpe], dim=2)  # [B, L_s, 1+k, pe_dim]
                
                # Add learnable RPE for global latents
                if G > 0:
                    global_rpe = self.global_rpe.view(1, 1, 1, -1).expand(B, L_s, G, -1)
                    rpe_cat = torch.cat([rpe_cat, global_rpe], dim=2)  # [B, L_s, 1+k+G, pe_dim]
                
                context = torch.cat([context, rpe_cat], dim=-1)  # [B, L_s, 1+k+G, D+pe_dim]
            else:
                # RPE requested but not provided - pad with zeros
                # This shouldn't happen in normal operation but prevents crashes
                total_ctx = 1 + k + G
                zero_rpe = torch.zeros(B, L_s, total_ctx, self.pe_dim, device=device, dtype=dtype)
                context = torch.cat([context, zero_rpe], dim=-1)
        
        # =====================================================================
        # Attention computation
        # =====================================================================
        Q = self.to_q(spatial)  # [B, L_s, inner]
        K = self.to_k(context)  # [B, L_s, 1+k+G, inner]
        V = self.to_v(context)  # [B, L_s, 1+k+G, inner]
        
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=H)
        K = rearrange(K, 'b l c (h d) -> b h l c d', h=H)
        V = rearrange(V, 'b l c (h d) -> b h l c d', h=H)
        
        # Attention scores
        attn = torch.einsum('b h l d, b h l c d -> b h l c', Q, K) * self.scale
        
        # =====================================================================
        # Gaussian bias
        # =====================================================================
        if self.use_gaussian_bias and self.sigma is not None:
            sigma_sq = (self.sigma ** 2).view(1, H, 1, 1)
            
            # Spatial part (self + k neighbors)
            spatial_dist = dist_cat[:, :, :1+k]  # [B, L_s, 1+k]
            dist_sq = (spatial_dist ** 2).unsqueeze(1)  # [B, 1, L_s, 1+k]
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
# GLOBAL FULL ATTENTION
# =============================================================================

class GlobalFullAttention(nn.Module):
    """
    Full attention for global latents.
    
    Q: global [B, G, D]
    K/V: [ALL spatial, ALL global] = [B, L_s + G, D]
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
        
        if use_gaussian_bias:
            self.global_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.global_bias = None
    
    def forward(
        self,
        global_latents: torch.Tensor,
        spatial_latents: torch.Tensor,
    ) -> torch.Tensor:
        B, G, D = global_latents.shape
        L_s = spatial_latents.shape[1]
        H = self.heads
        
        context = torch.cat([spatial_latents, global_latents], dim=1)
        
        Q = self.to_q(global_latents)
        K = self.to_k(context)
        V = self.to_v(context)
        
        Q = rearrange(Q, 'b g (h d) -> b h g d', h=H)
        K = rearrange(K, 'b c (h d) -> b h c d', h=H)
        V = rearrange(V, 'b c (h d) -> b h c d', h=H)
        
        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        
        if self.use_gaussian_bias and self.global_bias is not None:
            attn = attn + self.global_bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        
        out = rearrange(out, 'b h g d -> b g (h d)')
        return self.to_out(out)


# =============================================================================
# HYBRID SELF-ATTENTION BLOCK
# =============================================================================

class HybridSelfAttentionBlock(nn.Module):
    """
    Complete self-attention block.
    
    1. Spatial local attention: Q=spatial, K/V=[self, k-NN, ALL global]
    2. Global full attention: Q=global, K/V=[ALL spatial, ALL global]
    3. FeedForward for both
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rpe: bool = True,
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        has_global: bool = True,
    ):
        super().__init__()
        self.has_global = has_global
        self.use_rpe = use_rpe
        
        # Spatial attention
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
        L_s = num_spatial
        
        spatial = latents[:, :L_s]
        global_ = latents[:, L_s:] if latents.shape[1] > L_s else None
        
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
        
        # Global attention
        if self.has_global and global_ is not None and global_.shape[1] > 0:
            global_ = global_ + self.global_attn(
                self.global_norm(global_),
                spatial,
            )
            global_ = global_ + self.global_ff(self.global_ff_norm(global_))
        
        if global_ is not None and global_.shape[1] > 0:
            return torch.cat([spatial, global_], dim=1)
        return spatial


# =============================================================================
# MAIN MODULE: HybridSelfAttention
# =============================================================================

class HybridSelfAttention(nn.Module):
    """
    Hybrid self-attention with log-distance RPE.
    
    Key features:
    1. Log-distance RPE: handles all scales without aliasing
    2. Learnable self-embedding: replaces degenerate (0,0) case
    3. k-NN spatial attention + full global attention
    
    Memory: O(L_s × (k + 1 + G) + G × (L_s + G)) instead of O((L_s + G)²)
    """
    
    def __init__(
        self,
        dim: int,
        k: int = 64,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rpe: bool = True,
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        num_blocks: int = 4,
        has_global: bool = True,
        share_weights: bool = False,
        # Log-distance RPE parameters
        rpe_num_bands: int = 32,
        rpe_max_freq: int = 32,
        rpe_log_scale: float = 1.0,
    ):
        super().__init__()
        
        self.k = k
        self.use_rpe = use_rpe
        self.use_gaussian_bias = use_gaussian_bias
        self.num_blocks = num_blocks
        
        # Cache module with log-distance RPE
        self.local_cache = LocalAttentionCache(
            num_bands=rpe_num_bands,
            max_freq=rpe_max_freq,
            log_scale=rpe_log_scale,
        )
        self.pe_dim = self.local_cache.get_output_dim()
        
        print(f"[HybridSelfAttention] Log-distance RPE dim: {self.pe_dim}")
        print(f"[HybridSelfAttention] use_rpe={use_rpe}, use_gaussian_bias={use_gaussian_bias}")
        
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
    ) -> Dict[str, torch.Tensor]:
        """Compute k-NN cache with log-distance RPE."""
        k = k or self.k
        return self.local_cache(positions, k)
    
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
    
    def get_sigma_stats(self) -> Optional[Dict[str, Any]]:
        """Get Gaussian sigma statistics."""
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

def create_hybrid_self_attention(config: dict) -> HybridSelfAttention:
    """Factory function to create HybridSelfAttention from config."""
    cfg = config.get("Atomiser", config)
    
    return HybridSelfAttention(
        dim=cfg.get("latent_dim", 512),
        k=cfg.get("self_attn_k", 64),
        heads=cfg.get("latent_heads", 8),
        dim_head=cfg.get("latent_dim_head", 64),
        ff_mult=cfg.get("ff_mult", 4),
        dropout=cfg.get("attn_dropout", 0.0),
        use_rpe=cfg.get("use_rpe", True),
        use_gaussian_bias=cfg.get("use_gaussian_bias", True),
        sigma_init=cfg.get("sigma_init", 3.0),
        learnable_sigma=cfg.get("learnable_sigma", True),
        num_blocks=cfg.get("self_per_cross_attn", 4),
        has_global=cfg.get("global_latents", 0) > 0,
        share_weights=cfg.get("weight_tie_layers", False),
        rpe_num_bands=cfg.get("rpe_num_bands", 32),
        rpe_max_freq=cfg.get("rpe_max_freq", 32),
        rpe_log_scale=cfg.get("rpe_log_scale", 1.0),
    )