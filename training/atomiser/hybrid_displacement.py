"""
Hybrid Self-Attention with Attention-Guided Displacement

Key insight: Use attention patterns to guide latent displacement.
- Attention provides NON-ZERO base signal (softmax guarantee)
- MLP learns corrections/refinements
- Even if MLP collapses to zero, attention still provides reasonable displacement

Architecture matches SelfAttentionWithGaussianBias + learnable position updates.

Usage:
    hybrid = HybridSelfAttentionWithDisplacement(
        dim=512,
        pos_encoder=token_processor.pos_encoder,
        k=64,
        enable_displacement=True,
        max_displacement=5.0,
        ...
    )
    
    cache = hybrid.compute_cache(spatial_coords)
    latents, new_positions, disp_stats = hybrid(
        latents, cache, num_spatial, current_positions
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from einops import rearrange

from .nn_comp import FeedForward, CrossAttention


# =============================================================================
# ATTENTION-GUIDED DISPLACEMENT MODULE
# =============================================================================

class AttentionGuidedDisplacement(nn.Module):
    """
    Predicts displacement from attention patterns + MLP correction.
    
    Key properties:
    - Attention provides base signal that NEVER collapses to zero
    - MLP is initialized to zero, learns corrections over time
    - Learnable weights control attention vs MLP contribution
    
    Training dynamics:
    - Early: MLP ≈ 0, displacement ≈ attention-weighted center of mass
    - Late: MLP learns refinements, weights shift as needed
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_displacement: float = 5.0,
        mlp_hidden_ratio: float = 0.25,
        init_attn_weight: float = 1.0,
        init_mlp_weight: float = -2.0,
    ):
        """
        Args:
            dim: Latent dimension
            num_heads: Number of attention heads
            max_displacement: Maximum displacement in meters
            mlp_hidden_ratio: Hidden dim = dim * ratio
            init_attn_weight: Initial log-weight for attention (exp(1) ≈ 2.7)
            init_mlp_weight: Initial log-weight for MLP (exp(-2) ≈ 0.1)
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.max_displacement = max_displacement
        
        # === Head importance weights ===
        # Learn which heads contain positional vs semantic information
        self.head_weights = nn.Parameter(torch.zeros(num_heads))
        
        # === MLP correction network ===
        hidden_dim = int(dim * mlp_hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )
        # Initialize MLP to output near-zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
        # === Learnable balance between attention and MLP ===
        # Using log-space for numerical stability
        self.log_attn_weight = nn.Parameter(torch.tensor(init_attn_weight))
        self.log_mlp_weight = nn.Parameter(torch.tensor(init_mlp_weight))
    
    def forward(
        self,
        latents: torch.Tensor,           # [B, L_s, D]
        attn_weights: torch.Tensor,      # [B, H, L_s, 1+k+G]
        neighbor_positions: torch.Tensor, # [B, L_s, k, 2]
        current_positions: torch.Tensor,  # [B, L_s, 2]
        k: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute displacement for each spatial latent.
        
        Args:
            latents: Spatial latent features
            attn_weights: Attention weights from spatial local attention
            neighbor_positions: Positions of k-NN neighbors
            current_positions: Current latent positions
            k: Number of neighbors
            
        Returns:
            displacement: [B, L_s, 2] displacement in meters
            stats: Dictionary with diagnostic information
        """
        B, L_s, D = latents.shape
        device = latents.device
        
        # =====================================================================
        # 1. Attention-based displacement (NEVER zero due to softmax)
        # =====================================================================
        
        # Extract attention to spatial neighbors only (skip self at 0, skip global at end)
        neighbor_attn = attn_weights[:, :, :, 1:1+k]  # [B, H, L_s, k]
        
        # Compute head importance (softmax ensures all heads contribute)
        head_importance = F.softmax(self.head_weights, dim=0)  # [H]
        
        # Weighted average across heads
        # [H] x [B, H, L_s, k] -> [B, L_s, k]
        weighted_attn = torch.einsum('h, b h l k -> b l k', head_importance, neighbor_attn)
        
        # Normalize to sum to 1 (proper probability distribution over neighbors)
        weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute weighted center of mass
        # [B, L_s, k] x [B, L_s, k, 2] -> [B, L_s, 2]
        weighted_center = torch.einsum('b l k, b l k d -> b l d', weighted_attn, neighbor_positions)
        
        # Attention-based displacement: move towards weighted center
        attn_displacement = weighted_center - current_positions  # [B, L_s, 2]
        
        # =====================================================================
        # 2. MLP correction (can be zero, that's OK)
        # =====================================================================
        
        mlp_displacement = self.mlp(latents)  # [B, L_s, 2]
        
        # =====================================================================
        # 3. Combine with learnable weights
        # =====================================================================
        
        w_attn = self.log_attn_weight.exp()
        w_mlp = self.log_mlp_weight.exp()
        
        # Weighted combination (normalized)
        total_weight = w_attn + w_mlp + 1e-8
        combined = (w_attn * attn_displacement + w_mlp * mlp_displacement) / total_weight
        
        # =====================================================================
        # 4. Constrain magnitude
        # =====================================================================
        
        # Soft constraint using tanh
        displacement = torch.tanh(combined / self.max_displacement) * self.max_displacement
        
        # =====================================================================
        # 5. Compute diagnostics
        # =====================================================================
        
        with torch.no_grad():
            stats = {
                'attn_disp_magnitude': attn_displacement.norm(dim=-1).mean().item(),
                'mlp_disp_magnitude': mlp_displacement.norm(dim=-1).mean().item(),
                'final_disp_magnitude': displacement.norm(dim=-1).mean().item(),
                'w_attn': w_attn.item(),
                'w_mlp': w_mlp.item(),
                'w_attn_fraction': (w_attn / total_weight).item(),
                'head_importance': head_importance.tolist(),
                'max_displacement_used': displacement.norm(dim=-1).max().item(),
            }
        
        return displacement, stats


# =============================================================================
# LOCAL ATTENTION CACHE
# =============================================================================

class LocalAttentionCache(nn.Module):
    """
    Computes k-NN, distances, and RPE for local self-attention.
    Also stores neighbor positions for displacement computation.
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
        Compute k-nearest neighbors, RPE, distances, and neighbor positions.
        
        Returns cache dict with:
            - topk_indices: [B, L, k]
            - rpe: [B, L, k, pe_dim]
            - self_rpe: [B, L, 1, pe_dim]
            - distances: [B, L, k]
            - neighbor_positions: [B, L, k, 2]  <- NEW for displacement
        """
        B, L, _ = positions.shape
        device = positions.device
        dtype = positions.dtype
        
        k = min(k, L - 1)
        
        if physical_scale is None:
            physical_scale = self.latent_spacing * math.sqrt(k / math.pi)
        
        # Find k-nearest neighbors
        with torch.no_grad():
            diff_all = positions.unsqueeze(2) - positions.unsqueeze(1)
            dist_sq_all = (diff_all ** 2).sum(dim=-1)
            
            mask = torch.eye(L, dtype=torch.bool, device=device)
            dist_sq_all = dist_sq_all.masked_fill(mask.unsqueeze(0), float('inf'))
            
            _, topk_indices = torch.topk(dist_sq_all, k=k, dim=-1, largest=False)
        
        # Gather neighbor positions
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        positions_exp = positions.unsqueeze(1).expand(-1, L, -1, -1)
        neighbor_positions = torch.gather(positions_exp, dim=2, index=idx_exp)  # [B, L, k, 2]
        
        # Compute delta for RPE
        delta = neighbor_positions - positions.unsqueeze(2)
        delta_x = delta[..., 0]
        delta_y = delta[..., 1]
        
        # Distances for Gaussian bias
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        
        # RPE for neighbors: [B, L, k, pe_dim]
        rpe = self.pos_encoder(delta_x, delta_y, physical_scale, gsd=None)
        
        # Self-RPE: should be [B, L, 1, pe_dim]
        self_delta_x = torch.zeros(B, L, 1, device=device, dtype=dtype)
        self_delta_y = torch.zeros(B, L, 1, device=device, dtype=dtype)
        self_rpe = self.pos_encoder(self_delta_x, self_delta_y, physical_scale, gsd=None)
        
        # Ensure 4D shape (pos_encoder might squeeze singleton dim)
        if self_rpe.dim() == 3:
            self_rpe = self_rpe.unsqueeze(2)  # [B, L, pe_dim] -> [B, L, 1, pe_dim]
        
        return {
            'topk_indices': topk_indices,
            'rpe': rpe,
            'self_rpe': self_rpe,
            'distances': distances,
            'neighbor_positions': neighbor_positions,  # NEW
        }
    
    def get_output_dim(self) -> int:
        return self.pe_dim


# =============================================================================
# SPATIAL LOCAL ATTENTION (returns attention weights for displacement)
# =============================================================================

class SpatialLocalAttention(nn.Module):
    """
    Local self-attention for spatial latents.
    Returns attention weights for use in displacement computation.
    
    With RPE enabled (default), attention is direction-aware:
    - Gaussian bias encodes distance (scalar)
    - RPE encodes distance + direction (r, θ)
    
    This directional information helps displacement prediction
    by allowing heads to specialize in different directions.
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rpe: bool = True,  # Default True for directional awareness
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
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            spatial: [B, L_s, D]
            topk_indices: [B, L_s, k]
            rpe: [B, L_s, k, pe_dim] or None
            self_rpe: [B, L_s, 1, pe_dim] or None
            distances: [B, L_s, k]
            global_latents: [B, G, D] or None
            return_attn: If True, return attention weights
            
        Returns:
            output: [B, L_s, D]
            attn_weights: [B, H, L_s, 1+k+G] if return_attn else None
        """
        B, L_s, D = spatial.shape
        k = topk_indices.shape[-1]
        H = self.heads
        device = spatial.device
        dtype = spatial.dtype
        
        G = global_latents.shape[1] if global_latents is not None else 0
        
        # === Build context: [self, k neighbors, global] ===
        
        # Gather k-NN neighbors
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        spatial_exp = spatial.unsqueeze(1).expand(-1, L_s, -1, -1)
        neighbors = torch.gather(spatial_exp, dim=2, index=idx_exp)
        
        # Add self
        self_feat = spatial.unsqueeze(2)
        context = torch.cat([self_feat, neighbors], dim=2)  # [B, L_s, 1+k, D]
        
        # Add global
        if G > 0:
            global_exp = global_latents.unsqueeze(1).expand(-1, L_s, -1, -1)
            context = torch.cat([context, global_exp], dim=2)
        
        # Build distances
        self_dist = torch.zeros(B, L_s, 1, device=device, dtype=dtype)
        dist_cat = torch.cat([self_dist, distances], dim=2)
        if G > 0:
            global_dist = torch.full((B, L_s, G), float('inf'), device=device, dtype=dtype)
            dist_cat = torch.cat([dist_cat, global_dist], dim=2)
        
        # Optional RPE
        if self.use_rpe and rpe is not None:
            rpe_cat = torch.cat([self_rpe, rpe], dim=2)
            if G > 0:
                global_rpe = torch.zeros(B, L_s, G, self.pe_dim, device=device, dtype=dtype)
                rpe_cat = torch.cat([rpe_cat, global_rpe], dim=2)
            context = torch.cat([context, rpe_cat], dim=-1)
        
        # === Attention ===
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
            
            spatial_dist = dist_cat[:, :, :1+k]
            dist_sq = (spatial_dist ** 2).unsqueeze(1)
            gaussian_bias = -dist_sq / (2 * sigma_sq)
            attn[:, :, :, :1+k] = attn[:, :, :, :1+k] + gaussian_bias
            
            if G > 0 and self.global_bias is not None:
                attn[:, :, :, 1+k:] = attn[:, :, :, 1+k:] + self.global_bias
        
        attn_weights = F.softmax(attn, dim=-1)  # [B, H, L_s, 1+k+G]
        out = torch.einsum('b h l c, b h l c d -> b h l d', attn_weights, V)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.to_out(out)
        
        if return_attn:
            return out, attn_weights
        return out, None


# =============================================================================
# GLOBAL FULL ATTENTION
# =============================================================================

class GlobalFullAttention(nn.Module):
    """
    Full attention for global latents: Q=global, K/V=[ALL spatial, ALL global]
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
# HYBRID SELF-ATTENTION BLOCK WITH DISPLACEMENT
# =============================================================================

class HybridSelfAttentionBlockWithDisplacement(nn.Module):
    """
    Self-attention block with integrated displacement prediction.
    
    Architecture:
    1. Spatial local attention (with global in context), return attn weights
    2. Predict displacement from attention weights + MLP
    3. Global full attention
    4. FeedForward for both
    
    With RPE enabled (default), attention patterns are direction-aware,
    which helps displacement prediction understand "move LEFT" vs "move RIGHT".
    """
    
    def __init__(
        self,
        dim: int,
        pe_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rpe: bool = True,  # Default True for directional awareness
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        has_global: bool = True,
        enable_displacement: bool = True,
        max_displacement: float = 5.0,
    ):
        super().__init__()
        self.has_global = has_global
        self.enable_displacement = enable_displacement
        
        # 1. Spatial local attention
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
        
        # 2. Displacement predictor (optional)
        if enable_displacement:
            self.displacement_predictor = AttentionGuidedDisplacement(
                dim=dim,
                num_heads=heads,
                max_displacement=max_displacement,
            )
        else:
            self.displacement_predictor = None
        
        # 3. Global attention
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
        current_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """
        Args:
            latents: [B, L_total, D]
            cache: dict with topk_indices, rpe, self_rpe, distances, neighbor_positions
            num_spatial: L_s
            current_positions: [B, L_s, 2]
            
        Returns:
            latents: [B, L_total, D]
            displacement: [B, L_s, 2] or None
            disp_stats: dict
        """
        L_s = num_spatial
        k = cache['topk_indices'].shape[-1]
        
        spatial = latents[:, :L_s]
        global_ = latents[:, L_s:] if latents.shape[1] > L_s else None
        
        # 1. Spatial attention (return attention weights)
        spatial_out, attn_weights = self.spatial_attn(
            self.spatial_norm(spatial),
            cache['topk_indices'],
            cache.get('rpe'),
            cache.get('self_rpe'),
            cache['distances'],
            global_latents=global_,
            return_attn=True,
        )
        spatial = spatial + spatial_out
        
        # 2. Predict displacement (if enabled)
        displacement = None
        disp_stats = {}
        if self.enable_displacement and self.displacement_predictor is not None:
            displacement, disp_stats = self.displacement_predictor(
                spatial,  # Use updated spatial features
                attn_weights,
                cache['neighbor_positions'],
                current_positions,
                k,
            )
        
        # 3. FeedForward for spatial
        spatial = spatial + self.spatial_ff(self.spatial_ff_norm(spatial))
        
        # 4. Global attention
        if self.has_global and global_ is not None and global_.shape[1] > 0:
            global_ = global_ + self.global_attn(
                self.global_norm(global_),
                spatial,
            )
            global_ = global_ + self.global_ff(self.global_ff_norm(global_))
        
        # Recombine
        if global_ is not None and global_.shape[1] > 0:
            latents = torch.cat([spatial, global_], dim=1)
        else:
            latents = spatial
        
        return latents, displacement, disp_stats


# =============================================================================
# MAIN MODULE
# =============================================================================

class HybridSelfAttentionWithDisplacement(nn.Module):
    """
    Hybrid self-attention with attention-guided displacement.
    
    Key features:
    - Spatial: Q=spatial, K/V=[self, k-NN, global] - local attention
    - Global: Q=global, K/V=[ALL spatial, global] - full attention
    - Displacement: Attention patterns + MLP correction
    - RPE: Direction-aware attention (enabled by default)
    
    With RPE enabled, attention heads can specialize:
    - Some heads focus on semantic similarity
    - Some heads focus on specific directions (LEFT, RIGHT, etc.)
    
    This directional awareness significantly helps displacement prediction.
    
    Memory: O(L_s × (k+1+G) + G × (L_s+G))
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
        use_rpe: bool = True,  # Default True for directional awareness
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        num_blocks: int = 4,
        has_global: bool = True,
        share_weights: bool = False,
        enable_displacement: bool = True,
        max_displacement: float = 5.0,
        displacement_mode: str = 'per_block',  # 'per_block', 'last_only', 'accumulate'
    ):
        super().__init__()
        
        self.k = k
        self.use_rpe = use_rpe
        self.use_gaussian_bias = use_gaussian_bias
        self.num_blocks = num_blocks
        self.enable_displacement = enable_displacement
        self.displacement_mode = displacement_mode
        
        # Cache computation
        self.local_cache = LocalAttentionCache(
            pos_encoder=pos_encoder,
            latent_spacing=latent_spacing,
        )
        self.pe_dim = self.local_cache.get_output_dim()
        
        # Attention blocks
        if share_weights:
            # Determine which blocks have displacement
            block_has_displacement = self._get_block_displacement_flags(num_blocks)
            
            block = HybridSelfAttentionBlockWithDisplacement(
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
                enable_displacement=enable_displacement,
                max_displacement=max_displacement,
            )
            self.blocks = nn.ModuleList([block] * num_blocks)
        else:
            block_has_displacement = self._get_block_displacement_flags(num_blocks)
            
            self.blocks = nn.ModuleList([
                HybridSelfAttentionBlockWithDisplacement(
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
                    enable_displacement=block_has_displacement[i],
                    max_displacement=max_displacement,
                )
                for i in range(num_blocks)
            ])
    
    def _get_block_displacement_flags(self, num_blocks: int) -> List[bool]:
        """Determine which blocks should predict displacement."""
        if not self.enable_displacement:
            return [False] * num_blocks
        
        if self.displacement_mode == 'last_only':
            flags = [False] * num_blocks
            flags[-1] = True
            return flags
        else:  # 'per_block' or 'accumulate'
            return [True] * num_blocks
    
    def compute_cache(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None,
        physical_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute k-NN cache including neighbor positions."""
        k = k or self.k
        return self.local_cache(positions, k, physical_scale)
    
    def forward(
        self,
        latents: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        num_spatial: int,
        current_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass.
        
        Args:
            latents: [B, L_total, D]
            cache: dict from compute_cache
            num_spatial: L_s
            current_positions: [B, L_s, 2]
            
        Returns:
            latents: [B, L_total, D]
            total_displacement: [B, L_s, 2] or None
            stats: dict with diagnostics
        """
        all_stats = {
            'per_block': [],
        }
        
        # Track displacement based on mode
        if self.displacement_mode == 'accumulate':
            total_displacement = torch.zeros_like(current_positions)
        else:
            total_displacement = None
        
        for i, block in enumerate(self.blocks):
            latents, block_displacement, block_stats = block(
                latents, cache, num_spatial, current_positions
            )
            
            block_stats['block_idx'] = i
            all_stats['per_block'].append(block_stats)
            
            # Handle displacement based on mode
            if block_displacement is not None:
                if self.displacement_mode == 'accumulate':
                    total_displacement = total_displacement + block_displacement
                elif self.displacement_mode == 'per_block':
                    # Return last block's displacement
                    total_displacement = block_displacement
                elif self.displacement_mode == 'last_only':
                    total_displacement = block_displacement
        
        # Compute aggregate stats
        if self.enable_displacement and len(all_stats['per_block']) > 0:
            valid_stats = [s for s in all_stats['per_block'] if 'attn_disp_magnitude' in s]
            if valid_stats:
                all_stats['mean_attn_disp'] = sum(s['attn_disp_magnitude'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_mlp_disp'] = sum(s['mlp_disp_magnitude'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_w_attn_fraction'] = sum(s['w_attn_fraction'] for s in valid_stats) / len(valid_stats)
        
        if total_displacement is not None:
            all_stats['total_displacement_magnitude'] = total_displacement.norm(dim=-1).mean().item()
        
        return latents, total_displacement, all_stats
    
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
    
    def get_displacement_stats(self) -> Dict[str, Any]:
        """Get displacement predictor statistics."""
        stats = {}
        
        for i, block in enumerate(self.blocks):
            if block.displacement_predictor is not None:
                pred = block.displacement_predictor
                stats[f'block_{i}'] = {
                    'w_attn': pred.log_attn_weight.exp().item(),
                    'w_mlp': pred.log_mlp_weight.exp().item(),
                    'head_importance': F.softmax(pred.head_weights, dim=0).tolist(),
                }
        
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_hybrid_self_attention_with_displacement(
    config: dict,
    pos_encoder: nn.Module,
) -> HybridSelfAttentionWithDisplacement:
    """Factory function."""
    cfg = config.get("Atomiser", config)
    
    return HybridSelfAttentionWithDisplacement(
        dim=cfg.get("latent_dim", 512),
        pos_encoder=pos_encoder,
        k=cfg.get("self_attn_k", 64),
        latent_spacing=cfg.get("latent_spacing", 3.0),
        heads=cfg.get("latent_heads", 8),
        dim_head=cfg.get("latent_dim_head", 64),
        ff_mult=cfg.get("ff_mult", 4),
        dropout=cfg.get("attn_dropout", 0.0),
        use_rpe=cfg.get("use_rpe", True),  # Default True
        use_gaussian_bias=cfg.get("use_gaussian_bias", True),
        sigma_init=cfg.get("sigma_init", 3.0),
        learnable_sigma=cfg.get("learnable_sigma", True),
        num_blocks=cfg.get("self_per_cross_attn", 4),
        has_global=cfg.get("global_latents", 0) > 0,
        share_weights=cfg.get("weight_tie_layers", False),
        enable_displacement=cfg.get("enable_displacement", True),
        max_displacement=cfg.get("max_displacement", 5.0),
        displacement_mode=cfg.get("displacement_mode", "per_block"),
    )