"""
Hybrid Self-Attention with Explicit Displacement + Learned Weights

Key Design Principles:
1. EXPLICIT SIGNALS provide DIRECTIONS (similarity-based repulsion, attention-based attraction)
2. MLP learns per-latent WEIGHTS for those directions (not directions themselves!)
3. Fully interpretable: each component has clear semantic meaning
4. Easy to ablate: can disable MLP modulation to test explicit-only

Why MLP outputs weights, not directions:
- MLP only sees latent features, NOT neighbor positions/features
- Therefore MLP CANNOT compute meaningful directions
- But MLP CAN learn "given my content, how much should I trust attraction vs repulsion?"

Usage:
    hybrid = HybridSelfAttentionWithDisplacement(
        dim=512,
        pos_encoder=pos_encoder,
        k=64,
        enable_displacement=True,
        use_mlp_weights=True,  # Set False for explicit-only (ablation)
    )
    
    cache = hybrid.compute_cache(spatial_coords)
    latents, displacement, stats = hybrid(latents, cache, num_spatial, positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from einops import rearrange

from .nn_comp import FeedForward


# =============================================================================
# EXPLICIT SIGNAL COMPUTATION
# =============================================================================

def compute_explicit_signals(
    latents: torch.Tensor,            # [B, L_s, D]
    neighbor_features: torch.Tensor,  # [B, L_s, k, D]
    neighbor_positions: torch.Tensor, # [B, L_s, k, 2]
    current_positions: torch.Tensor,  # [B, L_s, 2]
) -> Dict[str, torch.Tensor]:
    """
    Compute explicit (non-learned) signals from features and positions.
    
    These are EXPERT PRIORS - meaningful from epoch 1, no training required!
    
    Returns:
        similarity: [B, L_s, k] - cosine similarity with each neighbor
        complexity: [B, L_s] - feature complexity (normalized std)
        uniqueness: [B, L_s] - how different from neighbors (1 - mean similarity)
        distances: [B, L_s, k] - distance to each neighbor
        delta: [B, L_s, k, 2] - vector to each neighbor
    """
    B, L_s, D = latents.shape
    k = neighbor_features.shape[2]
    
    # === 1. Similarity with neighbors ===
    latents_exp = latents.unsqueeze(2)  # [B, L_s, 1, D]
    similarity = F.cosine_similarity(latents_exp, neighbor_features, dim=-1)  # [B, L_s, k]
    
    # === 2. Feature complexity (high std = rich content) ===
    feature_std = latents.std(dim=-1)  # [B, L_s]
    std_min = feature_std.min(dim=-1, keepdim=True)[0]
    std_max = feature_std.max(dim=-1, keepdim=True)[0]
    complexity = (feature_std - std_min) / (std_max - std_min + 1e-8)
    
    # === 3. Uniqueness (high = different from neighbors) ===
    uniqueness = 1 - similarity.mean(dim=-1)  # [B, L_s]
    
    # === 4. Distances and directions ===
    delta = neighbor_positions - current_positions.unsqueeze(2)  # [B, L_s, k, 2]
    distances = delta.norm(dim=-1)  # [B, L_s, k]
    
    return {
        'similarity': similarity,
        'complexity': complexity,
        'uniqueness': uniqueness,
        'distances': distances,
        'delta': delta,
    }


# =============================================================================
# EXPLICIT DISPLACEMENT WITH LEARNED WEIGHTS
# =============================================================================

class ExplicitDisplacementWithLearnedWeights(nn.Module):
    """
    Displacement predictor where:
    - DIRECTIONS come from explicit signals (attraction, repulsion)
    - WEIGHTS come from MLP (content-dependent modulation)
    
    This is more principled than MLP outputting directions because:
    - MLP can't see neighbors, so it CAN'T compute directions
    - But MLP CAN learn when to trust each direction based on content
    
    Ablation modes:
    - use_mlp_weights=True: Full model with learned weight modulation
    - use_mlp_weights=False: Explicit signals only with global weights
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_displacement: float = 5.0,
        use_mlp_weights: bool = True,
        importance_weight: float = 0.8,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_displacement = max_displacement
        self.use_mlp_weights = use_mlp_weights
        
        # === Attention head importance (which heads for attraction) ===
        self.head_weights = nn.Parameter(torch.zeros(num_heads))
        
        # === Repulsion temperature (sharpness) ===
        self.repulsion_temperature = nn.Parameter(torch.tensor(1.0))
        
        # === Global base weights ===
        self.log_base_attn = nn.Parameter(torch.tensor(1.0))      # exp(1) ≈ 2.7
        self.log_base_repulsion = nn.Parameter(torch.tensor(0.0)) # exp(0) = 1.0
        
        # === Importance strength (how much complexity+uniqueness reduces movement) ===
        self.importance_strength = nn.Parameter(torch.tensor(importance_weight))
        
        # === MLP for per-latent weight modulation ===
        if use_mlp_weights:
            self.weight_mlp = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.LayerNorm(dim // 4),
                nn.Linear(dim // 4, 3),  # [w_attn_mod, w_repulsion_mod, scale_mod]
            )
            # Initialize to output ~0 → sigmoid(0)=0.5 → modulation ≈ 1.0
            nn.init.zeros_(self.weight_mlp[-1].weight)
            nn.init.zeros_(self.weight_mlp[-1].bias)
        else:
            self.weight_mlp = None
    
    def forward(
        self,
        latents: torch.Tensor,            # [B, L_s, D]
        attn_weights: torch.Tensor,       # [B, H, L_s, 1+k+G]
        neighbor_features: torch.Tensor,  # [B, L_s, k, D]
        neighbor_positions: torch.Tensor, # [B, L_s, k, 2]
        current_positions: torch.Tensor,  # [B, L_s, 2]
        k: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute displacement using explicit directions + learned weights.
        
        Returns:
            displacement: [B, L_s, 2]
            stats: dict with interpretable diagnostics
        """
        B, L_s, D = latents.shape
        
        # =====================================================================
        # 1. EXPLICIT SIGNALS (no learning, meaningful from epoch 1)
        # =====================================================================
        signals = compute_explicit_signals(
            latents, neighbor_features, neighbor_positions, current_positions
        )
        
        # =====================================================================
        # 2. ATTRACTION DIRECTION (from attention patterns)
        # =====================================================================
        neighbor_attn = attn_weights[:, :, :, 1:1+k]  # [B, H, L_s, k]
        
        # Weight attention heads by learned importance
        head_importance = F.softmax(self.head_weights, dim=0)  # [H]
        weighted_attn = torch.einsum('h, b h l k -> b l k', head_importance, neighbor_attn)
        weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute weighted center of attended neighbors
        weighted_center = torch.einsum('b l k, b l k d -> b l d', weighted_attn, neighbor_positions)
        attraction = weighted_center - current_positions  # [B, L_s, 2]
        
        # =====================================================================
        # 3. REPULSION DIRECTION (from similarity - explicit!)
        # =====================================================================
        # Direction away from each neighbor (unit vectors)
        direction_away = -signals['delta'] / (signals['distances'].unsqueeze(-1) + 1e-8)
        
        # Repulsion strength: high similarity + close = strong repulsion
        temp = self.repulsion_temperature.abs() + 1e-8
        similarity_factor = torch.exp(signals['similarity'] / temp)
        distance_factor = 1.0 / (signals['distances'] + 0.1)
        
        repulsion_strength = similarity_factor * distance_factor
        repulsion_strength = repulsion_strength / (repulsion_strength.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum of repulsion directions
        repulsion = torch.einsum('b l k, b l k d -> b l d', repulsion_strength, direction_away)
        
        # =====================================================================
        # 4. IMPORTANCE (explicit: complexity + uniqueness)
        # =====================================================================
        importance = 0.5 * signals['complexity'] + 0.5 * signals['uniqueness']
        imp_strength = torch.sigmoid(self.importance_strength)
        explicit_scale = 1.0 - imp_strength * importance  # [B, L_s]
        
        # =====================================================================
        # 5. WEIGHT MODULATION (learned, content-dependent)
        # =====================================================================
        base_attn = self.log_base_attn.exp()
        base_repulsion = self.log_base_repulsion.exp()
        
        if self.use_mlp_weights and self.weight_mlp is not None:
            # MLP predicts per-latent weight modulation
            weight_mods = self.weight_mlp(latents)  # [B, L_s, 3]
            
            # Modulation factors: sigmoid → [0, 1], then scale to [0, 2]
            # This allows both suppressing (×0.5) and amplifying (×1.5) base weights
            w_attn_mod = 2 * torch.sigmoid(weight_mods[..., 0])      # [B, L_s]
            w_repulsion_mod = 2 * torch.sigmoid(weight_mods[..., 1]) # [B, L_s]
            scale_mod = torch.sigmoid(weight_mods[..., 2])           # [B, L_s]
            
            # Per-latent weights = base × modulation
            w_attn = base_attn * w_attn_mod
            w_repulsion = base_repulsion * w_repulsion_mod
            final_scale = explicit_scale * scale_mod
        else:
            # Ablation: no MLP, just global weights
            w_attn = base_attn * torch.ones(B, L_s, device=latents.device)
            w_repulsion = base_repulsion * torch.ones(B, L_s, device=latents.device)
            w_attn_mod = torch.ones(B, L_s, device=latents.device)
            w_repulsion_mod = torch.ones(B, L_s, device=latents.device)
            scale_mod = torch.ones(B, L_s, device=latents.device)
            final_scale = explicit_scale
        
        # =====================================================================
        # 6. COMBINE DIRECTIONS WITH WEIGHTS
        # =====================================================================
        total_w = w_attn + w_repulsion + 1e-8
        
        combined = (
            w_attn.unsqueeze(-1) * attraction +
            w_repulsion.unsqueeze(-1) * repulsion
        ) / total_w.unsqueeze(-1)
        
        # Apply importance scaling (explicit) × learned scale
        combined = combined * final_scale.unsqueeze(-1)
        
        # =====================================================================
        # 7. CONSTRAIN MAGNITUDE
        # =====================================================================
        displacement = torch.tanh(combined / self.max_displacement) * self.max_displacement
        
        # =====================================================================
        # 8. DIAGNOSTICS (all interpretable!)
        # =====================================================================
        with torch.no_grad():
            stats = {
                # Directions (explicit)
                'attraction_magnitude': attraction.norm(dim=-1).mean().item(),
                'repulsion_magnitude': repulsion.norm(dim=-1).mean().item(),
                'final_displacement_magnitude': displacement.norm(dim=-1).mean().item(),
                
                # Global weights
                'base_w_attn': base_attn.item(),
                'base_w_repulsion': base_repulsion.item(),
                
                # Per-latent weight modulation (if MLP enabled)
                'w_attn_mod_mean': w_attn_mod.mean().item(),
                'w_repulsion_mod_mean': w_repulsion_mod.mean().item(),
                'w_attn_mod_std': w_attn_mod.std().item(),
                'w_repulsion_mod_std': w_repulsion_mod.std().item(),
                'scale_mod_mean': scale_mod.mean().item(),
                
                # Effective per-latent weights
                'w_attn_effective_mean': w_attn.mean().item(),
                'w_repulsion_effective_mean': w_repulsion.mean().item(),
                'w_attn_fraction': (w_attn / total_w).mean().item(),
                
                # Explicit signals
                'similarity_mean': signals['similarity'].mean().item(),
                'complexity_mean': signals['complexity'].mean().item(),
                'uniqueness_mean': signals['uniqueness'].mean().item(),
                'importance_mean': importance.mean().item(),
                
                # Scaling
                'explicit_scale_mean': explicit_scale.mean().item(),
                'final_scale_mean': final_scale.mean().item(),
                
                # Parameters
                'repulsion_temperature': temp.item(),
                'importance_strength': imp_strength.item(),
                'head_importance': head_importance.tolist(),
                
                # For visualization (per-latent tensors)
                'similarity_per_latent': signals['similarity'].mean(dim=-1),  # [B, L_s]
                'complexity_per_latent': signals['complexity'],                # [B, L_s]
                'uniqueness_per_latent': signals['uniqueness'],                # [B, L_s]
                'importance_per_latent': importance,                           # [B, L_s]
                'movement_scale_per_latent': final_scale,                      # [B, L_s]
            }
        
        return displacement, stats


# =============================================================================
# LOCAL ATTENTION CACHE
# =============================================================================

class LocalAttentionCache(nn.Module):
    """
    Computes k-NN, distances, RPE, and neighbor positions for local attention.
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
        neighbor_positions = torch.gather(positions_exp, dim=2, index=idx_exp)
        
        # Compute delta for RPE
        delta = neighbor_positions - positions.unsqueeze(2)
        delta_x = delta[..., 0]
        delta_y = delta[..., 1]
        
        # Distances
        distances = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        
        # RPE for neighbors
        rpe = self.pos_encoder(delta_x, delta_y, physical_scale, gsd=None)
        
        # Self-RPE
        self_delta_x = torch.zeros(B, L, 1, device=device, dtype=dtype)
        self_delta_y = torch.zeros(B, L, 1, device=device, dtype=dtype)
        self_rpe = self.pos_encoder(self_delta_x, self_delta_y, physical_scale, gsd=None)
        
        if self_rpe.dim() == 3:
            self_rpe = self_rpe.unsqueeze(2)
        
        return {
            'topk_indices': topk_indices,
            'rpe': rpe,
            'self_rpe': self_rpe,
            'distances': distances,
            'neighbor_positions': neighbor_positions,
        }
    
    def get_output_dim(self) -> int:
        return self.pe_dim


# =============================================================================
# SPATIAL LOCAL ATTENTION
# =============================================================================

class SpatialLocalAttention(nn.Module):
    """
    Local self-attention for spatial latents.
    Returns attention weights + neighbor features for displacement computation.
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
        return_neighbor_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weights and neighbor features.
        """
        B, L_s, D = spatial.shape
        k = topk_indices.shape[-1]
        H = self.heads
        device = spatial.device
        dtype = spatial.dtype
        
        G = global_latents.shape[1] if global_latents is not None else 0
        
        # Gather k-NN neighbors
        idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        spatial_exp = spatial.unsqueeze(1).expand(-1, L_s, -1, -1)
        neighbors = torch.gather(spatial_exp, dim=2, index=idx_exp)
        
        neighbor_features_out = neighbors.clone() if return_neighbor_features else None
        
        # Build context: [self, k neighbors, global]
        self_feat = spatial.unsqueeze(2)
        context = torch.cat([self_feat, neighbors], dim=2)
        
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
        
        # Attention
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
        
        attn_weights = F.softmax(attn, dim=-1)
        out = torch.einsum('b h l c, b h l c d -> b h l d', attn_weights, V)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.to_out(out)
        
        attn_out = attn_weights if return_attn else None
        
        return out, attn_out, neighbor_features_out


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
    Self-attention block with integrated displacement.
    
    Architecture:
    1. Spatial local attention → attn_weights + neighbor_features
    2. Displacement from explicit signals + learned weights
    3. Global full attention
    4. FeedForward
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
        enable_displacement: bool = True,
        max_displacement: float = 5.0,
        use_mlp_weights: bool = True,
    ):
        super().__init__()
        self.has_global = has_global
        self.enable_displacement = enable_displacement
        
        # Spatial local attention
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
        
        # Displacement predictor
        if enable_displacement:
            self.displacement_predictor = ExplicitDisplacementWithLearnedWeights(
                dim=dim,
                num_heads=heads,
                max_displacement=max_displacement,
                use_mlp_weights=use_mlp_weights,
            )
        else:
            self.displacement_predictor = None
        
        # Global attention
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass.
        """
        L_s = num_spatial
        k = cache['topk_indices'].shape[-1]
        
        spatial = latents[:, :L_s]
        global_ = latents[:, L_s:] if latents.shape[1] > L_s else None
        
        # Spatial attention
        need_disp_info = self.enable_displacement and self.displacement_predictor is not None
        
        spatial_out, attn_weights, neighbor_features = self.spatial_attn(
            self.spatial_norm(spatial),
            cache['topk_indices'],
            cache.get('rpe'),
            cache.get('self_rpe'),
            cache['distances'],
            global_latents=global_,
            return_attn=need_disp_info,
            return_neighbor_features=need_disp_info,
        )
        spatial = spatial + spatial_out
        
        # Displacement
        displacement = None
        disp_stats = {}
        if need_disp_info:
            displacement, disp_stats = self.displacement_predictor(
                spatial,
                attn_weights,
                neighbor_features,
                cache['neighbor_positions'],
                current_positions,
                k,
            )
        
        # FeedForward for spatial
        spatial = spatial + self.spatial_ff(self.spatial_ff_norm(spatial))
        
        # Global attention
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
    Hybrid self-attention with explicit displacement + learned weights.
    
    Key Design:
    - DIRECTIONS from explicit signals (attention-based attraction, similarity-based repulsion)
    - WEIGHTS from MLP (content-dependent modulation of directions)
    
    Ablation-friendly:
    - use_mlp_weights=False → test explicit-only baseline
    - enable_displacement=False → test no-displacement baseline
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
        use_rpe: bool = True,
        use_gaussian_bias: bool = True,
        sigma_init: float = 3.0,
        learnable_sigma: bool = True,
        num_blocks: int = 4,
        has_global: bool = True,
        share_weights: bool = False,
        enable_displacement: bool = True,
        max_displacement: float = 5.0,
        displacement_mode: str = 'last_only',
        use_mlp_weights: bool = True,
    ):
        super().__init__()
        
        self.k = k
        self.use_rpe = use_rpe
        self.use_gaussian_bias = use_gaussian_bias
        self.num_blocks = num_blocks
        self.enable_displacement = enable_displacement
        self.displacement_mode = displacement_mode
        self.use_mlp_weights = use_mlp_weights
        
        # Cache computation
        self.local_cache = LocalAttentionCache(
            pos_encoder=pos_encoder,
            latent_spacing=latent_spacing,
        )
        self.pe_dim = self.local_cache.get_output_dim()
        
        # Determine which blocks have displacement
        block_has_displacement = self._get_block_displacement_flags(num_blocks)
        
        # Attention blocks
        if share_weights:
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
                use_mlp_weights=use_mlp_weights,
            )
            self.blocks = nn.ModuleList([block] * num_blocks)
        else:
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
                    use_mlp_weights=use_mlp_weights,
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
        else:
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
        
        Returns:
            latents: [B, L_total, D]
            displacement: [B, L_s, 2] or None
            stats: dict with interpretable diagnostics
        """
        all_stats = {
            'per_block': [],
        }
        
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
            
            if block_displacement is not None:
                if self.displacement_mode == 'accumulate':
                    total_displacement = total_displacement + block_displacement
                else:
                    total_displacement = block_displacement
        
        # Aggregate stats
        if self.enable_displacement and len(all_stats['per_block']) > 0:
            valid_stats = [s for s in all_stats['per_block'] if 'attraction_magnitude' in s]
            if valid_stats:
                all_stats['mean_attraction'] = sum(s['attraction_magnitude'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_repulsion'] = sum(s['repulsion_magnitude'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_similarity'] = sum(s['similarity_mean'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_complexity'] = sum(s['complexity_mean'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_uniqueness'] = sum(s['uniqueness_mean'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_w_attn_mod'] = sum(s['w_attn_mod_mean'] for s in valid_stats) / len(valid_stats)
                all_stats['mean_w_repulsion_mod'] = sum(s['w_repulsion_mod_mean'] for s in valid_stats) / len(valid_stats)
        
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
        """Get displacement predictor parameter statistics."""
        stats = {}
        
        for i, block in enumerate(self.blocks):
            if block.displacement_predictor is not None:
                pred = block.displacement_predictor
                stats[f'block_{i}'] = {
                    'base_w_attn': pred.log_base_attn.exp().item(),
                    'base_w_repulsion': pred.log_base_repulsion.exp().item(),
                    'repulsion_temperature': pred.repulsion_temperature.abs().item(),
                    'importance_strength': torch.sigmoid(pred.importance_strength).item(),
                    'head_importance': F.softmax(pred.head_weights, dim=0).tolist(),
                    'use_mlp_weights': pred.use_mlp_weights,
                }
        
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_hybrid_self_attention_with_displacement(
    config: dict,
    pos_encoder: nn.Module,
) -> HybridSelfAttentionWithDisplacement:
    """Factory function for creating from config."""
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
        use_rpe=cfg.get("use_rpe", True),
        use_gaussian_bias=cfg.get("use_gaussian_bias", True),
        sigma_init=cfg.get("sigma_init", 3.0),
        learnable_sigma=cfg.get("learnable_sigma", True),
        num_blocks=cfg.get("self_per_cross_attn", 4),
        has_global=cfg.get("global_latents", 0) > 0,
        share_weights=cfg.get("weight_tie_layers", False),
        enable_displacement=cfg.get("enable_displacement", True),
        max_displacement=cfg.get("max_displacement", 5.0),
        displacement_mode=cfg.get("displacement_mode", "last_only"),
        use_mlp_weights=cfg.get("use_mlp_weights", True),
    )