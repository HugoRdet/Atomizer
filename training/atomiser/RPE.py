import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from functools import wraps
from einops import repeat, rearrange
from typing import Optional, Tuple, List, Dict, Any, Union
from torch.utils.checkpoint import checkpoint
from training.utils.token_building.fourier_features import fourier_encode




class PairwiseRPEEncoder(nn.Module):
    """
    Computes pairwise relative position encodings using Fourier features.
    
    For positions (p_i, p_j), computes fourier(p_j - p_i).
    """
    
    def __init__(
        self,
        num_bands: int = 32,
        max_freq: float = 32.0,
        normalize_scale: float = 10.0,
    ):
        super().__init__()
        
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.normalize_scale = normalize_scale
        
        # Output dim: (num_bands * 2 + 1) per coordinate × 2 coordinates
        self.output_dim = (num_bands * 2 + 1) * 2
    
    def forward(
        self,
        query_pos: torch.Tensor,  # [B, N_q, 2]
        key_pos: torch.Tensor,    # [B, N_k, 2]
    ) -> torch.Tensor:
        """
        Compute pairwise RPE features.
        
        Returns: [B, N_q, N_k, output_dim]
        """
        # Compute pairwise deltas: key - query
        # query_pos: [B, N_q, 1, 2]
        # key_pos:   [B, 1, N_k, 2]
        # delta:     [B, N_q, N_k, 2]
        delta = key_pos.unsqueeze(1) - query_pos.unsqueeze(2)
        
        delta_x = delta[..., 0]  # [B, N_q, N_k]
        delta_y = delta[..., 1]  # [B, N_q, N_k]
        
        # Normalize
        dx = delta_x / self.normalize_scale
        dy = delta_y / self.normalize_scale
        
        # Compress to (-1, 1)
        dx_comp = dx / (1.0 + torch.abs(dx))
        dy_comp = dy / (1.0 + torch.abs(dy))
        
        # Fourier encode
        x_enc = fourier_encode(dx_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        y_enc = fourier_encode(dy_comp, max_freq=self.max_freq, num_bands=self.num_bands)
        
        # Concatenate: [B, N_q, N_k, output_dim]
        return torch.cat([x_enc, y_enc], dim=-1)


# =============================================================================
# SELF-ATTENTION WITH RPE CONCATENATED TO K
# =============================================================================

class SelfAttentionRPEConcat(nn.Module):
    """
    Self-attention where RPE Fourier features are concatenated to K.
    
    For each (Q_i, K_j) pair:
    - Q_i = W_q · x_i                              (content only)
    - K_ij = W_k · concat(x_j, fourier(p_j - p_i)) (content + relative position)
    - V_j = W_v · x_j                              (content only)
    - score_ij = Q_i · K_ij / sqrt(d)
    
    This allows K to be position-aware relative to each query.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        # RPE parameters
        rpe_num_bands: int = 32,
        rpe_max_freq: float = 32.0,
        rpe_normalize_scale: float = 51.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        rpe_normalize_scale=51.0
        rpe_num_bands= 16
        rpe_max_freq= 16
        
        # RPE encoder
        self.rpe_encoder = PairwiseRPEEncoder(
            num_bands=rpe_num_bands,
            max_freq=rpe_max_freq,
            normalize_scale=rpe_normalize_scale,
        )
        self.rpe_dim = self.rpe_encoder.output_dim
        
        # Q: from content only
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        
        # K: from content + RPE features
        self.to_k = nn.Linear(dim + self.rpe_dim, inner_dim, bias=False)
        
        # V: from content only
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,                           # [B, N, dim]
        positions: Optional[torch.Tensor] = None,  # [B, N_spatial, 2]
        num_spatial: Optional[int] = None,
        gsd: Optional[torch.Tensor] = None,        # unused for now
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tokens [B, N, dim]
            positions: Spatial positions [B, N_spatial, 2]
            num_spatial: Number of spatial tokens (rest are global)
        
        Returns:
            Output tokens [B, N, dim]
        """
        B, N, D = x.shape
        H, d = self.heads, self.dim_head
        
        L_spatial = num_spatial if num_spatial is not None else N
        L_global = N - L_spatial
        
        # =================================================================
        # CASE 1: All tokens are spatial
        # =================================================================
        if L_global == 0 and positions is not None:
            return self._forward_all_spatial(x, positions)
        
        # =================================================================
        # CASE 2: Mixed spatial and global tokens
        # =================================================================
        if positions is not None and L_spatial > 0:
            return self._forward_mixed(x, positions, L_spatial, L_global)
        
        # =================================================================
        # CASE 3: No positions provided - standard attention
        # =================================================================
        return self._forward_no_positions(x)
    
    def _forward_all_spatial(
        self,
        x: torch.Tensor,       # [B, N, dim]
        positions: torch.Tensor,  # [B, N, 2]
    ) -> torch.Tensor:
        """All tokens are spatial - full RPE."""
        B, N, D = x.shape
        H, d = self.heads, self.dim_head
        
        # Compute pairwise RPE features: [B, N, N, rpe_dim]
        rpe_features = self.rpe_encoder(positions, positions)
        
        # Q: [B, N, H, d]
        q = self.to_q(x).view(B, N, H, d)
        
        # V: [B, N, H, d]
        v = self.to_v(x).view(B, N, H, d)
        
        # K: needs to incorporate RPE for each (i, j) pair
        # x_j expanded: [B, N_q, N_k, dim] = [B, N, N, dim]
        x_expanded = x.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, dim]
        
        # Concatenate content and RPE: [B, N, N, dim + rpe_dim]
        k_input = torch.cat([x_expanded, rpe_features], dim=-1)
        
        # Project to K: [B, N, N, inner_dim] -> [B, N, N, H, d]
        k = self.to_k(k_input).view(B, N, N, H, d)
        
        # Attention scores: Q_i · K_ij
        # q: [B, N_q, H, d]
        # k: [B, N_q, N_k, H, d]
        # scores: [B, H, N_q, N_k]
        scores = torch.einsum('b i h d, b i j h d -> b h i j', q, k) * self.scale
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum of values
        # attn: [B, H, N_q, N_k]
        # v: [B, N_k, H, d]
        # out: [B, N_q, H, d]
        out = torch.einsum('b h i j, b j h d -> b i h d', attn, v)
        
        return self.to_out(out.reshape(B, N, H * d))
    
    def _forward_mixed(
        self,
        x: torch.Tensor,          # [B, N, dim]
        positions: torch.Tensor,  # [B, L_spatial, 2]
        L_spatial: int,
        L_global: int,
    ) -> torch.Tensor:
        """Mixed spatial and global tokens."""
        B, N, D = x.shape
        H, d = self.heads, self.dim_head
        
        x_spatial = x[:, :L_spatial]   # [B, L_spatial, dim]
        x_global = x[:, L_spatial:]    # [B, L_global, dim]
        
        # =================================================================
        # Compute Q for all tokens
        # =================================================================
        q = self.to_q(x).view(B, N, H, d)  # [B, N, H, d]
        
        # =================================================================
        # Compute V for all tokens
        # =================================================================
        v = self.to_v(x).view(B, N, H, d)  # [B, N, H, d]
        
        # =================================================================
        # Compute K - different for spatial vs global
        # =================================================================
        
        # --- Spatial-to-Spatial RPE ---
        # For spatial queries attending to spatial keys
        rpe_spatial = self.rpe_encoder(positions, positions)  # [B, L_spatial, L_spatial, rpe_dim]
        
        # x_spatial expanded for spatial keys: [B, L_spatial, L_spatial, dim]
        x_spatial_expanded = x_spatial.unsqueeze(1).expand(B, L_spatial, L_spatial, D)
        
        # K for spatial keys (from spatial queries): [B, L_spatial, L_spatial, H, d]
        k_spatial_spatial = self.to_k(
            torch.cat([x_spatial_expanded, rpe_spatial], dim=-1)
        ).view(B, L_spatial, L_spatial, H, d)
        
        # --- Global-to-Spatial RPE (global queries attending to spatial keys) ---
        # Global tokens have no position, use zeros for RPE
        zero_rpe_global_spatial = torch.zeros(
            B, L_global, L_spatial, self.rpe_dim, 
            device=x.device, dtype=x.dtype
        )
        x_spatial_for_global = x_spatial.unsqueeze(1).expand(B, L_global, L_spatial, D)
        
        k_global_spatial = self.to_k(
            torch.cat([x_spatial_for_global, zero_rpe_global_spatial], dim=-1)
        ).view(B, L_global, L_spatial, H, d)
        
        # --- Spatial-to-Global (spatial queries attending to global keys) ---
        # No RPE for global keys
        zero_rpe_spatial_global = torch.zeros(
            B, L_spatial, L_global, self.rpe_dim,
            device=x.device, dtype=x.dtype
        )
        x_global_for_spatial = x_global.unsqueeze(1).expand(B, L_spatial, L_global, D)
        
        k_spatial_global = self.to_k(
            torch.cat([x_global_for_spatial, zero_rpe_spatial_global], dim=-1)
        ).view(B, L_spatial, L_global, H, d)
        
        # --- Global-to-Global (global queries attending to global keys) ---
        zero_rpe_global_global = torch.zeros(
            B, L_global, L_global, self.rpe_dim,
            device=x.device, dtype=x.dtype
        )
        x_global_expanded = x_global.unsqueeze(1).expand(B, L_global, L_global, D)
        
        k_global_global = self.to_k(
            torch.cat([x_global_expanded, zero_rpe_global_global], dim=-1)
        ).view(B, L_global, L_global, H, d)
        
        # =================================================================
        # Assemble full K matrix: [B, N_q, N_k, H, d]
        # =================================================================
        # Layout:
        #                  Keys
        #              Spatial  Global
        # Queries  Spatial  [SS]    [SG]
        #          Global   [GS]    [GG]
        
        # Concatenate along key dimension for spatial queries
        k_from_spatial_queries = torch.cat([k_spatial_spatial, k_spatial_global], dim=2)  # [B, L_spatial, N, H, d]
        
        # Concatenate along key dimension for global queries
        k_from_global_queries = torch.cat([k_global_spatial, k_global_global], dim=2)  # [B, L_global, N, H, d]
        
        # Concatenate along query dimension
        k = torch.cat([k_from_spatial_queries, k_from_global_queries], dim=1)  # [B, N, N, H, d]
        
        # =================================================================
        # Compute attention
        # =================================================================
        scores = torch.einsum('b i h d, b i j h d -> b h i j', q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b j h d -> b i h d', attn, v)
        
        return self.to_out(out.reshape(B, N, H * d))
    
    def _forward_no_positions(self, x: torch.Tensor) -> torch.Tensor:
        """No positions - use zero RPE features."""
        B, N, D = x.shape
        H, d = self.heads, self.dim_head
        
        # Zero RPE features
        zero_rpe = torch.zeros(B, N, N, self.rpe_dim, device=x.device, dtype=x.dtype)
        
        # Q
        q = self.to_q(x).view(B, N, H, d)
        
        # V
        v = self.to_v(x).view(B, N, H, d)
        
        # K with zero RPE
        x_expanded = x.unsqueeze(1).expand(B, N, N, D)
        k_input = torch.cat([x_expanded, zero_rpe], dim=-1)
        k = self.to_k(k_input).view(B, N, N, H, d)
        
        # Attention
        scores = torch.einsum('b i h d, b i j h d -> b h i j', q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b j h d -> b i h d', attn, v)
        
        return self.to_out(out.reshape(B, N, H * d))


# =============================================================================
# PRENORM WRAPPER
# =============================================================================

class PreNormRPEConcat(nn.Module):
    """PreNorm wrapper for SelfAttentionRPEConcat."""
    
    def __init__(self, dim: int, fn: SelfAttentionRPEConcat):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        num_spatial: Optional[int] = None,
        gsd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fn(
            self.norm(x), 
            positions=positions, 
            num_spatial=num_spatial, 
            gsd=gsd
        )

# =============================================================================
# CARTESIAN RPE MODULE (Standalone, reusable)
# =============================================================================

class CartesianRPE(nn.Module):
    """
    Computes Cartesian Fourier RPE and projects to attention bias.
    
    Pipeline:
    (pos_i, pos_j) -> delta = pos_j - pos_i -> Fourier encode -> project to bias
    """
    
    def __init__(
        self,
        num_heads: int,
        num_bands: int = 32,
        max_freq: float = 32.0,
        normalize_scale: float = 50.0,
        include_gsd: bool = False,
        G_ref: float = 0.2,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.normalize_scale = normalize_scale
        self.include_gsd = include_gsd
        self.G_ref = G_ref
        
        # Fourier encoding dimension per component: num_bands * 2 (sin, cos) + 1 (raw)
        self.per_component_dim = num_bands * 2 + 1
        
        # Total RPE dimension: X + Y (+ optional GSD)
        if include_gsd:
            self.rpe_dim = self.per_component_dim * 3
        else:
            self.rpe_dim = self.per_component_dim * 2
        
        # Project RPE features to per-head bias
        self.to_bias = nn.Sequential(
            nn.Linear(self.rpe_dim, num_heads * 2),
            nn.GELU(),
            nn.Linear(num_heads * 2, num_heads),
        )
    
    def forward(
        self,
        query_pos: torch.Tensor,    # [B, N_q, 2] query positions
        key_pos: torch.Tensor,      # [B, N_k, 2] key positions
        gsd: Optional[torch.Tensor] = None,  # [B, N_k] or scalar
    ) -> torch.Tensor:
        """
        Compute attention bias from relative positions.
        
        Args:
            query_pos: [B, N_q, 2] positions of queries
            key_pos: [B, N_k, 2] positions of keys
            gsd: Optional GSD for resolution-aware encoding
        
        Returns:
            bias: [B, H, N_q, N_k] attention bias
        """
        B, N_q, _ = query_pos.shape
        N_k = key_pos.shape[1]
        device = query_pos.device
        dtype = query_pos.dtype


        
        # Compute pairwise deltas: key - query (for each query, relative position of each key)
        # query_pos: [B, N_q, 1, 2]
        # key_pos:   [B, 1, N_k, 2]
        # delta:     [B, N_q, N_k, 2]
        delta = key_pos.unsqueeze(1) - query_pos.unsqueeze(2)
        
        delta_x = delta[..., 0]  # [B, N_q, N_k]
        delta_y = delta[..., 1]  # [B, N_q, N_k]
        
        # Encode with Cartesian Fourier features
        rpe_features = self._encode_cartesian(delta_x, delta_y, gsd)  # [B, N_q, N_k, rpe_dim]
        
        # Project to attention bias
        bias = self.to_bias(rpe_features)  # [B, N_q, N_k, H]
        bias = bias.permute(0, 3, 1, 2)    # [B, H, N_q, N_k]
        
        return bias
    
    def _encode_cartesian(
        self,
        delta_x: torch.Tensor,  # [B, N_q, N_k]
        delta_y: torch.Tensor,  # [B, N_q, N_k]
        gsd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode relative positions with Fourier features."""
        
        # Normalize by scale
        dx = delta_x / self.normalize_scale
        dy = delta_y / self.normalize_scale
        
        # Signed compression: (-inf, inf) -> (-1, 1)
        dx_comp = dx / (1.0 + torch.abs(dx))
        dy_comp = dy / (1.0 + torch.abs(dy))
        
        # Fourier encoding
        x_enc = self._fourier_encode(dx_comp)  # [B, N_q, N_k, per_component_dim]
        y_enc = self._fourier_encode(dy_comp)  # [B, N_q, N_k, per_component_dim]
        
        if self.include_gsd and gsd is not None:
            # Handle GSD
            if not isinstance(gsd, torch.Tensor):
                gsd = torch.full_like(delta_x, gsd)
            
            if gsd.dim() == 2:  # [B, N_k]
                gsd = gsd.unsqueeze(1).expand(-1, delta_x.shape[1], -1)  # [B, N_q, N_k]
            
            log_gsd = torch.log((gsd / self.G_ref).clamp(min=1e-8))
            gsd_enc = self._fourier_encode(log_gsd)
            return torch.cat([x_enc, y_enc, gsd_enc], dim=-1)
        
        return torch.cat([x_enc, y_enc], dim=-1)
    
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier encoding to input."""
        device = x.device
        dtype = x.dtype
        
        # Frequencies linearly spaced
        freqs = torch.linspace(1.0, self.max_freq, self.num_bands, device=device, dtype=dtype)
        
        # Compute sin and cos
        angles = x.unsqueeze(-1) * freqs * math.pi  # [..., num_bands]
        
        # Concatenate: [raw, sin, cos]
        return torch.cat([
            x.unsqueeze(-1),
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)


# =============================================================================
# SELF-ATTENTION WITH CARTESIAN RPE
# =============================================================================

class SelfAttentionCartesianRPE(nn.Module):
    """
    Self-attention with Cartesian Fourier RPE.
    
    RPE is computed as attention bias from pairwise relative positions.
    Supports both full self-attention and mixed spatial/global latents.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        # RPE parameters
        use_rpe: bool = True,
        rpe_num_bands: int = 32,
        rpe_max_freq: float = 32.0,
        rpe_normalize_scale: float = 10.0,
        include_gsd: bool = False,
    ):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rpe = use_rpe
        
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RPE module
        if use_rpe:
            self.rpe = CartesianRPE(
                num_heads=heads,
                num_bands=rpe_num_bands,
                max_freq=rpe_max_freq,
                normalize_scale=rpe_normalize_scale,
                include_gsd=include_gsd,
            )
        else:
            self.rpe = None
    
    def forward(
        self,
        x: torch.Tensor,                        # [B, N, dim]
        positions: Optional[torch.Tensor] = None,  # [B, N_spatial, 2]
        num_spatial: Optional[int] = None,      # Number of spatial tokens (rest are global)
        gsd: Optional[torch.Tensor] = None,     # [B, N] or scalar
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, dim]
            positions: Spatial positions [B, N_spatial, 2] (only for spatial tokens)
            num_spatial: Number of spatial tokens (if None, all tokens are spatial)
            gsd: Optional GSD for resolution-aware RPE
        
        Returns:
            Output tokens [B, N, dim]
        """
        B, N, _ = x.shape
        H, d = self.heads, self.dim_head
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, H, d) for t in qkv]
        
        # Compute attention scores
        scores = torch.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        
        # Add RPE bias
        if self.use_rpe and self.rpe is not None and positions is not None:
            scores = self._add_rpe_bias(scores, positions, num_spatial, gsd)
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.einsum('b h i j, b j h d -> b i h d', attn, v)
        
        return self.to_out(out.reshape(B, N, H * d))
    
    def _add_rpe_bias(
        self,
        scores: torch.Tensor,           # [B, H, N, N]
        positions: torch.Tensor,        # [B, N_spatial, 2]
        num_spatial: Optional[int],
        gsd: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Add RPE bias to attention scores."""
        
        B, H, N, _ = scores.shape
        N_spatial = positions.shape[1] if num_spatial is None else num_spatial
        
        if N_spatial == N:
            # All tokens are spatial - simple case
            bias = self.rpe(positions, positions, gsd)  # [B, H, N, N]
            return scores + bias
        
        # Mixed spatial and global tokens
        # Only add RPE bias for spatial-spatial interactions
        spatial_bias = self.rpe(positions, positions, gsd)  # [B, H, N_spatial, N_spatial]
        
        # Create full bias matrix (zeros for global interactions)
        full_bias = scores.new_zeros(B, H, N, N)
        full_bias[:, :, :N_spatial, :N_spatial] = spatial_bias
        
        return scores + full_bias


# =============================================================================
# LOCAL CROSS-ATTENTION WITH CARTESIAN RPE (for decoder)
# =============================================================================

class LocalCrossAttentionCartesianRPE(nn.Module):
    """
    Local cross-attention with Cartesian Fourier RPE.
    
    Each query attends to its own local context (k nearest neighbors).
    Used in decoder: queries are tokens, context are k-nearest latents.
    """
    
    def __init__(
        self,
        dim_query: int,
        dim_context: int,
        dim_out: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        # RPE parameters
        use_rpe: bool = True,
        rpe_num_bands: int = 32,
        rpe_max_freq: float = 32.0,
        rpe_normalize_scale: float = 10.0,
        include_gsd: bool = True,
    ):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rpe = use_rpe
        
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim_query, inner_dim, bias=False)
        self.to_k = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_out)
        self.dropout = nn.Dropout(dropout)
        
        # RPE projection (local version - per query-key pair)
        if use_rpe:
            self.rpe = LocalCartesianRPE(
                num_heads=heads,
                num_bands=rpe_num_bands,
                max_freq=rpe_max_freq,
                normalize_scale=rpe_normalize_scale,
                include_gsd=include_gsd,
            )
        else:
            self.rpe = None
    
    def forward(
        self,
        x: torch.Tensor,                # [B, N, dim_query] queries
        context: torch.Tensor,          # [B, N, k, dim_context] local context per query
        mask: Optional[torch.Tensor] = None,   # [B, N, k] attention mask
        query_pos: Optional[torch.Tensor] = None,   # [B, N, 2] query positions
        context_pos: Optional[torch.Tensor] = None, # [B, N, k, 2] context positions
        gsd: Optional[torch.Tensor] = None,         # [B, N, k] or scalar
    ) -> torch.Tensor:
        """
        Args:
            x: Query tokens [B, N, dim_query]
            context: Local context for each query [B, N, k, dim_context]
            mask: Attention mask [B, N, k]
            query_pos: Query positions [B, N, 2]
            context_pos: Context positions [B, N, k, 2]
            gsd: GSD values [B, N, k] or scalar
        
        Returns:
            Output [B, N, dim_out]
        """
        B, N, _ = x.shape
        k = context.shape[2]
        H, d = self.heads, self.dim_head
        
        # Project Q, K, V
        q = self.to_q(x).view(B, N, H, d)           # [B, N, H, d]
        K = self.to_k(context).view(B, N, k, H, d)  # [B, N, k, H, d]
        V = self.to_v(context).view(B, N, k, H, d)  # [B, N, k, H, d]
        
        # Compute attention scores
        scores = torch.einsum('b n h d, b n k h d -> b n h k', q, K) * self.scale
        
        # Add RPE bias
        if self.use_rpe and self.rpe is not None and query_pos is not None and context_pos is not None:
            bias = self.rpe(query_pos, context_pos, gsd)  # [B, N, H, k]
            scores = scores + bias
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.einsum('b n h k, b n k h d -> b n h d', attn, V)
        
        return self.to_out(out.reshape(B, N, H * d))


class LocalCartesianRPE(nn.Module):
    """
    Cartesian RPE for local cross-attention.
    
    Computes RPE for each query attending to its local context.
    More memory efficient than full pairwise computation.
    """
    
    def __init__(
        self,
        num_heads: int,
        num_bands: int = 32,
        max_freq: float = 32.0,
        normalize_scale: float = 10.0,
        include_gsd: bool = True,
        G_ref: float = 0.2,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.normalize_scale = normalize_scale
        self.include_gsd = include_gsd
        self.G_ref = G_ref
        
        # Fourier encoding dimension
        self.per_component_dim = num_bands * 2 + 1
        if include_gsd:
            self.rpe_dim = self.per_component_dim * 3
        else:
            self.rpe_dim = self.per_component_dim * 2
        
        # Project to per-head bias
        self.to_bias = nn.Sequential(
            nn.Linear(self.rpe_dim, num_heads * 2),
            nn.GELU(),
            nn.Linear(num_heads * 2, num_heads),
        )
    
    def forward(
        self,
        query_pos: torch.Tensor,    # [B, N, 2]
        context_pos: torch.Tensor,  # [B, N, k, 2]
        gsd: Optional[Union[torch.Tensor, float]] = None,  # [B, N, k] or scalar
    ) -> torch.Tensor:
        """
        Compute attention bias for local cross-attention.
        
        Returns:
            bias: [B, N, H, k]
        """
        B, N, k, _ = context_pos.shape
        device = query_pos.device
        dtype = query_pos.dtype
        
        # Compute deltas: context - query
        # query_pos:   [B, N, 1, 2]
        # context_pos: [B, N, k, 2]
        delta = context_pos - query_pos.unsqueeze(2)  # [B, N, k, 2]
        
        delta_x = delta[..., 0]  # [B, N, k]
        delta_y = delta[..., 1]  # [B, N, k]
        
        # Encode
        rpe_features = self._encode_cartesian(delta_x, delta_y, gsd)  # [B, N, k, rpe_dim]
        
        # Project to bias
        bias = self.to_bias(rpe_features)  # [B, N, k, H]
        bias = bias.permute(0, 1, 3, 2)    # [B, N, H, k]
        
        return bias
    
    def _encode_cartesian(
        self,
        delta_x: torch.Tensor,
        delta_y: torch.Tensor,
        gsd: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """Encode with Cartesian Fourier features."""
        device = delta_x.device
        dtype = delta_x.dtype
        
        # Normalize
        dx = delta_x / self.normalize_scale
        dy = delta_y / self.normalize_scale
        
        # Compress
        dx_comp = dx / (1.0 + torch.abs(dx))
        dy_comp = dy / (1.0 + torch.abs(dy))
        
        # Fourier encode
        x_enc = self._fourier_encode(dx_comp)
        y_enc = self._fourier_encode(dy_comp)
        
        if self.include_gsd and gsd is not None:
            if not isinstance(gsd, torch.Tensor):
                gsd = torch.full_like(delta_x, gsd)
            
            if gsd.shape != delta_x.shape:
                try:
                    gsd = gsd.expand_as(delta_x)
                except RuntimeError:
                    gsd = torch.full_like(delta_x, gsd.mean().item())
            
            log_gsd = torch.log((gsd / self.G_ref).clamp(min=1e-8))
            gsd_enc = self._fourier_encode(log_gsd)
            return torch.cat([x_enc, y_enc, gsd_enc], dim=-1)
        
        return torch.cat([x_enc, y_enc], dim=-1)
    
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Fourier encoding."""
        device = x.device
        dtype = x.dtype
        
        freqs = torch.linspace(1.0, self.max_freq, self.num_bands, device=device, dtype=dtype)
        angles = x.unsqueeze(-1) * freqs * math.pi
        
        return torch.cat([
            x.unsqueeze(-1),
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)


# =============================================================================
# PRENORM WRAPPER
# =============================================================================

class PreNormCartesianRPE(nn.Module):
    """PreNorm wrapper for SelfAttentionCartesianRPE."""
    
    def __init__(self, dim: int, fn: SelfAttentionCartesianRPE):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        num_spatial: Optional[int] = None,
        gsd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fn(self.norm(x), positions=positions, num_spatial=num_spatial, gsd=gsd)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_self_attention_with_rpe(
    dim: int,
    heads: int = 8,
    dim_head: int = 64,
    dropout: float = 0.0,
    rpe_type: str = "cartesian",  # "cartesian", "rope", or "none"
    rpe_num_bands: int = 32,
    rpe_max_freq: float = 32.0,
    rpe_normalize_scale: float = 10.0,
    include_gsd: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create self-attention with different RPE types.
    
    Args:
        rpe_type: "cartesian" (Fourier bias), "rope" (rotation), or "none"
    """
    if rpe_type == "cartesian":
        
        return SelfAttentionCartesianRPE(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            use_rpe=True,
            rpe_num_bands=rpe_num_bands,
            rpe_max_freq=rpe_max_freq,
            rpe_normalize_scale=100,
            include_gsd=include_gsd,
        )
    elif rpe_type == "rope":

        return SelfAttentionRoPE(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            use_rope=True,
            **kwargs,
        )
    else:
        # Standard self-attention without RPE
        from .nn_comp import SelfAttention
        return SelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )


# =============================================================================
# LOCAL ROPE 2D
# =============================================================================

class LocalRoPE2D(nn.Module):
    """
    Local Rotary Position Encoding for 2D.
    
    Two modes:
    1. Cross-attention: Q at origin (no rotation), K rotated by delta
    2. Self-attention: Both Q and K rotated by their positions
    
    Resolution-aware via log-scale GSD modulation.
    """
    
    def __init__(
        self,
        dim_head: int,
        base: float = 10.0,
        reference_gsd: float = 0.2,
        learnable_scale: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"
        
        self.dim_head = dim_head
        self.reference_gsd = reference_gsd
        
        quarter_dim = dim_head // 4
        
        # Base frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim).float() / quarter_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Log reference for resolution scaling
        self.register_buffer('log_ref', torch.tensor(math.log(1.0 + reference_gsd)))
        
        # Learnable frequency scales
        if learnable_scale:
            self.scale_x = nn.Parameter(torch.ones(quarter_dim))
            self.scale_y = nn.Parameter(torch.ones(quarter_dim))
            nn.init.normal_(self.scale_x, mean=1.0, std=0.1)
            nn.init.normal_(self.scale_y, mean=1.0, std=0.1)
        else:
            self.register_buffer('scale_x', torch.ones(quarter_dim))
            self.register_buffer('scale_y', torch.ones(quarter_dim))
        
        # Resolution sensitivity
        self.res_sensitivity = nn.Parameter(torch.tensor(0.5))
    
    def forward_cross(
        self,
        q: torch.Tensor,        # [B, N, H, d]
        k: torch.Tensor,        # [B, N, k, H, d]
        delta_x: torch.Tensor,  # [B, N, k]
        delta_y: torch.Tensor,  # [B, N, k]
        gsd: Optional[torch.Tensor] = None,  # [B, N, k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention: Q unchanged (at origin), K rotated by delta."""
        res_scale = self._compute_resolution_scale(gsd)
        k_rotated = self._rotate_2d_cross(k, delta_x, delta_y, res_scale)
        return q, k_rotated
    
    def forward_self(
        self,
        q: torch.Tensor,        # [B, N, H, d]
        k: torch.Tensor,        # [B, N, H, d]
        pos_x: torch.Tensor,    # [B, N]
        pos_y: torch.Tensor,    # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-attention: Both Q and K rotated by their positions."""
        q_rotated = self._rotate_2d_self(q, pos_x, pos_y)
        k_rotated = self._rotate_2d_self(k, pos_x, pos_y)
        return q_rotated, k_rotated
    
    def _compute_resolution_scale(
        self,
        gsd: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Log-scale resolution factor."""
        if gsd is None:
            return None
        
        log_gsd = torch.log(1.0 + gsd)
        log_ratio = self.log_ref - log_gsd
        res_scale = torch.exp(log_ratio * self.res_sensitivity)
        return res_scale.clamp(min=0.1, max=10.0)
    
    def _rotate_2d_cross(
        self,
        x: torch.Tensor,        # [B, N, k, H, d]
        delta_x: torch.Tensor,  # [B, N, k]
        delta_y: torch.Tensor,  # [B, N, k]
        res_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotate for cross-attention (5D input)."""
        half_d = self.dim_head // 2
        
        x_part = x[..., :half_d]
        y_part = x[..., half_d:]
        
        x_rotated = self._apply_rotary_cross(x_part, delta_x, self.scale_x, res_scale)
        y_rotated = self._apply_rotary_cross(y_part, delta_y, self.scale_y, res_scale)
        
        return torch.cat([x_rotated, y_rotated], dim=-1)
    
    def _rotate_2d_self(
        self,
        x: torch.Tensor,        # [B, N, H, d]
        pos_x: torch.Tensor,    # [B, N]
        pos_y: torch.Tensor,    # [B, N]
    ) -> torch.Tensor:
        """Rotate for self-attention (4D input)."""
        half_d = self.dim_head // 2
        
        x_part = x[..., :half_d]
        y_part = x[..., half_d:]
        
        x_rotated = self._apply_rotary_self(x_part, pos_x, self.scale_x)
        y_rotated = self._apply_rotary_self(y_part, pos_y, self.scale_y)
        
        return torch.cat([x_rotated, y_rotated], dim=-1)
    
    def _apply_rotary_cross(
        self,
        x: torch.Tensor,        # [B, N, k, H, half_d]
        delta: torch.Tensor,    # [B, N, k]
        freq_scale: torch.Tensor,
        res_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotary for cross-attention."""
        effective_delta = delta
        if res_scale is not None:
            effective_delta = delta * res_scale
        
        freq = self.inv_freq * freq_scale
        angles = effective_delta.unsqueeze(-1) * freq  # [B, N, k, quarter_dim]
        angles = angles.unsqueeze(3)  # [B, N, k, 1, quarter_dim]
        
        cos = angles.cos()
        sin = angles.sin()
        
        def rotate_half(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat((-t2, t1), dim=-1)
        
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        return x * cos + rotate_half(x) * sin
    
    def _apply_rotary_self(
        self,
        x: torch.Tensor,        # [B, N, H, half_d]
        pos: torch.Tensor,      # [B, N]
        freq_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Rotary for self-attention."""
        freq = self.inv_freq * freq_scale
        angles = pos.unsqueeze(-1) * freq  # [B, N, quarter_dim]
        angles = angles.unsqueeze(2)  # [B, N, 1, quarter_dim]
        
        cos = angles.cos()
        sin = angles.sin()
        
        def rotate_half(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat((-t2, t1), dim=-1)
        
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        return x * cos + rotate_half(x) * sin


# =============================================================================
# CROSS-ATTENTION WITH LOCAL ROPE
# =============================================================================

class LocalCrossAttentionRoPE(nn.Module):
    """
    Cross-attention with Resolution-Aware Local RoPE.
    
    Q (latents) at origin, K (tokens) rotated by relative position.
    """
    
    def __init__(
        self,
        dim_query: int,
        dim_context: int,
        dim_out: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10.0,
        rope_reference_gsd: float = 0.2,
        rope_learnable_scale: bool = True,
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope
        
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim_query, inner_dim, bias=False)
        self.to_k = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_out)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = LocalRoPE2D(
                dim_head=dim_head,
                base=rope_base,
                reference_gsd=rope_reference_gsd,
                learnable_scale=rope_learnable_scale,
                num_heads=heads,
            )
        else:
            self.rope = None
    
    def forward(
        self,
        x: torch.Tensor,        # [B, N, dim_query]
        context: torch.Tensor,  # [B, N, k, dim_context]
        mask: Optional[torch.Tensor] = None,
        delta_x: Optional[torch.Tensor] = None,
        delta_y: Optional[torch.Tensor] = None,
        gsd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        k = context.shape[2]
        H, d = self.heads, self.dim_head
        
        q = self.to_q(x).view(B, N, H, d)
        K = self.to_k(context).view(B, N, k, H, d)
        V = self.to_v(context).view(B, N, k, H, d)
        
        # Apply RoPE
        if self.use_rope and self.rope is not None and delta_x is not None:
            q, K = self.rope.forward_cross(q, K, delta_x, delta_y, gsd=gsd)
        
        scores = torch.einsum('b n h d, b n k h d -> b n h k', q, K) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('b n h k, b n k h d -> b n h d', attn, V)
        return self.to_out(out.reshape(B, N, H * d))


# =============================================================================
# SELF-ATTENTION WITH LOCAL ROPE
# =============================================================================

class SelfAttentionRoPE(nn.Module):
    """
    Self-attention with 2D RoPE for spatial latents.
    
    Both Q and K get rotated by their positions.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10.0,
        rope_learnable_scale: bool = True,
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope
        
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = LocalRoPE2D(
                dim_head=dim_head,
                base=rope_base,
                reference_gsd=0.2,
                learnable_scale=rope_learnable_scale,
                num_heads=heads,
            )
        else:
            self.rope = None
    
    def forward(
        self,
        x: torch.Tensor,
        pos_x: Optional[torch.Tensor] = None,
        pos_y: Optional[torch.Tensor] = None,
        num_spatial: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, d = self.heads, self.dim_head
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, H, d) for t in qkv]
        
        # Apply RoPE only to spatial latents
        if self.use_rope and self.rope is not None and pos_x is not None:
            if num_spatial is not None and num_spatial < N:
                q_spatial, q_global = q[:, :num_spatial], q[:, num_spatial:]
                k_spatial, k_global = k[:, :num_spatial], k[:, num_spatial:]
                
                q_spatial, k_spatial = self.rope.forward_self(
                    q_spatial, k_spatial, pos_x, pos_y
                )
                
                q = torch.cat([q_spatial, q_global], dim=1)
                k = torch.cat([k_spatial, k_global], dim=1)
            else:
                q, k = self.rope.forward_self(q, k, pos_x, pos_y)
        
        scores = torch.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('b h i j, b j h d -> b i h d', attn, v)
        return self.to_out(out.reshape(B, N, H * d))


# =============================================================================
# PRENORM WRAPPERS
# =============================================================================

class PreNormRoPE(nn.Module):
    """PreNorm that passes positions to the inner RoPE attention."""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, pos_x=None, pos_y=None, num_spatial=None, **kwargs):
        x = self.norm(x)
        return self.fn(x, pos_x=pos_x, pos_y=pos_y, num_spatial=num_spatial, **kwargs)


class PreNormWithPositions(nn.Module):
    """PreNorm wrapper that passes positions to the inner function."""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, positions=None, num_spatial=None, **kwargs):
        x = self.norm(x)
        if positions is not None:
            return self.fn(x, positions=positions, num_spatial=num_spatial, **kwargs)
        return self.fn(x, **kwargs)

