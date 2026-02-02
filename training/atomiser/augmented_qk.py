"""
Self-Attention with Augmented Q/K (Absolute Position Features)

Position features are added to latents BEFORE Q/K projection,
making both queries and keys position-aware.

Key benefits:
- Cheap: O(Ls × D) cache vs O(Ls² × d) for targeting
- Multi-resolution friendly: compression normalizes physical distances
- Relative position emerges naturally from dot product
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionEncoder(nn.Module):
    """
    Encodes 2D positions using Fourier features + MLP.
    
    Uses compression function p/(scale + |p|) for multi-resolution support:
    - Same physical distance → same encoding regardless of image span
    - Graceful saturation for large positions (no wrapping)
    """
    
    def __init__(
        self,
        output_dim: int,
        num_bands: int = 32,
        max_freq: float = 32.0,
        compression_scale: float = 50.0,
        include_gsd: bool = True,
        reference_gsd: float = 1.0,
    ):
        super().__init__()
        
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.compression_scale = compression_scale
        self.include_gsd = include_gsd
        self.reference_gsd = reference_gsd
        
        # Fourier dim: (raw + sin + cos) × num_bands × 2 axes
        fourier_dim = (2 * num_bands + 1) * 2
        
        # Optional GSD feature
        input_dim = fourier_dim + (1 if include_gsd else 0)
        
        # MLP to project to output dimension
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        # Pre-compute frequencies
        freqs = torch.linspace(1.0, max_freq, num_bands)
        self.register_buffer('freqs', freqs)
    
    def _compress(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compress positions to (-1, 1) range.
        
        Same physical distance → same compressed value regardless of image span.
        """
        return p / (self.compression_scale + torch.abs(p))
    
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier features: [x, sin(f₁πx), cos(f₁πx), ..., sin(fₙπx), cos(fₙπx)]
        """
        # x: [..., 1] or [...]
        if x.dim() == len(x.shape) and x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        
        angles = x * self.freqs * math.pi  # [..., num_bands]
        
        return torch.cat([
            x,
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)  # [..., 2*num_bands + 1]
    
    def forward(
        self,
        positions: torch.Tensor,  # [B, L, 2] in meters
        gsd: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, L, 2] positions in physical units (meters)
            gsd: Ground sample distance (optional)
            
        Returns:
            pos_features: [B, L, output_dim]
        """
        B, L, _ = positions.shape
        
        # Compress to bounded range
        px = self._compress(positions[..., 0])  # [B, L]
        py = self._compress(positions[..., 1])  # [B, L]
        
        # Fourier encode each axis
        with torch.cuda.amp.autocast(enabled=False):
            px_enc = self._fourier_encode(px.float())  # [B, L, 2*bands+1]
            py_enc = self._fourier_encode(py.float())  # [B, L, 2*bands+1]
            
            fourier_feats = torch.cat([px_enc, py_enc], dim=-1)  # [B, L, fourier_dim]
            
            # Optionally add GSD
            if self.include_gsd and gsd is not None:
                if gsd.dim() == 0:
                    gsd = gsd.expand(B)
                log_gsd = torch.log(gsd / self.reference_gsd + 1e-8)  # [B]
                log_gsd = log_gsd.view(B, 1, 1).expand(-1, L, 1)  # [B, L, 1]
                fourier_feats = torch.cat([fourier_feats, log_gsd], dim=-1)
            
            pos_features = self.mlp(fourier_feats)
        
        return pos_features.to(positions.dtype)


class SelfAttentionWithAugmentedQK(nn.Module):
    """
    Self-attention where Q and K are computed from position-augmented inputs.
    
    For each latent i at position p_i:
        x_aug_i = x_i + PE(p_i)
        q_i = W_q(x_aug_i)
        k_i = W_k(x_aug_i)
        v_i = W_v(x_i)  # Values from original (no position)
    
    Relative position emerges from:
        q_i^T k_j = (W_q x_i + W_q PE_i)^T (W_k x_j + W_k PE_j)
                  = content-content + content-position + position-content + position-position
    
    Benefits:
    - Cheap cache: [B, Ls, D] instead of [B, Ls, Ls, d]
    - Multi-resolution friendly via compression
    - Both Q and K are position-aware
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        # Position encoding params
        num_bands: int = 32,
        max_freq: float = 32.0,
        compression_scale: float = 50.0,
        include_gsd: bool = True,
        reference_gsd: float = 1.0,
        # Gaussian bias (optional locality prior)
        use_gaussian_bias: bool = True,
        sigma_init: float = 9.0,
        learnable_sigma: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_gaussian_bias = use_gaussian_bias
        
        inner_dim = heads * dim_head
        
        # =====================================================================
        # Position Encoder
        # =====================================================================
        self.pos_encoder = PositionEncoder(
            output_dim=dim,
            num_bands=num_bands,
            max_freq=max_freq,
            compression_scale=compression_scale,
            include_gsd=include_gsd,
            reference_gsd=reference_gsd,
        )
        
        # =====================================================================
        # Attention Projections
        # =====================================================================
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # Gaussian Bias (optional)
        # =====================================================================
        if use_gaussian_bias:
            if learnable_sigma:
                self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma_init)))
            else:
                self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma_init)))
    
    @property
    def sigma(self) -> torch.Tensor:
        """Per-head sigma values."""
        return self.log_sigma.exp()
    
    def compute_position_cache(
        self,
        positions: torch.Tensor,  # [B, Ls, 2]
        gsd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute position features for caching.
        
        Args:
            positions: [B, Ls, 2] spatial latent positions in meters
            gsd: Ground sample distance
            
        Returns:
            pos_features: [B, Ls, D] position features
        """
        return self.pos_encoder(positions, gsd)
    
    def compute_gaussian_bias(
        self,
        positions: torch.Tensor,  # [B, Ls, 2]
        num_spatial: int,
        total_latents: int,
    ) -> torch.Tensor:
        """
        Compute Gaussian distance bias.
        
        Args:
            positions: [B, Ls, 2]
            num_spatial: Number of spatial latents
            total_latents: Total number of latents (spatial + global)
            
        Returns:
            bias: [B, H, L, L]
        """
        B = positions.shape[0]
        Ls = num_spatial
        L = total_latents
        device = positions.device
        dtype = positions.dtype
        
        # Pairwise squared distances
        delta = positions.unsqueeze(1) - positions.unsqueeze(2)  # [B, Ls, Ls, 2]
        dist_sq = (delta ** 2).sum(dim=-1)  # [B, Ls, Ls]
        
        # Gaussian: -d² / 2σ²
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)  # [1, H, 1, 1]
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq + 1e-8)  # [B, H, Ls, Ls]
        
        # Full bias matrix (with global latents)
        if L > Ls:
            full_bias = torch.zeros(B, self.heads, L, L, device=device, dtype=dtype)
            full_bias[:, :, :Ls, :Ls] = spatial_bias
            # Global latents get zero bias (attend equally everywhere)
            return full_bias
        
        return spatial_bias
    
    def forward(
        self,
        x: torch.Tensor,              # [B, L, D]
        positions: torch.Tensor,      # [B, Ls, 2]
        num_spatial: int,
        pos_cache: Optional[torch.Tensor] = None,  # [B, Ls, D]
        gsd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input latents [B, L, D]
            positions: Spatial latent positions [B, Ls, 2]
            num_spatial: Number of spatial latents
            pos_cache: Pre-computed position features [B, Ls, D]
            gsd: Ground sample distance
            
        Returns:
            output: [B, L, D]
            pos_cache: [B, Ls, D] for reuse
        """
        B, L, D = x.shape
        H, d = self.heads, self.dim_head
        Ls = num_spatial
        
        # =====================================================================
        # 1. Get or Compute Position Features
        # =====================================================================
        if pos_cache is None:
            pos_cache = self.compute_position_cache(positions, gsd)  # [B, Ls, D]
        
        # =====================================================================
        # 2. Augment Spatial Latents with Position
        # =====================================================================
        x_spatial = x[:, :Ls]  # [B, Ls, D]
        x_global = x[:, Ls:] if L > Ls else None  # [B, Lg, D] or None
        
        x_spatial_aug = x_spatial + pos_cache  # [B, Ls, D]
        
        # Reconstruct full sequence (global latents not augmented)
        if x_global is not None:
            x_aug = torch.cat([x_spatial_aug, x_global], dim=1)  # [B, L, D]
        else:
            x_aug = x_spatial_aug
        
        # =====================================================================
        # 3. Q, K from Augmented; V from Original
        # =====================================================================
        q = self.to_q(x_aug).view(B, L, H, d)   # [B, L, H, d]
        k = self.to_k(x_aug).view(B, L, H, d)   # [B, L, H, d]
        v = self.to_v(x).view(B, L, H, d)       # [B, L, H, d] - original, no position
        
        # =====================================================================
        # 4. Attention Scores
        # =====================================================================
        # Transpose for matmul: [B, H, L, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        # =====================================================================
        # 5. Add Gaussian Bias (optional)
        # =====================================================================
        if self.use_gaussian_bias:
            gaussian_bias = self.compute_gaussian_bias(positions, Ls, L)  # [B, H, L, L]
            scores = scores + gaussian_bias
        
        # =====================================================================
        # 6. Softmax, Dropout, Output
        # =====================================================================
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, H, L, d]
        out = out.transpose(1, 2).reshape(B, L, H * d)  # [B, L, H*d]
        
        return self.to_out(out), pos_cache


# =============================================================================
# PreNorm Wrapper
# =============================================================================

class PreNormAugmentedQK(nn.Module):
    """PreNorm wrapper for SelfAttentionWithAugmentedQK."""
    
    def __init__(self, dim: int, fn: SelfAttentionWithAugmentedQK):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        num_spatial: int,
        pos_cache: Optional[torch.Tensor] = None,
        gsd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fn(self.norm(x), positions, num_spatial, pos_cache, gsd)