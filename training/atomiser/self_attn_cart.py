import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TargetingSelfAttention(nn.Module):
    """
    Self-attention where queries "target" positions via learned positional keys.
    
    Score formula:
        score[i,j] = (q_i^T k_j + q_i^T rpe_ij) / √d
        
    where rpe_ij = MLP(fourier(p_j - p_i)) is a position-dependent "key".
    
    This allows each query to learn position-specific targeting patterns.
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
        normalize_scale: float = 51.5,
        # Gaussian bias (optional, additive)
        use_gaussian_bias: bool = False,
        sigma: float = 9.0,
        learnable_sigma: bool = True,
    ):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.normalize_scale = normalize_scale
        self.use_gaussian_bias = use_gaussian_bias
        
        inner_dim = heads * dim_head
        
        # =====================================================================
        # Position Encoder
        # =====================================================================
        self.num_bands = num_bands
        self.max_freq = max_freq
        
        # Fourier features: (2 * num_bands + 1) per coordinate × 2 coordinates
        fourier_dim = (2 * num_bands + 1) * 2
        
        # Project to per-head dimension (shared across heads for efficiency)
        self.rpe_mlp = nn.Sequential(
            nn.Linear(fourier_dim, dim_head * 2),
            nn.GELU(),
            nn.Linear(dim_head * 2, dim_head),
        )
        
        # =====================================================================
        # Attention Projections
        # =====================================================================
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # Gaussian Bias (optional)
        # =====================================================================
        if use_gaussian_bias:
            if learnable_sigma:
                self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma)))
            else:
                self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma)))
            self.global_bias = nn.Parameter(torch.tensor(0.0))
    
    @property
    def sigma(self):
        return self.log_sigma.exp() if hasattr(self, 'log_sigma') else None
    
    def compute_rpe_cache(
        self,
        positions: torch.Tensor,  # [B, Ls, 2]
    ) -> torch.Tensor:
        """
        Compute position encoding cache.
        
        Returns:
            rpe_cache: [B, Ls, Ls, d] position-dependent keys
        """
        B, Ls, _ = positions.shape
        device = positions.device
        dtype = positions.dtype
        
        # Pairwise deltas: [B, Ls, Ls, 2]
        # delta[b, i, j] = positions[b, j] - positions[b, i]
        delta = positions.unsqueeze(1) - positions.unsqueeze(2)
        
        dx = delta[..., 0]  # [B, Ls, Ls]
        dy = delta[..., 1]  # [B, Ls, Ls]
        
        # Normalize and compress
        dx_norm = dx / self.normalize_scale
        dy_norm = dy / self.normalize_scale
        
        dx_comp = dx_norm / (1.0 + torch.abs(dx_norm))
        dy_comp = dy_norm / (1.0 + torch.abs(dy_norm))
        
        # Fourier encode
        with torch.amp.autocast("cuda",enabled=False):
            dx_comp = dx_comp.float()
            dy_comp = dy_comp.float()
            
            x_enc = self._fourier_encode(dx_comp)  # [B, Ls, Ls, 2*num_bands+1]
            y_enc = self._fourier_encode(dy_comp)  # [B, Ls, Ls, 2*num_bands+1]
            
            fourier_features = torch.cat([x_enc, y_enc], dim=-1)  # [B, Ls, Ls, fourier_dim]
            
            # Project to dim_head
            rpe_cache = self.rpe_mlp(fourier_features)  # [B, Ls, Ls, d]

        
        return rpe_cache.to(dtype)
    
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Fourier encoding: [raw, sin(f1*x), cos(f1*x), ..., sin(fL*x), cos(fL*x)]"""
        device = x.device
        freqs = torch.linspace(1.0, self.max_freq, self.num_bands, device=device)
        angles = x.unsqueeze(-1) * freqs * math.pi  # [..., num_bands]
        
        return torch.cat([
            x.unsqueeze(-1),
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)
    
    def compute_gaussian_bias(
        self,
        positions: torch.Tensor,  # [B, Ls, 2]
        num_spatial: int,
        total_latents: int,
    ) -> torch.Tensor:
        """Compute Gaussian distance bias (optional additive component)."""
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
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)  # [B, H, Ls, Ls]
        
        # Full bias matrix
        if L > Ls:
            full_bias = torch.zeros(B, self.heads, L, L, device=device, dtype=dtype)
            full_bias[:, :, :Ls, :Ls] = spatial_bias
            full_bias[:, :, Ls:, :] = self.global_bias
            full_bias[:, :, :, Ls:] = self.global_bias
            return full_bias
        
        return spatial_bias
    
    def forward(
        self,
        x: torch.Tensor,                           # [B, L, D]
        positions: torch.Tensor,                   # [B, Ls, 2]
        num_spatial: int,
        rpe_cache: Optional[torch.Tensor] = None,  # [B, Ls, Ls, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens [B, L, D]
            positions: Spatial positions [B, Ls, 2]
            num_spatial: Number of spatial tokens
            rpe_cache: Pre-computed RPE (optional, for layer reuse)
        
        Returns:
            output: [B, L, D]
            rpe_cache: [B, Ls, Ls, d] for reuse
        """
        B, L, _ = x.shape
        H, d = self.heads, self.dim_head
        Ls = num_spatial
        
        # =====================================================================
        # 1. QKV Projection
        # =====================================================================
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, H, d) for t in qkv]
        
        # =====================================================================
        # 2. Compute or Reuse RPE Cache
        # =====================================================================
        if rpe_cache is None:
            rpe_cache = self.compute_rpe_cache(positions)
        
        # =====================================================================
        # 3. Content Attention Scores
        # =====================================================================
        # [B, L, H, d] × [B, L, H, d] -> [B, H, L, L]
        scores = torch.einsum('bihd, bjhd -> bhij', q, k) * self.scale
        
        # =====================================================================
        # 4. Targeting Scores (Query-Position Coupling)
        # =====================================================================
        # q_spatial: [B, Ls, H, d]
        # rpe_cache: [B, Ls, Ls, d] (shared across heads)
        # Result: [B, H, Ls, Ls]
        q_spatial = q[:, :Ls]
        targeting_scores = torch.einsum('bihd, bijd -> bhij', q_spatial, rpe_cache) * self.scale
        
        # Add to spatial-spatial block
        scores[:, :, :Ls, :Ls] = scores[:, :, :Ls, :Ls] + targeting_scores
        
        # =====================================================================
        # 5. Gaussian Bias (Optional, Additive)
        # =====================================================================
        if self.use_gaussian_bias:
            gaussian_bias = self.compute_gaussian_bias(positions, Ls, L)
            scores = scores + gaussian_bias
        
        # =====================================================================
        # 6. Softmax, Dropout, Output
        # =====================================================================
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhij, bjhd -> bihd', attn, v)
        out = self.to_out(out.reshape(B, L, H * d))
        
        return out, rpe_cache


# =============================================================================
# PreNorm Wrapper
# =============================================================================

class PreNormTargeting(nn.Module):
    """PreNorm wrapper for TargetingSelfAttention."""
    
    def __init__(self, dim: int, fn: TargetingSelfAttention):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        num_spatial: int,
        rpe_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fn(self.norm(x), positions, num_spatial, rpe_cache)
