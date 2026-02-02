import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .RPE import LocalRoPE2D  

class SelfAttentionRoPEWithGaussianBias(nn.Module):
    """
    Self-attention combining:
    1. 2D RoPE for relative position encoding (through rotation)
    2. Gaussian distance bias for explicit distance-based attention weighting
    
    This gives both:
    - Content-position coupling via RoPE (learned how position affects content similarity)
    - Content-independent distance preference via Gaussian (strong locality prior)
    
    Score formula:
        score[i,j] = (R(p_i)q_i)ᵀ(R(p_j)k_j) / √d  +  (-||p_i - p_j||² / 2σ²)
                     \_________________________/      \______________________/
                           RoPE term                    Gaussian bias term
                     (content-position coupled)       (pure distance prior)
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        # RoPE parameters
        use_rope: bool = True,
        rope_base: float = 1000.0,
        rope_learnable_scale: bool = True,
        # Gaussian bias parameters
        use_gaussian_bias: bool = True,
        sigma: float = 9.0,
        learnable_sigma: bool = True,
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope
        self.use_gaussian_bias = use_gaussian_bias
        
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # RoPE Module
        # =====================================================================
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
        
        # =====================================================================
        # Gaussian Bias Parameters
        # =====================================================================
        if use_gaussian_bias:
            if learnable_sigma:
                # Per-head learnable sigma (in log space for numerical stability)
                self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma)))
            else:
                self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma)))
            
            # Learned bias for global latents attending to/from spatial latents
            self.global_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.log_sigma = None
            self.global_bias = None
    
    @property
    def sigma(self) -> torch.Tensor:
        """Get sigma values (exponentiated from log space)."""
        return self.log_sigma.exp()
    
    def compute_gaussian_bias(
        self,
        pos_x: torch.Tensor,      # [B, L_spatial]
        pos_y: torch.Tensor,      # [B, L_spatial]
        num_spatial: int,
        total_latents: int,
    ) -> torch.Tensor:
        """
        Compute Gaussian distance bias: -d² / (2σ²)
        
        Returns:
            bias: [B, H, N, N] attention bias matrix
        """
        B = pos_x.shape[0]
        L_spatial = num_spatial
        L_total = total_latents
        device = pos_x.device
        dtype = pos_x.dtype
        
        # Compute pairwise squared distances
        # pos_x: [B, L_spatial] -> [B, L_spatial, 1] - [B, 1, L_spatial] = [B, L_spatial, L_spatial]
        dx = pos_x.unsqueeze(2) - pos_x.unsqueeze(1)  # [B, L_spatial, L_spatial]
        dy = pos_y.unsqueeze(2) - pos_y.unsqueeze(1)  # [B, L_spatial, L_spatial]
        dist_sq = dx ** 2 + dy ** 2                    # [B, L_spatial, L_spatial]
        
        # Gaussian bias: -d² / (2σ²), per head
        # sigma: [H] -> [1, H, 1, 1]
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)  # [B, H, L_spatial, L_spatial]
        
        # Handle global latents if present
        if L_total > L_spatial:
            full_bias = torch.zeros(B, self.heads, L_total, L_total, device=device, dtype=dtype)
            
            # Spatial-to-spatial: Gaussian bias
            full_bias[:, :, :L_spatial, :L_spatial] = spatial_bias
            
            # Global interactions: learned constant bias
            full_bias[:, :, L_spatial:, :] = self.global_bias
            full_bias[:, :, :, L_spatial:] = self.global_bias
            
            return full_bias
        
        return spatial_bias
    
    def forward(
        self,
        x: torch.Tensor,                              # [B, N, dim]
        pos_x: Optional[torch.Tensor] = None,         # [B, N_spatial]
        pos_y: Optional[torch.Tensor] = None,         # [B, N_spatial]
        positions: Optional[torch.Tensor] = None,     # [B, N_spatial, 2] (alternative)
        num_spatial: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE + Gaussian bias.
        
        Args:
            x: Input tokens [B, N, dim]
            pos_x: X coordinates [B, N_spatial]
            pos_y: Y coordinates [B, N_spatial]
            positions: Alternative input [B, N_spatial, 2] (will extract pos_x, pos_y)
            num_spatial: Number of spatial tokens (rest are global)
        
        Returns:
            Output tokens [B, N, dim]
        """
        B, N, _ = x.shape
        H, d = self.heads, self.dim_head
        
        # Handle alternative position input format
        if positions is not None and pos_x is None:
            pos_x = positions[..., 0]
            pos_y = positions[..., 1]
        
        L_spatial = num_spatial if num_spatial is not None else N
        
        # =====================================================================
        # 1. Project to Q, K, V
        # =====================================================================
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, H, d) for t in qkv]
        
        # =====================================================================
        # 2. Apply RoPE (Rotary Position Embeddings)
        # =====================================================================
        if self.use_rope and self.rope is not None and pos_x is not None:
            if num_spatial is not None and num_spatial < N:
                # Hybrid latents: only rotate spatial, leave global unchanged
                q_spatial, q_global = q[:, :L_spatial], q[:, L_spatial:]
                k_spatial, k_global = k[:, :L_spatial], k[:, L_spatial:]
                
                q_spatial, k_spatial = self.rope.forward_self(
                    q_spatial, k_spatial, pos_x, pos_y
                )
                
                q = torch.cat([q_spatial, q_global], dim=1)
                k = torch.cat([k_spatial, k_global], dim=1)
            else:
                q, k = self.rope.forward_self(q, k, pos_x, pos_y)
        
        # =====================================================================
        # 3. Compute Attention Scores
        # =====================================================================
        # Reshape for matmul: [B, N, H, d] -> [B, H, N, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Content scores (with RoPE already baked in)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, N, N]
        
        # =====================================================================
        # 4. Add Gaussian Bias (Additive, Content-Independent)
        # =====================================================================
        if self.use_gaussian_bias and pos_x is not None:
            gaussian_bias = self.compute_gaussian_bias(pos_x, pos_y, L_spatial, N)
            scores = scores + gaussian_bias
        
        # =====================================================================
        # 5. Softmax, Dropout, Weighted Sum
        # =====================================================================
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, H, N, d]
        
        # =====================================================================
        # 6. Final Projection
        # =====================================================================
        out = out.transpose(1, 2).reshape(B, N, H * d)
        return self.to_out(out)


# =============================================================================
# PreNorm Wrapper
# =============================================================================

class PreNormRoPEGaussian(nn.Module):
    """PreNorm wrapper for SelfAttentionRoPEWithGaussianBias."""
    
    def __init__(self, dim: int, fn: SelfAttentionRoPEWithGaussianBias):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(
        self,
        x: torch.Tensor,
        pos_x: Optional[torch.Tensor] = None,
        pos_y: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        num_spatial: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.fn(
            self.norm(x),
            pos_x=pos_x,
            pos_y=pos_y,
            positions=positions,
            num_spatial=num_spatial,
        )
