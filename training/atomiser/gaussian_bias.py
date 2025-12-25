import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from functools import wraps
from einops import repeat, rearrange
from typing import Optional, Tuple, List, Dict, Any



# =============================================================================
# Self-Attention with Gaussian Distance Bias (FULL L×L, MEMORY INTENSIVE)
# =============================================================================

class SelfAttentionWithGaussianBias(nn.Module):
    """
    Self-attention with Gaussian distance bias for spatial latents.
    
    This is the FULL L×L version - use only when you need global attention.
    For memory-efficient local attention, use HybridSelfAttentionWithDisplacement instead.
    
    Bias formula: bias[i,j] = -dist(i,j)² / (2σ²)
    
    Memory: O(B × H × L × L) for bias matrix (~50 MB for L=1225)
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        sigma: float = 3.0,
        learnable_sigma: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma)))
        else:
            self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma)))
        
        self.global_bias = nn.Parameter(torch.tensor(0.0))
    
    @property
    def sigma(self):
        return self.log_sigma.exp()
    
    def compute_distance_bias(
        self,
        positions: torch.Tensor,
        num_spatial: int,
        total_latents: int,
    ) -> torch.Tensor:
        B = positions.shape[0]
        L_spatial = num_spatial
        L_total = total_latents
        device = positions.device
        dtype = positions.dtype
        
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)
        
        full_bias = torch.zeros(B, self.heads, L_total, L_total, device=device, dtype=dtype)
        full_bias[:, :, :L_spatial, :L_spatial] = spatial_bias
        
        if L_total > L_spatial:
            full_bias[:, :, L_spatial:, :] = self.global_bias
            full_bias[:, :, :, L_spatial:] = self.global_bias
        
        return full_bias
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor = None,
        num_spatial: int = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if positions is not None and num_spatial is not None:
            bias = self.compute_distance_bias(positions, num_spatial, L)
            attn = attn + bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.to_out(out)