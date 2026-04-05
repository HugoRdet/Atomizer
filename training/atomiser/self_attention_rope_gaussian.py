"""
Self-Attention with RoPE + Optional Gaussian Distance Bias
==========================================================

Two Gaussian bias modes (both compatible with scaled_dot_product_attention):

  "soft":  Full Gaussian bias as float attn_mask → memory-efficient backend
           score_ij = QK/√d + (-dist²/2σ²)
           
  "hard":  Threshold to boolean mask → potentially Flash Attention
           score_ij = QK/√d   where dist_ij < cutoff_σ * σ
           score_ij = -inf     otherwise

RoPE handles content-dependent spatial awareness.
Gaussian bias adds spatial prior (soft = smooth falloff, hard = binary cutoff).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelfAttentionRoPE(nn.Module):
    """
    Self-attention with 2D RoPE and optional Gaussian distance bias.
    
    Args:
        use_gaussian_bias: Enable distance-based attention bias/mask
        gaussian_sigma: Initial σ per head
        gaussian_mode: "soft" (float bias, memory-efficient SDPA) or
                       "hard" (boolean mask, potential Flash Attention)
        gaussian_cutoff: For "hard" mode, mask out latents beyond cutoff*σ
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_compression_scale: float = 50.0,
        rope_learnable_scale: bool = True,
        # Gaussian bias parameters
        use_gaussian_bias: bool = False,
        gaussian_sigma: float = 3.0,
        learnable_sigma: bool = True,
        gaussian_mode: str = "hard",    # "soft" or "hard"
        gaussian_cutoff: float = 3.0,   # for "hard": mask beyond cutoff*σ
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"
        
        self.heads = heads
        self.dim_head = dim_head
        self.use_rope = use_rope
        self.use_gaussian_bias = use_gaussian_bias
        self.gaussian_mode = gaussian_mode
        self.gaussian_cutoff = gaussian_cutoff
        self.dropout_p = dropout
        
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = dim_head ** -0.5
        
        # RoPE module
        if use_rope:
            from .RPE import LocalRoPE2D
            self.rope = LocalRoPE2D(
                dim_head=dim_head,
                base=rope_base,
                compression_scale=rope_compression_scale,
                learnable_scale=rope_learnable_scale,
                num_heads=heads,
            )
        else:
            self.rope = None
        
        # Gaussian distance bias
        if use_gaussian_bias:
            if learnable_sigma:
                self.log_sigma = nn.Parameter(
                    torch.full((heads,), math.log(gaussian_sigma)))
            else:
                self.register_buffer(
                    'log_sigma', torch.full((heads,), math.log(gaussian_sigma)))
            
            print(f"[SelfAttentionRoPE] Gaussian bias: mode={gaussian_mode}, "
                  f"σ={gaussian_sigma:.1f}, learnable={learnable_sigma}"
                  + (f", cutoff={gaussian_cutoff}σ" if gaussian_mode == "hard" else ""))
    
    @property
    def sigma(self):
        return self.log_sigma.exp()
    
    def _compute_distance_mask(
        self,
        pos_x: torch.Tensor,   # [B, N_spatial]
        pos_y: torch.Tensor,   # [B, N_spatial]
        num_spatial: int,
        total_N: int,
    ) -> torch.Tensor:
        """
        Compute attention mask from pairwise distances.
        
        "soft": returns float bias [B, H, N, N] added to attention scores
        "hard": returns boolean mask [B, 1, N, N] (True = attend)
        """
        B = pos_x.shape[0]
        device = pos_x.device
        dtype = pos_x.dtype
        
        # Pairwise squared distance: [B, N_s, N_s]
        dx = pos_x.unsqueeze(2) - pos_x.unsqueeze(1)
        dy = pos_y.unsqueeze(2) - pos_y.unsqueeze(1)
        dist_sq = dx ** 2 + dy ** 2
        
        if self.gaussian_mode == "soft":
            # Float bias: -dist²/(2σ²) per head
            sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)  # [1, H, 1, 1]
            spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq + 1e-8)
            
            if total_N == num_spatial:
                return spatial_bias
            
            # Pad for global latents (zero bias = neutral)
            full_bias = torch.zeros(
                B, self.heads, total_N, total_N,
                device=device, dtype=dtype)
            full_bias[:, :, :num_spatial, :num_spatial] = spatial_bias
            return full_bias
        
        else:  # "hard"
            # Boolean mask: attend if within cutoff*σ (use max σ across heads)
            max_sigma = self.sigma.max()
            cutoff_sq = (self.gaussian_cutoff * max_sigma) ** 2
            spatial_mask = dist_sq < cutoff_sq  # [B, N_s, N_s]
            
            if total_N == num_spatial:
                return spatial_mask.unsqueeze(1)  # [B, 1, N, N]
            
            # Global latents always visible
            full_mask = torch.ones(
                B, 1, total_N, total_N,
                dtype=torch.bool, device=device)
            full_mask[:, :, :num_spatial, :num_spatial] = spatial_mask.unsqueeze(1)
            return full_mask
    
    def forward(
        self,
        x: torch.Tensor,
        pos_x: Optional[torch.Tensor] = None,
        pos_y: Optional[torch.Tensor] = None,
        num_spatial: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,
        gsd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, d = self.heads, self.dim_head
        
        # Handle positions in [B, N, 2] format
        if positions is not None and pos_x is None:
            pos_x = positions[..., 0]
            pos_y = positions[..., 1]
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, H, d) for t in qkv]
        
        # Apply RoPE
        if self.use_rope and self.rope is not None and pos_x is not None:
            if num_spatial is not None and num_spatial < N:
                q_spatial, q_global = q[:, :num_spatial], q[:, num_spatial:]
                k_spatial, k_global = k[:, :num_spatial], k[:, num_spatial:]
                
                q_spatial, k_spatial = self.rope.forward_self(
                    q_spatial, k_spatial,
                    pos_x[:, :num_spatial], pos_y[:, :num_spatial]
                )
                
                q = torch.cat([q_spatial, q_global], dim=1)
                k = torch.cat([k_spatial, k_global], dim=1)
            else:
                q, k = self.rope.forward_self(q, k, pos_x, pos_y)
        
        # ── Transpose for SDPA: [B, H, N, d] ─────────────────────
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ── Compute attention mask if Gaussian bias is active ─────
        attn_mask = None
        if self.use_gaussian_bias and pos_x is not None:
            n_spatial = num_spatial if num_spatial is not None else N
            attn_mask = self._compute_distance_mask(
                pos_x[:, :n_spatial], pos_y[:, :n_spatial],
                n_spatial, N,
            )
        
        # ── SDPA: chooses best backend automatically ──────────────
        # No mask → Flash Attention
        # Float mask (soft) → memory-efficient (xformers) backend
        # Boolean mask (hard) → memory-efficient or math backend
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).reshape(B, N, H * d)
        return self.to_out(out)