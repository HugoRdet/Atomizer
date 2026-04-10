"""
nn_comp.py — Core neural network components for Atomiser
=========================================================

Used by Atomiser_Senflood:
    PreNorm              — LayerNorm wrapper for encoder cross-attn and FF blocks
    FeedForward          — GEGLU feedforward block
    LatentAttentionPooling — attention pooling for classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from typing import Optional


# =============================================================================
# PRENORM
# =============================================================================

class PreNorm(nn.Module):
    """
    Apply LayerNorm before the wrapped module.

    Optionally normalizes a `context` kwarg as well (for cross-attention
    where both query and context should be normed).
    """

    def __init__(self, dim: int, fn: nn.Module, context_dim: Optional[int] = None):
        super().__init__()
        self.norm         = nn.LayerNorm(dim)
        self.fn           = fn
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        if self.norm_context is not None and "context" in kwargs and kwargs["context"] is not None:
            kwargs = {**kwargs, "context": self.norm_context(kwargs["context"])}
        return self.fn(x, **kwargs)


# =============================================================================
# FEEDFORWARD
# =============================================================================

class FeedForward(nn.Module):
    """
    Two-layer feedforward block with GEGLU activation.

    GEGLU: x → W1(x) ⊙ σ(W2(x))
    Uses a 4× expansion factor with gated output.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(dim, dim * mult * 2)
        self.w2      = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, gate = self.w1(x).chunk(2, dim=-1)
        x = x1 * F.gelu(gate)
        return self.dropout(self.w2(x))


# =============================================================================
# CROSS-ATTENTION (used internally by LatentAttentionPooling)
# =============================================================================

class CrossAttention(nn.Module):
    """
    Standard multi-head cross-attention with Flash Attention support.

    Used by LatentAttentionPooling — not called directly by the model.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim   = dim_head * heads

        self.heads     = heads
        self.scale     = dim_head ** -0.5
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

        self.to_q   = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, _ = x.shape
        Nk = context.shape[1]
        H  = self.heads

        q = rearrange(self.to_q(x),       "b n (h d) -> b h n d", h=H)
        k = rearrange(self.to_k(context), "b n (h d) -> b h n d", h=H)
        v = rearrange(self.to_v(context), "b n (h d) -> b h n d", h=H)

        if self.use_flash:
            attn_mask = None
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).expand(-1, Nq, -1)
                attn_mask = mask.unsqueeze(1).expand(-1, H, -1, -1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).expand(-1, Nq, -1)
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out  = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# =============================================================================
# LATENT ATTENTION POOLING
# =============================================================================

class LatentAttentionPooling(nn.Module):
    """
    Compress a sequence of latents into a single vector via cross-attention.

    A single learned query attends over all latents, producing one vector
    that aggregates global information. Used as the classifier input.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.cross  = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] sequence of latents
        Returns:
            [B, dim] pooled representation
        """
        b   = x.size(0)
        q   = repeat(self.query, "1 1 d -> b 1 d", b=b)
        out = self.cross(q, context=x, mask=None)
        return out.squeeze(1)