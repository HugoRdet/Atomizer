"""
Perceiver Attention Building Blocks
=====================================

Core components for the Perceiver-IO baseline:
  - GEGLU activation
  - FeedForward (GEGLU-gated)
  - Multi-head Attention (cross and self)
  - PreNorm wrapper with optional context normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# =============================================================================
# HELPERS
# =============================================================================

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# =============================================================================
# ACTIVATION
# =============================================================================

class GEGLU(nn.Module):
    """Gated GELU activation. Splits input in half, gates one with GELU of other."""
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# =============================================================================
# FEEDFORWARD
# =============================================================================

class FeedForward(nn.Module):
    """GEGLU-gated feedforward block."""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),  # 2x for GEGLU split
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# ATTENTION
# =============================================================================

class Attention(nn.Module):
    """
    Multi-head attention supporting both self-attention and cross-attention.

    Self-attention: forward(x)
    Cross-attention: forward(x, context=data, mask=mask)

    Args:
        query_dim: Dimension of query input (latents).
        context_dim: Dimension of context input (data). None = self-attention.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        dropout: Attention dropout rate.
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for multi-head: [B, N, H*D] → [B*H, N, D]
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
            (q, k, v),
        )

        # Attention scores
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # Apply mask (True = keep, False = ignore)
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


# =============================================================================
# PRENORM
# =============================================================================

class PreNorm(nn.Module):
    """
    Pre-LayerNorm wrapper. Optionally normalizes context for cross-attention.

    Usage:
        PreNorm(dim, Attention(...))                          # self-attn
        PreNorm(dim, Attention(...), context_dim=input_dim)   # cross-attn
    """
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            kwargs['context'] = self.norm_context(context)

        return self.fn(x, **kwargs)