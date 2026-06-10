"""
QK-Normalized Multi-Head Attention
===================================

A drop-in replacement for `nn.MultiheadAttention` that applies RMSNorm to
queries and keys after projection but before computing attention scores.

This is the established mitigation for "attention entropy collapse," where
Q and K projection magnitudes grow unboundedly during transformer training,
saturating the softmax and producing degenerate attention patterns. Used in
ViT-22B [1], Gemma 2 [2], Mistral [3], and other stable-at-scale transformer
training recipes.

The interface matches `nn.MultiheadAttention` so you can swap it in without
changing call sites:

    # Before
    self.decoder_cross_attn = nn.MultiheadAttention(
        embed_dim=768, kdim=K, vdim=V, num_heads=16,
        dropout=0.05, batch_first=True,
    )

    # After
    self.decoder_cross_attn = QKNormMultiheadAttention(
        embed_dim=768, kdim=K, vdim=V, num_heads=16,
        dropout=0.05, batch_first=True,
    )

Forward call signature is identical (query, key, value, key_padding_mask,
need_weights). The need_weights argument is accepted for API compatibility
but always returns None for the weights (matching memory-efficient kernels).

References:
    [1] Dehghani et al. "Scaling Vision Transformers to 22 Billion Parameters" (2023)
    [2] Gemma Team. "Gemma 2: Improving Open Language Models at a Practical Size" (2024)
    [3] Jiang et al. "Mistral 7B" (2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square layer normalization (no mean subtraction).

    For input x of shape [..., D], computes:
        x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps).

    Lighter than LayerNorm (no mean subtraction, no bias) and is the
    standard choice in modern transformers (Llama, Mistral, Gemma).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back
        in_dtype = x.dtype
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * rms).to(in_dtype) * self.weight


class QKNormMultiheadAttention(nn.Module):
    """
    Multi-head attention with RMSNorm applied to Q and K per-head before
    computing attention scores.

    Drop-in replacement for nn.MultiheadAttention with the same interface
    (when batch_first=True).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        if not batch_first:
            raise NotImplementedError(
                "QKNormMultiheadAttention currently only supports batch_first=True"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.dropout = dropout

        # ── Q / K / V projections ─────────────────────────────────────
        # Match the parameter names from nn.MultiheadAttention so a
        # checkpoint trained with MHA can theoretically be loaded into
        # this module with state_dict surgery (modulo the new norm
        # parameters). Not used today, but cheap to preserve.
        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, self.kdim))
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        # ── Output projection ─────────────────────────────────────────
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # ── QK normalization (per-head) ───────────────────────────────
        # RMSNorm with dimension = head_dim, applied separately per head.
        # The same norm layer is shared across heads (one gamma per
        # head_dim feature), which is the standard formulation.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform for projections, matching nn.MultiheadAttention's
        # default (which uses Xavier for in_proj_weight).
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: [B, Tq, embed_dim]
            key:   [B, Tk, kdim]
            value: [B, Tk, vdim]
            key_padding_mask: [B, Tk] bool, True = mask out
            need_weights: kept for API compat, always returns None
            attn_mask: optional additive attention mask [Tq, Tk] or
                       [B*num_heads, Tq, Tk]

        Returns:
            attn_out: [B, Tq, embed_dim]
            None  (weights not returned; matches sdpa kernel behavior)
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # ── Projections ───────────────────────────────────────────────
        # Split the bias into 3 chunks (q, k, v) to match nn.MHA layout.
        if self.in_proj_bias is not None:
            q_bias = self.in_proj_bias[:self.embed_dim]
            k_bias = self.in_proj_bias[self.embed_dim:2 * self.embed_dim]
            v_bias = self.in_proj_bias[2 * self.embed_dim:]
        else:
            q_bias = k_bias = v_bias = None

        q = F.linear(query, self.q_proj_weight, q_bias)   # [B, Tq, embed_dim]
        k = F.linear(key,   self.k_proj_weight, k_bias)   # [B, Tk, embed_dim]
        v = F.linear(value, self.v_proj_weight, v_bias)   # [B, Tk, embed_dim]

        # ── Reshape to [B, num_heads, T, head_dim] ────────────────────
        q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        # ── QK-norm (the actual fix) ──────────────────────────────────
        # RMSNorm operates over the last dim, which is head_dim. Each
        # head is normalized independently. This bounds the magnitude
        # of Q and K such that ||q|| ≈ ||k|| ≈ sqrt(head_dim) regardless
        # of how large the underlying projection weights become.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ── Build the attention mask for sdpa ─────────────────────────
        # sdpa expects either:
        #   - attn_mask: [B*H, Tq, Tk] additive float mask, OR
        #   - is_causal flag, OR
        #   - key_padding_mask broadcast manually
        # We combine key_padding_mask + attn_mask into a single bool mask.
        mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, Tk], True = mask
            # expand to [B, 1, 1, Tk] for broadcasting against [B, H, Tq, Tk]
            mask = key_padding_mask[:, None, None, :]
        if attn_mask is not None:
            # attn_mask: [Tq, Tk] or [B*H, Tq, Tk]
            if attn_mask.dim() == 2:
                am = attn_mask[None, None]   # [1, 1, Tq, Tk]
            elif attn_mask.dim() == 3:
                am = attn_mask.view(B, self.num_heads, Tq, Tk)
            else:
                am = attn_mask
            mask = am if mask is None else (mask | am.bool())

        # ── Attention ─────────────────────────────────────────────────
        # F.scaled_dot_product_attention handles the softmax, scaling by
        # sqrt(head_dim), and dropout for us. Pass attn_mask as a bool
        # mask (True = mask out).
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )   # [B, H, Tq, head_dim]

        # ── Reshape and project ───────────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            B, Tq, self.embed_dim
        )
        attn_out = self.out_proj(attn_out)

        # Always return None for weights — matches sdpa kernel behavior
        # and signals "don't depend on this for analysis."
        return attn_out, None
