"""
Perceiver-IO Encoder-Decoder
==============================

Task-agnostic Perceiver-IO following Jaegle et al. (2021).

Encoder:
    Input tokens [B, N, D_in] → cross-attention with latents →
    self-attention layers → latent representation [B, L, D_latent]

Decoder:
    Query tokens [B, M, D_query] × cross-attention with latents →
    output [B, M, D_query] → logits [B, M, D_out]

The encoder uses iterative cross-attention (depth > 1 repeats
cross-attn → self-attn blocks). Weight tying optional.
"""

from functools import wraps

import torch
import torch.nn as nn
from einops import repeat

from .attention import Attention, PreNorm, FeedForward


# =============================================================================
# CACHE HELPER (for weight tying)
# =============================================================================

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


# =============================================================================
# PERCEIVER-IO
# =============================================================================

class PerceiverIO(nn.Module):
    """
    Perceiver-IO: encoder maps arbitrary input to fixed latents,
    decoder maps latents to arbitrary output via query cross-attention.

    Args:
        input_dim: Dimension of input tokens (channels + fourier pos).
        query_dim: Dimension of decoder query tokens.
        output_dim: Dimension of final output (e.g., num_classes).
        num_latents: Number of latent vectors.
        latent_dim: Dimension of each latent vector.
        depth: Number of encoder cross-attn → self-attn blocks.
        cross_heads: Heads for cross-attention.
        latent_heads: Heads for latent self-attention.
        cross_dim_head: Dim per head in cross-attention.
        latent_dim_head: Dim per head in self-attention.
        self_per_cross_attn: Number of self-attn blocks per cross-attn.
        weight_tie_layers: Share weights across depth > 1 blocks.
        attn_dropout: Attention dropout.
        ff_dropout: Feedforward dropout.
        decoder_ff: Add feedforward after decoder cross-attention.
    """

    def __init__(
        self,
        input_dim,
        query_dim,
        output_dim,
        num_latents=512,
        latent_dim=512,
        depth=6,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        self_per_cross_attn=1,
        weight_tie_layers=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
        decoder_ff=False,
    ):
        super().__init__()

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        # ── Encoder layers ──────────────────────────────────────────
        get_cross_attn = cache_fn(lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim, input_dim,
                heads=cross_heads, dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=input_dim,
        ))
        get_cross_ff = cache_fn(lambda: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout),
        ))
        get_self_attn = cache_fn(lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                heads=latent_heads, dim_head=latent_dim_head,
                dropout=attn_dropout,
            ),
        ))
        get_self_ff = cache_fn(lambda: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout),
        ))

        self.encoder_layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            # Self-attention blocks
            self_attns = nn.ModuleList([])
            for sa_idx in range(self_per_cross_attn):
                sa_key = 0 if should_cache else f"{i}_{sa_idx}"
                self_attns.append(nn.ModuleList([
                    get_self_attn(**cache_args, key=f"sa_{sa_key}"),
                    get_self_ff(**cache_args, key=f"sf_{sa_key}"),
                ]))

            cross_key = 0 if should_cache else i
            self.encoder_layers.append(nn.ModuleList([
                get_cross_attn(**cache_args, key=f"ca_{cross_key}"),
                get_cross_ff(**cache_args, key=f"cf_{cross_key}"),
                self_attns,
            ]))

        # ── Decoder ─────────────────────────────────────────────────
        self.decoder_cross_attn = PreNorm(
            query_dim,
            Attention(
                query_dim, latent_dim,
                heads=cross_heads, dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(query_dim, FeedForward(query_dim, dropout=ff_dropout))
            if decoder_ff else None
        )

        # Output projection
        self.to_logits = nn.Linear(query_dim, output_dim)

    def encode(self, data, mask=None):
        """
        Encode input tokens into latent representation.

        Args:
            data: [B, N, input_dim] input tokens.
            mask: [B, N] bool mask (True = valid token).

        Returns:
            latents: [B, L, latent_dim]
        """
        B = data.shape[0]
        x = repeat(self.latents, 'n d -> b n d', b=B)

        for cross_attn, cross_ff, self_attns in self.encoder_layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        return x

    def decode(self, latents, queries):
        """
        Decode from latents using query tokens.

        Args:
            latents: [B, L, latent_dim] from encoder.
            queries: [B, M, query_dim] decoder queries.

        Returns:
            logits: [B, M, output_dim]
        """
        out = self.decoder_cross_attn(queries, context=latents)

        if self.decoder_ff is not None:
            out = out + self.decoder_ff(out)

        return self.to_logits(out)

    def forward(self, data, mask=None, queries=None):
        """
        Full forward: encode data, decode with queries.

        Args:
            data: [B, N, input_dim]
            mask: [B, N] optional
            queries: [B, M, query_dim]

        Returns:
            logits: [B, M, output_dim]
        """
        latents = self.encode(data, mask=mask)

        if queries is None:
            return latents

        return self.decode(latents, queries)