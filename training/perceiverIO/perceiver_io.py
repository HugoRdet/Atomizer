"""
Perceiver-IO Encoder-Decoder
==============================

Task-agnostic Perceiver-IO following Jaegle et al. (2021).

Now factored into three classes:

    PerceiverEncoder  : input tokens -> latent representation.
                        Shared across tasks in multi-task setups.
    PerceiverDecoder  : latents + queries -> output logits.
                        One decoder per task in multi-task setups
                        (different num_classes -> different to_logits;
                         different query strategies live above this).
    PerceiverIO       : composition of the two for single-task use.
                        Public API unchanged from the original module:
                        same constructor signature, same forward(),
                        same encode()/decode() methods.

Existing single-task code (PerceiverSeg, PerceiverCls, etc.) using
PerceiverIO works unchanged — only the internal layout changed.
For multi-task, a wrapper can instantiate one PerceiverEncoder and a
ModuleDict of PerceiverDecoders, sharing the encoder while specialising
the decoder per task.
"""

from functools import wraps

import torch
import torch.nn as nn
from einops import repeat

from .attention import Attention, PreNorm, FeedForward
from torch.profiler import record_function



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
# ENCODER
# =============================================================================

class PerceiverEncoder(nn.Module):
    """
    Perceiver-IO encoder: input tokens -> latent representation via
    iterative cross-attention + self-attention.

    Args:
        input_dim: Dimension of input tokens.
        num_latents: Number of latent vectors.
        latent_dim: Dimension of each latent vector.
        depth: Number of cross-attn -> self-attn blocks.
        cross_heads, latent_heads: head counts.
        cross_dim_head, latent_dim_head: dim per head.
        self_per_cross_attn: self-attn blocks per cross-attn.
        weight_tie_layers: share weights across blocks (block 0 always unique).
        attn_dropout, ff_dropout: dropouts.
    """

    def __init__(
        self,
        input_dim,
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
    ):
        super().__init__()

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.latent_dim = latent_dim

        # ── Encoder layer factories ─────────────────────────────────
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

    def forward(self, data, mask=None):
        """
        Encode input tokens into latent representation.

        Args:
            data: [B, N, input_dim] input tokens.
            mask: [B, N] bool mask (True = valid token), or None.

        Returns:
            latents: [B, L, latent_dim]
        """
        B = data.shape[0]
        x = repeat(self.latents, 'n d -> b n d', b=B)

        for cross_attn, cross_ff, self_attns in self.encoder_layers:
            with record_function("Encoder Cross Attention"):
                x = cross_attn(x, context=data, mask=mask) + x
                x = cross_ff(x) + x

            with record_function("Self Attention"):

                for self_attn, self_ff in self_attns:
                    x = self_attn(x) + x
                    x = self_ff(x) + x

        return x


# =============================================================================
# DECODER
# =============================================================================

class PerceiverDecoder(nn.Module):
    """
    Perceiver-IO decoder: queries cross-attend to latents, optional
    feedforward, then linear projection to output_dim.

    A single PerceiverDecoder is task-specific: the to_logits projection
    fixes the output dimension. In multi-task settings, instantiate one
    PerceiverDecoder per task and dispatch on task name.

    Note: this module operates on already-prepared queries — query
    construction (Fourier position encodings, learned CLS tokens, etc.)
    lives in the higher-level model wrapper.

    Args:
        query_dim: Dimension of decoder query tokens.
        latent_dim: Dimension of encoder latents.
        output_dim: Dimension of final output (e.g., num_classes).
        cross_heads: Heads for the decoder cross-attention.
        cross_dim_head: Dim per head.
        attn_dropout, ff_dropout: dropouts.
        decoder_ff: Add feedforward after the decoder cross-attention.
    """

    def __init__(
        self,
        query_dim,
        latent_dim,
        output_dim,
        cross_heads=1,
        cross_dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        decoder_ff=False,
    ):
        super().__init__()

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
        self.to_logits = nn.Linear(query_dim, output_dim)

    def forward(self, latents, queries):
        """
        Args:
            latents: [B, L, latent_dim] from encoder.
            queries: [B, M, query_dim] decoder queries.

        Returns:
            logits: [B, M, output_dim]
        """
        with record_function("Decoder"):
            out = self.decoder_cross_attn(queries, context=latents)
            if self.decoder_ff is not None:
                out = out + self.decoder_ff(out)
            return self.to_logits(out)


# =============================================================================
# PERCEIVER-IO (composition; preserves original public API)
# =============================================================================

class PerceiverIO(nn.Module):
    """
    Perceiver-IO: encoder maps arbitrary input to fixed latents,
    decoder maps latents to arbitrary output via query cross-attention.

    Thin composition of PerceiverEncoder + PerceiverDecoder. The public
    API (constructor signature, forward(), encode(), decode()) is the
    same as the pre-refactor implementation, so existing single-task
    code (PerceiverSeg, PerceiverCls) keeps working without changes.

    Args:
        input_dim: Dimension of input tokens.
        query_dim: Dimension of decoder query tokens.
        output_dim: Dimension of final output (e.g., num_classes).
        num_latents, latent_dim, depth: encoder size config.
        cross_heads, latent_heads: head counts (cross_heads is shared
                                   between encoder and decoder cross-attention).
        cross_dim_head, latent_dim_head: dim per head.
        self_per_cross_attn: encoder self-attn per cross-attn.
        weight_tie_layers: share weights across encoder blocks > 0.
        attn_dropout, ff_dropout: dropouts.
        decoder_ff: add feedforward after decoder cross-attention.
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

        self.encoder = PerceiverEncoder(
            input_dim=input_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
            depth=depth,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            self_per_cross_attn=self_per_cross_attn,
            weight_tie_layers=weight_tie_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self.decoder = PerceiverDecoder(
            query_dim=query_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            decoder_ff=decoder_ff,
        )

    # ── Backward-compat: keep .latents and .encoder_layers reachable ──
    # Some external code might inspect these attributes; expose them via
    # property pass-throughs so the refactor is fully transparent.
    @property
    def latents(self):
        return self.encoder.latents

    @property
    def encoder_layers(self):
        return self.encoder.encoder_layers

    @property
    def decoder_cross_attn(self):
        return self.decoder.decoder_cross_attn

    @property
    def decoder_ff(self):
        return self.decoder.decoder_ff

    @property
    def to_logits(self):
        return self.decoder.to_logits

    # ── Original API ────────────────────────────────────────────────

    def encode(self, data, mask=None):
        """
        Encode input tokens into latent representation.

        Args:
            data: [B, N, input_dim] input tokens.
            mask: [B, N] bool mask (True = valid token), or None.

        Returns:
            latents: [B, L, latent_dim]
        """
        return self.encoder(data, mask=mask)

    def decode(self, latents, queries):
        """
        Decode from latents using query tokens.

        Args:
            latents: [B, L, latent_dim] from encoder.
            queries: [B, M, query_dim] decoder queries.

        Returns:
            logits: [B, M, output_dim]
        """
        return self.decoder(latents, queries)

    def forward(self, data, mask=None, queries=None):
        """
        Full forward: encode data, decode with queries.

        Args:
            data: [B, N, input_dim]
            mask: [B, N] optional
            queries: [B, M, query_dim] (or None to return latents only).

        Returns:
            logits: [B, M, output_dim] if queries is provided,
                    else latents: [B, L, latent_dim]
        """
        latents = self.encode(data, mask=mask)
        if queries is None:
            return latents
        return self.decode(latents, queries)
