"""
RPE.py — Rotary Position Encoding for Atomiser
===============================================

Used classes:
    LocalRoPE2D               — core 2D RoPE with physical compression
    SelfAttentionRoPE         — self-attention + RoPE (Flash Attention)
    LocalCrossAttentionRoPE   — local cross-attention + optional RoPE
    PreNormRoPE               — PreNorm wrapper for RoPE self-attention

Physical compression:
    pos → pos / (scale + |pos|)  ∈ (-1, 1)
    Same physical distance (meters) → same rotation regardless of resolution.

RoPE convention:
    Self-attention:  Q and K both rotated by absolute position.
                     Relative position emerges from the dot product.
    Cross-attention: Only K rotated by relative delta (token - latent).
                     Q (latent) stays unrotated, treated as origin.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# CORE 2D ROPE
# =============================================================================

class LocalRoPE2D(nn.Module):
    """
    2D RoPE with physical distance compression.

    dim_head must be divisible by 4:
        - first half  (dim_head/2): X-axis rotation
        - second half (dim_head/2): Y-axis rotation
        - each half further split into two for the complex rotation pair

    Compression:  pos / (scale + |pos|)  →  (-1, 1)
        Linear near origin → good resolution for nearby tokens.
        Saturates at large distances → numerical stability.
    """

    def __init__(
        self,
        dim_head: int,
        base: float = 10000.0,
        compression_scale: float = 50.0,
        learnable_scale: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"

        self.dim_head  = dim_head
        self.num_heads = num_heads
        quarter_dim    = dim_head // 4

        # Standard RoPE inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim).float() / quarter_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Learnable compression scale + per-frequency amplitude scales
        if learnable_scale:
            self.log_scale = nn.Parameter(torch.tensor(math.log(compression_scale)))
            self.scale_x   = nn.Parameter(torch.ones(quarter_dim))
            self.scale_y   = nn.Parameter(torch.ones(quarter_dim))
        else:
            self.register_buffer("log_scale", torch.tensor(math.log(compression_scale)))
            self.register_buffer("scale_x",   torch.ones(quarter_dim))
            self.register_buffer("scale_y",   torch.ones(quarter_dim))

    @property
    def compression_scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def _compress(self, pos: torch.Tensor) -> torch.Tensor:
        """Map physical position (meters) → (-1, 1)."""
        return pos / (self.compression_scale + torch.abs(pos))

    # ── Public API ──────────────────────────────────────────────────────────

    def forward_self(
        self,
        q: torch.Tensor,      # [B, N, H, d]
        k: torch.Tensor,      # [B, N, H, d]
        pos_x: torch.Tensor,  # [B, N] meters (absolute)
        pos_y: torch.Tensor,  # [B, N] meters (absolute)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention: rotate both Q and K by absolute position.
        Relative position emerges from q_i · k_j via RoPE identity.
        """
        px = self._compress(pos_x)
        py = self._compress(pos_y)
        return self._rotate_self(q, px, py), self._rotate_self(k, px, py)

    def forward_cross(
        self,
        q: torch.Tensor,        # [B, L, H, d]
        k: torch.Tensor,        # [B, L, m, H, d]
        delta_x: torch.Tensor,  # [B, L, m] meters (relative: token - latent)
        delta_y: torch.Tensor,  # [B, L, m] meters (relative: token - latent)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention: rotate only K by relative delta.
        Q (latent) is treated as the origin — stays unrotated.
        """
        dx = self._compress(delta_x)
        dy = self._compress(delta_y)
        return q, self._rotate_cross(k, dx, dy)

    # ── Internal rotation helpers ────────────────────────────────────────────

    def _rotate_self(
        self,
        x: torch.Tensor,  # [B, N, H, d]
        px: torch.Tensor, # [B, N] compressed
        py: torch.Tensor, # [B, N] compressed
    ) -> torch.Tensor:
        half_d = self.dim_head // 2

        # [B, N] → [B, N, quarter_dim] → [B, N, 1, quarter_dim]
        angles_x = (px.unsqueeze(-1) * self.inv_freq * self.scale_x).unsqueeze(2)
        angles_y = (py.unsqueeze(-1) * self.inv_freq * self.scale_y).unsqueeze(2)

        x_rot = self._apply_rotation(x[..., :half_d], angles_x)
        y_rot = self._apply_rotation(x[..., half_d:], angles_y)
        return torch.cat([x_rot, y_rot], dim=-1)

    def _rotate_cross(
        self,
        x: torch.Tensor,  # [B, L, m, H, d]
        dx: torch.Tensor, # [B, L, m] compressed
        dy: torch.Tensor, # [B, L, m] compressed
    ) -> torch.Tensor:
        half_d = self.dim_head // 2

        # [B, L, m] → [B, L, m, quarter_dim] → [B, L, m, 1, quarter_dim]
        angles_x = (dx.unsqueeze(-1) * self.inv_freq * self.scale_x).unsqueeze(3)
        angles_y = (dy.unsqueeze(-1) * self.inv_freq * self.scale_y).unsqueeze(3)

        x_rot = self._apply_rotation(x[..., :half_d], angles_x)
        y_rot = self._apply_rotation(x[..., half_d:], angles_y)
        return torch.cat([x_rot, y_rot], dim=-1)

    @staticmethod
    def _apply_rotation(
        x: torch.Tensor,      # [..., H, quarter_dim*2]  (the half being rotated)
        angles: torch.Tensor, # [..., 1, quarter_dim]  (broadcast over H)
    ) -> torch.Tensor:
        """
        Apply complex rotation:
            [x1, x2] → [x1·cos(θ) − x2·sin(θ), x2·cos(θ) + x1·sin(θ)]

        x is split into two equal halves (x1, x2) along the last dim.
        angles broadcasts over the head dimension.
        """
        cos = angles.cos()  # [..., 1, quarter_dim]
        sin = angles.sin()

        x1, x2 = x.chunk(2, dim=-1)  # each [..., H, quarter_dim]
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)


# =============================================================================
# SELF-ATTENTION WITH ROPE
# =============================================================================

class SelfAttentionRoPE(nn.Module):
    """
    Multi-head self-attention with 2D RoPE using Flash Attention.

    Supports hybrid spatial + global latent layout:
        tokens[:num_spatial]  → spatial latents, get RoPE
        tokens[num_spatial:]  → global latents, no RoPE
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
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"

        self.heads      = heads
        self.dim_head   = dim_head
        self.use_rope   = use_rope
        self.dropout_p  = dropout

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        if use_rope:
            self.rope = LocalRoPE2D(
                dim_head=dim_head,
                base=rope_base,
                compression_scale=rope_compression_scale,
                learnable_scale=rope_learnable_scale,
                num_heads=heads,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        pos_x: Optional[torch.Tensor] = None,       # [B, N] meters
        pos_y: Optional[torch.Tensor] = None,       # [B, N] meters
        num_spatial: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,   # [B, N, 2] alternative
        gsd: Optional[torch.Tensor] = None,         # unused, kept for API compat
        attn_mask: Optional[torch.Tensor] = None,   # [B, 1, 1, N] or broadcastable
                                                      # to [B, H, N, N]; bool,
                                                      # True = attend/keep. None ->
                                                      # no masking (existing
                                                      # behavior, fully backward
                                                      # compatible).
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, d    = self.heads, self.dim_head

        # Accept positions in [B, N, 2] format
        if positions is not None and pos_x is None:
            pos_x = positions[..., 0]
            pos_y = positions[..., 1]

        qkv    = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, H, d) for t in qkv]

        # Apply RoPE
        if self.use_rope and self.rope is not None and pos_x is not None:
            if num_spatial is not None and num_spatial < N:
                # Spatial tokens get RoPE, global tokens do not
                q_s, q_g = q[:, :num_spatial], q[:, num_spatial:]
                k_s, k_g = k[:, :num_spatial], k[:, num_spatial:]
                q_s, k_s = self.rope.forward_self(
                    q_s, k_s,
                    pos_x[:, :num_spatial],
                    pos_y[:, :num_spatial],
                )
                q = torch.cat([q_s, q_g], dim=1)
                k = torch.cat([k_s, k_g], dim=1)
            else:
                q, k = self.rope.forward_self(q, k, pos_x, pos_y)

        # Flash Attention — transpose to [B, H, N, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, H * d)
        return self.to_out(out)


# =============================================================================
# LOCAL CROSS-ATTENTION WITH ROPE
# =============================================================================

class LocalCrossAttentionRoPE(nn.Module):
    """
    Local cross-attention with optional 2D RoPE.

    Each latent attends to its own set of m nearby tokens (from geographic
    pruning). RoPE is applied to K only (relative delta: token - latent).

    When use_rope=False (encoder_use_rpe: False in config), this reduces
    to standard local cross-attention with no positional encoding.
    """

    def __init__(
        self,
        dim_query: int,
        dim_context: int,
        dim_out: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_compression_scale: float = 50.0,
        rope_learnable_scale: bool = True,
        rope_reference_gsd: float = 0.2,  # kept for API compatibility
    ):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"

        self.heads    = heads
        self.dim_head = dim_head
        self.scale    = dim_head ** -0.5
        self.use_rope = use_rope

        inner_dim = heads * dim_head
        self.to_q   = nn.Linear(dim_query,   inner_dim, bias=False)
        self.to_k   = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_v   = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_out)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = LocalRoPE2D(
                dim_head=dim_head,
                base=rope_base,
                compression_scale=rope_compression_scale,
                learnable_scale=rope_learnable_scale,
                num_heads=heads,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,        # [B, L, dim_query]   latents
        context: torch.Tensor,  # [B, L, m, dim_context] sampled tokens
        mask: Optional[torch.Tensor] = None,     # [B, L, m] True = valid
        delta_x: Optional[torch.Tensor] = None,  # [B, L, m] meters
        delta_y: Optional[torch.Tensor] = None,  # [B, L, m] meters
        gsd: Optional[torch.Tensor] = None,      # unused, kept for API compat
    ) -> torch.Tensor:
        B, L, _ = x.shape
        m       = context.shape[2]
        H, d    = self.heads, self.dim_head

        q = self.to_q(x).view(B, L, H, d)              # [B, L, H, d]
        K = self.to_k(context).view(B, L, m, H, d)     # [B, L, m, H, d]
        V = self.to_v(context).view(B, L, m, H, d)     # [B, L, m, H, d]

        # Rotate K by relative delta (Q stays as origin)
        if self.use_rope and self.rope is not None and delta_x is not None:
            q, K = self.rope.forward_cross(q, K, delta_x, delta_y)

        # Attention scores: [B, L, H, m]
        scores = torch.einsum("b l h d, b l m h d -> b l h m", q, K) * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b l h m, b l m h d -> b l h d", attn, V)
        return self.to_out(out.reshape(B, L, H * d))


# =============================================================================
# PRENORM WRAPPER
# =============================================================================

class PreNormRoPE(nn.Module):
    """
    PreNorm wrapper that forwards pos_x/pos_y/num_spatial to the inner
    SelfAttentionRoPE module.
    """

    def __init__(self, dim: int, fn: SelfAttentionRoPE):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn

    def forward(
        self,
        x: torch.Tensor,
        pos_x: Optional[torch.Tensor] = None,
        pos_y: Optional[torch.Tensor] = None,
        num_spatial: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.fn(
            self.norm(x),
            pos_x=pos_x,
            pos_y=pos_y,
            num_spatial=num_spatial,
            positions=positions,
            **kwargs,
        )
