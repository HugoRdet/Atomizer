import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pytorch_lightning as pl
from training.utils.token_building.fourier_features import fourier_encode
# ---------------------------------
# Utilities
# ---------------------------------

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

# ---------------------------------
# PreNorm wrapper
# ---------------------------------
class PreNorm(nn.Module):
    """Apply LayerNorm before the function, with optional context normalization."""
    
    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fn = fn
        self.norm_context = nn.LayerNorm(context_dim, eps=eps) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None and ('context' in kwargs) and (kwargs['context'] is not None):
            # Create new kwargs dict to avoid mutating caller's dict
            kwargs = {**kwargs, 'context': self.norm_context(kwargs['context'])}
        return self.fn(x, **kwargs)



def exists(val):
    return val is not None


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


#@torch.compile
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult * 2)
        self.w2 = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, gate = self.w1(x).chunk(2, dim=-1)
        x = x1 * F.gelu(gate)
        return self.dropout(self.w2(x))


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.,
        use_flash: bool = True
    ):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # one linear for Q,K,V
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (B, N, dim)
        B, N, _ = x.shape

        # project and split
        qkv = self.to_qkv(x)         # (B, N, 3·inner)
        q, k, v = qkv.chunk(3, dim=-1)

        # to (B·H, N, D)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=self.heads)

        if self.use_flash:
            # prepare mask for FlashAttention
            attn_mask = None
            if exists(mask):
                # mask: (B, N) -> (B, N, N)
                m = mask.unsqueeze(1).expand(-1, N, -1)            # (B, N, N)
                attn_mask = repeat(m, "b i j -> (b h) i j", h=self.heads)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.to_out[1].p,
                is_causal=False
            )
        else:
            # classic
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                m = mask.unsqueeze(1).expand(-1, N, -1)            # (B, N, N)
                m = repeat(m, "b i j -> (b h) i j", h=self.heads)
                sim = sim.masked_fill(~m, float("-inf"))
            attn = sim.softmax(dim=-1)
            attn = self.to_out[1](attn)
            out = einsum("b i j, b j d -> b i d", attn, v)

        # back to (B, N, inner)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out[0](out)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LocalCrossAttention(nn.Module):
    """
    Cross-attention where each query has its own local context.
    Works for both encoder (latent→token) and decoder (token→latent).
    """
    
    def __init__(self, dim_query, dim_context, dim_out, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim_query, inner_dim, bias=False)
        self.to_k = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context, mask=None, bias=None):
        """
        Args:
            x: [B, N, dim_query] - queries
            context: [B, N, k, dim_context] - local context per query
            mask: [B, N, k] optional
            bias: [B, N, k] optional
        Returns:
            [B, N, dim_out]
        """
        B, N, _ = x.shape
        k = context.shape[2]
        H, d = self.heads, self.dim_head
        
        q = self.to_q(x).view(B, N, H, d)
        K = self.to_k(context).view(B, N, k, H, d)
        V = self.to_v(context).view(B, N, k, H, d)
        
        scores = torch.einsum('b n h d, b n k h d -> b n h k', q, K) * self.scale
        
        if bias is not None:
            scores = scores + bias.unsqueeze(2)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_weights = F.softmax(attn, dim=-1)  # [B, H, N, k]


        
        out = torch.einsum('b n h k, b n k h d -> b n h d', attn, V)
        
        return self.to_out(out.reshape(B, N, H * d))
    

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_flash=True, id=0):
        super().__init__()
        context_dim = context_dim or query_dim
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, query_dim),
            nn.Dropout(dropout)
        )
        
        # Store dropout separately for manual attention
        self.dropout = nn.Dropout(dropout)
        # Will hold the last attention weights (manual path only)
        self.last_attn = None
        self.viz=False
        
        


    
    


    def forward(self, x, context, mask=None, bias=None, id=0):
        """
        Args:
            x: Query tensor (B, Nq, query_dim)
            context: Key/Value tensor (B, Nk, context_dim)
            mask: Optional attention mask (B, Nq, Nk) or (B, Nk)
            bias: Optional attention bias to add to attention scores
                  Shape should be (B, heads, Nq, Nk) or broadcastable to it
            id: Layer identifier
        """
        B, Nq, _ = x.shape
        Nk = context.shape[1]
        
        # 1) Project Q, K, V
        q = self.to_q(x)  # (B, Nq, inner)
        k = self.to_k(context)  # (B, Nk, inner)
        v = self.to_v(context)  # (B, Nk, inner)
        
        # 2) Split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        
        # Handle bias - if provided, we need to use manual attention
        if bias is not None or not self.use_flash:
            # Manual attention computation
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, Nq, Nk)
            
            # Add bias if provided
            if bias is not None:
                # Create bias_weight parameter lazily on first use
                
                #if not self.bias_weight is None:
                #    print("layer idx:", id,"    bias weight:", self.bias_weight.item(),bias.max().item(),bias.min().item())
            
                # Handle bias shape [batch, nb_tokens]
                if bias.dim() == 2:  # (B, nb_tokens)
                    # Assuming nb_tokens corresponds to key positions (Nk)
                    # Reshape to (B, 1, 1, Nk) for broadcasting across heads and queries
                    bias = bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, nb_tokens)
                elif bias.dim() == 3:  # (B, Nq, Nk)
                    bias = bias.unsqueeze(1)  # (B, 1, Nq, Nk)
                elif bias.dim() == 1:  # (nb_tokens,)
                    bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, nb_tokens)
                
                scores = scores +  bias 
            
            # Apply mask if provided
            if mask is not None:
                if mask.dim() == 2:  # (B, Nk)
                    mask = mask.unsqueeze(1).expand(-1, Nq, -1)  # (B, Nq, Nk)
                if mask.dim() == 3:  # (B, Nq, Nk)
                    mask = mask.unsqueeze(1)  # (B, 1, Nq, Nk)
                scores = scores.masked_fill(~mask, float('-inf'))
            
            # Apply softmax
            #print("scores :",scores.shape)

            
            attn = F.softmax(scores, dim=-1)
            #print("scores :",attn.shape)

            if self.viz:
                return attn
            

            
            
            
            attn = self.dropout(attn)
            self.last_attn = attn.detach()


            
            
            # Apply attention to values
            out = torch.matmul(attn, v)  # (B, heads, Nq, dim_head)

            
        else:
            # Flash attention path (only when no bias is provided)
            attn_mask = None
            if mask is not None:
                # mask should be (B, Nq, Nk) - True for valid positions
                if mask.dim() == 2:
                    # If mask is (B, Nk), expand to (B, Nq, Nk)
                    mask = mask.unsqueeze(1).expand(-1, Nq, -1)
                attn_mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            self.last_attn = None
        
        # 3) Recombine heads and project
        
        out = rearrange(out, "b h n d -> b n (h d)")
        
        return self.to_out(out)
    
class LatentAttentionPooling(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.cross  = CrossAttention(
            query_dim   = dim,
            context_dim = dim,
            heads       = heads,
            dim_head    = dim_head,
            dropout     = dropout
        )

    def forward(self, x):
        b = x.size(0)
        q = repeat(self.query, '1 1 d -> b 1 d', b=b)
        out = self.cross(q, context=x, mask=None)
        return out.squeeze(1)
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
    
    
class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        h = self.heads
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        q = q.softmax(dim=-1) * self.scale
        k = k.softmax(dim=-2)
        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)
    

class SelfAttentionWithGaussianBias(nn.Module):
    """
    Standard self-attention + Gaussian distance bias for spatial latents.
    
    Attention bias matrix:
                    spatial    global
        spatial   [Gaussian]   [0]
        global       [0]       [0]
    
    Global latents attend freely everywhere, spatial latents have locality prior.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        sigma: float = 3.0,
        learnable_sigma: bool = False,
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
    
    @property
    def sigma(self):
        return self.log_sigma.exp()
    
    def compute_position_bias(
        self, 
        positions: torch.Tensor, 
        num_spatial: int,
        total_latents: int
    ) -> torch.Tensor:
        """
        Compute Gaussian bias for spatial latents, 0 for global.
        
        Args:
            positions: [B, L_spatial, 2] spatial latent positions
            num_spatial: L_spatial
            total_latents: L_spatial + L_global
            
        Returns:
            [B, H, L_total, L_total] attention bias
        """
        B = positions.shape[0]
        L_spatial = num_spatial
        L_total = total_latents
        L_global = L_total - L_spatial
        device = positions.device
        
        # Spatial-spatial: Gaussian bias
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, Ls, Ls, 2]
        dist_sq = (diff ** 2).sum(dim=-1)  # [B, Ls, Ls]
        
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)  # [1, H, 1, 1]
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)  # [B, H, Ls, Ls]
        
        # Build full bias matrix with zeros for global
        # [B, H, L_total, L_total]
        full_bias = torch.zeros(B, self.heads, L_total, L_total, device=device)
        full_bias[:, :, :L_spatial, :L_spatial] = spatial_bias
        
        return full_bias
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor = None,
        num_spatial: int = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] all latents (spatial + global)
            positions: [B, L_spatial, 2] spatial latent positions
            num_spatial: number of spatial latents (first num_spatial are spatial)
        """
        B, L, _ = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, L, L]
        
        # Add position bias if provided
        if positions is not None and num_spatial is not None:
            pos_bias = self.compute_position_bias(positions, num_spatial, L)
            attn = attn + pos_bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.to_out(out)
    


class PreNormWithPositions(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, positions=None, num_spatial=None, **kwargs):
        return self.fn(self.norm(x), positions=positions, num_spatial=num_spatial, **kwargs)


def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache: return f(*args, **kwargs)
        nonlocal cache
        if key in cache: return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn



"""
Self-Attention with Polar Positional Encoding

FIXED VERSION: Uses pre-registered frequency buffers to avoid memory explosion
during backward pass.

The key fix: frequencies are registered as buffers in __init__, not created
dynamically in every forward call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttentionWithPolarPE(nn.Module):
    """
    Self-attention with polar positional encoding concatenated to keys/values.
    
    For each query latent i, keys/values from latent j are augmented with
    polar-encoded relative position from i to j. This matches the encoder/decoder
    cross-attention approach.
    
    Architecture:
        Q[i] = W_q @ latent[i]
        K[i,j] = W_k @ concat(latent[j], polar_PE(pos[j] - pos[i]))
        V[i,j] = W_v @ concat(latent[j], polar_PE(pos[j] - pos[i]))
    
    Distance normalization:
        - Cross-attention uses small scale (~3m) since tokens are local to each latent
        - Self-attention uses larger scale (~50m) since latents span the full image
        - Sigmoid compression: r / (scale + r) maps [0, ∞) → [0, 1)
    
    Memory: O(L² × (D + pos_dim)) for expanded K/V tensors
    
    IMPORTANT: Frequencies are pre-registered as buffers to avoid memory leaks
    during backward pass.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_freq_bands: int = 32,
        max_freq: int = 32,
        physical_scale: float = 51.5,
    ):
        """
        Args:
            dim: Latent dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Attention dropout
            num_freq_bands: Number of Fourier frequency bands
            max_freq: Maximum frequency for Fourier encoding
            physical_scale: Distance normalization scale in meters.
                Default ~51.5m (half image extent) gives:
                - 3m → 0.06 (fine local resolution)
                - 50m → 0.5 (midpoint)
                - 100m → 0.67 (still distinguishable)
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.physical_scale = physical_scale
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq
        
        inner_dim = heads * dim_head
        
        # =====================================================================
        # PRE-REGISTER Fourier frequencies as buffer (CRITICAL FOR MEMORY!)
        # This avoids creating new tensors on every forward call
        # =====================================================================
        if num_freq_bands == -1:
            freqs = torch.linspace(1., max_freq, int(max_freq))
        else:
            freqs = torch.linspace(1., max_freq / 2, num_freq_bands)
        self.register_buffer('freqs', freqs)
        
        # Position encoding dimension: 2 * (num_bands * 2 + 1) for r and theta
        self.pos_dim = 2 * (len(freqs) * 2 + 1)
        
        # Q projection: just from latent embedding
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        
        # K, V projection: from latent embedding + polar position encoding
        self.to_k = nn.Linear(dim + self.pos_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim + self.pos_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier encoding using PRE-REGISTERED frequencies.
        
        Args:
            x: [...] input values (any shape)
            
        Returns:
            [..., F*2 + 1] encoded values (sin, cos, original)
        """
        x_expanded = x.unsqueeze(-1)  # [..., 1]
        x_scaled = x_expanded * self.freqs * math.pi  # [..., F] - uses buffer!
        encoding = torch.cat([
            x_scaled.sin(),   # [..., F]
            x_scaled.cos(),   # [..., F]
            x_expanded,       # [..., 1] (original value)
        ], dim=-1)  # [..., F*2 + 1]
        return encoding
    
    def compute_polar_encoding(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute polar encoding for all spatial latent pairs.
        
        Args:
            positions: [B, L_spatial, 2] spatial latent positions in meters
            
        Returns:
            encoding: [B, L_spatial, L_spatial, pos_dim]
        """
        # Relative positions: pos[j] - pos[i]
        diff = positions.unsqueeze(1) - positions.unsqueeze(2)  # [B, Ls, Ls, 2]
        delta_x = diff[..., 0]  # [B, Ls, Ls]
        delta_y = diff[..., 1]  # [B, Ls, Ls]
        
        # Polar coordinates
        r = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
        theta = torch.atan2(delta_y, delta_x)
        
        # Sigmoid compression for r: [0, ∞) → [0, 1)
        r_compressed = r / (self.physical_scale + r)
        
        # Normalize theta: [-π, π] → [-1, 1]
        theta_norm = theta / math.pi
        
        # Fourier encode using pre-registered frequencies
        r_encoded = self._fourier_encode(r_compressed)      # [B, Ls, Ls, F*2+1]
        theta_encoded = self._fourier_encode(theta_norm)    # [B, Ls, Ls, F*2+1]
        
        # Concatenate
        encoding = torch.cat([r_encoded, theta_encoded], dim=-1)  # [B, Ls, Ls, pos_dim]
        
        return encoding
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor = None,
        num_spatial: int = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] all latents (spatial + global concatenated)
            positions: [B, L_spatial, 2] spatial latent positions in meters
            num_spatial: number of spatial latents (first num_spatial are spatial)
        """
        B, L, D = x.shape
        
        # If no positions provided, fall back to standard self-attention
        if positions is None or num_spatial is None:
            # Standard self-attention without position encoding
            qkv = torch.cat([
                self.to_q(x),
                self.to_k(F.pad(x, (0, self.pos_dim))),  # Pad with zeros
                self.to_v(F.pad(x, (0, self.pos_dim)))
            ], dim=-1)
            q, k, v = qkv.chunk(3, dim=-1)
            
            q = q.view(B, L, self.heads, self.dim_head).transpose(1, 2)
            k = k.view(B, L, self.heads, self.dim_head).transpose(1, 2)
            v = v.view(B, L, self.heads, self.dim_head).transpose(1, 2)
            
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, -1)
            return self.to_out(out)
        
        L_spatial = num_spatial
        L_global = L - L_spatial
        device = x.device
        
        # Split spatial and global latents
        x_spatial = x[:, :L_spatial, :]  # [B, Ls, D]
        x_global = x[:, L_spatial:, :]   # [B, Lg, D]
        
        # === QUERIES ===
        # All queries from all latents (no position info)
        q = self.to_q(x)  # [B, L, inner_dim]
        q = q.view(B, L, self.heads, self.dim_head).transpose(1, 2)  # [B, H, L, D_head]
        
        # === KEYS & VALUES for spatial-spatial attention ===
        # Compute polar encoding: [B, Ls, Ls, pos_dim]
        polar_encoding = self.compute_polar_encoding(positions)
        
        # Expand spatial latents for all query-key pairs: [B, Ls, Ls, D]
        x_spatial_expanded = x_spatial.unsqueeze(1).expand(B, L_spatial, L_spatial, D)
        
        # Concatenate latents with position encoding: [B, Ls, Ls, D + pos_dim]
        x_spatial_with_pe = torch.cat([x_spatial_expanded, polar_encoding], dim=-1)
        
        # Project to K, V: [B, Ls, Ls, inner_dim]
        k_spatial = self.to_k(x_spatial_with_pe)
        v_spatial = self.to_v(x_spatial_with_pe)
        
        # Reshape: [B, H, Ls, Ls, D_head]
        k_spatial = k_spatial.view(B, L_spatial, L_spatial, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
        v_spatial = v_spatial.view(B, L_spatial, L_spatial, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
        
        # === KEYS & VALUES for global latents (no position info) ===
        if L_global > 0:
            # Global latents with zero position encoding
            zeros_pe = torch.zeros(B, L_global, self.pos_dim, device=device, dtype=x.dtype)
            x_global_with_pe = torch.cat([x_global, zeros_pe], dim=-1)  # [B, Lg, D + pos_dim]
            
            k_global = self.to_k(x_global_with_pe)  # [B, Lg, inner_dim]
            v_global = self.to_v(x_global_with_pe)
            
            k_global = k_global.view(B, L_global, self.heads, self.dim_head).transpose(1, 2)  # [B, H, Lg, D_head]
            v_global = v_global.view(B, L_global, self.heads, self.dim_head).transpose(1, 2)
        
        # === ATTENTION COMPUTATION ===
        # Split into: spatial queries and global queries
        q_spatial = q[:, :, :L_spatial, :]  # [B, H, Ls, D_head]
        q_global = q[:, :, L_spatial:, :] if L_global > 0 else None  # [B, H, Lg, D_head]
        
        # --- Spatial queries ---
        # Spatial query i attending to spatial key j (with position encoding)
        # q_spatial: [B, H, Ls, D_head] -> [B, H, Ls, 1, D_head]
        attn_spatial_to_spatial = (q_spatial.unsqueeze(3) * k_spatial).sum(dim=-1) * self.scale
        # [B, H, Ls, Ls]
        
        if L_global > 0:
            # Spatial queries attending to global keys (no position info)
            attn_spatial_to_global = torch.matmul(q_spatial, k_global.transpose(-1, -2)) * self.scale
            # [B, H, Ls, Lg]
            
            # Concatenate attention scores for spatial queries
            attn_spatial = torch.cat([attn_spatial_to_spatial, attn_spatial_to_global], dim=-1)
            # [B, H, Ls, L]
        else:
            attn_spatial = attn_spatial_to_spatial
        
        # --- Global queries ---
        if L_global > 0:
            # Global queries attend to all keys with zero position encoding
            # For spatial keys: need k with zero PE for all spatial latents
            zeros_pe_spatial = torch.zeros(B, L_spatial, self.pos_dim, device=device, dtype=x.dtype)
            x_spatial_zero_pe = torch.cat([x_spatial, zeros_pe_spatial], dim=-1)
            
            k_spatial_for_global = self.to_k(x_spatial_zero_pe)
            k_spatial_for_global = k_spatial_for_global.view(B, L_spatial, self.heads, self.dim_head).transpose(1, 2)
            
            v_spatial_for_global = self.to_v(x_spatial_zero_pe)
            v_spatial_for_global = v_spatial_for_global.view(B, L_spatial, self.heads, self.dim_head).transpose(1, 2)
            
            # Global-to-spatial attention
            attn_global_to_spatial = torch.matmul(q_global, k_spatial_for_global.transpose(-1, -2)) * self.scale
            # [B, H, Lg, Ls]
            
            # Global-to-global attention
            attn_global_to_global = torch.matmul(q_global, k_global.transpose(-1, -2)) * self.scale
            # [B, H, Lg, Lg]
            
            # Concatenate attention scores for global queries
            attn_global = torch.cat([attn_global_to_spatial, attn_global_to_global], dim=-1)
            # [B, H, Lg, L]
            
            # Combine all attention scores
            attn = torch.cat([attn_spatial, attn_global], dim=2)
            # [B, H, L, L]
        else:
            attn = attn_spatial
        
        # Softmax
        attn = F.softmax(attn, dim=-1)  # [B, H, L, L]
        
        # === VALUE AGGREGATION ===
        # Spatial queries aggregate from spatial values (with PE) and global values
        # Global queries aggregate from all values (with zero PE)
        
        # Spatial queries -> spatial values
        attn_spatial_to_spatial_weights = attn[:, :, :L_spatial, :L_spatial]  # [B, H, Ls, Ls]
        out_spatial_from_spatial = torch.einsum('bhij,bhijd->bhid', attn_spatial_to_spatial_weights, v_spatial)
        # [B, H, Ls, D_head]
        
        if L_global > 0:
            # Spatial queries -> global values
            attn_spatial_to_global_weights = attn[:, :, :L_spatial, L_spatial:]  # [B, H, Ls, Lg]
            out_spatial_from_global = torch.matmul(attn_spatial_to_global_weights, v_global)
            # [B, H, Ls, D_head]
            
            out_spatial = out_spatial_from_spatial + out_spatial_from_global
            
            # Global queries -> all values (with zero PE)
            attn_global_to_spatial_weights = attn[:, :, L_spatial:, :L_spatial]  # [B, H, Lg, Ls]
            attn_global_to_global_weights = attn[:, :, L_spatial:, L_spatial:]   # [B, H, Lg, Lg]
            
            out_global_from_spatial = torch.matmul(attn_global_to_spatial_weights, v_spatial_for_global)
            out_global_from_global = torch.matmul(attn_global_to_global_weights, v_global)
            out_global = out_global_from_spatial + out_global_from_global
            # [B, H, Lg, D_head]
            
            # Combine outputs
            out = torch.cat([out_spatial, out_global], dim=2)
            # [B, H, L, D_head]
        else:
            out = out_spatial_from_spatial
        
        out = out.transpose(1, 2).reshape(B, L, -1)  # [B, L, inner_dim]
        return self.to_out(out)


class PreNormWithPositions(nn.Module):
    """PreNorm wrapper that passes positions and num_spatial to the inner function."""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.dim = dim
    
    def forward(self, x, positions=None, num_spatial=None, **kwargs):
        # Safety check: x should be [B, L, D] with D == self.dim
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"PreNormWithPositions: expected x with shape [B, L, {self.dim}], "
                f"got shape {list(x.shape)}. "
                f"positions shape: {list(positions.shape) if positions is not None else None}"
            )
        
        x = self.norm(x)
        if positions is not None:
            return self.fn(x, positions=positions, num_spatial=num_spatial, **kwargs)
        return self.fn(x, **kwargs)
    





class SelfAttentionWithGaussianBias(nn.Module):
    """
    Self-attention with Gaussian distance bias for spatial latents.
    
    Instead of concatenating position encodings to K/V (expensive),
    this adds a distance-based bias directly to attention scores (cheap).
    
    Bias formula: bias[i,j] = -dist(i,j)² / (2σ²)
    
    This creates a soft locality prior where nearby latents attend more strongly.
    
    Memory: O(B × H × L × L) for bias matrix
    vs PolarPE: O(B × L × L × (D + pos_dim)) for expanded K/V
    
    With L=1225, H=8:
        GaussianBias: 1225² × 8 × 4 bytes = 48 MB
        PolarPE: 1225² × 642 × 4 bytes = 3.8 GB
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
        """
        Args:
            dim: Latent dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Attention dropout
            sigma: Initial Gaussian sigma (in meters, same units as positions)
            learnable_sigma: If True, sigma is learned per-head
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        
        # Fused QKV projection (memory efficient)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Learnable sigma per head (in log space for numerical stability)
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma)))
        else:
            self.register_buffer('log_sigma', torch.full((heads,), math.log(sigma)))
        
        # Learnable bias for global latents (they have no position)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
    
    @property
    def sigma(self):
        """Get sigma values (exponentiated from log space)."""
        return self.log_sigma.exp()
    
    def compute_distance_bias(
        self,
        positions: torch.Tensor,
        num_spatial: int,
        total_latents: int,
    ) -> torch.Tensor:
        """
        Compute Gaussian distance bias matrix.
        
        Args:
            positions: [B, L_spatial, 2] spatial latent positions in meters
            num_spatial: Number of spatial latents
            total_latents: Total number of latents (spatial + global)
            
        Returns:
            bias: [B, H, L_total, L_total] attention bias
        """
        B = positions.shape[0]
        L_spatial = num_spatial
        L_total = total_latents
        device = positions.device
        dtype = positions.dtype
        
        # Compute pairwise squared distances for spatial latents
        # positions: [B, Ls, 2]
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, Ls, Ls, 2]
        dist_sq = (diff ** 2).sum(dim=-1)  # [B, Ls, Ls]
        
        # Gaussian bias: -dist² / (2σ²)
        # sigma: [H] -> [1, H, 1, 1]
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)
        
        # Spatial-spatial bias: [B, H, Ls, Ls]
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)
        
        # Build full bias matrix including global latents
        full_bias = torch.zeros(B, self.heads, L_total, L_total, device=device, dtype=dtype)
        
        # Fill spatial-spatial block
        full_bias[:, :, :L_spatial, :L_spatial] = spatial_bias
        
        # Global latents get a learnable constant bias
        # This allows the model to learn how much global latents should attend
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
        """
        Forward pass.
        
        Args:
            x: [B, L, D] all latents (spatial + global)
            positions: [B, L_spatial, 2] positions for spatial latents
            num_spatial: Number of spatial latents
            
        Returns:
            out: [B, L, D] output features
        """
        B, L, D = x.shape
        
        # Fused QKV projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        # q, k, v: [B, H, L, D_head]
        
        # Standard attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, L, L]
        
        # Add distance bias if positions provided
        if positions is not None and num_spatial is not None:
            bias = self.compute_distance_bias(positions, num_spatial, L)
            attn = attn + bias
        
        # Softmax and value aggregation
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, D_head]
        
        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, L, -1)  # [B, L, inner_dim]
        return self.to_out(out)

