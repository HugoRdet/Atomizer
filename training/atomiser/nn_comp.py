import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pytorch_lightning as pl

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


@torch.compile
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