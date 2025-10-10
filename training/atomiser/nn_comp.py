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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        
        # Learned parameter to control bias importance (initialized to 1.0)
        if id>0:
            #self.bias_weight = nn.Parameter(torch.randn(1)*10+1)
            self.bias_weight = nn.Parameter(torch.Tensor([10.0]))
        else:
            self.bias_weight = 0


    
    


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
                
                scores = scores + self.bias_weight * bias 
            
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