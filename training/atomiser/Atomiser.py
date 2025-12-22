"""
Atomiser Model with Attention-Guided Latent Displacement

Supports three self-attention modes:
1. use_local_attention=True: Local k-NN + RPE + Gaussian bias + attention-guided displacement
2. use_gaussian_bias=True: Global self-attention with Gaussian distance bias (O(L²))
3. Both False: Standard self-attention (no position info)

Key innovation: Displacement is predicted FROM attention patterns, not separately.
- Attention provides base signal (never collapses to zero)
- MLP learns corrections/refinements
- Directional RPE enables "move LEFT" vs "move RIGHT" signals

Usage:
    from atomiser import Atomiser
    
    model = Atomiser(config=config, lookup_table=lookup_table)
    output = model(data, mask, mae_tokens, mae_tokens_mask, task="reconstruction")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from functools import wraps
from einops import repeat, rearrange
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
# IMPORTS
# =============================================================================

# Input processing (the refactored module)
from training.utils.token_building.processor import TokenProcessor

# Neural network components
from .nn_comp import (
    PreNorm, 
    SelfAttention, 
    FeedForward, 
    LatentAttentionPooling, 
    LocalCrossAttention,
)

# Hybrid self-attention with integrated displacement
from .hybrid_displacement import (
    HybridSelfAttentionWithDisplacement,
    create_hybrid_self_attention_with_displacement,
)


# =============================================================================
# Utilities
# =============================================================================

def cache_fn(f):
    """Cache function results for weight sharing across layers."""
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
# Self-Attention with Gaussian Distance Bias (FULL L×L, MEMORY INTENSIVE)
# =============================================================================

class SelfAttentionWithGaussianBias(nn.Module):
    """
    Self-attention with Gaussian distance bias for spatial latents.
    
    This is the FULL L×L version - use only when you need global attention.
    For memory-efficient local attention, use HybridSelfAttentionWithDisplacement instead.
    
    Bias formula: bias[i,j] = -dist(i,j)² / (2σ²)
    
    Memory: O(B × H × L × L) for bias matrix (~50 MB for L=1225)
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
        
        self.global_bias = nn.Parameter(torch.tensor(0.0))
    
    @property
    def sigma(self):
        return self.log_sigma.exp()
    
    def compute_distance_bias(
        self,
        positions: torch.Tensor,
        num_spatial: int,
        total_latents: int,
    ) -> torch.Tensor:
        B = positions.shape[0]
        L_spatial = num_spatial
        L_total = total_latents
        device = positions.device
        dtype = positions.dtype
        
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)
        
        full_bias = torch.zeros(B, self.heads, L_total, L_total, device=device, dtype=dtype)
        full_bias[:, :, :L_spatial, :L_spatial] = spatial_bias
        
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
        B, L, D = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if positions is not None and num_spatial is not None:
            bias = self.compute_distance_bias(positions, num_spatial, L)
            attn = attn + bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.to_out(out)


class PreNormWithPositions(nn.Module):
    """PreNorm wrapper that passes positions to the inner function."""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, positions=None, num_spatial=None, **kwargs):
        x = self.norm(x)
        if positions is not None:
            return self.fn(x, positions=positions, num_spatial=num_spatial, **kwargs)
        return self.fn(x, **kwargs)


# =============================================================================
# MAIN ATOMISER CLASS
# =============================================================================

class Atomiser(pl.LightningModule):
    """
    Atomizer model with attention-guided latent displacement.
    
    Spatial latents: Arranged on a grid, use geographic attention (local)
    Global latents: No spatial position, participate in self-attention only
    
    Self-attention modes:
    1. use_local_attention=True: Local k-NN + RPE + Gaussian bias + attention-guided displacement
       - Memory efficient: O(Lk) instead of O(L²)
       - Direction-aware via polar RPE (enabled by default)
       - Displacement predicted from attention patterns + MLP correction
    2. use_gaussian_bias=True: Global attention with Gaussian distance bias (O(L²))
    3. Both False: Standard self-attention
    """
    
    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=['lookup_table'])
        self.config = config
        
        # =====================================================================
        # 1. INPUT PROCESSOR
        # =====================================================================
        self.input_processor = TokenProcessor(config, lookup_table)
        
        # =====================================================================
        # 2. LATENT CONFIGURATION
        # =====================================================================
        self.spatial_latents_per_row = config["Atomiser"]["spatial_latents"]
        self.num_spatial_latents = self.spatial_latents_per_row ** 2
        self.num_global_latents = config["Atomiser"].get("global_latents", 0)
        self.num_latents = self.num_spatial_latents + self.num_global_latents
        
        # =====================================================================
        # 3. DIMENSIONS
        # =====================================================================
        self.input_dim = self.input_processor.get_encoder_output_dim()
        self.query_dim_recon = self.input_processor.get_decoder_output_dim()
        self.latent_dim = config["Atomiser"].get("latent_dim", self.input_dim)
        
        self.decoder_pe_dim = self.input_processor.pos_encoder.get_output_dim(include_gsd=True)
        
        # =====================================================================
        # 4. MODEL ARCHITECTURE PARAMETERS
        # =====================================================================
        self.depth = config["Atomiser"]["depth"]
        self.cross_heads = config["Atomiser"]["cross_heads"]
        self.latent_heads = config["Atomiser"]["latent_heads"]
        self.cross_dim_head = config["Atomiser"]["cross_dim_head"]
        self.latent_dim_head = config["Atomiser"]["latent_dim_head"]
        self.attn_dropout = config["Atomiser"]["attn_dropout"]
        self.ff_dropout = config["Atomiser"]["ff_dropout"]
        self.weight_tie_layers = config["Atomiser"]["weight_tie_layers"]
        self.self_per_cross_attn = config["Atomiser"]["self_per_cross_attn"]
        self.num_classes = config["trainer"]["num_classes"]
        
        # Geographic attention parameters
        self.geo_k = config["Atomiser"].get("geo_k", 1500)
        self.geo_m_train = config["Atomiser"].get("geo_m_train", 500)
        self.geo_m_val = config["Atomiser"].get("geo_m_val", 500)
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)
        
        # =====================================================================
        # 5. SELF-ATTENTION MODE SELECTION
        # =====================================================================
        self.use_local_attention = config["Atomiser"].get("use_local_attention", False)
        self.use_gaussian_bias = config["Atomiser"].get("use_gaussian_bias", False)
        
        # Local attention parameters
        self.self_attn_k = config["Atomiser"].get("self_attn_k", 128)
        self.latent_spacing = config["Atomiser"].get("latent_spacing", 3.0)
        self.sigma_init = config["Atomiser"].get("sigma_init", 3.0)
        self.learnable_sigma = config["Atomiser"].get("learnable_sigma", True)
        
        # Displacement parameters (NEW - integrated into attention)
        self.enable_displacement = config["Atomiser"].get("enable_displacement", True)
        self.max_displacement = config["Atomiser"].get("max_displacement", 5.0)
        self.displacement_mode = config["Atomiser"].get("displacement_mode", "per_block")
        
        # RPE is now enabled by default for directional awareness
        self.use_rpe = config["Atomiser"].get("use_rpe", True)
        
        # Gaussian bias parameters (for mode 2)
        self.gaussian_sigma = config["Atomiser"].get("gaussian_sigma", 9.0)
        
        # =====================================================================
        # 6. INITIALIZE COMPONENTS
        # =====================================================================
        self._init_latents()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()
        
        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log important configuration settings."""
        print(f"[Atomiser] Input dim: {self.input_dim}, Latent dim: {self.latent_dim}")
        print(f"[Atomiser] Spatial latents: {self.num_spatial_latents}, Global: {self.num_global_latents}")
        print(f"[Atomiser] Depth: {self.depth}, self_per_cross_attn: {self.self_per_cross_attn}")
        
        if self.use_local_attention:
            print(f"[Atomiser] Local Self-Attention with Displacement ENABLED")
            print(f"[Atomiser]   k={self.self_attn_k}, σ_init={self.sigma_init}m, learnable_σ={self.learnable_sigma}")
            print(f"[Atomiser]   RPE (directional): {self.use_rpe}")
            print(f"[Atomiser]   Displacement: {self.enable_displacement}, mode={self.displacement_mode}, max={self.max_displacement}m")
            print(f"[Atomiser]   Memory: O(L×k) = O({self.num_spatial_latents}×{self.self_attn_k}) = {self.num_spatial_latents * self.self_attn_k // 1000}K pairs")
        elif self.use_gaussian_bias:
            print(f"[Atomiser] Gaussian bias ENABLED (full L×L): sigma={self.gaussian_sigma}m")
            print(f"[Atomiser]   Memory: O(L²) = O({self.num_spatial_latents}²) = {self.num_spatial_latents ** 2 // 1000}K pairs")
        else:
            print(f"[Atomiser] Standard self-attention (no position info)")

    def _init_latents(self):
        """Initialize learnable latent vectors."""
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with optional weight sharing."""
        
        # Cross-attention factory
        get_cross_attn = cache_fn(lambda: PreNorm(
            self.latent_dim,
            LocalCrossAttention(
                dim_query=self.latent_dim,
                dim_context=self.input_dim,
                dim_out=self.latent_dim,
                heads=self.cross_heads,
                dim_head=self.cross_dim_head,
                dropout=self.attn_dropout
            )
        ))
        
        get_cross_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))
        
        # =====================================================================
        # Self-attention: THREE MODES
        # =====================================================================
        
        if self.use_local_attention:
            # MODE 1: Local k-NN + RPE + Gaussian bias + INTEGRATED DISPLACEMENT
            self.hybrid_self_attns = nn.ModuleList()
            
            for layer_idx in range(self.depth):
                if self.weight_tie_layers and layer_idx > 0:
                    self.hybrid_self_attns.append(self.hybrid_self_attns[0])
                else:
                    hybrid = HybridSelfAttentionWithDisplacement(
                        dim=self.latent_dim,
                        pos_encoder=self.input_processor.pos_encoder,
                        k=self.self_attn_k,
                        latent_spacing=self.latent_spacing,
                        heads=self.latent_heads,
                        dim_head=self.latent_dim_head,
                        ff_mult=4,
                        dropout=self.attn_dropout,
                        use_rpe=self.use_rpe,  # Enabled by default for directional info
                        use_gaussian_bias=True,
                        sigma_init=self.sigma_init,
                        learnable_sigma=self.learnable_sigma,
                        num_blocks=self.self_per_cross_attn,
                        has_global=self.num_global_latents > 0,
                        share_weights=False,
                        enable_displacement=self.enable_displacement,
                        max_displacement=self.max_displacement,
                        displacement_mode=self.displacement_mode,
                    )
                    self.hybrid_self_attns.append(hybrid)
            
            # Build encoder layers (cross-attention only)
            self.encoder_layers = nn.ModuleList([])
            for layer_idx in range(self.depth):
                should_cache = self.weight_tie_layers and layer_idx > 0
                cache_key = 0 if should_cache else layer_idx
                
                cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
                cross_ff = get_cross_ff(_cache=should_cache, key=f"cross_ff_{cache_key}")
                
                self_attns = nn.ModuleList([])  # Handled by hybrid_self_attns
                
                self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))
        
        else:
            # MODE 2 or 3: Gaussian bias or standard self-attention
            self.hybrid_self_attns = None
            
            if self.use_gaussian_bias:
                get_latent_attn = cache_fn(lambda: PreNormWithPositions(
                    self.latent_dim,
                    SelfAttentionWithGaussianBias(
                        dim=self.latent_dim,
                        heads=self.latent_heads,
                        dim_head=self.latent_dim_head,
                        dropout=self.attn_dropout,
                        sigma=self.gaussian_sigma,
                        learnable_sigma=self.learnable_sigma
                    )
                ))
            else:
                get_latent_attn = cache_fn(lambda: PreNorm(
                    self.latent_dim,
                    SelfAttention(
                        dim=self.latent_dim,
                        heads=self.latent_heads,
                        dim_head=self.latent_dim_head,
                        dropout=self.attn_dropout,
                    )
                ))
            
            get_latent_ff = cache_fn(lambda: PreNorm(
                self.latent_dim,
                FeedForward(self.latent_dim, dropout=self.ff_dropout)
            ))
            
            self.encoder_layers = nn.ModuleList([])
            
            for layer_idx in range(self.depth):
                should_cache = self.weight_tie_layers and layer_idx > 0
                cache_key = 0 if should_cache else layer_idx
                
                cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
                cross_ff = get_cross_ff(_cache=should_cache, key=f"cross_ff_{cache_key}")
                
                self_attns = nn.ModuleList([])
                for sa_idx in range(self.self_per_cross_attn):
                    sa_cache_key = f"{cache_key}_{sa_idx}" if should_cache else f"{layer_idx}_{sa_idx}"
                    self_attn = get_latent_attn(_cache=should_cache, key=f"self_attn_{sa_cache_key}")
                    self_ff = get_latent_ff(_cache=should_cache, key=f"self_ff_{sa_cache_key}")
                    self_attns.append(nn.ModuleList([self_attn, self_ff]))
                
                self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _init_decoder(self):
        """Initialize decoder cross-attention and output head."""
        decoder_context_dim = self.latent_dim + self.decoder_pe_dim
        
        self.decoder_cross_attn = LocalCrossAttention(
            dim_query=self.query_dim_recon,
            dim_context=decoder_context_dim,
            dim_out=self.latent_dim,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head,
            dropout=self.attn_dropout
        )
        
        hidden_dim = self.latent_dim * 2
        mlp_input_dim = self.latent_dim + self.query_dim_recon
        
        self.output_head = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def _init_classifier(self):
        """Initialize classification head."""
        if self.config["Atomiser"].get("final_classifier_head", True):
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(
                    self.latent_dim, 
                    heads=self.latent_heads, 
                    dim_head=self.latent_dim_head, 
                    dropout=self.attn_dropout
                ),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.num_classes)
            )
        else:
            self.to_logits = nn.Identity()

    # =========================================================================
    # Coordinate Utilities
    # =========================================================================

    def _get_default_latent_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get default grid coordinates for spatial latents."""
        grid = self.input_processor.geometry.get_default_latent_grid(device)
        return grid.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # =========================================================================
    # Geographic Pruning
    # =========================================================================
    
    def compute_token_latent_affinity(
        self, 
        tokens: torch.Tensor, 
        latents_coords: torch.Tensor = None, 
        sigma: float = 0.5
    ) -> torch.Tensor:
        """Compute affinity between tokens and latents using 2D Gaussian integrals."""
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        
        x_indices = tokens[:, :, 1].long()
        y_indices = tokens[:, :, 2].long()
        
        mu_x = latents_coords[:, :, 0]
        mu_y = latents_coords[:, :, 1]
        
        integral_x_lut, integral_y_lut = self._precompute_integral_lut(
            x_indices, y_indices, mu_x, mu_y, sigma, device, normalize=False
        )
        
        x_indices_exp = x_indices.unsqueeze(-1).expand(-1, -1, L)
        integral_x = torch.gather(integral_x_lut, dim=1, index=x_indices_exp)
        
        y_indices_exp = y_indices.unsqueeze(-1).expand(-1, -1, L)
        integral_y = torch.gather(integral_y_lut, dim=1, index=y_indices_exp)
        
        affinity = integral_x * integral_y
        
        return affinity.permute(0, 2, 1)

    def _precompute_integral_lut(
        self, 
        x_indices: torch.Tensor, 
        y_indices: torch.Tensor, 
        mu_x: torch.Tensor, 
        mu_y: torch.Tensor, 
        sigma: float, 
        device: torch.device,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute 1D Gaussian integrals as lookup tables."""
        B, L = mu_x.shape
        
        num_x_positions = x_indices.max().item() + 1
        num_y_positions = y_indices.max().item() + 1
        
        sqrt_2, half_width, token_centers = self.input_processor.geometry.get_integral_constants()
        
        x_pos_indices = torch.arange(num_x_positions, device=device)
        x_centers = token_centers[x_pos_indices]
        
        x_min = (x_centers - half_width).view(1, -1, 1)
        x_max = (x_centers + half_width).view(1, -1, 1)
        mu_x_exp = mu_x.unsqueeze(1)
        
        sigma_sqrt_2 = sigma * sqrt_2
        z_x_min = (x_min - mu_x_exp) / sigma_sqrt_2
        z_x_max = (x_max - mu_x_exp) / sigma_sqrt_2
        
        integral_x_lut = 0.5 * (torch.erf(z_x_max) - torch.erf(z_x_min))
        
        y_pos_indices = torch.arange(num_y_positions, device=device)
        y_centers = token_centers[y_pos_indices]
        
        y_min = (y_centers - half_width).view(1, -1, 1)
        y_max = (y_centers + half_width).view(1, -1, 1)
        mu_y_exp = mu_y.unsqueeze(1)
        
        z_y_min = (y_min - mu_y_exp) / sigma_sqrt_2
        z_y_max = (y_max - mu_y_exp) / sigma_sqrt_2
        
        integral_y_lut = 0.5 * (torch.erf(z_y_max) - torch.erf(z_y_min))
        
        if normalize:
            integral_x_lut = integral_x_lut / (integral_x_lut.sum(dim=1, keepdim=True) + 1e-8)
            integral_y_lut = integral_y_lut / (integral_y_lut.sum(dim=1, keepdim=True) + 1e-8)
        
        return integral_x_lut, integral_y_lut

    def geographic_pruning(
        self, 
        tokens: torch.Tensor, 
        mask: torch.Tensor, 
        latents_coords: torch.Tensor,
        sigma: float = 0.5,
        chunk_size: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Geographic pruning using Gaussian affinity with memory-efficient chunking."""
        k = self.geo_k
        B, N, D = tokens.shape
        L = self.num_spatial_latents
        device = tokens.device
        
        with torch.no_grad():
            x_indices = tokens[:, :, 1].long()
            y_indices = tokens[:, :, 2].long()
            
            mu_x = latents_coords[:, :, 0]
            mu_y = latents_coords[:, :, 1]
            
            integral_x_lut, integral_y_lut = self._precompute_integral_lut(
                x_indices, y_indices, mu_x, mu_y, sigma, device, normalize=True
            )
            
            all_indices = []
            all_bias_values = []
            
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                chunk_L = chunk_end - chunk_start
                
                x_lut_chunk = integral_x_lut[:, :, chunk_start:chunk_end]
                y_lut_chunk = integral_y_lut[:, :, chunk_start:chunk_end]
                
                x_idx_exp = x_indices.unsqueeze(-1).expand(-1, -1, chunk_L)
                y_idx_exp = y_indices.unsqueeze(-1).expand(-1, -1, chunk_L)
                
                integral_x_chunk = torch.gather(x_lut_chunk, dim=1, index=x_idx_exp)
                integral_y_chunk = torch.gather(y_lut_chunk, dim=1, index=y_idx_exp)
                
                chunk_affinity = integral_x_chunk * integral_y_chunk
                chunk_affinity = chunk_affinity.permute(0, 2, 1)
                chunk_affinity = torch.log(chunk_affinity + 1e-8)
                
                topk_result = torch.topk(
                    chunk_affinity, 
                    k=k, 
                    dim=-1, 
                    largest=True, 
                    sorted=False
                )
                
                all_indices.append(topk_result.indices)
                all_bias_values.append(topk_result.values)
                
                del integral_x_chunk, integral_y_chunk, chunk_affinity, x_lut_chunk, y_lut_chunk
            
            selected_indices = torch.cat(all_indices, dim=1)
            selected_bias = torch.cat(all_bias_values, dim=1)
            
            del integral_x_lut, integral_y_lut
        
        tokens_per_latent = self._gather_tokens(tokens, selected_indices)
        masks_per_latent = self._gather_masks(mask, selected_indices)
        
        return tokens_per_latent, masks_per_latent, selected_bias

    def _gather_tokens(self, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        L, k = indices.shape[1], indices.shape[2]
        
        flat_indices = indices.reshape(B, L * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(tokens, dim=1, index=flat_indices_exp)
        
        return gathered.reshape(B, L, k, D)

    def _gather_masks(self, mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, N = mask.shape
        L, k = indices.shape[1], indices.shape[2]
        
        flat_indices = indices.reshape(B, L * k)
        gathered = torch.gather(mask, dim=1, index=flat_indices)
        
        return gathered.reshape(B, L, k).bool()

    # =========================================================================
    # Encoder
    # =========================================================================

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        latents_coords: Optional[torch.Tensor] = None,
        training: bool = True,
        return_trajectory: bool = False,
        return_displacement_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], Optional[Dict[str, Any]]]:
        """
        Encode tokens into latent representations.
        
        Args:
            tokens: [B, N, 6] input tokens
            mask: [B, N] attention mask
            latents_coords: [B, L_spatial, 2] initial positions or None for grid
            training: bool
            return_trajectory: If True, return position history
            return_displacement_stats: If True, return displacement statistics
            
        Returns:
            latents: [B, L_total, D]
            final_coords: [B, L_spatial, 2]
            trajectory: List of coords if return_trajectory else None
            displacement_stats: Dict if return_displacement_stats else None
        """
        B = tokens.shape[0]
        L_spatial = self.num_spatial_latents
        device = tokens.device
        
        # Initialize latents
        latents = repeat(self.latents, 'n d -> b n d', b=B)
        
        # Force fresh grid coordinates
        del latents_coords
        latents_coords = None
        
        # Initialize coordinates
        if latents_coords is not None:
            current_coords = latents_coords.clone()
        else:
            current_coords = self._get_default_latent_coords(B, device)
        
        trajectory = [current_coords.clone()] if return_trajectory else None
        all_displacement_stats = [] if return_displacement_stats else None
        
        # Initial geographic pruning
        geo_tokens, geo_masks, _ = self.geographic_pruning(tokens, mask, current_coords)
        k = geo_tokens.shape[2]
        
        num_layers = len(self.encoder_layers)
        
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            
            # =================================================================
            # CROSS-ATTENTION (latents ← tokens)
            # =================================================================
            
            m = self.geo_m_train if training else self.geo_m_val
            m = min(m, k)
            
            if m < k:
                perm = torch.randperm(k, device=device)[:m]
                sampled_tokens = geo_tokens[:, :, perm, :]
                sampled_masks = geo_masks[:, :, perm]
            else:
                sampled_tokens = geo_tokens
                sampled_masks = geo_masks
            
            processed_tokens = self.input_processor.process_data_for_encoder(
                sampled_tokens,
                sampled_masks,
                latent_positions=current_coords
            )
            
            latents_spatial = latents[:, :L_spatial, :]
            latents_global = latents[:, L_spatial:, :] if self.num_global_latents > 0 else None
            
            spatial_out = cross_attn(
                latents_spatial,
                context=processed_tokens,
                mask=~sampled_masks,
            )
            
            latents_spatial = spatial_out + latents_spatial
            latents_spatial = cross_ff(latents_spatial) + latents_spatial
            
            if latents_global is not None:
                latents = torch.cat([latents_spatial, latents_global], dim=1)
            else:
                latents = latents_spatial
            
            # =================================================================
            # SELF-ATTENTION + DISPLACEMENT (integrated)
            # =================================================================
            
            if self.use_local_attention:
                # MODE 1: Local k-NN + RPE + Gaussian bias + INTEGRATED DISPLACEMENT
                cache = self.hybrid_self_attns[layer_idx].compute_cache(current_coords)
                
                latents, displacement, disp_stats = self.hybrid_self_attns[layer_idx](
                    latents, 
                    cache, 
                    num_spatial=L_spatial,
                    current_positions=current_coords,
                )

                
                
                # Apply displacement
                if displacement is not None:
                    current_coords = current_coords + displacement
                
                # Collect stats
                if return_displacement_stats:
                    disp_stats['layer'] = layer_idx
                    all_displacement_stats.append(disp_stats)
                
            elif self.use_gaussian_bias:
                # MODE 2: Gaussian bias self-attention (full L×L)
                for self_attn, self_ff in self_attns:
                    latents = self_attn(
                        latents, 
                        positions=current_coords,
                        num_spatial=L_spatial
                    ) + latents
                    latents = self_ff(latents) + latents
                    
            else:
                # MODE 3: Standard self-attention
                for self_attn, self_ff in self_attns:
                    latents = self_attn(latents) + latents
                    latents = self_ff(latents) + latents
            
            # Record trajectory
            if return_trajectory:
                trajectory.append(current_coords.clone())
            
            # Re-compute geographic pruning for next layer
            if layer_idx < num_layers - 1:
                geo_tokens, geo_masks, _ = self.geographic_pruning(
                    tokens, mask, current_coords
                )
        
        # Aggregate displacement stats
        final_disp_stats = None
        if return_displacement_stats and all_displacement_stats:
            final_disp_stats = self._aggregate_displacement_stats(all_displacement_stats)
        
        return latents, current_coords, trajectory, final_disp_stats
    
    def _aggregate_displacement_stats(self, stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate displacement statistics across layers."""
        aggregated = {
            'per_layer': stats_list,
        }
        
        # Extract key metrics
        total_displacements = []
        attn_fractions = []
        
        for stats in stats_list:
            if 'total_displacement_magnitude' in stats:
                total_displacements.append(stats['total_displacement_magnitude'])
            if 'mean_w_attn_fraction' in stats:
                attn_fractions.append(stats['mean_w_attn_fraction'])
        
        if total_displacements:
            aggregated['mean_displacement'] = sum(total_displacements) / len(total_displacements)
            aggregated['max_displacement'] = max(total_displacements)
        
        if attn_fractions:
            aggregated['mean_attn_fraction'] = sum(attn_fractions) / len(attn_fractions)
        
        return aggregated

    # =========================================================================
    # Decoder
    # =========================================================================

    def reconstruct(
        self, 
        latents: torch.Tensor, 
        latents_coords: torch.Tensor, 
        query_tokens: torch.Tensor, 
        query_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct query tokens using spatial latents.
        """
        B, N, _ = query_tokens.shape
        L_spatial = self.num_spatial_latents
        device = latents.device
        D = latents.shape[-1]
        k = self.decoder_k_spatial
        
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask
        )
        
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)
        
        dists_sq = (
            query_coords.unsqueeze(2) - latents_coords.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        
        _, topk_indices = torch.topk(dists_sq, k=k, dim=-1, largest=False)
        
        spatial_latents = latents[:, :L_spatial, :]
        
        flat_indices = topk_indices.reshape(B, N * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_exp)
        selected_latents = gathered.reshape(B, N, k, D)
        
        flat_coord_indices = flat_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(latents_coords, dim=1, index=flat_coord_indices)
        selected_coords = gathered_coords.reshape(B, N, k, 2)
        
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)
        
        self.latent_surface = 103
        scale = self.latent_surface / (self.spatial_latents_per_row - 1)
        
        relative_pe = self.input_processor.pos_encoder(
            delta_x, delta_y, scale, gsd=0.2
        )
        
        context = torch.cat([selected_latents, relative_pe], dim=-1)
        output = self.decoder_cross_attn(query_features, context)
        
        output_with_query = torch.cat([output, query_features], dim=-1)
        predictions = self.output_head(output_with_query)
        
        return predictions
    
    def classify(self, latents: torch.Tensor) -> torch.Tensor:
        """Classify from latent representations."""
        return self.to_logits(latents)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self, 
        data: torch.Tensor, 
        mask: torch.Tensor, 
        mae_tokens: Optional[torch.Tensor] = None, 
        mae_tokens_mask: Optional[torch.Tensor] = None, 
        latents_coords: Optional[torch.Tensor] = None,
        training: bool = True, 
        task: str = "reconstruction",
        return_trajectory: bool = False,
        return_displacement_stats: bool = False,
    ):
        """
        Forward pass.
        
        Args:
            data: [B, N, 6] input tokens
            mask: [B, N] attention mask
            mae_tokens: [B, M, 6] query tokens for reconstruction
            mae_tokens_mask: [B, M] query mask
            latents_coords: [B, L, 2] initial positions or None
            training: bool
            task: "reconstruction", "visualization", "encoder", or "classification"
            return_trajectory: If True, return position history
            return_displacement_stats: If True, return displacement statistics
        """
        need_trajectory = return_trajectory or task == "visualization"
        need_disp_stats = return_displacement_stats or task == "visualization"
        
        latents, final_coords, trajectory, disp_stats = self.encode(
            data, mask, latents_coords, training, 
            return_trajectory=need_trajectory,
            return_displacement_stats=need_disp_stats,
        )
        
        if task == "encoder":
            result = {'latents': latents, 'final_coords': final_coords}
            if trajectory is not None:
                result['trajectory'] = trajectory
            if disp_stats is not None:
                result['displacement_stats'] = disp_stats
            return result
        
        if task == "reconstruction" or task == "visualization":
            chunk_size = 100000
            N = mae_tokens.shape[1]
            
            if N > chunk_size:
                preds_list = []
                for i in range(0, N, chunk_size):
                    chunk_tokens = mae_tokens[:, i:i + chunk_size]
                    chunk_mask = mae_tokens_mask[:, i:i + chunk_size]
                    preds_list.append(self.reconstruct(
                        latents, final_coords, chunk_tokens, chunk_mask
                    ))
                predictions = torch.cat(preds_list, dim=1)
            else:
                predictions = self.reconstruct(
                    latents, final_coords, mae_tokens, mae_tokens_mask
                )
            
            if task == "visualization":
                return {
                    'predictions': predictions,
                    'trajectory': trajectory,
                    'displacement_stats': disp_stats,
                    'final_coords': final_coords,
                }
            return predictions
        
        else:  # classification
            return self.classify(latents)

    # =========================================================================
    # Statistics and Diagnostics
    # =========================================================================
    
    def get_sigma_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about learned sigma values."""
        if not self.use_local_attention or self.hybrid_self_attns is None:
            return None
        
        all_stats = []
        for layer_idx, hybrid in enumerate(self.hybrid_self_attns):
            stats = hybrid.get_sigma_stats()
            if stats:
                stats['layer'] = layer_idx
                all_stats.append(stats)
        
        if not all_stats:
            return None
        
        all_sigmas = []
        for stats in all_stats:
            all_sigmas.extend([s for block in stats['per_block'] for s in block])
        
        all_sigmas = torch.tensor(all_sigmas)
        
        return {
            'per_layer': all_stats,
            'global_mean': all_sigmas.mean().item(),
            'global_min': all_sigmas.min().item(),
            'global_max': all_sigmas.max().item(),
            'global_std': all_sigmas.std().item(),
        }
    
    def get_displacement_predictor_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about displacement predictors (attention vs MLP weights)."""
        if not self.use_local_attention or self.hybrid_self_attns is None:
            return None
        
        all_stats = {}
        for layer_idx, hybrid in enumerate(self.hybrid_self_attns):
            stats = hybrid.get_displacement_stats()
            if stats:
                all_stats[f'layer_{layer_idx}'] = stats
        
        return all_stats if all_stats else None

    # =========================================================================
    # Freeze/Unfreeze Utilities
    # =========================================================================
    
    def _set_requires_grad(self, module, flag: bool):
        if isinstance(module, torch.Tensor):
            module.requires_grad = flag
        elif hasattr(module, 'parameters'):
            for param in module.parameters():
                param.requires_grad = flag
    
    def freeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, False)
        self.latents.requires_grad = False
        self._set_requires_grad(self.input_processor, False)
        if self.hybrid_self_attns is not None:
            self._set_requires_grad(self.hybrid_self_attns, False)
    
    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.latents.requires_grad = True
        self._set_requires_grad(self.input_processor, True)
        if self.hybrid_self_attns is not None:
            self._set_requires_grad(self.hybrid_self_attns, True)
    
    def freeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, False)
        self._set_requires_grad(self.output_head, False)
    
    def unfreeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, True)
        self._set_requires_grad(self.output_head, True)
    
    def freeze_classifier(self):
        self._set_requires_grad(self.to_logits, False)
    
    def unfreeze_classifier(self):
        self._set_requires_grad(self.to_logits, True)
    
    def freeze_displacement(self):
        """Freeze only the displacement predictors (keep attention trainable)."""
        if self.hybrid_self_attns is not None:
            for hybrid in self.hybrid_self_attns:
                for block in hybrid.blocks:
                    if block.displacement_predictor is not None:
                        self._set_requires_grad(block.displacement_predictor, False)
    
    def unfreeze_displacement(self):
        """Unfreeze displacement predictors."""
        if self.hybrid_self_attns is not None:
            for hybrid in self.hybrid_self_attns:
                for block in hybrid.blocks:
                    if block.displacement_predictor is not None:
                        self._set_requires_grad(block.displacement_predictor, True)

    # =========================================================================
    # Trajectory Analysis
    # =========================================================================
    
    def compute_trajectory_stats(self, trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        """Compute statistics about latent movement from trajectory."""
        if trajectory is None or len(trajectory) < 2:
            return {}
        
        stats = {
            'num_steps': len(trajectory) - 1,
            'per_step_displacement': [],
            'cumulative_displacement': [],
        }
        
        initial_coords = trajectory[0]
        
        for i in range(1, len(trajectory)):
            step_disp = (trajectory[i] - trajectory[i-1]).norm(dim=-1).mean().item()
            cumul_disp = (trajectory[i] - initial_coords).norm(dim=-1).mean().item()
            
            stats['per_step_displacement'].append(step_disp)
            stats['cumulative_displacement'].append(cumul_disp)
        
        stats['total_displacement'] = stats['cumulative_displacement'][-1] if stats['cumulative_displacement'] else 0
        stats['mean_step_displacement'] = sum(stats['per_step_displacement']) / len(stats['per_step_displacement']) if stats['per_step_displacement'] else 0
        
        return stats