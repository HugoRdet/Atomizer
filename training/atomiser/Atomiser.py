"""
Atomiser Model with Learnable Latent Positions

Supports three self-attention modes:
1. use_local_attention=True: Local k-NN + RPE + Gaussian bias + bidirectional global cross-attention
2. use_gaussian_bias=True: Global self-attention with Gaussian distance bias (O(L²))
3. Both False: Standard self-attention (no position info)

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

# Hybrid self-attention (local k-NN + RPE + Gaussian bias)
from .hybrid_self_attention import (
    HybridSelfAttention,
    create_hybrid_self_attention,
)

# Displacement strategies
from .displacement import (
    PositionUpdateStrategy,
    NoPositionUpdate,
    MLPDisplacementUpdate,
    DeformableOffsetUpdate,
    create_position_updater,
    compute_displacement_stats,
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
    For memory-efficient local attention, use HybridSelfAttention instead.
    
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
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, Ls, Ls, 2]
        dist_sq = (diff ** 2).sum(dim=-1)  # [B, Ls, Ls]
        
        # Gaussian bias: -dist² / (2σ²)
        sigma_sq = (self.sigma ** 2).view(1, -1, 1, 1)
        
        # Spatial-spatial bias: [B, H, Ls, Ls]
        spatial_bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq)
        
        # Build full bias matrix including global latents
        full_bias = torch.zeros(B, self.heads, L_total, L_total, device=device, dtype=dtype)
        
        # Fill spatial-spatial block
        full_bias[:, :, :L_spatial, :L_spatial] = spatial_bias
        
        # Global latents get a learnable constant bias
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
        
        # Standard attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Add distance bias if positions provided
        if positions is not None and num_spatial is not None:
            bias = self.compute_distance_bias(positions, num_spatial, L)
            attn = attn + bias
        
        # Softmax and value aggregation
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project output
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
    Atomizer model for satellite image processing with learnable latent positions.
    
    Spatial latents: Arranged on a grid, use geographic attention (local)
    Global latents: No spatial position, participate in self-attention only
    
    Self-attention modes:
    1. use_local_attention=True: Local k-NN + RPE + Gaussian bias + bidirectional global cross-attention
       - Memory efficient: O(Lk) instead of O(L²)
       - Full directional info via polar RPE
       - Multi-scale via learnable per-head σ
    2. use_gaussian_bias=True: Global attention with Gaussian distance bias (O(L²))
    3. Both False: Standard self-attention
    """
    
    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=['lookup_table'])
        self.config = config
        
        # =====================================================================
        # 1. INPUT PROCESSOR (Replaces self.transform)
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
        # 3. DIMENSIONS (from processor)
        # =====================================================================
        self.input_dim = self.input_processor.get_encoder_output_dim()
        self.query_dim_recon = self.input_processor.get_decoder_output_dim()
        self.latent_dim = config["Atomiser"].get("latent_dim", self.input_dim)
        
        # Positional encoding dimension (for decoder context)
        self.decoder_pe_dim = self.input_processor.pos_encoder.get_output_dim(include_gsd=False)
        
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
        self.geo_k = config["Atomiser"].get("geo_k", 2000)
        self.geo_m_train = config["Atomiser"].get("geo_m_train", 500)
        self.geo_m_val = config["Atomiser"].get("geo_m_val", 500)
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)
        
        # =====================================================================
        # 5. SELF-ATTENTION MODE SELECTION
        # =====================================================================
        # Priority: use_local_attention > use_gaussian_bias > standard
        self.use_local_attention = config["Atomiser"].get("use_local_attention", False)
        self.use_gaussian_bias = config["Atomiser"].get("use_gaussian_bias", False)
        
        # Local attention parameters (k-NN + RPE + Gaussian bias)
        self.self_attn_k = config["Atomiser"].get("self_attn_k", 128)
        self.latent_spacing = config["Atomiser"].get("latent_spacing", 3.0)
        self.sigma_init = config["Atomiser"].get("sigma_init", 3.0)
        self.learnable_sigma = config["Atomiser"].get("learnable_sigma", True)
        
        # Gaussian bias parameters (only used if use_gaussian_bias=True and use_local_attention=False)
        self.gaussian_sigma = config["Atomiser"].get("gaussian_sigma", 9.0)
        
        # =====================================================================
        # 6. INITIALIZE COMPONENTS
        # =====================================================================
        self._init_latents()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()
        self._init_position_updater()
        
        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log important configuration settings."""
        print(f"[Atomiser] Input dim: {self.input_dim}, Latent dim: {self.latent_dim}")
        print(f"[Atomiser] Spatial latents: {self.num_spatial_latents}, Global: {self.num_global_latents}")
        print(f"[Atomiser] Depth: {self.depth}, self_per_cross_attn: {self.self_per_cross_attn}")
        
        if self.use_local_attention:
            print(f"[Atomiser] Local Self-Attention ENABLED (k-NN + RPE + Gaussian bias)")
            print(f"[Atomiser]   k={self.self_attn_k}, σ_init={self.sigma_init}m, learnable_σ={self.learnable_sigma}")
            print(f"[Atomiser]   latent_spacing={self.latent_spacing}m")
            print(f"[Atomiser]   RPE: using same pos_encoder as cross-attention")
            print(f"[Atomiser]   Memory: O(L×k) = O({self.num_spatial_latents}×{self.self_attn_k}) = {self.num_spatial_latents * self.self_attn_k // 1000}K pairs")
        elif self.use_gaussian_bias:
            print(f"[Atomiser] Gaussian bias ENABLED (full L×L): sigma={self.gaussian_sigma}m, learnable={self.learnable_sigma}")
            print(f"[Atomiser]   Memory: O(L²) = O({self.num_spatial_latents}²) = {self.num_spatial_latents ** 2 // 1000}K pairs")
        else:
            print(f"[Atomiser] Standard self-attention (no position info)")

    def _init_latents(self):
        """Initialize learnable latent vectors."""
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with optional weight sharing."""
        
        # =====================================================================
        # Cross-attention factory (spatial latents attending to tokens)
        # =====================================================================
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
            # MODE 1: Local k-NN + RPE + Gaussian bias + bidirectional global cross-attention
            # Memory efficient: O(Lk) instead of O(L²)
            # Uses same pos_encoder as cross-attention for consistency
            self.hybrid_self_attns = nn.ModuleList()
            
            for layer_idx in range(self.depth):
                if self.weight_tie_layers and layer_idx > 0:
                    # Share weights with first layer
                    self.hybrid_self_attns.append(self.hybrid_self_attns[0])
                else:
                    hybrid = HybridSelfAttention(
                        dim=self.latent_dim,
                        pos_encoder=self.input_processor.pos_encoder,  # Use same encoder!
                        k=self.self_attn_k,
                        latent_spacing=self.latent_spacing,
                        heads=self.latent_heads,
                        dim_head=self.latent_dim_head,
                        ff_mult=4,
                        dropout=self.attn_dropout,
                        sigma_init=self.sigma_init,
                        learnable_sigma=self.learnable_sigma,
                        num_blocks=self.self_per_cross_attn,
                        has_global=self.num_global_latents > 0,
                        share_weights=False,
                    )
                    self.hybrid_self_attns.append(hybrid)
            
            # Build encoder layers (cross-attention only, self-attention handled by hybrid)
            self.encoder_layers = nn.ModuleList([])
            for layer_idx in range(self.depth):
                should_cache = self.weight_tie_layers and layer_idx > 0
                cache_key = 0 if should_cache else layer_idx
                
                cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
                cross_ff = get_cross_ff(_cache=should_cache, key=f"cross_ff_{cache_key}")
                
                # Empty self_attns (handled by hybrid_self_attns)
                self_attns = nn.ModuleList([])
                
                self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))
        
        else:
            # MODE 2 or 3: Gaussian bias or standard self-attention (full L×L)
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
            
            # Build layers
            self.encoder_layers = nn.ModuleList([])
            self.hybrid_self_attns = None  # Not used
            
            for layer_idx in range(self.depth):
                should_cache = self.weight_tie_layers and layer_idx > 0
                cache_key = 0 if should_cache else layer_idx
                
                cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
                cross_ff = get_cross_ff(_cache=should_cache, key=f"cross_ff_{cache_key}")
                
                # Self-attention blocks
                self_attns = nn.ModuleList([])
                for sa_idx in range(self.self_per_cross_attn):
                    sa_cache_key = f"{cache_key}_{sa_idx}" if should_cache else f"{layer_idx}_{sa_idx}"
                    self_attn = get_latent_attn(_cache=should_cache, key=f"self_attn_{sa_cache_key}")
                    self_ff = get_latent_ff(_cache=should_cache, key=f"self_ff_{sa_cache_key}")
                    self_attns.append(nn.ModuleList([self_attn, self_ff]))
                
                self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _init_decoder(self):
        """Initialize decoder cross-attention and output head."""
        
        # Context = latent features + relative PE
        decoder_context_dim = self.latent_dim + self.decoder_pe_dim
        
        self.decoder_cross_attn = LocalCrossAttention(
            dim_query=self.query_dim_recon,
            dim_context=decoder_context_dim,
            dim_out=self.latent_dim,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head,
            dropout=self.attn_dropout
        )
        
        # Output MLP
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

    def _init_position_updater(self):
        """Initialize the position update strategy."""
        use_displacement = self.config["Atomiser"].get("use_displacement", False)
        
        if use_displacement:
            updater_config = {
                "use_displacement": True,
                "position_strategy": self.config["Atomiser"].get("position_strategy", "mlp"),
                "latent_dim": self.latent_dim,
                "depth": self.depth,
                "max_displacement": self.config["Atomiser"].get("max_displacement", 5.0),
                "share_displacement_weights": self.config["Atomiser"].get("share_displacement_weights", True),
            }
            self.position_updater = create_position_updater(updater_config)
            print(f"[Atomiser] Position updates ENABLED: {updater_config['position_strategy']}")
        else:
            self.position_updater = NoPositionUpdate()

    # =========================================================================
    # Coordinate Utilities
    # =========================================================================

    def _get_default_latent_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get default grid coordinates for spatial latents."""
        grid = self.input_processor.geometry.get_default_latent_grid(device)
        return grid.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # =========================================================================
    # Geographic Pruning (Gaussian Affinity with Chunking)
    # =========================================================================
    
    def compute_token_latent_affinity(
        self, 
        tokens: torch.Tensor, 
        latents_coords: torch.Tensor = None, 
        sigma: float = 0.5
    ) -> torch.Tensor:
        """
        Compute affinity between tokens and latents using 2D Gaussian integrals.
        """
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
        
        # X-axis integral LUT
        x_pos_indices = torch.arange(num_x_positions, device=device)
        x_centers = token_centers[x_pos_indices]
        
        x_min = (x_centers - half_width).view(1, -1, 1)
        x_max = (x_centers + half_width).view(1, -1, 1)
        mu_x_exp = mu_x.unsqueeze(1)
        
        sigma_sqrt_2 = sigma * sqrt_2
        z_x_min = (x_min - mu_x_exp) / sigma_sqrt_2
        z_x_max = (x_max - mu_x_exp) / sigma_sqrt_2
        
        integral_x_lut = 0.5 * (torch.erf(z_x_max) - torch.erf(z_x_min))
        
        # Y-axis integral LUT
        y_pos_indices = torch.arange(num_y_positions, device=device)
        y_centers = token_centers[y_pos_indices]
        
        y_min = (y_centers - half_width).view(1, -1, 1)
        y_max = (y_centers + half_width).view(1, -1, 1)
        mu_y_exp = mu_y.unsqueeze(1)
        
        z_y_min = (y_min - mu_y_exp) / sigma_sqrt_2
        z_y_max = (y_max - mu_y_exp) / sigma_sqrt_2
        
        integral_y_lut = 0.5 * (torch.erf(z_y_max) - torch.erf(z_y_min))
        
        # Optional normalization
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
        """Efficiently gather tokens by indices."""
        B, N, D = tokens.shape
        L, k = indices.shape[1], indices.shape[2]
        
        flat_indices = indices.reshape(B, L * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(tokens, dim=1, index=flat_indices_exp)
        
        return gathered.reshape(B, L, k, D)

    def _gather_masks(self, mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Efficiently gather masks by indices."""
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
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Encode tokens into latent representations.
        
        Args:
            tokens: [B, N, 6] input tokens
            mask: [B, N] attention mask
            latents_coords: [B, L_spatial, 2] initial positions or None for grid
            training: bool
            return_trajectory: If True, return position history
            
        Returns:
            latents: [B, L_total, D]
            final_coords: [B, L_spatial, 2]
            trajectory: List of coords if return_trajectory else None
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
        
        # Initial geographic pruning
        geo_tokens, geo_masks, _ = self.geographic_pruning(tokens, mask, current_coords)
        k = geo_tokens.shape[2]
        
        num_layers = len(self.encoder_layers)
        
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            
            # =================================================================
            # CROSS-ATTENTION (latents ← tokens)
            # =================================================================
            
            # Sample tokens from geographic pool
            m = self.geo_m_train if training else self.geo_m_val
            m = min(m, k)
            
            if m < k:
                perm = torch.randperm(k, device=device)[:m]
                sampled_tokens = geo_tokens[:, :, perm, :]
                sampled_masks = geo_masks[:, :, perm]
            else:
                sampled_tokens = geo_tokens
                sampled_masks = geo_masks
            
            # Process tokens with TokenProcessor
            processed_tokens = self.input_processor.process_data_for_encoder(
                sampled_tokens,
                sampled_masks,
                latent_positions=current_coords
            )
            
            # Cross attention (spatial latents only)
            latents_spatial = latents[:, :L_spatial, :]
            latents_global = latents[:, L_spatial:, :] if self.num_global_latents > 0 else None
            
            spatial_out = cross_attn(
                latents_spatial,
                context=processed_tokens,
                mask=~sampled_masks,
            )
            
            latents_spatial = spatial_out + latents_spatial
            latents_spatial = cross_ff(latents_spatial) + latents_spatial
            
            # Recombine
            if latents_global is not None:
                latents = torch.cat([latents_spatial, latents_global], dim=1)
            else:
                latents = latents_spatial
            
            # =================================================================
            # SELF-ATTENTION (three modes)
            # =================================================================
            
            if self.use_local_attention:
                # MODE 1: Local k-NN + RPE + Gaussian bias + bidirectional global
                # Compute cache ONCE per layer (k-NN indices, RPE, distances)
                cache = self.hybrid_self_attns[layer_idx].compute_cache(current_coords)
                
                # Apply all self_per_cross_attn blocks (cache is reused!)
                latents = self.hybrid_self_attns[layer_idx](
                    latents, 
                    cache, 
                    num_spatial=L_spatial
                )
                
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
            
            # =================================================================
            # POSITION UPDATE
            # =================================================================
            latents_spatial = latents[:, :L_spatial, :]
            current_coords, _ = self.position_updater(latents_spatial, current_coords, layer_idx)
            
            if return_trajectory:
                trajectory.append(current_coords.clone())
            
            # Re-compute geographic pruning for next layer
            if layer_idx < num_layers - 1:
                geo_tokens, geo_masks, _ = self.geographic_pruning(
                    tokens, mask, current_coords
                )
        
        if return_trajectory:
            return latents, current_coords, trajectory
        return latents, current_coords, None

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
        
        Args:
            latents: [B, L_total, D]
            latents_coords: [B, L_spatial, 2]
            query_tokens: [B, N, 6]
            query_mask: [B, N]
            
        Returns:
            predictions: [B, N, 1]
        """
        B, N, _ = query_tokens.shape
        L_spatial = self.num_spatial_latents
        device = latents.device
        D = latents.shape[-1]
        k = self.decoder_k_spatial
        
        # 1. Get query content features
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask
        )
        
        # 2. Find k-nearest latents for each query
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)
        
        dists_sq = (
            query_coords.unsqueeze(2) - latents_coords.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        
        _, topk_indices = torch.topk(dists_sq, k=k, dim=-1, largest=False)
        
        # 3. Gather selected latents
        spatial_latents = latents[:, :L_spatial, :]
        
        flat_indices = topk_indices.reshape(B, N * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_exp)
        selected_latents = gathered.reshape(B, N, k, D)
        
        # 4. Compute relative positional encoding
        flat_coord_indices = flat_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(latents_coords, dim=1, index=flat_coord_indices)
        selected_coords = gathered_coords.reshape(B, N, k, 2)
        
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)
        
        scale = self.input_processor.geometry.get_physical_scale(query_tokens)
        
        relative_pe = self.input_processor.pos_encoder(
            delta_x, delta_y, scale, gsd=None
        )
        
        # 5. Cross-attend
        context = torch.cat([selected_latents, relative_pe], dim=-1)
        output = self.decoder_cross_attn(query_features, context)
        
        # 6. Output head
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
        return_trajectory: bool = False
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
        """
        need_trajectory = return_trajectory or task == "visualization"
        
        latents, final_coords, trajectory = self.encode(
            data, mask, latents_coords, training, return_trajectory=need_trajectory
        )
        
        if task == "encoder":
            result = {'latents': latents, 'final_coords': final_coords}
            if trajectory is not None:
                result['trajectory'] = trajectory
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
                return predictions, trajectory
            return predictions
        
        else:  # classification
            return self.classify(latents)

    # =========================================================================
    # Sigma Statistics (for local attention mode)
    # =========================================================================
    
    def get_sigma_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about learned sigma values (only for local attention mode)."""
        if not self.use_local_attention or self.hybrid_self_attns is None:
            return None
        
        all_stats = []
        for layer_idx, hybrid in enumerate(self.hybrid_self_attns):
            stats = hybrid.get_sigma_stats()
            stats['layer'] = layer_idx
            all_stats.append(stats)
        
        # Aggregate
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

    # =========================================================================
    # Freeze/Unfreeze Utilities
    # =========================================================================
    
    def _set_requires_grad(self, module, flag: bool):
        """Recursively set requires_grad."""
        if isinstance(module, torch.Tensor):
            module.requires_grad = flag
        elif hasattr(module, 'parameters'):
            for param in module.parameters():
                param.requires_grad = flag
    
    def freeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, False)
        self.latents.requires_grad = False
        self._set_requires_grad(self.position_updater, False)
        self._set_requires_grad(self.input_processor, False)
        if self.hybrid_self_attns is not None:
            self._set_requires_grad(self.hybrid_self_attns, False)
    
    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.latents.requires_grad = True
        self._set_requires_grad(self.position_updater, True)
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
    
    def freeze_position_updater(self):
        self._set_requires_grad(self.position_updater, False)
    
    def unfreeze_position_updater(self):
        self._set_requires_grad(self.position_updater, True)
    
    def get_displacement_stats(self, trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        """Compute statistics about latent movement from trajectory."""
        return compute_displacement_stats(trajectory)