"""
Atomiser Model with Local RoPE for Positional Encoding

Key Features:
- Local RoPE: Q at origin (unchanged), K rotated by relative position
- Resolution-aware: log-scale GSD modulation
- 37x faster than Fourier+MLP approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from functools import wraps
from einops import repeat, rearrange
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.checkpoint import checkpoint

# =============================================================================
# IMPORTS
# =============================================================================

from training.utils.token_building.processor import TokenProcessor

from .nn_comp import (
    PreNorm, 
    SelfAttention, 
    FeedForward, 
    LatentAttentionPooling, 
    PreNormWithPositions,
    LatentPositionEncoder
)

from .RPE import (
    LocalCrossAttentionRoPE,
    PreNormRPEConcat,
    SelfAttentionRPEConcat,
    LocalCrossAttentionRoPE
)

from .displacement import (
    create_position_updater,
    PositionUpdateStrategy,
    compute_displacement_stats,
)

from .gaussian_bias import (
    SelfAttentionWithGaussianBias
)

from .geographic_pruning import (
    GeographicPruning,
    create_geographic_pruning,
)

from .error_guided_displacement import (
    ErrorGuidedDisplacement,
    create_error_guided_displacement,
)

from .gravity_displacement import (
    GravityDisplacement,
    create_gravity_displacement,
)

from .hybrid_self_attention import (
    HybridSelfAttention,
    create_hybrid_self_attention,
)


"""
Self-Attention with Cartesian Fourier RPE

Two modes:
1. Self-attention (encoder): All tokens attend to all tokens
2. Local cross-attention (decoder): Queries attend to k nearest contexts

RPE is computed as attention bias from pairwise relative positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, Union


    



# =============================================================================
# UTILITIES
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
# MAIN ATOMISER CLASS
# =============================================================================

class Atomiser_error(pl.LightningModule):
    """
    Atomizer model with Local RoPE for positional encoding.
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
        self.latent_surface = config["Atomiser"].get("latent_surface", 103.0)
        
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
        self.geo_sigma = config["Atomiser"].get("geo_sigma", 0.5)
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)
        
        # =====================================================================
        # 5. MEMORY OPTIMIZATION
        # =====================================================================
        self.use_checkpoint = config["Atomiser"].get("use_checkpoint", False)
        
        # =====================================================================
        # 6. ROPE CONFIGURATION
        # =====================================================================
        self.encoder_use_rpe = config["Atomiser"].get("encoder_use_rpe", False)
        self.decoder_use_rpe = config["Atomiser"].get("decoder_use_rpe", False)
        self.use_rpe = config["Atomiser"].get("use_rpe", False)  # For self-attention
        self.rope_base = config["Atomiser"].get("rope_base", 10.0)
        self.rope_reference_gsd = config["Atomiser"].get("rope_reference_gsd", 0.2)
        self.rope_learnable_scale = config["Atomiser"].get("rope_learnable_scale", True)
        
        # =====================================================================
        # 7. SELF-ATTENTION MODE
        # =====================================================================
        self.use_gaussian_bias = config["Atomiser"].get("use_gaussian_bias", False)
        self.gaussian_sigma = config["Atomiser"].get("gaussian_sigma", 9.0)
        self.learnable_sigma = config["Atomiser"].get("learnable_sigma", True)
        self.use_hybrid_self_attention = config["Atomiser"].get("use_hybrid_self_attention", False)
        self.self_attn_k = config["Atomiser"].get("self_attn_k", 64)
        self.latent_spacing = self.latent_surface / (self.spatial_latents_per_row - 1)
        
        # =====================================================================
        # 8. DISPLACEMENT STRATEGY
        # =====================================================================
        self.use_displacement = config["Atomiser"].get("use_displacement", False)
        self.position_strategy = config["Atomiser"].get("position_strategy", "mlp")
        self.max_displacement = config["Atomiser"].get("max_displacement", 3.0)
        self.min_displacement = config["Atomiser"].get("min_displacement", 0.5)
        self.share_displacement_weights = config["Atomiser"].get("share_displacement_weights", True)
        self.stable_depth = config["Atomiser"].get("stable_depth", 0)
        
        self.use_error_guided_displacement = config["Atomiser"].get("use_error_guided_displacement", False)
        self.share_error_predictor_weights = config["Atomiser"].get("share_error_predictor_weights", True)
        
        self.use_gravity_displacement = config["Atomiser"].get("use_gravity_displacement", False)
        self.repulsion_strength = config["Atomiser"].get("repulsion_strength", 0.5)
        self.gravity_power = config["Atomiser"].get("gravity_power", 2.0)
        self.error_offset = config["Atomiser"].get("error_offset", 0.1)
        self.danger_zone_divisor = config["Atomiser"].get("danger_zone_divisor", 2.0)
        
        self.use_density_spreading = config["Atomiser"].get("use_density_spreading", True)
        self.density_iters = config["Atomiser"].get("density_iters", 3)
        self.density_sigma_mult = config["Atomiser"].get("density_sigma_mult", 0.5)
        self.density_step_mult = config["Atomiser"].get("density_step_mult", 0.1)
        self.max_density_step_mult = config["Atomiser"].get("max_density_step_mult", 0.25)
        
        # =====================================================================
        # 9. PREDICTOR-ONLY MODE
        # =====================================================================
        self.predictor_only = config["Atomiser"].get("predictor_only", False)
        
        # =====================================================================
        # 10. INITIALIZE COMPONENTS
        # =====================================================================
        self._init_latents()
        self._init_geographic_pruning()
        self._init_displacement_updater()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()
        
        self._log_config()
        
        if self.predictor_only:
            self._apply_predictor_only_mode()

    def _log_config(self):
        """Log important configuration settings."""
        print(f"[Atomiser] Input dim: {self.input_dim}, Latent dim: {self.latent_dim}")
        print(f"[Atomiser] Spatial latents: {self.num_spatial_latents}, Global: {self.num_global_latents}")
        print(f"[Atomiser] Depth: {self.depth}, self_per_cross_attn: {self.self_per_cross_attn}")
        
        print(f"[Atomiser] RoPE Configuration:")
        print(f"[Atomiser]   encoder_use_rpe={self.encoder_use_rpe}")
        print(f"[Atomiser]   decoder_use_rpe={self.decoder_use_rpe}")
        print(f"[Atomiser]   self_attn_use_rpe={self.use_rpe}")
        if self.encoder_use_rpe or self.decoder_use_rpe or self.use_rpe:
            print(f"[Atomiser]   rope_base={self.rope_base}")
            print(f"[Atomiser]   rope_reference_gsd={self.rope_reference_gsd}")
        
        print(f"[Atomiser] Geographic: k={self.geo_k}, σ={self.geo_sigma}")
        
        if self.use_hybrid_self_attention:
            print(f"[Atomiser] Self-Attention: HYBRID")
        elif self.use_rpe:
            print(f"[Atomiser] Self-Attention: RoPE")
        elif self.use_gaussian_bias:
            print(f"[Atomiser] Self-Attention: Gaussian bias")
        else:
            print(f"[Atomiser] Self-Attention: Standard")
        
        if self.use_gravity_displacement:
            print(f"[Atomiser] Displacement: TWO-PHASE DYNAMICS")
        elif self.use_error_guided_displacement:
            print(f"[Atomiser] Displacement: ERROR-GUIDED")
        elif self.use_displacement:
            print(f"[Atomiser] Displacement: MLP-BASED")
        else:
            print(f"[Atomiser] Displacement: DISABLED")

    

    def _init_latents(self):
        """Initialize learnable latent vectors."""

        self.use_learned_latents = self.config["latent_grids"].get("use_learned_latents", False)
        
        if self.use_learned_latents:
            # OLD STYLE: Each spatial latent has its own learned embedding
            self.spatial_latents = nn.Parameter(torch.randn(self.num_spatial_latents, self.latent_dim))
            nn.init.trunc_normal_(self.spatial_latents, std=0.02, a=-2., b=2.)
            self.latent_pos_encoder = None
            
        else:
            # NEW STYLE: Shared content + position encoding
            self.spatial_latent_content = nn.Parameter(torch.randn(self.latent_dim))
            nn.init.trunc_normal_(self.spatial_latent_content, std=0.02, a=-2., b=2.)

            self.latent_pos_encoder = LatentPositionEncoder(
                output_dim=self.latent_dim,
                num_bands=self.config["latent_grids"].get("latent_pos_num_bands", 32),
                max_freq=self.config["latent_grids"].get("latent_pos_max_freq", 32),
                normalize_scale=self.latent_surface,
                init_scale=0.02,
            )
            print("[Atomiser] Using NEW STYLE: shared content + APE")
        
        # Global latents (same for both)
        self.global_latents = nn.Parameter(torch.randn(self.num_global_latents, self.latent_dim))
        nn.init.trunc_normal_(self.global_latents, std=0.02, a=-2., b=2.)


    
    def _init_geographic_pruning(self):
        """Initialize geographic pruning module."""
        self.geo_pruning = GeographicPruning(
            geometry=self.input_processor.geometry,
            num_spatial_latents=self.num_spatial_latents,
            geo_k=self.geo_k,
            default_sigma=self.geo_sigma,
        )


    def _init_latents_with_positions(
            self, 
            batch_size: int, 
            coords: torch.Tensor,
            device: torch.device
        ) -> torch.Tensor:
        """Initialize latents based on mode."""
        
        if self.use_learned_latents:
            # OLD STYLE
            spa_latents = repeat(self.spatial_latents, 'n d -> b n d', b=batch_size)
        else:
            
            # NEW STYLE
            L_spatial = self.num_spatial_latents
            spa_content = repeat(self.spatial_latent_content, 'd -> b n d', b=batch_size, n=L_spatial)
            pos_encoding = self.latent_pos_encoder(coords)
            spa_latents = spa_content + pos_encoding*0
        
        glob_latents = repeat(self.global_latents, 'n d -> b n d', b=batch_size)
        return torch.cat([spa_latents, glob_latents], dim=1)
    
    def _init_displacement_updater(self):
        """Initialize the position update strategy from config."""
        self.error_displacement = None
        self.gravity_displacement = None
        self.position_updater = None
        
        if self.use_gravity_displacement:
            self.gravity_displacement = GravityDisplacement(
                latent_dim=self.latent_dim,
                num_latents_per_row=self.spatial_latents_per_row,
                max_displacement=self.max_displacement,
                min_displacement=self.min_displacement,
                repulsion_strength=self.repulsion_strength,
                gravity_power=self.gravity_power,
                depth=self.depth,
                share_weights=self.share_error_predictor_weights,
                latent_surface=self.latent_surface,
                error_offset=self.error_offset,
                danger_zone_divisor=self.danger_zone_divisor,
                use_density_spreading=self.use_density_spreading,
                density_iters=self.density_iters,
                density_sigma_mult=self.density_sigma_mult,
                density_step_mult=self.density_step_mult,
                max_density_step_mult=self.max_density_step_mult,
                freeze_boundary=self.config["Atomiser"].get("freeze_boundary", False)
            )
        elif self.use_error_guided_displacement:
            self.error_displacement = ErrorGuidedDisplacement(
                latent_dim=self.latent_dim,
                num_latents_per_row=self.spatial_latents_per_row,
                max_displacement=self.max_displacement,
                min_displacement=self.min_displacement,
                depth=self.depth,
                share_weights=self.share_error_predictor_weights,
                latent_surface=self.latent_surface,
            )
        elif self.use_displacement:
            displacement_config = {
                "use_displacement": self.use_displacement,
                "position_strategy": self.position_strategy,
                "latent_dim": self.latent_dim,
                "depth": self.depth,
                "max_displacement": self.max_displacement,
                "share_displacement_weights": self.share_displacement_weights,
                "num_spatial_latents": self.num_spatial_latents,
            }
            self.position_updater = create_position_updater(displacement_config)
    
    def _init_encoder_layers(self):
        """Initialize encoder layers with Local RoPE."""
        
        # Cross-attention with RoPE
        get_cross_attn = cache_fn(lambda: PreNorm(
            self.latent_dim,
            LocalCrossAttentionRoPE(
                dim_query=self.latent_dim,
                dim_context=self.input_dim,
                dim_out=self.latent_dim,
                heads=self.cross_heads,
                dim_head=self.cross_dim_head,
                dropout=self.attn_dropout,
                use_rope=self.encoder_use_rpe,
                rope_base=self.rope_base,
                rope_reference_gsd=self.rope_reference_gsd,
                rope_learnable_scale=self.rope_learnable_scale,
            )
        ))
        
        get_cross_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))
        
        # Self-attention options
        if self.use_hybrid_self_attention:
            self.hybrid_self_attn = HybridSelfAttention(
                dim=self.latent_dim,
                k=self.self_attn_k,
                heads=self.latent_heads,
                dim_head=self.latent_dim_head,
                ff_mult=4,
                dropout=self.attn_dropout,
                use_rpe=self.use_rpe,
                use_gaussian_bias=self.use_gaussian_bias,
                sigma_init=self.gaussian_sigma,
                learnable_sigma=self.learnable_sigma,
                num_blocks=self.self_per_cross_attn,
                has_global=self.num_global_latents > 0,
                share_weights=self.weight_tie_layers,
                rpe_num_bands=32,
                rpe_max_freq=32,
                rpe_normalize_scale=self.latent_spacing,
            )
            get_latent_attn = None
            get_latent_ff = None
        
        

        elif self.use_rpe:
            self.hybrid_self_attn = None
            get_latent_attn = cache_fn(lambda: PreNormRPEConcat(
                self.latent_dim,
                SelfAttentionRPEConcat(
                    dim=self.latent_dim,
                    heads=self.latent_heads,
                    dim_head=self.latent_dim_head,
                    dropout=self.attn_dropout,
                    rpe_num_bands=self.config["Atomiser"].get("rpe_num_bands", 32),
                    rpe_max_freq=self.config["Atomiser"].get("rpe_max_freq", 32.0),
                    rpe_normalize_scale=self.latent_spacing,
                )
            ))
            get_latent_ff = cache_fn(lambda: PreNorm(
                self.latent_dim,
                FeedForward(self.latent_dim, dropout=self.ff_dropout)
            ))
            
        elif self.use_gaussian_bias:
   
            self.hybrid_self_attn = None
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
            get_latent_ff = cache_fn(lambda: PreNorm(
                self.latent_dim,
                FeedForward(self.latent_dim, dropout=self.ff_dropout)
            ))
        else:
            self.hybrid_self_attn = None
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
        
        # Build encoder layers
        self.encoder_layers = nn.ModuleList([])
        
        for layer_idx in range(self.depth):
            should_cache = self.weight_tie_layers and layer_idx > 0
            cache_key = 0 if should_cache else layer_idx
            
            cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
            cross_ff = get_cross_ff(_cache=should_cache, key=f"cross_ff_{cache_key}")
            
            if self.use_hybrid_self_attention:
                self_attns = None
            else:
                self_attns = nn.ModuleList([])
                for sa_idx in range(self.self_per_cross_attn):
                    sa_cache_key = f"{cache_key}_{sa_idx}" if should_cache else f"{layer_idx}_{sa_idx}"
                    self_attn = get_latent_attn(_cache=should_cache, key=f"self_attn_{sa_cache_key}")
                    self_ff = get_latent_ff(_cache=should_cache, key=f"self_ff_{sa_cache_key}")
                    self_attns.append(nn.ModuleList([self_attn, self_ff]))
            
            self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _init_decoder(self):
        """Initialize decoder with Local RoPE."""
        decoder_context_dim = self.latent_dim + self.decoder_pe_dim
        
        self.decoder_cross_attn = LocalCrossAttentionRoPE(
            dim_query=self.query_dim_recon,
            dim_context=decoder_context_dim,
            dim_out=self.latent_dim,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head,
            dropout=self.attn_dropout,
            use_rope=self.decoder_use_rpe,
            rope_base=self.rope_base,
            rope_reference_gsd=self.rope_reference_gsd,
            rope_learnable_scale=self.rope_learnable_scale,
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
            nn.Linear(hidden_dim, self.num_classes)
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
    # Predictor-Only Mode
    # =========================================================================
    
    def _apply_predictor_only_mode(self):
        """Freeze all model components EXCEPT the error predictor."""
        has_error_predictor = (
            self.gravity_displacement is not None or 
            self.error_displacement is not None
        )
        
        if not has_error_predictor:
            raise ValueError(
                "predictor_only=True requires use_gravity_displacement=True or "
                "use_error_guided_displacement=True!"
            )
        
        total_params_before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.freeze_all()
        self._unfreeze_error_predictor_only()
        total_params_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"[Atomiser] Predictor-only mode: {total_params_before:,} → {total_params_after:,} trainable params")
    
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def _unfreeze_error_predictor_only(self):
        if self.gravity_displacement is not None:
            if self.gravity_displacement.share_weights:
                for param in self.gravity_displacement.error_predictor.parameters():
                    param.requires_grad = True
            else:
                for predictor in self.gravity_displacement.error_predictors:
                    for param in predictor.parameters():
                        param.requires_grad = True
        
        if self.error_displacement is not None:
            if self.error_displacement.share_weights:
                for param in self.error_displacement.error_predictor.parameters():
                    param.requires_grad = True
            else:
                for predictor in self.error_displacement.error_predictors:
                    for param in predictor.parameters():
                        param.requires_grad = True

    # =========================================================================
    # Coordinate Utilities
    # =========================================================================

    def _get_default_latent_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        grid = self.input_processor.geometry.get_default_latent_grid(device)
        return grid.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # =========================================================================
    # Single Encoder Layer
    # =========================================================================
    
    def _encode_single_layer(
        self,
        latents: torch.Tensor,
        current_coords: torch.Tensor,
        initial_spatial_latents: torch.Tensor,
        geo_tokens: torch.Tensor,
        geo_masks: torch.Tensor,
        perm: Optional[torch.Tensor],
        token_centers_lut: Optional[torch.Tensor],
        gsd_lut: Optional[torch.Tensor],
        layer_idx: int,
        cross_attn: nn.Module,
        cross_ff: nn.Module,
        self_attns: Optional[nn.ModuleList],
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Single encoder layer with Local RoPE."""
        B = latents.shape[0]
        L_spatial = self.num_spatial_latents
        
        # Sample tokens
        if perm is not None:
            sampled_tokens = geo_tokens[:, :, perm, :]
            sampled_masks = geo_masks[:, :, perm]
        else:
            sampled_tokens = geo_tokens
            sampled_masks = geo_masks
        
        processed_tokens = self.input_processor.process_data_for_encoder(
            sampled_tokens, sampled_masks, latent_positions=current_coords
        )
        
        latents_spatial = latents[:, :L_spatial, :]
        latents_global = latents[:, L_spatial:, :] if self.num_global_latents > 0 else None
        
        # =================================================================
        # COMPUTE POSITION AND GSD FOR ROPE
        # =================================================================
        delta_x = None
        delta_y = None
        gsd = None
        
        if self.encoder_use_rpe and token_centers_lut is not None:
            token_x_idx = sampled_tokens[:, :, :, 1].long()
            token_y_idx = sampled_tokens[:, :, :, 2].long()
            token_x = token_centers_lut[token_x_idx]
            token_y = token_centers_lut[token_y_idx]
            
            # Relative positions (local: latent at origin)
            delta_x = token_x - current_coords[:, :, 0:1]
            delta_y = token_y - current_coords[:, :, 1:2]
            
            # Token GSD
            if gsd_lut is not None:
                band_idx = sampled_tokens[:, :, :, 0].long()
                gsd = gsd_lut[band_idx]
        
        # =================================================================
        # CROSS-ATTENTION WITH ROPE
        # =================================================================
        
        spatial_out = cross_attn(
            latents_spatial,
            context=processed_tokens,
            mask=~sampled_masks,
            delta_x=delta_x,
            delta_y=delta_y,
            gsd=gsd,
        )
        
        latents_spatial = spatial_out + latents_spatial
        latents_spatial = cross_ff(latents_spatial) + latents_spatial
        
        if latents_global is not None:
            latents = torch.cat([latents_spatial, latents_global], dim=1)
        else:
            latents = latents_spatial
        
        # =================================================================
        # SELF-ATTENTION
        # =================================================================
        if self.use_hybrid_self_attention:
            hybrid_cache = self.hybrid_self_attn.compute_cache(current_coords)
            latents = self.hybrid_self_attn(latents, hybrid_cache, num_spatial=L_spatial)
        elif self.use_rpe:
            for self_attn, self_ff in self_attns:
                latents = self_attn(
                    latents,
                    positions=current_coords,  # [B, L_spatial, 2]
                    num_spatial=L_spatial,
                    gsd=None,
                ) + latents
                latents = self_ff(latents) + latents
        elif self.use_gaussian_bias:
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents, positions=current_coords, num_spatial=L_spatial) + latents
                latents = self_ff(latents) + latents
        else:
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents
        
        # =================================================================
        # DISPLACEMENT
        # =================================================================
        latents_spatial = latents[:, :L_spatial, :]
        displacement_enabled = layer_idx < (self.depth - self.stable_depth)
        predicted_error = None
        
        if displacement_enabled and self.use_gravity_displacement:
            new_coords, displacement, predicted_error = self.gravity_displacement(
                latents_spatial, current_coords, layer_idx
            )
        elif displacement_enabled and self.use_error_guided_displacement:
            new_coords, displacement, predicted_error = self.error_displacement(
                latents_spatial, current_coords, layer_idx
            )
        elif displacement_enabled and self.position_updater is not None:
            new_coords, displacement = self.position_updater(
                latents_spatial, current_coords, layer_idx
            )
        else:
            new_coords = current_coords
        
        # Reset spatial latents if displacement enabled
        if displacement_enabled and (
            self.use_gravity_displacement or 
            self.use_error_guided_displacement or 
            self.position_updater is not None
        ):
            latents = torch.cat([
                initial_spatial_latents.clone(),
                latents[:, L_spatial:, :]
            ], dim=1)
        
        return latents, new_coords, predicted_error

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
        return_predicted_errors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], Optional[Dict[str, Any]], Optional[List[torch.Tensor]]]:
        """Encode with Local RoPE."""
        B = tokens.shape[0]
        L_spatial = self.num_spatial_latents
        device = tokens.device
        
        # =========================================================================
        # INITIALIZE COORDINATES
        # =========================================================================
        
        current_coords = self._get_default_latent_coords(B, device)
      
        # =========================================================================
        # INITIALIZE LATENTS WITH POSITION ENCODING
        # =========================================================================
        latents = self._init_latents_with_positions(B, current_coords, device)

       
        num_spatial = latents.shape[1] - self.num_global_latents
        spatial_idx = torch.randperm(num_spatial, device=latents.device)

        # 2. Create the index for the global latents (keeping them at the end)
        # If self.num_global_latents > 1, this handles the whole block
        global_idx = torch.arange(num_spatial, latents.shape[1], device=latents.device)

        # 3. Concatenate them
        full_idx = torch.cat([spatial_idx, global_idx])

        # 4. Apply to latents and coordinates
        #latents = latents[:, full_idx, :]
        # Coordinates only exist for spatial latents, so we use the spatial_idx specifically
        current_coords = current_coords[:, spatial_idx, :]

        

       
        initial_spatial_latents = latents[:, :L_spatial, :].clone()

        # Tracking
        trajectory = [current_coords.clone()] if return_trajectory else None
        all_displacements = [] if return_displacement_stats else None
        predicted_errors_list = [] if return_predicted_errors else None
        
        # Initial geographic pruning
        geo_tokens, geo_masks, _ = self.geo_pruning(tokens, mask, current_coords)
        k = geo_tokens.shape[2]

  

        
        # Cache LUTs
        token_centers_lut = None
        gsd_lut = None
        if self.encoder_use_rpe:
            _, _, token_centers_lut = self.input_processor.geometry.get_integral_constants()
            if hasattr(self.input_processor, 'get_gsd_lut'):
                gsd_lut = self.input_processor.get_gsd_lut()
        
        # Pre-generate random permutations
        m = self.geo_m_train if training else self.geo_m_val
        m = min(m, k)
        
        if m < k:
            all_perms = [torch.randperm(k, device=device)[:m] for _ in range(self.depth)]
        else:
            all_perms = [None] * self.depth
        
        num_layers = len(self.encoder_layers)
        
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            perm = all_perms[layer_idx]
            
            # Run single layer
            if self.use_checkpoint and self.training:
                latents, new_coords, predicted_error = checkpoint(
                    self._encode_single_layer,
                    latents,
                    current_coords,
                    initial_spatial_latents,
                    geo_tokens,
                    geo_masks,
                    perm,
                    token_centers_lut,
                    gsd_lut,
                    layer_idx,
                    cross_attn,
                    cross_ff,
                    self_attns,
                    training,
                    use_reentrant=False
                )
            else:
                latents, new_coords, predicted_error = self._encode_single_layer(
                    latents,
                    current_coords,
                    initial_spatial_latents,
                    geo_tokens,
                    geo_masks,
                    perm,
                    token_centers_lut,
                    gsd_lut,
                    layer_idx,
                    cross_attn,
                    cross_ff,
                    self_attns,
                    training,
                )
            
            # Store predicted error
            if return_predicted_errors and predicted_error is not None:
                predicted_errors_list.append(predicted_error)
            
            # Track displacement statistics
            if return_displacement_stats:
                displacement = new_coords - current_coords
                disp_magnitude = torch.norm(displacement, dim=-1)
                displacement_enabled = layer_idx < (self.depth - self.stable_depth)
                all_displacements.append({
                    'layer': layer_idx,
                    'displacement_enabled': displacement_enabled,
                    'mean_magnitude': disp_magnitude.mean().item(),
                    'max_magnitude': disp_magnitude.max().item(),
                    'std_magnitude': disp_magnitude.std().item(),
                })
            
            # Update coordinates
            current_coords = new_coords
            
            # Record trajectory
            if return_trajectory:
                trajectory.append(current_coords.clone())
            
            # Re-compute geographic pruning
            if layer_idx < num_layers - 1:
                geo_tokens, geo_masks, _ = self.geo_pruning(tokens, mask, current_coords)
                k = geo_tokens.shape[2]
                
                m = self.geo_m_train if training else self.geo_m_val
                m = min(m, k)
                if m < k and layer_idx + 1 < num_layers:
                    all_perms[layer_idx + 1] = torch.randperm(k, device=device)[:m]
        
        # Aggregate displacement stats
        final_disp_stats = None
        if return_displacement_stats and all_displacements:
            final_disp_stats = self._aggregate_displacement_stats(all_displacements, trajectory)
        
        return latents, current_coords, trajectory, final_disp_stats, predicted_errors_list

    def _aggregate_displacement_stats(
        self, 
        stats_list: List[Dict[str, Any]], 
        trajectory: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        aggregated = {
            'per_layer': stats_list,
            'stable_depth': self.stable_depth,
        }
        
        enabled_stats = [s for s in stats_list if s.get('displacement_enabled', True)]
        mean_mags = [s['mean_magnitude'] for s in enabled_stats]
        max_mags = [s['max_magnitude'] for s in enabled_stats]
        
        aggregated['mean_displacement_per_layer'] = mean_mags
        aggregated['cumulative_mean_displacement'] = sum(mean_mags) if mean_mags else 0.0
        aggregated['max_single_layer_displacement'] = max(max_mags) if max_mags else 0.0
        aggregated['num_displacement_layers'] = len(enabled_stats)
        
        if trajectory is not None and len(trajectory) > 1:
            total_disp = trajectory[-1] - trajectory[0]
            total_mag = torch.norm(total_disp, dim=-1)
            aggregated['total_displacement'] = {
                'mean': total_mag.mean().item(),
                'max': total_mag.max().item(),
                'std': total_mag.std().item(),
            }
        
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
        """Reconstruct query tokens using spatial latents with Local RoPE."""
        B, N, _ = query_tokens.shape
        L_spatial = self.num_spatial_latents
        device = latents.device
        D = latents.shape[-1]
        k = self.decoder_k_spatial
        
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask
        )
        
        # Query positions in meters
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)
        
        # Find k nearest latents
        dists_sq = (
            query_coords.unsqueeze(2) - latents_coords.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        
        _, topk_indices = torch.topk(dists_sq, k=k, dim=-1, largest=False)
        
        spatial_latents = latents[:, :L_spatial, :]
        
        # Gather latents
        flat_indices = topk_indices.reshape(B, N * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_exp)
        selected_latents = gathered.reshape(B, N, k, D)
        
        # Gather latent positions
        flat_coord_indices = flat_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(latents_coords, dim=1, index=flat_coord_indices)
        selected_latent_coords = gathered_coords.reshape(B, N, k, 2)
        
        # Compute relative deltas for RoPE (latent - query, since query is the "origin")
        delta_x = selected_latent_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)  # [B, N, k]
        delta_y = selected_latent_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)  # [B, N, k]
        
        # Compute relative PE for concatenation (existing approach)
        scale = self.latent_surface / (self.spatial_latents_per_row - 1)
        relative_pe = self.input_processor.pos_encoder(
            delta_x, delta_y, scale, gsd=0.2
        )
        
        # Context = latent features + relative PE
        context = torch.cat([selected_latents, relative_pe], dim=-1)
        
        # Cross-attention with RoPE
        output = self.decoder_cross_attn(
            query_features,
            context,
            delta_x=delta_x,
            delta_y=delta_y,
            gsd=None,  # Decoder doesn't need GSD modulation
        )
        
        output_with_query = torch.cat([output, query_features], dim=-1)
        predictions = self.output_head(output_with_query)
        
        return predictions
    
    def classify(self, latents: torch.Tensor) -> torch.Tensor:
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
        return_predicted_errors: bool = False,
    ):
        need_trajectory = return_trajectory or task == "visualization"
        need_disp_stats = return_displacement_stats or task == "visualization"
        need_pred_errors = return_predicted_errors or task == "visualization"
        
        latents, final_coords, trajectory, disp_stats, predicted_errors = self.encode(
            data, mask, latents_coords, training, 
            return_trajectory=need_trajectory,
            return_displacement_stats=need_disp_stats,
            return_predicted_errors=need_pred_errors,
        )
        
        if task == "encoder":
            result = {'latents': latents, 'final_coords': final_coords}
            if trajectory is not None:
                result['trajectory'] = trajectory
            if disp_stats is not None:
                result['displacement_stats'] = disp_stats
            if predicted_errors is not None:
                result['predicted_errors'] = predicted_errors
            return result
        
        if task == "reconstruction" or task == "visualization":
            chunk_size = 10000
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
                    'latents': latents,
                    'trajectory': trajectory,
                    'displacement_stats': disp_stats,
                    'final_coords': final_coords,
                    'predicted_errors': predicted_errors,
                }
            
            if return_predicted_errors:
                return {
                    'predictions': predictions,
                    'latents': latents,
                    'final_coords': final_coords,
                    'trajectory': trajectory,
                    'predicted_errors': predicted_errors,
                }
            return predictions
        
        else:  # classification
            return self.classify(latents)

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
        self.spatial_latent_content.requires_grad = False  # Changed from spatial_latent_content
        self.global_latents.requires_grad = False
        self._set_requires_grad(self.input_processor, False)

    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.spatial_latent_content.requires_grad = True  # Changed from spatial_latent_content
        self.global_latents.requires_grad = True
        self._set_requires_grad(self.input_processor, True)
    
    def unfreeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, True)
        self._set_requires_grad(self.output_head, True)

    def freeze_decoder(self):
        """NEW: Explicitly freeze decoder components for classification-only tasks."""
        if hasattr(self, 'decoder_cross_attn'):
            self._set_requires_grad(self.decoder_cross_attn, False)
        if hasattr(self, 'output_head'):
            self._set_requires_grad(self.output_head, False)
        print("[Atomiser] Decoder and Reconstruction Head FROZEN.")
    
    def freeze_classifier(self):
        self._set_requires_grad(self.to_logits, False)
    
    def unfreeze_classifier(self):
        self._set_requires_grad(self.to_logits, True)
    
    def freeze_displacement(self):
        if self.position_updater is not None:
            self._set_requires_grad(self.position_updater, False)
        if self.error_displacement is not None:
            self._set_requires_grad(self.error_displacement, False)
        if self.gravity_displacement is not None:
            self._set_requires_grad(self.gravity_displacement, False)
    
    def unfreeze_displacement(self):
        if self.position_updater is not None:
            self._set_requires_grad(self.position_updater, True)
        if self.error_displacement is not None:
            self._set_requires_grad(self.error_displacement, True)
        if self.gravity_displacement is not None:
            self._set_requires_grad(self.gravity_displacement, True)

    # =========================================================================
    # Trajectory Analysis
    # =========================================================================
    
    def compute_trajectory_stats(self, trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        if trajectory is None or len(trajectory) < 2:
            return {}
        
        stats = {
            'num_steps': len(trajectory) - 1,
            'per_step_displacement': [],
            'cumulative_displacement': [],
            'stable_depth': self.stable_depth,
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