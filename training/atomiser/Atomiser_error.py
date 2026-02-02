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


# =============================================================================
# IMPORTS
# =============================================================================

from training.utils.token_building.processor import TokenProcessor

from .augmented_qk import (
    SelfAttentionWithAugmentedQK,
    PreNormAugmentedQK,
)

from .nn_comp import (
    PreNorm, 
    SelfAttention, 
    FeedForward, 
    LatentAttentionPooling, 
    PreNormWithPositions,
    LatentPositionEncoder
)

from .self_attn_cart import(
    PreNormTargeting,
    TargetingSelfAttention,
)

from .RPE import (
    LocalCrossAttentionRoPE,
    PreNormRPEConcat,
    SelfAttentionRPEConcat,
    SelfAttentionRoPE,
    PreNormRoPE
)

from .RPE_gaussian import (
    SelfAttentionRoPEWithGaussianBias,
    PreNormRoPEGaussian
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
        self.geo_pruning = dict()
        
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
        self.decoder_pe_dim = self.input_processor.pos_encoder.get_output_dim()
        
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
        self.geo_m_train = config["Atomiser"].get("geo_m_train", 500)
        self.geo_m_val = config["Atomiser"].get("geo_m_val", 500)
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)
        
        # =====================================================================
        # 5. MEMORY OPTIMIZATION
        # =====================================================================
        self.use_checkpoint = config["Atomiser"].get("use_checkpoint", False)
        
        # =====================================================================
        # 6. ROPE CONFIGURATION
        # =====================================================================
        self.encoder_use_rpe = config["Atomiser"]["RPE"].get("encoder_use_rpe", False)
        self.decoder_use_rpe = config["Atomiser"]["RPE"].get("decoder_use_rpe", False)
        self.use_rpe = config["Atomiser"]["RPE"].get("selfattn_use_rpe", False)
        
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

        self.use_augmented_qk = config["Atomiser"].get("use_augmented_qk", False)
        self.augmented_qk_num_bands = config["Atomiser"].get("augmented_qk_num_bands", 32)
        self.augmented_qk_max_freq = config["Atomiser"].get("augmented_qk_max_freq", 32.0)
        self.augmented_qk_compression_scale = config["Atomiser"].get("augmented_qk_compression_scale", 50.0)
        self.augmented_qk_include_gsd = config["Atomiser"].get("augmented_qk_include_gsd", True)
        self.augmented_qk_reference_gsd = config["Atomiser"].get("augmented_qk_reference_gsd", 1.0)

        self.use_targeting_self_attention = config["Atomiser"].get("use_targeting_self_attention", False)
        self.targeting_num_bands = config["Atomiser"].get("targeting_num_bands", 32)
        self.targeting_max_freq = config["Atomiser"].get("targeting_max_freq", 32.0)
        self.targeting_normalize_scale = config["Atomiser"].get("targeting_normalize_scale", self.latent_surface / 2)
        
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
        
        if self.predictor_only:
            self._apply_predictor_only_mode()

    def _init_latents(self):
        """Initialize learnable latent vectors."""        
        self.spatial_latent_content = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.spatial_latent_content, std=0.02, a=-2., b=2.)

        self.global_latents = nn.Parameter(torch.randn(self.num_global_latents, self.latent_dim))
        nn.init.trunc_normal_(self.global_latents, std=0.02, a=-2., b=2.)

    def _init_geographic_pruning(self):
        """Initialize geographic pruning module."""
        for grid_name in self.config["latent_grids"]:
            grid_options = self.config["latent_grids"][grid_name]
            
            span = grid_options["span"]
            latents_per_row = grid_options["latents"]
            spacing = span / (latents_per_row - 1)
            auto_sigma = 1.0 * spacing
            
            self.geo_pruning[grid_name] = GeographicPruning(
                geometry=self.input_processor.geometry,
                num_spatial_latents=latents_per_row**2,
                geo_k=grid_options["geo_k"],
                default_sigma=auto_sigma,
            )

    def get_spatial_latents(self, batch_size: int, grid_options: dict) -> torch.Tensor:
        """Initialize spatial latents."""
        return repeat(self.spatial_latent_content, 'd -> b n d', b=batch_size, n=grid_options["latents"]**2)
    
    def get_global_latents(self, batch_size: int) -> torch.Tensor:
        """Initialize global latents."""
        return repeat(self.global_latents, 'n d -> b n d', b=batch_size)
    
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
        """Initialize encoder layers with Local RoPE using compression."""
        
        self_rope_compression_scale = self.config["RoPE"].get("self_compression_scale", 50.0)
        cross_rope_compression_scale = self.config["RoPE"].get("cross_compression_scale", 50.0)
        rope_base = self.config["RoPE"].get("base", 10000.0)
        
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
                rope_base=rope_base,
                rope_compression_scale=10,
                rope_learnable_scale=self.rope_learnable_scale,
            )
        ))
        
        get_cross_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))

        # =========================================================================
        # Self-Attention Configuration
        # =========================================================================
        
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
                rpe_num_bands=self.config["latent_config"].get("latent_pos_num_bands", 32),
                rpe_max_freq=self.config["latent_config"].get("latent_pos_max_freq", 32),
                rpe_normalize_scale=self.latent_spacing,
            )
            get_latent_attn = None
            get_latent_ff = None

        elif self.use_rpe and self.use_gaussian_bias:
            self.hybrid_self_attn = None
            
            get_latent_attn = cache_fn(lambda: PreNormRoPEGaussian(
                self.latent_dim,
                SelfAttentionRoPEWithGaussianBias(
                    dim=self.latent_dim,
                    heads=self.latent_heads,
                    dim_head=self.latent_dim_head,
                    dropout=self.attn_dropout,
                    use_rope=True,
                    rope_base=rope_base,
                    rope_compression_scale=self_rope_compression_scale,
                    rope_learnable_scale=self.rope_learnable_scale,
                    use_gaussian_bias=True,
                    sigma=self.gaussian_sigma,
                    learnable_sigma=self.learnable_sigma,
                )
            ))
            
            get_latent_ff = cache_fn(lambda: PreNorm(
                self.latent_dim,
                FeedForward(self.latent_dim, dropout=self.ff_dropout)
            ))

        elif self.use_rpe:
            self.hybrid_self_attn = None
            
            get_latent_attn = cache_fn(lambda: PreNormRoPE(
                self.latent_dim,
                SelfAttentionRoPE(
                    dim=self.latent_dim,
                    heads=self.latent_heads,
                    dim_head=self.latent_dim_head,
                    dropout=self.attn_dropout,
                    use_rope=True,
                    rope_base=rope_base,
                    rope_compression_scale=self_rope_compression_scale,
                    rope_learnable_scale=self.rope_learnable_scale,
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
        
        # =========================================================================
        # Build Encoder Layers
        # =========================================================================
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
        """Initialize decoder with Local RoPE using compression."""
        
        rope_compression_scale = self.config["RoPE"].get("compression_scale", 50.0)
        rope_base = self.config["RoPE"].get("base", 10000.0)
        
        if self.decoder_use_rpe:
            decoder_context_dim = self.latent_dim
        else:
            decoder_context_dim = self.latent_dim + self.decoder_pe_dim
        
        self.decoder_cross_attn = LocalCrossAttentionRoPE(
            dim_query=self.query_dim_recon,
            dim_context=decoder_context_dim,
            dim_out=self.latent_dim,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head,
            dropout=self.attn_dropout,
            use_rope=self.decoder_use_rpe,
            rope_base=rope_base,
            rope_compression_scale=10,
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

    def _get_default_latent_coords(self, batch_size: int, device: torch.device, config: dict) -> torch.Tensor:
        grid = self.input_processor.geometry.get_default_latent_grid(config)
        return grid.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
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
    # Encode
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
        """
        Encode tokens into latent representations.
        
        Args:
            tokens: [B, N, token_dim] input tokens
            mask: [B, N] boolean mask (True = valid)
            latents_coords: Optional initial latent coordinates
            training: Whether in training mode (affects subsampling)
            return_trajectory: Whether to return position trajectory
            return_displacement_stats: Whether to return displacement statistics
            return_predicted_errors: Whether to return predicted errors
        
        Returns:
            latents: [B, L, D] final latent representations
            current_coords: [B, L_spatial, 2] final latent coordinates
            trajectory: Optional list of coordinate tensors
            disp_stats: Optional displacement statistics dict
            predicted_errors: Optional list of predicted error tensors
        """
        B = tokens.shape[0]
        L_spatial = self.num_spatial_latents
        device = tokens.device
        
        modality = "FLAIR"

        del latents_coords
        latents_coords=None

        # Initialize coordinates
        if latents_coords is not None:
            current_coords = latents_coords.clone()
        else:
            current_coords = self._get_default_latent_coords(
                B, device, self.config["latent_grids"][modality]
            )
        
        # Initialize latents
        spatial_latents = self.get_spatial_latents(B, self.config["latent_grids"][modality])
        global_latents = self.get_global_latents(B)
        
        if global_latents is not None:
            latents = torch.cat([spatial_latents, global_latents], dim=1)
        else:
            latents = spatial_latents
        
        initial_spatial_latents = latents[:, :L_spatial, :].clone()

        # Tracking lists
        trajectory = [current_coords.clone()] if return_trajectory else None
        all_displacements = [] if return_displacement_stats else None
        predicted_errors_list = [] if return_predicted_errors else None

        # Geographic pruning
        geo_tokens, geo_masks, _ = self.geo_pruning[modality](
            tokens, mask, current_coords, id_modality=modality
        )
        
        k = geo_tokens.shape[2]

        # Cache lookup tables
        token_centers_lut = None
        gsd_lut = None
        
        if self.encoder_use_rpe:
            _, _, token_centers_lut = self.input_processor.geometry.get_integral_constants()
            if hasattr(self.input_processor, 'get_gsd_lut'):
                gsd_lut = self.input_processor.get_gsd_lut()

        # Subsampling configuration
        geo_train_k = self.config["latent_grids"][modality].get("train_k", 500)
        geo_val_k = self.config["latent_grids"][modality].get("val_k", 500)
        
        m = geo_train_k if training else geo_val_k
        m = min(m, k)
        
        # Pre-generate random permutations for subsampling
        if m < k:
            all_perms = [torch.randperm(k, device=device)[:m] for _ in range(self.depth)]
        else:
            all_perms = [None] * self.depth
        
        num_layers = len(self.encoder_layers)
        
        # Encoder layers
        for layer_idx, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_layers):
            perm = all_perms[layer_idx]
            
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
            
            # Re-compute geographic pruning for next layer (if coords changed)
            if layer_idx < num_layers - 1:
                displacement_enabled = layer_idx < (self.depth - self.stable_depth)
                should_reprune = displacement_enabled and (
                    self.use_gravity_displacement or 
                    self.use_error_guided_displacement or 
                    self.position_updater is not None
                )
                
                if should_reprune:
                    del geo_tokens, geo_masks
                    
                    geo_tokens, geo_masks, _ = self.geo_pruning[modality](
                        tokens, mask, current_coords, id_modality=modality
                    )
                    
                    k = geo_tokens.shape[2]
                    m = geo_train_k if training else geo_val_k
                    m = min(m, k)

                    if m < k and layer_idx + 1 < num_layers:
                        all_perms[layer_idx + 1] = torch.randperm(k, device=device)[:m]
        
        # Aggregate displacement stats
        final_disp_stats = None
        if return_displacement_stats and all_displacements:
            final_disp_stats = self._aggregate_displacement_stats(all_displacements, trajectory)
        
        return latents, current_coords, trajectory, final_disp_stats, predicted_errors_list

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
        """Single encoder layer."""
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
        
        # Compute position and GSD for RoPE (cross-attention)
        delta_x = None
        delta_y = None
        gsd = None
        
        if self.encoder_use_rpe and token_centers_lut is not None:
            token_x_idx = sampled_tokens[:, :, :, 1].long()
            token_y_idx = sampled_tokens[:, :, :, 2].long()
            token_x = token_centers_lut[token_x_idx]
            token_y = token_centers_lut[token_y_idx]
            
            delta_x = token_x - current_coords[:, :, 0:1]
            delta_y = token_y - current_coords[:, :, 1:2]
            
            if gsd_lut is not None:
                band_idx = sampled_tokens[:, :, :, 0].long()
                gsd = gsd_lut[band_idx]
        
        # Cross-attention
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
        
        # Self-attention
        if self.use_hybrid_self_attention:
            hybrid_cache = self.hybrid_self_attn.compute_cache(current_coords)
            latents = self.hybrid_self_attn(latents, hybrid_cache, num_spatial=L_spatial)
        
        elif self.use_rpe:
            px = current_coords[..., 0]
            py = current_coords[..., 1]

            for self_attn, self_ff in self_attns:
                latents = self_attn(
                    latents,
                    pos_x=px,
                    pos_y=py,
                    num_spatial=L_spatial
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
        
        # Displacement
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
        
        # Compute relative deltas for RoPE
        delta_x = selected_latent_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_latent_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)


        relative_pe = self.input_processor.pos_encoder(
            delta_x, delta_y
        )
        
        # Context = latent features + relative PE
        context = torch.cat([selected_latents, relative_pe], dim=-1)
        
        # Cross-attention with RoPE
        output = self.decoder_cross_attn(
            query_features,
            context,
            delta_x=delta_x,
            delta_y=delta_y,
            gsd=torch.tensor([0.2], device=device),
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

        # Encoder
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
        self.spatial_latent_content.requires_grad = False
        self.global_latents.requires_grad = False
        self._set_requires_grad(self.input_processor, False)

    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.spatial_latent_content.requires_grad = True
        self.global_latents.requires_grad = True
        self._set_requires_grad(self.input_processor, True)
    
    def unfreeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, True)
        self._set_requires_grad(self.output_head, True)

    def freeze_decoder(self):
        """Freeze decoder components for classification-only tasks."""
        if hasattr(self, 'decoder_cross_attn'):
            self._set_requires_grad(self.decoder_cross_attn, False)
        if hasattr(self, 'output_head'):
            self._set_requires_grad(self.output_head, False)
    
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