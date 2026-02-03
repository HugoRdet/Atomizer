"""
Atomiser Model with Local RoPE for Positional Encoding

Key Features:
- Local RoPE: Q at origin (unchanged), K rotated by relative position
- Resolution-aware: log-scale GSD modulation
- Modular encode() with separate cross/self attention steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from functools import wraps
from dataclasses import dataclass
from einops import repeat, rearrange
from typing import Optional, Tuple, List, Dict, Any


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
)

from .RPE import (
    LocalCrossAttentionRoPE,
    SelfAttentionRoPE,
    PreNormRoPE
)

from .RPE_gaussian import (
    SelfAttentionRoPEWithGaussianBias,
    PreNormRoPEGaussian
)

from .gaussian_bias import (
    SelfAttentionWithGaussianBias
)

from .geographic_pruning import (
    GeographicPruning,
)

from .hybrid_self_attention import (
    HybridSelfAttention,
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


@dataclass
class EncoderOutput:
    """Structured output from encoder."""
    latents: torch.Tensor
    coords: torch.Tensor
    trajectory: Optional[List[torch.Tensor]] = None
    displacement_stats: Optional[Dict[str, Any]] = None
    predicted_errors: Optional[List[torch.Tensor]] = None


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
        self.geo_pruning = nn.ModuleDict()
        
        # =====================================================================
        # 2. DIMENSIONS
        # =====================================================================
        self.input_dim = self.input_processor.get_encoder_output_dim()
        self.query_dim_recon = self.input_processor.get_decoder_output_dim()
        self.latent_dim = config["Atomiser"].get("latent_dim", self.input_dim)
        self.decoder_pe_dim = self.input_processor.pos_encoder.get_output_dim()
        
        # =====================================================================
        # 3. GLOBAL LATENTS (shared across modalities)
        # =====================================================================
        self.num_global_latents = config["Atomiser"].get("global_latents", 0)
        
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
        
        # Decoder parameters
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)
        
        # =====================================================================
        # 5. ROPE CONFIGURATION
        # =====================================================================
        self.encoder_use_rpe = config["Atomiser"]["RPE"].get("encoder_use_rpe", False)
        self.decoder_use_rpe = config["Atomiser"]["RPE"].get("decoder_use_rpe", False)
        self.use_rpe = config["Atomiser"]["RPE"].get("selfattn_use_rpe", False)
        self.rope_learnable_scale = config["Atomiser"].get("rope_learnable_scale", True)
        
        # =====================================================================
        # 6. SELF-ATTENTION MODE
        # =====================================================================
        self.use_gaussian_bias = config["Atomiser"].get("use_gaussian_bias", False)
        self.gaussian_sigma = config["Atomiser"].get("gaussian_sigma", 9.0)
        self.learnable_sigma = config["Atomiser"].get("learnable_sigma", True)
        self.use_hybrid_self_attention = config["Atomiser"].get("use_hybrid_self_attention", False)
        self.self_attn_k = config["Atomiser"].get("self_attn_k", 64)
        
        # =====================================================================
        # 7. INITIALIZE COMPONENTS
        # =====================================================================
        self._init_latents()
        self._init_geographic_pruning()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_latents(self):
        """Initialize learnable latent vectors (shared across modalities)."""
        self.spatial_latent_content = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.spatial_latent_content, std=0.02, a=-2., b=2.)

        if self.num_global_latents > 0:
            self.global_latents = nn.Parameter(torch.randn(self.num_global_latents, self.latent_dim))
            nn.init.trunc_normal_(self.global_latents, std=0.02, a=-2., b=2.)
        else:
            self.register_buffer('global_latents', None)

    def _init_geographic_pruning(self):
        """Initialize geographic pruning module for each modality."""
        for grid_name, grid_options in self.config["latent_grids"].items():
            span = grid_options["span"]
            latents_per_row = grid_options["latents"]
            spacing = span / (latents_per_row - 1)
            auto_sigma = 1.0 * spacing
            
            self.geo_pruning[grid_name] = GeographicPruning(
                geometry=self.input_processor.geometry,
                num_spatial_latents=latents_per_row ** 2,
                geo_k=grid_options["geo_k"],
                default_sigma=auto_sigma,
            )

    def _init_encoder_layers(self):
        """Initialize encoder layers with Local RoPE."""
        
        self_rope_compression_scale = self.config["RoPE"].get("self_compression_scale", 50.0)
        cross_rope_compression_scale = self.config["RoPE"].get("cross_compression_scale", 10.0)
        rope_base = self.config["RoPE"].get("base", 10000.0)
        
        # Cross-attention factory
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
                rope_compression_scale=cross_rope_compression_scale,
                rope_learnable_scale=self.rope_learnable_scale,
            )
        ))
        
        get_cross_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))

        # Self-attention factory (depends on mode)
        get_latent_attn, get_latent_ff = self._create_self_attention_factories(
            rope_base, self_rope_compression_scale
        )
        
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

    def _create_self_attention_factories(self, rope_base: float, compression_scale: float):
        """Create self-attention factories based on configuration."""
        
        # Compute latent spacing for hybrid attention (use first modality as reference)
        first_modality = next(iter(self.config["latent_grids"].values()))
        latent_spacing = first_modality["span"] / (first_modality["latents"] - 1)
        
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
                rpe_normalize_scale=latent_spacing,
            )
            return None, None

        self.hybrid_self_attn = None

        if self.use_rpe and self.use_gaussian_bias:
            get_latent_attn = cache_fn(lambda: PreNormRoPEGaussian(
                self.latent_dim,
                SelfAttentionRoPEWithGaussianBias(
                    dim=self.latent_dim,
                    heads=self.latent_heads,
                    dim_head=self.latent_dim_head,
                    dropout=self.attn_dropout,
                    use_rope=True,
                    rope_base=rope_base,
                    rope_compression_scale=compression_scale,
                    rope_learnable_scale=self.rope_learnable_scale,
                    use_gaussian_bias=True,
                    sigma=self.gaussian_sigma,
                    learnable_sigma=self.learnable_sigma,
                )
            ))
        elif self.use_rpe:
            get_latent_attn = cache_fn(lambda: PreNormRoPE(
                self.latent_dim,
                SelfAttentionRoPE(
                    dim=self.latent_dim,
                    heads=self.latent_heads,
                    dim_head=self.latent_dim_head,
                    dropout=self.attn_dropout,
                    use_rope=True,
                    rope_base=rope_base,
                    rope_compression_scale=compression_scale,
                    rope_learnable_scale=self.rope_learnable_scale,
                )
            ))
        elif self.use_gaussian_bias:
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
        
        return get_latent_attn, get_latent_ff

    def _init_decoder(self):
        """Initialize decoder with Local RoPE."""
        
        rope_compression_scale = self.config["RoPE"].get("cross_compression_scale", 10.0)
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
            rope_compression_scale=rope_compression_scale,
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
    # Latent & Coordinate Initialization
    # =========================================================================

    def get_spatial_latents(self, batch_size: int, grid_config: dict) -> torch.Tensor:
        """Initialize spatial latents for a given grid configuration."""
        num_latents = grid_config["latents"] ** 2
        return repeat(self.spatial_latent_content, 'd -> b n d', b=batch_size, n=num_latents)
    
    def get_global_latents(self, batch_size: int) -> Optional[torch.Tensor]:
        """Initialize global latents (shared across modalities)."""
        if self.global_latents is None:
            return None
        return repeat(self.global_latents, 'n d -> b n d', b=batch_size)

    def get_default_coords(self, batch_size: int, device: torch.device, grid_config: dict) -> torch.Tensor:
        """Get default latent coordinates for a modality."""
        grid = self.input_processor.geometry.get_default_latent_grid(grid_config)
        return grid.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)

    def combine_latents(
        self, 
        spatial_latents: torch.Tensor, 
        global_latents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Combine spatial and global latents."""
        if global_latents is not None:
            return torch.cat([spatial_latents, global_latents], dim=1)
        return spatial_latents

    # =========================================================================
    # Token Sampling
    # =========================================================================

    def _sample_tokens(
        self, 
        geo_tokens: torch.Tensor, 
        geo_masks: torch.Tensor, 
        grid_config: dict,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from geographic pruning output."""
        k = geo_tokens.shape[2]
        m = grid_config.get("train_k", 500) if training else grid_config.get("val_k", 500)
        m = min(m, k)
        
        if m < k:
            perm = torch.randperm(k, device=geo_tokens.device)[:m]
            return geo_tokens[:, :, perm, :], geo_masks[:, :, perm]
        
        return geo_tokens, geo_masks

    # =========================================================================
    # Compute Deltas (for RoPE)
    # =========================================================================

    def _compute_deltas(
        self, 
        sampled_tokens: torch.Tensor, 
        coords: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute relative positions and GSD for RoPE."""
        if not self.encoder_use_rpe:
            return None, None, None
        
        _, _, token_centers_lut = self.input_processor.geometry.get_integral_constants()
        
        token_x_idx = sampled_tokens[:, :, :, 1].long()
        token_y_idx = sampled_tokens[:, :, :, 2].long()
        token_x = token_centers_lut[token_x_idx]
        token_y = token_centers_lut[token_y_idx]
        
        delta_x = token_x - coords[:, :, 0:1]
        delta_y = token_y - coords[:, :, 1:2]
        
        # GSD (optional)
        gsd = None
        if hasattr(self.input_processor, 'get_gsd_lut'):
            gsd_lut = self.input_processor.get_gsd_lut()
            if gsd_lut is not None:
                band_idx = sampled_tokens[:, :, :, 0].long()
                gsd = gsd_lut[band_idx]
        
        return delta_x, delta_y, gsd

    # =========================================================================
    # Attention Steps (Modular)
    # =========================================================================

    def _cross_attention_step(
        self,
        latents: torch.Tensor,
        sampled_tokens: torch.Tensor,
        sampled_masks: torch.Tensor,
        coords: torch.Tensor,
        cross_attn: nn.Module,
        cross_ff: nn.Module,
        L_spatial: int,
    ) -> torch.Tensor:
        """Single cross-attention step: latents attend to tokens."""
        
        # Process tokens
        processed_tokens = self.input_processor.process_data_for_encoder(
            sampled_tokens, sampled_masks, latent_positions=coords
        )
        
        # Compute deltas for RoPE
        delta_x, delta_y, gsd = self._compute_deltas(sampled_tokens, coords)
        
        # Cross-attention on spatial latents only
        spatial = latents[:, :L_spatial]
        spatial = cross_attn(
            spatial,
            context=processed_tokens,
            mask=~sampled_masks,
            delta_x=delta_x,
            delta_y=delta_y,
            gsd=gsd,
        ) + spatial
        spatial = cross_ff(spatial) + spatial
        
        # Recombine with global latents
        return torch.cat([spatial, latents[:, L_spatial:]], dim=1)

    def _self_attention_step(
        self,
        latents: torch.Tensor,
        coords: torch.Tensor,
        self_attns: Optional[nn.ModuleList],
        L_spatial: int,
    ) -> torch.Tensor:
        """Single self-attention step: latents attend to each other."""
        
        if self.use_hybrid_self_attention:
            hybrid_cache = self.hybrid_self_attn.compute_cache(coords)
            return self.hybrid_self_attn(latents, hybrid_cache, num_spatial=L_spatial)
        
        if self.use_rpe or self.use_gaussian_bias:
            px = coords[..., 0]
            py = coords[..., 1]
            
            for self_attn, self_ff in self_attns:
                if self.use_rpe:
                    latents = self_attn(latents, pos_x=px, pos_y=py, num_spatial=L_spatial) + latents
                else:
                    latents = self_attn(latents, positions=coords, num_spatial=L_spatial) + latents
                latents = self_ff(latents) + latents
        else:
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents
        
        return latents

    # =========================================================================
    # Main Encode
    # =========================================================================
    def prepare_cross_attention_step(self,geo_tuple,layers,L_spatial, training,latents):
        geo_tokens, geo_masks, grid_config,latent_coords = geo_tuple
        sampled_tokens, sampled_masks = self._sample_tokens(
            geo_tokens, geo_masks, grid_config, training
        )
        
        cross_attn, cross_ff=layers
        latents = self._cross_attention_step(
            latents, sampled_tokens, sampled_masks, latent_coords,
            cross_attn, cross_ff, L_spatial
        )

        return latents

    

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        modality: str = "FLAIR",
        initial_coords: Optional[torch.Tensor] = None,
        do_cross_attention: bool = True,
        do_self_attention: bool = True,
        num_layers: Optional[int] = None,
        training: bool = True,
        return_trajectory: bool = False,
    ) -> EncoderOutput:
        """
        Encode tokens into latent representations.
        
        Args:
            tokens: [B, N, token_dim] input tokens
            mask: [B, N] boolean mask (True = masked/invalid)
            modality: Which modality config to use (e.g., "FLAIR", "S2")
            initial_coords: Optional initial latent coordinates [B, L_spatial, 2]
            do_cross_attention: Whether to perform cross-attention
            do_self_attention: Whether to perform self-attention
            num_layers: Override number of layers (default: self.depth)
            training: Whether in training mode (affects token sampling)
            return_trajectory: Whether to return coordinate trajectory
        
        Returns:
            EncoderOutput with latents, coords, and optional trajectory
        """
        # =========================================
        # 1. GET MODALITY CONFIG
        # =========================================
      
        grid_config = self.config["latent_grids"][modality]
        
        L_spatial = grid_config["latents"] ** 2
        
        B = tokens.shape[0]
        device = tokens.device
        
        # =========================================
        # 2. INITIALIZE LATENTS & COORDS
        # =========================================
        spatial_latents = self.get_spatial_latents(B, grid_config)
        global_latents = self.get_global_latents(B)
        latents = self.combine_latents(spatial_latents, global_latents)
        del initial_coords
        initial_coords=None
        if initial_coords is not None:
            coords = initial_coords.clone()
        else:
            coords = self.get_default_coords(B, device, grid_config)
        
        #pruning
        geo_tokens, geo_masks, _ = self.geo_pruning[modality](tokens, mask, coords, id_modality=modality)
        
        #traj tracking
        trajectory = [coords.clone()] if return_trajectory else None
        
        #layer loop
        depth = num_layers if num_layers is not None else self.depth
        
        for layer_idx in range(depth):
            cross_attn, cross_ff, self_attns = self.encoder_layers[layer_idx]
            
            # Sample tokens
            #sampled_tokens, sampled_masks = self._sample_tokens(
            #    geo_tokens, geo_masks, grid_config, training
            #)

            geo_tuple=geo_tokens, geo_masks, grid_config,coords
            latents=self.prepare_cross_attention_step(geo_tuple,(cross_attn, cross_ff),L_spatial, training,latents)
            
            # Cross-attention: latents ← tokens
            #if do_cross_attention:
            #    latents = self._cross_attention_step(
            #        latents, sampled_tokens, sampled_masks, coords,
            #        cross_attn, cross_ff, L_spatial
            #    )
            
            # Self-attention: latents ↔ latents
            if do_self_attention:
                latents = self._self_attention_step(
                    latents, coords, self_attns, L_spatial
                )
            
            # Record trajectory
            if return_trajectory:
                trajectory.append(coords.clone())
        
        return EncoderOutput(
            latents=latents,
            coords=coords,
            trajectory=trajectory,
        )

    # =========================================================================
    # Reconstruct
    # =========================================================================

    def reconstruct(
        self, 
        latents: torch.Tensor, 
        latents_coords: torch.Tensor, 
        query_tokens: torch.Tensor, 
        query_mask: torch.Tensor,
        L_spatial: Optional[int] = None,
    ) -> torch.Tensor:
        """Reconstruct query tokens using spatial latents."""
        
        B, N, _ = query_tokens.shape
        device = latents.device
        D = latents.shape[-1]
        k = self.decoder_k_spatial


        
        
        # Infer L_spatial if not provided
        if L_spatial is None:
            L_spatial = latents_coords.shape[1]
        
        # Query features
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask
        )
        
        # Query positions in meters
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)
        
        # Find k nearest latents
        spatial_latents = latents[:, :L_spatial, :]
        dists_sq = (query_coords.unsqueeze(2) - latents_coords.unsqueeze(1)).pow(2).sum(dim=-1)
        _, topk_indices = torch.topk(dists_sq, k=k, dim=-1, largest=False)
        
        # Gather latents
        flat_indices = topk_indices.reshape(B, N * k)
        flat_indices_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(spatial_latents, dim=1, index=flat_indices_exp)
        selected_latents = gathered.reshape(B, N, k, D)
        
        # Gather latent positions
        flat_coord_indices = flat_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(latents_coords, dim=1, index=flat_coord_indices)
        selected_latent_coords = gathered_coords.reshape(B, N, k, 2)
        
        # Compute relative deltas
        delta_x = selected_latent_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_latent_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)
        
        # Relative position encoding
        relative_pe = self.input_processor.pos_encoder(delta_x, delta_y)
        
        # Context = latent features + relative PE
        context = torch.cat([selected_latents, relative_pe], dim=-1)
        
        # Cross-attention
        output = self.decoder_cross_attn(
            query_features,
            context,
            delta_x=delta_x,
            delta_y=delta_y,
            gsd=torch.tensor([0.2], device=device),
        )
        
        # Output head
        output_with_query = torch.cat([output, query_features], dim=-1)
        return self.output_head(output_with_query)
    
    def classify(self, latents: torch.Tensor) -> torch.Tensor:
        """Classification from latents."""
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
        initial_coords: Optional[torch.Tensor] = None,
        training: bool = True, 
        task: str = "reconstruction",
        do_cross_attention: bool = True,
        do_self_attention: bool = True,
        return_trajectory: bool = False,
        # Backward compatibility (ignored for now)
        latents_coords: Optional[torch.Tensor] = None,  # Old name for initial_coords
        return_displacement_stats: bool = False,
        return_predicted_errors: bool = False,
    ):
        """
        Main forward pass.
        
        Args:
            data: Input tokens [B, N, token_dim]
            mask: Token mask [B, N]
            modality: Which modality to use
            mae_tokens: Query tokens for reconstruction [B, M, token_dim]
            mae_tokens_mask: Query mask [B, M]
            initial_coords: Optional initial coordinates
            training: Training mode flag
            task: "reconstruction", "classification", "encoder", or "visualization"
            do_cross_attention: Enable cross-attention
            do_self_attention: Enable self-attention
            return_trajectory: Return coordinate trajectory
            latents_coords: (Deprecated) Use initial_coords instead
            return_displacement_stats: (Deprecated) Not implemented in refactored version
            return_predicted_errors: (Deprecated) Not implemented in refactored version
        """
        # Backward compatibility: latents_coords -> initial_coords
        if latents_coords is not None and initial_coords is None:
            initial_coords = latents_coords
        
        need_trajectory = return_trajectory or task == "visualization"
        modality="FLAIR"
        # Encode
        encoder_output = self.encode(
            data, mask,
            modality=modality,
            initial_coords=initial_coords,
            do_cross_attention=do_cross_attention,
            do_self_attention=do_self_attention,
            training=training, 
            return_trajectory=need_trajectory,
        )
        
        latents = encoder_output.latents
        final_coords = encoder_output.coords
        trajectory = encoder_output.trajectory
        
        # Get L_spatial from modality config
        grid_config = self.config["latent_grids"][modality]
        L_spatial = grid_config["latents"] ** 2
        
        if task == "encoder":
            result = {
                'latents': latents,
                'final_coords': final_coords,
                'trajectory': trajectory,
            }
            # Backward compatibility: add empty fields
            if return_displacement_stats:
                result['displacement_stats'] = None
            if return_predicted_errors:
                result['predicted_errors'] = None
            return result
        
        if task in ("reconstruction", "visualization"):
            # Chunked reconstruction for memory efficiency
            chunk_size = 10000
            N = mae_tokens.shape[1]
  
            if N > chunk_size:
                preds_list = []
                for i in range(0, N, chunk_size):
                    chunk_tokens = mae_tokens[:, i:i + chunk_size]
                    chunk_mask = mae_tokens_mask[:, i:i + chunk_size]
                    preds_list.append(self.reconstruct(
                        latents, final_coords, chunk_tokens, chunk_mask, L_spatial
                    ))
                predictions = torch.cat(preds_list, dim=1)
            else:
                predictions = self.reconstruct(
                    latents, final_coords, mae_tokens, mae_tokens_mask, L_spatial
                )
            
            if task == "visualization":
                result = {
                    'predictions': predictions,
                    'latents': latents,
                    'trajectory': trajectory,
                    'final_coords': final_coords,
                }
                if return_predicted_errors:
                    result['predicted_errors'] = None
                if return_displacement_stats:
                    result['displacement_stats'] = None
                return result
            
            # Backward compatibility for return_predicted_errors
            if return_predicted_errors:
                return {
                    'predictions': predictions,
                    'latents': latents,
                    'final_coords': final_coords,
                    'trajectory': trajectory,
                    'predicted_errors': None,
                }
            
            return predictions
        
        else:  # classification
            return self.classify(latents)

    # =========================================================================
    # Freeze/Unfreeze Utilities
    # =========================================================================
    
    def _set_requires_grad(self, module, flag: bool):
        if module is None:
            return
        if isinstance(module, torch.Tensor):
            module.requires_grad = flag
        elif hasattr(module, 'parameters'):
            for param in module.parameters():
                param.requires_grad = flag
    
    def freeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, False)
        self.spatial_latent_content.requires_grad = False
        if self.global_latents is not None:
            self.global_latents.requires_grad = False
        self._set_requires_grad(self.input_processor, False)

    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.spatial_latent_content.requires_grad = True
        if self.global_latents is not None:
            self.global_latents.requires_grad = True
        self._set_requires_grad(self.input_processor, True)
    
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

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True