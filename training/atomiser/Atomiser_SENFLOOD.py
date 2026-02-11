"""
Atomiser Model — Grouped Token Format
======================================

Accepts batch dict from dataloader:
{
    "groups": {res: {"tokens": [B,N,6], "mask": [B,N], "shape": (C,H,W)}},
    "queries": [B, M, 6],
    "queries_mask": [B, M],
    "label": [B, H, W],
    "target_resolution": float,
    "image": [B, C, H, W],
}

Key changes from original:
- No modality strings — grid configs computed at runtime from resolution + shape
- Single GeographicPruning instance (params overridden per resolution)
- encode() loops over resolution groups for cross-attention
- Latent grid computed from runtime config, not yaml
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

from training.utils.token_building.processor import TokenProcessor
from training.utils.datasets.token_grouping import compute_grid_config

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
    PreNormRoPE,
)

from .RPE_gaussian import (
    SelfAttentionRoPEWithGaussianBias,
    PreNormRoPEGaussian,
)

from .gaussian_bias import SelfAttentionWithGaussianBias
from .geographic_pruning import GeographicPruning
from .hybrid_self_attention import HybridSelfAttention


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

class Atomiser_Senflood(pl.LightningModule):

    def __init__(self, *, config, lookup_table):
        super().__init__()
        self.save_hyperparameters(ignore=['lookup_table'])
        self.config = config

        # =====================================================================
        # 1. INPUT PROCESSOR
        # =====================================================================
        self.input_processor = TokenProcessor(config, lookup_table)

        # =====================================================================
        # 2. DIMENSIONS
        # =====================================================================
        self.input_dim = self.input_processor.get_encoder_output_dim()
        self.query_dim_recon = self.input_processor.get_decoder_output_dim()
        self.latent_dim = config["Atomiser"].get("latent_dim", self.input_dim)
        self.decoder_pe_dim = self.input_processor.pos_encoder.get_output_dim()

        # =====================================================================
        # 3. LATENT GRID CONFIG (replaces per-modality yaml blocks)
        # =====================================================================
        latent_cfg = config.get("latent_grid", {})
        self.pixels_per_latent = latent_cfg.get("pixels_per_latent", 50)
        self.sigma_factor = latent_cfg.get("sigma_factor", 1.5)
        self.max_k = latent_cfg.get("max_k", 2000)
        self.hexagonal = latent_cfg.get("hexagonal", False)

        # =====================================================================
        # 4. GLOBAL LATENTS
        # =====================================================================
        self.num_global_latents = config["Atomiser"].get("global_latents", 0)

        # =====================================================================
        # 5. ARCHITECTURE PARAMETERS
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
        self.decoder_k_spatial = config["Atomiser"].get("decoder_k_spatial", 4)

        # =====================================================================
        # 6. ROPE CONFIGURATION
        # =====================================================================
        self.encoder_use_rpe = config["Atomiser"]["RPE"].get("encoder_use_rpe", False)
        self.decoder_use_rpe = config["Atomiser"]["RPE"].get("decoder_use_rpe", False)
        self.use_rpe = config["Atomiser"]["RPE"].get("selfattn_use_rpe", False)
        self.rope_learnable_scale = config["Atomiser"].get("rope_learnable_scale", True)

        # =====================================================================
        # 7. SELF-ATTENTION MODE
        # =====================================================================
        self.use_gaussian_bias = config["Atomiser"].get("use_gaussian_bias", False)
        self.gaussian_sigma = config["Atomiser"].get("gaussian_sigma", 9.0)
        self.learnable_sigma = config["Atomiser"].get("learnable_sigma", True)
        self.use_hybrid_self_attention = config["Atomiser"].get("use_hybrid_self_attention", False)
        self.self_attn_k = config["Atomiser"].get("self_attn_k", 64)

        # =====================================================================
        # 8. INITIALIZE COMPONENTS
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
        """Initialize learnable latent vectors (resolution-agnostic)."""
        self.spatial_latent_content = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.spatial_latent_content, std=0.02, a=-2., b=2.)

        if self.num_global_latents > 0:
            self.global_latents = nn.Parameter(
                torch.randn(self.num_global_latents, self.latent_dim)
            )
            nn.init.trunc_normal_(self.global_latents, std=0.02, a=-2., b=2.)
        else:
            self.register_buffer('global_latents', None)

    def _init_geographic_pruning(self):
        """Single GeographicPruning instance. All params passed at call time."""
        self.geo_pruning = GeographicPruning(
            geometry=self.input_processor.geometry,
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

        # Self-attention factory
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
                    sa_key = f"{cache_key}_{sa_idx}" if should_cache else f"{layer_idx}_{sa_idx}"
                    self_attn = get_latent_attn(_cache=should_cache, key=f"self_attn_{sa_key}")
                    self_ff = get_latent_ff(_cache=should_cache, key=f"self_ff_{sa_key}")
                    self_attns.append(nn.ModuleList([self_attn, self_ff]))

            self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _create_self_attention_factories(self, rope_base: float, compression_scale: float):
        """Create self-attention factories based on configuration."""
        # Approximate latent spacing for RPE normalization
        rpe_normalize_scale = self.config["Atomiser"].get(
            "rpe_normalize_scale",
            self.pixels_per_latent * 10.0,  # fallback: pixels_per_latent * ~resolution
        )

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
                rpe_normalize_scale=rpe_normalize_scale,
            )
            return None, None

        self.hybrid_self_attn = None

        if self.use_rpe and self.use_gaussian_bias:
            get_latent_attn = cache_fn(lambda: PreNormRoPEGaussian(
                self.latent_dim,
                SelfAttentionRoPEWithGaussianBias(
                    dim=self.latent_dim, heads=self.latent_heads,
                    dim_head=self.latent_dim_head, dropout=self.attn_dropout,
                    use_rope=True, rope_base=rope_base,
                    rope_compression_scale=compression_scale,
                    rope_learnable_scale=self.rope_learnable_scale,
                    use_gaussian_bias=True, sigma=self.gaussian_sigma,
                    learnable_sigma=self.learnable_sigma,
                )
            ))
        elif self.use_rpe:
            get_latent_attn = cache_fn(lambda: PreNormRoPE(
                self.latent_dim,
                SelfAttentionRoPE(
                    dim=self.latent_dim, heads=self.latent_heads,
                    dim_head=self.latent_dim_head, dropout=self.attn_dropout,
                    use_rope=True, rope_base=rope_base,
                    rope_compression_scale=compression_scale,
                    rope_learnable_scale=self.rope_learnable_scale,
                )
            ))
        elif self.use_gaussian_bias:
            get_latent_attn = cache_fn(lambda: PreNormWithPositions(
                self.latent_dim,
                SelfAttentionWithGaussianBias(
                    dim=self.latent_dim, heads=self.latent_heads,
                    dim_head=self.latent_dim_head, dropout=self.attn_dropout,
                    sigma=self.gaussian_sigma, learnable_sigma=self.learnable_sigma,
                )
            ))
        else:
            get_latent_attn = cache_fn(lambda: PreNorm(
                self.latent_dim,
                SelfAttention(
                    dim=self.latent_dim, heads=self.latent_heads,
                    dim_head=self.latent_dim_head, dropout=self.attn_dropout,
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
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _init_classifier(self):
        """Initialize classification head."""
        if self.config["Atomiser"].get("final_classifier_head", True):
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(
                    self.latent_dim, heads=self.latent_heads,
                    dim_head=self.latent_dim_head, dropout=self.attn_dropout,
                ),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.num_classes),
            )
        else:
            self.to_logits = nn.Identity()

    # =========================================================================
    # Latent Helpers
    # =========================================================================

    def get_spatial_latents(self, batch_size: int, L_spatial: int) -> torch.Tensor:
        """Repeat the single learned latent vector to fill the grid."""
        return repeat(self.spatial_latent_content, 'd -> b n d', b=batch_size, n=L_spatial)

    def get_global_latents(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.global_latents is None:
            return None
        return repeat(self.global_latents, 'n d -> b n d', b=batch_size)

    def combine_latents(
        self,
        spatial_latents: torch.Tensor,
        global_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if global_latents is not None:
            return torch.cat([spatial_latents, global_latents], dim=1)
        return spatial_latents

    # ============================================================================
    # In Atomiser_Senflood (replace _compute_latent_grid method)
    # ============================================================================

    def _compute_latent_grid(
        self,
        grid_config: dict,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute latent positions from runtime grid config.
        
        Supports two layouts:
        - Square: latents at cell centers (all Voronoi cells same size)
        - Hexagonal: staggered rows for better 2D coverage
        
        Returns [B, L_spatial, 2] in meters (centered at image origin).
        """
        lx = grid_config["latents_x"]
        ly = grid_config["latents_y"]
        span_x = grid_config["span_x"]
        span_y = grid_config["span_y"]
        hexagonal = grid_config.get("hexagonal", False)
        
        if hexagonal:
            grid = self._create_hexagonal_grid(lx, ly, span_x, span_y, device)
        else:
            grid = self._create_square_grid(lx, ly, span_x, span_y, device)
        
        return grid.unsqueeze(0).expand(batch_size, -1, -1)


    def _create_square_grid(
        self,
        lx: int,
        ly: int,
        span_x: float,
        span_y: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a square grid with latents at cell centers.
        
        All Voronoi cells have equal size (including edges).
        
        Example (lx=4, span_x=100):
            step = 25
            positions: -37.5, -12.5, +12.5, +37.5
            
            ┌────┬────┬────┬────┐
            │ ●  │ ●  │ ●  │ ●  │  Each cell is 25×25
            ├────┼────┼────┼────┤
            │ ●  │ ●  │ ●  │ ●  │
            └────┴────┴────┴────┘
        
        Returns:
            grid: [lx*ly, 2] tensor of (x, y) coordinates
        """
        # Cell-centered: latents at center of each cell
        step_x = span_x / lx
        step_y = span_y / ly
        
        start_x = -span_x / 2.0 + step_x / 2.0
        end_x = span_x / 2.0 - step_x / 2.0
        
        start_y = -span_y / 2.0 + step_y / 2.0
        end_y = span_y / 2.0 - step_y / 2.0
        
        xs = torch.linspace(start_x, end_x, lx, device=device) if lx > 1 \
            else torch.zeros(1, device=device)
        ys = torch.linspace(start_y, end_y, ly, device=device) if ly > 1 \
            else torch.zeros(1, device=device)
        
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        return grid


    def _create_hexagonal_grid(
        self,
        lx: int,
        ly: int,
        span_x: float,
        span_y: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a hexagonal grid with staggered rows.
        
        Odd rows are offset by half the horizontal step.
        Some latents may extend slightly past image boundaries - this is intentional
        to maintain the hexagonal pattern without deformation.
        
        Example (lx=5, ly=5):
            Row 0: ●     ●     ●     ●     ●      (even: starts at -span/2)
            Row 1:   ●     ●     ●     ●     ●    (odd: offset by step/2)
            Row 2: ●     ●     ●     ●     ●
            Row 3:   ●     ●     ●     ●     ●
            Row 4: ●     ●     ●     ●     ●
        
        Returns:
            grid: [lx*ly, 2] tensor of (x, y) coordinates
        """
        half_span_x = span_x / 2.0
        half_span_y = span_y / 2.0
        
        # Spacing between latents
        step_x = span_x / (lx - 1) if lx > 1 else 0
        step_y = span_y / (ly - 1) if ly > 1 else 0
        
        # Offset for odd rows (half horizontal step)
        offset = step_x / 2.0
        
        grid_points = []
        
        for row_idx in range(ly):
            # Y coordinate: from -half_span_y to +half_span_y
            y = -half_span_y + row_idx * step_y if ly > 1 else 0.0
            
            # X offset for odd rows
            x_offset = offset if (row_idx % 2 == 1) else 0.0
            
            for col_idx in range(lx):
                # X coordinate: may extend past boundary on odd rows (intentional)
                x = -half_span_x + col_idx * step_x + x_offset if lx > 1 else 0.0
                grid_points.append([x, y])
        
        grid = torch.tensor(grid_points, dtype=torch.float32, device=device)
        
        return grid

    # =========================================================================
    # Geographic Pruning
    # =========================================================================

    def _apply_pruning(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        grid_config: dict,
        L_spatial: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run geographic pruning for one resolution group."""
        geo_tokens, geo_masks, _ = self.geo_pruning(
            tokens, mask, coords,
            geo_k=grid_config["geo_k"],
            sigma=grid_config["geo_sigma"],
            L_spatial=L_spatial,
            hexagonal=grid_config.get("hexagonal", False),  
        )
        return geo_tokens, geo_masks

    # =========================================================================
    # Token Sampling
    # =========================================================================

    def _sample_tokens(
        self,
        geo_tokens: torch.Tensor,
        geo_masks: torch.Tensor,
        grid_config: dict,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sub-sample tokens per latent for cross-attention."""
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
        coords: torch.Tensor,
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

        gsd = None
        if hasattr(self.input_processor, 'get_gsd_lut'):
            gsd_lut = self.input_processor.get_gsd_lut()
            if gsd_lut is not None:
                band_idx = sampled_tokens[:, :, :, 0].long()
                gsd = gsd_lut[band_idx]

        return delta_x, delta_y, gsd

    # =========================================================================
    # Attention Steps
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
        """Single cross-attention step: spatial latents attend to tokens."""
        processed_tokens = self.input_processor.process_data_for_encoder(
            sampled_tokens, sampled_masks, latent_positions=coords
        )

        delta_x, delta_y, gsd = self._compute_deltas(sampled_tokens, coords)

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

        return torch.cat([spatial, latents[:, L_spatial:]], dim=1)

    def _self_attention_step(
        self,
        latents: torch.Tensor,
        coords: torch.Tensor,
        self_attns: Optional[nn.ModuleList],
        L_spatial: int,
    ) -> torch.Tensor:
        """Self-attention: all latents attend to each other."""
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
    # Encode
    # =========================================================================

    def encode(
        self,
        groups: Dict[float, dict],
        grid_configs: Dict[float, dict],
        primary_config: dict,
        training: bool = True,
        return_trajectory: bool = False,
    ) -> EncoderOutput:
        """
        Encode resolution-grouped tokens into latent representations.
        
        For each encoder layer:
          1. Cross-attention: latents attend to each resolution group sequentially
          2. Self-attention: all latents attend to each other
        
        Args:
            groups:         {resolution: {"tokens": [B,N,6], "mask": [B,N], "shape": ...}}
            grid_configs:   {resolution: compute_grid_config output}
            primary_config: grid config for the finest resolution (determines latent grid)
            training:       affects token sub-sampling
            return_trajectory: whether to record latent coords per layer
        """
        first_group = next(iter(groups.values()))
        B = first_group["tokens"].shape[0]
        device = first_group["tokens"].device

        L_spatial = primary_config["L_spatial"]

        # ── Init latents & coords ───────────────────────────
        spatial_latents = self.get_spatial_latents(B, L_spatial)
        global_latents = self.get_global_latents(B)
        latents = self.combine_latents(spatial_latents, global_latents)

        coords = self._compute_latent_grid(primary_config, B, device)

        # ── Geographic pruning (once, before layer loop) ────
        geo_cache = {}
        for res in sorted(groups.keys()):
            tokens = groups[res]["tokens"]
            mask = groups[res]["mask"]
            gc = grid_configs[res]
            geo_tokens, geo_masks = self._apply_pruning(tokens, mask, coords, gc, L_spatial)
        
            geo_cache[res] = (geo_tokens, geo_masks, gc)

        # ── Trajectory tracking ─────────────────────────────
        trajectory = [coords.clone()] if return_trajectory else None

        # ── Layer loop ──────────────────────────────────────
        for layer_idx in range(self.depth):
            cross_attn, cross_ff, self_attns = self.encoder_layers[layer_idx]

            # Cross-attention: one pass per resolution group
            for res in sorted(groups.keys()):
                geo_tokens, geo_masks, gc = geo_cache[res]

                sampled_tokens, sampled_masks = self._sample_tokens(
                    geo_tokens, geo_masks, gc, training
                )

                latents = self._cross_attention_step(
                    latents, sampled_tokens, sampled_masks, coords,
                    cross_attn, cross_ff, L_spatial,
                )

            # Self-attention: latents ↔ latents (global)
            latents = self._self_attention_step(
                latents, coords, self_attns, L_spatial
            )

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
        L_spatial: int,
        target_resolution: float = None,
    ) -> torch.Tensor:
        """Reconstruct query tokens using nearest spatial latents.
        
        Args:
            latents:           [B, L_total, D] all latent vectors
            latents_coords:    [B, L_spatial, 2] spatial latent positions in meters
            query_tokens:      [B, N, 8] raw query data
            query_mask:        [B, N] valid query mask
            L_spatial:         number of spatial latents
            target_resolution: float (m/px), the resolution to reconstruct at.
                               Passed to decoder so query features are 
                               resolution-aware. If None, uses per-token 
                               resolution_idx from query_tokens[:, :, 6].
        """
        B, N, _ = query_tokens.shape
        device = latents.device
        D = latents.shape[-1]
        k = self.decoder_k_spatial

        # ── Query features (spectral + resolution) ──────────
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask,
            target_resolution=target_resolution,
        )

        # ── Query positions in meters ───────────────────────
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)

        # ── Find k nearest spatial latents ──────────────────
        spatial_latents = latents[:, :L_spatial, :]
        dists_sq = (
            query_coords.unsqueeze(2) - latents_coords.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        _, topk_indices = torch.topk(dists_sq, k=k, dim=-1, largest=False)

        # ── Gather latents & coords ─────────────────────────
        flat_indices = topk_indices.reshape(B, N * k)

        flat_exp = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_latents = torch.gather(
            spatial_latents, 1, flat_exp
        ).reshape(B, N, k, D)

        flat_coord_exp = flat_indices.unsqueeze(-1).expand(-1, -1, 2)
        selected_coords = torch.gather(
            latents_coords, 1, flat_coord_exp
        ).reshape(B, N, k, 2)

        # ── Relative deltas ─────────────────────────────────
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)

        # ── Relative PE + context ───────────────────────────
        relative_pe = self.input_processor.pos_encoder(delta_x, delta_y)
        context = torch.cat([selected_latents, relative_pe], dim=-1)

        # ── Cross-attention ─────────────────────────────────
        output = self.decoder_cross_attn(
            query_features, context,
            delta_x=delta_x, delta_y=delta_y,
            gsd=target_resolution if target_resolution is not None else 10.0,
        )

        # ── Output head ─────────────────────────────────────
        return self.output_head(torch.cat([output, query_features], dim=-1))

    def classify(self, latents: torch.Tensor) -> torch.Tensor:
        return self.to_logits(latents)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        batch: dict,
        training: bool = True,
        task: str = "reconstruction",
        return_trajectory: bool = False,
        return_predicted_errors: bool = False,
    ):
        """
        Main forward pass.
        
        Args:
            batch: grouped dict from dataloader:
                {
                    "groups": {res: {"tokens": [B,N,8], "mask": [B,N], ...}},
                    "queries":           [B, M, 8],
                    "queries_mask":      [B, M],
                    "label":             [B, H, W],
                    "target_resolution": float (m/px),
                    "image":             [B, C, H, W],
                }
            training: training mode flag
            task: "reconstruction", "classification", "encoder", or "visualization"
            return_trajectory: return latent coords per layer
            return_predicted_errors: (compat) not implemented, always None
        """
        groups = batch["groups"]
        queries = batch["queries"]
        queries_mask = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)       # ← NEW

        # ── Compute grid configs per resolution ─────────────
        resolutions = sorted(groups.keys())
        grid_configs = {}
        for res in resolutions:
            grid_configs[res] = compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                pixels_per_latent=self.pixels_per_latent,
                sigma_factor=self.sigma_factor,
                max_k=self.max_k,
            )

        # Primary = finest resolution (determines latent grid)
        primary_res = resolutions[0]
        primary_config = grid_configs[primary_res]
        L_spatial = primary_config["L_spatial"]

        need_trajectory = return_trajectory or task == "visualization"

        # ── Encode ──────────────────────────────────────────
        encoder_output = self.encode(
            groups=groups,
            grid_configs=grid_configs,
            primary_config=primary_config,
            training=training,
            return_trajectory=need_trajectory,
        )

        latents = encoder_output.latents
        final_coords = encoder_output.coords
        trajectory = encoder_output.trajectory

        # ── Task dispatch ───────────────────────────────────
        if task == "encoder":
            result = {
                'latents': latents,
                'final_coords': final_coords,
                'trajectory': trajectory,
            }
            if return_predicted_errors:
                result['predicted_errors'] = None
            return result

        if task in ("reconstruction", "visualization"):
            # Chunked reconstruction for memory efficiency
            chunk_size = 10000
            N = queries.shape[1]

            if N > chunk_size:
                preds = []
                for i in range(0, N, chunk_size):
                    preds.append(self.reconstruct(
                        latents, final_coords,
                        queries[:, i:i + chunk_size],
                        queries_mask[:, i:i + chunk_size],
                        L_spatial,
                        target_resolution=target_resolution,            # ← NEW
                    ))
                predictions = torch.cat(preds, dim=1)
            else:
                predictions = self.reconstruct(
                    latents, final_coords, queries, queries_mask, L_spatial,
                    target_resolution=target_resolution,                # ← NEW
                )

            if task == "visualization" or return_predicted_errors:
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
    # Freeze/Unfreeze
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