"""
Atomiser Model — Multi-Resolution Encoder/Decoder
===================================================

Architecture:
- Per-resolution latent grids (different sizes based on token count)
- Cross-attention: separate per resolution
- Self-attention: concatenate → mix → split (cross-resolution information flow)
- Decoder: predict-then-interpolate with full-context MLP

Decoder pipeline:
  Per grid:
    1. Fetch top-(k+1) nearest latents during training, top-k during eval
    2. Randomly drop 1 latent during training (forces RPE usage)
    3. concat(latent_i, RPE_i, query_features, task_embed) → shared MLP → local prediction_i
    4. IDW blend (grid-spacing normalized) → grid feature
  Across grids:
    5. Learned grid gate → fused feature
  Output:
    6. reconstruction_head → scalar prediction

All tokens are flat [B, N, 8] — temporal information is encoded in column 7.
No temporal chunking or averaging. Single encode call processes the full
spatio-temporal-spectral token set.

Accepts batch dict from dataloader:
{
    "groups": {res: {"tokens": [B,N,8], "mask": [B,N], "shape": (H,W) or (C,H,W)}},
    "queries": [B, M, 8],
    "queries_mask": [B, M],
    "label": [B, H, W],
    "target_resolution": float,
    "task": str,
    "image": [B, C, H, W],
}
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
    """Structured output from encoder with per-resolution latents."""
    latents_per_res: Dict[float, torch.Tensor]  # {resolution: [B, L_res, D]}
    coords_per_res: Dict[float, torch.Tensor]   # {resolution: [B, L_res, 2]}
    trajectory: Optional[List[Dict[float, torch.Tensor]]] = None
    global_latents: Optional[torch.Tensor] = None


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
        # 3. LATENT GRID CONFIG
        # =====================================================================
        latent_cfg = config.get("latent_grid", {})
        self.tokens_per_latent = latent_cfg.get("tokens_per_latent", 2000)
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
        # 8. TASK EMBEDDING
        # =====================================================================
        self.task_embed_dim = config["Atomiser"].get("task_embed_dim", 32)

        # =====================================================================
        # 9. INITIALIZE COMPONENTS
        # =====================================================================
        self._init_latents()
        self._init_geographic_pruning()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_task_embeddings()
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

        get_latent_attn, get_latent_ff = self._create_self_attention_factories(
            rope_base, self_rope_compression_scale
        )

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
        rpe_normalize_scale = self.config["Atomiser"].get(
            "rpe_normalize_scale",
            self.tokens_per_latent * 0.1,
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

    def _init_task_embeddings(self):
        """
        Initialize learned task embeddings.

        Each task gets a [task_embed_dim] vector concatenated to the decoder
        MLP input alongside latent features, RPE, and query features.
        This lets the shared decoder specialize its output per task.

        Tasks are registered dynamically via register_task().
        A default embedding is used for unknown tasks.
        """
        self.task_embeddings = nn.ParameterDict()
        self.task_embed_default = nn.Parameter(torch.randn(self.task_embed_dim))
        nn.init.trunc_normal_(self.task_embed_default, std=0.02, a=-2., b=2.)

    def register_task(self, task_name: str):
        """
        Register a learned embedding for a task.

        Called by the trainer during setup for each active task.
        Safe to call multiple times — skips if already registered.
        """
        # nn.ParameterDict keys cannot contain '.', replace with '_'
        key = task_name.replace(".", "_")
        if key not in self.task_embeddings:
            param = nn.Parameter(torch.randn(self.task_embed_dim, device=self.task_embed_default.device))
            nn.init.trunc_normal_(param, std=0.02, a=-2., b=2.)
            self.task_embeddings[key] = param
            print(f"[Atomiser] Registered task embedding: '{task_name}' (dim={self.task_embed_dim})")

    def get_task_embedding(self, task_name: str) -> torch.Tensor:
        """Get the learned embedding for a task, falling back to default."""
        key = task_name.replace(".", "_")
        return self.task_embeddings.get(key, self.task_embed_default)

    def _init_decoder(self):
        """
        Initialize decoder: full-context MLP + IDW + grid gate.

        Local MLP receives [latent, RPE, query_features, task_embed]:
          - latent: encoded spatial content from encoder
          - RPE: relative position of query w.r.t. latent
          - query_features: spectral/temporal context (what to predict)
          - task_embed: learned task identity vector
        """
        decoder_hidden = self.latent_dim
        local_input_dim = (
            self.latent_dim
            + self.decoder_pe_dim
            + self.query_dim_recon
            + self.task_embed_dim
        )

        # ── Local predictor: full context per latent ─────────────
        self.local_predictor = nn.Sequential(
            nn.Linear(local_input_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
        )

        # ── IDW temperature ──────────────────────────────────────
        self.register_buffer(
            'decoder_temperature', torch.tensor(2.0)
        )

        # ── Grid gate ────────────────────────────────────────────
        self.grid_gate = nn.Linear(decoder_hidden, 1)

        # ── Reconstruction head ──────────────────────────────────
        hidden_dim = self.latent_dim * 2

        self.reconstruction_head = nn.Sequential(
            nn.Linear(decoder_hidden, hidden_dim),
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
        return repeat(self.spatial_latent_content, 'd -> b n d', b=batch_size, n=L_spatial)

    def get_global_latents(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.global_latents is None:
            return None
        return repeat(self.global_latents, 'n d -> b n d', b=batch_size)

    def init_latents_per_resolution(
        self,
        batch_size: int,
        grid_configs: Dict[float, dict],
        device: torch.device,
    ) -> Tuple[Dict[float, torch.Tensor], Dict[float, torch.Tensor]]:
        latents_per_res = {}
        coords_per_res = {}

        for res in sorted(grid_configs.keys()):
            gc = grid_configs[res]
            L_spatial = gc["L_spatial"]
            latents_per_res[res] = self.get_spatial_latents(batch_size, L_spatial)
            coords_per_res[res] = self._compute_latent_grid(gc, batch_size, device)

        return latents_per_res, coords_per_res

    def _compute_latent_grid(self, grid_config, batch_size, device):
        lx = grid_config["latents_x"]
        ly = grid_config["latents_y"]
        span_x = grid_config["span_x"]
        span_y = grid_config["span_y"]
        hexagonal = grid_config.get("hexagonal", False)

        if hexagonal:
            grid = self._create_hexagonal_grid(lx, ly, span_x, span_y, device)
        else:
            grid = self._create_square_grid(lx, ly, span_x, span_y, device)

        grid_config["L_spatial"] = grid.shape[0]
        return grid.unsqueeze(0).expand(batch_size, -1, -1)

    def _create_square_grid(self, lx, ly, span_x, span_y, device):
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
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    def _create_hexagonal_grid(self, lx, ly, span_x, span_y, device):
        half_span_x = span_x / 2.0
        half_span_y = span_y / 2.0

        step_x = span_x / (lx - 1) if lx > 1 else 0
        step_y = span_y / (ly - 1) if ly > 1 else 0
        offset = step_x / 2.0

        grid_points = []
        for row_idx in range(ly):
            y = -half_span_y + row_idx * step_y if ly > 1 else 0.0
            x_offset = offset if (row_idx % 2 == 1) else 0.0

            for col_idx in range(lx):
                x = -half_span_x + col_idx * step_x + x_offset if lx > 1 else 0.0
                if abs(x) > half_span_x or abs(y) > half_span_y:
                    continue
                grid_points.append([x, y])

        return torch.tensor(grid_points, dtype=torch.float32, device=device)

    # =========================================================================
    # Multi-Resolution Self-Attention Helpers
    # =========================================================================

    def concatenate_latents_for_self_attn(self, latents_per_res, coords_per_res, global_latents):
        all_spatial = []
        all_coords = []
        split_sizes = []

        for res in sorted(latents_per_res.keys()):
            all_spatial.append(latents_per_res[res])
            all_coords.append(coords_per_res[res])
            split_sizes.append(latents_per_res[res].shape[1])

        latents_concat = torch.cat(all_spatial, dim=1)
        coords_concat = torch.cat(all_coords, dim=1)

        if global_latents is not None:
            latents_concat = torch.cat([latents_concat, global_latents], dim=1)

        return latents_concat, coords_concat, split_sizes

    def split_latents_after_self_attn(self, latents_concat, split_sizes, resolutions):
        total_spatial = sum(split_sizes)
        spatial_concat = latents_concat[:, :total_spatial]
        latents_list = torch.split(spatial_concat, split_sizes, dim=1)

        latents_per_res = {}
        for i, res in enumerate(resolutions):
            latents_per_res[res] = latents_list[i]

        global_latents = latents_concat[:, total_spatial:] \
            if latents_concat.shape[1] > total_spatial else None

        return latents_per_res, global_latents

    # =========================================================================
    # Geographic Pruning & Sampling
    # =========================================================================

    def _apply_pruning(self, tokens, mask, coords, grid_config, L_spatial):
        geo_tokens, geo_masks, _ = self.geo_pruning(
            tokens, mask, coords,
            geo_k=grid_config["geo_k"],
            sigma=grid_config["geo_sigma"],
            L_spatial=L_spatial,
            hexagonal=grid_config.get("hexagonal", False),
        )
        return geo_tokens, geo_masks

    def _sample_tokens(self, geo_tokens, geo_masks, grid_config, training=True):
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

    def _compute_deltas(self, sampled_tokens, coords):
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

    def _cross_attention_step(self, latents, sampled_tokens, sampled_masks,
                               coords, cross_attn, cross_ff, L_spatial):
        processed_tokens = self.input_processor.process_data_for_encoder(
            sampled_tokens, sampled_masks, latent_positions=coords
        )
        delta_x, delta_y, gsd = self._compute_deltas(sampled_tokens, coords)

        spatial = latents[:, :L_spatial]
        spatial = cross_attn(
            spatial, context=processed_tokens, mask=~sampled_masks,
            delta_x=delta_x, delta_y=delta_y, gsd=gsd,
        ) + spatial
        spatial = cross_ff(spatial) + spatial

        return torch.cat([spatial, latents[:, L_spatial:]], dim=1)

    def _self_attention_step_multiresolution(self, latents_per_res, coords_per_res,
                                              global_latents, self_attns):
        resolutions = sorted(latents_per_res.keys())

        latents_concat, coords_concat, split_sizes = self.concatenate_latents_for_self_attn(
            latents_per_res, coords_per_res, global_latents
        )
        total_spatial = sum(split_sizes)

        if self.use_hybrid_self_attention:
            hybrid_cache = self.hybrid_self_attn.compute_cache(coords_concat)
            latents_concat = self.hybrid_self_attn(
                latents_concat, hybrid_cache, num_spatial=total_spatial
            )
        elif self.use_rpe or self.use_gaussian_bias:
            px = coords_concat[..., 0]
            py = coords_concat[..., 1]

            for self_attn, self_ff in self_attns:
                if self.use_rpe:
                    latents_concat = self_attn(
                        latents_concat, pos_x=px, pos_y=py, num_spatial=total_spatial
                    ) + latents_concat
                else:
                    latents_concat = self_attn(
                        latents_concat, positions=coords_concat, num_spatial=total_spatial
                    ) + latents_concat
                latents_concat = self_ff(latents_concat) + latents_concat
        else:
            for self_attn, self_ff in self_attns:
                latents_concat = self_attn(latents_concat) + latents_concat
                latents_concat = self_ff(latents_concat) + latents_concat

        latents_per_res, global_latents = self.split_latents_after_self_attn(
            latents_concat, split_sizes, resolutions
        )
        return latents_per_res, global_latents

    # =========================================================================
    # Encode
    # =========================================================================

    def encode(self, groups, grid_configs, training=True, return_trajectory=False):
        first_group = next(iter(groups.values()))
        B = first_group["tokens"].shape[0]
        device = first_group["tokens"].device
        resolutions = sorted(groups.keys())

        latents_per_res, coords_per_res = self.init_latents_per_resolution(
            B, grid_configs, device
        )
        global_latents = self.get_global_latents(B)

        # Geographic pruning (once)
        geo_cache = {}
        for res in resolutions:
            tokens = groups[res]["tokens"]
            mask = groups[res]["mask"]
            gc = grid_configs[res]
            coords = coords_per_res[res]
            L_spatial = gc["L_spatial"]

            geo_tokens, geo_masks = self._apply_pruning(
                tokens, mask, coords, gc, L_spatial
            )
            geo_cache[res] = (geo_tokens, geo_masks, gc)

        trajectory = [coords_per_res.copy()] if return_trajectory else None

        # Layer loop
        for layer_idx in range(self.depth):
            cross_attn, cross_ff, self_attns = self.encoder_layers[layer_idx]

            for res in resolutions:
                geo_tokens, geo_masks, gc = geo_cache[res]
                coords = coords_per_res[res]
                L_spatial = gc["L_spatial"]

                sampled_tokens, sampled_masks = self._sample_tokens(
                    geo_tokens, geo_masks, gc, training
                )

                latents_per_res[res] = self._cross_attention_step(
                    latents_per_res[res], sampled_tokens, sampled_masks,
                    coords, cross_attn, cross_ff, L_spatial,
                )

            latents_per_res, global_latents = self._self_attention_step_multiresolution(
                latents_per_res, coords_per_res, global_latents, self_attns
            )

            if return_trajectory:
                trajectory.append(coords_per_res.copy())

        return EncoderOutput(
            latents_per_res=latents_per_res,
            coords_per_res=coords_per_res,
            trajectory=trajectory,
            global_latents=global_latents,
        )

    # =========================================================================
    # Decoder: Full-Context MLP + IDW + Grid Gate
    # =========================================================================

    def _compute_grid_spacing(self, coords):
        with torch.no_grad():
            c = coords[0]
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
            dists.fill_diagonal_(float('inf'))
            nn_dists = dists.min(dim=-1).values
            grid_spacing = nn_dists.median()
        return grid_spacing

    def _decode_single_grid(self, latents, coords, query_coords, query_gsd,
                             query_features, grid_spacing, k, task_embed,
                             training=True):
        """
        Decode from a single resolution grid.

        Args:
            latents: [B, L, D] encoder output for this grid
            coords: [B, L, 2] latent positions
            query_coords: [B, M, 2] query spatial positions
            query_gsd: scalar or [B, M] ground sampling distance
            query_features: [B, M, Q] processed query token features
            grid_spacing: scalar, median nearest-neighbor distance in grid
            k: number of nearest latents to use
            task_embed: [task_embed_dim] learned task vector
            training: enables latent dropping
        """
        B, M, _ = query_coords.shape
        D = latents.shape[-1]

        k_fetch = k + 1 if training else k
        k_fetch = min(k_fetch, coords.shape[1])
        k_keep = min(k, k_fetch)

        dists_sq = (
            query_coords.unsqueeze(2) - coords.unsqueeze(1)
        ).pow(2).sum(dim=-1)

        topk_dists_sq, topk_indices = torch.topk(
            dists_sq, k=k_fetch, dim=-1, largest=False,
        )

        # Latent dropping during training
        if training and k_fetch > k_keep:
            drop_idx = torch.randint(0, k_fetch, (B, M, 1), device=coords.device)
            keep_mask = torch.ones(B, M, k_fetch, dtype=torch.bool, device=coords.device)
            keep_mask.scatter_(2, drop_idx, False)
            topk_indices = topk_indices[keep_mask].reshape(B, M, k_keep)
            topk_dists_sq = topk_dists_sq[keep_mask].reshape(B, M, k_keep)

        # Gather latents and coords
        flat_idx = topk_indices.reshape(B, M * k_keep)

        selected_latents = torch.gather(
            latents, 1, flat_idx.unsqueeze(-1).expand(-1, -1, D),
        ).reshape(B, M, k_keep, D)

        selected_coords = torch.gather(
            coords, 1, flat_idx.unsqueeze(-1).expand(-1, -1, 2),
        ).reshape(B, M, k_keep, 2)

        # RPE
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)

        if isinstance(query_gsd, torch.Tensor) and query_gsd.dim() >= 2:
            compression_scale = self.input_processor.compression_alpha * query_gsd.unsqueeze(-1)
        else:
            compression_scale = self.input_processor.compression_alpha * query_gsd

        rel_pe = self.input_processor.pos_encoder(
            delta_x, delta_y, compression_scale=compression_scale,
        )

        # Expand query features and task embedding to match [B, M, k, ...]
        query_expanded = query_features.unsqueeze(2).expand(-1, -1, k_keep, -1)
        task_expanded = task_embed.expand(B, M, k_keep, -1)

        # Local prediction: MLP(latent, RPE, query, task)
        local_input = torch.cat(
            [selected_latents, rel_pe, query_expanded, task_expanded], dim=-1
        )
        local_preds = self.local_predictor(local_input)

        # IDW blend
        gs_sq = grid_spacing.pow(2).clamp(min=1e-8)
        dists_norm = topk_dists_sq / gs_sq
        temperature = self.decoder_temperature
        weights = F.softmax(-dists_norm * temperature, dim=-1)

        grid_feature = (weights.unsqueeze(-1) * local_preds).sum(dim=2)
        return grid_feature

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                     query_mask, target_resolution=None, task_name="reconstruction",
                     training=True, return_features=False):
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        # Process query tokens
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask, target_resolution=target_resolution,
        )
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)

        # Extract GSD
        if self.input_processor.use_constant_gsd:
            query_gsd = self.input_processor._constant_gsd
        else:
            query_gsd = self.input_processor.geometry.get_token_gsd(query_tokens)

        # Task embedding: [task_embed_dim] → will be broadcast in _decode_single_grid
        task_embed = self.get_task_embedding(task_name)

        # Decode from each resolution grid
        grid_features = []
        for res in sorted(latents_per_res.keys()):
            grid_spacing = self._compute_grid_spacing(coords_per_res[res])

            grid_feat = self._decode_single_grid(
                latents=latents_per_res[res],
                coords=coords_per_res[res],
                query_coords=query_coords,
                query_gsd=query_gsd,
                query_features=query_features,
                grid_spacing=grid_spacing,
                k=k,
                task_embed=task_embed,
                training=training,
            )
            grid_features.append(grid_feat)

        # Fuse across grids
        if len(grid_features) == 1:
            fused = grid_features[0]
        else:
            stacked = torch.stack(grid_features, dim=2)           # [B, M, G, D]
            scores = self.grid_gate(stacked).squeeze(-1)          # [B, M, G]
            weights = F.softmax(scores, dim=-1)                   # [B, M, G]
            fused = (weights.unsqueeze(-1) * stacked).sum(dim=2)  # [B, M, D]

        # ── Output head ──────────────────────────────────────────
        if return_features:
            return fused  # [B, M, D] — pre-head features for task-specific heads
        return self.reconstruction_head(fused)

    def classify(self, latents_per_res):
        all_latents = torch.cat(
            [latents_per_res[res] for res in sorted(latents_per_res.keys())], dim=1
        )
        return self.to_logits(all_latents)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, batch, training=True, task="reconstruction",
                return_trajectory=False, return_predicted_errors=False,
                return_features=False, tokens_per_latent_override=None):
        """
        Forward pass with multi-task support.

        Args:
            batch: Standard batch dict. Must contain "task" key (str).
            training: Training mode (enables pruning, dropout, etc.)
            task: "reconstruction" | "encoder" | "visualization"
            return_trajectory: Return latent trajectories per layer.
            return_predicted_errors: Return error predictions.
            return_features: If True, return pre-head features [B, M, D]
                            instead of final predictions. Used by multi-task
                            trainer to apply task-specific heads.
            tokens_per_latent_override: If set, overrides self.tokens_per_latent
                                        for this forward pass. Used to vary
                                        latent density during reconstruction
                                        training.

        Returns:
            If return_features=True:
                {"features": [B, M, D], "latents_per_res": ..., "coords_per_res": ...}
            Else:
                predictions tensor or dict (existing behavior).
        """
        groups = batch["groups"]
        queries = batch["queries"]
        queries_mask = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)
        task_name = batch.get("task", "reconstruction")

        # Variable latent density
        tpl = tokens_per_latent_override if tokens_per_latent_override is not None \
            else self.tokens_per_latent

        resolutions = sorted(groups.keys())
        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                tokens_per_latent=tpl,
                total_tokens=groups[res]["tokens"].shape[1],
                sigma_factor=self.sigma_factor,
                max_k=self.max_k,
            )
            for res in resolutions
        }

        need_trajectory = return_trajectory or task == "visualization"
        encoder_output = self.encode(
            groups=groups, grid_configs=grid_configs,
            training=training, return_trajectory=need_trajectory,
        )

        latents_per_res = encoder_output.latents_per_res
        coords_per_res = encoder_output.coords_per_res
        trajectory = encoder_output.trajectory

        if task == "encoder":
            result = {
                'latents_per_res': latents_per_res,
                'coords_per_res': coords_per_res,
                'trajectory': trajectory,
            }
            if return_predicted_errors:
                result['predicted_errors'] = None
            return result

        if task in ("reconstruction", "visualization"):
            chunk_size = 10000
            N = queries.shape[1]

            if N > chunk_size:
                preds = []
                for i in range(0, N, chunk_size):
                    preds.append(self.reconstruct(
                        latents_per_res, coords_per_res,
                        queries[:, i:i + chunk_size],
                        queries_mask[:, i:i + chunk_size],
                        target_resolution=target_resolution,
                        task_name=task_name,
                        training=training,
                        return_features=return_features,
                    ))
                output = torch.cat(preds, dim=1)
            else:
                output = self.reconstruct(
                    latents_per_res, coords_per_res,
                    queries, queries_mask,
                    target_resolution=target_resolution,
                    task_name=task_name,
                    training=training,
                    return_features=return_features,
                )

            if return_features:
                return {
                    "features": output,  # [B, M, D] pre-head
                    "latents_per_res": latents_per_res,
                    "coords_per_res": coords_per_res,
                }

            if task == "visualization" or return_predicted_errors:
                return {
                    'predictions': output,
                    'latents_per_res': latents_per_res,
                    'coords_per_res': coords_per_res,
                    'trajectory': trajectory,
                    'predicted_errors': None,
                }

            return output

        else:
            return self.classify(latents_per_res)

    # =========================================================================
    # Freeze/Unfreeze
    # =========================================================================

    def _set_requires_grad(self, module, flag):
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
        self._set_requires_grad(self.local_predictor, False)
        self._set_requires_grad(self.grid_gate, False)
        self._set_requires_grad(self.reconstruction_head, False)

    def unfreeze_decoder(self):
        self._set_requires_grad(self.local_predictor, True)
        self._set_requires_grad(self.grid_gate, True)
        self._set_requires_grad(self.reconstruction_head, True)

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