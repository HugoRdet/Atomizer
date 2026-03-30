"""
Atomiser Model — Multi-Resolution Encoder/Decoder with MAE Pretraining
=======================================================================

Changes from previous version:
1. MAE masking in encode():
   - mask_ratio argument (0.0 = disabled, 0.75 = MAE pretraining)
   - Split latents into visible / masked ONCE before the layer loop
   - Cross-attention: visible latents only (contiguous tensor, no indexing tricks)
   - Self-attention: ALL latents (visible + mask tokens as registers)
     → cat([visible, masked]) before self-attn, split back after
   - After loop: cat([visible, masked]) → full grid for decoder
   - Decoder is completely unaware of masking
   - geo_cache and masked_indices_per_res returned in EncoderOutput

2. Task embedding removed:
   - _init_task_embeddings, register_task, get_task_embedding gone
   - local_input_dim no longer includes task_embed_dim
   - _decode_single_grid and reconstruct signatures cleaned up

Architecture:
- Per-resolution latent grids (different sizes based on token count)
- Cross-attention: visible latents only (MAE), all latents (mask_ratio=0)
- Self-attention: all latents including mask tokens (register pattern)
- Decoder: predict-then-interpolate, unaware of masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from dataclasses import dataclass
from einops import repeat
from typing import Optional, Tuple, List, Dict

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
    latents_per_res: Dict[float, torch.Tensor]      # {res: [B, L, D]}
    coords_per_res:  Dict[float, torch.Tensor]      # {res: [B, L, 2]}
    trajectory:      Optional[List[Dict[float, torch.Tensor]]] = None
    global_latents:  Optional[torch.Tensor] = None
    # MAE fields — None when mask_ratio == 0.0
    geo_cache:               Optional[Dict] = None  # {res: (geo_tokens [B,L,k,8], geo_masks, gc)}
    masked_indices_per_res:  Optional[Dict[float, torch.Tensor]] = None  # {res: [n_mask]}


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
        self.input_dim       = self.input_processor.get_encoder_output_dim()
        self.query_dim_recon = self.input_processor.get_decoder_output_dim()
        self.latent_dim      = config["Atomiser"].get("latent_dim", self.input_dim)
        self.decoder_pe_dim  = self.input_processor.pos_encoder.get_output_dim()

        # =====================================================================
        # 3. LATENT GRID CONFIG
        # =====================================================================
        latent_cfg             = config.get("latent_grid", {})
        self.tokens_per_latent = latent_cfg.get("tokens_per_latent", 2000)
        self.sigma_factor      = latent_cfg.get("sigma_factor", 1.5)
        self.max_k             = latent_cfg.get("max_k", 2000)
        self.hexagonal         = latent_cfg.get("hexagonal", False)

        # =====================================================================
        # 4. GLOBAL LATENTS
        # =====================================================================
        self.num_global_latents = config["Atomiser"].get("global_latents", 0)

        # =====================================================================
        # 5. ARCHITECTURE PARAMETERS
        # =====================================================================
        self.depth               = config["Atomiser"]["depth"]
        self.cross_heads         = config["Atomiser"]["cross_heads"]
        self.latent_heads        = config["Atomiser"]["latent_heads"]
        self.cross_dim_head      = config["Atomiser"]["cross_dim_head"]
        self.latent_dim_head     = config["Atomiser"]["latent_dim_head"]
        self.attn_dropout        = config["Atomiser"]["attn_dropout"]
        self.ff_dropout          = config["Atomiser"]["ff_dropout"]
        self.weight_tie_layers   = config["Atomiser"]["weight_tie_layers"]
        self.self_per_cross_attn = config["Atomiser"]["self_per_cross_attn"]
        self.num_classes         = config["trainer"]["num_classes"]
        self.decoder_k_spatial   = config["Atomiser"].get("decoder_k_spatial", 4)

        # =====================================================================
        # 6. ROPE CONFIGURATION
        # =====================================================================
        self.encoder_use_rpe      = config["Atomiser"]["RPE"].get("encoder_use_rpe", False)
        self.decoder_use_rpe      = config["Atomiser"]["RPE"].get("decoder_use_rpe", False)
        self.use_rpe              = config["Atomiser"]["RPE"].get("selfattn_use_rpe", False)
        self.rope_learnable_scale = config["Atomiser"].get("rope_learnable_scale", True)

        # =====================================================================
        # 7. SELF-ATTENTION MODE
        # =====================================================================
        self.use_gaussian_bias         = config["Atomiser"].get("use_gaussian_bias", False)
        self.gaussian_sigma            = config["Atomiser"].get("gaussian_sigma", 9.0)
        self.learnable_sigma           = config["Atomiser"].get("learnable_sigma", True)
        self.use_hybrid_self_attention = config["Atomiser"].get("use_hybrid_self_attention", False)
        self.self_attn_k               = config["Atomiser"].get("self_attn_k", 64)

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
        """Initialize learnable latent vector and MAE mask token."""
        self.spatial_latent_content = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.spatial_latent_content, std=0.02, a=-2., b=2.)

        # Learned mask token — replaces content of masked latents before
        # the layer loop. Participates in self-attention as a register,
        # absorbing context from visible neighbours so the decoder can
        # interpolate across masked regions via IDW.
        self.mask_token = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-2., b=2.)

        if self.num_global_latents > 0:
            self.global_latents = nn.Parameter(
                torch.randn(self.num_global_latents, self.latent_dim)
            )
            nn.init.trunc_normal_(self.global_latents, std=0.02, a=-2., b=2.)
        else:
            self.register_buffer('global_latents', None)

    def _init_geographic_pruning(self):
        self.geo_pruning = GeographicPruning(
            geometry=self.input_processor.geometry,
        )

    def _init_encoder_layers(self):
        self_rope_compression_scale  = self.config["RoPE"].get("self_compression_scale", 50.0)
        cross_rope_compression_scale = self.config["RoPE"].get("cross_compression_scale", 10.0)
        rope_base                    = self.config["RoPE"].get("base", 10000.0)

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
            cache_key    = 0 if should_cache else layer_idx

            cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
            cross_ff   = get_cross_ff(_cache=should_cache,   key=f"cross_ff_{cache_key}")

            if self.use_hybrid_self_attention:
                self_attns = None
            else:
                self_attns = nn.ModuleList([])
                for sa_idx in range(self.self_per_cross_attn):
                    sa_key    = f"{cache_key}_{sa_idx}" if should_cache else f"{layer_idx}_{sa_idx}"
                    self_attn = get_latent_attn(_cache=should_cache, key=f"self_attn_{sa_key}")
                    self_ff   = get_latent_ff(_cache=should_cache,   key=f"self_ff_{sa_key}")
                    self_attns.append(nn.ModuleList([self_attn, self_ff]))

            self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _create_self_attention_factories(self, rope_base: float, compression_scale: float):
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

    def _init_decoder(self):
        """
        Upgraded decoder for supervised segmentation:
          1. Wider & deeper local predictor (3 layers, 2× hidden dim)
          2. Content-aware neighbor gating (replaces pure IDW)
          3. Post-fusion residual MLP before the reconstruction head
        """
        decoder_hidden  = self.latent_dim * 2
        local_input_dim = (
            self.latent_dim
            + self.decoder_pe_dim
            + self.query_dim_recon
        )

        # --- Option 1: wider & deeper per-neighbor MLP ---
        self.local_predictor = nn.Sequential(
            nn.Linear(local_input_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )

        # --- Option 3: content-aware neighbor gating ---
        # Input: per-neighbor prediction + RPE → scalar score per neighbor.
        # Combined with distance prior via additive logits before softmax.
        self.register_buffer('decoder_temperature', torch.tensor(2.0))
        self.neighbor_gate = nn.Sequential(
            nn.Linear(self.latent_dim + self.decoder_pe_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, 1),
        )

        # Multi-resolution grid gate (unchanged)
        self.grid_gate = nn.Linear(self.latent_dim, 1)

        # --- Option 2: post-fusion residual MLP ---
        self.post_fusion = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )

        # Final prediction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.num_classes),
        )

    def _init_classifier(self):
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
        coords_per_res  = {}
        for res in sorted(grid_configs.keys()):
            gc        = grid_configs[res]
            L_spatial = gc["L_spatial"]
            latents_per_res[res] = self.get_spatial_latents(batch_size, L_spatial)
            coords_per_res[res]  = self._compute_latent_grid(gc, batch_size, device)
        return latents_per_res, coords_per_res

    def _compute_latent_grid(self, grid_config, batch_size, device):
        lx        = grid_config["latents_x"]
        ly        = grid_config["latents_y"]
        span_x    = grid_config["span_x"]
        span_y    = grid_config["span_y"]
        hexagonal = grid_config.get("hexagonal", False)

        grid = (self._create_hexagonal_grid(lx, ly, span_x, span_y, device)
                if hexagonal else
                self._create_square_grid(lx, ly, span_x, span_y, device))

        grid_config["L_spatial"] = grid.shape[0]
        return grid.unsqueeze(0).expand(batch_size, -1, -1)

    def _create_square_grid(self, lx, ly, span_x, span_y, device):
        step_x  = span_x / lx
        step_y  = span_y / ly
        start_x = -span_x / 2.0 + step_x / 2.0
        end_x   =  span_x / 2.0 - step_x / 2.0
        start_y = -span_y / 2.0 + step_y / 2.0
        end_y   =  span_y / 2.0 - step_y / 2.0

        xs = (torch.linspace(start_x, end_x, lx, device=device)
              if lx > 1 else torch.zeros(1, device=device))
        ys = (torch.linspace(start_y, end_y, ly, device=device)
              if ly > 1 else torch.zeros(1, device=device))

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
            y        = -half_span_y + row_idx * step_y if ly > 1 else 0.0
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

    def concatenate_latents_for_self_attn(self, latents_per_res, coords_per_res,
                                           global_latents):
        all_spatial = []
        all_coords  = []
        split_sizes = []

        for res in sorted(latents_per_res.keys()):
            all_spatial.append(latents_per_res[res])
            all_coords.append(coords_per_res[res])
            split_sizes.append(latents_per_res[res].shape[1])

        latents_concat = torch.cat(all_spatial, dim=1)
        coords_concat  = torch.cat(all_coords,  dim=1)

        if global_latents is not None:
            latents_concat = torch.cat([latents_concat, global_latents], dim=1)

        return latents_concat, coords_concat, split_sizes

    def split_latents_after_self_attn(self, latents_concat, split_sizes, resolutions):
        total_spatial  = sum(split_sizes)
        spatial_concat = latents_concat[:, :total_spatial]
        latents_list   = torch.split(spatial_concat, split_sizes, dim=1)

        latents_per_res = {res: latents_list[i] for i, res in enumerate(resolutions)}
        global_latents  = (latents_concat[:, total_spatial:]
                           if latents_concat.shape[1] > total_spatial else None)
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
        token_x     = token_centers_lut[token_x_idx]
        token_y     = token_centers_lut[token_y_idx]

        delta_x = token_x - coords[:, :, 0:1]
        delta_y = token_y - coords[:, :, 1:2]

        gsd = None
        if hasattr(self.input_processor, 'get_gsd_lut'):
            gsd_lut = self.input_processor.get_gsd_lut()
            if gsd_lut is not None:
                band_idx = sampled_tokens[:, :, :, 0].long()
                gsd      = gsd_lut[band_idx]

        return delta_x, delta_y, gsd

    # =========================================================================
    # Attention Steps
    # =========================================================================

    def _cross_attention_step(self, latents, sampled_tokens, sampled_masks,
                            coords, cross_attn, cross_ff, L_spatial):
        """
        Local cross-attention for a single resolution.

        latents  [B, L, D]  — L_vis during MAE, L_all during standard encoding.
        coords   [B, L, 2]  — must match the latents passed in.
        L_spatial            — number of spatial latents (for global latent split).

        For latents in pure-padding regions where all tokens are masked,
        we force-unmask one token to prevent softmax over all -inf → NaN.
        That token is zero-valued (padding), so cross-attention produces
        a near-zero output and the residual preserves the latent's value.
        """
        processed_tokens = self.input_processor.process_data_for_encoder(
            sampled_tokens, sampled_masks, latent_positions=coords
        )
        delta_x, delta_y, gsd = self._compute_deltas(sampled_tokens, coords)

        spatial = latents[:, :L_spatial]

        # Prevent all-masked latents: force-unmask first token for latents
        # where every token is masked (padding region). This prevents
        # softmax(all -inf) → NaN. The unmasked token is zero-padded,
        # so its contribution is minimal.
        all_masked = sampled_masks.all(dim=-1, keepdim=True)  # [B, L, 1]
        if all_masked.any():
            sampled_masks = sampled_masks.clone()
            sampled_masks[:, :, 0] = sampled_masks[:, :, 0] & ~all_masked.squeeze(-1)

        attn_out = cross_attn(
            spatial, context=processed_tokens, mask=~sampled_masks,
            delta_x=delta_x, delta_y=delta_y, gsd=gsd,
        )
        # Safety net: catch any remaining NaN from numerical edge cases
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        spatial = attn_out + spatial
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
            hybrid_cache   = self.hybrid_self_attn.compute_cache(coords_concat)
            latents_concat = self.hybrid_self_attn(
                latents_concat, hybrid_cache, num_spatial=total_spatial
            )
        elif self.use_rpe or self.use_gaussian_bias:
            px = coords_concat[..., 0]
            py = coords_concat[..., 1]
            for self_attn, self_ff in self_attns:
                if self.use_rpe:
                    latents_concat = self_attn(
                        latents_concat, pos_x=px, pos_y=py,
                        num_spatial=total_spatial
                    ) + latents_concat
                else:
                    latents_concat = self_attn(
                        latents_concat, positions=coords_concat,
                        num_spatial=total_spatial
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

    def encode(self, groups, grid_configs, training=True,
               return_trajectory=False, mask_ratio: float = 0.0):
        """
        Encode input token groups into per-resolution latent representations.

        MAE masking (mask_ratio > 0.0)
        ──────────────────────────────
        Before the layer loop, split each resolution's latents into two
        contiguous tensors:

            visible_latents [B, L_vis, D]  normal init  — do cross-attention
            masked_latents  [B, L_msk, D]  mask_token   — skip cross-attention

        Layer loop
            Cross-attn : visible_latents only, with their matching geo_tokens
                         and coords. Completely contiguous — no indexing.
            Self-attn  : cat([visible, masked]) so mask tokens act as registers
                         and absorb context from visible neighbours.
                         Split back to visible / masked after self-attn.

        After loop
            Rebuild full grid: cat([visible, masked])
            Decoder is distance-based (IDW) and masking-unaware — ordering
            within the grid doesn't matter.

        Mask indices shared across the batch (same spatial mask per step).

        Args:
            mask_ratio: 0.0 = standard, 0.75 = MAE pretraining.
        """
        first_group = next(iter(groups.values()))
        B           = first_group["tokens"].shape[0]
        device      = first_group["tokens"].device
        resolutions = sorted(groups.keys())

        latents_per_res, coords_per_res = self.init_latents_per_resolution(
            B, grid_configs, device
        )
        global_latents = self.get_global_latents(B)

        # ── Geographic pruning (once, outside gradient tracking) ──────────
        geo_cache = {}
        for res in resolutions:
            tokens    = groups[res]["tokens"]
            mask      = groups[res]["mask"]
            gc        = grid_configs[res]
            coords    = coords_per_res[res]
            L_spatial = gc["L_spatial"]

            geo_tokens, geo_masks = self._apply_pruning(
                tokens, mask, coords, gc, L_spatial
            )
            geo_cache[res] = (geo_tokens, geo_masks, gc)

        # ── MAE: split once before layer loop ─────────────────────────────
        mae_active = mask_ratio > 0.0

        # Per-resolution split state carried through the loop
        vis_latents = {}   # {res: [B, L_vis, D]}
        vis_coords  = {}   # {res: [B, L_vis, 2]}
        msk_latents = {}   # {res: [B, L_msk, D]}
        msk_coords  = {}   # {res: [B, L_msk, 2]}
        masked_indices_per_res = {}

        # geo_cache slices for visible latents — precomputed once
        vis_geo_tokens = {}  # {res: [B, L_vis, k, 8]}
        vis_geo_masks  = {}  # {res: [B, L_vis, k]}

        if mae_active:
            mask_token_vec = self.mask_token.view(1, 1, -1)  # [1, 1, D]

            for res in resolutions:
                L      = latents_per_res[res].shape[1]
                n_mask = min(int(mask_ratio * L), L - 1)  # keep ≥ 1 visible

                # Same mask for all samples in the batch
                perm        = torch.randperm(L, device=device)
                mask_idx    = perm[:n_mask]    # [n_mask]
                visible_idx = perm[n_mask:]    # [L_vis]

                masked_indices_per_res[res] = mask_idx

                # Visible — normal latent init
                vis_latents[res] = latents_per_res[res][:, visible_idx]   # [B, L_vis, D]
                vis_coords[res]  = coords_per_res[res][:, visible_idx]    # [B, L_vis, 2]

                # Masked — learned mask token, no gradient from tokens
                msk_latents[res] = mask_token_vec.expand(B, n_mask, -1).clone()
                msk_coords[res]  = coords_per_res[res][:, mask_idx]       # [B, n_mask, 2]

                # Precompute geo_cache slices for visible latents
                gt, gm, _ = geo_cache[res]
                vis_geo_tokens[res] = gt[:, visible_idx]   # [B, L_vis, k, 8]
                vis_geo_masks[res]  = gm[:, visible_idx]   # [B, L_vis, k]

        trajectory = [coords_per_res.copy()] if return_trajectory else None

        # ── Layer loop ────────────────────────────────────────────────────
        for layer_idx in range(self.depth):
            cross_attn, cross_ff, self_attns = self.encoder_layers[layer_idx]

            # ── Cross-attention ──────────────────────────────────────────
            for res in resolutions:
                _, _, gc = geo_cache[res]

                if mae_active:
                    # Visible latents only — contiguous, no fancy indexing
                    sampled_tokens, sampled_masks = self._sample_tokens(
                        vis_geo_tokens[res], vis_geo_masks[res], gc, training
                    )
                    vis_latents[res] = self._cross_attention_step(
                        vis_latents[res], sampled_tokens, sampled_masks,
                        vis_coords[res], cross_attn, cross_ff,
                        L_spatial=vis_latents[res].shape[1],
                    )
                else:
                    # Standard — all latents
                    gt, gm, _ = geo_cache[res]
                    L_spatial  = gc["L_spatial"]
                    sampled_tokens, sampled_masks = self._sample_tokens(
                        gt, gm, gc, training
                    )
                    latents_per_res[res] = self._cross_attention_step(
                        latents_per_res[res], sampled_tokens, sampled_masks,
                        coords_per_res[res], cross_attn, cross_ff, L_spatial,
                    )

            # ── Self-attention ───────────────────────────────────────────
            if mae_active:
                # Assemble full grid per resolution: [visible | masked]
                # Self-attention sees all latents — mask tokens as registers.
                full_latents_per_res = {}
                full_coords_per_res  = {}
                L_vis_per_res        = {}

                for res in resolutions:
                    full_latents_per_res[res] = torch.cat(
                        [vis_latents[res], msk_latents[res]], dim=1
                    )
                    full_coords_per_res[res] = torch.cat(
                        [vis_coords[res], msk_coords[res]], dim=1
                    )
                    L_vis_per_res[res] = vis_latents[res].shape[1]

                full_latents_per_res, global_latents = \
                    self._self_attention_step_multiresolution(
                        full_latents_per_res, full_coords_per_res,
                        global_latents, self_attns,
                    )

                # Split back: first L_vis = visible, rest = masked
                for res in resolutions:
                    L_vis            = L_vis_per_res[res]
                    vis_latents[res] = full_latents_per_res[res][:, :L_vis]
                    msk_latents[res] = full_latents_per_res[res][:, L_vis:]

            else:
                latents_per_res, global_latents = \
                    self._self_attention_step_multiresolution(
                        latents_per_res, coords_per_res,
                        global_latents, self_attns,
                    )

            if return_trajectory:
                trajectory.append(coords_per_res.copy())

        # ── Rebuild full grid for decoder ──────────────────────────────────
        # cat([visible, masked]) per resolution.
        # Decoder queries by distance → ordering doesn't matter.
        if mae_active:
            for res in resolutions:
                latents_per_res[res] = torch.cat(
                    [vis_latents[res], msk_latents[res]], dim=1
                )
                coords_per_res[res] = torch.cat(
                    [vis_coords[res], msk_coords[res]], dim=1
                )

        return EncoderOutput(
            latents_per_res=latents_per_res,
            coords_per_res=coords_per_res,
            trajectory=trajectory,
            global_latents=global_latents,
            geo_cache=geo_cache if mae_active else None,
            masked_indices_per_res=masked_indices_per_res if mae_active else None,
        )

    # =========================================================================
    # Decoder: Full-Context MLP + IDW + Grid Gate
    # =========================================================================

    def _compute_grid_spacing(self, coords):
        with torch.no_grad():
            c     = coords[0]
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
            dists.fill_diagonal_(float('inf'))
            nn_dists     = dists.min(dim=-1).values
            grid_spacing = nn_dists.median()
        return grid_spacing

    def _decode_single_grid(self, latents, coords, query_coords, query_gsd,
                             query_features, grid_spacing, k, training=True):
        """
        Decode from a single resolution grid.

        latents [B, L, D] — full grid including mask-token latents.
        Decoder is distance-based and masking-unaware.
        """
        B, M, _ = query_coords.shape
        D       = latents.shape[-1]

        k_fetch = k + 1 if training else k
        k_fetch = min(k_fetch, coords.shape[1])
        k_keep  = min(k, k_fetch)

        dists_sq = (
            query_coords.unsqueeze(2) - coords.unsqueeze(1)
        ).pow(2).sum(dim=-1)

        topk_dists_sq, topk_indices = torch.topk(
            dists_sq, k=k_fetch, dim=-1, largest=False,
        )

        # Latent dropping during training (prevents shortcut learning)
        if training and k_fetch > k_keep:
            drop_idx  = torch.randint(0, k_fetch, (B, M, 1), device=coords.device)
            keep_mask = torch.ones(B, M, k_fetch, dtype=torch.bool, device=coords.device)
            keep_mask.scatter_(2, drop_idx, False)
            topk_indices  = topk_indices[keep_mask].reshape(B, M, k_keep)
            topk_dists_sq = topk_dists_sq[keep_mask].reshape(B, M, k_keep)

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

        B_dec, M_dec, K_dec = delta_x.shape
        delta_x_flat = delta_x.reshape(B_dec, M_dec * K_dec)
        delta_y_flat = delta_y.reshape(B_dec, M_dec * K_dec)

        if isinstance(query_gsd, torch.Tensor) and query_gsd.dim() >= 2:
            compression_scale = (self.input_processor.compression_alpha
                                 * query_gsd.unsqueeze(-1))
            compression_scale = compression_scale.reshape(B_dec, M_dec * K_dec)
        else:
            compression_scale = self.input_processor.compression_alpha * query_gsd

        rel_pe = self.input_processor.pos_encoder(
            delta_x_flat, delta_y_flat, compression_scale=compression_scale,
        )
        rel_pe = rel_pe.reshape(B_dec, M_dec, K_dec, -1)

        query_expanded = query_features.unsqueeze(2).expand(-1, -1, k_keep, -1)

        local_input  = torch.cat([selected_latents, rel_pe, query_expanded], dim=-1)
        local_preds  = self.local_predictor(local_input)   # [B, M, k, latent_dim]

        # Content-aware neighbor gating (Option 3)
        # Learned gate scores from prediction content + relative position,
        # combined additively with distance prior so the model can override
        # pure distance weighting at class boundaries.
        gate_input    = torch.cat([local_preds, rel_pe], dim=-1)  # [B, M, k, latent_dim + pe_dim]
        content_score = self.neighbor_gate(gate_input).squeeze(-1)  # [B, M, k]

        gs_sq      = grid_spacing.pow(2).clamp(min=1e-8)
        dists_norm = topk_dists_sq / gs_sq
        weights    = F.softmax(
            content_score - dists_norm * self.decoder_temperature, dim=-1,
        )  # [B, M, k]

        grid_feature = (weights.unsqueeze(-1) * local_preds).sum(dim=2)
        return grid_feature

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                     query_mask, target_resolution=None,
                     training=True, return_features=False):
        """
        Decode: latents + query positions → predictions.

        Completely unaware of MAE masking. latents_per_res contains the
        full grid: visible encoded latents + mask-token registers.

        Returns:
            return_features=False : [B, M, num_classes]
            return_features=True  : [B, M, D]  pre-head features
        """
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask, target_resolution=target_resolution,
        )
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)

        if self.input_processor.use_constant_gsd:
            query_gsd = self.input_processor._constant_gsd
        else:
            query_gsd = self.input_processor.geometry.get_token_gsd(query_tokens)

        grid_features = []
        for res in sorted(latents_per_res.keys()):
            grid_spacing = self._compute_grid_spacing(coords_per_res[res])
            grid_feat    = self._decode_single_grid(
                latents=latents_per_res[res],
                coords=coords_per_res[res],
                query_coords=query_coords,
                query_gsd=query_gsd,
                query_features=query_features,
                grid_spacing=grid_spacing,
                k=k,
                training=training,
            )
            grid_features.append(grid_feat)

        # Fuse across grids (DDP-safe: no conditional branching)
        stacked = torch.stack(grid_features, dim=2)         # [B, M, G, D]
        scores  = self.grid_gate(stacked).squeeze(-1)       # [B, M, G]
        weights = F.softmax(scores, dim=-1)                 # [B, M, G]
        fused   = (weights.unsqueeze(-1) * stacked).sum(dim=2)  # [B, M, D]

        # Post-fusion residual refinement (Option 2)
        fused = self.post_fusion(fused) + fused

        if return_features:
            return fused

        return self.reconstruction_head(fused)              # [B, M, num_classes]

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
                return_features=False, tokens_per_latent_override=None,
                mask_ratio: float = 0.0):

        groups            = batch["groups"]
        queries           = batch["queries"]
        queries_mask      = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)

        tpl = (tokens_per_latent_override if tokens_per_latent_override is not None
               else self.tokens_per_latent)

        resolutions  = sorted(groups.keys())
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
        encoder_output  = self.encode(
            groups=groups, grid_configs=grid_configs,
            training=training, return_trajectory=need_trajectory,
            mask_ratio=mask_ratio,
        )

        latents_per_res = encoder_output.latents_per_res
        coords_per_res  = encoder_output.coords_per_res
        trajectory      = encoder_output.trajectory

        if task == "encoder":
            return {
                'latents_per_res': latents_per_res,
                'coords_per_res':  coords_per_res,
                'trajectory':      trajectory,
                'encoder_output':  encoder_output,
            }

        if task in ("reconstruction", "visualization"):
            chunk_size = 10_000
            N = queries.shape[1]

            if N > chunk_size:
                preds = []
                for i in range(0, N, chunk_size):
                    preds.append(self.reconstruct(
                        latents_per_res, coords_per_res,
                        queries[:, i:i + chunk_size],
                        queries_mask[:, i:i + chunk_size],
                        target_resolution=target_resolution,
                        training=training,
                        return_features=return_features,
                    ))
                output = torch.cat(preds, dim=1)
            else:
                output = self.reconstruct(
                    latents_per_res, coords_per_res,
                    queries, queries_mask,
                    target_resolution=target_resolution,
                    training=training,
                    return_features=return_features,
                )

            if return_features:
                return {
                    "features":        output,
                    "latents_per_res": latents_per_res,
                    "coords_per_res":  coords_per_res,
                    "encoder_output":  encoder_output,
                }

            if task == "visualization" or return_predicted_errors:
                return {
                    'predictions':      output,
                    'latents_per_res':  latents_per_res,
                    'coords_per_res':   coords_per_res,
                    'trajectory':       trajectory,
                    'predicted_errors': None,
                }

            return output

        else:
            return self.classify(latents_per_res)

    # =========================================================================
    # Freeze / Unfreeze
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
        self.mask_token.requires_grad = False
        if self.global_latents is not None:
            self.global_latents.requires_grad = False
        self._set_requires_grad(self.input_processor, False)

    def unfreeze_encoder(self):
        self._set_requires_grad(self.encoder_layers, True)
        self.spatial_latent_content.requires_grad = True
        self.mask_token.requires_grad = True
        if self.global_latents is not None:
            self.global_latents.requires_grad = True
        self._set_requires_grad(self.input_processor, True)

    def freeze_decoder(self):
        self._set_requires_grad(self.local_predictor, False)
        self._set_requires_grad(self.neighbor_gate, False)
        self._set_requires_grad(self.grid_gate, False)
        self._set_requires_grad(self.post_fusion, False)
        self._set_requires_grad(self.reconstruction_head, False)

    def unfreeze_decoder(self):
        self._set_requires_grad(self.local_predictor, True)
        self._set_requires_grad(self.neighbor_gate, True)
        self._set_requires_grad(self.grid_gate, True)
        self._set_requires_grad(self.post_fusion, True)
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