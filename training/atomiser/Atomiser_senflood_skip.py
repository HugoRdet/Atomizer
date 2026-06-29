"""
Atomiser Model (SKIP variant) — Multi-Resolution Encoder/Decoder
================================================================

Identical to Atomiser_Senflood EXCEPT for a decoder PIXEL-SKIP CASCADE,
gated behind config["Atomiser"]["use_decoder_skip"].

Cascade (when enabled):
    learned pixel_query
        -> [pixel cross-attention over the query-pixel's own encoded band-tokens]
        -> enriched per-pixel query
        -> [existing geographic cross-attention over k-nearest latents]
        -> head

The pixel cross-attention has NO relative position encoding: token and query
share the same pixel, so displacement is zero and the positional block in
process_data_for_encoder collapses to a constant. Tokens are distinguished
purely by spectral / reflectance / resolution / time.

All additions are tagged  # >>> SKIP  for easy diffing against the original.

Config additions:
  Atomiser:
    use_decoder_skip: true
    decoder_skip_drop_p: 0.5   # aggressive train-only random drop on the
                               # pixel's own band-tokens (can go down to 1)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import wraps
from dataclasses import dataclass
from einops import repeat
from typing import Optional, Tuple, List, Dict
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from training.utils.token_building.processor import TokenProcessor
from training.utils.datasets.token_grouping import compute_grid_config

from .nn_comp import (
    PreNorm,
    FeedForward,
    LatentAttentionPooling,
)

from .RPE import (
    LocalCrossAttentionRoPE,
    SelfAttentionRoPE,
    PreNormRoPE,
)

from .geographic_pruning import GeographicPruning
from .error_supervision import compute_latent_errors, compute_error_predictor_loss


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
# ENCODER OUTPUT
# =============================================================================

@dataclass
class EncoderOutput:
    """Structured output from encoder with per-resolution latents."""
    latents_per_res: Dict[float, torch.Tensor]
    coords_per_res:  Dict[float, torch.Tensor]
    trajectory:      Optional[List[Dict[float, torch.Tensor]]] = None
    global_latents:  Optional[torch.Tensor] = None
    geo_cache:               Optional[Dict] = None
    masked_indices_per_res:  Optional[Dict[float, torch.Tensor]] = None


# =============================================================================
# MAIN ATOMISER CLASS (SKIP VARIANT)
# =============================================================================

class Atomiser_Senflood_Skip(pl.LightningModule):

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
        # 3. LATENT GRID + STOCHASTIC TOKEN SAMPLING
        # =====================================================================
        latent_cfg        = config.get("latent_grid", {})
        self.sigma_factor = latent_cfg.get("sigma_factor", 1.5)
        self.hexagonal    = latent_cfg.get("hexagonal", False)

        default_sampling = [[8192, 1024]]
        self.train_sampling = [
            tuple(p) for p in latent_cfg.get("train_sampling", default_sampling)
        ]
        self.val_sampling = [
            tuple(p) for p in latent_cfg.get("val_sampling", default_sampling)
        ]
        self.max_k = max(p[0] for p in self.train_sampling + self.val_sampling)

        print(f"[Atomiser] Train sampling: {self.train_sampling}")
        print(f"[Atomiser] Val sampling:   {self.val_sampling}")

        # =====================================================================
        # 4. GLOBAL LATENTS
        # Global latents participate in self-attention for global context
        # aggregation. They are intentionally not wired into the decoder —
        # the decoder uses only spatial latents for local prediction.
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
        self.gradient_checkpointing = config["Atomiser"].get("gradient_checkpointing", True)

        # =====================================================================
        # 6. ROPE CONFIGURATION
        # =====================================================================
        self.encoder_use_rpe      = config["Atomiser"]["RPE"].get("encoder_use_rpe", False)
        self.use_rpe              = config["Atomiser"]["RPE"].get("selfattn_use_rpe", False)
        self.rope_learnable_scale = config["Atomiser"].get("rope_learnable_scale", True)

        if self.gradient_checkpointing:
            print("[Atomiser] Gradient checkpointing: ENABLED")

        # =====================================================================
        # 7. ERROR PREDICTOR
        # =====================================================================
        self.use_error_predictor = config["Atomiser"].get("use_error_predictor", False)
        if self.use_error_predictor:
            self.lambda_error = float(config["Atomiser"].get("lambda_error", 0.1))
            print(f"[Atomiser] Error predictor ENABLED (lambda={self.lambda_error})")
        else:
            print(f"[Atomiser] Error predictor DISABLED")

        # =====================================================================
        # 7c. TARGETED DEPTH-2
        # Second cross-attention + self-attention pass on top-k high-error
        # latents only. Uses shared encoder weights (zero new parameters).
        # Much cheaper than full depth=2: only 2.5% of latents re-encoded.
        # =====================================================================
        self.use_targeted_depth2 = config["Atomiser"].get(
            "use_targeted_depth2", False)
        if self.use_targeted_depth2:
            td2 = config["Atomiser"].get("targeted_depth2", {})
            self.td2_k          = int(td2.get("k", 50))
            self.td2_cross_k    = int(td2.get("cross_k", 2000))  # full geo pool
            self.td2_self_attn  = int(td2.get("self_attn", 2))
            print(f"[Atomiser] Targeted depth-2 ENABLED: "
                  f"k={self.td2_k}, cross_k={self.td2_cross_k}, "
                  f"self_attn={self.td2_self_attn}")
        else:
            print(f"[Atomiser] Targeted depth-2 DISABLED")

        # =====================================================================
        # 7b. REFINEMENT
        # =====================================================================
        self.use_refinement = config["Atomiser"].get("use_refinement", False)
        if self.use_refinement:
            ref_cfg = config["Atomiser"].get("refinement", {})
            self.k_refine            = int(ref_cfg.get("k_refine", 50))
            self.refine_grid_size    = int(ref_cfg.get("grid_size", 2))   # 2→2×2=4 latents
            self.refine_tpl          = int(ref_cfg.get("tokens_per_latent", 500))
            self.refine_cross_k      = int(ref_cfg.get("cross_k", 500))
            self.refine_self_local   = int(ref_cfg.get("self_attn_local", 2))
            self.refine_self_global  = int(ref_cfg.get("self_attn_global", 2))
            # grid offset: half the typical latent spacing fraction
            self.refine_offset_factor = float(ref_cfg.get("offset_factor", 0.25))
            print(f"[Atomiser] Refinement ENABLED: k={self.k_refine}, "
                  f"grid={self.refine_grid_size}×{self.refine_grid_size}, "
                  f"tpl={self.refine_tpl}, cross_k={self.refine_cross_k}")

            refine_cs = float(ref_cfg.get(
                "self_attn_compression_scale",
                1250.0,
            ))
            rope_base = config["RoPE"].get("base", 100.0)

            self.refine_self_attn = PreNormRoPE(
                self.latent_dim,
                SelfAttentionRoPE(
                    dim=self.latent_dim,
                    heads=self.latent_heads,
                    dim_head=self.latent_dim_head,
                    dropout=self.attn_dropout,
                    use_rope=True,
                    rope_base=rope_base,
                    rope_compression_scale=refine_cs,
                    rope_learnable_scale=self.rope_learnable_scale,
                )
            )
            self.refine_self_ff = PreNorm(
                self.latent_dim,
                FeedForward(self.latent_dim, dropout=self.ff_dropout),
            )
            print(f"[Atomiser] Refinement local SA compression scale: {refine_cs:.1f}m")

            cross_rope_cs = self.config["RoPE"].get("cross_compression_scale", 50.0)
            self.refine_cross_attn = PreNorm(
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
                    rope_compression_scale=cross_rope_cs,
                    rope_learnable_scale=self.rope_learnable_scale,
                )
            )
            self.refine_cross_ff = PreNorm(
                self.latent_dim,
                FeedForward(self.latent_dim, dropout=self.ff_dropout),
            )
            print(f"[Atomiser] Refinement dedicated cross-attention ENABLED")
        else:
            print(f"[Atomiser] Refinement DISABLED")

        # =====================================================================
        # 8. INITIALIZE COMPONENTS
        # =====================================================================
        self._init_latents()
        self._init_geographic_pruning()
        self._init_encoder_layers()
        self._init_decoder()
        self._init_classifier()

    # =========================================================================
    # Sampling Configuration
    # =========================================================================

    def sample_config(self, training: bool = True):
        if training:
            return random.choice(self.train_sampling)
        else:
            return random.choice(self.val_sampling)

    @property
    def tokens_per_latent(self):
        return self.train_sampling[0][0]

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_latents(self):
        self.spatial_latent_content = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.spatial_latent_content, std=0.02, a=-2., b=2.)

        self.mask_token = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-2., b=2.)

        self.refinement_latent_content = nn.Parameter(torch.randn(self.latent_dim))
        nn.init.trunc_normal_(self.refinement_latent_content, std=0.02, a=-2., b=2.)

        if self.num_global_latents > 0:
            self.global_latents = nn.Parameter(
                torch.randn(self.num_global_latents, self.latent_dim))
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
        rope_base                    = self.config["RoPE"].get("base", 100.0)

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
        get_latent_attn = cache_fn(lambda: PreNormRoPE(
            self.latent_dim,
            SelfAttentionRoPE(
                dim=self.latent_dim, heads=self.latent_heads,
                dim_head=self.latent_dim_head, dropout=self.attn_dropout,
                use_rope=True, rope_base=rope_base,
                rope_compression_scale=self_rope_compression_scale,
                rope_learnable_scale=self.rope_learnable_scale,
            )
        ))
        get_latent_ff = cache_fn(lambda: PreNorm(
            self.latent_dim,
            FeedForward(self.latent_dim, dropout=self.ff_dropout)
        ))

        self.encoder_layers = nn.ModuleList([])
        for layer_idx in range(self.depth):
            should_cache = self.weight_tie_layers and layer_idx > 0
            cache_key    = 0 if should_cache else layer_idx

            cross_attn = get_cross_attn(_cache=should_cache, key=f"cross_attn_{cache_key}")
            cross_ff   = get_cross_ff(_cache=should_cache,   key=f"cross_ff_{cache_key}")

            self_attns = nn.ModuleList([])
            for sa_idx in range(self.self_per_cross_attn):
                sa_key    = f"{cache_key}_{sa_idx}" if should_cache else f"{layer_idx}_{sa_idx}"
                self_attn = get_latent_attn(_cache=should_cache, key=f"self_attn_{sa_key}")
                self_ff   = get_latent_ff(_cache=should_cache,   key=f"self_ff_{sa_key}")
                self_attns.append(nn.ModuleList([self_attn, self_ff]))

            self.encoder_layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

    def _init_decoder(self):
        """
        Pooled decoder: global learned query + context with rel_pe + random drop.
        (See original Atomiser_Senflood for full design notes.)
        """
        # ── Context dimension = latent_dim + rel_pe_dim ───────────────
        self.decoder_context_dim = self.latent_dim + self.decoder_pe_dim

        # ── Global learned query ──────────────────────────────────────
        self.global_query = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        nn.init.trunc_normal_(self.global_query, std=0.02, a=-2., b=2.)

        # ── Cross-attention (Q dim != K/V dim) ────────────────────────
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            kdim=self.decoder_context_dim,
            vdim=self.decoder_context_dim,
            num_heads=self.cross_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        self.dec_q_norm = nn.LayerNorm(self.latent_dim)
        self.dec_ctx_norm = nn.LayerNorm(self.decoder_context_dim)

        # ── Bernoulli drop probability ────────────────────────────────
        self.decoder_drop_p = self.config["Atomiser"].get("decoder_drop_p", 0.25)
        print(f"[Atomiser] Pooled decoder: k={self.decoder_k_spatial}, "
              f"drop_p={self.decoder_drop_p}")

        # ── Final segmentation head ───────────────────────────────────
        hidden_dim = self.latent_dim * 2
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

        # ── Error predictor ───────────────────────────────────────────
        if self.use_error_predictor:
            self.error_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 4),
                nn.GELU(),
                nn.Linear(self.latent_dim // 4, 1),
                nn.Softplus(),
            )

        # >>> SKIP ─────────────────────────────────────────────────────
        # Decoder pixel-skip cascade: a learned query first attends to the
        # query-pixel's own encoded band-tokens, producing an enriched query
        # that then drives the existing latent cross-attention.
        self.use_decoder_skip = self.config["Atomiser"].get("use_decoder_skip", False)
        if self.use_decoder_skip:
            self.decoder_skip_drop_p = float(
                self.config["Atomiser"].get("decoder_skip_drop_p", 0.5))

            # Learned query for the pixel-CA (shared; K/V differ per pixel).
            self.pixel_query = nn.Parameter(torch.randn(1, 1, self.latent_dim))
            nn.init.trunc_normal_(self.pixel_query, std=0.02, a=-2., b=2.)

            # Pixel cross-attention: Q=latent_dim, K/V = encoded band-tokens
            # (dim = input_dim == encoder_output_dim).
            self.pixel_cross_attn = nn.MultiheadAttention(
                embed_dim=self.latent_dim,
                kdim=self.input_dim,
                vdim=self.input_dim,
                num_heads=self.cross_heads,
                dropout=self.attn_dropout,
                batch_first=True,
            )

            self.pixel_q_norm = nn.LayerNorm(self.latent_dim)
            print(f"[Atomiser] Decoder pixel-skip: ENABLED "
                  f"(drop_p={self.decoder_skip_drop_p}, K/V dim={self.input_dim})")
        else:
            print("[Atomiser] Decoder pixel-skip: DISABLED")
        # >>> END SKIP ─────────────────────────────────────────────────

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

    def get_spatial_latents(self, batch_size, L_spatial):
        return repeat(self.spatial_latent_content, 'd -> b n d', b=batch_size, n=L_spatial)

    def get_global_latents(self, batch_size):
        if self.global_latents is None:
            return None
        return repeat(self.global_latents, 'n d -> b n d', b=batch_size)

    def init_latents_per_resolution(self, batch_size, grid_configs, device):
        latents_per_res = {}
        coords_per_res  = {}
        for res in sorted(grid_configs.keys()):
            gc        = grid_configs[res]
            L_spatial = gc["L_spatial"]
            latents_per_res[res] = self.get_spatial_latents(batch_size, L_spatial)
            coords_per_res[res]  = self._compute_latent_grid(gc, batch_size, device)
        return latents_per_res, coords_per_res

    def _compute_latent_grid(self, grid_config, batch_size, device):
        lx   = grid_config["latents_x"]
        ly   = grid_config["latents_y"]
        sx   = grid_config["span_x"]
        sy   = grid_config["span_y"]
        hexa = grid_config.get("hexagonal", False)

        grid = (self._create_hexagonal_grid(lx, ly, sx, sy, device) if hexa
                else self._create_square_grid(lx, ly, sx, sy, device))

        grid_config["L_spatial"] = grid.shape[0]
        return grid.unsqueeze(0).expand(batch_size, -1, -1)

    def _create_square_grid(self, lx, ly, span_x, span_y, device):
        step_x = span_x / lx
        step_y = span_y / ly
        xs = (torch.linspace(-span_x/2 + step_x/2, span_x/2 - step_x/2, lx, device=device)
              if lx > 1 else torch.zeros(1, device=device))
        ys = (torch.linspace(-span_y/2 + step_y/2, span_y/2 - step_y/2, ly, device=device)
              if ly > 1 else torch.zeros(1, device=device))
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx.flatten(), gy.flatten()], dim=-1)

    def _create_hexagonal_grid(self, lx, ly, span_x, span_y, device):
        hx = span_x / 2.0
        hy = span_y / 2.0
        sx = span_x / (lx - 1) if lx > 1 else 0
        sy = span_y / (ly - 1) if ly > 1 else 0
        offset = sx / 2.0
        pts = []
        for r in range(ly):
            y = -hy + r * sy if ly > 1 else 0.0
            xo = offset if (r % 2 == 1) else 0.0
            for c in range(lx):
                x = -hx + c * sx + xo if lx > 1 else 0.0
                if abs(x) > hx or abs(y) > hy:
                    continue
                pts.append([x, y])
        return torch.tensor(pts, dtype=torch.float32, device=device)

    # =========================================================================
    # Multi-Resolution Self-Attention Helpers
    # =========================================================================

    def concatenate_latents_for_self_attn(self, latents_per_res, coords_per_res,
                                           global_latents):
        all_spatial = []
        all_coords  = []
        split_sizes = []
        for res in sorted(latents_per_res.keys(), key=str):
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

    def _sample_tokens(self, geo_tokens, geo_masks, cross_k):
        k = geo_tokens.shape[2]
        m = min(cross_k, k)
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
                gsd = gsd_lut[sampled_tokens[:, :, :, 0].long()]
        return delta_x, delta_y, gsd

    # =========================================================================
    # Attention Steps
    # =========================================================================

    def _cross_attention_step(self, latents, sampled_tokens, sampled_masks,
                              coords, cross_attn, cross_ff, L_spatial):
        processed_tokens = self.input_processor.process_data_for_encoder(
            sampled_tokens, sampled_masks, latent_positions=coords)
        delta_x, delta_y, gsd = self._compute_deltas(sampled_tokens, coords)

        spatial = latents[:, :L_spatial]

        all_masked = sampled_masks.all(dim=-1, keepdim=True)
        if all_masked.any():
            sampled_masks = sampled_masks.clone()
            sampled_masks[:, :, 0] = sampled_masks[:, :, 0] & ~all_masked.squeeze(-1)

        attn_out = cross_attn(
            spatial, context=processed_tokens, mask=~sampled_masks,
            delta_x=delta_x, delta_y=delta_y, gsd=gsd)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        spatial = attn_out + spatial
        spatial = cross_ff(spatial) + spatial
        return torch.cat([spatial, latents[:, L_spatial:]], dim=1)

    def _self_attention_step_multiresolution(self, latents_per_res, coords_per_res,
                                              global_latents, self_attns):
        resolutions = sorted(latents_per_res.keys(), key=str)
        latents_concat, coords_concat, split_sizes = self.concatenate_latents_for_self_attn(
            latents_per_res, coords_per_res, global_latents)
        total_spatial = sum(split_sizes)

        if self.use_rpe:
            px = coords_concat[..., 0]
            py = coords_concat[..., 1]
            for self_attn, self_ff in self_attns:
                latents_concat = self_attn(
                    latents_concat, pos_x=px, pos_y=py,
                    num_spatial=total_spatial) + latents_concat
                latents_concat = self_ff(latents_concat) + latents_concat
        else:
            for self_attn, self_ff in self_attns:
                latents_concat = self_attn(latents_concat) + latents_concat
                latents_concat = self_ff(latents_concat) + latents_concat

        latents_per_res, global_latents = self.split_latents_after_self_attn(
            latents_concat, split_sizes, resolutions)
        return latents_per_res, global_latents

    # =========================================================================
    # Encode
    # =========================================================================

    def encode(self, groups, grid_configs, training=True,
               return_trajectory=False, mask_ratio: float = 0.0,
               cross_k: int = 1024):
        first_group = next(iter(groups.values()))
        B      = first_group["tokens"].shape[0]
        device = first_group["tokens"].device
        resolutions = sorted(groups.keys())

        latents_per_res, coords_per_res = self.init_latents_per_resolution(
            B, grid_configs, device)
        global_latents = self.get_global_latents(B)

        geo_cache = {}
        for res in resolutions:
            tokens    = groups[res]["tokens"]
            mask      = groups[res]["mask"]
            gc        = dict(grid_configs[res])
            coords    = coords_per_res[res]
            L_spatial = gc["L_spatial"]

            geo_tokens, geo_masks = self._apply_pruning(
                tokens, mask, coords, gc, L_spatial)
            geo_cache[res] = (geo_tokens, geo_masks, gc, cross_k)

        mae_active = mask_ratio > 0.0
        vis_latents = {}; vis_coords = {}
        msk_latents = {}; msk_coords = {}
        masked_indices_per_res = {}
        vis_geo_tokens = {}; vis_geo_masks = {}; vis_cross_k = {}

        if mae_active:
            mask_token_vec = self.mask_token.view(1, 1, -1)
            for res in resolutions:
                L      = latents_per_res[res].shape[1]
                n_mask = min(int(mask_ratio * L), L - 1)
                perm        = torch.randperm(L, device=device)
                mask_idx    = perm[:n_mask]
                visible_idx = perm[n_mask:]
                masked_indices_per_res[res] = mask_idx
                vis_latents[res] = latents_per_res[res][:, visible_idx]
                vis_coords[res]  = coords_per_res[res][:, visible_idx]
                msk_latents[res] = mask_token_vec.expand(B, n_mask, -1).clone()
                msk_coords[res]  = coords_per_res[res][:, mask_idx]
                gt, gm, _, ck = geo_cache[res]
                vis_geo_tokens[res] = gt[:, visible_idx]
                vis_geo_masks[res]  = gm[:, visible_idx]
                vis_cross_k[res]    = ck

        trajectory = [coords_per_res.copy()] if return_trajectory else None

        for layer_idx in range(self.depth):
            cross_attn, cross_ff, self_attns = self.encoder_layers[layer_idx]

            for res in resolutions:
                if mae_active:
                    st, sm = self._sample_tokens(
                        vis_geo_tokens[res], vis_geo_masks[res], vis_cross_k[res])
                    if self.gradient_checkpointing and self.training:
                        vis_latents[res] = torch_checkpoint(
                            self._cross_attention_step,
                            vis_latents[res], st, sm,
                            vis_coords[res], cross_attn, cross_ff,
                            vis_latents[res].shape[1],
                            use_reentrant=False)
                    else:
                        vis_latents[res] = self._cross_attention_step(
                            vis_latents[res], st, sm,
                            vis_coords[res], cross_attn, cross_ff,
                            L_spatial=vis_latents[res].shape[1])
                else:
                    gt, gm, gc, ck = geo_cache[res]
                    L_spatial = gc["L_spatial"]
                    st, sm = self._sample_tokens(gt, gm, ck)
                    if self.gradient_checkpointing and self.training:
                        latents_per_res[res] = torch_checkpoint(
                            self._cross_attention_step,
                            latents_per_res[res], st, sm,
                            coords_per_res[res], cross_attn, cross_ff,
                            L_spatial, use_reentrant=False)
                    else:
                        latents_per_res[res] = self._cross_attention_step(
                            latents_per_res[res], st, sm,
                            coords_per_res[res], cross_attn, cross_ff, L_spatial)

            if mae_active:
                full_lpr = {}; full_cpr = {}; L_vis_pr = {}
                for res in resolutions:
                    full_lpr[res] = torch.cat([vis_latents[res], msk_latents[res]], dim=1)
                    full_cpr[res] = torch.cat([vis_coords[res], msk_coords[res]], dim=1)
                    L_vis_pr[res] = vis_latents[res].shape[1]
                full_lpr, global_latents = self._self_attention_step_multiresolution(
                    full_lpr, full_cpr, global_latents, self_attns)
                for res in resolutions:
                    lv = L_vis_pr[res]
                    vis_latents[res] = full_lpr[res][:, :lv]
                    msk_latents[res] = full_lpr[res][:, lv:]
            else:
                latents_per_res, global_latents = self._self_attention_step_multiresolution(
                    latents_per_res, coords_per_res, global_latents, self_attns)

            if return_trajectory:
                trajectory.append(coords_per_res.copy())

        if mae_active:
            for res in resolutions:
                latents_per_res[res] = torch.cat([vis_latents[res], msk_latents[res]], dim=1)
                coords_per_res[res]  = torch.cat([vis_coords[res], msk_coords[res]], dim=1)

        return EncoderOutput(
            latents_per_res=latents_per_res,
            coords_per_res=coords_per_res,
            trajectory=trajectory,
            global_latents=global_latents,
            geo_cache={r: (gt, gm, gc, ck)
                       for r, (gt, gm, gc, ck) in geo_cache.items()},
            masked_indices_per_res=masked_indices_per_res if mae_active else None,
        )

    # =========================================================================
    # >>> SKIP: pixel-token cross-attention (enriched query builder)
    # =========================================================================

    def _pixel_skip(self, query_tokens, query_token_idx, query_token_valid,
                    pool_tokens, pool_mask, training):
        """
        Build the enriched per-pixel query by attending over each query-pixel's
        own encoded band-tokens.

        Args:
            query_tokens      : [B, M, 8]   decoder queries (one per pixel)
            query_token_idx   : [B, M, C]   rows into pool_tokens for each query's
                                            own band-tokens (C = bands_per_pixel)
            query_token_valid : [B, M]      bool, True = real query (not padding)
            pool_tokens       : [B, N, 8]   FULL un-subsampled token pool
            pool_mask         : [B, N]      True/1 = padded OR band-dropped token
            training          : bool

        Returns:
            enriched_query : [B, M, latent_dim]
        """
        B, M, C = query_token_idx.shape
        device = pool_tokens.device

        # ── 1. Gather each pixel's own C band-tokens + their pool mask ─
        flat_idx = query_token_idx.reshape(B, M * C)
        gathered = torch.gather(
            pool_tokens, 1,
            flat_idx.unsqueeze(-1).expand(-1, -1, pool_tokens.shape[-1])
        ).reshape(B, M, C, pool_tokens.shape[-1])                      # [B, M, C, 8]

        pool_mask_b = pool_mask.bool() if pool_mask.dtype != torch.bool else pool_mask
        gathered_mask = torch.gather(
            pool_mask_b, 1, flat_idx
        ).reshape(B, M, C)                                             # [B, M, C] True=masked

        # ── 2. Encode band-tokens (no RPE: zero displacement -> constant
        #        positional block; pass pixel's own coords) ────────────
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)  # [B,M,2]
        encoded = self.input_processor.process_data_for_encoder(
            gathered, gathered_mask, latent_positions=query_coords
        )                                                             # [B, M, C, input_dim]

        # ── 3. key_padding_mask (True = DON'T attend) ─────────────────
        key_pad = gathered_mask.clone()                              # band-drop / pool-pad
        invalid_q = ~query_token_valid.bool()                        # [B, M]
        key_pad = key_pad | invalid_q.unsqueeze(-1)

        # Aggressive random drop, TRAIN ONLY (can go down to 1 token).
        if training and self.decoder_skip_drop_p > 0:
            real = ~key_pad
            drop_roll = torch.bernoulli(
                torch.full((B, M, C), self.decoder_skip_drop_p, device=device)
            ).bool()
            key_pad = key_pad | (drop_roll & real)

        # ── 4. Force-keep guard: every pixel keeps >=1 REAL token ─────
        real_token = ~(gathered_mask | invalid_q.unsqueeze(-1))      # eligible (not band-drop/pad)
        none_kept = (~key_pad).sum(dim=-1) == 0                       # [B, M]
        if none_kept.any():
            has_real   = real_token.any(dim=-1)                      # [B, M]
            first_real = torch.argmax(real_token.float(), dim=-1)    # [B, M]
            fix = none_kept & has_real
            if fix.any():
                bi, mi = torch.where(fix)
                key_pad[bi, mi, first_real[bi, mi]] = False
            still = none_kept & ~has_real
            if still.any():
                bi, mi = torch.where(still)
                key_pad[bi, mi, 0] = False  # fully padded query; output discarded

        # ── 5. Pixel cross-attention ──────────────────────────────────
        BM = B * M
        kv  = encoded.reshape(BM, C, -1)                             # [B*M, C, input_dim]
        q   = self.pixel_query.expand(BM, 1, -1).contiguous()        # [B*M, 1, latent_dim]
        q   = self.pixel_q_norm(q)                                   # >>> QNORM
        kpm = key_pad.reshape(BM, C)                                 # [B*M, C]

        enriched, _ = self.pixel_cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=kpm, need_weights=False,
        )                                                            # [B*M, 1, latent_dim]
        enriched = enriched.squeeze(1).reshape(B, M, -1)             # [B, M, latent_dim]
        return enriched

    # =========================================================================
    # Decoder
    # =========================================================================

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None,
                    training=True, return_features=False,
                    return_topk=False,
                    # >>> SKIP: optional cascade inputs (default None => original behavior)
                    query_token_idx=None, query_token_valid=None,
                    pool_tokens=None, pool_mask=None):
        """
        Decode query pixels into class logits using pooled decoder.
        (See original for the base pipeline; SKIP cascade swaps the query.)
        """
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)

        all_latents = torch.cat(
            [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)], dim=1)
        all_coords = torch.cat(
            [coords_per_res[r] for r in sorted(coords_per_res.keys(), key=str)], dim=1)

        dists_sq = (query_coords.unsqueeze(2) - all_coords.unsqueeze(1)).pow(2).sum(-1)

        k_fetch = min(k, all_coords.shape[1])
        k_keep  = k_fetch

        _, topk_indices = torch.topk(dists_sq, k=k_fetch, dim=-1, largest=False)

        D = all_latents.shape[-1]
        flat_idx = topk_indices.reshape(B, M * k_keep)

        selected_latents = torch.gather(
            all_latents, 1,
            flat_idx.unsqueeze(-1).expand(-1, -1, D)
        ).reshape(B, M, k_keep, D)

        selected_coords = torch.gather(
            all_coords, 1,
            flat_idx.unsqueeze(-1).expand(-1, -1, 2)
        ).reshape(B, M, k_keep, 2)

        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)

        B_d, M_d, K_d = delta_x.shape
        if self.input_processor.use_constant_gsd:
            cs = self.input_processor.compression_alpha * self.input_processor._constant_gsd
        else:
            query_gsd = self.input_processor.geometry.get_token_gsd(query_tokens)
            cs = self.input_processor.compression_alpha * query_gsd

        dx_flat = delta_x.reshape(B_d, M_d * K_d)
        dy_flat = delta_y.reshape(B_d, M_d * K_d)
        rel_pe  = self.input_processor.pos_encoder(dx_flat, dy_flat, compression_scale=cs)
        rel_pe  = rel_pe.reshape(B_d, M_d, K_d, -1)

        context = torch.cat([selected_latents, rel_pe], dim=-1)

        if training and self.decoder_drop_p > 0:
            keep_probs = torch.full(
                (B, M, k_keep), 1.0 - self.decoder_drop_p,
                device=context.device)
            keep_mask = torch.bernoulli(keep_probs).bool()

            none_kept = ~keep_mask.any(dim=-1, keepdim=True)
            if none_kept.any():
                keep_mask = keep_mask.clone()
                keep_mask[..., 0] = keep_mask[..., 0] | none_kept.squeeze(-1)
        else:
            keep_mask = torch.ones(B, M, k_keep, dtype=torch.bool,
                                    device=context.device)

        BM = B * M
        kv_flat     = context.reshape(BM, k_keep, -1)
        kv_flat     = self.dec_ctx_norm(kv_flat)

        # >>> SKIP: build the per-pixel enriched query if the cascade is on.
        # Otherwise fall back to the shared global_query (original behavior).

        if self.use_decoder_skip and query_token_idx is not None:
            enriched = self._pixel_skip(
                query_tokens, query_token_idx, query_token_valid,
                pool_tokens, pool_mask, training)                    # [B, M, latent_dim]
            q_flat = enriched.reshape(BM, 1, -1).contiguous()        # [B*M, 1, latent_dim]
        else:
            q_flat = self.global_query.expand(BM, 1, -1).contiguous()
        # >>> QNORM: pre-norm the decoder query (enriched OR global_query)
        # before the cross-attention. No residual is added on the
        # input-derived enriched query (Perceiver-IO decoder note).
        q_flat = self.dec_q_norm(q_flat)
        # >>> END QNORM
        # >>> END SKIP

        key_pad_flat = (~keep_mask).reshape(BM, k_keep)

        attn_out, _ = self.decoder_cross_attn(
            query=q_flat, key=kv_flat, value=kv_flat,
            key_padding_mask=key_pad_flat,
            need_weights=False,
        )
        attn_out = attn_out.squeeze(1).reshape(B, M, -1)

        if return_features:
            return attn_out

        logits = self.reconstruction_head(attn_out)

        if return_topk:
            topk_dists_sq_kept = torch.gather(dists_sq, 2, topk_indices)
            return logits, topk_indices, topk_dists_sq_kept

        return logits

    # =========================================================================
    # Refinement
    # =========================================================================

    def _spawn_refinement_grid(
        self,
        topk_coords:   torch.Tensor,
        latent_spacing: float,
    ) -> torch.Tensor:
        B, k, _ = topk_coords.shape
        g        = self.refine_grid_size
        offset   = latent_spacing * self.refine_offset_factor
        device   = topk_coords.device

        steps = torch.linspace(-offset * (g - 1) / 2,
                                offset * (g - 1) / 2, g, device=device)
        gy, gx = torch.meshgrid(steps, steps, indexing="ij")
        offsets = torch.stack([gx.flatten(), gy.flatten()], dim=-1)

        refine_coords = (topk_coords.unsqueeze(2)
                         + offsets.unsqueeze(0).unsqueeze(0))
        return refine_coords.reshape(B, k * g * g, 2)

    def _estimate_latent_spacing(self, coords: torch.Tensor) -> float:
        c = coords[0, :min(50, coords.shape[1])]
        with torch.no_grad():
            d = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
            d.fill_diagonal_(float("inf"))
            spacing = d.min(dim=-1).values.median().item()
        return spacing

    def refine(
        self,
        latents_per_res: dict,
        coords_per_res:  dict,
        groups:          dict,
        predicted_errors: torch.Tensor,
        geo_cache:       dict,
        training:        bool,
    ) -> tuple:
        B      = predicted_errors.shape[0]
        device = predicted_errors.device

        resolutions  = sorted(latents_per_res.keys(), key=str)
        all_latents  = torch.cat([latents_per_res[r] for r in resolutions], dim=1)
        all_coords   = torch.cat([coords_per_res[r]  for r in resolutions], dim=1)
        L            = all_latents.shape[1]

        res = resolutions[0]
        geo_tokens_all, geo_masks_all, gc, _ = geo_cache[res]
        L_geo = geo_tokens_all.shape[1]

        errors_for_topk = predicted_errors[:, :L_geo]
        k               = min(self.k_refine, L_geo)
        topk_idx        = torch.topk(errors_for_topk, k, dim=-1).indices

        all_coords_original = all_coords[:, :L_geo]
        topk_coords = torch.gather(
            all_coords_original, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, 2))
        spacing       = self._estimate_latent_spacing(all_coords)
        refine_coords = self._spawn_refinement_grid(topk_coords, spacing)
        n_refine = refine_coords.shape[1]

        refine_latents = repeat(
            self.refinement_latent_content,
            "d -> b n d", b=B, n=n_refine)

        g2 = self.refine_grid_size ** 2
        topk_idx_exp = (topk_idx
                        .unsqueeze(-1).unsqueeze(-1)
                        .expand(-1, -1, geo_tokens_all.shape[2], 8))
        topk_geo_tokens = torch.gather(
            geo_tokens_all, 1, topk_idx_exp)
        topk_geo_masks = torch.gather(
            geo_masks_all, 1,
            topk_idx.unsqueeze(-1).expand(
                -1, -1, geo_masks_all.shape[2]))

        refine_geo_tokens = topk_geo_tokens.unsqueeze(2).expand(
            -1, -1, g2, -1, -1).reshape(B, n_refine,
                                         topk_geo_tokens.shape[2], 8)
        refine_geo_masks  = topk_geo_masks.unsqueeze(2).expand(
            -1, -1, g2, -1).reshape(B, n_refine,
                                     topk_geo_masks.shape[2])

        st, sm = self._sample_tokens(
            refine_geo_tokens, refine_geo_masks,
            cross_k=self.refine_cross_k)

        refine_latents = self._cross_attention_step(
            refine_latents, st, sm,
            refine_coords, self.refine_cross_attn, self.refine_cross_ff,
            L_spatial=n_refine,
        )

        px = refine_coords[..., 0]
        py = refine_coords[..., 1]
        for _ in range(self.refine_self_local):
            refine_latents = (self.refine_self_attn(
                refine_latents, pos_x=px, pos_y=py,
                num_spatial=n_refine) + refine_latents)
            refine_latents = self.refine_self_ff(refine_latents) + refine_latents

        _, _, self_attns = self.encoder_layers[0]
        merged_latents = torch.cat([all_latents, refine_latents], dim=1)
        merged_coords  = torch.cat([all_coords,  refine_coords],  dim=1)

        for sa_idx in range(self.refine_self_global):
            self_attn, self_ff = self_attns[sa_idx % len(self_attns)]
            if self.use_rpe:
                px = merged_coords[..., 0]
                py = merged_coords[..., 1]
                merged_latents = (self_attn(
                    merged_latents, pos_x=px, pos_y=py,
                    num_spatial=merged_latents.shape[1]) + merged_latents)
            else:
                merged_latents = self_attn(merged_latents) + merged_latents
            merged_latents = self_ff(merged_latents) + merged_latents

        latents_per_res_updated = {}
        coords_per_res_updated  = {}

        offset = 0
        for r in resolutions:
            n = latents_per_res[r].shape[1]
            latents_per_res_updated[r] = merged_latents[:, offset:offset + n]
            coords_per_res_updated[r]  = merged_coords[:, offset:offset + n]
            offset += n

        latents_per_res_updated["refinement"] = merged_latents[:, offset:]
        coords_per_res_updated["refinement"]  = merged_coords[:, offset:]

        return latents_per_res_updated, coords_per_res_updated

    # =========================================================================
    # Targeted Depth-2
    # =========================================================================

    def _targeted_depth2(
        self,
        latents_per_res:  dict,
        coords_per_res:   dict,
        predicted_errors: torch.Tensor,
        geo_cache:        dict,
        global_latents:   Optional[torch.Tensor],
    ) -> dict:
        resolutions = sorted(latents_per_res.keys(), key=str)
        res         = resolutions[0]

        all_latents = torch.cat(
            [latents_per_res[r] for r in resolutions
             if r != "refinement"], dim=1)
        all_coords  = torch.cat(
            [coords_per_res[r]  for r in resolutions
             if r != "refinement"], dim=1)
        B, L, D = all_latents.shape

        geo_tokens_all, geo_masks_all, gc, _ = geo_cache[res]
        L_geo = geo_tokens_all.shape[1]

        errors_for_topk = predicted_errors[:, :L_geo]
        k = min(self.td2_k, L_geo)
        topk_idx = torch.topk(errors_for_topk, k, dim=-1).indices

        topk_latents = torch.gather(
            all_latents, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, D))
        topk_coords  = torch.gather(
            all_coords, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, 2))

        geo_k = geo_tokens_all.shape[2]
        topk_geo_tokens = torch.gather(
            geo_tokens_all, 1,
            topk_idx.unsqueeze(-1).unsqueeze(-1)
                    .expand(-1, -1, geo_k, 8))
        topk_geo_masks  = torch.gather(
            geo_masks_all, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, geo_k))

        cross_attn, cross_ff, _ = self.encoder_layers[0]
        st, sm = self._sample_tokens(
            topk_geo_tokens, topk_geo_masks,
            cross_k=self.td2_cross_k)

        topk_latents_updated = self._cross_attention_step(
            topk_latents, st, sm,
            topk_coords, cross_attn, cross_ff,
            L_spatial=k,
        )

        all_latents = all_latents.clone()
        all_latents.scatter_(
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, D),
            topk_latents_updated,
        )

        updated_per_res = {}
        updated_coords  = {}
        offset = 0
        for r in resolutions:
            if r == "refinement":
                continue
            n = latents_per_res[r].shape[1]
            updated_per_res[r] = all_latents[:, offset:offset + n]
            updated_coords[r]  = all_coords[:,  offset:offset + n]
            offset += n

        if "refinement" in latents_per_res:
            updated_per_res["refinement"] = latents_per_res["refinement"]
            updated_coords["refinement"]  = coords_per_res["refinement"]

        _, _, self_attns = self.encoder_layers[0]
        updated_per_res, global_latents = \
            self._self_attention_step_multiresolution(
                updated_per_res, updated_coords, global_latents, self_attns)

        return updated_per_res

    def classify(self, latents_per_res):
        all_latents = torch.cat(
            [latents_per_res[res] for res in sorted(latents_per_res.keys(), key=str)], dim=1)
        return self.to_logits(all_latents)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, batch, training=True, task="reconstruction",
                return_trajectory=False, return_predicted_errors=False,
                return_features=False, tokens_per_latent_override=None,
                mask_ratio: float = 0.0, return_for_error=False):

        groups       = batch["groups"]
        queries      = batch["queries"]
        queries_mask = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)

        # >>> SKIP: read cascade inputs + the intact full token pool.
        query_token_idx   = batch.get("query_token_idx", None)    # [B, M, C]
        query_token_valid = batch.get("query_token_valid", None)  # [B, M]
        skip_pool_tokens = None
        skip_pool_mask   = None
        if self.use_decoder_skip and query_token_idx is not None:
            skip_res = sorted(groups.keys())[0]   # single-res Sen1Floods11
            skip_pool_tokens = groups[skip_res]["tokens"]   # [B, N, 8] intact
            skip_pool_mask   = groups[skip_res]["mask"]     # [B, N]
        # >>> END SKIP

        if tokens_per_latent_override is not None:
            tpl = tokens_per_latent_override
            batch_cross_k = self.val_sampling[0][1]
        else:
            tpl, batch_cross_k = self.sample_config(training)

        resolutions   = sorted(groups.keys())
        geo_k_budget  = batch_cross_k * 2

        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                tokens_per_latent=tpl,
                total_tokens=groups[res]["tokens"].shape[1],
                sigma_factor=self.sigma_factor,
                max_k=geo_k_budget,
            )
            for res in resolutions
        }

        need_trajectory = return_trajectory or task == "visualization"
        encoder_output = self.encode(
            groups=groups, grid_configs=grid_configs,
            training=training, return_trajectory=need_trajectory,
            mask_ratio=mask_ratio, cross_k=batch_cross_k)

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
            predicted_errors = None
            all_latents_for_err = torch.cat(
                [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)], dim=1)
            if self.use_error_predictor:
                predicted_errors = self.error_predictor(
                    all_latents_for_err.detach()
                ).squeeze(-1)

            if (self.use_refinement
                    and predicted_errors is not None
                    and encoder_output.geo_cache is not None):
                latents_per_res, coords_per_res = self.refine(
                    latents_per_res=latents_per_res,
                    coords_per_res=coords_per_res,
                    groups=groups,
                    predicted_errors=predicted_errors,
                    geo_cache=encoder_output.geo_cache,
                    training=training,
                )

            if (self.use_targeted_depth2
                    and predicted_errors is not None
                    and encoder_output.geo_cache is not None):
                latents_per_res = self._targeted_depth2(
                    latents_per_res=latents_per_res,
                    coords_per_res=coords_per_res,
                    predicted_errors=predicted_errors,
                    geo_cache=encoder_output.geo_cache,
                    global_latents=encoder_output.global_latents,
                )

            chunk_size = 10_000
            N = queries.shape[1]
            need_topk = return_for_error and self.use_error_predictor

            if N > chunk_size:
                preds_list      = []
                topk_idx_list   = []
                topk_dists_list = []
                for i in range(0, N, chunk_size):
                    # >>> SKIP: slice cascade inputs in lockstep with queries.
                    chunk_qti = (query_token_idx[:, i:i+chunk_size]
                                 if query_token_idx is not None else None)
                    chunk_qtv = (query_token_valid[:, i:i+chunk_size]
                                 if query_token_valid is not None else None)
                    chunk_result = self.reconstruct(
                        latents_per_res, coords_per_res,
                        queries[:, i:i+chunk_size],
                        queries_mask[:, i:i+chunk_size],
                        target_resolution=target_resolution,
                        training=training,
                        return_features=return_features,
                        return_topk=need_topk,
                        query_token_idx=chunk_qti,
                        query_token_valid=chunk_qtv,
                        pool_tokens=skip_pool_tokens,
                        pool_mask=skip_pool_mask,
                    )
                    if need_topk:
                        preds_list.append(chunk_result[0])
                        topk_idx_list.append(chunk_result[1])
                        topk_dists_list.append(chunk_result[2])
                    else:
                        preds_list.append(chunk_result)
                output = torch.cat(preds_list, dim=1)
                if need_topk:
                    topk_indices  = torch.cat(topk_idx_list,   dim=1)
                    topk_dists_sq = torch.cat(topk_dists_list, dim=1)
            else:
                chunk_result = self.reconstruct(
                    latents_per_res, coords_per_res,
                    queries, queries_mask,
                    target_resolution=target_resolution,
                    training=training,
                    return_features=return_features,
                    return_topk=need_topk,
                    # >>> SKIP: full (unchunked) cascade inputs
                    query_token_idx=query_token_idx,
                    query_token_valid=query_token_valid,
                    pool_tokens=skip_pool_tokens,
                    pool_mask=skip_pool_mask,
                )
                if need_topk:
                    output, topk_indices, topk_dists_sq = chunk_result
                else:
                    output = chunk_result

            if return_features:
                return {"features": output, "latents_per_res": latents_per_res,
                        "coords_per_res": coords_per_res, "encoder_output": encoder_output}

            if return_for_error and need_topk:
                all_coords = torch.cat(
                    [coords_per_res[r] for r in sorted(coords_per_res.keys(), key=str)], dim=1)
                all_latents_post = torch.cat(
                    [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)], dim=1)
                return {
                    "predictions":      output,
                    "predicted_errors": predicted_errors,
                    "topk_indices":     topk_indices,
                    "topk_dists_sq":    topk_dists_sq,
                    "num_latents":      all_latents_post.shape[1],
                    "latent_coords":    all_coords,
                }

            if task == "visualization" or return_predicted_errors:
                return {'predictions': output, 'latents_per_res': latents_per_res,
                        'coords_per_res': coords_per_res, 'trajectory': trajectory,
                        'predicted_errors': predicted_errors}
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
        self._set_requires_grad(self.dec_q_norm,False)
        self._set_requires_grad(self.dec_ctx_norm, False)
        self._set_requires_grad(self.decoder_cross_attn, False)
        self._set_requires_grad(self.reconstruction_head, False)

        if self.use_error_predictor:
            self._set_requires_grad(self.error_predictor, False)
        if self.use_refinement:
            self.refinement_latent_content.requires_grad = False
            self._set_requires_grad(self.refine_cross_attn, False)
            self._set_requires_grad(self.refine_cross_ff, False)
            self._set_requires_grad(self.refine_self_attn, False)
            self._set_requires_grad(self.refine_self_ff, False)
        # >>> SKIP
        if self.use_decoder_skip:
            self._set_requires_grad(self.pixel_cross_attn, False)
            self._set_requires_grad(self.pixel_q_norm, False)
            self.pixel_query.requires_grad = False
        # >>> END SKIP

    def unfreeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, True)
        self._set_requires_grad(self.dec_q_norm,True)
        self._set_requires_grad(self.reconstruction_head, True)
        self._set_requires_grad(self.dec_ctx_norm, True)
        if self.use_error_predictor:
            self._set_requires_grad(self.error_predictor, True)
        if self.use_refinement:
            self.refinement_latent_content.requires_grad = True
            self._set_requires_grad(self.refine_cross_attn, True)
            self._set_requires_grad(self.refine_cross_ff, True)
            self._set_requires_grad(self.refine_self_attn, True)
            self._set_requires_grad(self.refine_self_ff, True)
        # >>> SKIP
        if self.use_decoder_skip:
            self._set_requires_grad(self.pixel_cross_attn, True)
            self._set_requires_grad(self.pixel_q_norm, True)
            self.pixel_query.requires_grad = True
        # >>> END SKIP

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
