"""
Atomiser Model — Multi-Resolution Encoder/Decoder
==================================================

Changes vs previous version:
  - decoder_learned_dim removed (processor no longer has it)
  - _compute_grid_spacing cached per latent count (avoids O(L²) cdist every decode)
  - geographic pruning bias capture cleaned up (was silently discarded anyway)
  - global_latents kept: participates in self-attention for global context,
    intentionally not wired into decoder

Config:
  latent_grid:
    sigma_factor: 1.5
    hexagonal: true
    train_sampling:
      - [3000, 1000]
    val_sampling:
      - [3000, 1000]
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

            # Compression scale for mini-latent local self-attention.
            # Should be tuned to the physical scale of the refinement zone.
            # With offset_factor=0.5 and spacing=115m, mini-latents span
            # ~230m diagonally. A scale of ~1250m keeps positions well within
            # the linear regime (compressed pos ≈ 0.044 at 57.5m offset)
            # while still giving meaningful RoPE rotations.
            # The formula-derived value (offset * spacing * 3 ≈ 172m) was
            # too aggressive — positions saturated too quickly.
            refine_cs = float(ref_cfg.get(
                "self_attn_compression_scale",
                1250.0,   # safe default: linear regime, learnable from here
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

            # Dedicated cross-attention for refinement level.
            # Separate from encoder_layers[0] so it can specialize in
            # high-frequency boundary detection rather than inheriting
            # the coarse averaging behavior of the main encoder.
            # Gradient signal comes exclusively from boundary pixels
            # via mini-latents → learns discriminative feature extraction.
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

        # Learned initialization for refinement (mini) latents.
        # Separate from spatial_latent_content so mini-latents can specialize
        # in detecting fine-grained boundary/hole patterns from the start.
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

        Pipeline:
            1. Select k nearest latents per query pixel
            2. Build context = cat([latent, rel_pe])  — no MLP projection
            3. Bernoulli drop on context tokens (training only)
            4. Cross-attention: Q = global learned vector, K/V = context
            5. MLP head → logits

        Design motivation:
          - Context is just cat([latent, rel_pe]). The cross-attention's
            internal K/V projections absorb the positional signal —
            no redundant MLP needed. Latents already refined by encoder.
          - Global learned query is task-transferable and pixel-agnostic —
            forces K/V to carry all pixel-specific information.
          - Bernoulli drop prevents collapse onto the nearest latent.
        """
        # ── Context dimension = latent_dim + rel_pe_dim ───────────────
        # We use a MultiheadAttention with separate kdim/vdim since the
        # K/V carry the position-augmented context, while Q is latent_dim.
        self.decoder_context_dim = self.latent_dim + self.decoder_pe_dim

        # ── Global learned query ──────────────────────────────────────
        # Single vector, shared across all query pixels.
        # K/V differ per pixel, so attention output still differs per pixel.
        self.global_query = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        nn.init.trunc_normal_(self.global_query, std=0.02, a=-2., b=2.)

        # ── Cross-attention (Q dim != K/V dim) ────────────────────────
        # Q     : [B*M, 1, latent_dim]
        # K,V   : [B*M, k, latent_dim + decoder_pe_dim]
        # Output: [B*M, 1, latent_dim]
        # MultiheadAttention handles kdim/vdim != embed_dim via internal projections.
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            kdim=self.decoder_context_dim,
            vdim=self.decoder_context_dim,
            num_heads=self.cross_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        # ── Bernoulli drop probability ────────────────────────────────
        # Applied per-latent during training only.
        # p=0.25 drops 1 of 4 latents on average, forcing the decoder
        # to extract useful info from non-nearest neighbors.
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
        # Predicts per-latent segmentation difficulty (soft CE).
        # Input latents are DETACHED — no gradient flows to the encoder.
        # Supervised by compute_latent_errors() using the k-nearest
        # assignment already computed in reconstruct().
        if self.use_error_predictor:
            self.error_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 4),
                nn.GELU(),
                nn.Linear(self.latent_dim // 4, 1),
                nn.Softplus(),  # positive output — matches CE target range
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
        # Geographic bias returned by geo_pruning is not used here —
        # cross-attention uses RoPE instead of the Gaussian distance bias.
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

        # Prevent all-masked latents (softmax over all -inf → NaN)
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
            # Use raw meter coordinates — consistent across all datasets
            # since latent positions are always in meters.
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

        # ── Geographic pruning (once, no gradients) ───────────────────
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

        # ── MAE: split once before layer loop ─────────────────────────
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

        # ── Layer loop ────────────────────────────────────────────────
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
    # Decoder
    # =========================================================================

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None,
                    training=True, return_features=False,
                    return_topk=False):
        """
        Decode query pixels into class logits using pooled decoder.

        Pipeline:
            1. Select k nearest latents per query pixel
            2. Compute relative displacement (query → latent)
            3. context = cat([latent, rel_pe])           (no MLP)
            4. Bernoulli drop (training only) — forces non-reliance on nearest
            5. Cross-attn: Q=global_query, K/V=context
            6. Segmentation head → logits

        Shapes:
            query_coords:   [B, M, 2]
            selected:       [B, M, k, latent_dim]
            rel_pe:         [B, M, k, pe_dim]
            context:        [B, M, k, latent_dim + pe_dim]
            attn_out:       [B, M, latent_dim]
            output logits:  [B, M, num_classes]
        """
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        # ── Query coords (for k-nearest + rel_pe) ─────────────────────
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)

        # ── Concat all latents across resolutions ─────────────────────
        all_latents = torch.cat(
            [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)], dim=1)
        all_coords = torch.cat(
            [coords_per_res[r] for r in sorted(coords_per_res.keys(), key=str)], dim=1)

        # ── Select k nearest latents per query ────────────────────────
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

        # ── Relative displacement query → latent ──────────────────────
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)

        # ── Relative positional encoding ──────────────────────────────
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
        # rel_pe: [B, M, k, pe_dim]

        # ── Build context = cat([latent, rel_pe]) ─────────────────────
        # No MLP — K/V projections inside cross-attention absorb the
        # positional signal. Saves memory and compute.
        context = torch.cat([selected_latents, rel_pe], dim=-1)
        # context: [B, M, k, latent_dim + pe_dim]

        # ── Bernoulli drop (training only) ────────────────────────────
        # Drop each latent independently with p=drop_p.
        # Ensures at least 1 latent kept per pixel.
        if training and self.decoder_drop_p > 0:
            keep_probs = torch.full(
                (B, M, k_keep), 1.0 - self.decoder_drop_p,
                device=context.device)
            keep_mask = torch.bernoulli(keep_probs).bool()          # [B, M, k]

            # Guard: if a pixel has 0 kept latents, force keep its nearest
            none_kept = ~keep_mask.any(dim=-1, keepdim=True)        # [B, M, 1]
            if none_kept.any():
                keep_mask = keep_mask.clone()
                keep_mask[..., 0] = keep_mask[..., 0] | none_kept.squeeze(-1)
        else:
            keep_mask = torch.ones(B, M, k_keep, dtype=torch.bool,
                                    device=context.device)

        # ── Cross-attention: Q=global_query, K/V=context ──────────────
        # Q dim = latent_dim, K/V dim = latent_dim + pe_dim
        # MultiheadAttention handles kdim/vdim != embed_dim internally.
        BM = B * M
        kv_flat     = context.reshape(BM, k_keep, -1)                  # [B*M, k, D+pe]
        q_flat      = self.global_query.expand(BM, 1, -1).contiguous() # [B*M, 1, D]
        key_pad_flat = (~keep_mask).reshape(BM, k_keep)                # [B*M, k]

        attn_out, _ = self.decoder_cross_attn(
            query=q_flat, key=kv_flat, value=kv_flat,
            key_padding_mask=key_pad_flat,
            need_weights=False,
        )
        # attn_out: [B*M, 1, latent_dim]
        attn_out = attn_out.squeeze(1).reshape(B, M, -1)               # [B, M, latent_dim]

        if return_features:
            return attn_out

        # ── Segmentation head ─────────────────────────────────────────
        logits = self.reconstruction_head(attn_out)                   # [B, M, num_classes]

        if return_topk:
            # Reuse topk assignment for error supervision
            topk_dists_sq_kept = torch.gather(dists_sq, 2, topk_indices)
            return logits, topk_indices, topk_dists_sq_kept

        return logits

    # =========================================================================
    # Refinement
    # =========================================================================

    def _spawn_refinement_grid(
        self,
        topk_coords:   torch.Tensor,   # [B, k, 2]  high-error latent positions
        latent_spacing: float,          # meters — used to set grid offset
    ) -> torch.Tensor:
        """
        Spawn a grid_size×grid_size pattern of mini-latent positions around
        each of the k high-error latents.

        With grid_size=2 and offset_factor=0.25:
            offset = latent_spacing * 0.25
            offsets = [(-o,-o), (-o,+o), (+o,-o), (+o,+o)]

        Returns:
            refine_coords: [B, k*grid_size², 2]  mini-latent meter positions
        """
        B, k, _ = topk_coords.shape
        g        = self.refine_grid_size
        offset   = latent_spacing * self.refine_offset_factor
        device   = topk_coords.device

        # Build offset grid: g×g offsets centered at 0
        steps = torch.linspace(-offset * (g - 1) / 2,
                                offset * (g - 1) / 2, g, device=device)
        gy, gx = torch.meshgrid(steps, steps, indexing="ij")
        offsets = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [g², 2]

        # Expand and add: [B, k, 1, 2] + [1, 1, g², 2] → [B, k, g², 2]
        refine_coords = (topk_coords.unsqueeze(2)
                         + offsets.unsqueeze(0).unsqueeze(0))
        return refine_coords.reshape(B, k * g * g, 2)                # [B, k*g², 2]

    def _estimate_latent_spacing(self, coords: torch.Tensor) -> float:
        """
        Estimate typical latent spacing from the first few latents.
        Uses a small sample to avoid O(L²) cost.
        Fast approximation: median of nearest-neighbor distances on a 20-pt subset.
        """
        c = coords[0, :min(50, coords.shape[1])]  # [≤50, 2]
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
        predicted_errors: torch.Tensor,  # [B, L]
        geo_cache:       dict,
        training:        bool,
    ) -> tuple:
        """
        Refinement step: spawn mini-latents at high-error zones and re-encode.

        Pipeline:
            1. TopK high-error latents → positions [B, k, 2]
            2. Spawn g×g grid → refine_coords [B, k*g², 2]
            3. Init refine_latents from refinement_latent_content
            4. Reuse geo_cache token pools for the topk latents (free)
            5. Cross-attention: refine_latents × local tokens (500 tpl, 500 cross_k)
            6. Self-attention ×refine_self_local: mini-latents only
            7. Self-attention ×refine_self_global: full merged pool
            8. Return updated latents_per_res + refine_coords merged in

        Args:
            latents_per_res:  {res: [B, L, D]}  original latents after encode
            coords_per_res:   {res: [B, L, 2]}  original latent coords
            groups:           batch["groups"]
            predicted_errors: [B, L]  from error_predictor (detached)
            geo_cache:        {res: (geo_tokens, geo_masks, gc, cross_k)}
            training:         bool

        Returns:
            latents_per_res_updated: dict with refined latents merged in
            coords_per_res_updated:  dict with refined coords merged in
        """
        B      = predicted_errors.shape[0]
        device = predicted_errors.device

        # ── Concatenate all latents/coords ────────────────────────────
        resolutions  = sorted(latents_per_res.keys(), key=str)
        all_latents  = torch.cat([latents_per_res[r] for r in resolutions], dim=1)
        all_coords   = torch.cat([coords_per_res[r]  for r in resolutions], dim=1)
        L            = all_latents.shape[1]

        # ── 1. TopK high-error latents ────────────────────────────────
        # Use only the first L_geo latents for topk selection —
        # geo_cache only has entries for the original L latents from
        # the first encode, not for any refinement latents added later.
        res = resolutions[0]
        geo_tokens_all, geo_masks_all, gc, _ = geo_cache[res]
        L_geo = geo_tokens_all.shape[1]  # original latent count

        # Only score original latents, not refinement latents
        errors_for_topk = predicted_errors[:, :L_geo]
        k               = min(self.k_refine, L_geo)
        topk_idx        = torch.topk(errors_for_topk, k, dim=-1).indices  # [B, k]

        # Gather coords from original latents only
        all_coords_original = all_coords[:, :L_geo]
        topk_coords = torch.gather(
            all_coords_original, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, 2))                  # [B, k, 2]
        # ── 2. Spawn g×g grid around each high-error latent ──────────
        spacing       = self._estimate_latent_spacing(all_coords)
        refine_coords = self._spawn_refinement_grid(topk_coords, spacing)
        # refine_coords: [B, k*g², 2]
        n_refine = refine_coords.shape[1]  # k * g²

        # ── 3. Initialize refinement latents ─────────────────────────
        refine_latents = repeat(
            self.refinement_latent_content,
            "d -> b n d", b=B, n=n_refine)                             # [B, k*g², D]

        # ── 4. Reuse geo_cache token pools ────────────────────────────
        # Each group of g² mini-latents shares the token pool of its
        # parent high-error latent — already computed, zero extra cost.

        g2 = self.refine_grid_size ** 2
        topk_idx_exp = (topk_idx
                        .unsqueeze(-1).unsqueeze(-1)
                        .expand(-1, -1, geo_tokens_all.shape[2], 8))   # [B, k, geo_k, 8]
        topk_geo_tokens = torch.gather(
            geo_tokens_all, 1, topk_idx_exp)                           # [B, k, geo_k, 8]
        topk_geo_masks = torch.gather(
            geo_masks_all, 1,
            topk_idx.unsqueeze(-1).expand(
                -1, -1, geo_masks_all.shape[2]))                       # [B, k, geo_k]

        # Repeat token pool for each of the g² mini-latents per parent
        # [B, k, geo_k, 8] → [B, k*g², geo_k, 8]
        refine_geo_tokens = topk_geo_tokens.unsqueeze(2).expand(
            -1, -1, g2, -1, -1).reshape(B, n_refine,
                                         topk_geo_tokens.shape[2], 8)
        refine_geo_masks  = topk_geo_masks.unsqueeze(2).expand(
            -1, -1, g2, -1).reshape(B, n_refine,
                                     topk_geo_masks.shape[2])

        # ── 5. Cross-attention for refinement latents ─────────────────
        # Uses dedicated refine_cross_attn — separate weights from the
        # main encoder so it can specialize in high-frequency boundary
        # detection rather than inheriting coarse averaging behavior.
        st, sm = self._sample_tokens(
            refine_geo_tokens, refine_geo_masks,
            cross_k=self.refine_cross_k)

        refine_latents = self._cross_attention_step(
            refine_latents, st, sm,
            refine_coords, self.refine_cross_attn, self.refine_cross_ff,
            L_spatial=n_refine,
        )

        # ── 6. Self-attention: mini-latents only ──────────────────────
        # Uses refine_self_attn with a small compression scale tuned to
        # the mini-latent spacing (~29m) rather than the full image extent.
        # This ensures the 4 latents in each 2×2 grid are distinguishable
        # from each other by their position. The standard encoder self-attn
        # (compression_scale=2560m) would make them nearly indistinguishable.
        px = refine_coords[..., 0]
        py = refine_coords[..., 1]
        for _ in range(self.refine_self_local):
            refine_latents = (self.refine_self_attn(
                refine_latents, pos_x=px, pos_y=py,
                num_spatial=n_refine) + refine_latents)
            refine_latents = self.refine_self_ff(refine_latents) + refine_latents

        # ── 7. Self-attention: full merged pool ───────────────────────
        # Uses encoder_layers[0] self-attns — large compression scale
        # (2560m) well-calibrated for original latent spacing (~115m).
        _, _, self_attns = self.encoder_layers[0]
        merged_latents = torch.cat([all_latents, refine_latents], dim=1)  # [B, L+n, D]
        merged_coords  = torch.cat([all_coords,  refine_coords],  dim=1)  # [B, L+n, 2]

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

        # ── 8. Pack back into per-res dicts ───────────────────────────
        # Keep original per-resolution structure intact.
        # Add refinement latents under a dedicated "refinement" key.
        # The decoder already does:
        #   torch.cat([latents_per_res[r] for r in sorted(keys)], dim=1)
        # so "refinement" will be concatenated automatically alongside
        # the original resolution slots. Works for both single-res
        # (Sen1Floods11) and multi-res (PASTIS) without modification.
        #
        # Note: after global self-attention the merged_latents contain
        # updated versions of both original and refinement latents.
        # We split them back here so each slot stays consistent.
        latents_per_res_updated = {}
        coords_per_res_updated  = {}

        # Restore original resolution slots with their updated latents
        # (they were updated by the global self-attention)
        offset = 0
        for r in resolutions:
            n = latents_per_res[r].shape[1]
            latents_per_res_updated[r] = merged_latents[:, offset:offset + n]
            coords_per_res_updated[r]  = merged_coords[:, offset:offset + n]
            offset += n

        # Add refinement latents as a separate slot
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
        predicted_errors: torch.Tensor,  # [B, L]
        geo_cache:        dict,
        global_latents:   Optional[torch.Tensor],
    ) -> dict:
        """
        Second cross-attention + self-attention pass on top-k high-error latents.

        Uses shared encoder_layers[0] weights — zero new parameters.
        Cost: O(k × geo_k) cross-attn + O(L²) self-attn
              ≈ 2.5% of a full second encode for k=50, L=1966.

        Pipeline:
            1. topk(errors, k) → select k high-error latent indices
            2. Gather their geo_cache token pools (full pool, no subsampling)
            3. Cross-attention: k latents × geo_k tokens (shared weights)
            4. Write updated k latents back into latents_per_res
            5. Full self-attention on all L latents to propagate updates
        """
        resolutions = sorted(latents_per_res.keys(), key=str)
        res         = resolutions[0]  # geo_cache uses first resolution

        # ── Concat original latents ───────────────────────────────────
        all_latents = torch.cat(
            [latents_per_res[r] for r in resolutions
             if r != "refinement"], dim=1)                 # [B, L, D]
        all_coords  = torch.cat(
            [coords_per_res[r]  for r in resolutions
             if r != "refinement"], dim=1)                 # [B, L, 2]
        B, L, D = all_latents.shape

        # ── 1. TopK high-error latents ────────────────────────────────
        geo_tokens_all, geo_masks_all, gc, _ = geo_cache[res]
        L_geo = geo_tokens_all.shape[1]  # original latent count only

        errors_for_topk = predicted_errors[:, :L_geo]
        k = min(self.td2_k, L_geo)
        topk_idx = torch.topk(errors_for_topk, k, dim=-1).indices  # [B, k]

        # ── 2. Gather latents + coords + token pools ──────────────────
        topk_latents = torch.gather(
            all_latents, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, D))                # [B, k, D]
        topk_coords  = torch.gather(
            all_coords, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, 2))                # [B, k, 2]

        geo_k = geo_tokens_all.shape[2]
        topk_geo_tokens = torch.gather(
            geo_tokens_all, 1,
            topk_idx.unsqueeze(-1).unsqueeze(-1)
                    .expand(-1, -1, geo_k, 8))                       # [B, k, geo_k, 8]
        topk_geo_masks  = torch.gather(
            geo_masks_all, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, geo_k))            # [B, k, geo_k]

        # ── 3. Second cross-attention (shared weights) ────────────────
        # Use up to td2_cross_k tokens — default 2000 (full geo pool)
        cross_attn, cross_ff, _ = self.encoder_layers[0]
        st, sm = self._sample_tokens(
            topk_geo_tokens, topk_geo_masks,
            cross_k=self.td2_cross_k)

        # _cross_attention_step expects [B, L_total, D] where first L_spatial
        # are spatial. Wrap topk_latents with no global latents appended.
        topk_latents_updated = self._cross_attention_step(
            topk_latents, st, sm,
            topk_coords, cross_attn, cross_ff,
            L_spatial=k,
        )
        # Output: [B, k, D]  (no global latents so shape unchanged)

        # ── 4. Write updated latents back ─────────────────────────────
        all_latents = all_latents.clone()
        all_latents.scatter_(
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, D),
            topk_latents_updated,
        )

        # ── 5. Full self-attention to propagate boundary updates ──────
        # Rebuild per-res dicts with updated latents
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

        # Carry over refinement slot if present
        if "refinement" in latents_per_res:
            updated_per_res["refinement"] = latents_per_res["refinement"]
            updated_coords["refinement"]  = coords_per_res["refinement"]

        _, _, self_attns = self.encoder_layers[0]
        updated_per_res, global_latents =             self._self_attention_step_multiresolution(
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
            # ── Error predictor (runs on detached latents) ─────────────
            # Always computed when enabled — used for supervision (train)
            # and for refinement selection (val/test).
            predicted_errors = None
            all_latents_for_err = torch.cat(
                [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)], dim=1)
            if self.use_error_predictor:
                predicted_errors = self.error_predictor(
                    all_latents_for_err.detach()   # detach: no grad to encoder
                ).squeeze(-1)                       # [B, L]

            # ── Refinement (mini-latents) ──────────────────────────────
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

            # ── Targeted depth-2 ───────────────────────────────────────
            # Second cross-attention + self-attention on top-k high-error
            # latents. Shared weights, zero new parameters, ~2.5% extra cost.
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

            # ── Decode ─────────────────────────────────────────────────
            # return_topk=True on first chunk only — for error supervision
            # we only need the assignment from a representative chunk.
            # For simplicity we collect topk from all chunks and concat.
            chunk_size = 10_000
            N = queries.shape[1]
            need_topk = return_for_error and self.use_error_predictor

            if N > chunk_size:
                preds_list      = []
                topk_idx_list   = []
                topk_dists_list = []
                for i in range(0, N, chunk_size):
                    chunk_result = self.reconstruct(
                        latents_per_res, coords_per_res,
                        queries[:, i:i+chunk_size],
                        queries_mask[:, i:i+chunk_size],
                        target_resolution=target_resolution,
                        training=training,
                        return_features=return_features,
                        return_topk=need_topk,
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
                )
                if need_topk:
                    output, topk_indices, topk_dists_sq = chunk_result
                else:
                    output = chunk_result

            if return_features:
                return {"features": output, "latents_per_res": latents_per_res,
                        "coords_per_res": coords_per_res, "encoder_output": encoder_output}

            # ── Return dict for error supervision ───────────────────────
            if return_for_error and need_topk:
                all_coords = torch.cat(
                    [coords_per_res[r] for r in sorted(coords_per_res.keys(), key=str)], dim=1)
                # num_latents must reflect the TOTAL latent count after
                # refinement — topk_indices in reconstruct index into the
                # full merged pool (original + refinement latents).
                all_latents_post = torch.cat(
                    [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)], dim=1)
                return {
                    "predictions":      output,
                    "predicted_errors": predicted_errors,  # [B, L_original]
                    "topk_indices":     topk_indices,       # [B, M, k]
                    "topk_dists_sq":    topk_dists_sq,      # [B, M, k]
                    "num_latents":      all_latents_post.shape[1],  # L_original + n_refine
                    "latent_coords":    all_coords,          # [B, L_total, 2] for viz
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

    def unfreeze_decoder(self):
        self._set_requires_grad(self.decoder_cross_attn, True)
        self._set_requires_grad(self.reconstruction_head, True)
        if self.use_error_predictor:
            self._set_requires_grad(self.error_predictor, True)
        if self.use_refinement:
            self.refinement_latent_content.requires_grad = True
            self._set_requires_grad(self.refine_cross_attn, True)
            self._set_requires_grad(self.refine_cross_ff, True)
            self._set_requires_grad(self.refine_self_attn, True)
            self._set_requires_grad(self.refine_self_ff, True)

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