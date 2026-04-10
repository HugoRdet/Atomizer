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
        # 7. INITIALIZE COMPONENTS
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
        k-nearest cross-attention decoder with RoPE and query skip.

        Pipeline:
            1. Select k nearest latents per query pixel
            2. Build context = [latent | rel_pe]  (explicit relative geometry)
            3. LocalCrossAttentionRoPE: Q=query_features, K/V=context
            4. output_head([cross_attn_output | query_features]) → logits

        The concatenation of query_features in the output_head gives the MLP
        explicit access to modality metadata (spectral index, resolution)
        alongside the spatial content from cross-attention.
        This is not a residual — it's a richer input to the final projection.
        """
        rope_compression_scale = self.config["RoPE"].get("cross_compression_scale", 50.0)
        rope_base              = self.config["RoPE"].get("base", 100.0)

        # context_dim = latent features + relative PE
        decoder_context_dim = self.latent_dim + self.decoder_pe_dim

        # Position-aware cross-attention in decoder
        self.decoder_cross_attn = LocalCrossAttentionRoPE(
            dim_query=self.query_dim_recon,
            dim_context=decoder_context_dim,
            dim_out=self.latent_dim,
            heads=self.cross_heads,
            dim_head=self.cross_dim_head,
            dropout=self.attn_dropout,
            use_rope=self.encoder_use_rpe,
            rope_base=rope_base,
            rope_compression_scale=rope_compression_scale,
            rope_learnable_scale=self.rope_learnable_scale,
        )

        # Final MLP: [cross_attn_output (latent_dim) | query_features (query_dim_recon)]
        mlp_input_dim = self.latent_dim + self.query_dim_recon
        hidden_dim    = self.latent_dim * 2

        self.reconstruction_head = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
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
        resolutions = sorted(latents_per_res.keys())
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
            geo_cache={r: (gt, gm, gc)
                       for r, (gt, gm, gc, _) in geo_cache.items()} if mae_active else None,
            masked_indices_per_res=masked_indices_per_res if mae_active else None,
        )

    # =========================================================================
    # Decoder
    # =========================================================================

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None,
                    training=True, return_features=False):
        """
        Decode query pixels into class logits.

        Pipeline:
            1. Select k nearest latents per query pixel
            2. Compute relative displacement (query → latent)
            3. Build context = cat([latent_features | rel_pe])
            4. LocalCrossAttentionRoPE: Q=query_features, K/V=context
            5. output_head(cat([cross_attn_out | query_features])) → logits

        Shapes:
            query_features:  [B, M, query_dim_recon]
            context:         [B*M, k, latent_dim + pe_dim]
            output logits:   [B, M, num_classes]
        """
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        # ── Query features & coords ───────────────────────────────────
        query_features, _, _ = self.input_processor.process_data_for_decoder(
            query_tokens, query_mask, target_resolution=target_resolution)
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)
        # query_features: [B, M, query_dim_recon]
        # query_coords:   [B, M, 2]

        # ── Concat all latents across resolutions ─────────────────────
        all_latents = torch.cat(
            [latents_per_res[r] for r in sorted(latents_per_res.keys())], dim=1)
        all_coords = torch.cat(
            [coords_per_res[r] for r in sorted(coords_per_res.keys())], dim=1)
        # all_latents: [B, L, latent_dim]
        # all_coords:  [B, L, 2]

        # ── Select k nearest latents per query ────────────────────────
        dists_sq = (query_coords.unsqueeze(2) - all_coords.unsqueeze(1)).pow(2).sum(-1)

        k_fetch = min(k + 1, all_coords.shape[1]) if training else min(k, all_coords.shape[1])
        k_keep  = min(k, k_fetch)

        _, topk_indices = torch.topk(dists_sq, k=k_fetch, dim=-1, largest=False)

        if training and k_fetch > k_keep:
            drop_idx  = torch.randint(0, k_fetch, (B, M, 1), device=all_coords.device)
            keep_mask = torch.ones(B, M, k_fetch, dtype=torch.bool, device=all_coords.device)
            keep_mask.scatter_(2, drop_idx, False)
            topk_indices = topk_indices[keep_mask].reshape(B, M, k_keep)

        D = all_latents.shape[-1]
        flat_idx = topk_indices.reshape(B, M * k_keep)

        selected_latents = torch.gather(
            all_latents, 1,
            flat_idx.unsqueeze(-1).expand(-1, -1, D)
        ).reshape(B, M, k_keep, D)
        # selected_latents: [B, M, k, latent_dim]

        selected_coords = torch.gather(
            all_coords, 1,
            flat_idx.unsqueeze(-1).expand(-1, -1, 2)
        ).reshape(B, M, k_keep, 2)
        # selected_coords: [B, M, k, 2]

        # ── Relative displacement query → latent ──────────────────────
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)
        # delta_x, delta_y: [B, M, k]

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

        # ── Build context = [latent | rel_pe] ─────────────────────────
        context = torch.cat([selected_latents, rel_pe], dim=-1)
        # context: [B, M, k, latent_dim + pe_dim]

        # ── Cross-attention: Q=query_features, K/V=context ───────────
        # LocalCrossAttentionRoPE expects:
        #   queries: [B, L, query_dim]          → [B*M, 1, query_dim]
        #   context: [B, L, m, ctx_dim]         → [B*M, 1, k, ctx_dim]
        #   delta_x: [B, L, m]                  → [B*M, 1, k]
        q_flat       = query_features.reshape(B * M, 1, -1)           # [B*M, 1, query_dim]
        ctx_flat     = context.reshape(B * M, 1, k_keep, -1)          # [B*M, 1, k, ctx_dim]
        dx_flat_attn = delta_x.reshape(B * M, 1, k_keep)              # [B*M, 1, k]
        dy_flat_attn = delta_y.reshape(B * M, 1, k_keep)              # [B*M, 1, k]

        attn_out = self.decoder_cross_attn(
            q_flat,
            context=ctx_flat,
            delta_x=dx_flat_attn,
            delta_y=dy_flat_attn,
        )
        # attn_out: [B*M, 1, latent_dim]
        attn_out = attn_out.squeeze(1).reshape(B, M, -1)              # [B, M, latent_dim]

        if return_features:
            return attn_out

        # ── Output head: [cross_attn_out | query_features] → logits ──
        output = torch.cat([attn_out, query_features], dim=-1)       # [B, M, latent_dim + query_dim]
        return self.reconstruction_head(output)                       # [B, M, num_classes]

    def classify(self, latents_per_res):
        all_latents = torch.cat(
            [latents_per_res[res] for res in sorted(latents_per_res.keys())], dim=1)
        return self.to_logits(all_latents)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, batch, training=True, task="reconstruction",
                return_trajectory=False, return_predicted_errors=False,
                return_features=False, tokens_per_latent_override=None,
                mask_ratio: float = 0.0):

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
            chunk_size = 10_000
            N = queries.shape[1]
            if N > chunk_size:
                preds = []
                for i in range(0, N, chunk_size):
                    preds.append(self.reconstruct(
                        latents_per_res, coords_per_res,
                        queries[:, i:i+chunk_size],
                        queries_mask[:, i:i+chunk_size],
                        target_resolution=target_resolution,
                        training=training, return_features=return_features))
                output = torch.cat(preds, dim=1)
            else:
                output = self.reconstruct(
                    latents_per_res, coords_per_res,
                    queries, queries_mask,
                    target_resolution=target_resolution,
                    training=training, return_features=return_features)

            if return_features:
                return {"features": output, "latents_per_res": latents_per_res,
                        "coords_per_res": coords_per_res, "encoder_output": encoder_output}

            if task == "visualization" or return_predicted_errors:
                return {'predictions': output, 'latents_per_res': latents_per_res,
                        'coords_per_res': coords_per_res, 'trajectory': trajectory,
                        'predicted_errors': None}
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
        self._set_requires_grad(self.neighbor_cross_attn, False)
        self.decoder_temperature.requires_grad = False
        self._set_requires_grad(self.grid_gate, False)
        self._set_requires_grad(self.post_fusion, False)
        self._set_requires_grad(self.reconstruction_head, False)

    def unfreeze_decoder(self):
        self._set_requires_grad(self.local_predictor, True)
        self._set_requires_grad(self.neighbor_cross_attn, True)
        self.decoder_temperature.requires_grad = True
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