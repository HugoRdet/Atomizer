"""
Atomiser_Fractal — Subclass for LIDAR + VHR semantic segmentation
==================================================================

Differs from Atomiser_Senflood in two places:

1. Token processor: replaces the parent's TokenProcessor with
   FractalTokenProcessor, which routes col 7 of each token through the
   correct encoder — echo metadata for LIDAR tokens, time for everything
   else. The parent treats col 7 as time_idx unconditionally, which
   produces meaningless temporal embeddings for LIDAR tokens.

2. Decoder query construction: the parent uses a single learned global
   query broadcast to all pixels:

       Q[b, m] = global_query

   This works when every query is fully characterized by its (x, y)
   position — rel_pe on the K/V side lets the model distinguish queries
   spatially. FRACTAL breaks this assumption: multiple LIDAR points share
   the same (x, y) but differ in z (canopy above ground, bridge deck above
   road, eaves above sidewalk). With a position-only query they all get
   the same prediction.

   FRACTAL builds a per-pixel query that depends on z:

       Q[b, m] = global_query + z_projection(reflectance_encoder(z[b, m]))

   where z is stored in col 0 of each query token by FractalDataset
   (ground-relative-clipped, scaled to roughly [-1, 2]).


z_query_projection: MLP rather than single linear
--------------------------------------------------
The original implementation used a single nn.Linear for z_query_projection.
We replace it with a small MLP (build_mlp factory): 2 layers, GELU + LayerNorm,
hidden=128. This provides:
  - Nonlinear capacity to map Fourier-encoded z features to a 768-dim offset
  - LayerNorm in the middle for stable output magnitudes (the original
    single Linear's bias inflated 2.89x during training, the largest growth
    of any parameter in the model, because it had to encode both direction
    and scale of the desired output without normalization to control magnitude)

The final layer is zero-initialized so z_proj starts at zero. The model
starts behaving like Senflood (Q ≈ global_query at init) and gradually
learns to incorporate z as training progresses.

Everything else (encoder, refinement, error predictor, classifier, freeze
helpers) is inherited unchanged.
"""

import torch
import torch.nn as nn
from einops import repeat

from .Atomiser_SENFLOOD import Atomiser_Senflood, EncoderOutput
from training.utils.token_building.processor import build_mlp
from training.utils.token_building.fractal_token_processor import FractalTokenProcessor


# Token column indices — must match TokenProcessor / TokenBuilder.
TOKEN_VALUE_IDX = 0


class Atomiser_Fractal(Atomiser_Senflood):
    """
    Atomiser variant for FRACTAL.

    Inherits all encoder/decoder/refinement logic from Atomiser_Senflood.
    Adds:
        - input_processor: replaced with FractalTokenProcessor (echo routing)
        - z_query_projection (MLP): per-pixel z conditioning for queries
    """

    def __init__(self, *, config, lookup_table):
        super().__init__(config=config, lookup_table=lookup_table)

        # ── Replace input_processor with FRACTAL-aware version ─────────
        # The parent class instantiated a vanilla TokenProcessor. We replace
        # it with FractalTokenProcessor which detects LIDAR tokens via
        # spectral_idx and routes col 7 through an echo encoder instead of
        # the time encoder.
        #
        # The replacement preserves all output dims (encoder/decoder MLP
        # output dims are identical) so the parent's pre-computed
        # self.input_dim, self.query_dim_recon, self.decoder_pe_dim etc.
        # remain valid. Only the temporal feature *content* changes for
        # LIDAR tokens.
        #
        # Sub-components that other parts of Atomiser_Senflood reference
        # via self.input_processor.* (e.g., geometry, pos_encoder,
        # reflectance_encoder) all still exist with the same names because
        # FractalTokenProcessor inherits from TokenProcessor.
        self.input_processor = FractalTokenProcessor(config, lookup_table)

        # GeographicPruning was initialized with the OLD input_processor's
        # geometry instance. Re-wire it to the new one so they stay in sync.
        # (Both geometries are identical in behavior — they're built from
        # the same config and lookup table — but keeping a single reference
        # avoids confusion when the model is serialized.)
        self.geo_pruning.geometry = self.input_processor.geometry

        print(f"[Atomiser_Fractal] input_processor replaced with "
              f"FractalTokenProcessor (echo-aware col-7 routing)")

        # ── z-aware query projection (MLP) ─────────────────────────────
        # The reflectance encoder produces a Fourier embedding of the
        # query's z value. We project that into latent_dim through a small
        # MLP and ADD it to the global query.
        #
        # An MLP (not a single Linear) is used so that:
        #   - The LayerNorm in the middle stabilizes output magnitudes
        #   - GELU provides nonlinear capacity over the Fourier features
        # hidden_dim=128 is intentionally smaller than the encoder MLPs
        # (768) because the input is a single scalar — there's no
        # justification for the same capacity as 600-dim feature inputs.
        z_feature_dim    = self.input_processor.reflectance_encoder.out_dim
        z_hidden         = 128
        tokenizer_layers = config["Atomiser"]["tokenizer_nb_layers"]

        self.z_query_projection = build_mlp(
            in_dim=z_feature_dim,
            hidden_dim=z_hidden,
            out_dim=self.latent_dim,
            num_layers=tokenizer_layers,
        )

        # Zero-init the FINAL Linear layer so z_proj starts at zero.
        # per_pixel_q = global_query + z_proj ≈ global_query at init,
        # matching Senflood's behavior. The model gradually learns to
        # incorporate z as training progresses.
        last_linear = None
        for module in self.z_query_projection:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        print(f"[Atomiser_Fractal] z-aware query projection: "
              f"reflectance_encoder({z_feature_dim}) → "
              f"MLP({tokenizer_layers}L, h={z_hidden}) → "
              f"latent_dim({self.latent_dim})  "
              f"[final layer zero-initialized]")
        print(f"[Atomiser_Fractal]   Q per pixel = global_query + "
              f"z_projection(reflectance_encoder(z))")

    # =========================================================================
    # Decoder override: z-aware query
    # =========================================================================

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None,
                    training=True, return_features=False,
                    return_topk=False):
        """
        FRACTAL-specific decoder. Identical to Atomiser_Senflood.reconstruct
        except Q is derived per-pixel from (global_query + z_projection).

        Steps:
            1. Select k nearest latents per query pixel    (same as parent)
            2. Compute rel_pe                              (same as parent)
            3. context = cat([latent, rel_pe])             (same as parent)
            4. Bernoulli drop                              (same as parent)
            5. Build per-pixel Q from global_query + MLP(z) (NEW)
            6. Cross-attention with per-pixel Q            (modified)
            7. Segmentation head                           (same as parent)
        """
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        # ── Query coords (for k-nearest + rel_pe) ─────────────────────
        # geometry.get_token_centers reads cols 1-2 (x, y) and ignores col 0.
        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)

        # ── Concat all latents across resolutions ─────────────────────
        all_latents = torch.cat(
            [latents_per_res[r] for r in sorted(latents_per_res.keys(), key=str)],
            dim=1,
        )
        all_coords = torch.cat(
            [coords_per_res[r] for r in sorted(coords_per_res.keys(), key=str)],
            dim=1,
        )

        # ── Select k nearest latents per query ────────────────────────
        dists_sq = (
            (query_coords.unsqueeze(2) - all_coords.unsqueeze(1)).pow(2).sum(-1)
        )

        k_fetch = min(k, all_coords.shape[1])
        k_keep  = k_fetch
        _, topk_indices = torch.topk(dists_sq, k=k_fetch, dim=-1, largest=False)

        D = all_latents.shape[-1]
        flat_idx = topk_indices.reshape(B, M * k_keep)

        selected_latents = torch.gather(
            all_latents, 1, flat_idx.unsqueeze(-1).expand(-1, -1, D)
        ).reshape(B, M, k_keep, D)
        selected_coords = torch.gather(
            all_coords, 1, flat_idx.unsqueeze(-1).expand(-1, -1, 2)
        ).reshape(B, M, k_keep, 2)

        # ── Relative displacement query → latent ──────────────────────
        delta_x = selected_coords[..., 0] - query_coords[..., 0].unsqueeze(-1)
        delta_y = selected_coords[..., 1] - query_coords[..., 1].unsqueeze(-1)

        # ── Relative positional encoding ──────────────────────────────
        B_d, M_d, K_d = delta_x.shape
        if self.input_processor.use_constant_gsd:
            cs = (self.input_processor.compression_alpha
                  * self.input_processor._constant_gsd)
        else:
            query_gsd = self.input_processor.geometry.get_token_gsd(query_tokens)
            cs = self.input_processor.compression_alpha * query_gsd

        dx_flat = delta_x.reshape(B_d, M_d * K_d)
        dy_flat = delta_y.reshape(B_d, M_d * K_d)
        rel_pe  = self.input_processor.pos_encoder(
            dx_flat, dy_flat, compression_scale=cs
        )
        rel_pe = rel_pe.reshape(B_d, M_d, K_d, -1)

        # ── Build context = cat([latent, rel_pe]) ─────────────────────
        context = torch.cat([selected_latents, rel_pe], dim=-1)

        # ── Bernoulli drop (training only) ────────────────────────────
        if training and self.decoder_drop_p > 0:
            keep_probs = torch.full(
                (B, M, k_keep), 1.0 - self.decoder_drop_p,
                device=context.device,
            )
            keep_mask = torch.bernoulli(keep_probs).bool()
            none_kept = ~keep_mask.any(dim=-1, keepdim=True)
            if none_kept.any():
                keep_mask = keep_mask.clone()
                keep_mask[..., 0] = keep_mask[..., 0] | none_kept.squeeze(-1)
        else:
            keep_mask = torch.ones(
                B, M, k_keep, dtype=torch.bool, device=context.device,
            )

        # ─────────────────────────────────────────────────────────────
        # Build per-pixel Q from (global_query + z_projection)
        # ─────────────────────────────────────────────────────────────
        # Read z (normalized to roughly [-1, 2]) from query col 0.
        # FractalDataset writes ground-relative-clipped z into this column
        # for real queries; padding queries have z=0 (and label=IGNORE_INDEX
        # so they don't contribute to loss).
        z_values = query_tokens[..., TOKEN_VALUE_IDX]                   # [B, M]

        # Encode z via the reflectance encoder. This is the same Fourier
        # feature encoder used for input token values on the encoder side
        # — consistent encoding between encoder and decoder paths.
        z_features = self.input_processor.reflectance_encoder(z_values)
        # reflectance_encoder typically returns [B, M, 1, feat_dim] for
        # batched scalar inputs; squeeze the singleton dim if present.
        if z_features.dim() == 4 and z_features.shape[-2] == 1:
            z_features = z_features.squeeze(-2)                          # [B, M, feat_dim]

        # Project z features into latent_dim space via MLP, add to global query.
        # Final MLP layer is zero-initialized so z_proj starts at zero and
        # per_pixel_q ≈ global_query at init.
        z_proj = self.z_query_projection(z_features)                     # [B, M, latent_dim]
        global_q = self.global_query.expand(B, M, -1)                    # [B, M, latent_dim]
        per_pixel_q = global_q + z_proj                                  # [B, M, latent_dim]

        # ── Cross-attention with per-pixel Q ──────────────────────────
        BM = B * M
        kv_flat = context.reshape(BM, k_keep, -1)
        q_flat  = per_pixel_q.reshape(BM, 1, -1).contiguous()
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
    # Freeze / unfreeze: include z_query_projection
    # =========================================================================

    def freeze_decoder(self):
        super().freeze_decoder()
        self._set_requires_grad(self.z_query_projection, False)

    def unfreeze_decoder(self):
        super().unfreeze_decoder()
        self._set_requires_grad(self.z_query_projection, True)
