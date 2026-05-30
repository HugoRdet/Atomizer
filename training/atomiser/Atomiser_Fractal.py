"""
Atomiser_Fractal — Subclass for LIDAR + VHR semantic segmentation
==================================================================

The only difference from Atomiser_Senflood is in the decoder's query
construction. Atomiser_Senflood uses a single learned global query vector
broadcast to all pixels:

    Q[b, m] = global_query  (same for every pixel)

This works when every query is fully characterized by its (x, y) position
— the K/V side via rel_pe lets the model distinguish queries spatially.

FRACTAL breaks this assumption: multiple LIDAR points share the same (x, y)
but differ in z (tree canopy above ground, bridge deck above road, building
eaves above sidewalk). With a position-only query, all such points get the
same prediction.

Atomiser_Fractal modifies Q to depend on the query's z value:

    Q[b, m] = global_query + z_projection(reflectance_encoder(z[b, m]))

The z value is stored in col 0 of each query token by FractalDataset, after
ground-relative normalization (z minus local ground median, clipped to
[-15, 30], scaled by 1/15 → roughly [-1, 2]).

Everything else (encoder, refinement, error predictor, classifier, freeze
helpers) is inherited unchanged. The only new parameter is z_query_projection.
"""

import torch
import torch.nn as nn
from einops import repeat

from .Atomiser_SENFLOOD import Atomiser_Senflood, EncoderOutput


# Token column indices — must match TokenProcessor / TokenBuilder.
TOKEN_VALUE_IDX = 0


class Atomiser_Fractal(Atomiser_Senflood):
    """
    Atomiser variant for FRACTAL.

    Inherits all encoder/decoder/refinement logic from Atomiser_Senflood.
    Adds one z-aware projection in the decoder that lets queries at the
    same (x, y) but different z produce different predictions.
    """

    def __init__(self, *, config, lookup_table):
        super().__init__(config=config, lookup_table=lookup_table)

        # ── z-aware query projection ─────────────────────────────────
        # The reflectance encoder produces a Fourier embedding of the
        # query's z value. We project that into latent_dim and ADD it to
        # the global query (additive rather than concat keeps the
        # cross-attention dim unchanged: latent_dim).
        z_feature_dim = self.input_processor.reflectance_encoder.out_dim
        self.z_query_projection = nn.Linear(z_feature_dim, self.latent_dim)

        # Initialize the projection to small values so the model starts
        # behaving like Atomiser_Senflood (Q ≈ global_query at init) and
        # gradually learns to incorporate z.
        nn.init.trunc_normal_(
            self.z_query_projection.weight, std=0.02, a=-2., b=2.
        )
        nn.init.zeros_(self.z_query_projection.bias)

        print(f"[Atomiser_Fractal] z-aware query projection: "
              f"reflectance_encoder({z_feature_dim}) → "
              f"linear → latent_dim({self.latent_dim})")
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
            5. Build per-pixel Q from global_query + z     (NEW)
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
        # NEW: Build per-pixel Q from (global_query + z_projection)
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

        # Project z features into latent_dim space, add to global query.
        # This makes Q a per-pixel function of z while preserving the
        # learned global query as the base.
        z_proj = self.z_query_projection(z_features)                     # [B, M, latent_dim]
        global_q = self.global_query.expand(B, M, -1)                    # [B, M, latent_dim]
        per_pixel_q = global_q + z_proj                                   # [B, M, latent_dim]

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
