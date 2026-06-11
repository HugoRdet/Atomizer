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
helpers) is inherited unchanged. The new pieces in Atomiser_Fractal are:
    1. z_query_projection: MLP mapping z_features → latent_dim
    2. decoder_cross_attn: REPLACED with a QK-normalized variant (see below)


z_query_projection: MLP rather than single linear
--------------------------------------------------
The original implementation used a single nn.Linear for z_query_projection.
Diagnostic analysis after early-run divergence showed this layer's bias
inflated 2.89x (the largest growth of any parameter in the model, even
larger than the decoder Q projection at 2.28x). A single linear cannot
express enough nonlinear functions of z, so the model compensates by
inflating weights to push z_features through magnitude scaling.

We replace it with the same MLP structure used by the parent class's
decoder_mlp (build_mlp factory): 2 layers, GELU activation, LayerNorm.
This provides genuine nonlinear capacity and stable output magnitudes,
matching the architectural pattern already established in TokenProcessor.

The final layer is zero-initialized so the model starts behaving like
Atomiser_Senflood (Q ≈ global_query at init) and gradually learns to
incorporate z as training progresses.


Decoder cross-attention QK-normalization
-----------------------------------------
After observing training divergence around epoch 12 in early FRACTAL runs,
we ran a per-parameter diagnostic across checkpoints. The decoder
cross-attention's Q projection inflated 2.28x in max_abs between epoch 9
(healthy, val_mIoU=0.72) and epoch 14 (collapsed, val_mIoU=0.11). All other
attention layers in the model — encoder cross-attention, encoder
self-attention (4 blocks), classifier head — remained stable (<1.3x over
the same window).

The diagnosed failure mode is attention entropy collapse [ViT-22B, Gemma 2]:
unbounded Q/K magnitudes saturate the softmax, gradients vanish through the
saturated attention, and the model freezes in a degenerate predict-one-class
state. The decoder is the load-bearing site because it has the largest
attention matrix in the model (millions of pixel/point queries against the
latent set).

To prevent this, we replace the inherited `decoder_cross_attn` (a standard
nn.MultiheadAttention) with `QKNormMultiheadAttention`, which applies
RMSNorm to queries and keys per-head before computing attention scores.

Encoder attention layers are NOT modified — the diagnostic showed they are
stable, and over-normalizing them would unnecessarily restrict the encoder's
representational flexibility.

The MLP-based z_projection and QK-norm are complementary: the MLP reduces
the pressure that causes attention magnitudes to inflate in the first place,
while QK-norm provides defense-in-depth against any residual instability.
"""

import torch
import torch.nn as nn
from einops import repeat

from .Atomiser_SENFLOOD import Atomiser_Senflood, EncoderOutput
from .QK_norm_attention import QKNormMultiheadAttention
from training.utils.token_building.processor import build_mlp


# Token column indices — must match TokenProcessor / TokenBuilder.
TOKEN_VALUE_IDX = 0


class Atomiser_Fractal(Atomiser_Senflood):
    """
    Atomiser variant for FRACTAL.

    Inherits all encoder/decoder/refinement logic from Atomiser_Senflood.
    Adds:
        - z_query_projection (MLP): per-pixel z conditioning for queries
        - decoder_cross_attn (QK-normalized): prevents attention collapse
    """

    def __init__(self, *, config, lookup_table):
        super().__init__(config=config, lookup_table=lookup_table)

        # ── z-aware query projection (MLP) ────────────────────────────
        # Builds an MLP matching the structure used by the parent's
        # decoder_mlp: 2 layers, GELU + LayerNorm, hidden=tokenizer_hidden.
        # This gives z_projection real nonlinear capacity instead of just
        # a linear remap of Fourier features.
        z_feature_dim    = self.input_processor.reflectance_encoder.out_dim
        tokenizer_hidden = 128 #config["Atomiser"]["tokenizer_hidden_size"]
        tokenizer_layers = config["Atomiser"]["tokenizer_nb_layers"]

        self.z_query_projection = build_mlp(
            in_dim=z_feature_dim,
            hidden_dim=tokenizer_hidden,
            out_dim=self.latent_dim,
            num_layers=tokenizer_layers,
        )

        # Zero-init the FINAL layer so z_proj starts at zero. This way
        # per_pixel_q = global_query + z_proj ≈ global_query at init,
        # matching Senflood's behavior. The model gradually learns to
        # incorporate z as training progresses.
        #
        # We assume the final layer is the last nn.Linear in the
        # Sequential. build_mlp's structure puts Linear last by design
        # (see token_processor.build_mlp).
        last_linear = None
        for module in self.z_query_projection:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        print(f"[Atomiser_Fractal] z-aware query projection: "
              f"reflectance_encoder({z_feature_dim}) → "
              f"MLP({tokenizer_layers}L, h={tokenizer_hidden}) → "
              f"latent_dim({self.latent_dim})  "
              f"[final layer zero-initialized]")
        print(f"[Atomiser_Fractal]   Q per pixel = global_query + "
              f"z_projection(reflectance_encoder(z))")

        # ── Replace decoder cross-attention with QK-normalized variant ────
        # The parent class (Atomiser_Senflood) instantiated a standard
        # nn.MultiheadAttention for self.decoder_cross_attn. We replace it
        # here with a QK-normalized version. Same constructor args, same
        # forward signature — no other code changes needed.
        self.decoder_cross_attn = QKNormMultiheadAttention(
            embed_dim=self.latent_dim,
            kdim=self.decoder_context_dim,
            vdim=self.decoder_context_dim,
            num_heads=self.cross_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        print(f"[Atomiser_Fractal] decoder_cross_attn: "
              f"QK-normalized (RMSNorm per-head on Q and K) "
              f"— prevents attention entropy collapse")

    # =========================================================================
    # Decoder override: z-aware query
    # =========================================================================

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None,
                    training=True, return_features=False,
                    return_topk=False):
        """
        FRACTAL-specific decoder. Identical to Atomiser_Senflood.reconstruct
        except Q is derived per-pixel from (global_query + z_projection),
        and the decoder cross-attention internally normalizes Q and K via
        RMSNorm per-head.

        Steps:
            1. Select k nearest latents per query pixel    (same as parent)
            2. Compute rel_pe                              (same as parent)
            3. context = cat([latent, rel_pe])             (same as parent)
            4. Bernoulli drop                              (same as parent)
            5. Build per-pixel Q from global_query + MLP(z) (NEW: MLP-based)
            6. Cross-attention with per-pixel Q            (NOTE: now uses
               QKNormMultiheadAttention internally; call signature is
               identical to nn.MultiheadAttention)
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
        # This makes Q a per-pixel function of z while preserving the
        # learned global query as the base. Final MLP layer is zero-initialized,
        # so z_proj starts at zero and per_pixel_q ≈ global_query at init.
        z_proj = self.z_query_projection(z_features)                     # [B, M, latent_dim]
        global_q = self.global_query.expand(B, M, -1)                    # [B, M, latent_dim]
        per_pixel_q = global_q + z_proj                                   # [B, M, latent_dim]

        # ── Cross-attention with per-pixel Q ──────────────────────────
        # Note: self.decoder_cross_attn is now a QKNormMultiheadAttention
        # (set in __init__). Same interface as nn.MultiheadAttention so the
        # call below is identical — but Q and K are RMSNorm'd inside the
        # attention before computing scores.
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
