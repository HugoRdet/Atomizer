"""
Atomiser_Dales -- Subclass for LIDAR-only semantic segmentation (DALES)
==========================================================================

Now based on Atomiser_Senflood_Skip (not the plain Atomiser_Senflood),
adding THREE substitutions/combinations on top of it:

1. Token processor: replaces the parent's TokenProcessor with
   DalesTokenProcessor (echo + intensity routing) -- see
   dales_token_processor.py.

2. Geographic pruning: replaces the parent's (shared-batch) GeographicPruning
   with the DALES-specific variant from geographic_pruning_dales.py.

   AUTHORITATIVE path: token_latent_assignment (precomputed offline by
   precompute_dales_latent_assignment.py, one nearest-latent index per
   token per D4 grid variant, loaded/selected by DalesDataset and passed
   through the batch) -- no distance computation happens at train time at
   all. `_apply_pruning`, `encode`, and `forward` are overridden to plumb
   `batch["token_latent_assignment"]` (and, as fallback, `batch["patch_id"]`)
   down to the pruning call -- see geographic_pruning_dales.py's docstring
   for the staleness caveat the fallback path carries.

3. Decoder query construction: COMBINES the parent's decoder-skip cascade
   (pixel_query attends over the query-pixel's own encoded tokens, gated
   by config["Atomiser"]["use_decoder_skip"]) with Atomiser_Fractal's
   z-aware query projection. DALES has the same "points sharing (x, y) but
   differing in z" issue FRACTAL solves for -- e.g. a power line point
   above a ground point, a car roof above the road, a building eave above
   a sidewalk point.

   The two mechanisms are ADDITIVE, not exclusive:
       base_q       = enriched_skip_query   (if use_decoder_skip, else global_query)
       per_pixel_q  = base_q + z_projection(reflectance_encoder(z))

   So with use_decoder_skip enabled, a query pixel's decoder input reflects
   BOTH its own local atoms (via the skip cascade) AND its z value (via the
   z-projection) -- the skip cascade alone can't distinguish two points at
   the same (x,y) if it only sees position-keyed atoms with no z signal of
   its own, so the z-projection remains additive value even with skip on.

   DalesDataset already writes elevation into query column 0
   (`queries[:, 0] = values_query`), matching the expected convention.

Everything else (encoder, refinement, error predictor, classifier, freeze
helpers, adaptive/quadtree decode branches) is inherited unchanged from
Atomiser_Senflood_Skip.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.profiler import record_function

from .Atomiser_senflood_skip import Atomiser_Senflood_Skip, EncoderOutput
from training.utils.token_building.processor import build_mlp
from training.utils.token_building.DALES_token_processor import DalesTokenProcessor
from training.utils.datasets.token_grouping import compute_grid_config

from .geographic_pruning_dales import GeographicPruning as GeographicPruningDales


TOKEN_VALUE_IDX = 0


class Atomiser_Dales(Atomiser_Senflood_Skip):
    """
    Atomiser variant for DALES, with decoder-skip cascade enabled.
    """

    def __init__(self, *, config, lookup_table):
        super().__init__(config=config, lookup_table=lookup_table)

        self.input_processor = DalesTokenProcessor(config, lookup_table)

        self.geo_pruning = GeographicPruningDales(
            geometry=self.input_processor.geometry,
        )
        self.zone_probe.geometry = self.input_processor.geometry

        print(f"[Atomiser_Dales] input_processor replaced with "
              f"DalesTokenProcessor (echo + intensity aware routing)")
        print(f"[Atomiser_Dales] geo_pruning replaced with "
              f"GeographicPruningDales (precompute-consuming / "
              f"per-sample fallback)")
        print(f"[Atomiser_Dales] decoder-skip cascade: "
              f"{'ENABLED' if self.use_decoder_skip else 'DISABLED'} "
              f"(inherited from Atomiser_Senflood_Skip config)")

        z_feature_dim    = self.input_processor.reflectance_encoder.out_dim
        z_hidden         = 128
        tokenizer_layers = config["Atomiser"]["tokenizer_nb_layers"]

        self.z_query_projection = build_mlp(
            in_dim=z_feature_dim,
            hidden_dim=z_hidden,
            out_dim=self.latent_dim,
            num_layers=tokenizer_layers,
        )

        last_linear = None
        for module in self.z_query_projection:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        print(f"[Atomiser_Dales] z-aware query projection: "
              f"reflectance_encoder({z_feature_dim}) -> "
              f"MLP({tokenizer_layers}L, h={z_hidden}) -> "
              f"latent_dim({self.latent_dim})  "
              f"[final layer zero-initialized]")
        print(f"[Atomiser_Dales]   Q per pixel = "
              f"{'skip_enriched_query' if self.use_decoder_skip else 'global_query'} "
              f"+ z_projection(reflectance_encoder(z))")

    def _apply_pruning(self, tokens, mask, coords, grid_config, L_spatial,
                       token_latent_assignment=None, patch_ids=None):
        geo_tokens, geo_masks, _ = self.geo_pruning(
            tokens, mask, coords,
            geo_k=grid_config["geo_k"],
            sigma=grid_config["geo_sigma"],
            L_spatial=L_spatial,
            hexagonal=grid_config.get("hexagonal", False),
            token_latent_assignment=token_latent_assignment,
            patch_ids=patch_ids,
        )
        return geo_tokens, geo_masks

    def encode(self, groups, grid_configs, training=True,
               return_trajectory=False, mask_ratio: float = 0.0,
               cross_k: int = 1024, token_latent_assignment=None,
               patch_ids=None):
        first_group = next(iter(groups.values()))
        B      = first_group["tokens"].shape[0]
        device = first_group["tokens"].device
        resolutions = sorted(groups.keys())

        with record_function("Latents init "):
            latents_per_res, coords_per_res = self.init_latents_per_resolution(
                B, grid_configs, device)
            global_latents = self.get_global_latents(B)

        geo_cache = {}

        with record_function("geo pruning"):
            for res in resolutions:
                tokens    = groups[res]["tokens"]
                mask      = groups[res]["mask"]
                gc        = dict(grid_configs[res])
                coords    = coords_per_res[res]
                L_spatial = gc["L_spatial"]

                geo_tokens, geo_masks = self._apply_pruning(
                    tokens, mask, coords, gc, L_spatial,
                    token_latent_assignment=token_latent_assignment,
                    patch_ids=patch_ids)
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
                with record_function("Cross Attention - Encoder"):
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

            with record_function("Self Attention"):
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

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None,
                    training=True, return_features=False,
                    return_topk=False,
                    query_token_idx=None, query_token_valid=None,
                    pool_tokens=None, pool_mask=None):
        B, M, _ = query_tokens.shape
        k = self.decoder_k_spatial

        with record_function("Decoder pre processing"):
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

        with record_function("Decoder Cross attention"):

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

            with record_function("Decoder skip"):
                if self.use_decoder_skip and query_token_idx is not None:
                    base_q = self._pixel_skip(
                        query_tokens, query_token_idx, query_token_valid,
                        pool_tokens, pool_mask, training)
                else:
                    base_q = self.global_query.expand(B, M, -1)

                z_values = query_tokens[..., TOKEN_VALUE_IDX]
                z_features = self.input_processor.reflectance_encoder(z_values)
                if z_features.dim() == 4 and z_features.shape[-2] == 1:
                    z_features = z_features.squeeze(-2)
                z_proj = self.z_query_projection(z_features)

                per_pixel_q = base_q + z_proj
                q_flat = per_pixel_q.reshape(BM, 1, -1).contiguous()
                q_flat = self.dec_q_norm(q_flat)

            key_pad_flat = (~keep_mask).reshape(BM, k_keep)

            with record_function("Decoder Cross Attention"):

                attn_out, _ = self.decoder_cross_attn(
                    query=q_flat, key=kv_flat, value=kv_flat,
                    key_padding_mask=key_pad_flat,
                    need_weights=False,
                )
                attn_out = attn_out.squeeze(1).reshape(B, M, -1)

                if return_features:
                    return attn_out

                with record_function("Decoder Logits"):
                    logits = self.reconstruction_head(attn_out)

                if return_topk:
                    topk_dists_sq_kept = torch.gather(dists_sq, 2, topk_indices)
                    return logits, topk_indices, topk_dists_sq_kept

                return logits

    def forward(self, batch, training=True, task="reconstruction",
                return_trajectory=False, return_predicted_errors=False,
                return_features=False, tokens_per_latent_override=None,
                mask_ratio: float = 0.0, return_for_error=False):

        groups       = batch["groups"]
        queries      = batch["queries"]
        queries_mask = batch["queries_mask"]
        target_resolution = batch.get("target_resolution", None)

        query_token_idx   = batch.get("query_token_idx", None)
        query_token_valid = batch.get("query_token_valid", None)
        skip_pool_tokens = None
        skip_pool_mask   = None
        if self.use_decoder_skip and query_token_idx is not None:
            skip_res = self.config["Atomiser"].get("skip_resolution", 10.0)
            if skip_res not in groups:
                skip_res = min(groups.keys())
            skip_pool_tokens = groups[skip_res]["tokens"]
            skip_pool_mask   = groups[skip_res]["mask"]

        token_latent_assignment = batch.get("token_latent_assignment", None)
        patch_ids = batch.get("patch_id", None)

        if tokens_per_latent_override is not None:
            tpl = tokens_per_latent_override
            batch_cross_k = self.val_sampling[0][1]
        else:
            tpl, batch_cross_k = self.sample_config(training)

        resolutions   = sorted(groups.keys())
        geo_k_budget  = batch_cross_k * 2

        with record_function("Compute grid config"):
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

        with record_function("encode"):
            encoder_output = self.encode(
                groups=groups, grid_configs=grid_configs,
                training=training, return_trajectory=need_trajectory,
                mask_ratio=mask_ratio, cross_k=batch_cross_k,
                token_latent_assignment=token_latent_assignment,
                patch_ids=patch_ids)

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

            if getattr(self, "use_quadtree_decode", False) and not training:
                with record_function("Quadtree decode branch"):
                    output = self._reconstruct_quadtree(
                        latents_per_res, coords_per_res, batch)
                if task == "visualization" or return_predicted_errors:
                    return {'predictions': output,
                            'latents_per_res': latents_per_res,
                            'coords_per_res': coords_per_res,
                            'trajectory': trajectory,
                            'predicted_errors': predicted_errors}
                return output

            if getattr(self, "use_adaptive_decode", False) and not training:
                with record_function("Adaptive decode"):
                    output = self._reconstruct_adaptive(
                        latents_per_res, coords_per_res, batch)
                if task == "visualization" or return_predicted_errors:
                    return {'predictions': output,
                            'latents_per_res': latents_per_res,
                            'coords_per_res': coords_per_res,
                            'trajectory': trajectory,
                            'predicted_errors': predicted_errors}
                return output

            chunk_size = 10_000
            N = queries.shape[1]
            need_topk = False

            if N > chunk_size:
                preds_list      = []
                topk_idx_list   = []
                topk_dists_list = []
                for i in range(0, N, chunk_size):
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

    def freeze_decoder(self):
        super().freeze_decoder()
        self._set_requires_grad(self.z_query_projection, False)

    def unfreeze_decoder(self):
        super().unfreeze_decoder()
        self._set_requires_grad(self.z_query_projection, True)
