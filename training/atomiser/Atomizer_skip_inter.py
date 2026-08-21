"""
Atomiser Model (SKIP variant) — Multi-Resolution Encoder/Decoder
================================================================

Identical to Atomiser_Senflood EXCEPT for a decoder PIXEL-SKIP CASCADE,
gated behind config["Atomiser"]["use_decoder_skip"], plus an inference-only
ADAPTIVE probe-and-broadcast decode gated behind
config["Atomiser"]["use_adaptive_decode"].

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

All skip additions are tagged  # >>> SKIP  and adaptive ones  # >>> ADAPTIVE.

Config additions:
  Atomiser:
    use_decoder_skip: true
    decoder_skip_drop_p: 0.5     # train-only physical drop FRACTION on the
                                 # pixel's own band-tokens
    use_adaptive_decode: true    # inference-only (B=1) probe-and-broadcast
    adaptive_probe_k:    10
    adaptive_seed:       42
    skip_resolution:     10.0    # SAT pool the gather index targets
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
from torch.profiler import record_function

from training.utils.token_building.processor import TokenProcessor
from training.utils.datasets.token_grouping import compute_grid_config

from .Atomiser_senflood_skip import Atomiser_Senflood_Skip

class Atomizer_skip_inter(Atomiser_Senflood_Skip):
    """
    Analysis-only subclass: identical forward pass, but exposes pixel-skip
    cross-attention weights (spectral attention per query pixel) when
    return_attention=True. No new parameters -- state_dict is 100%
    compatible with the base checkpoint.
    """

    def _pixel_skip(self, query_tokens, query_token_idx, query_token_valid,
                    pool_tokens, pool_mask, training, return_attn=False):
        B, M, A = query_token_idx.shape
        device  = pool_tokens.device
        Dtok    = pool_tokens.shape[-1]

        pool_mask_b = pool_mask.bool() if pool_mask.dtype != torch.bool else pool_mask

        drop_p = getattr(self, "decoder_skip_drop_p", 0.0)
        if training and drop_p > 0.0:
            keep_frac = 1.0 - drop_p
            C_keep = max(1, int(torch.ceil(torch.tensor(keep_frac * A)).item()))
            rand = torch.rand(B, M, A, device=device)
            sel  = rand.argsort(dim=-1)[..., :C_keep]
            qti  = torch.gather(query_token_idx, 2, sel)
            C    = C_keep
        else:
            qti = query_token_idx
            C   = A

        flat_idx = qti.reshape(B, M * C)
        gathered = torch.gather(
            pool_tokens, 1, flat_idx.unsqueeze(-1).expand(-1, -1, Dtok)
        ).reshape(B, M, C, Dtok)

        gathered_mask = torch.gather(pool_mask_b, 1, flat_idx).reshape(B, M, C)

        query_coords = self.input_processor.geometry.get_token_centers(query_tokens)
        encoded = self.input_processor.process_data_for_encoder(
            gathered, gathered_mask, latent_positions=query_coords)

        key_pad   = gathered_mask.clone()
        invalid_q = ~query_token_valid.bool()
        key_pad   = key_pad | invalid_q.unsqueeze(-1)

        real_token = ~(gathered_mask | invalid_q.unsqueeze(-1))
        none_kept  = (~key_pad).sum(dim=-1) == 0
        if none_kept.any():
            has_real   = real_token.any(dim=-1)
            first_real = torch.argmax(real_token.float(), dim=-1)
            fix = none_kept & has_real
            if fix.any():
                bi, mi = torch.where(fix)
                key_pad[bi, mi, first_real[bi, mi]] = False
            still = none_kept & ~has_real
            if still.any():
                bi, mi = torch.where(still)
                key_pad[bi, mi, 0] = False

        BM  = B * M
        kv  = encoded.reshape(BM, C, -1)
        q   = self.pixel_query.expand(BM, 1, -1).contiguous()
        q   = self.pixel_q_norm(q)
        kpm = key_pad.reshape(BM, C)

        enriched, attn_w = self.pixel_cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=kpm,
            need_weights=return_attn,
            average_attn_weights=False,   # keep per-head if requested
        )
        enriched = enriched.squeeze(1).reshape(B, M, -1)

        if return_attn:
            # attn_w: [BM, heads, 1, C] -> [B, M, heads, C]
            attn_w = attn_w.squeeze(2).reshape(B, M, self.cross_heads, C)
            return enriched, attn_w, qti
        return enriched

    def reconstruct(self, latents_per_res, coords_per_res, query_tokens,
                    query_mask, target_resolution=None, training=True,
                    return_features=False, return_topk=False,
                    query_token_idx=None, query_token_valid=None,
                    pool_tokens=None, pool_mask=None,
                    return_attention=False):   # NEW
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
            all_latents, 1, flat_idx.unsqueeze(-1).expand(-1, -1, D)
        ).reshape(B, M, k_keep, D)
        selected_coords = torch.gather(
            all_coords, 1, flat_idx.unsqueeze(-1).expand(-1, -1, 2)
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

        keep_mask = torch.ones(B, M, k_keep, dtype=torch.bool, device=context.device)
        BM = B * M
        kv_flat = self.dec_ctx_norm(context.reshape(BM, k_keep, -1))

        attn_w = None
        qti_used = None
        if self.use_decoder_skip and query_token_idx is not None:
            if return_attention:
                enriched, attn_w, qti_used = self._pixel_skip(
                    query_tokens, query_token_idx, query_token_valid,
                    pool_tokens, pool_mask, training, return_attn=True)
            else:
                enriched = self._pixel_skip(
                    query_tokens, query_token_idx, query_token_valid,
                    pool_tokens, pool_mask, training, return_attn=False)
            q_flat = enriched.reshape(BM, 1, -1).contiguous()
        else:
            q_flat = self.global_query.expand(BM, 1, -1).contiguous()
        q_flat = self.dec_q_norm(q_flat)

        key_pad_flat = (~keep_mask).reshape(BM, k_keep)
        attn_out, _ = self.decoder_cross_attn(
            query=q_flat, key=kv_flat, value=kv_flat,
            key_padding_mask=key_pad_flat, need_weights=False)
        attn_out = attn_out.squeeze(1).reshape(B, M, -1)

        logits = self.reconstruction_head(attn_out)

        if return_attention:
            return logits, attn_w, qti_used
        return logits

    @torch.no_grad()
    def forward_with_attention(self, batch, target_resolution=None):
        """
        Convenience entry point: full forward pass at inference, chunked over
        queries, returning (logits, spectral_attn_mass, dominant_group) for
        the whole tile. spectral_attn_mass: [B, M, 3] (SAR/SWIR/REST).
        """
        self.eval()
        groups = batch["groups"]
        queries = batch["queries"]
        queries_mask = batch["queries_mask"]

        query_token_idx   = batch["query_token_idx"]
        query_token_valid = batch["query_token_valid"]
        skip_res = self.config["Atomiser"].get("skip_resolution", 10.0)
        if skip_res not in groups:
            skip_res = min(groups.keys())
        pool_tokens = groups[skip_res]["tokens"]
        pool_mask   = groups[skip_res]["mask"]

        tpl, batch_cross_k = self.val_sampling[0]
        resolutions = sorted(groups.keys())
        grid_configs = {
            res: compute_grid_config(
                resolution=res, shape=groups[res]["shape"],
                tokens_per_latent=tpl, total_tokens=groups[res]["tokens"].shape[1],
                sigma_factor=self.sigma_factor, max_k=batch_cross_k * 2,
            ) for res in resolutions
        }
        encoder_output = self.encode(groups=groups, grid_configs=grid_configs,
                                      training=False, cross_k=batch_cross_k)
        latents_per_res = encoder_output.latents_per_res
        coords_per_res  = encoder_output.coords_per_res

        chunk_size = 10_000
        N = queries.shape[1]
        logits_list, mass_list = [], []
        for i in range(0, N, chunk_size):
            logits, attn_w, qti = self.reconstruct(
                latents_per_res, coords_per_res,
                queries[:, i:i+chunk_size], queries_mask[:, i:i+chunk_size],
                target_resolution=target_resolution, training=False,
                query_token_idx=query_token_idx[:, i:i+chunk_size],
                query_token_valid=query_token_valid[:, i:i+chunk_size],
                pool_tokens=pool_tokens, pool_mask=pool_mask,
                return_attention=True,
            )
            spectral_idx_atoms = pool_tokens[..., 3].long()          # [B, N_pool]
            B_, M_, C_ = qti.shape
            atom_band = torch.gather(
                spectral_idx_atoms.unsqueeze(1).expand(-1, M_, -1), 2, qti)  # [B,M,C]
            group_id = self.band_group_lut.to(qti.device)[atom_band]         # [B,M,C]
            attn_avg = attn_w.mean(dim=2)                                    # [B,M,C] (avg heads)
            mass = torch.zeros(B_, M_, 3, device=attn_w.device)
            mass.scatter_add_(2, group_id, attn_avg)

            logits_list.append(logits)
            mass_list.append(mass)

        logits_full = torch.cat(logits_list, dim=1)
        mass_full   = torch.cat(mass_list, dim=1)
        return logits_full, mass_full
