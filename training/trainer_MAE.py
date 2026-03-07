"""
MAE Pre-training Trainer
========================

Dedicated trainer for Masked Autoencoder pretraining with Atomiser.

Training flow per step:
    1. encode(groups, mask_ratio=0.75)
       → latents_per_res  [visible cross-attn, all self-attn]
       → geo_cache        [B, L, k, 8] per resolution
       → masked_indices   [n_mask] per resolution

    2. _build_mae_queries(encoder_output)
       → gather tokens from masked latent pools
       → set col 4 = col 0 (target = normalized reflectance)
       → queries [B, M, 8],  queries_mask [B, M]

    3. reconstruct(latents, coords, queries)
       → predictions [B, M, 1]

    4. loss = MSE(predictions.squeeze(-1)[valid], queries[:,:,4][valid])

Validation uses the same mask_ratio=0.75 for a comparable loss curve.

Note on duplicates: tokens near latent boundaries appear in multiple
latent pools. We do not deduplicate — this adds mild redundancy but
does not bias the loss and avoids extra indexing complexity.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.utils.datasets.token_grouping import compute_grid_config


class Model_MAE(pl.LightningModule):
    """
    MAE pretraining trainer for Atomiser.

    Single task: reconstruction via latent masking.
    No segmentation heads, no uncertainty weighting, no legacy paths.
    """

    def __init__(self, config, wand, name, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.wand = wand
        self.name = name
        self.lookup_table = lookup_table

        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mask_ratio   = float(config.get("pretrain", {}).get("mask_ratio", 0.75))

        # =====================================================================
        # ENCODER
        # =====================================================================
        self.encoder = Atomiser_Senflood(
            config=self.config, lookup_table=self.lookup_table
        )

        # =====================================================================
        # RECONSTRUCTION HEAD
        # encoder.reconstruction_head outputs [B, M, num_classes]
        # For MAE, num_classes=1 → we squeeze to [B, M] scalar predictions.
        # No external head needed — reconstruction_head lives inside encoder.
        # =====================================================================

        # =====================================================================
        # METRICS
        # =====================================================================
        self.train_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mse   = torchmetrics.MeanSquaredError()
        self.val_mae   = torchmetrics.MeanAbsoluteError()

        print(f"[MAE] Trainer initialised — mask_ratio={self.mask_ratio}")
        print(f"[MAE] Encoder latent_dim={config['Atomiser'].get('latent_dim', 256)}")

    # =========================================================================
    # ENCODE
    # =========================================================================

    def _encode(self, batch, training: bool):
        """
        Build grid configs and run encoder with MAE masking.

        Returns EncoderOutput with:
            .latents_per_res
            .coords_per_res
            .geo_cache              — {res: (geo_tokens [B,L,k,8], geo_masks [B,L,k], gc)}
            .masked_indices_per_res — {res: [n_mask]}
        """
        groups = batch["groups"]
        tpl    = self._sample_tokens_per_latent() if training else self.encoder.tokens_per_latent

        # ── Diagnostics (first 3 steps only) ──────────────────────────────
        if self.global_step < 3:
            self._diag_input(groups)

        resolutions  = sorted(groups.keys())
        grid_configs = {
            res: compute_grid_config(
                resolution=res,
                shape=groups[res]["shape"],
                tokens_per_latent=tpl,
                total_tokens=groups[res]["tokens"].shape[1],
                sigma_factor=self.encoder.sigma_factor,
                max_k=self.encoder.max_k,
            )
            for res in resolutions
        }

        encoder_output = self.encoder.encode(
            groups=groups,
            grid_configs=grid_configs,
            training=training,
            mask_ratio=self.mask_ratio,
        )

        # ── Diagnostics ───────────────────────────────────────────────────
        if self.global_step < 3:
            self._diag_latents(encoder_output)

        return encoder_output, grid_configs

    # =========================================================================
    # BUILD MAE QUERIES FROM GEO CACHE
    # =========================================================================

    def _build_mae_queries(self, encoder_output):
        """
        Gather tokens assigned to masked latents → reconstruction targets.

        For each resolution:
          - geo_tokens [B, L, k, 8]: all tokens per latent (from geographic pruning)
          - masked_indices [n_mask]:  which latents were masked
          → select [B, n_mask, k, 8], flatten to [B, n_mask*k, 8]
          → set col 4 = col 0  (target reflectance)
          → queries_mask = geo_masks for those tokens (True = padding, skip in loss)

        Tokens near latent boundaries appear in multiple pools — mild redundancy,
        no bias.

        Returns:
            queries      [B, M_total, 8]   — M_total = sum over res of n_mask*k
            queries_mask [B, M_total]       — True = invalid (padding token)
        """
        all_queries = []
        all_masks   = []

        for res, mask_idx in encoder_output.masked_indices_per_res.items():
            geo_tokens, geo_masks, _ = encoder_output.geo_cache[res]
            # geo_tokens: [B, L, k, 8]
            # geo_masks:  [B, L, k]   True = padding/invalid

            B, L, k, d = geo_tokens.shape
            n_mask      = mask_idx.shape[0]

            # Gather masked latents: [B, n_mask, k, 8]
            masked_tokens = geo_tokens[:, mask_idx]          # [B, n_mask, k, 8]
            masked_masks  = geo_masks[:, mask_idx]           # [B, n_mask, k]

            # Flatten latent and token dims: [B, n_mask*k, 8]
            masked_tokens = masked_tokens.reshape(B, n_mask * k, d)
            masked_masks  = masked_masks.reshape(B, n_mask * k)

            # Set col 4 = col 0 (target = normalized reflectance)
            queries = masked_tokens.clone()
            queries[:, :, 4] = queries[:, :, 0]

            all_queries.append(queries)
            all_masks.append(masked_masks)

        queries      = torch.cat(all_queries, dim=1)   # [B, M_total, 8]
        queries_mask = torch.cat(all_masks,   dim=1)   # [B, M_total]

        return queries, queries_mask

    # =========================================================================
    # DECODE + LOSS
    # =========================================================================

    def _decode(self, encoder_output, queries, queries_mask,
                target_resolution, training: bool):
        """
        Chunked decode: latents → predictions [B, M, 1].
        """
        chunk_size      = 10_000
        latents_per_res = encoder_output.latents_per_res
        coords_per_res  = encoder_output.coords_per_res
        N               = queries.shape[1]

        if N > chunk_size:
            preds = []
            for i in range(0, N, chunk_size):
                p = self.encoder.reconstruct(
                    latents_per_res, coords_per_res,
                    queries[:, i:i + chunk_size],
                    queries_mask[:, i:i + chunk_size],
                    target_resolution=target_resolution,
                    training=training,
                    return_features=False,
                )
                preds.append(p)
            return torch.cat(preds, dim=1)          # [B, M, 1]
        else:
            return self.encoder.reconstruct(
                latents_per_res, coords_per_res,
                queries, queries_mask,
                target_resolution=target_resolution,
                training=training,
                return_features=False,
            )                                        # [B, M, 1]

    def _compute_loss(self, predictions, queries, queries_mask):
        """
        MSE loss on valid (non-padding) masked tokens.

        predictions: [B, M, 1]
        queries:     [B, M, 8]  — col 4 = target reflectance
        queries_mask:[B, M]     — True = padding, exclude from loss

        Returns (loss, preds [B,M], targets [B,M]) or (None, None, None)
        if no valid tokens (dummy batch).
        """
        preds   = predictions.squeeze(-1)       # [B, M]
        targets = queries[:, :, 4]              # [B, M]
        valid   = ~queries_mask                 # [B, M]  True = valid

        valid_count = valid.sum()
        if valid_count == 0:
            return None, None, None

        loss = nn.functional.mse_loss(preds[valid], targets[valid])
        return loss, preds, targets

    # =========================================================================
    # TRAINING STEP
    # =========================================================================

    def training_step(self, batch, batch_idx):
        encoder_output, _  = self._encode(batch, training=True)
        queries, queries_mask = self._build_mae_queries(encoder_output)

        target_resolution = batch.get("target_resolution", None)
        predictions       = self._decode(
            encoder_output, queries, queries_mask,
            target_resolution=target_resolution, training=True,
        )

        # ── NaN guard ─────────────────────────────────────────────────────
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            if self.global_step < 50:
                print(f"[MAE step={self.global_step}] NaN/Inf in predictions — skipping",
                      flush=True)
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        loss, preds, targets = self._compute_loss(predictions, queries, queries_mask)

        if loss is None or not torch.isfinite(loss):
            # Dummy batch — zero loss attached to all params for DDP safety
            return sum(p.sum() * 0.0 for p in self.parameters())

        self.train_mse.update(preds[~queries_mask], targets[~queries_mask])
        self.train_mae.update(preds[~queries_mask], targets[~queries_mask])

        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)

        return loss

    # =========================================================================
    # VALIDATION STEP
    # =========================================================================

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        encoder_output, _     = self._encode(batch, training=False)
        queries, queries_mask = self._build_mae_queries(encoder_output)

        target_resolution = batch.get("target_resolution", None)
        predictions       = self._decode(
            encoder_output, queries, queries_mask,
            target_resolution=target_resolution, training=False,
        )

        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        loss, preds, targets = self._compute_loss(predictions, queries, queries_mask)

        if loss is None or not torch.isfinite(loss):
            self.log("val_loss", torch.tensor(0.0, device=self.device),
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return

        self.val_mse.update(preds[~queries_mask], targets[~queries_mask])
        self.val_mae.update(preds[~queries_mask], targets[~queries_mask])

        self.log("val_loss", loss,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    # =========================================================================
    # EPOCH END
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mse", self.train_mse.compute(), on_epoch=True, logger=True)
        self.log("train_mae", self.train_mae.compute(), on_epoch=True, logger=True)
        self.train_mse.reset()
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", self.val_mae.compute(), on_epoch=True, logger=True)
        self.val_mse.reset()
        self.val_mae.reset()

    # =========================================================================
    # VARIABLE LATENT DENSITY
    # =========================================================================

    def _sample_tokens_per_latent(self) -> int:
        """
        Deterministic variable-density sampling.
        DDP-safe: all ranks use global_step → same value.
        """
        choices = self.config.get("pretrain", {}).get(
            "tokens_per_latent_choices", [512, 768, 1024, 1500, 2000]
        )
        return choices[self.global_step % len(choices)]

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps  = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _diag_input(self, groups):
        for res, g in groups.items():
            tok = g["tokens"]
            msk = g["mask"]
            n_valid  = (~msk).sum(dim=-1).float().mean().item()
            has_nan  = torch.isnan(tok).any().item()
            has_inf  = torch.isinf(tok).any().item()
            if has_nan or has_inf:
                print(f"[DIAG input] res={res}: NaN={has_nan} Inf={has_inf} "
                      f"shape={list(tok.shape)} mean_valid={n_valid:.0f}", flush=True)
                col_names = ["value", "x", "y", "spec_idx",
                             "label", "query_idx", "res_idx", "time_idx"]
                for col in range(tok.shape[-1]):
                    col_data = tok[..., col]
                    finite   = col_data[torch.isfinite(col_data)]
                    if finite.numel() > 0:
                        print(f"    col[{col}] {col_names[col]:>9s}: "
                              f"[{finite.min():.4f}, {finite.max():.4f}] "
                              f"std={finite.std():.4f} "
                              f"nan_frac={torch.isnan(col_data).float().mean():.3f}",
                              flush=True)
                    else:
                        print(f"    col[{col}] {col_names[col]:>9s}: ALL NaN/Inf",
                              flush=True)

    def _diag_latents(self, encoder_output):
        for res, lat in encoder_output.latents_per_res.items():
            has_nan = torch.isnan(lat).any().item()
            if has_nan:
                frac = torch.isnan(lat).float().mean().item()
                finite_vals = lat[torch.isfinite(lat)]
                rng = (f"[{finite_vals.min():.4f}, {finite_vals.max():.4f}]"
                       if finite_vals.numel() > 0 else "ALL NaN")
                print(f"[DIAG latents] res={res}: NaN frac={frac:.3f} range={rng}",
                      flush=True)
            else:
                print(f"[DIAG latents] res={res}: OK "
                      f"[{lat.min():.4f}, {lat.max():.4f}]", flush=True)

        if encoder_output.masked_indices_per_res:
            for res, idx in encoder_output.masked_indices_per_res.items():
                L = encoder_output.latents_per_res[res].shape[1]
                print(f"[DIAG MAE] res={res}: {idx.shape[0]}/{L} latents masked "
                      f"({idx.shape[0]/L*100:.0f}%)", flush=True)

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/mae_{self.name}{suffix}.pth"
        torch.save({"encoder": self.encoder.state_dict()}, file_path)
        print(f"[MAE] Encoder saved to {file_path}")

    def load_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/mae_{self.name}{suffix}.pth"
        state     = torch.load(file_path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        print(f"[MAE] Encoder loaded from {file_path}")

    def load_encoder_for_downstream(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        print(f"[MAE] Encoder loaded for downstream from {checkpoint_path}")