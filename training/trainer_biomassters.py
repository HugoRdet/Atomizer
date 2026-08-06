"""
BioMassters Trainer (SKIP variant) for Atomizer
================================================

Mirrors Model_SenFlood_Skip's structure, adapted for:

  1. REGRESSION instead of segmentation classes. y_hat is [B, M, 1] (single
     output channel, per the earlier design decision: same decoder as
     segmentation, just 1 output channel). Loss is plain MSE.

  2. Reading from batch["tasks"][TASK_NAME] instead of batch["queries"]
     directly -- BioMasstersSkipDataset emits the PASTIS-style `tasks` dict
     convention (unlike Sen1Floods11's flat batch["queries"]), since it was
     built to match PastisHDDataset's multi-task layout. A small bridge in
     forward()/_forward_crop() mirrors the task's queries into the top-level
     "queries"/"queries_mask" keys, since the encoder (see below) expects
     that flat convention.

  3. IGNORE_VALUE = -1.0 (float sentinel, AGB is never negative) instead of
     255 -- masked out of loss/metrics the same way Sen1Floods11 masks
     ignore_index=255, just via a value comparison instead of an integer
     class id.

  4. Regression metrics: RMSE + MAE (the standard BioMassters leaderboard
     metrics) via torchmetrics, instead of IoU/Accuracy.

Everything else (optimizer/scheduler, error-predictor supervision, sliding-
window guard, save/load, complexity measurement hookup) follows the same
pattern as Model_SenFlood_Skip.

ENCODER: reuses Atomiser_Senflood_Skip -- same underlying architecture,
config-driven, no BioMassters-specific encoder class needed. See the
>>> BRIDGE comments in forward()/_forward_crop() for the one adaptation
this requires (batch["tasks"] -> batch["queries"] mirroring).

IMPORTANT — sliding window:
  Same guard as Sen1Floods11: _forward_crop would drop query_token_idx/valid,
  silently disabling the skip cascade at eval. BioMassters tiles are 256x256
  (smaller than Sen1Floods11's 512x512), so slide=False should be the
  natural default anyway -- this just fails loudly if misconfigured.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser.Atomiser_senflood_skip import Atomiser_Senflood_Skip
from training.atomiser.error_supervision import (
    compute_latent_errors,
    compute_error_predictor_loss,
)
from training.utils.datasets.sliding_window import stitch_predictions
from training.utils.datasets.utils_dataset_biomasters import BioMasstersSkipDataset


class Model_BioMassters_Skip(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table,
                 class_names=None, agb_mean: float = None, agb_std: float = None):
        super().__init__()
        self.strict_loading = False
        self.config         = config
        self.transform      = transform
        self.wand           = wand
        self.name           = name
        self.lookup_table   = lookup_table

        self.task_name    = BioMasstersSkipDataset.TASK_NAME
        self.ignore_value = BioMasstersSkipDataset.IGNORE_VALUE  # -1.0 (AGB never negative)
        self.use_sliding  = config["trainer"].get("slide", False)

        # Target normalization: plain z-score, computed once over train and
        # passed in from the launch script (reads normalization_stats.pt's
        # agb_mean/agb_std, the same file BioMasstersSkipDataset/
        # BioMasstersBaselineDataset use for input normalization). Loss
        # operates in this normalized space; predictions are inverse-
        # transformed before RMSE/MAE, which stay in real Mg/ha. Required --
        # raises if not provided, since silently falling back to raw-space
        # training would be a different experiment than what was decided,
        # not a safe default.
        #
        # NOTE: originally log1p(AGB) then z-score, switched to plain
        # z-score after observing val_RMSE: inf in practice -- a NaN/extreme
        # prediction survives torch.clamp (clamp doesn't fix NaN) and then
        # explodes exponentially through expm1. Plain z-score is linear, so
        # extreme predictions stay extreme rather than blowing up further.
        if agb_mean is None or agb_std is None:
            raise ValueError(
                "agb_mean/agb_std must be provided (read from "
                "normalization_stats.pt's agb_mean/agb_std, computed "
                "by BioMasstersSkipDataset). Target normalization (z-score) "
                "is required by this trainer, not optional."
            )
        self.register_buffer("agb_mean", torch.tensor(float(agb_mean)))
        self.register_buffer("agb_std",  torch.tensor(float(agb_std)))

        # >>> SKIP: refuse to run a silently-invalid configuration.
        # Sliding-window _forward_crop drops query_token_idx/valid, which would
        # disable the skip at eval without error. Fail loudly instead.
        use_decoder_skip = config["Atomiser"].get("use_decoder_skip", False)
        if use_decoder_skip and self.use_sliding:
            raise ValueError(
                "use_decoder_skip=True is incompatible with slide=True: the "
                "sliding-window path drops query_token_idx/valid and would "
                "silently disable the skip cascade at eval time, invalidating "
                "the comparison. Set trainer.slide=False for the skip run "
                "(BioMassters tiles are 256x256 and decode whole)."
            )
        # >>> END SKIP

        # =====================================================================
        # METRICS (regression: RMSE + MAE, the standard BioMassters
        # leaderboard metrics -- swap/add R2 here if you'd rather track that too)
        # =====================================================================
        self.metric_RMSE_train = torchmetrics.MeanSquaredError(squared=False)
        self.metric_RMSE_val   = torchmetrics.MeanSquaredError(squared=False)
        self.metric_RMSE_test  = torchmetrics.MeanSquaredError(squared=False)
        self.metric_MAE_train  = torchmetrics.MeanAbsoluteError()
        self.metric_MAE_val    = torchmetrics.MeanAbsoluteError()
        self.metric_MAE_test   = torchmetrics.MeanAbsoluteError()

        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood_Skip(
            config=self.config, lookup_table=self.lookup_table)

        # =====================================================================
        # LOSS
        # =====================================================================
        # MSE (L2) regression loss.
        self.loss = nn.MSELoss(reduction="none")

        # =====================================================================
        # ERROR PREDICTOR SUPERVISION
        # =====================================================================
        self.use_error_predictor = config["Atomiser"].get(
            "use_error_predictor", False)
        if self.use_error_predictor:
            self.lambda_error = float(
                config["Atomiser"].get("lambda_error", 0.1))
            self.error_warmup = int(
                config["Atomiser"].get("error_supervision_warmup_epochs", 0))
            print(f"[Trainer] Error predictor supervision ENABLED "
                  f"(lambda={self.lambda_error}, warmup={self.error_warmup} epochs)")
        else:
            print(f"[Trainer] Error predictor supervision DISABLED")

        self.lr           = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _should_supervise_error(self) -> bool:
        return (self.use_error_predictor
                and self.current_epoch >= self.error_warmup)

    def _task_queries(self, batch):
        task = batch["tasks"][self.task_name]
        return task["queries"], task["queries_mask"]

    # =========================================================================
    # TARGET NORMALIZATION (plain z-score)
    # =========================================================================

    def _transform_target(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Plain z-score, using train-split stats. Only meaningful where the
        input is a real AGB value; callers must mask out IGNORE_VALUE (-1.0)
        entries themselves -- this function doesn't special-case the
        sentinel, since z-score of -1.0 is just a finite (if useless) value,
        and letting it propagate into an otherwise-masked-out entry is fine
        as long as the caller never includes it in a loss/metric reduction.
        """
        return (raw - self.agb_mean) / self.agb_std.clamp(min=1e-6)

    def _inverse_transform(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Undo _transform_target: z*std + mean. Clamped to [0, MAX_PLAUSIBLE_AGB]
        for physical validity/numerical safety -- only affects metric
        reporting, never the loss (loss stays in normalized space, unclamped,
        for clean gradients).

        Plain z-score is a LINEAR transform, so unlike the earlier log1p+
        z-score version, an extreme/unconstrained prediction (e.g. from an
        under-trained model or a NaN that slipped through) stays extreme
        rather than exploding exponentially through expm1 -- that earlier
        version produced val_RMSE in the tens of millions (or inf, once a
        NaN appeared) at epoch 0/1. This clamp is now just a sane physical
        ceiling, not load-bearing numerical-stability machinery.
        """
        MAX_PLAUSIBLE_AGB = 2000.0  # Mg/ha -- generous ceiling, real values rarely exceed ~500
        pred = normalized * self.agb_std + self.agb_mean
        return pred.clamp(min=0.0, max=MAX_PLAUSIBLE_AGB)

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False, return_for_error=False):
        # >>> BRIDGE: Atomiser_Senflood_Skip.forward() reads batch["queries"]/
        # batch["queries_mask"] directly (Sen1Floods11's flat convention), but
        # BioMasstersSkipDataset produces batch["tasks"][TASK_NAME]["queries"]
        # (PASTIS-style). Mirror the task's queries into the top-level keys
        # the reused encoder expects, without mutating the caller's batch dict.
        # REMOVE this bridge once/if a BioMassters-native encoder that reads
        # batch["tasks"] directly replaces the reused Senflood architecture.
        if "queries" not in batch and "tasks" in batch:
            queries, queries_mask = self._task_queries(batch)
            batch = {**batch, "queries": queries, "queries_mask": queries_mask}
        return self.encoder(
            batch, training=training, return_for_error=return_for_error)

    # =========================================================================
    # SHARED STEP LOGIC
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        supervise_error = training and self._should_supervise_error()

        result = self.forward(
            batch, training=training, return_for_error=supervise_error)

        y_hat = result["predictions"] if isinstance(result, dict) else result  # [B, M, 1]
        y_hat = y_hat.squeeze(-1)  # [B, M] -- interpreted as NORMALIZED-space prediction

        queries, _ = self._task_queries(batch)
        raw_labels = queries[:, :, 4].float()  # continuous AGB, col 4, RAW Mg/ha

        valid_mask = raw_labels != self.ignore_value  # computed on RAW labels

        y_hat_flat      = rearrange(y_hat,       "b m -> (b m)")
        raw_labels_flat = rearrange(raw_labels,  "b m -> (b m)")
        mask_flat       = rearrange(valid_mask,  "b m -> (b m)")

        # Loss in normalized (plain z-score) space.
        norm_target_flat = self._transform_target(raw_labels_flat)
        per_elem_loss = self.loss(y_hat_flat, norm_target_flat)
        if mask_flat.any():
            reg_loss = per_elem_loss[mask_flat].mean()
        else:
            # Degenerate batch with no valid AGB pixels (shouldn't happen on
            # train/val, only possible on a test split without public AGB) --
            # zero loss rather than NaN from an empty mean.
            reg_loss = per_elem_loss.sum() * 0.0

        total_loss = reg_loss

        if supervise_error and isinstance(result, dict):
            predicted_errors = result.get("predicted_errors")
            topk_indices     = result.get("topk_indices")
            topk_dists_sq    = result.get("topk_dists_sq")
            num_latents      = result.get("num_latents")

            if (predicted_errors is not None
                    and topk_indices is not None
                    and topk_dists_sq is not None):
                # NOTE: compute_latent_errors was written for classification
                # logits/labels; for regression, per-query error is just the
                # (masked) absolute residual. Operates in normalized space
                # here (y_hat, norm_target are both normalized), consistent
                # with the loss above. If compute_latent_errors expects class
                # logits specifically, this call needs updating on the
                # encoder side to accept continuous predictions -- flagging
                # since I can't verify its internals from here.
                zone_error, err_valid_mask = compute_latent_errors(
                    logits        = y_hat.detach().unsqueeze(-1),
                    labels        = self._transform_target(raw_labels),
                    topk_indices  = topk_indices,
                    topk_dists_sq = topk_dists_sq,
                    num_latents   = num_latents,
                    ignore_index  = self.ignore_value,
                )
                L_pred = predicted_errors.shape[1]
                error_loss = compute_error_predictor_loss(
                    predicted_errors = predicted_errors,
                    zone_error       = zone_error[:, :L_pred],
                    valid_mask       = err_valid_mask[:, :L_pred],
                )
                total_loss = reg_loss + self.lambda_error * error_loss
                self.log("train_error_loss", error_loss,
                         on_step=False, on_epoch=True, logger=True)

        # Predictions inverse-transformed back to real Mg/ha for metric
        # logging -- RMSE/MAE stay in interpretable units, loss stays in
        # normalized space. raw_labels_flat is already real Mg/ha (untouched).
        preds_denorm_flat = self._inverse_transform(y_hat_flat)

        return total_loss, reg_loss, preds_denorm_flat, raw_labels_flat, mask_flat

    # =========================================================================
    # SLIDING WINDOW INFERENCE
    # =========================================================================
    # NOTE: guarded off for the skip run (see __init__). Kept intact so this
    # trainer remains a drop-in for the non-skip baseline if skip is disabled.

    def _forward_crop(self, batch, crop_idx):
        crop_queries      = batch["tasks"][self.task_name]["queries"][crop_idx:crop_idx + 1]
        crop_queries_mask = batch["tasks"][self.task_name]["queries_mask"][crop_idx:crop_idx + 1]
        mini_batch = {
            "groups": {},
            "tasks": {
                self.task_name: {
                    "queries":      crop_queries,
                    "queries_mask": crop_queries_mask,
                }
            },
            # >>> BRIDGE: same top-level mirroring as forward(), see comment there.
            "queries":      crop_queries,
            "queries_mask": crop_queries_mask,
        }
        for res, grp in batch["groups"].items():
            mini_batch["groups"][res] = {
                "tokens": grp["tokens"][crop_idx:crop_idx + 1],
                "mask":   grp["mask"][crop_idx:crop_idx + 1],
                "shape":  grp["shape"],
            }
        result = self.forward(mini_batch, training=False)
        preds = result["predictions"] if isinstance(result, dict) else result
        return preds.squeeze(-1)

    def _sliding_window_step(self, batch):
        num_crops      = batch["tasks"][self.task_name]["queries"].shape[0]
        positions      = batch["crop_positions"]
        crop_h, crop_w = batch["crop_size"]
        full_h, full_w = batch["full_size"]

        crop_preds_list = []
        for i in range(num_crops):
            with torch.no_grad():
                preds = self._forward_crop(batch, i)
            crop_preds_list.append(preds.squeeze(0))

        # NOTE: stitch_predictions was written for classification logits
        # (num_classes channels); for single-channel regression this should
        # still work if it treats the channel dim generically, but wasn't
        # verified for C=1 -- worth a quick sanity check before relying on it.
        preds_full, values_avg = stitch_predictions(
            crop_logits_list=crop_preds_list,
            crop_positions=positions,
            crop_h=crop_h, crop_w=crop_w,
            full_h=full_h, full_w=full_w,
            num_classes=1,
        )
        return preds_full, batch["label"].to(self.device), values_avg

    # =========================================================================
    # TRAINING / VALIDATION / TEST STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        total_loss, reg_loss, preds, labels, mask = self._compute_loss_and_preds(
            batch, training=True)

        if mask.any():
            self.metric_RMSE_train.update(preds[mask], labels[mask])
            self.metric_MAE_train.update(preds[mask], labels[mask])

        self.log("train_loss",     total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_reg_loss", reg_loss,   on_step=False, on_epoch=True,
                 logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch.get("sliding", False):
            # NOTE: values_avg here is normalized-space (model output, matching
            # _compute_loss_and_preds's y_hat before inverse-transform) since
            # _forward_crop -> self.forward() returns raw model predictions.
            # preds_full from stitch_predictions is therefore ALSO normalized-
            # space and must be inverse-transformed before use as real Mg/ha,
            # same as the non-sliding path. Loss uses the normalized target.
            preds_full_norm, label_full, values_avg_norm = self._sliding_window_step(batch)
            valid = label_full != self.ignore_value
            norm_target = self._transform_target(label_full)
            loss = self.loss(values_avg_norm, norm_target)
            loss = loss[valid].mean() if valid.any() else loss.sum() * 0.0
            preds_full = self._inverse_transform(preds_full_norm)
            if valid.sum() > 0:
                self.metric_RMSE_val.update(preds_full[valid], label_full[valid])
                self.metric_MAE_val.update(preds_full[valid], label_full[valid])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
            return loss

        _, reg_loss, preds, labels, mask = self._compute_loss_and_preds(
            batch, training=False)
        if mask.any():
            self.metric_RMSE_val.update(preds[mask], labels[mask])
            self.metric_MAE_val.update(preds[mask], labels[mask])
        self.log("val_loss", reg_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return reg_loss

    def test_step(self, batch, batch_idx):
        if batch.get("sliding", False):
            preds_full_norm, label_full, values_avg_norm = self._sliding_window_step(batch)
            valid = label_full != self.ignore_value
            norm_target = self._transform_target(label_full)
            loss = self.loss(values_avg_norm, norm_target)
            loss = loss[valid].mean() if valid.any() else loss.sum() * 0.0
            preds_full = self._inverse_transform(preds_full_norm)
            if valid.sum() > 0:
                self.metric_RMSE_test.update(preds_full[valid], label_full[valid])
                self.metric_MAE_test.update(preds_full[valid], label_full[valid])
            self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
            return loss

        _, reg_loss, preds, labels, mask = self._compute_loss_and_preds(
            batch, training=False)
        if mask.any():
            self.metric_RMSE_test.update(preds[mask], labels[mask])
            self.metric_MAE_test.update(preds[mask], labels[mask])
        self.log("test_loss", reg_loss, on_step=False, on_epoch=True, logger=True)
        return reg_loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_RMSE", self.metric_RMSE_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MAE",  self.metric_MAE_train.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.metric_RMSE_train.reset()
        self.metric_MAE_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_RMSE", self.metric_RMSE_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val_MAE",  self.metric_MAE_val.compute(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.metric_RMSE_val.reset()
        self.metric_MAE_val.reset()

    def on_test_epoch_end(self):
        test_rmse = self.metric_RMSE_test.compute()
        test_mae  = self.metric_MAE_test.compute()

        self.log("test_RMSE", test_rmse, on_epoch=True, logger=True)
        self.log("test_MAE",  test_mae,  on_epoch=True, logger=True)

        print(f"\n{'='*60}")
        print(f"TEST RESULTS")
        print(f"{'='*60}")
        print(f"  RMSE (Mg/ha): {test_rmse:.4f}")
        print(f"  MAE  (Mg/ha): {test_mae:.4f}")
        print(f"{'='*60}\n")

        self.metric_RMSE_test.reset()
        self.metric_MAE_test.reset()

    # =========================================================================
    # MODEL SAVE / LOAD
    # =========================================================================

    def save_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[BioMassters] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix    = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[BioMassters] Model loaded from {file_path}")

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def _compute_total_steps(self) -> int:
        override = self.config.get("trainer", {}).get("total_steps", None)
        if override is not None:
            print(f"[Trainer] total_steps override: {override}")
            return int(override)

        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1

        if est <= 0:
            fallback = max(1, self.trainer.max_epochs) * 1000
            print(f"[Trainer] WARN: cannot estimate total_steps. "
                  f"Falling back to {fallback}.")
            return fallback

        print(f"[Trainer] total_steps estimate: {est}")
        return est

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps  = self._compute_total_steps()
        warmup_steps = self.config.get("optimizer", {}).get(
            "warmup_steps", max(1, int(0.05 * total_steps))
        )

        print(f"[Trainer] LR schedule final: "
              f"total_steps={total_steps}, warmup={warmup_steps}, "
              f"peak_lr={self.lr}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
            },
        }
