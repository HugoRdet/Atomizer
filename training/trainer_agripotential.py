"""
AgriPotential Trainer
=====================

Same encoder selection, optimiser and schedule as PASTISTrainer, with the
metrics used on AgriPotential (paper and ImageCLEF AI4Agri 2026), all computed
over labelled pixels only:

    acc_pm1  accuracy with +-1 class tolerance   (ImageCLEF ranking metric)
    acc      exact accuracy
    mae      mean absolute error in classes
    mIoU     mean IoU over the 5 classes (absent class -> 0)

Loss (config trainer.loss):
    "ce"       cross-entropy over the 5 classes (default, as on PASTIS)
    "ordinal"  cumulative binary cross-entropy: logit k (k = 0..3) models
               P(y > k); prediction = number of logits with sigmoid > 0.5.
               Uses the first 4 of the 5 head outputs, so no model change.
               The dataset paper reports this formulation beats CE and MSE.
"""

import torch
import torch.nn.functional as F
import torchmetrics
from einops import rearrange

from training.trainer_PASTIS import PASTISTrainer


class AgriPotentialTrainer(PASTISTrainer):

    CROP_NAMES = ["very_low", "low", "average", "high", "very_high"]

    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__(config, wand, name, transform, lookup_table)
        K = len(self.CROP_NAMES)
        if self.num_classes != K:
            raise ValueError(f"AgriPotential has {K} classes, config trainer.num_classes="
                             f"{self.num_classes}")

        self.loss_type = str(config["trainer"].get("loss", "ce")).lower()
        if self.loss_type not in ("ce", "ordinal"):
            raise ValueError("trainer.loss must be 'ce' or 'ordinal'")
        if self.loss_type == "ordinal" and self.use_error_predictor:
            print("[AGRIPOTENTIAL] error predictor disabled with the ordinal loss")
            self.use_error_predictor = False
        self.register_buffer("_ord_k", torch.arange(K - 1), persistent=False)

        for split in ("train", "val", "test"):
            for m in ("mae", "exact", "pm1"):
                setattr(self, f"ord_{m}_{split}", torchmetrics.MeanMetric())
        print(f"[AGRIPOTENTIAL] loss={self.loss_type}")

    # =========================================================================
    # LOSS
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        if self.loss_type == "ce":
            return super()._compute_loss_and_preds(batch, training=training)

        result = self.forward(batch, training=training)
        y_hat = result.get("predictions") if isinstance(result, dict) else result
        batch = self._ensure_flat_batch(batch)
        labels = batch["queries"][:, :, 4].long()

        z = y_hat[..., : len(self.CROP_NAMES) - 1].float()            # [B, M, K-1]
        valid = labels != self.ignore_index
        if valid.any():
            target = (labels[valid].unsqueeze(-1) > self._ord_k).float()
            loss = F.binary_cross_entropy_with_logits(z[valid], target)
        else:
            loss = z.sum() * 0.0
        preds = (torch.sigmoid(z) > 0.5).sum(-1)
        return loss, loss, preds, labels

    # =========================================================================
    # ORDINAL METRICS
    # =========================================================================

    def _update_ordinal(self, split, preds, labels):
        p = rearrange(preds, "b m -> (b m)")
        y = rearrange(labels, "b m -> (b m)")
        v = y != self.ignore_index
        if not v.any():
            return
        err = (p[v] - y[v]).abs().float()
        getattr(self, f"ord_mae_{split}").update(err)
        getattr(self, f"ord_exact_{split}").update((err == 0).float())
        getattr(self, f"ord_pm1_{split}").update((err <= 1).float())

    def _log_ordinal(self, split, prog_bar=False):
        out = {}
        for m in ("mae", "exact", "pm1"):
            metric = getattr(self, f"ord_{m}_{split}")
            out[m] = metric.compute()
            metric.reset()
        self.log(f"{split}_mae", out["mae"], on_epoch=True, logger=True)
        self.log(f"{split}_acc_exact", out["exact"], on_epoch=True, logger=True)
        self.log(f"{split}_acc_pm1", out["pm1"], on_epoch=True, prog_bar=prog_bar, logger=True)
        return out

    def training_step(self, batch, batch_idx):
        loss, class_loss, preds, labels = self._compute_loss_and_preds(batch, training=True)
        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)
        self._update_ordinal("train", preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", class_loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, preds, labels = self._compute_loss_and_preds(batch, training=False)
        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)
        self._update_ordinal("val", preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, loss, preds, labels = self._compute_loss_and_preds(batch, training=False)
        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)
        self._update_ordinal("test", preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    # all 5 classes count (no background class here)
    def _compute_crop_miou(self, per_class_iou):
        return torch.nan_to_num(per_class_iou[: len(self.CROP_NAMES)], nan=0.0).mean()

    def _compute_crop_acc(self, per_class_acc):
        return torch.nan_to_num(per_class_acc[: len(self.CROP_NAMES)], nan=0.0).mean()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self._log_ordinal("train")

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_ordinal("val", prog_bar=True)

    def on_test_epoch_end(self):
        iou = self.metric_IoU_test.compute()
        miou = self._compute_crop_miou(iou)
        self.log("test_mIoU", miou, on_epoch=True, logger=True)
        self.log("test_accuracy", self._compute_crop_acc(self.metric_acc_test.compute()),
                 on_epoch=True, logger=True)
        for i, name in enumerate(self.CROP_NAMES):
            self.log(f"test_IoU_{name}", iou[i], on_epoch=True, logger=True)
        o = self._log_ordinal("test")

        print(f"\n{'='*60}\nTEST RESULTS (labelled pixels only)\n{'='*60}")
        print(f"  acc +-1:  {o['pm1']:.4f}\n  acc:      {o['exact']:.4f}\n"
              f"  MAE:      {o['mae']:.4f}\n  mIoU:     {miou:.4f}")
        for i, name in enumerate(self.CROP_NAMES):
            print(f"  IoU {name}: {iou[i]:.4f}")
        print(f"{'='*60}\n")
        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()
