"""
FRACTAL PTv3 Trainer
====================

Lightning module for PTv3Fractal, mirroring Model_PerceiverFractal so the
two are directly comparable:
  - same 7 classes, ignore_index=255
  - same loss: plain CrossEntropy. NOTE: Model_PerceiverFractal computes
    inverse-frequency class weights but never applies them (the lines
    passing them to the loss are commented out). This trainer mirrors
    that, i.e. NO class weighting, unless apply_class_weights=True.
  - same mIoU + macro accuracy, same per-class test IoU logging
  - same AdamW + cosine warmup schedule

Difference: logits come per voxel ([Nv, K]) and are gathered onto the
evaluated LiDAR points with batch["eval_index"] before loss/metrics.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from training.PTv3.model_ptv3_fractal import PTv3Fractal


FRACTAL_CLASS_NAMES = [
    "other", "ground", "vegetation", "building",
    "water", "bridge", "permanent_structure",
]
NUM_CLASSES_FRACTAL = 7
FRACTAL_CLASS_FREQS = [0.0056, 0.3897, 0.5698, 0.0280, 0.0052, 0.0013, 0.0005]


def default_fractal_class_weights(freqs=FRACTAL_CLASS_FREQS, weight_clip=50.0):
    raw = torch.tensor([1.0 / max(f, 1e-9) for f in freqs], dtype=torch.float32)
    return raw.clamp(max=weight_clip)


class Model_PTv3Fractal(pl.LightningModule):

    def __init__(
        self,
        in_channels: int = 9,
        grid_size: float = 0.1,
        patch_size: int = 1024,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 6, 2),
        dec_channels=(64, 64, 128, 256),
        dec_depths=(2, 2, 2, 2),
        head_dim: int = 16,
        drop_path: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = None,
        ignore_index: int = 255,
        apply_class_weights: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes  = NUM_CLASSES_FRACTAL
        self.class_names  = FRACTAL_CLASS_NAMES
        self.ignore_index = ignore_index
        self.lr           = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.model = PTv3Fractal(
            in_channels=in_channels,
            num_classes=self.num_classes,
            grid_size=grid_size,
            patch_size=patch_size,
            enc_channels=enc_channels,
            enc_depths=enc_depths,
            dec_channels=dec_channels,
            dec_depths=dec_depths,
            head_dim=head_dim,
            drop_path=drop_path,
        )

        ce_kwargs = {"ignore_index": int(ignore_index)}
        if apply_class_weights:
            self.register_buffer("_class_weights", default_fractal_class_weights())
            ce_kwargs["weight"] = self._class_weights
            print(f"[PTv3Fractal-Trainer] Class weights APPLIED: "
                  f"{self._class_weights.tolist()}")
        else:
            print("[PTv3Fractal-Trainer] No class weighting "
                  "(matches Model_PerceiverFractal as written).")
        self.loss_fn = nn.CrossEntropyLoss(**ce_kwargs)

        mk = dict(num_classes=self.num_classes, average="macro",
                  ignore_index=int(ignore_index))
        self.train_miou      = MulticlassJaccardIndex(**mk)
        self.val_miou        = MulticlassJaccardIndex(**mk)
        self.test_miou       = MulticlassJaccardIndex(**mk)
        self.train_macro_acc = MulticlassAccuracy(**mk)
        self.val_macro_acc   = MulticlassAccuracy(**mk)
        self.test_macro_acc  = MulticlassAccuracy(**mk)
        self.test_per_class_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes, average=None,
            ignore_index=int(ignore_index),
        )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[PTv3Fractal-Trainer] {self.num_classes} classes, "
              f"grid_size={grid_size} m, params={n_params:,}, lr={lr}")

    # =========================================================================

    def forward(self, batch):
        return self.model(batch)                         # [Nv, K]

    def _shared_step(self, batch, stage: str):
        is_train   = (stage == "train")
        logits_vox = self.forward(batch)                 # [Nv, K]
        logits     = logits_vox[batch["eval_index"]]     # [Ne, K]
        labels     = batch["eval_labels"].long()         # [Ne]
        loss       = self.loss_fn(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            if stage == "train":
                self.train_miou.update(preds, labels)
                self.train_macro_acc.update(preds, labels)
            elif stage == "val":
                self.val_miou.update(preds, labels)
                self.val_macro_acc.update(preds, labels)
            else:
                self.test_miou.update(preds, labels)
                self.test_macro_acc.update(preds, labels)
                self.test_per_class_iou.update(preds, labels)

        bs = len(batch["patch_id"])
        self.log(f"{stage}_loss", loss, on_step=is_train, on_epoch=True,
                 prog_bar=is_train, sync_dist=True, batch_size=bs)
        # share of context points kept after voxelization (1.0 = no merging)
        self.log(f"{stage}_voxel_keep_ratio",
                 logits_vox.shape[0] / max(batch["n_context"], 1),
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mIoU", self.train_miou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macro_acc", self.train_macro_acc, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_mIoU", self.val_miou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macro_acc", self.val_macro_acc, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        per_class = self.test_per_class_iou.compute()
        self.test_per_class_iou.reset()
        for i, name in enumerate(self.class_names):
            self.log(f"test_IoU/{name}", per_class[i].item(), on_epoch=True, sync_dist=True)
        self.log("test_mIoU", self.test_miou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_macro_acc", self.test_macro_acc, on_epoch=True, sync_dist=True)

    # =========================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = -1
        if total_steps <= 0:
            total_steps = max(1, self.trainer.max_epochs) * 1000
            print(f"[PTv3Fractal-Trainer] WARN: falling back to total_steps={total_steps}")
        warmup = (self.warmup_steps if self.warmup_steps is not None
                  else max(1, int(0.05 * total_steps)))
        print(f"[PTv3Fractal-Trainer] LR schedule: total_steps={total_steps}, "
              f"warmup={warmup}, peak_lr={self.lr}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=total_steps,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
