"""
trainer_DALES_unet.py -- Lightning module for Atomiser_UnetDales on DALES
========================================================================

Location: training/trainer_DALES_unet.py

Same structure as trainer_DALES.py (per-point cross-entropy, macro mIoU /
accuracy, AdamW + cosine warmup), adapted to the point-native pipeline:
    * model: Atomiser_UnetDales (no lookup table, no TokenBuilder)
    * batch: collate_dales -> labels are batch["query_labels"], padding = 255
    * class weights from the measured DALES training frequencies (raw codes
      1..8 counted on 200 train tiles), "sqrt" = sqrt-inverse frequency, as in
      the previous DALES run (comparable), "inv" = inverse frequency, "none"
    * per-class IoU logged for val and test
    * metrics on TILES: overlapping 50 m tiles count most points several times.
      Fine for model selection; the official score comes from the full-scene
      evaluation (one averaged prediction per physical point), so the test
      metrics here are logged as test_tiles_*.
    * the cosine warmup schedule is implemented here (no transformers dependency)
"""
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from training.atomiser.Atomiser_unet_dales import Atomiser_UnetDales
from training.utils.datasets.dataset_dales_points import CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX

# Raw-code counts on 200 training tiles (inspect_dales.py), codes 1..8 -> labels 0..7.
DALES_TRAIN_COUNTS = [11385804, 7705694, 171789, 26008, 41583, 112928, 12923, 4140650]


def dales_class_weights(mode="sqrt", counts=DALES_TRAIN_COUNTS, clip=50.0):
    """mode: "sqrt" (1 / sqrt(freq)), "inv" (1 / freq), "none". Normalised so
    the frequency-weighted mean weight is 1 (the loss scale stays comparable),
    then clipped."""
    if mode in (None, "none"):
        return None
    f = torch.tensor(counts, dtype=torch.float64)
    f = f / f.sum()
    w = f.pow(-0.5) if mode == "sqrt" else 1.0 / f
    w = w / (w * f).sum()
    return w.clamp(max=clip).float()


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_ratio=0.0):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class Model_DalesUnet(pl.LightningModule):
    def __init__(self, config: dict, name: str = "dales_unet", class_weighting: str = "sqrt",
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.strict_loading = False
        config = dict(config)
        config["trainer"] = dict(config.get("trainer", {}), num_classes=NUM_CLASSES)
        self.config, self.name = config, name
        self.num_classes, self.class_names = NUM_CLASSES, list(CLASS_NAMES)
        self.ignore_index = int(ignore_index)

        self.model = Atomiser_UnetDales(config)

        w = dales_class_weights(class_weighting)
        if w is not None:
            self.register_buffer("_class_weights", w)
            print(f"[DALES-Unet] class weights ({class_weighting}): "
                  + ", ".join(f"{n}={v:.2f}" for n, v in zip(self.class_names, w.tolist())))
        else:
            self._class_weights = None
            print("[DALES-Unet] unweighted cross-entropy")
        self.loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=self.ignore_index)

        mk = dict(num_classes=self.num_classes, ignore_index=self.ignore_index)
        # ModuleDict keys cannot be "train" (clashes with nn.Module.train): suffix "_m"
        self.miou = nn.ModuleDict({f"{s}_m": MulticlassJaccardIndex(average="macro", **mk)
                                   for s in ("train", "val", "test")})
        self.macc = nn.ModuleDict({f"{s}_m": MulticlassAccuracy(average="macro", **mk)
                                   for s in ("train", "val", "test")})
        self.class_iou = nn.ModuleDict({f"{s}_m": MulticlassJaccardIndex(average=None, **mk)
                                        for s in ("val", "test")})

        t = config["trainer"]
        self.lr = float(t["lr"])
        self.weight_decay = float(t.get("weight_decay", 0.05))

    # ------------------------------------------------------------------ steps
    def forward(self, batch):
        return self.model(batch)["predictions"]

    def _shared_step(self, batch, stage):
        logits = self.forward(batch)                                     # [B, M, C]
        labels = batch["query_labels"].long()                            # [B, M], 255 = ignore
        lf, yf = logits.reshape(-1, self.num_classes), labels.reshape(-1)
        loss = self.loss_fn(lf.float(), yf)
        k = f"{stage}_m"
        with torch.no_grad():
            keep = yf != self.ignore_index
            p, y = lf[keep].argmax(-1), yf[keep]
            self.miou[k].update(p, y)
            self.macc[k].update(p, y)
            if k in self.class_iou:
                self.class_iou[k].update(p, y)
        bs = labels.shape[0]
        name = "test_tiles" if stage == "test" else stage
        # DDP safety: nothing is synchronised per step. The step loss is logged
        # locally; epoch values are synchronised once, by Lightning, at epoch end
        # (same number of collectives on every rank).
        if stage == "train":
            self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True,
                     sync_dist=False, batch_size=bs)
        self.log(f"{name}_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(f"{name}_mIoU", self.miou[k], on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=bs)
        self.log(f"{name}_macro_acc", self.macc[k], on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        if batch_idx % 50 == 0:
            for k, v in self.model.gate_stats().items():
                if v is not None:
                    self.log(f"gates/{k}", v, on_step=True, on_epoch=False, batch_size=1)
            st = getattr(self.model, "last_partition_stats", None)
            if st:                                              # partition statistics (local, no sync)
                bs = batch["points"].shape[0]
                flat = {}
                for k2, v2 in st.items():
                    if isinstance(v2, dict):
                        flat.update({f"{k2}_{kk}": vv for kk, vv in v2.items()})
                    elif isinstance(v2, (int, float)):
                        flat[k2] = v2
                for k2, v2 in flat.items():
                    self.log(f"partition/{k2}_per_tile", float(v2) / bs, on_step=True, on_epoch=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ------------------------------------------------------------------ epoch ends
    def _log_epoch(self, stage):
        # mIoU / accuracy are logged as metric objects (Lightning computes, syncs and
        # resets them). Per-class IoU: one compute (torchmetrics syncs the confusion
        # matrix itself, identically on every rank), then local logging of the
        # already-synchronised values.
        name = "test_tiles" if stage == "test" else stage
        k = f"{stage}_m"
        per = self.class_iou[k].compute()
        self.class_iou[k].reset()
        for i, n in enumerate(self.class_names):
            self.log(f"{name}_IoU/{n}", per[i].item(), sync_dist=False, rank_zero_only=True)

    def on_train_start(self):
        print(f"[DALES-Unet] rank {self.global_rank}/{self.trainer.world_size}: "
              f"{self.trainer.num_training_batches} training batches per epoch", flush=True)

    def on_validation_epoch_end(self):
        self._log_epoch("val")

    def on_test_epoch_end(self):
        self._log_epoch("test")

    # ------------------------------------------------------------------ optimiser
    def _total_steps(self):
        override = self.config.get("trainer", {}).get("total_steps")
        if override is not None:
            return int(override)
        try:
            est = int(self.trainer.estimated_stepping_batches)
        except Exception:
            est = -1
        return est if est > 0 else max(1, self.trainer.max_epochs) * 1000

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or n.endswith(".bias") or "seed" in n or "log_mult" in n
             else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": self.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        total = self._total_steps()
        warm = int(self.config.get("optimizer", {}).get("warmup_steps", max(1, int(0.05 * total))))
        print(f"[DALES-Unet] AdamW lr={self.lr} wd={self.weight_decay} | cosine: total={total} warmup={warm}")
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": cosine_with_warmup(opt, warm, total), "interval": "step"}}
