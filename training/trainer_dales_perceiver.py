"""
trainer_dales_perceiver.py
=============================

PyTorch Lightning wrapper for PerceiverDales. Mirrors Model_PerceiverFractal's
constructor contract (inferred from script_train_fractal_perceiver.py's usage)
and Model_Dales's loss/metrics setup (class weighting, per-class IoU logging)
-- same 8-class DALES metrics as Atomizer's own trainer_DALES.py, for direct
comparability.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from .perceiverIO.perceiver_DALES import PerceiverDales


NUM_CLASSES_DALES = 8
DALES_CLASS_NAMES = [
    "ground", "vegetation", "cars", "trucks",
    "power_lines", "fences", "poles", "buildings",
]

# Same frequencies as Atomizer's trainer_DALES.py -- kept identical so any
# difference in results traces to the ARCHITECTURE, not a different loss
# weighting scheme.
DALES_CLASS_FREQS = [
    0.5040, 0.3034, 0.0078, 0.0011, 0.0017, 0.0046, 0.0007, 0.1716,
]


def default_dales_class_weights(freqs=DALES_CLASS_FREQS,
                                 weight_clip: float = 50.0) -> torch.Tensor:
    """SQRT-inverse-frequency weights -- see Atomizer's trainer_DALES.py
    for the reasoning (kept identical here for a fair comparison)."""
    raw = torch.tensor(
        [1.0 / max(f, 1e-9) ** 0.5 for f in freqs], dtype=torch.float32
    )
    return raw.clamp(max=weight_clip)


class Model_PerceiverDales(pl.LightningModule):
    def __init__(
        self,
        query_chunk_size: int = 100_000,
        num_latents: int = 256,
        latent_dim: int = 256,
        depth: int = 6,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        self_per_cross_attn: int = 1,
        weight_tie_layers: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        echo_hidden_dim: int = 64,
        intensity_hidden_dim: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = None,
        ignore_index: int = 255,
        class_weights=None,   # "auto" | None
        num_classes: int = NUM_CLASSES_DALES,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.query_chunk_size = query_chunk_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_names = DALES_CLASS_NAMES

        self.model = PerceiverDales(
            num_classes=num_classes,
            num_latents=num_latents,
            latent_dim=latent_dim,
            depth=depth,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            self_per_cross_attn=self_per_cross_attn,
            weight_tie_layers=weight_tie_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            echo_hidden_dim=echo_hidden_dim,
            intensity_hidden_dim=intensity_hidden_dim,
        )

        ce_kwargs = {}
        if ignore_index is not None:
            ce_kwargs["ignore_index"] = int(ignore_index)
        if class_weights == "auto":
            weights = default_dales_class_weights()
            self.register_buffer("_class_weights", weights)
            ce_kwargs["weight"] = self._class_weights
            print(f"[PerceiverDales-Trainer] class weights (sqrt-inv-freq, "
                  f"clip=50): {weights.tolist()}")
        self.loss_fn = nn.CrossEntropyLoss(**ce_kwargs)

        metric_kwargs = dict(num_classes=num_classes, average="macro")
        self.train_miou = MulticlassJaccardIndex(
            ignore_index=ignore_index, **metric_kwargs)
        self.val_miou = MulticlassJaccardIndex(
            ignore_index=ignore_index, **metric_kwargs)
        self.test_miou = MulticlassJaccardIndex(
            ignore_index=ignore_index, **metric_kwargs)

        self.train_acc = MulticlassAccuracy(
            ignore_index=ignore_index, num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(
            ignore_index=ignore_index, num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(
            ignore_index=ignore_index, num_classes=num_classes, average="macro")

        # Genuine OVERALL ACCURACY (point-weighted / micro), distinct from
        # the macro-average above. Macro-accuracy averages per-class recall
        # EQUALLY regardless of class frequency, so it's heavily dragged
        # down by rare, poorly-classified classes (e.g. poles at <0.1% of
        # points) even when the dominant classes (ground/vegetation/
        # buildings, ~97% of points here) are classified well. "OA" in a
        # results table conventionally means point-weighted accuracy, NOT
        # macro-accuracy -- log this one for that column, not test_macro_acc.
        self.test_acc_oa = MulticlassAccuracy(
            ignore_index=ignore_index, num_classes=num_classes, average="micro")

        per_class_kwargs = dict(num_classes=num_classes, average=None)
        self.test_iou_per_class = MulticlassJaccardIndex(
            ignore_index=ignore_index, **per_class_kwargs)

        print(f"[PerceiverDales-Trainer] {num_classes} classes, "
              f"ignore_index={ignore_index}")

    def forward(self, batch, training=True):
        qcs = None if training else self.query_chunk_size
        return self.model(batch, training=training, query_chunk_size=qcs)

    def _shared_step(self, batch, stage: str):
        is_train = (stage == "train")
        logits = self.forward(batch, training=is_train)  # [B, M, C]
        labels = batch["label"]                            # [B, M]

        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"Model produced {logits.shape[-1]} classes, "
                f"expected {self.num_classes}."
            )

        loss = self.loss_fn(logits.reshape(-1, self.num_classes),
                             labels.reshape(-1))

        preds = logits.argmax(dim=-1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, "train")
        self.train_miou.update(preds, labels)
        self.train_acc.update(preds, labels)
        self.log("train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_mIoU", self.train_miou.compute(), prog_bar=True)
        self.log("train_macro_acc", self.train_acc.compute())
        self.train_miou.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, "val")
        self.val_miou.update(preds, labels)
        self.val_acc.update(preds, labels)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_mIoU", self.val_miou.compute(), prog_bar=True)
        self.log("val_macro_acc", self.val_acc.compute())
        self.val_miou.reset()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, "test")
        self.test_miou.update(preds, labels)
        self.test_acc.update(preds, labels)
        self.test_acc_oa.update(preds, labels)
        self.test_iou_per_class.update(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test_mIoU", self.test_miou.compute())
        self.log("test_macro_acc", self.test_acc.compute())
        self.log("test_OA", self.test_acc_oa.compute())
        per_class = self.test_iou_per_class.compute()
        for name, val in zip(self.class_names, per_class.tolist()):
            self.log(f"test_IoU/{name}", val)
        self.test_miou.reset()
        self.test_acc.reset()
        self.test_acc_oa.reset()
        self.test_iou_per_class.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.warmup_steps is None or self.warmup_steps <= 0:
            return optimizer

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
