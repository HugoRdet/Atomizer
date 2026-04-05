"""
MNIST Classification Trainer — New Atomiser
=============================================

Accepts batch dict format directly from the updated MNIST dataset.
Uses classification task (attention pooling → logits).

Batch format:
    batch["groups"][0.2]["tokens"]: [B, N, 8]
    batch["groups"][0.2]["mask"]:   [B, N]
    batch["queries"]:               [B, M, 8]
    batch["queries_mask"]:          [B, M]
    batch["label"]:                 [B]  (digit class 0-9)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood


class Model_MNIST(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.name = name
        self.lookup_table = lookup_table

        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # Model
        self.encoder = Atomiser_Senflood(config=self.config, lookup_table=self.lookup_table)

        # Freeze decoder — classification only uses attention pooling
        self.encoder.freeze_decoder()

        # Freeze spectral encoder (1 band, nothing to learn)
        self.encoder.input_processor.spectral_encoder.requires_grad_(False)

        # Loss
        self.class_loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(
            self.num_classes, average="macro"
        )

    def forward(self, batch, training=False):
        """Forward — pass batch dict to encoder with task=classification."""
        return self.encoder(batch, training=training, task="classification")

    def on_fit_start(self):
        self.encoder.freeze_decoder()
        self.encoder.unfreeze_classifier()
        self.encoder.unfreeze_encoder()

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        logits = self.forward(batch, training=True)

        loss = self.class_loss_fn(logits, labels)
        self.train_acc.update(logits.argmax(dim=-1), labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True, logger=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        logits = self.forward(batch, training=False)

        loss = self.class_loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)

        self.val_acc.update(preds, labels)
        self.metric_IoU_val.update(preds, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, logger=True)
        self.log("val_IoU", self.metric_IoU_val.compute(), prog_bar=True, logger=True)
        self.val_acc.reset()
        self.metric_IoU_val.reset()

    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)

    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "cosine_warmup",
            },
        }