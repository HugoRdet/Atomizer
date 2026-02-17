"""
Sen1Floods11 / MADOS Trainer for Atomizer
==========================================

Semantic segmentation trainer with sliding window inference support.

Config:
    config["trainer"]["slide"]: bool (default False)
        If True, val/test datasets return sliding window batches.
        The trainer detects these via batch["sliding"] == True and
        processes each crop independently, stitching predictions
        before computing metrics on the full tile.

Token format (8 columns):
    [value, x, y, spectral_idx, label, query_flag, resolution_idx, time_idx]

Batch format (normal):
    groups[res]["tokens"]: [B, N, 8]
    queries:               [B, M, 8]
    label:                 [B, H, W]

Batch format (sliding window, after collate):
    groups[res]["tokens"]: [num_crops, N, 8]   ← crops stacked on dim 0
    queries:               [num_crops, M, 8]
    label:                 [H, W]              ← full tile, no batch dim
    crop_positions:        [(y0, x0), ...]
    crop_size:             (crop_h, crop_w)
    full_size:             (full_h, full_w)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from training.atomiser import Atomiser_Senflood
from training.atomiser.error_supervision import compute_error_supervision
from training.utils.datasets.sliding_window import stitch_predictions


class Model_SenFlood(pl.LightningModule):
    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform = transform
        self.wand = wand
        self.name = name
        self.lookup_table = lookup_table

        self.num_classes = config["trainer"]["num_classes"]
        self.ignore_index = 255

        # Sliding window inference for val/test
        self.use_sliding = config["trainer"].get("slide", False)

        # =====================================================================
        # METRICS
        # =====================================================================
        self.metric_IoU_train = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index
        )
        self.metric_IoU_val = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index
        )
        self.metric_IoU_test = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes,
            average=None, ignore_index=self.ignore_index
        )
        self.metric_acc_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index
        )
        self.metric_acc_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average="macro", ignore_index=self.ignore_index
        )
        self.metric_acc_test = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes,
            average=None, ignore_index=self.ignore_index
        )

        # =====================================================================
        # MODEL
        # =====================================================================
        self.encoder = Atomiser_Senflood(config=self.config, lookup_table=self.lookup_table)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])

        # =====================================================================
        # ERROR SUPERVISION (optional)
        # =====================================================================
        self.use_error_guided_displacement = config["Atomiser"].get("use_error_guided_displacement", False)
        self.use_gravity_displacement = config["Atomiser"].get("use_gravity_displacement", False)
        self.use_error_supervision = (
            self.use_error_guided_displacement or self.use_gravity_displacement
        )

        if self.use_error_supervision:
            self.lambda_error = config["Atomiser"].get("lambda_error", 0.1)
            self.error_grid_size = config["Atomiser"].get("error_grid_size", 7)
            self.error_grid_spacing = config["Atomiser"].get("error_grid_spacing", 2)
            self.error_channels_to_sample = config["Atomiser"].get("error_channels_to_sample", 1)
            self.error_loss_type = config["Atomiser"].get("error_loss_type", "mse")
            self.error_normalize = config["Atomiser"].get("error_normalize", True)
            self.error_warmup = config["Atomiser"].get("error_supervision_warmup_epochs", 0)
            self.stable_depth = config["Atomiser"].get("stable_depth", 0)

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, batch, training=False, return_trajectory=False,
                return_predicted_errors=False):
        """Forward pass — delegates entirely to encoder."""
        return self.encoder(
            batch,
            training=training,
            return_trajectory=return_trajectory,
            return_predicted_errors=return_predicted_errors,
        )

    def _should_supervise_error(self):
        return self.use_error_supervision and self.current_epoch >= self.error_warmup

    # =========================================================================
    # SHARED STEP LOGIC (normal path)
    # =========================================================================

    def _compute_loss_and_preds(self, batch, training=False):
        """
        Run forward, compute loss, return (loss, preds, labels, result).
        Used by train step and normal (non-sliding) val/test steps.
        """
        supervise_error = self._should_supervise_error() and training

        result = self.forward(
            batch,
            training=training,
            return_trajectory=supervise_error or (not training),
            return_predicted_errors=supervise_error or (not training),
        )

        if isinstance(result, dict):
            y_hat = result["predictions"]
        else:
            y_hat = result

        # Labels from column 4 of query tokens: [B, M, 8] → [B, M]
        labels = batch["queries"][:, :, 4].long()

        # Flatten: [B, M, C] → [B*M, C],  [B, M] → [B*M]
        y_hat_flat = rearrange(y_hat, "b t c -> (b t) c")
        labels_flat = rearrange(labels, "b n -> (b n)")

        class_loss = self.loss(y_hat_flat, labels_flat)

        # Error supervision (training only)
        total_loss = class_loss
        if supervise_error and isinstance(result, dict) and result.get("predicted_errors") is not None:
            error_loss, _ = compute_error_supervision(
                model=self.encoder,
                trajectory=result["trajectory"],
                predicted_errors=result["predicted_errors"],
                latents=result["latents"],
                final_coords=result["final_coords"],
                image_err=batch["image"],
                geometry=self.encoder.input_processor.geometry,
                grid_size=self.error_grid_size,
                spacing=self.error_grid_spacing,
                num_channels_to_sample=self.error_channels_to_sample,
                loss_type=self.error_loss_type,
                normalize=self.error_normalize,
            )
            total_loss = class_loss + (self.lambda_error * error_loss)
            self.log("train_error_loss", error_loss, on_step=False, on_epoch=True, logger=True)

        preds = torch.argmax(y_hat, dim=-1)  # [B, M]

        return total_loss, class_loss, preds, labels

    # =========================================================================
    # SLIDING WINDOW INFERENCE
    # =========================================================================

    def _forward_crop(self, batch, crop_idx):
        """
        Run forward on a single crop slice from a sliding window batch.

        Slices crop_idx from the stacked [num_crops, ...] tensors,
        keeping a batch dim of 1. Returns logits [1, M, C].
        """
        mini_batch = {
            "groups": {},
            "queries": batch["queries"][crop_idx:crop_idx + 1],
            "queries_mask": batch["queries_mask"][crop_idx:crop_idx + 1],
        }
        for res, grp in batch["groups"].items():
            mini_batch["groups"][res] = {
                "tokens": grp["tokens"][crop_idx:crop_idx + 1],
                "mask": grp["mask"][crop_idx:crop_idx + 1],
                "shape": grp["shape"],
            }

        result = self.forward(mini_batch, training=False)

        if isinstance(result, dict):
            return result["predictions"]  # [1, M, C]
        return result

    def _sliding_window_step(self, batch):
        """
        Process all crops, stitch predictions, compute metrics on full tile.

        Returns:
            preds_full: [H, W] full-tile prediction (argmax)
            label_full: [H, W] full-tile label
            logits_avg: [H*W, C] averaged logits
        """
        num_crops = batch["queries"].shape[0]

        positions = batch["crop_positions"]
        crop_h, crop_w = batch["crop_size"]
        full_h, full_w = batch["full_size"]

        crop_logits_list = []

        for i in range(num_crops):
            with torch.no_grad():
                logits = self._forward_crop(batch, i)  # [1, crop_h*crop_w, C]
            crop_logits_list.append(logits.squeeze(0))  # [crop_h*crop_w, C]

        # Stitch into full tile
        preds_full, logits_avg = stitch_predictions(
            crop_logits_list=crop_logits_list,
            crop_positions=positions,
            crop_h=crop_h, crop_w=crop_w,
            full_h=full_h, full_w=full_w,
            num_classes=self.num_classes,
        )

        # label: [H, W] (no batch dim from collate)
        label_full = batch["label"].to(self.device)

        return preds_full, label_full, logits_avg

    # =========================================================================
    # TRAINING / VALIDATION / TEST STEPS
    # =========================================================================

    def training_step(self, batch, batch_idx):
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(
            batch, training=True
        )

        self.metric_IoU_train.update(preds, labels)
        self.metric_acc_train.update(preds, labels)

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", class_loss, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # ── Sliding window path ─────────────────────────────
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)

            loss = self.loss(
                logits_avg.unsqueeze(0),
                label_full.unsqueeze(0),
            )

            valid = (label_full != self.ignore_index)
            if valid.sum() > 0:
                self.metric_IoU_val.update(preds_full[valid], label_full[valid])
                self.metric_acc_val.update(preds_full[valid], label_full[valid])

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss

        # ── Normal path ─────────────────────────────────────
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(batch, training=False)

        self.metric_IoU_val.update(preds, labels)
        self.metric_acc_val.update(preds, labels)

        self.log("val_loss", class_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return class_loss

    def test_step(self, batch, batch_idx):
        """Modified test_step with debugging for first batch."""
        
        # ═════════════════════════════════════════════════════════════════════
        # DEBUG BLOCK - ONLY FOR FIRST BATCH
        # ═════════════════════════════════════════════════════════════════════
        if batch_idx == 0:
            print("\n" + "="*70)
            print("🔍 TEST DEBUG - BATCH 0")
            print("="*70)
            
            # ─────────────────────────────────────────────────────────────────
            # 1. Check sliding window
            # ─────────────────────────────────────────────────────────────────
            is_sliding = batch.get("sliding", False)
            print(f"\n[1] Sliding window: {is_sliding}")
            
            if is_sliding:
                print("  ⚠️  Test is using SLIDING WINDOW")
                print("  Is validation also using sliding window?")
            
            # ─────────────────────────────────────────────────────────────────
            # 2. Check labels
            # ─────────────────────────────────────────────────────────────────
            print(f"\n[2] Labels:")
            labels = batch["queries"][:, :, 4].long()  # Column 4 = labels
            print(f"  Shape: {labels.shape}")
            print(f"  Unique values: {torch.unique(labels).tolist()}")
            
            # Distribution
            for cls in range(self.num_classes):
                count = (labels == cls).sum().item()
                pct = 100 * count / labels.numel()
                print(f"  Class {cls}: {count} ({pct:.1f}%)")
            
            ignore_count = (labels == self.ignore_index).sum().item()
            ignore_pct = 100 * ignore_count / labels.numel()
            print(f"  Ignore (255): {ignore_count} ({ignore_pct:.1f}%)")
            
            if ignore_pct > 80:
                print(f"  ⚠️  WARNING: {ignore_pct:.1f}% of pixels are IGNORED!")
                print(f"  This will tank your metrics!")
            
            # ─────────────────────────────────────────────────────────────────
            # 3. Run forward and check predictions
            # ─────────────────────────────────────────────────────────────────
            print(f"\n[3] Forward pass:")
            
            if not is_sliding:
                with torch.no_grad():
                    result = self.forward(batch, training=False)
                
                if isinstance(result, dict):
                    y_hat = result["predictions"]
                else:
                    y_hat = result
                
                preds = torch.argmax(y_hat, dim=-1)
                
                print(f"  Predictions shape: {preds.shape}")
                print(f"  Unique predictions: {torch.unique(preds).tolist()}")
                
                # Distribution
                for cls in range(self.num_classes):
                    count = (preds == cls).sum().item()
                    pct = 100 * count / preds.numel()
                    print(f"  Predicts class {cls}: {count} ({pct:.1f}%)")
                
                # ─────────────────────────────────────────────────────────
                # 4. Check alignment
                # ─────────────────────────────────────────────────────────
                print(f"\n[4] Alignment check:")
                
                # Flatten
                preds_flat = preds.flatten()
                labels_flat = labels.flatten()
                
                # Valid pixels (not ignored)
                valid_mask = (labels_flat != self.ignore_index)
                preds_valid = preds_flat[valid_mask]
                labels_valid = labels_flat[valid_mask]
                
                if len(preds_valid) > 0:
                    matches = (preds_valid == labels_valid).sum().item()
                    accuracy = 100 * matches / len(preds_valid)
                    
                    print(f"  Valid pixels: {len(preds_valid)}")
                    print(f"  Correct: {matches} ({accuracy:.1f}%)")
                    
                    if accuracy < 20:
                        print(f"\n  ❌ FOUND THE BUG!")
                        print(f"  Accuracy = {accuracy:.1f}% << 20%")
                        print(f"  → Predictions and labels are MISALIGNED")
                        print(f"  → Queries have WRONG coordinates")
                    elif accuracy < 50:
                        print(f"\n  ⚠️  Accuracy = {accuracy:.1f}% is suspiciously low")
                    else:
                        print(f"\n  ✓ Alignment looks OK ({accuracy:.1f}%)")
                
                # ─────────────────────────────────────────────────────────
                # 5. Confusion matrix
                # ─────────────────────────────────────────────────────────
                print(f"\n[5] Confusion matrix:")
                
                for true_cls in range(self.num_classes):
                    true_mask = (labels_valid == true_cls)
                    n_true = true_mask.sum().item()
                    
                    if n_true > 0:
                        preds_for_true = preds_valid[true_mask]
                        print(f"\n  When true label = {true_cls} ({n_true} pixels):")
                        
                        for pred_cls in range(self.num_classes):
                            n_pred = (preds_for_true == pred_cls).sum().item()
                            pct = 100 * n_pred / n_true
                            print(f"    Predicted {pred_cls}: {n_pred}/{n_true} ({pct:.1f}%)")
                
                # ─────────────────────────────────────────────────────────
                # 6. Check query coordinates
                # ─────────────────────────────────────────────────────────
                print(f"\n[6] Query coordinates:")
                
                query_x = batch["queries"][0, :10, 1]  # First 10 queries, x coord
                query_y = batch["queries"][0, :10, 2]  # First 10 queries, y coord
                
                print(f"  First 10 x coords: {query_x.tolist()}")
                print(f"  First 10 y coords: {query_y.tolist()}")
                print(f"  X range: [{batch['queries'][:, :, 1].min():.4f}, {batch['queries'][:, :, 1].max():.4f}]")
                print(f"  Y range: [{batch['queries'][:, :, 2].min():.4f}, {batch['queries'][:, :, 2].max():.4f}]")
                
                # Check if normalized
                max_coord = max(batch['queries'][:, :, 1].max(), batch['queries'][:, :, 2].max())
                if max_coord > 1.0:
                    print(f"  ⚠️  Coordinates > 1.0 (pixel space, not normalized)")
                else:
                    print(f"  ✓ Coordinates are normalized [0, 1]")
            
            print("\n" + "="*70 + "\n")
        
        # ═════════════════════════════════════════════════════════════════════
        # NORMAL TEST STEP (after debug)
        # ═════════════════════════════════════════════════════════════════════
        
        # ── Sliding window path ─────────────────────────────────────────────
        if batch.get("sliding", False):
            preds_full, label_full, logits_avg = self._sliding_window_step(batch)

            loss = self.loss(
                logits_avg.unsqueeze(0),
                label_full.unsqueeze(0),
            )

            valid = (label_full != self.ignore_index)
            if valid.sum() > 0:
                self.metric_IoU_test.update(preds_full[valid], label_full[valid])
                self.metric_acc_test.update(preds_full[valid], label_full[valid])

            self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
            return loss

        # ── Normal path ─────────────────────────────────────────────────────
        total_loss, class_loss, preds, labels = self._compute_loss_and_preds(batch, training=False)

        self.metric_IoU_test.update(preds, labels)
        self.metric_acc_test.update(preds, labels)

        self.log("test_loss", class_loss, on_step=False, on_epoch=True, logger=True)

        return class_loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================

    def on_train_epoch_end(self):
        self.log("train_mIoU", self.metric_IoU_train.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.metric_acc_train.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        self.log("val_mIoU", self.metric_IoU_val.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", self.metric_acc_val.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        test_iou = self.metric_IoU_test.compute()
        test_acc = self.metric_acc_test.compute()

        self.log("test_mIoU", test_iou.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc.mean(), on_epoch=True, logger=True)

        for i, name in enumerate(["no_flood", "flood"]):
            if i < len(test_iou):
                self.log(f"test_IoU_{name}", test_iou[i], on_epoch=True, logger=True)
            if i < len(test_acc):
                self.log(f"test_acc_{name}", test_acc[i], on_epoch=True, logger=True)

        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()

    # =========================================================================
    # MODEL SAVE/LOAD
    # =========================================================================

    def save_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        print(f"[SenFlood] Model saved to {file_path}")

    def load_model(self, name=None):
        suffix = f"_{name}" if name else ""
        file_path = f"./pth_files/{self.config['encoder']}_{self.name}{suffix}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"[SenFlood] Model loaded from {file_path}")

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1000, max(1, int(0.05 * total_steps)))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }