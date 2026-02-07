"""
U-Net Baseline for Sen1Floods11
================================
Simple U-Net for flood segmentation baseline comparison with Atomizer.
- Input: [B, 15, H, W] (13 S2 + 2 S1 bands)
- Output: [B, 2, H, W] (2 classes: no flood, flood)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup


# =============================================================================
# U-Net Architecture
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    
    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=15, num_classes=2, base_features=64):
        super().__init__()
        f = base_features
        
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)
        
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)
        
        self.outc = nn.Conv2d(f, num_classes, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


# =============================================================================
# Lightning Module
# =============================================================================

class Model_UNet_SenFlood(pl.LightningModule):
    def __init__(self, config, wand=True, name="unet"):
        super().__init__()
        self.config = config
        self.wand = wand
        self.name = name
        
        self.num_classes = 2
        self.ignore_index = 255
        
        # Model
        base_features = config.get("UNet", {}).get("base_features", 64)
        self.encoder = UNet(
            in_channels=15,
            num_classes=self.num_classes,
            base_features=base_features,
        )
        
        # Loss
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])
        
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
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[UNet] Initialized: {trainable_params:,} trainable params ({total_params:,} total)")
        print(f"[UNet] base_features={base_features}, in_channels=15, num_classes=2")

    # =========================================================================
    # FORWARD
    # =========================================================================
    
    def forward(self, image):
        return self.encoder(image)  # [B, 2, H, W]

    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def training_step(self, batch, batch_idx):
        image, label = batch  # [B, 15, H, W], [B, H, W]
        
        logits = self(image)  # [B, 2, H, W]
        loss = self.loss(logits, label)
        
        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        self.metric_IoU_train.update(preds, label)
        self.metric_acc_train.update(preds, label)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validation_step(self, batch, batch_idx):
        image, label = batch
        
        logits = self(image)
        loss = self.loss(logits, label)
        
        preds = torch.argmax(logits, dim=1)
        self.metric_IoU_val.update(preds, label)
        self.metric_acc_val.update(preds, label)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # =========================================================================
    # TEST
    # =========================================================================
    
    def test_step(self, batch, batch_idx):
        image, label = batch
        
        logits = self(image)
        loss = self.loss(logits, label)
        
        preds = torch.argmax(logits, dim=1)
        self.metric_IoU_test.update(preds, label)
        self.metric_acc_test.update(preds, label)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss

    # =========================================================================
    # EPOCH END HOOKS
    # =========================================================================
    
    def on_train_epoch_end(self):
        train_iou = self.metric_IoU_train.compute()
        train_acc = self.metric_acc_train.compute()
        self.log("train_mIoU", train_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_train.reset()
        self.metric_acc_train.reset()

    def on_validation_epoch_end(self):
        val_iou = self.metric_IoU_val.compute()
        val_acc = self.metric_acc_val.compute()
        self.log("val_mIoU", val_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.metric_IoU_val.reset()
        self.metric_acc_val.reset()

    def on_test_epoch_end(self):
        test_iou_per_class = self.metric_IoU_test.compute()
        test_acc_per_class = self.metric_acc_test.compute()
        
        self.log("test_mIoU", test_iou_per_class.mean(), on_epoch=True, logger=True)
        self.log("test_accuracy", test_acc_per_class.mean(), on_epoch=True, logger=True)
        
        class_names = ["no_flood", "flood"]
        for i, name in enumerate(class_names):
            self.log(f"test_IoU_{name}", test_iou_per_class[i], on_epoch=True, logger=True)
            self.log(f"test_acc_{name}", test_acc_per_class[i], on_epoch=True, logger=True)
        
        print(f"\n[UNet Test Results]")
        print(f"  mIoU: {test_iou_per_class.mean():.4f}")
        print(f"  Accuracy: {test_acc_per_class.mean():.4f}")
        for i, name in enumerate(class_names):
            print(f"  IoU ({name}): {test_iou_per_class[i]:.4f}")
            print(f"  Acc ({name}): {test_acc_per_class[i]:.4f}")
        
        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()

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