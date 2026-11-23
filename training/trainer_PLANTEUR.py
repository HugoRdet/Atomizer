"""
PLANTEUR Crop Segmentation Trainer
==================================

Same model selection, loss, optimiser and schedule as PASTISTrainer. Only the
metric convention changes, following the PLANTEUR benchmark card:

    mIoU = mean IoU over ALL 8 classes (0..7, background included); void (255)
    is ignored; a class absent from the evaluated patches counts as IoU 0
    (matters for leave-one-island-out, e.g. pineapple outside Guadeloupe).

PASTIS instead averages over classes 1..18 (background excluded).
"""

import torch

from training.trainer_PASTIS import PASTISTrainer


class PlanteurTrainer(PASTISTrainer):

    CROP_NAMES = [
        "background", "meadow", "fruits_veg_flowers", "orchard",
        "sugarcane", "banana", "pineapple", "tropical_tuber",
    ]

    def __init__(self, config, wand, name, transform, lookup_table):
        super().__init__(config, wand, name, transform, lookup_table)
        if self.num_classes != len(self.CROP_NAMES):
            raise ValueError(f"PLANTEUR has {len(self.CROP_NAMES)} classes, "
                             f"config trainer.num_classes={self.num_classes}")

    # absent classes -> 0 (torchmetrics' zero_division default; nan_to_num as a guard)
    def _compute_crop_miou(self, per_class_iou: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(per_class_iou[:len(self.CROP_NAMES)], nan=0.0).mean()

    def _compute_crop_acc(self, per_class_acc: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(per_class_acc[:len(self.CROP_NAMES)], nan=0.0).mean()

    def on_test_epoch_end(self):
        iou = self.metric_IoU_test.compute()
        acc = self.metric_acc_test.compute()
        miou, macc = self._compute_crop_miou(iou), self._compute_crop_acc(acc)

        self.log("test_mIoU", miou, on_epoch=True, logger=True)
        self.log("test_accuracy", macc, on_epoch=True, logger=True)
        for i, name in enumerate(self.CROP_NAMES):
            self.log(f"test_IoU_{name}", iou[i], on_epoch=True, logger=True)
            self.log(f"test_acc_{name}", acc[i], on_epoch=True, logger=True)

        print(f"\n{'='*60}\nTEST RESULTS (mIoU = mean over all 8 classes)\n{'='*60}")
        print(f"  mIoU:     {miou:.4f}\n  Accuracy: {macc:.4f}")
        for i, name in enumerate(self.CROP_NAMES):
            print(f"  IoU {name}: {iou[i]:.4f}")
        print(f"{'='*60}\n")

        self.metric_IoU_test.reset()
        self.metric_acc_test.reset()
