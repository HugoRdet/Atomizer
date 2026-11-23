"""Metrics with explicit conventions (doc §4.1): report both present-class and all-class mIoU."""
from __future__ import annotations

import torch


class SegMetrics:
    def __init__(self, k: int, ignore_index: int = -1):
        self.k, self.ignore = k, ignore_index
        self.cm = torch.zeros(k, k, dtype=torch.long)

    def update(self, logits, y):                      # logits [B,K,H,W], y [B,H,W]
        p = logits.argmax(1).flatten().cpu(); y = y.flatten().cpu()
        m = y != self.ignore
        self.cm += torch.bincount(y[m] * self.k + p[m], minlength=self.k ** 2).view(self.k, self.k)

    def compute(self):
        cm = self.cm.double(); tp = cm.diag()
        gt, pr = cm.sum(1), cm.sum(0)
        iou = tp / (gt + pr - tp).clamp(min=1)
        f1 = 2 * tp / (gt + pr).clamp(min=1)
        present = gt > 0
        return {
            "mIoU": iou[present].mean().item() * 100,           # PANGAEA-comparable: classes present in labels
            "mIoU_all_classes": iou.mean().item() * 100,
            "mF1": f1[present].mean().item() * 100,
            "OA": (tp.sum() / cm.sum().clamp(min=1)).item() * 100,
            "IoU_per_class": (iou * 100).tolist(),
        }


def cls_accuracy(pred, y):
    return (pred == y).float().mean().item() * 100
