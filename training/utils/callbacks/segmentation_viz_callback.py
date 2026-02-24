"""
Segmentation Visualization Callback
=====================================
Plots RGB image, GT segmentation mask, and predicted segmentation mask.
Uses get_viz_sample from the dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from training.utils.datasets.token_grouping import collate_grouped


class SegmentationVizCallback(pl.Callback):

    RGB_INDICES = [3, 2, 1]

    def __init__(
        self,
        sample_indices=(0, 1, 2),
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
        class_names=None,
    ):
        super().__init__()
        self.sample_indices = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb = use_wandb
        self.ignore_index = ignore_index
        self.class_names = class_names or ["no_flood", "flood"]

    def on_train_epoch_end(self, trainer, pl_module):
        print("ayoooooo")
        if trainer.global_rank != 0:
            print("[SEG VIZ] Not rank 0, skipping")
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            print("[SEG VIZ] Not logging epoch, skipping")
            return

        dataset = getattr(trainer.datamodule, "val_dataset", None)
        print(f"[SEG VIZ] val_dataset: {type(dataset)}")
        if dataset is None:
            dataset = getattr(trainer.datamodule, "train_dataset", None)
        print(f"[SEG VIZ] Using dataset: {type(dataset)}, has get_viz_sample: {hasattr(dataset, 'get_viz_sample')}")
        if dataset is None or not hasattr(dataset, "get_viz_sample"):
            print("[SEG VIZ] No get_viz_sample, returning")
            return

        print(f"[SEG VIZ] Starting viz for {len(self.sample_indices)} samples")

        device = pl_module.device
        pl_module.eval()

        figures = []

        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue

            try:
                sample = dataset.get_viz_sample(idx)

                H, W = sample["label"].shape
                image = sample["image"]  # [C, H, W]

                batch = collate_grouped([sample])
                batch = _batch_to_device(batch, device)

                with torch.no_grad():
                    result = pl_module(batch, training=False)

                if isinstance(result, dict):
                    y_hat = result["predictions"]
                else:
                    y_hat = result

                # [1, M, num_classes] → [M]
                preds = torch.argmax(y_hat, dim=-1).squeeze(0).cpu()
                pred_2d = preds[:H * W].reshape(H, W)

                label_2d = sample["label"]  # [H, W]

                # RGB
                rgb = image[self.RGB_INDICES]
                rgb = self._normalize_rgb(rgb)

                fig = self._make_figure(
                    rgb, label_2d.numpy(), pred_2d.numpy(),
                    idx, trainer.current_epoch
                )
                figures.append((f"seg_sample_{idx}", fig))

            except Exception as e:
                import traceback
                print(f"[SEG VIZ] Failed on sample {idx}: {e}")
                traceback.print_exc()

        if self.use_wandb and figures:
            import wandb
            for name, fig in figures:
                wandb.log({name: wandb.Image(fig)})
            plt.close("all")

        pl_module.train()

    @staticmethod
    def _normalize_rgb(rgb):
        rgb = rgb.numpy() if isinstance(rgb, torch.Tensor) else rgb
        for c in range(3):
            lo = np.percentile(rgb[c], 2)
            hi = np.percentile(rgb[c], 98)
            if hi - lo > 1e-6:
                rgb[c] = (rgb[c] - lo) / (hi - lo)
            else:
                rgb[c] = 0.0
        return np.clip(rgb, 0, 1)

    def _make_figure(self, rgb, label, pred, sample_idx, epoch):
        n_classes = len(self.class_names)
        cmap = plt.cm.get_cmap("tab10", n_classes)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.transpose(rgb, (1, 2, 0)))
        axes[0].set_title("RGB")
        axes[0].axis("off")

        masked_label = np.ma.masked_where(label == self.ignore_index, label)
        axes[1].imshow(masked_label, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[1].set_title("GT Label")
        axes[1].axis("off")

        im = axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        valid = label != self.ignore_index
        if valid.sum() > 0:
            acc = (pred[valid] == label[valid]).mean() * 100
            fig.suptitle(f"Sample {sample_idx} — Epoch {epoch} — Acc: {acc:.1f}%", fontsize=14)
        else:
            fig.suptitle(f"Sample {sample_idx} — Epoch {epoch}", fontsize=14)

        fig.tight_layout()
        return fig


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out