"""
BioMassters Baseline Regression Trainer
=========================================

PyTorch Lightning trainer for fixed-format baseline models (ResNet+double-
conv, ViT+LTAE, RAMEN) on BioMassters AGB regression.

Mirrors BaselineTrainer's structure, adapted for:

  1. REGRESSION instead of segmentation. Loss is plain MSE (matches the
     "let's just use a regular MSE" decision for the Atomizer trainer --
     kept consistent here for a fair comparison). Model output is
     [B, 1, H, W] (single channel), not [B, num_classes, H, W].

  2. Metrics: RMSE + MAE via torchmetrics.regression, computed only over
     valid pixels (target != IGNORE_VALUE), instead of IoU/Accuracy/
     per-class IoU. No class registry needed.

  3. batch["target"] is [B, 1, H, W] float (AGB, Mg/ha), not [B, H, W] long
     class ids -- see BioMasstersBaselineDataset's docstring note on this
     format difference from PastisBaselineDataset.

  4. Input extraction (`_get_image`): CORRECTED to match PASTIS's original
     BaselineTrainer behavior -- most baselines (ResNet+double-conv, ViT+LTAE)
     expect a single tensor selected via self.modality (default "s2", which
     after the fused collate holds the S2+S1-concatenated tensor), NOT a
     dict. Only models that opt in via `expects_full_image_dict = True`
     (e.g. RAMEN) get the full {"s2":..., "s1":...} dict passed through.
     An earlier version of this trainer always returned the full dict,
     assuming all three baselines needed it -- that broke ResNet/ViT+LTAE,
     whose forward() calls x.dim() on what it expects to be a plain tensor.

  5. No sliding-window inference: BioMassters tiles are already only 256x256
     (matching Atomizer's SAT_RESOLUTION setup), and none of the three
     baselines here are described as intractable at that size the way
     RAMEN's pixel-level tokenization was flagged for PASTIS/other tasks.
     ASSUMPTION -- if RAMEN's memory footprint is still a problem at 256x256
     with BioMassters' channel/timestep count, the window_size/window_stride
     mechanism from BaselineTrainer can be ported over the same way; omitted
     here since I don't know if it's actually needed for this task/scale.

Batch format (from BioMasstersBaselineDataset):
    {
        "image":    {"s2": [B, C*T, H, W] or [B, T, C, H, W],
                     "s1": [B, C*T, H, W] or [B, T, C, H, W]},
        "dates":    {"s2": [B, T], "s1": [B, T]},   (day-of-year, for LTAE)
        "target":   [B, 1, H, W] float,
        "metadata": [list of dicts],
    }

Architecture:
    Non-temporal (ResNet+double-conv): model(image_dict) -> [B, 1, H, W]
    Temporal (ViT+LTAE):  model(image_dict, doy=dates_dict) -> [B, 1, H, W]
    RAMEN:                model(image_dict) -> [B, 1, H, W]
                           (same expects_full_image_dict convention as
                           BaselineTrainer -- RAMENUPerNet looks up a
                           separate spectral projector per modality
                           internally, unchanged from the segmentation case)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup


IGNORE_VALUE = -1.0  # AGB (Mg/ha) is never negative -- matches trainer_biomassters.py


class BaselineRegressionTrainer(pl.LightningModule):
    """
    Single-task AGB regression trainer for baseline models (ResNet+double-
    conv, ViT+LTAE, RAMEN) on BioMassters.

    Parameters
    ----------
    model : nn.Module
        Regression model.
        Non-temporal: dict[modality] -> Tensor -> [B, 1, H, W]
        Temporal:     dict[modality] -> Tensor, doy=dict[modality] -> [B,T]
                      -> [B, 1, H, W]
    temporal : bool
        If True, pass doy to model (expects [B, T, C, H, W] per-modality
        input, i.e. BioMasstersBaselineDataset's temporal_mode="sequence").
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    ignore_value : float
        AGB sentinel value to exclude from loss/metrics (invalid/padding
        pixels). Matches BioMasstersBaselineDataset/trainer_biomassters.py's
        IGNORE_VALUE = -1.0.
    """

    def __init__(
        self,
        model: nn.Module,
        modality: str = "s2",
        temporal: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        ignore_value: float = IGNORE_VALUE,
        agb_mean: float = None,
        agb_std: float = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.modality = modality
        self.temporal = temporal
        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_value = ignore_value

        # Target normalization: plain z-score -- SAME transform and SAME
        # stats (read from normalization_stats.pt, computed by
        # BioMasstersSkipDataset) as trainer_biomassters.py, so baselines and
        # Atomizer are compared on identical footing. Required, not optional
        # -- see that trainer's __init__ for the rationale (originally
        # log1p+z-score, switched to plain z-score after val_RMSE: inf was
        # observed in practice -- see that trainer's docstring for the
        # mechanism).
        if agb_mean is None or agb_std is None:
            raise ValueError(
                "agb_mean/agb_std must be provided (read from "
                "normalization_stats.pt via the dataset's norm_stats, e.g. "
                "train_dataset.norm_stats['agb_mean']). Target "
                "normalization (z-score) is required, not optional, "
                "to match trainer_biomassters.py's convention."
            )
        self.register_buffer("agb_mean", torch.tensor(float(agb_mean)))
        self.register_buffer("agb_std",  torch.tensor(float(agb_std)))

        # ── Loss ────────────────────────────────────────────────────
        self.loss_fn = nn.MSELoss(reduction="none")

        # ── Metrics ─────────────────────────────────────────────────
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_RMSE", torchmetrics.MeanSquaredError(squared=False))
            setattr(self, f"{split}_MAE", torchmetrics.MeanAbsoluteError())

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode_str = "temporal" if temporal else "standard"
        print(f"[BaselineRegressionTrainer] task='biomassters', modality='{modality}', "
              f"mode={mode_str}, params={param_count:,}")

    # ─────────────────────────────────────────────────────────────────
    # Target normalization (plain z-score) -- identical to trainer_biomassters.py
    # ─────────────────────────────────────────────────────────────────

    def _transform_target(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.agb_mean) / self.agb_std.clamp(min=1e-6)

    def _inverse_transform(self, normalized: torch.Tensor) -> torch.Tensor:
        """See trainer_biomassters.py's _inverse_transform for the full
        rationale. Linear transform -- clamp here is just a sane physical
        ceiling, not load-bearing numerical-stability machinery the way it
        was under the earlier log1p+expm1 version."""
        MAX_PLAUSIBLE_AGB = 2000.0  # Mg/ha
        pred = normalized * self.agb_std + self.agb_mean
        return pred.clamp(min=0.0, max=MAX_PLAUSIBLE_AGB)

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, image, doy=None):
        if self.temporal and doy is not None:
            # Different models expect the temporal-positions kwarg under
            # different names: ViT+LTAE wants `doy`, RAMENUPerNet wants
            # `dates` (see model_ramen_upernet.py's forward signature).
            # `temporal_kwarg` is set on the model instance by build_model()
            # for models that deviate from the `doy` default -- avoids
            # hardcoding a model-specific kwarg name here.
            temporal_kwarg = getattr(self.model, "temporal_kwarg", "doy")
            return self.model(image, **{temporal_kwarg: doy})
        return self.model(image)

    # ─────────────────────────────────────────────────────────────────
    # Input extraction
    # ─────────────────────────────────────────────────────────────────

    def _get_image(self, batch):
        """
        CORRECTED: most baselines (ResNet+double-conv, ViT+LTAE) expect a
        single tensor, NOT a dict -- e.g. model_resnet_upernet.py's forward
        does `x.dim()`, which crashes on a dict. Only models that explicitly
        opt in via `expects_full_image_dict = True` (e.g. RAMEN, which looks
        up a separate spectral projector per modality internally) get the
        full dict. This matches PASTIS's original BaselineTrainer._get_image
        exactly -- an earlier version of this trainer incorrectly always
        returned the full dict, assuming all three baselines needed it.

        Since the fused collate (make_fused_collate) merges S2+S1 into a
        single "s2" key before this trainer ever sees the batch, selecting
        batch["image"]["s2"] already gives ResNet/ViT+LTAE the fused
        [B, T, 15, H, W] tensor they expect.
        """
        if getattr(self.model, "expects_full_image_dict", False):
            return batch["image"]  # dict[modality] -> Tensor, passed through as-is
        return batch["image"][self.modality]  # [B, T, C, H, W] or [B, C, H, W]

    # ─────────────────────────────────────────────────────────────────
    # Shared step
    # ─────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, split: str):
        image = self._get_image(batch)
        raw_target = batch["target"]  # [B, 1, H, W] float, RAW Mg/ha

        doy = None
        if self.temporal and "dates" in batch:
            if getattr(self.model, "expects_full_image_dict", False):
                doy = batch["dates"]  # dict[modality] -> [B, T], matches full-dict image
            else:
                doy = batch["dates"].get(self.modality)  # [B, T], matches single-tensor image

        preds_norm = self.forward(image, doy=doy)  # [B, 1, H, W] -- NORMALIZED-space prediction

        # NOTE: PerceiverSeg was patched to query only the most recent
        # frame's tokens (see perceiver_seg.py), so it now always returns
        # 4D [B, num_classes, H, W] regardless of T -- no per-model shape
        # handling needed here anymore. If a future model genuinely needs
        # different treatment, add an explicit, model-specific branch here
        # rather than a generic squeeze/aggregate (see git history for why:
        # a generic squeeze silently mishandled PerceiverSeg's real T=3
        # per-timestep output).
        if preds_norm.dim() != 4:
            raise RuntimeError(
                f"preds_norm has unexpected shape {tuple(preds_norm.shape)} "
                f"({preds_norm.dim()}D) -- expected 4D [B, C, H, W]. This "
                f"model's output shape needs explicit handling here."
            )

        # Handle spatial size mismatch
        if preds_norm.shape[2:] != raw_target.shape[2:]:
            preds_norm = nn.functional.interpolate(
                preds_norm, size=raw_target.shape[2:],
                mode="bilinear", align_corners=False,
            )

        valid_mask = raw_target != self.ignore_value  # [B, 1, H, W] bool, computed on RAW target

        preds_norm_flat = preds_norm.reshape(-1)
        raw_target_flat = raw_target.reshape(-1)
        mask_flat       = valid_mask.reshape(-1)

        # Loss in normalized (plain z-score) space.
        norm_target_flat = self._transform_target(raw_target_flat)
        per_elem_loss = self.loss_fn(preds_norm_flat, norm_target_flat)
        if mask_flat.any():
            loss = per_elem_loss[mask_flat].mean()
        else:
            # Degenerate batch with no valid AGB pixels -- zero loss rather
            # than NaN from an empty mean (shouldn't occur on train/val given
            # this dataset's chips missing AGBM: 0%, but kept for safety).
            loss = per_elem_loss.sum() * 0.0

        # Metrics in real Mg/ha: inverse-transform predictions, compare
        # against the untouched raw target.
        preds_flat = self._inverse_transform(preds_norm_flat)

        if mask_flat.any():
            rmse_metric = getattr(self, f"{split}_RMSE")
            mae_metric  = getattr(self, f"{split}_MAE")
            rmse_metric.update(preds_flat[mask_flat], raw_target_flat[mask_flat])
            mae_metric.update(preds_flat[mask_flat], raw_target_flat[mask_flat])

        self.log(f"{split}_loss", loss,
                 on_step=(split == "train"),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=(split != "train"))

        return loss

    # ─────────────────────────────────────────────────────────────────
    # Train / Val / Test steps
    # ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "test")

    # ─────────────────────────────────────────────────────────────────
    # Epoch end — log metrics
    # ─────────────────────────────────────────────────────────────────

    def _on_epoch_end(self, split: str):
        rmse_metric = getattr(self, f"{split}_RMSE")
        mae_metric  = getattr(self, f"{split}_MAE")

        rmse = rmse_metric.compute()
        mae  = mae_metric.compute()

        self.log(f"{split}_RMSE", rmse,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{split}_MAE", mae,
                 on_epoch=True, logger=True, sync_dist=True)

        rmse_metric.reset()
        mae_metric.reset()

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    # ─────────────────────────────────────────────────────────────────
    # Optimizer
    # ─────────────────────────────────────────────────────────────────

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
