#!/usr/bin/env python3
"""
Train baseline models on MultiEarth Deforestation Segmentation.

Experiments:
  1. Same-sensor:   --sensor s2                     (S2→S2)
  2. Same-sensor:   --sensor l8                     (L8→L8)
  3. Cross-sensor:  --sensor s2 --eval_sensor l8    (train S2, eval L8 interp)
  4. Cross-sensor:  --sensor l8 --eval_sensor s2    (train L8, eval S2 interp)

Models:
  --model unet      Standard UNet (temporal stacking)
  --model vit       ViT + UPerNet
  --model utae      U-TAE (temporal sequence model)

Usage:
  # Train UNet on S2
  python train_multiearth_baseline.py --model unet --sensor s2 \
      --batch_size 8 --lr 1e-3 --epochs 100 --xp_name unet_s2

  # Train UNet on L8
  python train_multiearth_baseline.py --model unet --sensor l8 \
      --batch_size 8 --lr 1e-3 --epochs 100 --xp_name unet_l8

  # Train on S2, evaluate cross-sensor on L8
  python train_multiearth_baseline.py --model unet --sensor s2 \
      --eval_sensor l8 --xp_name unet_s2_eval_l8
"""

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from training.utils.datasets_baselines.utils_dataset_multiearth_baseline import (
    MultiEarthBaselineDataset,
    NUM_S2_BANDS, NUM_L8_BANDS, NUM_CLASSES, IMG_SIZE,
)
from training.utils.datasets_baselines.collate import get_collate_fn
from training.trainer_baselines import BaselineTrainer


# ═══════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════

def build_model(model_name: str, in_channels: int, num_classes: int,
                temporal_mode: str = "stack", n_timesteps: int = 3):
    """Build segmentation model."""

    if model_name == "unet":
        from training.unet.model_unet import UNet
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            topology=(64, 128, 256, 512, 1024),
        )

    elif model_name == "vit":
        from training.VIT.model_vit_upernet import ViTLTAEUPerNet
        # ViT uses temporal mode: shared encoder per frame + LTAE fusion
        n_bands = in_channels // n_timesteps if temporal_mode == "stack" else in_channels
        return ViTLTAEUPerNet(
            in_channels=n_bands,
            num_classes=num_classes,
            img_size=256,
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            output_layers=(2, 5, 8, 11),
            decoder_channels=256,
            ltae_n_head=16,
            ltae_d_k=4,
            ltae_d_model=256,
        )

    elif model_name == "utae":
        # U-TAE uses sequence mode [B, T, C, H, W]
        from training.utae.utae import UTAE
        n_bands = in_channels // n_timesteps if temporal_mode == "stack" else in_channels
        return UTAE(
            input_dim=n_bands,
            encoder_widths=[64, 64, 128, 128],
            decoder_widths=[64, 64, 128, 128],
            out_conv=[32, num_classes],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=32,
            pad_value=0,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MultiEarth Baseline Training")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data/multi_earth/")
    parser.add_argument("--csv_path", type=str, default="multiearth_split.csv")
    parser.add_argument("--sensor", type=str, default="s2", choices=["s2", "l8"])
    parser.add_argument("--eval_sensor", type=str, default=None, choices=["s2", "l8", None],
                        help="Cross-sensor eval: load eval_sensor data, interpolate to train sensor grid")
    parser.add_argument("--n_timesteps", type=int, default=3)
    parser.add_argument("--temporal_mode", type=str, default="stack",
                        choices=["stack", "sequence"])

    # Model
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "vit", "utae"])
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    # Experiment
    parser.add_argument("--xp_name", type=str, default="multiearth_baseline")
    parser.add_argument("--project", type=str, default="multiearth")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ── Sensor config ───────────────────────────────────────────────
    n_bands = NUM_S2_BANDS if args.sensor == "s2" else NUM_L8_BANDS

    if args.temporal_mode == "stack":
        in_channels = n_bands * args.n_timesteps
    else:
        in_channels = n_bands  # U-TAE handles temporal dim internally

    temporal = (args.model in ("utae", "vit"))
    if temporal:
        args.temporal_mode = "sequence"
        in_channels = n_bands

    print(f"\n{'='*60}")
    print(f"MultiEarth Baseline Training")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Sensor:     {args.sensor} ({n_bands} bands)")
    print(f"  Eval sensor: {args.eval_sensor or 'same'}")
    print(f"  Temporal:   {args.temporal_mode} ({args.n_timesteps} steps)")
    print(f"  In channels: {in_channels}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    print(f"{'='*60}\n")

    # ── Datasets ────────────────────────────────────────────────────
    train_ds = MultiEarthBaselineDataset(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        split="train",
        sensor=args.sensor,
        n_timesteps=args.n_timesteps,
        temporal_mode=args.temporal_mode,
        augment=True,
    )

    val_ds = MultiEarthBaselineDataset(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        split="val",
        sensor=args.sensor,
        n_timesteps=args.n_timesteps,
        temporal_mode=args.temporal_mode,
        augment=False,
    )

    # Cross-sensor eval dataset (optional)
    cross_eval_ds = None
    if args.eval_sensor and args.eval_sensor != args.sensor:
        cross_eval_ds = MultiEarthBaselineDataset(
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            split="test",
            sensor=args.eval_sensor,
            cross_sensor_target=args.sensor,  # interpolate to training sensor grid
            n_timesteps=args.n_timesteps,
            temporal_mode=args.temporal_mode,
            augment=False,
        )

    test_ds = MultiEarthBaselineDataset(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        split="test",
        sensor=args.sensor,
        n_timesteps=args.n_timesteps,
        temporal_mode=args.temporal_mode,
        augment=False,
    )

    # ── Dataloaders ─────────────────────────────────────────────────
    modality = args.sensor
    collate_fn = get_collate_fn([modality])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Model ───────────────────────────────────────────────────────
    model = build_model(
        args.model, in_channels, args.num_classes,
        args.temporal_mode, args.n_timesteps,
    )

    trainer_module = BaselineTrainer(
        model=model,
        modality=modality,
        temporal=temporal,
        task="multiearth",
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
    )

    # ── Callbacks ───────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"./checkpoints/multi_earth/{args.xp_name}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        filename="best-{epoch}-{val_mIoU:.4f}",
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_mIoU", mode="max", patience=15, verbose=True,
    )

    logger = WandbLogger(
        project=args.project,
        name=args.xp_name,
        save_dir=args.log_dir,
    )

    # ── Trainer ─────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        logger=logger,
        log_every_n_steps=10,
    )

    # ── Train ───────────────────────────────────────────────────────
    trainer.fit(trainer_module, train_loader, val_loader)

    # ── GFLOPs count ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Model Complexity")
    print(f"{'='*60}")

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:  {param_count:>12,} ({param_count/1e6:.2f}M)")
    print(f"  Trainable:   {trainable:>12,} ({trainable/1e6:.2f}M)")

    try:
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.randn(1, in_channels, IMG_SIZE, IMG_SIZE, device=device)

        with torch.no_grad(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
        ) as prof:
            _ = model(dummy)

        # Sum all FLOPs from profiler events
        total_flops = sum(
            e.flops for e in prof.key_averages() if e.flops is not None and e.flops > 0
        )
        gflops = total_flops / 1e9
        print(f"  GFLOPs:      {gflops:>12.2f}")

        # Also show top-5 operators by FLOPs
        print(f"\n  Top operators:")
        events = sorted(
            [e for e in prof.key_averages() if e.flops and e.flops > 0],
            key=lambda e: e.flops, reverse=True,
        )
        for e in events[:5]:
            print(f"    {e.key:>40s}  {e.flops/1e9:>8.2f} GFLOPs")

    except Exception as e:
        print(f"  GFLOPs:      failed ({e})")

    print(f"{'='*60}")

    # ── Test (same sensor) ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Testing: {args.sensor}→{args.sensor} (same sensor)")
    print(f"{'='*60}")
    results_same = trainer.test(trainer_module, test_loader, ckpt_path="best")

    # Log same-sensor results to W&B with explicit prefix
    if results_same and logger.experiment:
        for k, v in results_same[0].items():
            logger.experiment.summary[f"same_sensor/{k}"] = v

    # ── Test (cross-sensor, if configured) ──────────────────────────
    if cross_eval_ds is not None:
        cross_loader = DataLoader(
            cross_eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )
        print(f"\n{'='*60}")
        print(f"Testing: {args.eval_sensor}→{args.sensor} (cross-sensor, spectral interp)")
        print(f"{'='*60}")
        results_cross = trainer.test(trainer_module, cross_loader, ckpt_path="best")

        # Log cross-sensor results with explicit prefix
        if results_cross and logger.experiment:
            for k, v in results_cross[0].items():
                logger.experiment.summary[f"cross_sensor/{k}"] = v

    # ── Final summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if results_same:
        miou = results_same[0].get("test_mIoU", "?")
        print(f"  {args.sensor}→{args.sensor}: mIoU={miou:.4f}" if isinstance(miou, float) else f"  {args.sensor}→{args.sensor}: mIoU={miou}")
    if cross_eval_ds is not None and results_cross:
        miou = results_cross[0].get("test_mIoU", "?")
        print(f"  {args.eval_sensor}→{args.sensor} (interp): mIoU={miou:.4f}" if isinstance(miou, float) else f"  {args.eval_sensor}→{args.sensor}: mIoU={miou}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()