"""
FRACTAL PTv3 Training Script
============================

PTv3 (random init) on FRACTAL LiDAR + VHR, 7-class point segmentation.
Data handling matches FractalPerceiverDataset; loss/metrics/optimizer
match Model_PerceiverFractal.

    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH

    python script_train_fractal_ptv3.py --xp_name ptv3_fractal \
        --data_dir ./data --valid_patches_file ./data/valid_patches.json \
        --batch_size 8 --lr 1e-4 --epochs 80

    # test only, scoring every LiDAR point of each patch:
    python script_train_fractal_ptv3.py --xp_name ptv3_fractal \
        --data_dir ./data --test_only path/to.ckpt --eval_full_scene
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

# Adjust this import: the dataset file must sit in the same package as
# FractalPerceiverDataset (it imports .augmentations from there).
from training.utils.datasets.utils_dataset_fractal_ptv3 import (
    FractalPTv3Dataset, fractal_ptv3_collate,
)
from training.trainer_ptv3_fractal import Model_PTv3Fractal


# =============================================================================
# ARGS
# =============================================================================

p = argparse.ArgumentParser(description="FRACTAL PTv3 training")
p.add_argument("--xp_name", type=str, required=True)
p.add_argument("--data_dir", type=str, default="./data",
               help="Root containing FRACTAL/ and FRACTAL-IRGB/.")
p.add_argument("--valid_patches_file", type=str, default=None,
               help="Same allow-list JSON as the PerceiverIO runs.")
p.add_argument("--test_only", type=str, default=None)
p.add_argument("--resume", type=str, default=None)

p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight_decay", type=float, default=1e-2)
p.add_argument("--epochs", type=int, default=80)
p.add_argument("--num_workers", type=int, default=8)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--grad_accum", type=int, default=1)
p.add_argument("--precision", type=str, default="bf16-mixed")
p.add_argument("--apply_class_weights", action="store_true",
               help="Use inverse-frequency CE weights. Off by default, "
                    "matching Model_PerceiverFractal as written.")

# data
p.add_argument("--max_points", type=int, default=16_000,
               help="Context points per patch (PerceiverIO: max_lidar_points).")
p.add_argument("--grid_size", type=float, default=0.1,
               help="Voxel size in metres. Watch *_voxel_keep_ratio: "
                    "well below 1 means many points share a voxel.")
p.add_argument("--no_vhr", action="store_true",
               help="LiDAR-only run: no ortho colours (5 input features "
                    "instead of 9). Same patches as the VHR runs.")
p.add_argument("--sigma_xy_pixels", type=float, default=0.25)
p.add_argument("--sigma_z_normed", type=float, default=0.003)
p.add_argument("--eval_full_scene", action="store_true",
               help="Test: score ALL LiDAR points of each patch.")
p.add_argument("--full_scene_context", type=str, default="subsample",
               choices=["subsample", "all"],
               help="subsample: model sees max_points points (like "
                    "PerceiverIO), others take their nearest neighbour's "
                    "prediction. all: model sees every point.")

# PTv3
p.add_argument("--ptv3_patch", type=int, default=1024)
p.add_argument("--ptv3_enc_channels", type=int, nargs=5, default=[32, 64, 128, 256, 512])
p.add_argument("--ptv3_enc_depths", type=int, nargs=5, default=[2, 2, 2, 6, 2])
p.add_argument("--ptv3_dec_channels", type=int, nargs=4, default=[64, 64, 128, 256])
p.add_argument("--ptv3_dec_depths", type=int, nargs=4, default=[2, 2, 2, 2])
p.add_argument("--ptv3_head_dim", type=int, default=16)
p.add_argument("--ptv3_drop_path", type=float, default=0.3)

args = p.parse_args()

_bad = [c for c in (*args.ptv3_enc_channels, *args.ptv3_dec_channels)
        if c % args.ptv3_head_dim]
if _bad:
    raise ValueError(f"PTv3 widths {_bad} not divisible by --ptv3_head_dim "
                     f"({args.ptv3_head_dim}).")
if args.resume is not None and not os.path.isfile(args.resume):
    raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

print(f"\n{'=' * 60}")
print(f"  FRACTAL PTv3")
print(f"  Inputs:      LiDAR{'' if args.no_vhr else ' + VHR colours'}")
print(f"  Grid size:   {args.grid_size} m   Context: {args.max_points} pts")
print(f"  Enc ch:      {tuple(args.ptv3_enc_channels)}  patch={args.ptv3_patch}")
print(f"  Test:        {'full scene (' + args.full_scene_context + ')' if args.eval_full_scene else 'context points'}")
print(f"  BS / LR:     {args.batch_size} / {args.lr}   Precision: {args.precision}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"{'=' * 60}\n")


# =============================================================================
# DATA
# =============================================================================

common = dict(
    root_path=args.data_dir,
    max_points=args.max_points,
    grid_size=args.grid_size,
    valid_patches_file=args.valid_patches_file,
    sigma_xy_pixels=args.sigma_xy_pixels,
    sigma_z_normed=args.sigma_z_normed,
    use_vhr=not args.no_vhr,
)
train_ds = FractalPTv3Dataset(mode="train", use_augmentation=True, **common)
val_ds   = FractalPTv3Dataset(mode="val", use_augmentation=False, **common)
test_ds  = FractalPTv3Dataset(mode="test", use_augmentation=False,
                              eval_full_scene=args.eval_full_scene,
                              full_scene_context=args.full_scene_context,
                              **common)

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=fractal_ptv3_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL
# =============================================================================

module = Model_PTv3Fractal(
    in_channels=train_ds.num_features,      # 9 with VHR, 5 with --no_vhr
    grid_size=args.grid_size,
    patch_size=args.ptv3_patch,
    enc_channels=tuple(args.ptv3_enc_channels),
    enc_depths=tuple(args.ptv3_enc_depths),
    dec_channels=tuple(args.ptv3_dec_channels),
    dec_depths=tuple(args.ptv3_dec_depths),
    head_dim=args.ptv3_head_dim,
    drop_path=args.ptv3_drop_path,
    lr=args.lr,
    weight_decay=args.weight_decay,
    apply_class_weights=args.apply_class_weights,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        run_name = f"PTv3_{args.xp_name}_g{args.grid_size}"
        if args.no_vhr:
            run_name += "_lidar_only"
        wandb.init(name=run_name,
                   project="Atomizer_Fractal_Baselines", config=vars(args))
        wandb_logger = WandbLogger(project="Atomizer_Fractal_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN
# =============================================================================

ckpt_dir = "./checkpoints/fractal_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir,
                        filename=f"ptv3_{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
                        monitor="val_mIoU", mode="max", save_top_k=1, verbose=True),
        ModelCheckpoint(dirpath=ckpt_dir, filename=f"ptv3_{args.xp_name}-last",
                        every_n_epochs=1, save_top_k=1, save_last=True),
        EarlyStopping(monitor="val_mIoU", mode="max", patience=args.patience, verbose=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=-1,
        accelerator="gpu",
        max_epochs=args.epochs,
        precision=args.precision,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
    best_ckpt = trainer.checkpoint_callback.best_model_path

    import torch.distributed as dist
    is_rank_zero = trainer.is_global_zero
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if not is_rank_zero:
        if wandb_logger:
            import wandb
            wandb.finish()
        raise SystemExit(0)
else:
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(f"--test_only checkpoint not found: {args.test_only}")
    best_ckpt = args.test_only


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================

print(f"\n{'=' * 60}\n  Testing checkpoint: {best_ckpt}\n{'=' * 60}\n")
test_trainer = Trainer(devices=1, accelerator="gpu", precision=args.precision,
                       logger=wandb_logger, default_root_dir=ckpt_dir)
test_trainer.test(module, test_loader, ckpt_path=best_ckpt)

if wandb_logger:
    import wandb
    wandb.finish()
