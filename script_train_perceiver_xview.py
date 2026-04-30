"""
xView2 Perceiver-IO Single-Task Training Script
==================================================

Trains the multi-temporal Perceiver-IO baseline on xView2 — 5-class
post-disaster damage segmentation with bi-temporal RGB inputs.

Token layout (matches PerceiverSeg.forward):
    Per-token feature = [reflectance(C=3), Fourier(x, y), Fourier(DOY)]
    Input tokens     = [B, 2*H*W, C + pos_dim + time_dim]
    Output queries   = [B, H*W, query_dim]    # time-agnostic

Synthetic DOY scheme:
    xView2 has no calendar dates — pre/post is positional, not temporal.
    The collate synthesizes ordinal day-of-year values:
        pre  -> doy = 1   (Jan 1, normalized to ~-1.0)
        post -> doy = 183 (Jul 2, normalized to ~0.0)
    Half-year separation in Fourier-space lets the time encoder produce
    maximally distinguishable embeddings for the two frames. Day 1 (not 0)
    avoids collision with the padding sentinel used by PASTIS in the
    multi-task setup.

Same dataset protocol as the xView2 baseline:
    - PANGAEA splits (90/10 stratified train/val from train+tier3, test/ for test)
    - Same building-biased crop (random 512x512 with retry, building-centered fallback)
    - Same oversampling for damage classes during training
    - BGR ordering, PANGAEA's BGR mean/std normalization
    - 5 classes (0=BG, 1=NoDamage, 2=Minor, 3=Major, 4=Destroyed)

Examples:
    python script_train_xview_perceiver.py --xp_name perceiver_xview \
        --batch_size 2 --grad_accum 2 --lr 1e-4 --epochs 80

    # Test-only mode
    python script_train_xview_perceiver.py --xp_name perceiver_xview \
        --test_only ./checkpoints/xview_perceiver/bl_perceiver_xview-best.ckpt
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_xview_baseline import (
    XView2BaselineDataset,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CHANNELS = 3                # RGB (BGR ordering, kept channels-first)
MODALITY_KEY = "s2"             # baseline convention; collate keys output here
NUM_FRAMES   = 2                # pre + post

# Synthetic ordinal DOY for the two frames.
# pre  = day  1  -> Fourier-normalized to ~-1.0
# post = day 183 -> Fourier-normalized to ~ 0.0  (half-year separation)
PRE_DOY  = 1
POST_DOY = 183


# =============================================================================
# COLLATE
# =============================================================================

def xview_collate(batch):
    """
    Collate xView2 samples and synthesize per-sample DOY.

    The dataset returns:
        {"image": [T=2, C=3, H, W], "target": [H, W], "metadata": {...}}

    Output (to BaselineTrainer with temporal=True, modality='s2'):
        {
            "image":    {"s2": [B, 2, 3, H, W]},
            "dates":    {"s2": [B, 2]},          # ordinal [PRE_DOY, POST_DOY]
            "target":   [B, H, W],
            "metadata": [list],
        }
    """
    images   = torch.stack([s["image"]  for s in batch], dim=0)   # [B, 2, 3, H, W]
    targets  = torch.stack([s["target"] for s in batch], dim=0)
    metadata = [s["metadata"] for s in batch]

    B = images.shape[0]
    dates = torch.tensor([[PRE_DOY, POST_DOY]] * B, dtype=torch.long)

    return {
        "image":    {MODALITY_KEY: images},
        "dates":    {MODALITY_KEY: dates},
        "target":   targets,
        "metadata": metadata,
    }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="xView2 Perceiver-IO Baseline")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--data_dir",  type=str, default="./data/xView2")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Training
parser.add_argument("--batch_size",   type=int, default=2)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=2)

# Crop / image size — xView2 native 1024x1024, baseline crop is 512.
# 512^2 * T=2 = 524k tokens. 1024^2 * T=2 = 2M tokens (likely OOM).
parser.add_argument("--crop_size", type=int, default=512,
                    help="Crop size (random for train, building-biased center for eval).")
parser.add_argument("--img_size", type=int, default=512,
                    help="Spatial size for token construction. Must match crop_size.")

# Disable damage-class oversampling during training (matches baseline default behavior).
parser.add_argument("--no_oversample", action="store_true",
                    help="Disable building-damage oversampling at train time.")

# Perceiver-IO config (matches the PASTIS/MADOS/BurnScars/SenFlood runs)
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=768)
parser.add_argument("--depth",              type=int, default=6)
parser.add_argument("--cross_heads",        type=int, default=1)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=1)
parser.add_argument("--no_weight_tie",      action="store_true",
                    help="Disable weight-tying across encoder blocks.")
parser.add_argument("--num_freq_bands",     type=int, default=16)
parser.add_argument("--max_freq",           type=float, default=16.0)
parser.add_argument("--attn_dropout",       type=float, default=0.0)
parser.add_argument("--ff_dropout",         type=float, default=0.0)

args = parser.parse_args()


# =============================================================================
# SANITY
# =============================================================================

if args.crop_size != args.img_size:
    raise ValueError(
        f"--crop_size ({args.crop_size}) must equal --img_size ({args.img_size}). "
        f"The Perceiver token positional encoding uses img_size at construction; "
        f"crops at training/eval must match."
    )


# =============================================================================
# DATASETS
# =============================================================================

train_ds = XView2BaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
    oversample_building_damage=(not args.no_oversample),
)
val_ds = XView2BaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.crop_size, augment=False,
    oversample_building_damage=False,
)
test_ds = XView2BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size, augment=False,
    oversample_building_damage=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  xView2 Perceiver-IO Baseline")
print(f"  Channels:     {NUM_CHANNELS} (RGB, BGR-ordered)")
print(f"  Frames:       T={NUM_FRAMES} (pre + post)")
print(f"  Patch size:   {args.img_size}x{args.img_size}")
print(f"  Tokens:       {NUM_FRAMES * args.img_size ** 2:,} per sample "
      f"(2 frames x {args.img_size}^2)")
print(f"  Queries:      {args.img_size ** 2:,} per sample (time-agnostic)")
print(f"  Synthetic DOY: pre={PRE_DOY}, post={POST_DOY} "
      f"(half-year separation)")
print(f"  Latents:      {args.num_latents} x {args.latent_dim}")
print(f"  Depth:        {args.depth}")
print(f"  Classes:      {NUM_CLASSES}")
print(f"  Ignore index: {IGNORE_INDEX}")
print(f"  Epochs:       {args.epochs}")
print(f"  Batch size:   {args.batch_size}")
print(f"  Grad accum:   {args.grad_accum}")
print(f"  LR:           {args.lr}")
print(f"  GPUs:         {torch.cuda.device_count()}")
print(f"{'='*60}\n")

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=xview_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=4 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                          shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL
# =============================================================================

model = PerceiverSeg(
    in_channels=NUM_CHANNELS,
    num_classes=NUM_CLASSES,
    img_size=args.img_size,
    num_latents=args.num_latents,
    latent_dim=args.latent_dim,
    depth=args.depth,
    cross_heads=args.cross_heads,
    latent_heads=args.latent_heads,
    cross_dim_head=args.cross_dim_head,
    latent_dim_head=args.latent_dim_head,
    self_per_cross_attn=args.self_per_cross_attn,
    weight_tie_layers=(not args.no_weight_tie),
    num_freq_bands=args.num_freq_bands,
    max_freq=args.max_freq,
    attn_dropout=args.attn_dropout,
    ff_dropout=args.ff_dropout,
)


# =============================================================================
# TRAINER MODULE
# =============================================================================

# temporal=True so BaselineTrainer reads batch["dates"]["s2"] and passes
# it to model.forward(image, doy=...). PerceiverSeg.forward extracts the
# DOY into per-frame Fourier features; the synthesized [1, 183] gives
# the model a clean half-year separation between pre and post.
trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    temporal=True,
    task="xview2",
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        run_name = f"BL_{args.xp_name}_perceiver"
        wandb.init(
            name=run_name,
            project="Atomizer_xView2_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_xView2_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/xview_perceiver/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_perceiver-{{epoch:02d}}-{{val_mIoU:.4f}}",
            monitor="val_mIoU",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_perceiver-last",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_mIoU",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=-1,
        max_epochs=args.epochs,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
    )

    print(f"\n{'='*60}")
    print(f"  Starting: perceiver on xView2")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)

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
        raise FileNotFoundError(
            f"--test_only checkpoint not found: {args.test_only}"
        )
    best_ckpt = args.test_only
    print(f"\n[test-only mode] Skipping training, testing: {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(trainer_module, test_loader, ckpt_path=best_ckpt)

if wandb_logger:
    import wandb
    wandb.finish()