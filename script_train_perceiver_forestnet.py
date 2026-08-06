"""
ForestNet Perceiver-IO Classification Script
================================================

Trains the Perceiver-IO classification baseline on geo-bench m-forestnet —
12-class Landsat-8 deforestation classification, 6 bands, 332x332 native
(center-cropped to 320x320 to be a multiple of 16).

Token layout (matches PerceiverCls.forward):
    Per-token feature = [reflectance(C=6), Fourier(x, y), no_time_vector]
    Input tokens     = [B, H*W, C + pos_dim + time_dim]    # H*W = 102,400
    Output query     = [B, 1, query_dim]                    # single CLS token
    Output           = [B, num_classes]

Single-frame so no DOY is passed.

Same dataset protocol as the ForestNet baseline:
    - Geo-bench default partition (train/valid/test)
    - Per-band z-score normalization from band_stats.json
    - Center crop 332 -> 320
    - D4 augmentation in training

Memory note: 320x320 produces ~102k tokens. With latent_dim=768,
num_latents=512, BS=8 fits comfortably; bump up if you have room.

Examples:
    python script_train_forestnet_perceiver.py --xp_name perceiver_forestnet \
        --batch_size 8 --lr 1e-4 --epochs 80

    # Test-only mode
    python script_train_forestnet_perceiver.py --xp_name perceiver_forestnet \
        --test_only ./checkpoints/forestnet_perceiver/bl_perceiver_forestnet-best.ckpt
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

from training.utils.datasets_baselines.utils_dataset_forestnet_baselines import (
    ForestNetBaselineDataset,
)
from training.perceiverIO.perceiver_cls import PerceiverCls
from training.trainer_baselines_classification import ClassificationBaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = ForestNetBaselineDataset.NUM_CLASSES        # 12
NUM_CHANNELS = ForestNetBaselineDataset.NUM_CHANNELS       # 6 (Landsat-8)
DEFAULT_CROP = 320                                         # divisible by 16
MODALITY_KEY = "landsat"


# =============================================================================
# COLLATE
# =============================================================================

def forestnet_collate(batch):
    """
    Stack per-modality images, stack int targets into a long tensor,
    keep metadata as a list.

    Differs from eurosat_collate: ForestNet returns target as a Python int
    (not torch.tensor), so torch.stack would fail.
    """
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets  = torch.tensor([s["target"] for s in batch], dtype=torch.long)
    metadata = [s["metadata"] for s in batch]

    return {
        "image":    images,
        "target":   targets,
        "metadata": metadata,
    }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="ForestNet Perceiver-IO Classification")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--data_dir",  type=str,
                    default="./data/geo-bench-1.0/classification_v1.0/m-forestnet")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Training
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Spatial — ForestNet native is 332, cropped to 320 by default
parser.add_argument("--crop_size", type=int, default=DEFAULT_CROP,
                    help=f"Center crop size from 332 native. Default {DEFAULT_CROP}.")
parser.add_argument("--img_size", type=int, default=DEFAULT_CROP,
                    help="Spatial size for token construction. Must match crop_size.")

# Perceiver-IO config (matches the rest of the row for parameter parity)
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=389)
parser.add_argument("--depth",              type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=1)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=6)
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
        f"the cropped patch size must match."
    )


# =============================================================================
# DATASETS
# =============================================================================

train_ds = ForestNetBaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
)
val_ds = ForestNetBaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.crop_size, augment=False,
)
test_ds = ForestNetBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size, augment=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  ForestNet Perceiver-IO Classification")
print(f"  Channels:     {NUM_CHANNELS} (Landsat-8: B,G,R,NIR,SWIR1,SWIR2)")
print(f"  Patch size:   {args.img_size}x{args.img_size} (center-cropped from 332)")
print(f"  Tokens:       {args.img_size ** 2:,} per sample")
print(f"  Query:        single learned CLS token (attention pool)")
print(f"  Latents:      {args.num_latents} x {args.latent_dim}")
print(f"  Depth:        {args.depth}")
print(f"  Classes:      {NUM_CLASSES}")
print(f"  Epochs:       {args.epochs}")
print(f"  Batch size:   {args.batch_size}")
print(f"  LR:           {args.lr}")
print(f"  Grad accum:   {args.grad_accum}")
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
    collate_fn=forestnet_collate,
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

model = PerceiverCls(
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

trainer_module = ClassificationBaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    num_classes=NUM_CLASSES,
    lr=args.lr,
    weight_decay=args.weight_decay,
    label_smoothing=args.label_smoothing,
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
            project="Atomizer_ForestNet_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_ForestNet_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/forestnet_perceiver/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_perceiver-{{epoch:02d}}-{{val_top1:.4f}}",
            monitor="val_top1",
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
            monitor="val_top1",
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
        accumulate_grad_batches=args.grad_accum,
    )

    print(f"\n{'='*60}")
    print(f"  Starting: perceiver on ForestNet")
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
