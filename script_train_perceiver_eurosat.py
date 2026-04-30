"""
EuroSAT Perceiver-IO Classification Script
=============================================

Trains the Perceiver-IO classification baseline on EuroSAT (geo-bench
m-eurosat) — 10-class S2 land-cover classification, 13 bands, 64x64.

Token layout (matches PerceiverCls.forward):
    Per-token feature = [reflectance(C=13), Fourier(x, y), no_time_vector]
    Input tokens     = [B, H*W, C + pos_dim + time_dim]    # H*W = 4096
    Output query     = [B, 1, query_dim]                    # single CLS token
    Output           = [B, num_classes]

Single-frame so no DOY is passed; the model's `no_time_vector` is used
for both tokens and the implicit query-time slot.

Same dataset protocol as the EuroSAT baseline:
    - Geo-bench default partition (train/valid/test, ~2000/1000/1000)
    - Per-band z-score normalization from band_stats.json
    - D4 augmentation in training

Memory note: 64x64 input means only 4096 tokens. Even with the parameter-
parity Perceiver-IO config (latent_dim=768, num_latents=512), this is by
far the cheapest task in the suite. Comfortable at large batch sizes.

Examples:
    python script_train_eurosat_perceiver.py --xp_name perceiver_eurosat \
        --batch_size 32 --lr 1e-4 --epochs 80

    # Test-only mode
    python script_train_eurosat_perceiver.py --xp_name perceiver_eurosat \
        --test_only ./checkpoints/eurosat_perceiver/bl_perceiver_eurosat-best.ckpt
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

from training.utils.datasets_baselines.utils_dataset_eurosat_baseline import (
    EuroSATBaselineDataset,
)
from training.perceiverIO.perceiver_cls import PerceiverCls
from training.trainer_baselines_classification import ClassificationBaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = EuroSATBaselineDataset.NUM_CLASSES        # 10
NUM_CHANNELS = EuroSATBaselineDataset.NUM_CHANNELS       # 13 S2 bands
PATCH_SIZE   = EuroSATBaselineDataset.PATCH_SIZE         # 64
MODALITY_KEY = "s2"


# =============================================================================
# COLLATE
# =============================================================================

def eurosat_collate(batch):
    """Stack per-modality images, stack scalar targets, keep metadata as list."""
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets  = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image":    images,
        "target":   targets,
        "metadata": metadata,
    }


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="EuroSAT Perceiver-IO Classification")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--data_dir",  type=str,
                    default="./data/geo-bench-1.0/classification_v1.0/m-eurosat")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Training
parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Spatial — EuroSAT native is 64x64
parser.add_argument("--img_size", type=int, default=PATCH_SIZE,
                    help=f"Spatial size. Default {PATCH_SIZE} (full EuroSAT patch).")

# Perceiver-IO config (matches the rest of the row for parameter parity)
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
# DATASETS
# =============================================================================

train_ds = EuroSATBaselineDataset(
    root_path=args.data_dir, mode="train", augment=True,
)
val_ds = EuroSATBaselineDataset(
    root_path=args.data_dir, mode="validation", augment=False,
)
test_ds = EuroSATBaselineDataset(
    root_path=args.data_dir, mode="test", augment=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  EuroSAT Perceiver-IO Classification")
print(f"  Channels:     {NUM_CHANNELS} (S2 — 13 bands)")
print(f"  Patch size:   {args.img_size}x{args.img_size}")
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
    collate_fn=eurosat_collate,
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
            project="Atomizer_EuroSAT_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_EuroSAT_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/eurosat_perceiver/"
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
    print(f"  Starting: perceiver on EuroSAT")
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