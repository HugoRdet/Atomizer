"""
PASTIS-HD Baseline Training Script
=====================================

Train temporal segmentation models (U-TAE, UNet, ViT) on PASTIS-HD.

Uses TemporalUNet (U-TAE) by default: UNet encoder applied per-frame,
LTAE temporal aggregation at bottleneck, UNet decoder.

Same training conditions as Atomizer-IO:
  - Same fold splits (train: 1,2,3 / val: 4 / test: 5)
  - Same normalization (per-band z-score from normalization_stats.pt)
  - Same temporal sampling (uniform or last-N)
  - No pretraining (from scratch)

Examples:
    # U-TAE, S2-only, 10 temporal frames
    python train_pastis_baseline.py --xp_name utae_s2 \
        --model utae --multi_temporal 10 \
        --batch_size 4 --lr 1e-3 --epochs 100

    # U-TAE, S2-only, last 5 frames
    python train_pastis_baseline.py --xp_name utae_s2_last5 \
        --model utae --multi_temporal 5 --temporal_last \
        --batch_size 4 --lr 1e-3 --epochs 100

    # U-TAE with S1
    python train_pastis_baseline.py --xp_name utae_s2s1 \
        --model utae --use_s1 --multi_temporal 10 \
        --batch_size 4 --lr 1e-3 --epochs 100
"""

import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_PASTIS import (
    PastisBaselineDataset, NUM_CLASSES, IGNORE_INDEX, NUM_S2_BANDS, NUM_S1_BANDS,
)
from training.ltae.ltae import TemporalUNet
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# COLLATE — handles nested dicts (image, dates, target, metadata)
# =============================================================================

def pastis_collate(batch):
    """
    Collate for PastisBaselineDataset.

    Stacks image tensors per sensor, dates per sensor, and targets.
    Metadata is collected as a list of dicts (not stacked).
    """
    images = {}
    dates = {}
    targets = []
    metadata = []

    # Collect sensor keys from first sample
    sensor_keys = list(batch[0]["image"].keys())

    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
        dates[key] = torch.stack([s["dates"][key] for s in batch])

    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "dates": dates,
        "target": targets,
        "metadata": metadata,
    }


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, n_heads=16, d_k=4,
                d_model=256, base_channels=64):
    if model_name == "utae":
        return TemporalUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            n_heads=n_heads,
            d_k=d_k,
            d_model=d_model,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: 'utae'")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="PASTIS-HD Baseline Training")
parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--model",      type=str, default="utae", choices=["utae"])
parser.add_argument("--data_dir",   type=str, default="./data/PASTIS-HD")

# Modality
parser.add_argument("--use_s1",     action="store_true",
                    help="Include S1A SAR data (default: S2-only)")

# Temporal
parser.add_argument("--multi_temporal", type=int, default=10,
                    help="Number of temporal frames to use")
parser.add_argument("--temporal_last",  action="store_true",
                    help="Take last N timesteps instead of uniform sampling")

# Training
parser.add_argument("--batch_size",  type=int, default=4)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",     type=int, default=100)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience",   type=int, default=30)
parser.add_argument("--grad_accum", type=int, default=1)

# U-TAE architecture
parser.add_argument("--base_channels", type=int, default=64)
parser.add_argument("--n_heads",     type=int, default=16)
parser.add_argument("--d_k",        type=int, default=4)
parser.add_argument("--d_model",    type=int, default=256)

args = parser.parse_args()

# =============================================================================
# CONFIG
# =============================================================================

modalities = ["S2"]
in_channels = NUM_S2_BANDS  # 10

if args.use_s1:
    modalities.append("S1")
    # S1 is a separate sensor key — we'll train on S2 only for the model
    # but pass S1 alongside. For simplicity, the baseline model uses S2.
    # To fuse S1+S2, concatenate bands: in_channels = 10 + 2 = 12
    in_channels = NUM_S2_BANDS + NUM_S1_BANDS

modality_str = "+".join(modalities)
temporal_str = f"{args.multi_temporal} frames ({'last' if args.temporal_last else 'uniform'})"

print(f"\n{'='*60}")
print(f"  PASTIS-HD Baseline Training")
print(f"  Model:      {args.model}")
print(f"  Modalities: {modality_str} ({in_channels} bands)")
print(f"  Temporal:   {temporal_str}")
print(f"  Epochs:     {args.epochs}")
print(f"  BS:         {args.batch_size}")
print(f"  LR:         {args.lr}")
print(f"  Grad acc:   {args.grad_accum}")
print(f"  GPUs:       {torch.cuda.device_count()}")
print(f"{'='*60}\n")

# =============================================================================
# DATASETS
# =============================================================================

common = dict(
    root_path=args.data_dir,
    use_s1=args.use_s1,
    multi_temporal=args.multi_temporal,
    temporal_last=args.temporal_last,
    temporal_mode="sequence",  # Always sequence for LTAE
)

train_ds = PastisBaselineDataset(mode="train", augment=True, **common)
val_ds = PastisBaselineDataset(mode="validation", augment=False, **common)
test_ds = PastisBaselineDataset(mode="test", augment=False, **common)

print(f"  Train: {len(train_ds)} patches")
print(f"  Val:   {len(val_ds)} patches")
print(f"  Test:  {len(test_ds)} patches")

# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=pastis_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

# Determine input modality key for the trainer
# If using S1, we'll need a custom collate that concatenates S2+S1
# For now: single modality key
if args.use_s1:
    # We need to fuse S2 and S1 before feeding to the model
    # Override collate to concatenate along channel dim
    print("[PASTIS-BL] S2+S1 fusion: concatenating bands in collate")

    _base_collate = pastis_collate

    def fused_collate(batch):
        out = _base_collate(batch)
        s2 = out["image"]["s2"]  # [B, T, 10, H, W]
        s1 = out["image"]["s1"]  # [B, T, 2, H, W]

        # Match temporal length (take min if different)
        T = min(s2.shape[1], s1.shape[1])
        fused = torch.cat([s2[:, :T], s1[:, :T]], dim=2)  # [B, T, 12, H, W]

        out["image"] = {"s2": fused}
        # Use S2 dates (primary sensor)
        out["dates"] = {"s2": out["dates"]["s2"][:, :T]}
        return out

    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=fused_collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=fused_collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=fused_collate, pin_memory=True,
    )

model = build_model(
    args.model,
    in_channels=in_channels,
    num_classes=NUM_CLASSES,
    n_heads=args.n_heads,
    d_k=args.d_k,
    d_model=args.d_model,
    base_channels=args.base_channels,
)

trainer_module = BaselineTrainer(
    model=model,
    modality="s2",
    temporal=True,
    task="pastis",
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
        run_name = f"BL_{args.xp_name}_{args.model}_{modality_str}"
        wandb.init(
            name=run_name,
            project="Atomizer_PASTIS_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_PASTIS_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")

# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/pastis_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}-last",
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

# =============================================================================
# TRAINER
# =============================================================================

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
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

# =============================================================================
# TRAIN
# =============================================================================

print(f"\n{'='*60}")
print(f"  Starting: {args.model} — {modality_str}")
print(f"  Temporal: {temporal_str}")
print(f"  Train: folds 1,2,3 → Val: fold 4 → Test: fold 5")
print(f"{'='*60}\n")

trainer.fit(trainer_module, train_loader, val_loader)

# =============================================================================
# TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing best checkpoint")
print(f"{'='*60}\n")

trainer.test(trainer_module, test_loader, ckpt_path="best")

if wandb_logger:
    import wandb
    wandb.finish()