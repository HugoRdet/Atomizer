"""
PV4GER Baseline Training Script
================================

Train baselines on geo-bench m-pv4ger-seg (binary PV panel segmentation,
RGB aerial 320×320).

Models:
  - unet   : classic UNet (PANGAEA-style)
  - vit    : ViT + UPerNet
  - resnet : ResNet + UPerNet (variant via --resnet_variant)

All single-frame, no LTAE.

Examples:
    # ResNet50
    python script_train_pv4ger_baselines.py --xp_name resnet50 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 16 --lr 1e-4 --epochs 80

    # UNet
    python script_train_pv4ger_baselines.py --xp_name unet \
        --model unet --batch_size 16 --lr 1e-3 --epochs 80
"""

import os
import argparse

import torch
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

from training.utils.datasets_baselines.utils_dataset_pv4ger_baselines import (
    PV4GERBaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.trainer_baselines import BaselineTrainer


NUM_CLASSES  = PV4GERBaselineDataset.NUM_CLASSES
IGNORE_INDEX = PV4GERBaselineDataset.IGNORE_INDEX
NUM_CHANNELS = PV4GERBaselineDataset.NUM_CHANNELS
MODALITY_KEY = "rgb"


# =============================================================================
# COLLATE
# =============================================================================

def pv4ger_collate(batch):
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name: str, in_channels: int, num_classes: int, args):
    if model_name == "unet":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            topology=tuple(args.unet_topology),
        )
    elif model_name == "vit":
        return ViTUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )
    elif model_name == "resnet":
        return build_resnet_upernet(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=args.vit_decoder_channels,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="PV4GER Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet"])
parser.add_argument("--data_dir",  type=str,
                    default="./data/geo-bench-1.0/segmentation_v1.0/m-pv4ger-seg")

# Test-only mode: skip training, run test on a provided checkpoint
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. When set, skips training "
                         "and runs test directly on this checkpoint. "
                         "Uses a single GPU (no DDP).")

# Training
parser.add_argument("--batch_size",   type=int, default=16)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Crop / image size
parser.add_argument("--crop_size", type=int, default=None,
                    help="Crop size (None = full 320×320 native). "
                         "ViT requires --crop_size == --img_size if set.")
parser.add_argument("--img_size",  type=int, default=320,
                    help="ViT positional embedding size (must match input).")

# UNet
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])

# ViT
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

args = parser.parse_args()


# =============================================================================
# SANITY CHECKS
# =============================================================================

if args.model == "vit":
    eff_size = args.crop_size if args.crop_size is not None else 320
    if eff_size != args.img_size:
        raise ValueError(
            f"For ViT: input size ({eff_size}) must equal --img_size "
            f"({args.img_size}). Default: both 320."
        )


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  PV4GER Baseline Training")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
print(f"  Channels:    {NUM_CHANNELS} (RGB aerial)")
crop_str = f"{args.crop_size}×{args.crop_size}" if args.crop_size else "320×320 (full)"
print(f"  Input size:  {crop_str}")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = PV4GERBaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
)
val_ds = PV4GERBaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.crop_size, augment=False,
)
test_ds = PV4GERBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size, augment=False,
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=pv4ger_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                           shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                           shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    temporal=False,
    task="pv4ger",
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
        run_name = f"BL_{args.xp_name}_{args.model}"
        if args.model == "resnet":
            run_name += f"_{args.resnet_variant}"
        wandb.init(
            name=run_name,
            project="Atomizer_PV4GER_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_PV4GER_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/pv4ger_baselines/"
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

# =============================================================================
# TRAINER (DDP for fit) — skipped entirely in --test_only mode
# =============================================================================

if args.test_only is None:
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

    # =========================================================================
    # TRAIN + TEST (test on rank 0 only)
    # =========================================================================

    trainer.fit(trainer_module, train_loader, val_loader)

    # Capture best checkpoint path BEFORE destroying the process group.
    best_ckpt = trainer.checkpoint_callback.best_model_path

    # ─────────────────────────────────────────────────────────────────────
    # Destroy DDP process group BEFORE the test trainer is built.
    # Reasons:
    #   1. test_trainer = Trainer(devices=1) on rank 0 should start with no
    #      leftover NCCL state.
    #   2. Rank 1 doesn't need to do anything else — it can exit cleanly now.
    #   3. Avoids deadlocks where rank 1 waits at a barrier while rank 0
    #      runs its solo test.
    # ─────────────────────────────────────────────────────────────────────
    import torch.distributed as dist
    is_rank_zero = trainer.is_global_zero  # capture before teardown

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if not is_rank_zero:
        if wandb_logger:
            import wandb
            wandb.finish()
        raise SystemExit(0)

else:
    # Test-only mode — skip DDP setup entirely.
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(
            f"--test_only checkpoint not found: {args.test_only}"
        )
    best_ckpt = args.test_only
    print(f"\n[test-only mode] Skipping training, testing checkpoint:")
    print(f"  {best_ckpt}\n")


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