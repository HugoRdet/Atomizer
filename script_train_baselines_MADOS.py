"""
MADOS Baseline Training Script
================================

Train single-frame segmentation baselines on MADOS — 15-class marine debris
segmentation on Sentinel-2.

MADOS is single-temporal — no LTAE needed. Supports:
  - unet   : classic UNet (PANGAEA-style topology)
  - vit    : ViT encoder + UPerNet decoder
  - resnet : ResNet encoder + UPerNet decoder (variant via --resnet_variant)

Same conditions as the Atomiser MADOS run:
  - Same train/val/test splits (./data/MADOS/splits/{train|val|test}_X.txt)
  - Same per-band per-resolution normalization (normalization_stats.pt cache)
  - Same bands order (from bands.yaml's bands_mados section)
  - All bands upscaled to 10m → [C, 240, 240]
  - 15 classes, IGNORE_INDEX=255

D4 augmentation is automatic for training.
240×240 is divisible by 16 → ViT patch_size=16 works cleanly.

Examples:
    # ResNet50 + UPerNet
    python script_train_mados_baselines.py --xp_name resnet50 \\
        --model resnet --resnet_variant resnet50 \\
        --batch_size 8 --lr 1e-4 --epochs 80

    # UNet baseline
    python script_train_mados_baselines.py --xp_name unet \\
        --model unet --batch_size 8 --lr 1e-3 --epochs 80

    # ViT-S baseline
    python script_train_mados_baselines.py --xp_name vit \\
        --model vit --batch_size 8 --lr 1e-4 --epochs 80
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

from training.utils import read_yaml
from training.utils.datasets_baselines.utils_dataset_mados_baseline import (
    MADOSBaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = MADOSBaselineDataset.NUM_CLASSES        # 15
IGNORE_INDEX = MADOSBaselineDataset.IGNORE_INDEX       # 255
MODALITY_KEY = "s2"

# Spatial size of MADOS patches at 10m resolution
NATIVE_H, NATIVE_W = MADOSBaselineDataset.FULL_SIZE_10M  # (240, 240)


# =============================================================================
# COLLATE
# =============================================================================

def mados_collate(batch):
    """Stack per-modality images, stack targets, keep metadata as list."""
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
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'unet', 'vit', 'resnet'"
        )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="MADOS Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet"])
parser.add_argument("--data_dir",  type=str, default="./data/MADOS")
parser.add_argument("--bands_yaml", type=str, default="./data/bands_info/bands.yaml",
                    help="YAML file containing the bands_mados section")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Training
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Image size — MADOS is 240×240. ViT patch_size=16 → 15×15 patches works cleanly.
parser.add_argument("--img_size", type=int, default=NATIVE_H,
                    help=f"Spatial size baked into ViT pos_embed. "
                         f"Default {NATIVE_H} (full MADOS patch).")

# UNet
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024],
                    help="UNet feature widths per level")

# ViT
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256,
                    help="UPerNet decoder channels (also used by ResNet)")

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

args = parser.parse_args()


# =============================================================================
# SANITY CHECK FOR VIT
# =============================================================================

if args.model == "vit":
    if args.img_size % args.vit_patch_size != 0:
        raise ValueError(
            f"For ViT: --img_size ({args.img_size}) must be divisible by "
            f"--vit_patch_size ({args.vit_patch_size}). "
            f"Default img_size={NATIVE_H} works with patch_size 16, 12, 8."
        )


# =============================================================================
# LOAD BANDS METADATA
# =============================================================================

bands_yaml = read_yaml(args.bands_yaml)
if "bands_mados" not in bands_yaml:
    raise KeyError(
        f"[MADOS-BL] bands_yaml ({args.bands_yaml}) must contain a "
        f"'bands_mados' section."
    )
bands_info = bands_yaml["bands_mados"]


# =============================================================================
# DATASETS
# =============================================================================

train_ds = MADOSBaselineDataset(
    root_path=args.data_dir, mode="train",
    bands_info=bands_info,
)
val_ds = MADOSBaselineDataset(
    root_path=args.data_dir, mode="validation",
    bands_info=bands_info,
)
test_ds = MADOSBaselineDataset(
    root_path=args.data_dir, mode="test",
    bands_info=bands_info,
)

NUM_CHANNELS = train_ds.num_channels


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  MADOS Baseline Training")
print(f"  Model:        {args.model}")
if args.model == "resnet":
    print(f"  Variant:      {args.resnet_variant}")
print(f"  Channels:     {NUM_CHANNELS} bands (all upscaled to 10m)")
print(f"  Patch size:   {NATIVE_H}×{NATIVE_W}")
print(f"  Classes:      {NUM_CLASSES}")
print(f"  Ignore index: {IGNORE_INDEX}")
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
    collate_fn=mados_collate,
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
    task="mados",                     # registered in TASK_CLASS_NAMES
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
            project="Atomizer_MADOS_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_MADOS_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/mados_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
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
    print(f"  Starting: {args.model} on MADOS")
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