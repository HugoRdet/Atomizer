"""
Sen1Floods11 Baseline Training Script
======================================

Train single-frame segmentation baselines on Sen1Floods11 (binary flood / no-flood).

Sen1Floods11 is single-temporal — no LTAE needed. Supports:
  - unet   : classic UNet (PANGAEA-style topology [64, 128, 256, 512, 1024])
  - vit    : ViT encoder + UPerNet decoder
  - resnet : ResNet encoder + UPerNet decoder (variant via --resnet_variant)

Same conditions as Atomiser:
  - Same train/val/test splits
  - Same normalization (per-band z-score, normalization_stats.pt)
  - Same NaN cleanup, ignore_index=255
  - Same D4 augmentation
  - 15 input channels (13 S2 + 2 S1, merged)

Examples:
    # ResNet50 + UPerNet on S2+S1
    python script_train_senflood_baseline.py --xp_name resnet50_s2s1 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 8 --lr 1e-4 --epochs 80

    # UNet baseline (matches PANGAEA's UNet setup)
    python script_train_senflood_baseline.py --xp_name unet_s2s1 \
        --model unet \
        --batch_size 8 --lr 1e-3 --epochs 80

    # ViT-S baseline
    python script_train_senflood_baseline.py --xp_name vit_s2s1 \
        --model vit \
        --batch_size 8 --lr 1e-4 --epochs 80
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

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS (from the dataset)
# =============================================================================

NUM_CLASSES = Sen1Floods11BaselineDataset.NUM_CLASSES        # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX      # 255
NUM_CHANNELS = Sen1Floods11BaselineDataset.NUM_CHANNELS      # 15
MODALITY_KEY = "s2s1"  # dataset returns image[{MODALITY_KEY}]


# =============================================================================
# COLLATE — stacks per-modality images, stacks targets, keeps metadata as list
# =============================================================================

def senflood_collate(batch):
    """
    Collate for Sen1Floods11BaselineDataset.

    Each sample is a dict with image[modality_key]: [C, H, W], target: [H, W].
    Stack into batch tensors.
    """
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "target": targets,
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

parser = argparse.ArgumentParser(description="Sen1Floods11 Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet"])
parser.add_argument("--data_dir",  type=str, default="./data/SENFLOOD")

# Training
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Crop / image size
parser.add_argument("--crop_size", type=int, default=512,
                    help="Random crop size for training. Default 512 = no crop "
                         "(use full image). For ViT, must match --img_size.")
parser.add_argument("--img_size",  type=int, default=512,
                    help="Spatial size baked into ViT positional embeddings. "
                         "MUST equal --crop_size for ViT (and equal eval size). "
                         "UNet/ResNet ignore this.")

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
# SANITY CHECKS
# =============================================================================

# ViT bakes spatial size into pos_embed → must match the actual eval/train size.
# Sen1Floods11 native size is 512. Train and eval must use the same size for ViT.
if args.model == "vit":
    if args.crop_size != args.img_size:
        raise ValueError(
            f"For ViT: --crop_size ({args.crop_size}) must equal "
            f"--img_size ({args.img_size}). ViT's positional embedding is "
            f"baked at construction; train and eval must use the same spatial "
            f"size. Sen1Floods11 native size is 512; recommended: both 512."
        )
    if args.img_size != 512:
        print(f"[WARNING] ViT trained at {args.img_size}×{args.img_size}, but "
              f"Sen1Floods11 native size is 512×512. Eval at 512 will fail; "
              f"the dataset would need explicit cropping/resizing.")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  Sen1Floods11 Baseline Training")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
print(f"  Channels:    {NUM_CHANNELS} (13 S2 + 2 S1)")
print(f"  Crop (train):{args.crop_size}×{args.crop_size}")
print(f"  Eval size:   512×512 (full)")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  Grad acc:    {args.grad_accum}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
)
val_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=None, augment=False,
)
test_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

# Val/test use full 512×512 images → bigger memory footprint per sample.
# Use batch_size=1 for eval to be safe; train uses cropped 256×256 at full BS.
loader_kwargs_train = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=senflood_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

loader_kwargs_eval = dict(
    batch_size=1,                       # full 512 — memory-conservative
    num_workers=args.num_workers,
    collate_fn=senflood_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs_train)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs_eval)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs_eval)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    temporal=False,                   # single-frame
    task="senflood",                  # registered in TASK_CLASS_NAMES (or falls back)
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
            project="Atomizer_SenFlood_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_SenFlood_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/senflood_baselines/"
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
print(f"  Starting: {args.model} on Sen1Floods11")
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