"""
HLS BurnScars Baseline Training Script
========================================

Train single-frame segmentation baselines on HLS BurnScars (binary burn scar).

Single-temporal — no LTAE needed. Supports:
  - unet   : classic UNet (PANGAEA-style topology)
  - vit    : ViT encoder + UPerNet decoder
  - resnet : ResNet encoder + UPerNet decoder (variant via --resnet_variant)

Same protocol as Sen1Floods11 baseline:
  - Same splits as PANGAEA (90/10 stratified train/val from training/, validation/ for test)
  - Same normalization (loaded from normalization_stats.pt or computed on train)
  - Same D4 augmentation
  - Train: 256×256 random crops; Eval: full 512×512 (UNet/ResNet)
  - ViT: train+eval at the same fixed size (--crop_size == --img_size)

Examples:
    # ResNet50 + UPerNet (matches Sen1Floods11 setup)
    python script_train_burnscars_baselines.py --xp_name resnet50 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 8 --lr 1e-4 --epochs 80

    # UNet baseline
    python script_train_burnscars_baselines.py --xp_name unet \
        --model unet \
        --batch_size 8 --lr 1e-3 --epochs 80

    # ViT-S at 512 (Sen1Floods11-native size)
    python script_train_burnscars_baselines.py --xp_name vit_512 \
        --model vit \
        --crop_size 512 --img_size 512 \
        --batch_size 2 --grad_accum 4 --lr 1e-4 --epochs 80
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

from training.utils.datasets_baselines.utils_dataset_burnscars_baselines import (
    BurnScarsBaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES = BurnScarsBaselineDataset.NUM_CLASSES        # 2
IGNORE_INDEX = BurnScarsBaselineDataset.IGNORE_INDEX      # 255
NUM_CHANNELS = BurnScarsBaselineDataset.NUM_CHANNELS      # 6
MODALITY_KEY = "hls"


# =============================================================================
# COLLATE
# =============================================================================

def burnscars_collate(batch):
    """Stack per-modality images, stack targets, keep metadata as list."""
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

parser = argparse.ArgumentParser(description="HLS BurnScars Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet"])
parser.add_argument("--data_dir",  type=str, default="./data/hls_burn_scars")

# Training
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Crop / image size
parser.add_argument("--crop_size", type=int, default=256,
                    help="Random crop for train. UNet/ResNet eval at full image; "
                         "ViT requires --crop_size == --img_size == --eval_size.")
parser.add_argument("--img_size",  type=int, default=256,
                    help="ViT positional embedding size (ignored by UNet/ResNet).")
parser.add_argument("--eval_size", type=int, default=None,
                    help="Eval crop size. None = full image (UNet/ResNet). "
                         "ViT auto-forces this to --img_size.")

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
parser.add_argument("--vit_decoder_channels", type=int, default=256,
                    help="UPerNet decoder channels (also used by ResNet)")

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

args = parser.parse_args()


# =============================================================================
# SANITY CHECKS & SIZE RESOLUTION
# =============================================================================

if args.model == "vit":
    if args.crop_size != args.img_size:
        raise ValueError(
            f"For ViT: --crop_size ({args.crop_size}) must equal "
            f"--img_size ({args.img_size}). ViT positional embedding is "
            f"baked at construction; train and eval must use the same size."
        )
    if args.eval_size is None:
        args.eval_size = args.img_size
    elif args.eval_size != args.img_size:
        raise ValueError(
            f"For ViT: --eval_size ({args.eval_size}) must equal "
            f"--img_size ({args.img_size})."
        )


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  HLS BurnScars Baseline Training")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
print(f"  Channels:    {NUM_CHANNELS} (HLS optical)")
print(f"  Train crop:  {args.crop_size}×{args.crop_size}")
eval_str = f"{args.eval_size}×{args.eval_size} (center crop)" if args.eval_size else "full image"
print(f"  Eval size:   {eval_str}")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  Grad acc:    {args.grad_accum}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = BurnScarsBaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
)
val_ds = BurnScarsBaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.eval_size, augment=False,
)
test_ds = BurnScarsBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.eval_size, augment=False,
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs_train = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=burnscars_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

loader_kwargs_eval = dict(
    batch_size=1,                       # full image — memory-conservative
    num_workers=args.num_workers,
    collate_fn=burnscars_collate,
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
    task="burnscars",
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
            project="Atomizer_BurnScars_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_BurnScars_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/burnscars_baselines/"
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
print(f"  Starting: {args.model} on HLS BurnScars")
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