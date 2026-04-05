"""
C2Seg Baseline Training Script
================================

Train standard segmentation models (UNet, ViT) on C2Seg using
the existing BaselineTrainer (PyTorch Lightning).

One model per sensor — no cross-sensor capability.

Same training conditions as Atomizer-IO:
  - Same spatial split (train/val/test crops)
  - Same augmentations (D4 + spectral averaging in collate)
  - No pretraining (from scratch)
  - Same optimizer schedule

Examples:
    # UNet on HSI (EnMAP, 242 bands)
    python train_c2seg_baseline.py --xp_name unet_hsi \
        --model unet --sensor hsi --spectral_aug \
        --batch_size 8 --lr 1e-3 --epochs 300

    # UNet on MSI (Sentinel-2, 4 bands)
    python train_c2seg_baseline.py --xp_name unet_msi \
        --model unet --sensor msi \
        --batch_size 8 --lr 1e-3 --epochs 300

    # UNet on SAR (2 bands)
    python train_c2seg_baseline.py --xp_name unet_sar \
        --model unet --sensor sar \
        --batch_size 8 --lr 1e-3 --epochs 300
"""

import os
import argparse
import json

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

from training.utils.datasets_baselines.utils_dataset_Cseg import (
    C2SegBaselineDataset, NUM_CLASSES, IGNORE_INDEX, SENSOR_META_KEY, SENSOR_GSD,
)
from training.utils.datasets_baselines.collate import (
    get_collate_fn, get_augmented_collate_fn,
)
from training.utils.datasets.utils_dataset_C2SEG import build_spectral_aug_pool
from training.unet.model_unet import UNet
from training.threeDunet.unet3d import UNet3D
from training.VIT.vit_upernet import ViTUPerNet
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.senpa_seg.senpa import SenPaSeg
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, img_size=128,
                wavelengths=None, bandwidths=None, gsd=10.0):
    if model_name == "unet":
        return UNet(in_channels, num_classes)
    elif model_name == "unet3d":
        return UNet3D(in_channels, num_classes, base_features=32)
    elif model_name == "vit":
        return ViTUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=4,
        )
    elif model_name == "perceiver":
        return PerceiverSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            num_latents=256,
            latent_dim=256,
            depth=6,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            self_per_cross_attn=1,
            weight_tie_layers=True,
            num_freq_bands=6,
            max_freq=10.0,
            attn_dropout=0.0,
            ff_dropout=0.0,
        )
    elif model_name == "senpa":
        return SenPaSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=16 if in_channels > 50 else 8,
            emb_dim=256,
            num_layers=4 if in_channels > 50 else 6,
            num_heads=8,
            wavelengths=wavelengths,
            bandwidths=bandwidths,
            gsd=gsd,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: 'unet', 'vit', 'perceiver', 'senpa'")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="C2Seg Baseline Training")
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--model", type=str, required=True,
                    choices=["unet", "unet3d", "vit", "perceiver", "senpa"])
parser.add_argument("--sensor", type=str, required=True,
                    choices=["hsi", "msi", "msi12", "sar", "enmap_30m", "hyspex", "hyspex_10m"])

# Data
parser.add_argument("--subset", type=str, default="germany",
                    choices=["germany", "china"])
parser.add_argument("--city", type=str, default="augsburg")
parser.add_argument("--mat_path", type=str, default=None)
parser.add_argument("--processed_dir", type=str,
                    default="./data/CrossCity/c2seg_processed")

# Training
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--crop_size", type=int, default=128)

# Augmentation
parser.add_argument("--spectral_aug", action="store_true",
                    help="Enable spectral band averaging in collate")
parser.add_argument("--spectral_groups", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128])
parser.add_argument("--spectral_aug_prob", type=float, default=0.5)

# Normalization
parser.add_argument("--norm_mode", type=str, default="band_minmax",
                    choices=["band_minmax", "zscore", "identity"],
                    help="band_minmax: per-band min-max (default). "
                         "zscore: per-band z-score. identity: raw values")

# Multi-GPU
parser.add_argument("--grad_accum", type=int, default=1)

args = parser.parse_args()

# =============================================================================
# RESOLVE PATHS
# =============================================================================

if args.mat_path is None:
    subset_dir = "Germany" if args.subset == "germany" else "China"
    city_mat = {
        "augsburg": "augsburg_multimodal.mat",
        "berlin": "berlin_multimodal.mat",
        "beijing": "beijing.mat",
        "wuhan": "wuhan.mat",
    }
    args.mat_path = f"./data/CrossCity/{subset_dir}/{city_mat[args.city]}"

crop_index_path = os.path.join(args.processed_dir, "c2seg_crop_index_split.csv")
stats_path = os.path.join(args.processed_dir, "c2seg_norm_stats.json")
spectral_meta_path = os.path.join(args.processed_dir, "c2seg_spectral_meta.json")

# Get sensor info
meta_key = SENSOR_META_KEY[(args.subset, args.sensor)]
with open(spectral_meta_path) as f:
    all_meta = json.load(f)
n_bands = all_meta[meta_key]["n_bands"]
sensor_wavelengths = all_meta[meta_key].get("wavelengths", None)
sensor_bandwidths = all_meta[meta_key].get("bandwidths", None)
sensor_gsd = SENSOR_GSD.get((args.subset, args.sensor), 10.0)

print(f"\n{'='*60}")
print(f"  C2Seg Baseline Training")
print(f"  Model:    {args.model}")
print(f"  Sensor:   {args.sensor} ({n_bands} bands)")
print(f"  City:     {args.city} ({args.subset})")
print(f"  Epochs:   {args.epochs}")
print(f"  BS:       {args.batch_size}")
print(f"  LR:       {args.lr}")
print(f"  Spec aug: {args.spectral_aug}")
print(f"  Norm:     {args.norm_mode}")
print(f"  Grad acc: {args.grad_accum}")
print(f"  GPUs:     {torch.cuda.device_count()}")
print(f"{'='*60}\n")

# =============================================================================
# DATASETS
# =============================================================================

common = dict(
    mat_path=args.mat_path,
    subset=args.subset,
    city=args.city,
    sensor=args.sensor,
    crop_index_path=crop_index_path,
    stats_path=stats_path,
    spectral_meta_path=spectral_meta_path,
    crop_size=args.crop_size,
    norm_mode=args.norm_mode,
)

train_ds = C2SegBaselineDataset(split="train", mode="train", augment=True, **common)
val_ds = C2SegBaselineDataset(split="val", mode="test", augment=False, **common)
test_ds = C2SegBaselineDataset(split="test", mode="test", augment=False, **common)

print(f"  Train: {len(train_ds)} crops")
print(f"  Val:   {len(val_ds)} crops")
print(f"  Test:  {len(test_ds)} crops")

# =============================================================================
# COLLATE
# =============================================================================

if args.spectral_aug:
    spectral_pool = build_spectral_aug_pool(n_random=500)
    print(f"[Baseline] Spectral aug pool: {len(spectral_pool)} configs")
    train_collate = get_augmented_collate_fn(
        modalities=[args.sensor],
        spectral_aug_prob=args.spectral_aug_prob,
        spectral_aug_pool=spectral_pool,
    )
else:
    train_collate = get_collate_fn([args.sensor])

eval_collate = get_collate_fn([args.sensor])

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, collate_fn=train_collate,
    pin_memory=True, drop_last=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, collate_fn=eval_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None)
test_loader = DataLoader(
    test_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, collate_fn=eval_collate,
    pin_memory=True)

# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(
    args.model, n_bands, NUM_CLASSES, img_size=args.crop_size,
    wavelengths=sensor_wavelengths, bandwidths=sensor_bandwidths,
    gsd=sensor_gsd,
)

trainer_module = BaselineTrainer(
    model=model,
    modality=args.sensor,
    task="c2seg",
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
        wandb.init(
            name=f"BL_{args.xp_name}_{args.model}_{args.sensor}",
            project="Atomizer_C2Seg_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_C2Seg_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")

# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = f"./checkpoints/c2seg_baselines/{args.subset}/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}_{args.sensor}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}_{args.sensor}-last",
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

# =============================================================================
# TRAIN
# =============================================================================

print(f"\n{'='*60}")
print(f"  Starting: {args.model} on {args.sensor} ({args.city})")
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