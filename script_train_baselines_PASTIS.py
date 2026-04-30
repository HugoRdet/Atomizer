"""
PASTIS-HD Baseline Training Script
=====================================

Train temporal segmentation models on PASTIS-HD.

Supported models (via --model):
  - unet_ltae           : Classic UNet (per-frame, shared) + LTAE temporal
                          aggregation at full output resolution + 1×1 head
  - vit_ltae            : ViT per-frame + LTAE BETWEEN encoder and decoder
                          (one LTAE per FPN feature layer) + UPerNet
  - vit_upernet_ltae    : ViT per-frame + UPerNet (per-frame features)
                          + LTAE AFTER decoder at FPN resolution + 1×1 head
  - vit_upernet_mt      : ViT + UPerNet with channel-concat early fusion
                          (TimeMerge DoubleConv before encoder).
                          Mirror of resnet_upernet_mt for the ViT family —
                          PANGAEA-style early fusion.
  - vit                 : ViT encoder (channel-stacked frames, non-temporal) + UPerNet
  - prithvi             : Prithvi 3D ViT (3D tubelet conv) + UPerNet
  - resnet_upernet_mt   : ResNet (channel-concat early fusion via TimeMerge
                          DoubleConv) + UPerNet — PANGAEA-style.
                          Replaces the old resnet_upernet_ltae (late fusion)
                          for direct comparability with PANGAEA's reported
                          UNetMT/ViT numbers on PASTIS.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training and run
    test on a saved checkpoint (single GPU, no DDP).

Examples:
    # New ViT + channel-concat MT, S2-only, 6 frames (PANGAEA convention)
    python script_train_pastis_baseline.py --xp_name vit_mt_t6 \
        --model vit_upernet_mt --multi_temporal 6 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # ResNet + channel-concat MT, S2-only, 6 frames (PANGAEA convention)
    python script_train_pastis_baseline.py --xp_name resnet50_mt_s2 \
        --model resnet_upernet_mt --resnet_variant resnet50 \
        --multi_temporal 6 --batch_size 4 --lr 1e-4 --epochs 100

    # UNet + LTAE
    python script_train_pastis_baseline.py --xp_name unet_ltae_s2 \
        --model unet_ltae --multi_temporal 10 \
        --batch_size 4 --lr 1e-3 --epochs 100

    # ViT + LTAE
    python script_train_pastis_baseline.py --xp_name vit_ltae_s2 \
        --model vit_ltae --multi_temporal 10 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # ViT (non-temporal, channel-stacked), S2-only, 3 frames
    python script_train_pastis_baseline.py --xp_name vit_s2_t3 \
        --model vit --multi_temporal 3 \
        --batch_size 4 --lr 1e-4 --epochs 100

    # UNet+LTAE with S2+S1 fusion
    python script_train_pastis_baseline.py --xp_name unet_ltae_s2s1 \
        --model unet_ltae --use_s1 --multi_temporal 10 \
        --batch_size 4 --lr 1e-3 --epochs 100
"""

import os
import argparse

import torch
import torch.nn as nn
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
from training.ltae.ltae import UNetLTAE
from training.VIT.model_vit_upernet import (
    ViTUPerNet, ViTLTAEUPerNet, ViTUPerNetLTAE, build_vit_upernet_mt,
)
from training.ResNet.model_resnet_upernet import build_resnet_upernet_mt
from training.prithvi.prithvi import PrithviUPerNet
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# COLLATE — handles nested dicts (image, dates, target, metadata)
# =============================================================================

def pastis_collate(batch):
    images = {}
    dates = {}
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


def make_fused_collate(use_s1: bool):
    """Returns a collate that fuses S2 and S1 along the channel dim."""
    if not use_s1:
        return pastis_collate

    def fused_collate(batch):
        out = pastis_collate(batch)
        s2 = out["image"]["s2"]  # [B, T, 10, H, W]
        s1 = out["image"]["s1"]  # [B, T,  2, H, W]
        T = min(s2.shape[1], s1.shape[1])
        fused = torch.cat([s2[:, :T], s1[:, :T]], dim=2)  # [B, T, 12, H, W]
        out["image"] = {"s2": fused}
        out["dates"] = {"s2": out["dates"]["s2"][:, :T]}
        return out

    return fused_collate


def make_channel_stack_collate(base_collate):
    """[B, T, C, H, W] → [B, T*C, H, W] for non-temporal models."""
    def stacked_collate(batch):
        out = base_collate(batch)
        for key, img in out["image"].items():
            if img.dim() == 5:
                B, T, C, H, W = img.shape
                out["image"][key] = img.reshape(B, T * C, H, W)
        return out
    return stacked_collate


# =============================================================================
# PRITHVI ADAPTER
# =============================================================================

class PrithviAdapter(nn.Module):
    """Wraps PrithviUPerNet to accept [B, T, C, H, W] (Prithvi expects [B, C, T, H, W])."""
    def __init__(self, prithvi_model: nn.Module):
        super().__init__()
        self.prithvi = prithvi_model

    def forward(self, x: torch.Tensor, doy: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.prithvi(x, doy=doy)


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, args):
    """Dispatch to the requested model architecture."""
    if model_name == "unet_ltae":
        return UNetLTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            topology=tuple(args.unet_topology),
            n_heads=args.n_heads,
            d_k=args.d_k,
            d_model=args.d_model,
        )

    elif model_name == "vit_ltae":
        return ViTLTAEUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
            ltae_n_head=args.n_heads,
            ltae_d_k=args.d_k,
            ltae_d_model=args.d_model,
        )

    elif model_name == "vit_upernet_ltae":
        return ViTUPerNetLTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
            ltae_n_head=args.n_heads,
            ltae_d_k=args.d_k,
            ltae_d_model=args.d_model,
        )

    elif model_name == "vit_upernet_mt":
        # ViT + UPerNet with channel-concat early fusion (TimeMerge DoubleConv).
        # Mirror of resnet_upernet_mt for the ViT family.
        return build_vit_upernet_mt(
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=args.multi_temporal,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name in ("resnet_upernet_mt", "resnet_upernet_ltae"):
        # Channel-concat early fusion via TimeMerge DoubleConv (PANGAEA-style).
        # The legacy 'resnet_upernet_ltae' name is kept as an alias for
        # back-compat, but it now refers to early-fusion MT (LTAE variant
        # was removed when we switched to PANGAEA-style temporal handling).
        if model_name == "resnet_upernet_ltae":
            print("[WARN] 'resnet_upernet_ltae' is now an alias for "
                  "'resnet_upernet_mt' (channel-concat early fusion). "
                  "The LTAE late-fusion variant has been removed.")
        return build_resnet_upernet_mt(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            num_frames=args.multi_temporal,
            decoder_channels=args.vit_decoder_channels,
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

    elif model_name == "prithvi":
        prithvi = PrithviUPerNet(
            in_chans=in_channels,
            num_frames=args.multi_temporal,
            img_size=args.img_size,
            patch_size=args.vit_patch_size,
            tubelet_size=args.prithvi_tubelet_size,
            embed_dim=args.prithvi_embed_dim,
            depth=args.prithvi_depth,
            num_heads=args.prithvi_num_heads,
            num_classes=num_classes,
            decoder_channels=args.vit_decoder_channels,
            output_layers=tuple(args.vit_output_layers),
        )
        return PrithviAdapter(prithvi)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: 'unet_ltae', 'vit_ltae', 'vit_upernet_ltae', "
            f"'vit_upernet_mt', 'vit', 'prithvi', 'resnet_upernet_mt'"
        )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="PASTIS-HD Baseline Training")
parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--model",      type=str, default="unet_ltae",
                    choices=["unet_ltae", "vit_ltae", "vit_upernet_ltae",
                             "vit_upernet_mt",
                             "vit", "prithvi", "resnet_upernet_mt",
                             "resnet_upernet_ltae"])  # legacy alias kept
parser.add_argument("--data_dir",   type=str, default="./data/PASTIS-HD")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

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
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience",    type=int, default=30)
parser.add_argument("--grad_accum",  type=int, default=1)

# UNet+LTAE / LTAE shared params (still used by unet_ltae, vit_ltae, vit_upernet_ltae)
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])
parser.add_argument("--n_heads",    type=int, default=16)
parser.add_argument("--d_k",        type=int, default=4)
parser.add_argument("--d_model",    type=int, default=256)

# ViT-specific
parser.add_argument("--img_size",          type=int, default=128)
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+", default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)

# Prithvi-specific
parser.add_argument("--prithvi_embed_dim",     type=int, default=768)
parser.add_argument("--prithvi_depth",         type=int, default=12)
parser.add_argument("--prithvi_num_heads",     type=int, default=12)
parser.add_argument("--prithvi_tubelet_size",  type=int, default=1)

# ResNet-specific
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

args = parser.parse_args()


# =============================================================================
# CONFIG
# =============================================================================

modalities = ["S2"]
per_frame_channels = NUM_S2_BANDS  # 10

if args.use_s1:
    modalities.append("S1")
    per_frame_channels = NUM_S2_BANDS + NUM_S1_BANDS  # 12

modality_str = "+".join(modalities)
temporal_str = f"{args.multi_temporal} frames ({'last' if args.temporal_last else 'uniform'})"

# Models that accept 5D [B, T, C, H, W] input directly (with their own
# internal temporal handling — LTAE, 3D conv, or TimeMerge DoubleConv).
is_temporal_model = args.model in (
    "unet_ltae", "vit_ltae", "vit_upernet_ltae", "vit_upernet_mt",
    "prithvi", "resnet_upernet_mt", "resnet_upernet_ltae",
)
if is_temporal_model:
    model_in_channels = per_frame_channels         # model sees [B, T, C, H, W]
else:
    model_in_channels = per_frame_channels * args.multi_temporal  # [B, T*C, H, W]

# Print summary
if args.test_only:
    print(f"\n[Train] Test-only mode: {args.test_only}\n")

print(f"\n{'='*60}")
print(f"  PASTIS-HD Baseline Training")
print(f"  Model:      {args.model} ({'temporal' if is_temporal_model else 'non-temporal'})")
print(f"  Modalities: {modality_str} ({per_frame_channels} bands/frame)")
print(f"  Temporal:   {temporal_str}")
if not is_temporal_model:
    print(f"  In channels (stacked): {model_in_channels}")
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
    temporal_mode="sequence",
)

train_ds = PastisBaselineDataset(mode="train",      augment=True,  **common)
val_ds   = PastisBaselineDataset(mode="validation", augment=False, **common)
test_ds  = PastisBaselineDataset(mode="test",       augment=False, **common)

print(f"  Train: {len(train_ds)} patches")
print(f"  Val:   {len(val_ds)} patches")
print(f"  Test:  {len(test_ds)} patches")


# =============================================================================
# COLLATE SELECTION
# =============================================================================

base_collate = make_fused_collate(args.use_s1)
if args.use_s1:
    print("[PASTIS-BL] S2+S1 fusion: concatenating bands in collate")

if not is_temporal_model:
    print("[PASTIS-BL] Non-temporal model: stacking T frames into channels")
    collate_fn = make_channel_stack_collate(base_collate)
else:
    collate_fn = base_collate


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(
    args.model,
    in_channels=model_in_channels,
    num_classes=NUM_CLASSES,
    args=args,
)

trainer_module = BaselineTrainer(
    model=model,
    modality="s2",
    temporal=is_temporal_model,
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
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/pastis_baselines/"
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

    print(f"\n{'='*60}")
    print(f"  Starting: {args.model} — {modality_str}")
    print(f"  Temporal: {temporal_str}")
    print(f"  Train: folds 1,2,3 → Val: fold 4 → Test: fold 5")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path

    # ─────────────────────────────────────────────────────────────────────
    # Destroy DDP process group BEFORE the test trainer is built.
    # Rank 1 exits cleanly here; only rank 0 proceeds to test.
    # ─────────────────────────────────────────────────────────────────────
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