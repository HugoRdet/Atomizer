"""
PASTIS-HD Perceiver-IO Single-Task Training Script
=====================================================

Trains the multi-temporal-aware Perceiver-IO baseline on PASTIS-HD.

Token layout (matches PerceiverSeg.forward):
    Per-token feature = [reflectance(C), Fourier(x, y), Fourier(DOY)]
    Input tokens    = [B, T*H*W, C + pos_dim + time_dim]
    Output queries  = [B, H*W, query_dim]    # time-agnostic

Differences from the multi-model script_train_pastis_baseline.py:
    - Single model path: Perceiver-IO only (no model dispatch).
    - The dataset is used in `temporal_mode="sequence"` so the model
      receives [B, T, C, H, W] directly. PerceiverSeg.forward auto-detects
      4D vs 5D and routes through the time encoder when DOY is provided.

Examples:
    # S2-only, 6 frames (parity with multi-task PASTIS run)
    python script_train_pastis_perceiver.py --xp_name perceiver_s2_t6 \
        --multi_temporal 6 --batch_size 2 --lr 1e-4 --epochs 100

    # S2+S1 fused (10 + 3 = 13 channels per frame), 6 frames
    python script_train_pastis_perceiver.py --xp_name perceiver_s2s1_t6 \
        --use_s1 --multi_temporal 6 --batch_size 2 --lr 1e-4 --epochs 100

    # Test-only mode
    python script_train_pastis_perceiver.py --xp_name perceiver_s2_t6 \
        --test_only ./checkpoints/pastis_perceiver/bl_perceiver_s2_t6-best.ckpt
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

from training.utils.datasets_baselines.utils_dataset_PASTIS import (
    NUM_CLASSES, IGNORE_INDEX, NUM_S1_BANDS, NUM_S2_BANDS,
    PastisBaselineDataset,
)
# These collates are the same ones used by the existing PASTIS baseline
# script — re-importing to avoid duplication.

from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.trainer_baselines import BaselineTrainer


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
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="PASTIS-HD Perceiver-IO Baseline")
parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--data_dir",   type=str, default="./data/PASTIS-HD")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Modality
parser.add_argument("--use_s1",     action="store_true",
                    help="Include S1A SAR data (default: S2-only)")

# Temporal
parser.add_argument("--multi_temporal", type=int, default=6,
                    help="Number of temporal frames to use")
parser.add_argument("--temporal_last",  action="store_true",
                    help="Take last N timesteps instead of uniform sampling")

# Training
parser.add_argument("--batch_size",  type=int, default=4)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience",    type=int, default=30)
parser.add_argument("--grad_accum",  type=int, default=1)

# Perceiver-specific
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=768)
parser.add_argument("--depth",              type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=8)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=2)
parser.add_argument("--no_weight_tie",      action="store_true",
                    help="Disable weight-tying across encoder blocks.")
parser.add_argument("--num_freq_bands",     type=int, default=6)
parser.add_argument("--max_freq",           type=float, default=10.0)
parser.add_argument("--attn_dropout",       type=float, default=0.0)
parser.add_argument("--ff_dropout",         type=float, default=0.0)

args = parser.parse_args()


# =============================================================================
# CONFIG
# =============================================================================

modalities = ["S2"]
per_frame_channels = NUM_S2_BANDS  # 10

if args.use_s1:
    modalities.append("S1")
    per_frame_channels = NUM_S2_BANDS + NUM_S1_BANDS  # 13

modality_str = "+".join(modalities)
temporal_str = f"{args.multi_temporal} frames " \
               f"({'last' if args.temporal_last else 'uniform'})"

# Perceiver-IO is treated as a temporal model — it receives [B, T, C, H, W]
# and uses doy from batch["dates"]["s2"].
is_temporal_model = True

print(f"\n{'='*60}")
print(f"  PASTIS-HD Perceiver-IO Baseline")
print(f"  Modalities: {modality_str} ({per_frame_channels} bands/frame)")
print(f"  Temporal:   {temporal_str}")
print(f"  Img size:   {args.img_size}x{args.img_size}")
print(f"  Tokens:     {args.multi_temporal * args.img_size ** 2:,} per sample")
print(f"  Queries:    {args.img_size ** 2:,} per sample")
print(f"  Latents:    {args.num_latents} x {args.latent_dim}")
print(f"  Depth:      {args.depth}")
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
    temporal_mode="sequence",       # [B, T, C, H, W] — what PerceiverSeg expects
)

train_ds = PastisBaselineDataset(mode="train",      augment=True,  **common)
val_ds   = PastisBaselineDataset(mode="validation", augment=False, **common)
test_ds  = PastisBaselineDataset(mode="test",       augment=False, **common)

print(f"  Train: {len(train_ds)} patches")
print(f"  Val:   {len(val_ds)} patches")
print(f"  Test:  {len(test_ds)} patches")


# =============================================================================
# COLLATE
# =============================================================================

# Uses the same fused collate as the rest of the PASTIS baselines:
# S2-only -> {"image": {"s2": [B, T, 10, H, W]}, "dates": {"s2": [B, T]}}
# S2+S1   -> S2 channel-fused with S1 then keyed under "s2".
collate_fn = make_fused_collate(args.use_s1)
if args.use_s1:
    print("[PASTIS-Perceiver] S2+S1 fusion: concatenating bands in collate")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=4 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False,                  **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False,                  **loader_kwargs)


# =============================================================================
# MODEL
# =============================================================================

model = PerceiverSeg(
    in_channels=per_frame_channels,
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

trainer_module = BaselineTrainer(
    model=model,
    modality="s2",                  # collate keys S2(+S1) under "s2"
    temporal=is_temporal_model,     # passes batch["dates"]["s2"] as doy
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
        run_name = f"BL_{args.xp_name}_perceiver_{modality_str}"
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

ckpt_dir = "./checkpoints/pastis_perceiver/"
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
    print(f"  Starting: perceiver_io — {modality_str}")
    print(f"  Temporal: {temporal_str}")
    print(f"  Train: folds 1,2,3 -> Val: fold 4 -> Test: fold 5")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path

    # Destroy DDP process group BEFORE the test trainer is built.
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
