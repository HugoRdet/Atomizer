"""
PASTIS-HD Perceiver-IO (SKIP / temporal-query) Single-Task Training Script
==========================================================================

Trains PerceiverSegPASTIS (per-pixel, time-aggregated query via 1x1 conv over
the pixel's T*C temporal features) on PASTIS-HD.

Key differences vs the baseline perceiver script:
    - Model: PerceiverSegPASTIS (from training.perceiverIO.perceiver_temp)
    - num_frames passed to the model (fixed-T 1x1 conv query).
    - Fused collate pads/truncates to EXACTLY multi_temporal frames so the
      model's fixed-T assert never fires.

Examples:
    # S2+S1 fused (10 + 3 = 13 channels/frame), 10 frames
    python script_train_pastis_perceiver_skip.py --xp_name perc_skip_s2s1_t10 \
        --use_s1 --multi_temporal 10 --batch_size 2 --lr 1e-4 --epochs 100
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

# >>> SKIP: the temporal-query Perceiver lives in perceiver_temp.py
from training.perceiverIO.perceiver_temp import PerceiverSegPASTIS
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
    return {"image": images, "dates": dates, "target": targets, "metadata": metadata}


def _fix_T(x, T_target, time_dim_is_last_minus3=True):
    """Pad (zeros) or truncate the temporal axis (dim=1) of [B,T,C,H,W] to T_target."""
    T = x.shape[1]
    if T == T_target:
        return x
    if T > T_target:
        return x[:, :T_target]
    # pad with zeros at the end
    pad_shape = list(x.shape)
    pad_shape[1] = T_target - T
    pad = torch.zeros(pad_shape, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def make_fused_collate(use_s1: bool, num_frames: int):
    """
    Returns a collate that fuses S2 and S1 along channels and guarantees
    EXACTLY num_frames temporal slots (pad/truncate), so the model's fixed-T
    assert never fires. Dates are fixed to num_frames the same way.
    """
    def fused_collate(batch):
        out = pastis_collate(batch)
        s2 = out["image"]["s2"]                      # [B, T2, 10, H, W]
        s2_dates = out["dates"]["s2"]                # [B, T2]

        if use_s1:
            s1 = out["image"]["s1"]                  # [B, T1, 3, H, W]
            T = min(s2.shape[1], s1.shape[1])
            fused = torch.cat([s2[:, :T], s1[:, :T]], dim=2)   # [B, T, 13, H, W]
            dates = s2_dates[:, :T]
        else:
            fused = s2
            dates = s2_dates

        # guarantee exactly num_frames temporal slots
        fused = _fix_T(fused, num_frames)
        # fix dates length too (pad with zeros / truncate)
        if dates.shape[1] != num_frames:
            if dates.shape[1] > num_frames:
                dates = dates[:, :num_frames]
            else:
                padd = torch.zeros(dates.shape[0], num_frames - dates.shape[1],
                                   dtype=dates.dtype)
                dates = torch.cat([dates, padd], dim=1)

        out["image"] = {"s2": fused}
        out["dates"] = {"s2": dates}
        return out

    return fused_collate


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="PASTIS-HD Perceiver-IO SKIP")
parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--data_dir",   type=str, default="./data/PASTIS-HD")
parser.add_argument("--test_only", type=str, default=None)
parser.add_argument("--use_s1",     action="store_true")
parser.add_argument("--multi_temporal", type=int, default=10)
parser.add_argument("--temporal_last",  action="store_true")
parser.add_argument("--batch_size",  type=int, default=1)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience",    type=int, default=30)
parser.add_argument("--grad_accum",  type=int, default=1)
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=768)
parser.add_argument("--agg_dim",            type=int, default=128)
parser.add_argument("--depth",              type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=1)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=6)
parser.add_argument("--no_weight_tie",      action="store_true")
parser.add_argument("--num_freq_bands",     type=int, default=16)
parser.add_argument("--max_freq",           type=float, default=16.0)
parser.add_argument("--attn_dropout",       type=float, default=0.0)
parser.add_argument("--ff_dropout",         type=float, default=0.0)
parser.add_argument("--img_size",           type=int, default=128)
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
temporal_str = f"{args.multi_temporal} frames ({'last' if args.temporal_last else 'uniform'})"

print(f"\n{'='*60}")
print(f"  PASTIS-HD Perceiver-IO SKIP (temporal-query)")
print(f"  Modalities: {modality_str} ({per_frame_channels} bands/frame)")
print(f"  Temporal:   {temporal_str}  (FIXED T={args.multi_temporal})")
print(f"  Img size:   {args.img_size}x{args.img_size}")
print(f"  Query:      1x1conv(T*C={per_frame_channels*args.multi_temporal}->{args.agg_dim}) + pos")
print(f"  Latents:    {args.num_latents} x {args.latent_dim}")
print(f"  Epochs:     {args.epochs}   BS: {args.batch_size}   LR: {args.lr}")
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
    temporal_mode="sequence",       # [B, T, C, H, W]
)
train_ds = PastisBaselineDataset(mode="train",      augment=True,  **common)
val_ds   = PastisBaselineDataset(mode="validation", augment=False, **common)
test_ds  = PastisBaselineDataset(mode="test",       augment=False, **common)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")


# =============================================================================
# COLLATE  (guarantees exactly multi_temporal frames)
# =============================================================================
collate_fn = make_fused_collate(args.use_s1, args.multi_temporal)


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
train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False,                 **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False,                 **loader_kwargs)


# =============================================================================
# MODEL  (PerceiverSegPASTIS — fixed-T temporal-query skip)
# =============================================================================
model = PerceiverSegPASTIS(
    in_channels=per_frame_channels,
    num_classes=NUM_CLASSES,
    img_size=args.img_size,
    num_frames=args.multi_temporal,          # >>> SKIP: fixed-T for the 1x1 conv query
    agg_dim=args.agg_dim,                     # >>> SKIP: aggregator output dim
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
    modality="s2",
    temporal=True,                  # passes batch["dates"]["s2"] as doy
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
        run_name = f"perc_skip_{args.xp_name}_{modality_str}"
        wandb.init(name=run_name, project="Atomizer_PASTIS_Baselines", config=vars(args))
        wandb_logger = WandbLogger(project="Atomizer_PASTIS_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN
# =============================================================================
ckpt_dir = "./checkpoints/pastis_perceiver_skip/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"perc_skip_{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
            monitor="val_mIoU", mode="max", save_top_k=1, verbose=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"perc_skip_{args.xp_name}-last",
            every_n_epochs=1, save_top_k=1, save_last=True,
        ),
        EarlyStopping(monitor="val_mIoU", mode="max", patience=args.patience, verbose=True),
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

    print(f"\n{'='*60}\n  Starting: perceiver SKIP — {modality_str}\n"
          f"  {temporal_str}\n{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)
    best_ckpt = trainer.checkpoint_callback.best_model_path

    import torch.distributed as dist
    is_rank_zero = trainer.is_global_zero
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if not is_rank_zero:
        if wandb_logger:
            import wandb; wandb.finish()
        raise SystemExit(0)
else:
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(f"--test_only checkpoint not found: {args.test_only}")
    best_ckpt = args.test_only
    print(f"\n[test-only mode] testing checkpoint:\n  {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================
print(f"\n{'='*60}\n  Testing checkpoint: {best_ckpt}\n{'='*60}\n")
test_trainer = Trainer(
    devices=1, accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, default_root_dir=ckpt_dir,
)
test_trainer.test(trainer_module, test_loader, ckpt_path=best_ckpt)

if wandb_logger:
    import wandb; wandb.finish()
