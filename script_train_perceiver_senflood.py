"""
Sen1Floods11 Perceiver-IO Single-Task Training Script
========================================================

Trains the Perceiver-IO baseline on Sen1Floods11 — binary flood / no-flood
segmentation, single-frame, 13 S2 + 2 S1 channels merged.

Token layout (matches PerceiverSeg.forward):
    Per-token feature = [reflectance(C=15), Fourier(x, y), no_time_vector]
    Input tokens     = [B, H*W, C + pos_dim + time_dim]
    Output queries   = [B, H*W, query_dim]

Single-frame so no DOY is passed; the model's `no_time_vector` is used
in the time slot for both tokens and queries.

Same dataset protocol as the Sen1Floods11 baseline:
    - Same train/val/test splits
    - Same normalization (per-band z-score, normalization_stats.pt)
    - Same NaN cleanup, ignore_index=255
    - D4 augmentation in training

Differs from the baseline script: full 512x512 used throughout (no random
crop), matching the BurnScars Perceiver run. Token count = 512^2 = 262,144.
If you OOM, drop --batch_size or --num_latents.

Examples:
    python script_train_senflood_perceiver.py --xp_name perceiver_senflood \
        --batch_size 2 --lr 1e-4 --epochs 80

    # Test-only mode
    python script_train_senflood_perceiver.py --xp_name perceiver_senflood \
        --test_only ./checkpoints/senflood_perceiver/bl_perceiver_senflood-best.ckpt
"""

import argparse
import os
import csv

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

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = Sen1Floods11BaselineDataset.NUM_CLASSES        # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX       # 255
NUM_CHANNELS = Sen1Floods11BaselineDataset.NUM_CHANNELS       # 15
MODALITY_KEY = "s2s1"

# Sen1Floods11 patches are 512x512 natively — full image used throughout.
NATIVE_SIZE = 512


# =============================================================================
# COLLATE
# =============================================================================

def senflood_collate(batch):
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
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Sen1Floods11 Perceiver-IO Baseline")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--data_dir",  type=str, default="./data/SENFLOOD")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Training
parser.add_argument("--batch_size",   type=int, default=1)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=100)
parser.add_argument("--num_workers",  type=int, default=8)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Spatial — Sen1Floods11 native is 512x512, used throughout (no crop).
parser.add_argument("--img_size", type=int, default=NATIVE_SIZE,
                    help=f"Spatial size. Default {NATIVE_SIZE} (full Sen1Floods11 patch).")

# Perceiver-IO config (matches the PASTIS/MADOS/BurnScars runs for parameter parity)
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=768)
parser.add_argument("--depth",              type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=16)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=6)
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

# crop_size=None -> full image (no random crop, no center crop).
train_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=None, augment=True,
)
val_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=None, augment=False,
)
test_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  Sen1Floods11 Perceiver-IO Baseline")
print(f"  Channels:     {NUM_CHANNELS} (13 S2 + 2 S1)")
print(f"  Patch size:   {args.img_size}x{args.img_size} (full image)")
print(f"  Tokens:       {args.img_size ** 2:,} per sample")
print(f"  Queries:      {args.img_size ** 2:,} per sample")
print(f"  Latents:      {args.num_latents} x {args.latent_dim}")
print(f"  Depth:        {args.depth}")
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
    collate_fn=senflood_collate,
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

model = PerceiverSeg(
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

trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    temporal=False,                   # single-frame; no DOY
    task="senflood",
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
        run_name = f"BL_{args.xp_name}_perceiver"
        wandb.init(
            name=run_name,
            project="Atomizer_SenFlood_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_SenFlood_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/senflood_perceiver/"
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
    print(f"  Starting: perceiver on Sen1Floods11")
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


# =============================================================================
# GFLOPS MEASUREMENT (torch.profiler, after scoring)
# =============================================================================

def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b

gflops = float("nan")
PROFILE_DIR = "./profiler"
tag = f"{args.xp_name}_test"
out_dir = os.path.join(PROFILE_DIR, tag)
os.makedirs(out_dir, exist_ok=True)
print(f"\n[Profile] Saving profiler artifacts to {out_dir}/")

try:
    from torch.profiler import profile, ProfilerActivity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # To avoid the PyTorch Lightning wrapper overhead, we pull the base model
    # directly and pass the image tensor into it.
    profiler_model = trainer_module.model.to(device)
    profiler_model.eval()

    n_profile = 30
    n_warmup  = 1
    batches = []

    for i, b in enumerate(test_loader):
        batches.append(_to_device(b, device))
        if len(batches) >= n_profile + n_warmup:
            break

    if not batches:
        print("[Profile] No test batches available; skipping profiling.")
    else:
        with torch.no_grad():
            # Warmup
            for b in batches[:n_warmup]:
                _ = profiler_model(b["image"][MODALITY_KEY])
            if device == "cuda":
                torch.cuda.synchronize()

            flops_list = []
            prof_last = None

            for b in batches[n_warmup:]:
                img_tensor = b["image"][MODALITY_KEY]
                with profile(activities=[ProfilerActivity.CPU,
                                         ProfilerActivity.CUDA],
                             with_flops=True,
                             record_shapes=True,
                             profile_memory=True) as prof:
                    _ = profiler_model(img_tensor)
                    if device == "cuda":
                        torch.cuda.synchronize()

                total = sum(evt.flops for evt in prof.key_averages()
                            if getattr(evt, "flops", None))
                flops_list.append(total)
                prof_last = prof

            if flops_list:
                mean_flops = sum(flops_list) / len(flops_list)
                gflops = mean_flops / 1e9
                print(f"[Profile] GFLOPs/forward (mean of {len(flops_list)} "
                      f"passes): {gflops:.3f}  "
                      f"[lower bound; profiler-counted ops only]")

            if prof_last is not None:
                ka = prof_last.key_averages()

                # Chrome trace
                try:
                    trace_path = os.path.join(out_dir, f"trace_{tag}.json")
                    prof_last.export_chrome_trace(trace_path)
                except Exception as ee:
                    pass

                # Region Summary Logic
                try:
                    # Specific custom regions requested
                    REGION_LABELS = {
                        "Self Attention",
                        "Encoder Cross Attention",
                        "Decoder"
                    }

                    evlist = prof_last.events()

                    def _start(e):
                        for a in ("time_range",):
                            tr = getattr(e, a, None)
                            if tr is not None:
                                return tr.start, tr.end
                        s = getattr(e, "cpu_interval", None)
                        if s is not None:
                            return s.start, s.end
                        return None

                    label_iv = []
                    for e in evlist:
                        nm = getattr(e, "name", getattr(e, "key", ""))
                        if nm in REGION_LABELS:
                            iv = _start(e)
                            if iv:
                                label_iv.append((nm, iv[0], iv[1]))

                    region_flops = {n: 0 for n in REGION_LABELS}
                    region_cuda  = {n: 0.0 for n in REGION_LABELS}
                    region_count = {n: 0 for n in REGION_LABELS}

                    for e in evlist:
                        fl = getattr(e, "flops", None) or 0
                        cu = getattr(e, "cuda_time_total", 0) or 0
                        if fl == 0 and cu == 0:
                            continue
                        iv = _start(e)
                        if not iv:
                            continue
                        s, en = iv
                        best = None
                        best_span = None
                        for (nm, ls, le) in label_iv:
                            if ls <= s and en <= le:
                                span = le - ls
                                if best_span is None or span < best_span:
                                    best, best_span = nm, span
                        if best is not None:
                            region_flops[best] += fl
                            region_cuda[best]  += cu
                            region_count[best] += 1

                    region_path = os.path.join(out_dir, f"regions_{tag}.csv")
                    with open(region_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["region", "gflops", "cuda_ms", "leaf_ops"])
                        for nm in sorted(REGION_LABELS,
                                         key=lambda n: region_flops[n],
                                         reverse=True):
                            w.writerow([nm, region_flops[nm]/1e9,
                                        region_cuda[nm]/1e3, region_count[nm]])

                    print("\n[Profile] Per-region breakdown (GFLOPs | CUDA ms | leaf ops):")
                    any_nonzero = False
                    for nm in sorted(REGION_LABELS, key=lambda n: region_flops[n], reverse=True):
                        gf = region_flops[nm]/1e9
                        cu = region_cuda[nm]/1e3
                        if region_flops[nm] > 0 or region_cuda[nm] > 0:
                            any_nonzero = True
                        print(f"[Profile]   {nm:<26} {gf:>9.1f} | {cu:>7.2f} ms | {region_count[nm]:>4d}")

                    if not any_nonzero:
                        print("[Profile]   (no FLOPs attributed — verify record_function placement)")

                except Exception as ee:
                    print(f"[Profile] region summary failed: {ee}")

except Exception as e:
    import traceback
    print(f"[Profile] GFLOPs measurement failed: {e}")
    gflops = float("nan")

print(f"\nRESULT xp={args.xp_name} test_gflops={gflops:.6f}")

if wandb_logger:
    wandb.finish()
