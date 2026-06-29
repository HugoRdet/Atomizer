"""
FRACTAL PerceiverIO Baseline Training Script
=============================================

Single-task LIDAR + VHR semantic segmentation on FRACTAL with PerceiverIO.
Mirrors script_train_fractal.py as closely as possible so results are
directly comparable. Key differences:

  - No lookup table / band registration (PerceiverIO uses raw Fourier
    features, not spectral indices)
  - No collate_grouped (flat tensors, standard DataLoader collation)
  - No config_model YAML (hyperparameters passed directly as CLI args)
  - Dataset: FractalPerceiverDataset
  - Trainer: Model_PerceiverFractal

Examples
--------
    # From scratch
    python script_train_fractal_perceiver.py --xp_name perceiver_fractal_v1

    # Force fresh start ignoring existing checkpoints
    python script_train_fractal_perceiver.py --xp_name perceiver_fractal_v1 \\
        --no_auto_resume

    # Test-only evaluation
    python script_train_fractal_perceiver.py \\
        --xp_name perceiver_fractal_v1_test \\
        --ckpt_path ./checkpoints/fractal_perceiver/perceiver_fractal_v1-best.ckpt \\
        --test_only
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
import glob
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

from training.trainer_fractal_perceiver import Model_PerceiverFractal
from training.utils.datasets_baselines.utils_dataset_fractal_perceiver import (
    FractalPerceiverDataset,
)


# =============================================================================
# HELPERS
# =============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(
    description="FRACTAL PerceiverIO Baseline Training"
)

# Experiment
parser.add_argument("--xp_name",     type=str, required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--batch_size",  type=int, default=4)

# Dataset
parser.add_argument("--root_path",          type=str, default="./data",
                    help="Parent dir containing FRACTAL/ and FRACTAL-IRGB/.")
parser.add_argument("--max_lidar_points",   type=int, default=16_000,
                    help="Max LIDAR points per patch (context tokens).")
parser.add_argument("--max_queries",        type=int, default=32_000,
                    help="Val/test query padding target.")
parser.add_argument("--valid_patches_file", type=str, default=None,
                    help="Optional JSON listing valid patch IDs per split.")
parser.add_argument("--sigma_xy_pixels",    type=float, default=0.25,
                    help="LIDAR XY jitter std dev in pixels (0.2m/px). "
                         "Default 0.25 ≈ 5cm physical noise. "
                         "Set 0 to disable XY jitter.")
parser.add_argument("--sigma_z_normed",     type=float, default=0.003,
                    help="LIDAR Z jitter std dev in normalized units "
                         "(Z_GROUND_REL_SCALE=15m, so 0.003 ≈ 4.5cm). "
                         "Set 0 to disable Z jitter.")

# Model architecture
parser.add_argument("--num_latents",         type=int,   default=256)
parser.add_argument("--latent_dim",          type=int,   default=256)
parser.add_argument("--depth",               type=int,   default=6)
parser.add_argument("--cross_heads",         type=int,   default=1)
parser.add_argument("--latent_heads",        type=int,   default=8)
parser.add_argument("--cross_dim_head",      type=int,   default=64)
parser.add_argument("--latent_dim_head",     type=int,   default=64)
parser.add_argument("--self_per_cross_attn", type=int,            default=1)
parser.add_argument("--weight_tie_layers",   type=str2bool,       default=True,
                    help="Share encoder weights across blocks > 0. "
                         "Pass --weight_tie_layers false to disable.")
parser.add_argument("--attn_dropout",        type=float,          default=0.0)
parser.add_argument("--ff_dropout",          type=float, default=0.0)
parser.add_argument("--echo_hidden_dim",     type=int,   default=64)

# Training
parser.add_argument("--grad_accumulation", type=int, default=2,
                    help="Gradient accumulation steps. Effective batch size "
                         "= batch_size * grad_accumulation * num_gpus.")
parser.add_argument("--query_chunk_size",  type=int, default=100_000,
                    help="Decode queries in chunks during val/test to avoid "
                         "OOM on large full-scene query sets. Default 100_000.")

# Optimizer
parser.add_argument("--lr",            type=float, default=1e-4)
parser.add_argument("--weight_decay",  type=float, default=1e-2)
parser.add_argument("--warmup_steps",  type=int,   default=None)

# Loss
parser.add_argument("--ignore_index",    type=int, default=255)
parser.add_argument("--class_weighting", type=str, default="auto",
                    choices=["auto", "none"])

# Resume / test
parser.add_argument("--ckpt_path",      type=str, default=None,
                    help="Path to checkpoint for resume or test_only.")
parser.add_argument("--no_auto_resume", action="store_true",
                    help="Disable auto-resume from the latest last checkpoint.")
parser.add_argument("--wandb_run_id",   type=str, default=None)
parser.add_argument("--test_only",      action="store_true")

args = parser.parse_args()


# =============================================================================
# CHECKPOINT DIR + AUTO-RESUME LOOKUP
# =============================================================================

ckpt_dir = "./checkpoints/fractal_perceiver/"
os.makedirs(ckpt_dir, exist_ok=True)


def _find_latest_last_checkpoint(xp_name: str) -> str:
    """
    Find the most recent 'last' checkpoint for the given xp_name.
    Sorts by epoch number parsed from filename (not mtime).
    Returns path or None.
    """
    pattern = os.path.join(
        ckpt_dir, f"perceiver_fractal_{xp_name}-last-*.ckpt"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None

    def _epoch_from_name(path: str) -> int:
        nums = re.findall(r"\d+", os.path.basename(path))
        return int(nums[-1]) if nums else -1

    matches.sort(key=_epoch_from_name)
    return matches[-1]


auto_resume_ckpt = None
if (not args.test_only
        and args.ckpt_path is None
        and not args.no_auto_resume):
    auto_resume_ckpt = _find_latest_last_checkpoint(args.xp_name)
    if auto_resume_ckpt is not None:
        print(f"\n[PerceiverFRACTAL] Auto-resume: found checkpoint "
              f"{auto_resume_ckpt}")
    else:
        print(f"\n[PerceiverFRACTAL] Auto-resume: no prior checkpoint for "
              f"xp_name='{args.xp_name}' — starting fresh")


print(f"\n{'='*70}")
print(f"  FRACTAL PerceiverIO — Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Modalities:      VHR ortho (NIR/R/G/B) + LIDAR points @ 0.2m")
print(f"  Max LIDAR pts:   {args.max_lidar_points}")
print(f"  Max queries:     {args.max_queries}")
print(f"  Jitter XY:       {args.sigma_xy_pixels}px  "
      f"({'disabled' if args.sigma_xy_pixels == 0 else '~'+str(round(args.sigma_xy_pixels * 0.2 * 100))+'cm physical'})")
print(f"  Jitter Z:        {args.sigma_z_normed} normed  "
      f"({'disabled' if args.sigma_z_normed == 0 else '~'+str(round(args.sigma_z_normed * 15 * 100))+'cm physical'})")
print(f"  Batch size:      {args.batch_size}")
print(f"  Epochs:          {args.epochs}")
print(f"  Grad accum:      {args.grad_accumulation} (effective batch = {args.batch_size * args.grad_accumulation})")
print(f"  Latents:         {args.num_latents} x {args.latent_dim}")
print(f"  Depth:           {args.depth}")
print(f"  Ignore idx:      {args.ignore_index}")
print(f"  Class weights:   {args.class_weighting}")
if args.ckpt_path is not None:
    if args.test_only:
        print(f"  Mode:            TEST ONLY (ckpt: {args.ckpt_path})")
    else:
        print(f"  Resume ckpt:     {args.ckpt_path}")
elif auto_resume_ckpt is not None:
    print(f"  Auto-resume:     {auto_resume_ckpt}")


# =============================================================================
# WANDB
# =============================================================================

wandb_resume_id = args.wandb_run_id
if wandb_resume_id is None and auto_resume_ckpt is not None:
    run_id_path = (
        f"training/wandb_runs/perceiver_fractal_{args.xp_name}.txt"
    )
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            wandb_resume_id = f.read().strip()
        print(f"[PerceiverFRACTAL] Resuming wandb run id={wandb_resume_id}")

wandb_logger = WandbLogger(
    project="Atomizer-FRACTAL",
    name=f"PerceiverFRACTAL_{args.xp_name}",
    save_dir=os.environ.get("WANDB_DIR", "./wandb"),
    config={
        "num_latents":         args.num_latents,
        "latent_dim":          args.latent_dim,
        "depth":               args.depth,
        "cross_heads":         args.cross_heads,
        "latent_heads":        args.latent_heads,
        "self_per_cross_attn": args.self_per_cross_attn,
        "weight_tie_layers":   args.weight_tie_layers,
        "attn_dropout":        args.attn_dropout,
        "ff_dropout":          args.ff_dropout,
        "echo_hidden_dim":     args.echo_hidden_dim,
        "max_lidar_points":    args.max_lidar_points,
        "max_queries":         args.max_queries,
        "sigma_xy_pixels":     args.sigma_xy_pixels,
        "sigma_z_normed":      args.sigma_z_normed,
        "batch_size":          args.batch_size,
        "epochs":              args.epochs,
        "lr":                  args.lr,
        "weight_decay":        args.weight_decay,
        "ignore_index":        args.ignore_index,
        "class_weighting":     args.class_weighting,
        "grad_accumulation":   args.grad_accumulation,
        "auto_resume_ckpt":    auto_resume_ckpt,
    },
    id=wandb_resume_id,
    resume="must" if wandb_resume_id is not None else None,
)


# =============================================================================
# DATASETS + DATALOADERS
# =============================================================================

def build_dataset(mode: str) -> FractalPerceiverDataset:
    return FractalPerceiverDataset(
        root_path=args.root_path,
        mode=mode,
        max_lidar_points=args.max_lidar_points,
        max_queries=args.max_queries,
        valid_patches_file=args.valid_patches_file,
        use_augmentation=(mode == "train"),
        sigma_xy_pixels=args.sigma_xy_pixels,
        sigma_z_normed=args.sigma_z_normed,
    )


def make_loader(dataset: FractalPerceiverDataset,
                shuffle: bool) -> DataLoader:
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        # No custom collate_fn needed — all tensors are already padded to
        # fixed sizes in __getitem__, so default collation works.
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=shuffle,
    )


print(f"\n[PerceiverFRACTAL] Building datasets...")
train_ds = build_dataset("train")
val_ds   = build_dataset("val")
test_ds  = build_dataset("test")
print(f"[PerceiverFRACTAL] Sizes: "
      f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")


# =============================================================================
# DataModule
# =============================================================================

class FractalPerceiverDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return make_loader(train_ds, shuffle=True)

    def val_dataloader(self):
        return make_loader(val_ds, shuffle=False)

    def test_dataloader(self):
        return make_loader(test_ds, shuffle=False)


data_module = FractalPerceiverDataModule()


# =============================================================================
# MODEL
# =============================================================================

class_weights_arg = "auto" if args.class_weighting == "auto" else None

model = Model_PerceiverFractal(
    query_chunk_size=args.query_chunk_size,
    num_latents=args.num_latents,
    latent_dim=args.latent_dim,
    depth=args.depth,
    cross_heads=args.cross_heads,
    latent_heads=args.latent_heads,
    cross_dim_head=args.cross_dim_head,
    latent_dim_head=args.latent_dim_head,
    self_per_cross_attn=args.self_per_cross_attn,
    weight_tie_layers=args.weight_tie_layers,
    attn_dropout=args.attn_dropout,
    ff_dropout=args.ff_dropout,
    echo_hidden_dim=args.echo_hidden_dim,
    lr=args.lr,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    ignore_index=args.ignore_index,
    class_weights=class_weights_arg,
)


# =============================================================================
# CALLBACKS + TRAINER
# =============================================================================

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"perceiver_fractal_{args.xp_name}"
                 f"-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=5,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"perceiver_fractal_{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1,
    max_epochs=args.epochs,
    accelerator="gpu",
    precision="32-true",
    logger=wandb_logger,
    accumulate_grad_batches=args.grad_accumulation,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)


# =============================================================================
# RESUME / INIT
# =============================================================================

resume_ckpt_path = None

if args.ckpt_path is not None and not args.test_only:
    resume_ckpt_path = args.ckpt_path
    print(f"\n[PerceiverFRACTAL] Resuming from {args.ckpt_path}")
elif auto_resume_ckpt is not None and not args.test_only:
    resume_ckpt_path = auto_resume_ckpt
    print(f"\n[PerceiverFRACTAL] Auto-resuming from {resume_ckpt_path}")


# =============================================================================
# TRAIN / TEST
# =============================================================================

if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  PerceiverFRACTAL — TEST ONLY\n"
          f"  ckpt: {args.ckpt_path}\n{'='*70}\n")

    ckpt   = torch.load(args.ckpt_path, map_location="cpu",
                        weights_only=False)
    state  = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[PerceiverFRACTAL] missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")

    trainer.test(model, datamodule=data_module, verbose=True)

else:
    print(f"\n{'='*70}\n  PerceiverFRACTAL — TRAINING\n{'='*70}\n")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    print(f"\n{'='*70}\n  PerceiverFRACTAL — FINAL TEST\n{'='*70}\n")
    trainer.test(model, datamodule=data_module, verbose=True, ckpt_path="best")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================

if wandb_logger is not None and trainer.is_global_zero:
    import wandb
    run = getattr(wandb, "run", None)
    if run is not None:
        os.makedirs("training/wandb_runs", exist_ok=True)
        run_id_path = (
            f"training/wandb_runs/perceiver_fractal_{args.xp_name}.txt"
        )
        with open(run_id_path, "w") as f:
            f.write(run.id)
        print(f"WANDB_RUN_ID: {run.id}")
