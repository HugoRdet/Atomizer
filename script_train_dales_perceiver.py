"""
DALES PerceiverIO Baseline Training Script
=============================================

Single-task LIDAR-only semantic segmentation on DALES with PerceiverIO.
Mirrors script_train_fractal_perceiver.py as closely as possible so results
are directly comparable to the FRACTAL PerceiverIO baseline. Key differences
from that script, all DALES-specific:

  - No VHR modality (DALES is LIDAR-only) -- dataset/model have no
    vhr_tokens/vhr_mask at all, not just zeroed-out ones.
  - Intensity gets its own MLP alongside echo (see perceiver_dales.py) --
    a deliberate content addition since DALES has no VHR to compensate for
    omitting it, unlike FRACTAL where VHR carries most of the value signal.
  - Dataset: DalesPerceiverDataset, reusing the SAME tiled .laz patches as
    Atomizer's DalesDataset (see DALES_DATASET.md) -- no separate data prep.
  - Trainer: Model_PerceiverDales (8 DALES classes, sqrt-inverse-frequency
    loss weighting matching Atomizer's own DALES trainer, for a fair
    comparison of ARCHITECTURE, not loss setup).

Examples
--------
    # From scratch
    python script_train_dales_perceiver.py --xp_name perceiver_dales_v1

    # Force fresh start ignoring existing checkpoints
    python script_train_dales_perceiver.py --xp_name perceiver_dales_v1 \\
        --no_auto_resume

    # Test-only evaluation
    python script_train_dales_perceiver.py \\
        --xp_name perceiver_dales_v1_test \\
        --ckpt_path ./checkpoints/dales_perceiver/perceiver_dales_v1-best.ckpt \\
        --test_only
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
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

from training.trainer_dales_perceiver import Model_PerceiverDales
from training.utils.datasets_baselines.utils_dataset_DALES_perceiver import (
    DalesPerceiverDataset,
)


# =============================================================================
# HELPERS
# =============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


def wait_for_checkpoint(path: str, wait_seconds: int, poll_interval: int = 15) -> str:
    """
    Polls for `path` to exist, up to `wait_seconds` total, checking every
    `poll_interval` seconds. Useful for chained SLURM jobs where the next
    job in the chain can start before the previous job's checkpoint write
    (and any filesystem sync delay, common on Lustre) has actually landed.

    wait_seconds=0 means "check once, don't wait" -- matches the old
    --ckpt_path behavior of just failing fast if the file isn't there.

    Raises FileNotFoundError if the checkpoint never appears within the
    timeout, rather than silently falling back to training from scratch --
    resuming should be an explicit, verified action.
    """
    if os.path.exists(path):
        return path

    if wait_seconds <= 0:
        raise FileNotFoundError(
            f"--resume_from checkpoint not found: {path} "
            f"(use --resume_wait_seconds > 0 to poll for it instead of failing immediately)"
        )

    print(f"[PerceiverDALES] Checkpoint not found yet: {path}")
    print(f"[PerceiverDALES] Waiting up to {wait_seconds}s (polling every {poll_interval}s)...")
    waited = 0
    while waited < wait_seconds:
        time.sleep(poll_interval)
        waited += poll_interval
        if os.path.exists(path):
            print(f"[PerceiverDALES] Checkpoint appeared after {waited}s: {path}")
            return path
        print(f"[PerceiverDALES]   ...still waiting ({waited}/{wait_seconds}s)")

    raise FileNotFoundError(
        f"--resume_from checkpoint still not found after waiting {wait_seconds}s: {path}"
    )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="DALES PerceiverIO Baseline Training")

# Experiment
parser.add_argument("--xp_name",     type=str, required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--batch_size",  type=int, default=2)

# Dataset
parser.add_argument("--root_path",          type=str, default="./data",
                    help="Parent dir containing DALES/{train,val,test} -- "
                         "the SAME tiled patches used by Atomizer's "
                         "DalesDataset, see DALES_DATASET.md.")
parser.add_argument("--max_lidar_points",   type=int, default=256_000,
                    help="Max LIDAR points per patch (context tokens).")
parser.add_argument("--max_queries",        type=int, default=256_000,
                    help="Query padding target.")
parser.add_argument("--valid_patches_file", type=str, default=None)
parser.add_argument("--sigma_xy_m",         type=float, default=0.05,
                    help="LIDAR XY jitter std dev in METERS (physical "
                         "units directly, unlike FRACTAL's pixel-based "
                         "sigma_xy_pixels -- DalesPerceiverDataset has no "
                         "pixel grid to reference). Default 5cm. Set 0 "
                         "to disable.")
parser.add_argument("--sigma_z_normed",     type=float, default=0.003,
                    help="LIDAR Z jitter std dev in normalized units "
                         "(Z_GROUND_REL_SCALE=15m, so 0.003 ~ 4.5cm). "
                         "Set 0 to disable.")
parser.add_argument("--min_points",         type=int, default=1000,
                    help="Minimum points for a patch to be processed "
                         "(below this, silently substitutes a different "
                         "patch -- see DALES_DATASET.md section on "
                         "evaluation completeness; pass 1 for test-time "
                         "full coverage).")

# Model architecture
parser.add_argument("--num_latents",         type=int,   default=512)
parser.add_argument("--latent_dim",          type=int,   default=768)
parser.add_argument("--depth",               type=int,   default=1)
parser.add_argument("--cross_heads",         type=int,   default=8)
parser.add_argument("--latent_heads",        type=int,   default=8)
parser.add_argument("--cross_dim_head",      type=int,   default=64)
parser.add_argument("--latent_dim_head",     type=int,   default=64)
parser.add_argument("--self_per_cross_attn", type=int,   default=6)
parser.add_argument("--weight_tie_layers",   type=str2bool, default=True)
parser.add_argument("--attn_dropout",        type=float, default=0.0)
parser.add_argument("--ff_dropout",          type=float, default=0.0)
parser.add_argument("--echo_hidden_dim",     type=int,   default=64)
parser.add_argument("--intensity_hidden_dim", type=int,  default=64,
                    help="Hidden dim of the intensity MLP -- has no "
                         "FRACTAL equivalent, since FRACTAL's LIDAR "
                         "baseline doesn't use intensity at all.")

# Training
parser.add_argument("--grad_accumulation", type=int, default=2)
parser.add_argument("--query_chunk_size",  type=int, default=100_000)

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
                    help="[Deprecated alias for --resume_from when NOT "
                         "using --test_only, kept for backward compat] "
                         "Also still the required arg for --test_only.")
parser.add_argument("--resume_from",    type=str, default=None,
                    help="Path to a checkpoint to resume training from. If "
                         "the file doesn't exist yet, use "
                         "--resume_wait_seconds to poll for it instead of "
                         "failing immediately (useful for chained SLURM "
                         "jobs where the next job can start before the "
                         "previous one's checkpoint write has landed).")
parser.add_argument("--resume_wait_seconds", type=int, default=0,
                    help="How long to poll for --resume_from (or "
                         "--ckpt_path with --test_only) to appear before "
                         "giving up (0 = check once, fail immediately if "
                         "missing).")
parser.add_argument("--resume_poll_interval", type=int, default=15,
                    help="Seconds between polls while waiting.")
parser.add_argument("--no_auto_resume", action="store_true")
parser.add_argument("--wandb_run_id",   type=str, default=None)
parser.add_argument("--test_only",      action="store_true")

args = parser.parse_args()

# --resume_from takes precedence over --ckpt_path for the RESUME path;
# --ckpt_path stays required (unchanged) for --test_only.
resume_ckpt_path_explicit = args.resume_from or (args.ckpt_path if not args.test_only else None)
if args.ckpt_path and not args.resume_from and not args.test_only:
    print("[PerceiverDALES] NOTE: --ckpt_path is deprecated for resuming "
          "training, use --resume_from instead (still honored for "
          "backward compat).")


# =============================================================================
# CHECKPOINT DIR + AUTO-RESUME LOOKUP
# =============================================================================

ckpt_dir = "./checkpoints/dales_perceiver/"
os.makedirs(ckpt_dir, exist_ok=True)


def _find_latest_last_checkpoint(xp_name: str) -> str:
    pattern = os.path.join(ckpt_dir, f"perceiver_dales_{xp_name}-last-*.ckpt")
    matches = glob.glob(pattern)
    if not matches:
        return None

    def _epoch_from_name(path: str) -> int:
        nums = re.findall(r"\d+", os.path.basename(path))
        return int(nums[-1]) if nums else -1

    matches.sort(key=_epoch_from_name)
    return matches[-1]


auto_resume_ckpt = None
if not args.test_only and resume_ckpt_path_explicit is None and not args.no_auto_resume:
    auto_resume_ckpt = _find_latest_last_checkpoint(args.xp_name)
    if auto_resume_ckpt is not None:
        print(f"\n[PerceiverDALES] Auto-resume: found checkpoint {auto_resume_ckpt}")
    else:
        print(f"\n[PerceiverDALES] Auto-resume: no prior checkpoint for "
              f"xp_name='{args.xp_name}' -- starting fresh")


print(f"\n{'='*70}")
print(f"  DALES PerceiverIO -- Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Modality:        LIDAR-only (NO VHR)")
print(f"  Max LIDAR pts:   {args.max_lidar_points}")
print(f"  Max queries:     {args.max_queries}")
print(f"  Jitter XY:       {args.sigma_xy_m}m  "
      f"({'disabled' if args.sigma_xy_m == 0 else 'physical meters, no pixel grid'})")
print(f"  Jitter Z:        {args.sigma_z_normed} normed")
print(f"  Batch size:      {args.batch_size}")
print(f"  Epochs:          {args.epochs}")
print(f"  Grad accum:      {args.grad_accumulation} "
      f"(effective batch = {args.batch_size * args.grad_accumulation})")
print(f"  Latents:         {args.num_latents} x {args.latent_dim}")
print(f"  Depth:           {args.depth}")
print(f"  Intensity MLP:   hidden={args.intensity_hidden_dim} "
      f"(no FRACTAL equivalent -- see module docstring)")
print(f"  Ignore idx:      {args.ignore_index}")
print(f"  Class weights:   {args.class_weighting}")
if args.test_only:
    print(f"  Mode:            TEST ONLY (ckpt: {args.ckpt_path})")
elif resume_ckpt_path_explicit is not None:
    print(f"  Resume ckpt:     {resume_ckpt_path_explicit} "
          f"(wait up to {args.resume_wait_seconds}s if not found yet)")
elif auto_resume_ckpt is not None:
    print(f"  Auto-resume:     {auto_resume_ckpt}")


# =============================================================================
# WANDB
# =============================================================================

wandb_resume_id = args.wandb_run_id
if wandb_resume_id is None and auto_resume_ckpt is not None:
    run_id_path = f"training/wandb_runs/perceiver_dales_{args.xp_name}.txt"
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            wandb_resume_id = f.read().strip()
        print(f"[PerceiverDALES] Resuming wandb run id={wandb_resume_id}")

wandb_logger = WandbLogger(
    project="Atomizer-DALES-Perceiver",
    name=f"PerceiverDALES_{args.xp_name}",
    save_dir=os.environ.get("WANDB_DIR", "./wandb"),
    config={
        "num_latents":          args.num_latents,
        "latent_dim":           args.latent_dim,
        "depth":                args.depth,
        "cross_heads":          args.cross_heads,
        "latent_heads":         args.latent_heads,
        "self_per_cross_attn":  args.self_per_cross_attn,
        "weight_tie_layers":    args.weight_tie_layers,
        "attn_dropout":         args.attn_dropout,
        "ff_dropout":           args.ff_dropout,
        "echo_hidden_dim":      args.echo_hidden_dim,
        "intensity_hidden_dim": args.intensity_hidden_dim,
        "max_lidar_points":     args.max_lidar_points,
        "max_queries":          args.max_queries,
        "sigma_xy_m":           args.sigma_xy_m,
        "sigma_z_normed":       args.sigma_z_normed,
        "min_points":           args.min_points,
        "batch_size":           args.batch_size,
        "epochs":               args.epochs,
        "lr":                   args.lr,
        "weight_decay":         args.weight_decay,
        "ignore_index":         args.ignore_index,
        "class_weighting":      args.class_weighting,
        "grad_accumulation":    args.grad_accumulation,
        "auto_resume_ckpt":     auto_resume_ckpt,
    },
    id=wandb_resume_id,
    resume="must" if wandb_resume_id is not None else None,
)


# =============================================================================
# DATASETS + DATALOADERS
# =============================================================================

def build_dataset(mode: str) -> DalesPerceiverDataset:
    return DalesPerceiverDataset(
        root_path=args.root_path,
        mode=mode,
        max_lidar_points=args.max_lidar_points,
        max_queries=args.max_queries,
        valid_patches_file=args.valid_patches_file,
        use_augmentation=(mode == "train"),
        sigma_xy_m=args.sigma_xy_m,
        sigma_z_normed=args.sigma_z_normed,
        min_points=args.min_points,
    )


def make_loader(dataset: DalesPerceiverDataset, shuffle: bool) -> DataLoader:
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        # No custom collate_fn -- all tensors pre-padded to fixed sizes.
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=shuffle,
    )


print(f"\n[PerceiverDALES] Building datasets...")
train_ds = build_dataset("train")
val_ds   = build_dataset("val")
test_ds  = build_dataset("test")
print(f"[PerceiverDALES] Sizes: train={len(train_ds)}  val={len(val_ds)}  "
      f"test={len(test_ds)}")


# =============================================================================
# DataModule
# =============================================================================

class DalesPerceiverDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return make_loader(train_ds, shuffle=True)

    def val_dataloader(self):
        return make_loader(val_ds, shuffle=False)

    def test_dataloader(self):
        return make_loader(test_ds, shuffle=False)


data_module = DalesPerceiverDataModule()


# =============================================================================
# MODEL
# =============================================================================

class_weights_arg = "auto" if args.class_weighting == "auto" else None

model = Model_PerceiverDales(
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
    intensity_hidden_dim=args.intensity_hidden_dim,
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
        filename=f"perceiver_dales_{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=5,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"perceiver_dales_{args.xp_name}-last-{{epoch:02d}}",
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
if resume_ckpt_path_explicit is not None and not args.test_only:
    resume_ckpt_path = wait_for_checkpoint(
        resume_ckpt_path_explicit, args.resume_wait_seconds, args.resume_poll_interval
    )
    print(f"\n[PerceiverDALES] Resuming from {resume_ckpt_path}")
elif auto_resume_ckpt is not None and not args.test_only:
    # Found via glob -- exists by construction, no need to wait for it.
    resume_ckpt_path = auto_resume_ckpt
    print(f"\n[PerceiverDALES] Auto-resuming from {resume_ckpt_path}")


# =============================================================================
# TRAIN / TEST
# =============================================================================

if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    ckpt_to_load = wait_for_checkpoint(
        args.ckpt_path, args.resume_wait_seconds, args.resume_poll_interval
    )

    print(f"\n{'='*70}\n  PerceiverDALES -- TEST ONLY\n"
          f"  ckpt: {ckpt_to_load}\n{'='*70}\n")

    ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[PerceiverDALES] missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")

    trainer.test(model, datamodule=data_module, verbose=True)

else:
    print(f"\n{'='*70}\n  PerceiverDALES -- TRAINING\n{'='*70}\n")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    print(f"\n{'='*70}\n  PerceiverDALES -- FINAL TEST\n{'='*70}\n")
    trainer.test(model, datamodule=data_module, verbose=True, ckpt_path="best")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================

if wandb_logger is not None and trainer.is_global_zero:
    import wandb
    run = getattr(wandb, "run", None)
    if run is not None:
        os.makedirs("training/wandb_runs", exist_ok=True)
        run_id_path = f"training/wandb_runs/perceiver_dales_{args.xp_name}.txt"
        with open(run_id_path, "w") as f:
            f.write(run.id)
        print(f"WANDB_RUN_ID: {run.id}")
