"""
Atomiser ForestNet Training Script
====================================

Single-temporal classification training on geo-bench m-forestnet (12 classes).

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training entirely
    and run test on the provided checkpoint with a single GPU. Useful for
    re-evaluating saved checkpoints without retraining.

--resume_from mode (NEW):
    Pass --resume_from <path/to/checkpoint.ckpt> to resume TRAINING from
    that checkpoint (full trainer state: model, optimizer, LR scheduler,
    epoch/step counters, callback state e.g. EarlyStopping/ModelCheckpoint's
    best-score tracking), via Trainer.fit(ckpt_path=...). If the file
    doesn't exist yet, use --resume_wait_seconds to poll for it instead of
    failing immediately (useful for chained SLURM jobs where the next job
    in the chain can start before the previous job's checkpoint write --
    and any filesystem sync delay, common on Lustre -- has landed on disk).
    Mutually exclusive with --test_only (one skips training, the other
    continues it -- pick one).

Required:
    - bands_forestnet section in ./data/bands_info/bands.yaml
      with 6 bands: B02 Blue, B03 Green, B04 Red, B05 NIR, B06 SWIR1, B07 SWIR2
    - configs_dataset_forestnet.yaml under ./data/Tiny_BigEarthNet/
"""

import os
import time
import argparse

import torch
from collections import Counter

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    EarlyStopping,
)

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding

from training.trainer_FORESTNET import Model_ForestNet
from training.utils.datasets.utils_dataset_FORESTNET import ForestNetDataset
from training.utils.datasets.dataloaders import UnifiedDataModule

UnifiedDataModule.GROUPED_DATASET_CLASSES = (
    UnifiedDataModule.GROUPED_DATASET_CLASSES | {"ForestNetDataset"}
)


# =============================================================================
# RESUME HELPER (mirrors train_biomassters.py's wait_for_checkpoint)
# =============================================================================

def wait_for_checkpoint(path: str, wait_seconds: int, poll_interval: int = 15) -> str:
    """
    Polls for `path` to exist, up to `wait_seconds` total, checking every
    `poll_interval` seconds. Useful for chained SLURM jobs where the next
    job in the chain can start before the previous job's checkpoint write
    has actually landed on disk.

    wait_seconds=0 means "check once, don't wait" -- fails fast if the
    file isn't there.

    Raises FileNotFoundError if the checkpoint never appears within the
    timeout, rather than silently falling back to training from scratch --
    resuming should be an explicit, verified action.
    """
    if os.path.exists(path):
        return path

    if wait_seconds <= 0:
        raise FileNotFoundError(
            f"Checkpoint not found: {path} "
            f"(use --resume_wait_seconds > 0 to poll for it instead of failing immediately)"
        )

    print(f"[Train] Checkpoint not found yet: {path}")
    print(f"[Train] Waiting up to {wait_seconds}s (polling every {poll_interval}s)...")
    waited = 0
    while waited < wait_seconds:
        time.sleep(poll_interval)
        waited += poll_interval
        if os.path.exists(path):
            print(f"[Train] Checkpoint appeared after {waited}s: {path}")
            return path
        print(f"[Train]   ...still waiting ({waited}/{wait_seconds}s)")

    raise FileNotFoundError(
        f"Checkpoint still not found after waiting {wait_seconds}s: {path}"
    )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser ForestNet training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--dataset_name", type=str, default="forestnet")
parser.add_argument("--clipping",     action="store_true")
parser.add_argument("--use_class_weights", action="store_true")
parser.add_argument("--label_smoothing",   type=float, default=0.0)
parser.add_argument("--train_fraction", type=float, default=None,
                    choices=[0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00],
                    help="Limited-label train partition fraction (e.g. 0.10 for "
                         "the 10%% partition). Loads "
                         "'{fraction:.2f}x_train_partition.json' for the TRAIN "
                         "split only -- validation/test always use "
                         "default_partition.json regardless. Default (None) "
                         "uses the full default_partition.json train split. "
                         "NOTE: I haven't verified UnifiedDataModule forwards "
                         "arbitrary extra kwargs to dataset_class construction "
                         "-- see the comment at the DATA MODULE section below.")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. When set, skips training "
                         "and runs test directly on this checkpoint. "
                         "Uses a single GPU (no DDP).")

# Resume-training mode (NEW)
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint to resume TRAINING from (full "
                         "trainer state via Trainer.fit(ckpt_path=...)). If "
                         "the file doesn't exist yet, use --resume_wait_seconds "
                         "to poll for it instead of failing immediately. "
                         "Mutually exclusive with --test_only.")
parser.add_argument("--resume_wait_seconds", type=int, default=0,
                    help="How long to poll for --resume_from (or --test_only, "
                         "reused for both) to appear before giving up "
                         "(0 = check once, fail immediately if missing).")
parser.add_argument("--resume_poll_interval", type=int, default=15,
                    help="Seconds between polls while waiting for the checkpoint.")

args = parser.parse_args()

if args.test_only is not None and args.resume_from is not None:
    raise ValueError(
        "--test_only and --resume_from are mutually exclusive: --test_only "
        "skips training entirely (loads weights, runs test), --resume_from "
        "continues training from a checkpoint. Pick one."
    )

xp_name           = args.xp_name
config_model = read_yaml("./training/configs/config_test-FORESTNET.yaml")
configs_dataset   = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml        = "./data/bands_info/bands.yaml"

if os.environ.get("LOCAL_RANK", "0") == "0":
    if args.test_only:
        print(f"[Train] Test-only mode: {args.test_only}")
    else:
        print(f"[Train] Gradient clipping: {'ON (val=1.0)' if args.clipping else 'OFF'}")
        print(f"[Train] Class weights:     {'ON' if args.use_class_weights else 'OFF'}")
        print(f"[Train] Label smoothing:   {args.label_smoothing}")
        if args.resume_from:
            print(f"[Train] Resume requested:  {args.resume_from} "
                  f"(wait up to {args.resume_wait_seconds}s if not found yet)")


# =============================================================================
# LOOKUP TABLE
# =============================================================================

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset),
    read_yaml(bands_yaml),
    config_model,
)


# =============================================================================
# WANDB (skipped in test-only mode, same as BioMassters' convention --
# resuming training still logs, since training is still happening)
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and args.test_only is None:
    import wandb
    wandb.init(
        name=config_model["encoder"] + "_" + xp_name,
        project="Atomizer_ForestNet",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="Atomizer_ForestNet")
    wandb.define_metric("train_loss",     step_metric="trainer/global_step")
    wandb.define_metric("val_loss",       step_metric="trainer/global_step")
    wandb.define_metric("val_top1",       step_metric="trainer/global_step")
    wandb.define_metric("val_macro_f1",   step_metric="trainer/global_step")


# =============================================================================
# DATA MODULE
# =============================================================================

if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"[DEBUG] dataset_class.__name__ = '{ForestNetDataset.__name__}'")
    print(f"[DEBUG] match: {ForestNetDataset.__name__ in UnifiedDataModule.GROUPED_DATASET_CLASSES}")

# Limited-label train fraction: UnifiedDataModule._create_grouped_dataset
# hardcodes the exact kwargs forwarded to dataset_class(...) -- no
# passthrough for arbitrary extra args like train_fraction (confirmed:
# passing dataset_kwargs= raised TypeError, UnifiedDataModule doesn't
# accept it). config_model IS forwarded unchanged though, so
# ForestNetDataset reads train_fraction from
# config_model["dataset"]["train_fraction"] instead -- set it here,
# BEFORE constructing UnifiedDataModule.
if args.train_fraction is not None:
    config_model.setdefault("dataset", {})["train_fraction"] = args.train_fraction
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[Train] Limited-label train partition: {args.train_fraction:.2f}x "
              f"(set via config_model['dataset']['train_fraction'])")

data_module = UnifiedDataModule(
    path="./data/geo-bench-1.0/classification_v1.0/m-forestnet",
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=ForestNetDataset,
)


# =============================================================================
# CLASS WEIGHTS (skip in test-only mode)
# =============================================================================

class_weights = None
if args.use_class_weights and args.test_only is None:
    tmp_train = ForestNetDataset(
        root_path="./data/geo-bench-1.0/classification_v1.0/m-forestnet",
        mode="train",
        dataset_config=read_yaml(bands_yaml),
        config_model=config_model,
        look_up=lookup_table,
    )
    counts  = Counter(tmp_train.name_to_label[n] for n in tmp_train.sample_names)
    weights = torch.zeros(tmp_train.NUM_CLASSES, dtype=torch.float32)
    for c in range(tmp_train.NUM_CLASSES):
        weights[c] = 1.0 / max(counts.get(c, 1), 1)
    class_weights = weights / weights.sum() * tmp_train.NUM_CLASSES
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[Train] Class weights: {class_weights.tolist()}")
    del tmp_train


# =============================================================================
# MODEL
# =============================================================================

model = Model_ForestNet(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
    class_weights=class_weights,
    label_smoothing=args.label_smoothing,
)


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/forestnet/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    # Build callbacks only when training
    lr_monitor   = LearningRateMonitor(logging_interval="step")
    accumulator  = GradientAccumulationScheduler(scheduling={0: 4})

    checkpoint_val = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{config_model['encoder']}_{xp_name}-{{epoch:02d}}-{{val_macro_f1:.4f}}",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{config_model['encoder']}_{xp_name}-last",
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=int(config_model["trainer"].get("patience", 20)),
        verbose=True,
    )

    callbacks = [accumulator, checkpoint_val, checkpoint_last, early_stop, lr_monitor]

    # DDP trainer for fit
    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        devices=-1,
        max_epochs=config_model["trainer"]["epochs"],
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0 if args.clipping else None,
    )

    # Resolve --resume_from (with polling) BEFORE fit, so a chained-job
    # wait doesn't happen after the (potentially slow) data module / class
    # weights setup above has already run.
    fit_ckpt_path = None
    if args.resume_from is not None:
        fit_ckpt_path = wait_for_checkpoint(
            args.resume_from, args.resume_wait_seconds, args.resume_poll_interval)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[Train] RESUMING from: {fit_ckpt_path}")

    trainer.fit(model, datamodule=data_module, ckpt_path=fit_ckpt_path)

    # Capture best checkpoint path BEFORE destroying the process group.
    best_ckpt = checkpoint_val.best_model_path

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
    # Test-only mode — skip DDP setup entirely.
    best_ckpt = wait_for_checkpoint(
        args.test_only, args.resume_wait_seconds, args.resume_poll_interval)
    print(f"\n[test-only mode] Skipping training, testing checkpoint:")
    print(f"  {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────
# Load checkpoint manually with strict=False.
#
# Some buffers in the model are runtime caches that get rebuilt on first
# forward pass:
#   - input_processor.time_encoder.time_values  (regenerated from config)
#   - geo_pruning.cell_*_<dims>_<hash>          (recomputed lazily for each
#                                                 distinct image geometry)
# These appear in saved checkpoints because they're registered as buffers,
# but they don't need to be restored — they'll be recreated automatically.
# strict=False tells PyTorch to ignore unexpected keys silently.
#
# Lightning's test_trainer.test(ckpt_path=...) always uses strict=True,
# so we load manually and call test() without ckpt_path.
# ─────────────────────────────────────────────────────────────────────
ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
if missing:
    print(f"[load_state_dict] missing keys: {missing}")
if unexpected:
    print(f"[load_state_dict] ignored unexpected keys ({len(unexpected)}):")
    for k in unexpected[:5]:
        print(f"    {k}")
    if len(unexpected) > 5:
        print(f"    ... and {len(unexpected) - 5} more")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
# Note: NO ckpt_path here — weights already loaded above.
test_trainer.test(
    model,
    datamodule=data_module,
)


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
    wandb.finish()
