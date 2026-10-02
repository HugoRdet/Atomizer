"""
Atomiser EuroSAT Training Script (optical only)
===================================================

10-class land cover classification, S2 optical (13 bands, incl. B10
cirrus), single modality — no SAR. Uses EurosatDataset
(training/utils/datasets/utils_dataset_eurosat.py), the classification-
pattern dataset built earlier: dummy queries, scalar label,
task="classification", pickle-embedded label extraction (GeoBench
m-eurosat has no 'label' HDF5 key or label_map.json — see that file's
docstring).

Mirror of script_train_eurosat_sar.py: SAME Model_ForestNet trainer, same
"classification" task path, same val_top1-monitored checkpointing. Only
the dataset differs (13 optical bands here vs. 15 S2+S1 fused there), and
there is no bands.keep/drop modality-ablation story since there's no
second modality to ablate.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training entirely
    and run test on the provided checkpoint with a single GPU.

Required:
    - bands_senflood section in ./data/bands_info/bands.yaml — EurosatDataset
      reuses bands_senflood's B01..B12/B08A entries for wavelength/bandwidth
      (same trick as CashewSkipDataset), including B10, which Cashew's
      version dropped.
    - configs_dataset_u_regular.yaml under ./data/Tiny_BigEarthNet/
    - config_test-EUROSAT.yaml under ./training/configs/ (SAME config file
      the EuroSAT-SAR script uses — trainer.bands.keep/drop entries in it
      are simply irrelevant/unused here, since EurosatDataset has no
      modality-drop mechanism).
    - ./data/geo-bench-1.0/classification_v1.0/m-eurosat/
      (default_partition.json, band_stats.json, id_*.hdf5)
"""

import os
import argparse

import torch
from collections import Counter

import h5py
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
from training.utils.datasets.utils_dataset_Eurosat import (
    EurosatDataset,
    _PermissiveUnpickler,
)
from training.utils.datasets.dataloaders import UnifiedDataModule

# NOTE: if EurosatDataset is not yet registered in
# UnifiedDataModule.GROUPED_DATASET_CLASSES (see dataloaders.py, and how
# EuroSATSARDataset is handled there), it needs to be added the same way
# before this script will find it via dataset_class=EurosatDataset below.


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser EuroSAT training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--clipping",     action="store_true")
parser.add_argument("--use_class_weights", action="store_true")
parser.add_argument("--label_smoothing",   type=float, default=0.0)

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

parser.add_argument("--resume", action="store_true",
                    help="Resume training from the '-last' checkpoint for this "
                         "xp_name, if one exists. Restores full trainer state "
                         "(epoch, optimizer, LR schedule, EarlyStopping/"
                         "ModelCheckpoint state) — not just weights. Safe to "
                         "pass on every submission: if no checkpoint is found "
                         "yet, training just starts fresh. Ignored if "
                         "--test_only is set.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Explicit checkpoint path to resume from (overrides "
                         "--resume auto-detection).")

args = parser.parse_args()

xp_name           = args.xp_name
config_model      = read_yaml("./training/configs/config_test-EUROSAT.yaml")
configs_dataset   = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml        = "./data/bands_info/bands.yaml"
data_root         = "./data/geo-bench-1.0/classification_v1.0/m-eurosat"

if os.environ.get("LOCAL_RANK", "0") == "0":
    if args.test_only:
        print(f"[Train] Test-only mode: {args.test_only}")
    else:
        print(f"[Train] Gradient clipping: {'ON' if args.clipping else 'OFF'}")
        print(f"[Train] Class weights:     {'ON' if args.use_class_weights else 'OFF'}")
        print(f"[Train] Label smoothing:   {args.label_smoothing}")
        print(f"[Train] Resume:            "
              f"{'ON (' + (args.resume_from or 'auto-detect last ckpt') + ')' if (args.resume or args.resume_from) else 'OFF'}")
    print(f"[Train] Modality: optical only (13 S2 bands, no SAR)")


# =============================================================================
# LOOKUP TABLE
# =============================================================================

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset),
    read_yaml(bands_yaml),
    config_model,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"] + "_" + xp_name,
        project="Atomizer_EuroSAT",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="Atomizer_EuroSAT")


# =============================================================================
# DATA MODULE
# =============================================================================

data_module = UnifiedDataModule(
    path=data_root,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=EurosatDataset,
)


# =============================================================================
# CLASS WEIGHTS
# =============================================================================
#
# EurosatDataset doesn't precompute a (class, path) sample_list the way
# EuroSATSARDataset does (its labels come from folder names, cheap to
# read; EurosatDataset's labels are pickle-embedded per HDF5, see that
# file's docstring). To avoid loading every sample's full 13-band image
# just to count labels, this reads ONLY each file's label via
# EurosatDataset's own _extract_label + _PermissiveUnpickler, bypassing
# _load_sample entirely.

class_weights = None
if args.use_class_weights and args.test_only is None:
    tmp_train = EurosatDataset(
        root_path=data_root,
        mode="train",
        dataset_config=read_yaml(bands_yaml),
        config_model=config_model,
        look_up=lookup_table,
    )
    counts = Counter()
    for name in tmp_train.sample_names:
        path = os.path.join(data_root, f"{name}.hdf5")
        with h5py.File(path, "r") as f:
            counts[EurosatDataset._extract_label(f)] += 1

    weights = torch.zeros(tmp_train.NUM_CLASSES, dtype=torch.float32)
    for c in range(tmp_train.NUM_CLASSES):
        weights[c] = 1.0 / max(counts.get(c, 1), 1)
    class_weights = weights / weights.sum() * tmp_train.NUM_CLASSES
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[Train] Class counts: {dict(sorted(counts.items()))}")
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

ckpt_dir = "./checkpoints/eurosat/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    # ── Resume detection (full trainer state, not just weights) ──────────
    resume_ckpt_path = None
    if args.resume_from is not None:
        resume_ckpt_path = args.resume_from
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(
                f"--resume_from checkpoint not found: {resume_ckpt_path}"
            )
        print(f"[Resume] Using explicit checkpoint: {resume_ckpt_path}")
    elif args.resume:
        auto_path = os.path.join(ckpt_dir, f"{config_model['encoder']}_{xp_name}-last.ckpt")
        if os.path.exists(auto_path):
            resume_ckpt_path = auto_path
            print(f"[Resume] Found existing checkpoint, resuming: {auto_path}")
        else:
            print(f"[Resume] --resume set but no checkpoint found at "
                  f"{auto_path} — starting fresh.")

    lr_monitor   = LearningRateMonitor(logging_interval="step")
    accumulator  = GradientAccumulationScheduler(scheduling={0: 1})

    checkpoint_val = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{config_model['encoder']}_{xp_name}-{{epoch:02d}}-{{val_top1:.4f}}",
        monitor="val_top1",
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
        monitor="val_top1",
        mode="max",
        patience=int(config_model["trainer"].get("patience", 100)),
        verbose=True,
    )

    callbacks = [accumulator, checkpoint_val, checkpoint_last, early_stop, lr_monitor]

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

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    best_ckpt = checkpoint_val.best_model_path

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
    print(f"\n[test-only mode] Skipping training, testing: {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST (with strict=False to ignore runtime cache buffers)
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys "
          f"(runtime caches — recreated automatically)")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(model, datamodule=data_module)


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
