"""
PureForest Atomizer Training Script (single-task classification)
=================================================================

13-class pure forest tree species classification.
Input modalities (both at 0.2m resolution group):
    - RGB+NIR ortho: 4 ch NIR-R-G-B, 250×250, uint8
    - LiDAR points:  ~100k–180k pts/patch, sparse (x, y, z_norm)

Both modalities tokenized at 0.2m and merged into a single group.
The encoder learns the fusion via cross-attention — no hand-crafted
projection of RGB onto LiDAR (unlike the RandLA-Net baseline).

Optional cross-task transfer:
    --ckpt_path <flairhub_or_fractal_checkpoint.ckpt>
    Loads a pretrained encoder. strict_loading=False so mismatched
    heads (19-class FLAIR-HUB, 7-class FRACTAL) don't block loading.

Auto-resume:
    If --ckpt_path is NOT provided (and --test_only is not set), the
    script looks for the most recent "last" checkpoint for this
    --xp_name and resumes from it. Pass --no_auto_resume to force a
    fresh start.

Examples
--------
    # From scratch (auto-resumes if a prior run exists)
    python script_train_pureforest.py --xp_name pureforest_v1

    # Force fresh start
    python script_train_pureforest.py --xp_name pureforest_v1 --no_auto_resume

    # Transfer from FLAIR-HUB or FRACTAL encoder
    python script_train_pureforest.py --xp_name pureforest_v1_xfer \\
        --ckpt_path ./checkpoints/fractal/atomiser_fractal_v1-best.ckpt

    # LiDAR-only ablation
    python script_train_pureforest.py --xp_name pureforest_lidar_only \\
        --modality lidar

    # RGB-only ablation
    python script_train_pureforest.py --xp_name pureforest_rgb_only \\
        --modality rgb

    # Test-only evaluation
    python script_train_pureforest.py --xp_name pureforest_v1_test \\
        --ckpt_path ./checkpoints/pureforest/atomiser_pureforest_v1-best.ckpt \\
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

# Project — config + lookup + bands factory
from training.utils import read_yaml, Lookup_encoding, create_flairhub_bands_info
from training.utils.datasets.token_builder import TokenBuilder

# Trainer + dataset
from training.trainer_PUREFOREST import Model_PureForest
from training.utils.datasets.utils_dataset_pureforest import PureForestDataset
from training.utils.datasets.token_grouping import collate_grouped


# =============================================================================
# RESOLUTION REGISTRATION
# =============================================================================
# PureForest uses a single resolution (0.2m) for both ortho and LiDAR,
# identical to FRACTAL. Using the same reference size means a FRACTAL or
# FLAIR-HUB checkpoint transfers cleanly into this encoder.

ALL_PUREFOREST_RESOLUTIONS = {
    0.2: 2048,   # RGB+NIR ortho + LiDAR points
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_PUREFOREST_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# BAND REGISTRATION
# =============================================================================
# PureForest ortho band order: NIR, R, G, B  (verified from rasterio stats —
# 4 bands, uint8, 250×250, 0.2m GSD).
#
# We intentionally reuse the same (bandwidth, central_wavelength) values as
# FRACTAL-IRGB so that:
#   (a) spectral_idx assignments are identical to FRACTAL,
#   (b) a FRACTAL-pretrained encoder transfers without any spectral remapping.

def create_pureforest_bands_info():
    return {
        "bands_pureforest_irgb_info": {
            "NIR": {"bandwidth": 100, "central_wavelength": 833, "idx": 0},
            "R":   {"bandwidth":  90, "central_wavelength": 660, "idx": 1},
            "G":   {"bandwidth":  80, "central_wavelength": 559, "idx": 2},
            "B":   {"bandwidth":  80, "central_wavelength": 492, "idx": 3},
        },
    }


# =============================================================================
# ARGS
# =============================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="PureForest Atomizer Training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,
                    default="config_test-pureforest.yaml",
                    help="Atomizer config YAML. Reuse the FRACTAL config "
                         "family for architecture + encoder compatibility.")
parser.add_argument("--dataset_name", type=str, default="u_regular")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--epochs",       type=int, default=100)
parser.add_argument("--batch_size",   type=int, default=None,
                    help="Override config batchsize.")

# Dataset args
parser.add_argument("--root_path",        type=str,
                    default="./data/PureForest",
                    help="PureForest root directory "
                         "(contains data/ and metadata/).")
parser.add_argument("--max_lidar_points", type=int, default=16_000,
                    help="Max LiDAR points per patch. Also the padding "
                         "target for batching.")
parser.add_argument("--modality",         type=str, default="both",
                    choices=["both", "rgb", "lidar"],
                    help="Which modalities to use. 'both' = RGB+NIR ortho "
                         "+ LiDAR (default). 'rgb' / 'lidar' for ablations.")

# Loss / training options
parser.add_argument("--class_weighting", type=str, default="auto",
                    choices=["auto", "none"],
                    help="'auto' = inverse-frequency weights clipped at 20. "
                         "'none' = unweighted CE (matches RandLA-Net baseline).")

# Resume / test
parser.add_argument("--ckpt_path",     type=str, default=None,
                    help="Path to a checkpoint for init, resume, or test. "
                         "If unset and not --test_only, auto-resume kicks in.")
parser.add_argument("--no_auto_resume", action="store_true",
                    help="Disable auto-resume from the latest 'last' "
                         "checkpoint when --ckpt_path is unset.")
parser.add_argument("--wandb_run_id",  type=str, default=None)
parser.add_argument("--test_only",     action="store_true")
args = parser.parse_args()


# =============================================================================
# CONFIG + LOOKUP
# =============================================================================
config_model         = read_yaml(f"./training/configs/{args.config_model}")
configs_dataset_path = (f"./data/Tiny_BigEarthNet/"
                        f"configs_dataset_{args.dataset_name}.yaml")
configs_dataset      = read_yaml(configs_dataset_path)

pureforest_bands = create_pureforest_bands_info()
flair_bands      = create_flairhub_bands_info()
bands            = {**flair_bands, **pureforest_bands}

lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
register_all_resolutions(lookup_table)

# Abstract channels shared with FRACTAL — register the same set so a
# FRACTAL checkpoint loads without key mismatches.
lookup_table.register_abstract_channel("ELEVATION")
lookup_table.register_abstract_channel("VV")
lookup_table.register_abstract_channel("VH")
lookup_table.register_abstract_channel("DSM")
lookup_table.register_abstract_channel("DTM")

# ── Resolve batch size ────────────────────────────────────────────────
if args.batch_size is not None:
    batch_size = args.batch_size
else:
    batch_size = int(config_model["trainer"].get(
        "batchsize", config_model.get("dataset", {}).get("batchsize", 4)))

# ── Override epochs in config (cosine schedule reads from this) ───────
config_model["trainer"]["epochs"]      = args.epochs
config_model["trainer"]["num_classes"] = 13


# =============================================================================
# CHECKPOINT DIR + AUTO-RESUME LOOKUP
# =============================================================================
ckpt_dir = "./checkpoints/pureforest/"
os.makedirs(ckpt_dir, exist_ok=True)


def _find_latest_last_checkpoint(xp_name: str) -> str:
    """
    Find the most recent 'last' checkpoint for the given xp_name.
    Sort by epoch number parsed from filename (robust on Lustre/shared fs).
    Returns path or None.
    """
    pattern = os.path.join(
        ckpt_dir, f"atomiser_pureforest_{xp_name}-last-*.ckpt"
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
        print(f"\n[PureForest] Auto-resume: found {auto_resume_ckpt}")
    else:
        print(f"\n[PureForest] Auto-resume: no prior checkpoint for "
              f"xp_name='{args.xp_name}' — starting fresh")


print(f"\n{'='*70}")
print(f"  PureForest Atomizer — Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Modality:        {args.modality}")
print(f"  Max LiDAR pts:   {args.max_lidar_points}")
print(f"  Batch size:      {batch_size}")
print(f"  Epochs:          {args.epochs}")
print(f"  Config:          {args.config_model}")
print(f"  Lookup tbl:      {len(lookup_table.table_wave)} spectral entries")
print(f"  Class weights:   {args.class_weighting}")
if args.ckpt_path is not None:
    if args.test_only:
        print(f"  Mode:            TEST ONLY (ckpt: {args.ckpt_path})")
    else:
        print(f"  Init from ckpt:  {args.ckpt_path}")
elif auto_resume_ckpt is not None:
    print(f"  Auto-resume:     {auto_resume_ckpt}")


# =============================================================================
# WANDB
# =============================================================================
wandb_resume_id = args.wandb_run_id
if wandb_resume_id is None and auto_resume_ckpt is not None:
    run_id_path = (f"training/wandb_runs/"
                   f"atomiser_pureforest_{args.xp_name}.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            wandb_resume_id = f.read().strip()
        print(f"[PureForest] Resuming wandb run id={wandb_resume_id}")

wandb_logger = WandbLogger(
    project="Atomizer-PureForest",
    name=f"AtomizerPureForest_{args.xp_name}",
    save_dir=os.environ.get("WANDB_DIR", "./wandb"),
    config={
        **config_model,
        "modality":         args.modality,
        "max_lidar_points": args.max_lidar_points,
        "batch_size":       batch_size,
        "epochs":           args.epochs,
        "class_weighting":  args.class_weighting,
        "init_ckpt":        args.ckpt_path,
        "auto_resume_ckpt": auto_resume_ckpt,
    },
    id=wandb_resume_id,
    resume="must" if wandb_resume_id is not None else None,
)


# =============================================================================
# DATASETS + DATALOADERS
# =============================================================================

def build_dataset(mode: str) -> PureForestDataset:
    return PureForestDataset(
        root_path=args.root_path,
        mode=mode,
        dataset_config=bands,
        config_model=config_model,
        look_up=lookup_table,
        max_lidar_points=args.max_lidar_points,
        modality=args.modality,
        use_augmentation=(mode == "train"),
    )


def make_loader(dataset, shuffle: bool) -> DataLoader:
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_grouped,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=shuffle,
    )


print(f"\n[PureForest] Building datasets...")
train_ds = build_dataset("train")
val_ds   = build_dataset("val")
test_ds  = build_dataset("test")
print(f"[PureForest] Sizes: "
      f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")


# =============================================================================
# DataModule
# =============================================================================

class PureForestDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return make_loader(train_ds, shuffle=True)

    def val_dataloader(self):
        return make_loader(val_ds, shuffle=False)

    def test_dataloader(self):
        return make_loader(test_ds, shuffle=False)


data_module = PureForestDataModule()


# =============================================================================
# MODEL
# =============================================================================
class_weights_arg = "auto" if args.class_weighting == "auto" else None

model = Model_PureForest(
    config=config_model,
    wand=wandb_logger is not None,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
    class_weights=class_weights_arg,
    label_smoothing=0.0,
)


# =============================================================================
# CALLBACKS + TRAINER
# =============================================================================
callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=(f"atomiser_pureforest_{args.xp_name}"
                  f"-{{epoch:02d}}-{{val_top1:.4f}}"),
        monitor="val_top1",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=(f"atomiser_pureforest_{args.xp_name}"
                  f"-last-{{epoch:02d}}"),
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
    accumulate_grad_batches=1,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================
# Three paths (identical logic to FRACTAL script):
#
#   1. --ckpt_path with "flair" or "fractal" in name
#        => Init mode: load weights only, fresh optimizer.
#           Both transfer naturally because the encoder architecture and
#           spectral/resolution indices are identical (same 0.2m group,
#           same NIR-R-G-B band registrations).
#
#   2. --ckpt_path without those tags
#        => Resume mode: pass to trainer.fit() for full state restore.
#
#   3. No --ckpt_path, not --test_only
#        => Auto-resume from latest "last" checkpoint if one exists.

def _is_pretrained_checkpoint(path: str) -> bool:
    base = os.path.basename(path).lower()
    return any(tag in base for tag in ("flairhub", "flair_", "fractal"))


resume_ckpt_path = None

if args.ckpt_path is not None and not args.test_only:
    if _is_pretrained_checkpoint(args.ckpt_path):
        print(f"\n[PureForest] Loading pretrained encoder from "
              f"{args.ckpt_path}")
        ckpt   = torch.load(args.ckpt_path, map_location="cpu",
                            weights_only=False)
        state  = ckpt.get("state_dict", ckpt)
        result = model.load_state_dict(state, strict=False)
        print(f"[PureForest]   missing keys   : {len(result.missing_keys)} "
              f"(expected: classification head reinit)")
        print(f"[PureForest]   unexpected keys: {len(result.unexpected_keys)}")
    else:
        resume_ckpt_path = args.ckpt_path
        print(f"\n[PureForest] Resuming from {args.ckpt_path}")

elif auto_resume_ckpt is not None and not args.test_only:
    resume_ckpt_path = auto_resume_ckpt
    print(f"\n[PureForest] Auto-resuming from {resume_ckpt_path}")


# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  PureForest — TEST ONLY\n"
          f"  ckpt: {args.ckpt_path}\n{'='*70}\n")

    ckpt   = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state  = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[PureForest] missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")

    trainer.test(model, datamodule=data_module, verbose=True)

else:
    print(f"\n{'='*70}\n  PureForest — TRAINING\n{'='*70}\n")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    print(f"\n{'='*70}\n  PureForest — FINAL TEST\n{'='*70}\n")
    # Resolve best checkpoint explicitly from the monitored callback (callbacks[0])
    # to avoid Lightning's ambiguity error when two ModelCheckpoint callbacks
    # are registered — it picks the first one (the 'last' callback, no monitor).
    best_ckpt_path = callbacks[0].best_model_path
    if not best_ckpt_path:
        print("[PureForest] WARNING: no best checkpoint found — testing with current weights.")
        best_ckpt_path = None
    else:
        print(f"[PureForest] Testing with best checkpoint: {best_ckpt_path}")
    trainer.test(model, datamodule=data_module, verbose=True, ckpt_path=best_ckpt_path)


# =============================================================================
# SAVE WANDB RUN ID  (for auto-resume continuity)
# =============================================================================
if wandb_logger is not None and trainer.is_global_zero:
    import wandb
    run = getattr(wandb, "run", None)
    if run is not None:
        os.makedirs("training/wandb_runs", exist_ok=True)
        run_id_path = (f"training/wandb_runs/"
                       f"atomiser_pureforest_{args.xp_name}.txt")
        with open(run_id_path, "w") as f:
            f.write(run.id)
        print(f"WANDB_RUN_ID: {run.id}")
