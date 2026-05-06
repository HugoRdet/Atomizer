"""
FLAIR-HUB Atomizer Training Script (single-task)
==================================================

Single-task land cover segmentation on FLAIR-HUB. Multi-modal multi-resolution
input via Atomizer's resolution-grouped tokenization.

Default modalities (matches the headline experiment):
    VHR (AERIAL_RGBI) + DEM + Sentinel-2 + Sentinel-1 ASC + Sentinel-1 DESC

Cross-sensor transfer experiment (run as a SECOND launch):
    Train  with --use_vhr --no_use_spot   (default)
    Test   with --no_use_vhr --use_spot   (--test_only --ckpt_path <best>)
    Same checkpoint, different test-time modality flags. No retraining.

Examples
--------
    # Default training run (VHR + S2 + S1 + DEM, 30 epochs)
    python script_train_flairhub.py --xp_name flair_v1

    # Resume from checkpoint, same wandb run
    python script_train_flairhub.py --xp_name flair_v1_resumed \
        --ckpt_path ./checkpoints/flairhub/atomiser_flairhub_v1-best.ckpt \
        --wandb_run_id <id_of_aborted_run>

    # Cross-sensor transfer test: train was VHR, evaluate on SPOT
    python script_train_flairhub.py --xp_name flair_v1_spot_test \
        --ckpt_path ./checkpoints/flairhub/atomiser_flairhub_v1-best.ckpt \
        --test_only --no_use_vhr --use_spot
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
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
from training.trainer_flairhub import Model_FlairHub
from training.utils.datasets.utils_dataset_FLAIRHUB import FlairHubDataset
from training.utils.datasets.token_grouping import collate_grouped


# =============================================================================
# RESOLUTION REGISTRATION (FLAIR-HUB-specific)
# =============================================================================
# Resolutions used by FLAIR-HUB modalities. Reference grid size 2048
# matches other datasets — TokenBuilder normalizes coordinates against this.

ALL_FLAIR_RESOLUTIONS = {
    0.2:  2048,   # VHR (AERIAL_RGBI), DEM, AERIAL_LABEL-COSIA
    1.6:  2048,   # SPOT_RGBI
    10.0: 2048,   # Sentinel-2 TS, Sentinel-1 ASC/DESC TS
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_FLAIR_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# ARGS
# =============================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="FLAIR-HUB Atomizer Training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,
                    default="config_test-FLAIR.yaml")
parser.add_argument("--dataset_name", type=str, default="u_regular",
                    help="configs_dataset_<name>.yaml under data/Tiny_BigEarthNet/")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--epochs",       type=int, default=30)
parser.add_argument("--batch_size",   type=int, default=None,
                    help="Override config's batchsize")

# Modality flags
parser.add_argument("--use_vhr",     type=str2bool, default=True)
parser.add_argument("--use_spot",    type=str2bool, default=False)
parser.add_argument("--use_dem",     type=str2bool, default=True)
parser.add_argument("--use_s2",      type=str2bool, default=True)
parser.add_argument("--use_s1",      type=str2bool, default=True)
parser.add_argument("--multi_temporal", type=int, default=6,
                    help="Number of S2/S1 timesteps to sample (linspace)")

# Loss / training options
parser.add_argument("--ignore_index", type=int, default=None,
                    help="Class to ignore in loss/metrics. None = score all 19.")

# Data root
parser.add_argument("--root_path", type=str, default="./data/FLAIR-HUB")

# Subset (for compute-tractable experiments)
parser.add_argument("--subset_indices", type=str, default=None,
                    help="Path to subset_indices.json from select_flair_subset.py. "
                         "If provided, train/val/test are filtered to the patch_ids "
                         "listed under train_patch_ids/val_patch_ids/test_patch_ids.")

# Resume / test
parser.add_argument("--ckpt_path",    type=str, default=None)
parser.add_argument("--wandb_run_id", type=str, default=None)
parser.add_argument("--test_only",    action="store_true")
args = parser.parse_args()


# =============================================================================
# CONFIG + LOOKUP
# =============================================================================
config_model         = read_yaml(f"./training/configs/{args.config_model}")
configs_dataset_path = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
configs_dataset      = read_yaml(configs_dataset_path)

# Bands from in-code factory (self-contained, no YAML for FLAIR-HUB)
bands = create_flairhub_bands_info()

lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
register_all_resolutions(lookup_table)

# S1 abstract channels — register so VV/VH spectral_idx exist in table_wave
lookup_table.register_abstract_channel("VV")
lookup_table.register_abstract_channel("VH")

# Optional but harmless: register DSM/DTM explicitly (the bands_dem_info
# entries already triggered registration via init_lookup_table_wave, but
# this ensures abstract_channel_indices is populated for downstream uses).
lookup_table.register_abstract_channel("DSM")
lookup_table.register_abstract_channel("DTM")

# ── Resolve batch size ───────────────────────────────────────────────
if args.batch_size is not None:
    batch_size = args.batch_size
else:
    batch_size = int(config_model["trainer"].get(
        "batchsize", config_model.get("dataset", {}).get("batchsize", 4)))

# ── Override epochs in the config (cosine schedule reads from this) ──
config_model["trainer"]["epochs"] = args.epochs


print(f"\n{'='*70}")
print(f"  FLAIR-HUB Atomizer — Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Modalities:  VHR={args.use_vhr}  SPOT={args.use_spot}  "
      f"DEM={args.use_dem}  S2={args.use_s2}  S1={args.use_s1}")
print(f"  Temporal:    {args.multi_temporal} timesteps (linspace)")
print(f"  Batch size:  {batch_size}")
print(f"  Epochs:      {args.epochs}")
print(f"  Config:      {args.config_model}")
print(f"  Lookup tbl:  {len(lookup_table.table_wave)} spectral entries")
print(f"  Ignore idx:  {args.ignore_index}")


# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb

    run_name = f"AtomizerFLAIR_{args.xp_name}"
    wandb_init_kwargs = dict(
        name=run_name,
        project="Atomizer-FLAIR",
        config={
            **config_model,
            "modalities": {
                "vhr": args.use_vhr, "spot": args.use_spot,
                "dem": args.use_dem, "s2": args.use_s2, "s1": args.use_s1,
            },
            "multi_temporal": args.multi_temporal,
            "batch_size":     batch_size,
            "epochs":         args.epochs,
            "ignore_index":   args.ignore_index,
        },
    )
    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]     = args.wandb_run_id
        wandb_init_kwargs["resume"] = "must"
        print(f"  W&B:         resuming run {args.wandb_run_id}")
    else:
        print(f"  W&B:         new run {run_name}")
    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="Atomizer-FLAIR")


# =============================================================================
# DATASETS + DATALOADERS
# =============================================================================

def build_dataset(mode: str):
    """Build a FLAIR-HUB dataset for the given split."""
    return FlairHubDataset(
        root_path=args.root_path,
        mode=mode,
        dataset_config=bands,                  # in-code dict (self-contained)
        config_model=config_model,
        look_up=lookup_table,
        use_vhr=args.use_vhr,
        use_spot=args.use_spot,
        use_dem=args.use_dem,
        use_s2=args.use_s2,
        use_s1=args.use_s1,
        multi_temporal=args.multi_temporal,
    )


def make_loader(dataset, shuffle: bool):
    """
    DataLoader with manual DistributedSampler when DDP is initialized.
    Matches the convention used in MT and PASTIS launch scripts.
    """
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


print(f"\n[FLAIR-HUB] Building datasets...")
train_ds = build_dataset("train")
val_ds   = build_dataset("validation")
test_ds  = build_dataset("test")
print(f"[FLAIR-HUB] Pre-subset sizes: "
      f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

# ── Subset filtering ────────────────────────────────────────────────
# If a subset_indices.json was provided, filter each split's patch_rows
# to only those patches in the subset. Filtering happens in-place on the
# dataset's patch_rows list (mutates the dataset; cheaper than wrapping
# in torch.utils.data.Subset because we keep the dataset's own __len__
# and __getitem__ behavior intact).
if args.subset_indices is not None:
    import json as _json
    print(f"\n[FLAIR-HUB] Loading subset indices from: {args.subset_indices}")
    with open(args.subset_indices) as f:
        subset = _json.load(f)

    split_key = {
        "train":      "train_patch_ids",
        "validation": "val_patch_ids",
        "test":       "test_patch_ids",
    }
    for ds, name in [(train_ds, "train"),
                     (val_ds,   "validation"),
                     (test_ds,  "test")]:
        wanted = set(subset.get(split_key[name], []))
        if not wanted:
            print(f"[FLAIR-HUB] No subset for '{name}', keeping full split.")
            continue
        before = len(ds.patch_rows)
        ds.patch_rows = [r for r in ds.patch_rows
                         if r["patch_id"] in wanted]
        after = len(ds.patch_rows)
        # Sanity check: warn if the JSON listed patch_ids not present in CSV
        missing = len(wanted) - after
        print(f"[FLAIR-HUB]   {name}: {before:>6} → {after:>6} "
              f"({missing} JSON ids not found in CSV)")

print(f"[FLAIR-HUB] Final sizes: "
      f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")


# =============================================================================
# DataModule (light wrapper around the loaders)
# =============================================================================

class FlairHubDataModule(pl.LightningDataModule):
    """Lightning wrapper around FLAIR-HUB DataLoaders."""

    def setup(self, stage=None):
        # Datasets already built above — nothing to do here.
        pass

    def train_dataloader(self):
        return make_loader(train_ds, shuffle=True)

    def val_dataloader(self):
        return make_loader(val_ds, shuffle=False)

    def test_dataloader(self):
        return make_loader(test_ds, shuffle=False)


data_module = FlairHubDataModule()


# =============================================================================
# MODEL
# =============================================================================
model = Model_FlairHub(
    config=config_model,
    wand=wandb_logger is not None,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
    ignore_index=args.ignore_index,
)


# =============================================================================
# CALLBACKS + TRAINER
# =============================================================================
ckpt_dir = "./checkpoints/flairhub/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_flairhub_{args.xp_name}-{{epoch:02d}}-"
                 f"{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_flairhub_{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,         # we set DistributedSampler manually
    devices=-1,
    max_epochs=args.epochs,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
)


# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  FLAIR-HUB — TEST ONLY\n  ckpt: {args.ckpt_path}\n"
          f"{'='*70}\n")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[FLAIR-HUB] missing keys: {len(result.missing_keys)}, "
          f"unexpected keys: {len(result.unexpected_keys)}")

    trainer.test(model, datamodule=data_module, verbose=True)

else:
    print(f"\n{'='*70}\n  FLAIR-HUB — TRAINING\n{'='*70}\n")
    if args.ckpt_path is not None:
        print(f"  Resuming from: {args.ckpt_path}")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

    print(f"\n{'='*70}\n  FLAIR-HUB — FINAL TEST\n{'='*70}\n")
    trainer.test(model, datamodule=data_module, verbose=True, ckpt_path="best")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/atomiser_flairhub_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)
    print(f"WANDB_RUN_ID: {wandb.run.id}")