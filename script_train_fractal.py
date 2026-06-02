"""
FRACTAL Atomizer Training Script (single-task)
================================================

Single-task LIDAR + VHR semantic segmentation on FRACTAL. 7-class point cloud
segmentation: other, ground, vegetation, building, water, bridge, permanent.

Input modalities (both at 0.2m resolution group):
    - VHR ortho (FRACTAL-IRGB): 4 ch NIR-R-G-B, 250x250, uint8
    - LIDAR points:             ~80k pts/patch, sparse (x, y, elevation)

Both modalities tokenized at 0.2m. They share latent placement; cross-attention
learns the fusion. No colorization preprocessing (RandLa-Net baseline projects
VHR onto LIDAR points before training; we skip this and let the model learn
the fusion).

The architecture is identical to FLAIR-HUB Atomizer. Only the dataset, trainer,
and class count change.

Optional cross-task transfer:
    --ckpt_path <flairhub_checkpoint.ckpt>
    Loads FLAIR-HUB-trained encoder. strict_loading=False so the 19-class
    head doesn't fail to load (re-initialized for 7 classes).

Examples
--------
    # From scratch on FRACTAL
    python script_train_fractal.py --xp_name fractal_v1

    # Pretrain transfer from FLAIR-HUB
    python script_train_fractal.py --xp_name fractal_v1_xfer \\
        --ckpt_path ./checkpoints/flairhub/atomiser_flairhub_v1-best.ckpt

    # Test-only evaluation
    python script_train_fractal.py --xp_name fractal_v1_test \\
        --ckpt_path ./checkpoints/fractal/atomiser_fractal_v1-best.ckpt \\
        --test_only
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
from training.trainer_FRACTAL import Model_Fractal
from training.utils.datasets.utils_dataset_fractal import FractalDataset
from training.utils.datasets.token_grouping import collate_grouped


# =============================================================================
# RESOLUTION REGISTRATION (FRACTAL-specific)
# =============================================================================
# FRACTAL uses a single resolution (0.2m) for both VHR and LIDAR. The shared
# resolution_idx is what allows the encoder to mix them via cross-attention
# in a single group.

ALL_FRACTAL_RESOLUTIONS = {
    0.2: 2048,   # VHR ortho + LIDAR points
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_FRACTAL_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# FRACTAL BANDS REGISTRATION
# =============================================================================
# FRACTAL-IRGB ortho band order is NIR, R, G, B (verified via rasterio
# src.descriptions = ('Infrared', 'Red', 'Green', 'Blue')).
#
# Use the SAME (bandwidth, central_wavelength) values as FLAIR-HUB aerial
# bands. This way, the spectral_idx for FRACTAL VHR comes out identical to
# FLAIR-HUB's aerial bands -> encoder weights transfer naturally if you
# load a FLAIR-HUB checkpoint via --ckpt_path.
#
# Wavelengths/bandwidths from typical IGN ortho specs (matches FLAIR-HUB
# aerial bands convention; adjust if your bands.yaml uses different values).

def create_fractal_bands_info():
    """
    Build the bands_info dict that FractalDataset reads.

    Returns:
        dict containing 'bands_fractal_irgb_info' with NIR/R/G/B band specs.
    """
    return {
        "bands_fractal_irgb_info": {
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


parser = argparse.ArgumentParser(description="FRACTAL Atomizer Training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,
                    default="config_test-FRACTAL.yaml",
                    help="Atomizer config YAML. Use the same config family "
                         "as FLAIR-HUB for encoder-architecture reuse.")
parser.add_argument("--dataset_name", type=str, default="u_regular",
                    help="configs_dataset_<name>.yaml under data/Tiny_BigEarthNet/")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--epochs",       type=int, default=30)
parser.add_argument("--batch_size",   type=int, default=None,
                    help="Override config's batchsize")

# FRACTAL-specific dataset args
parser.add_argument("--root_path",          type=str, default="./data",
                    help="Parent dir containing FRACTAL/ and FRACTAL-IRGB/.")
parser.add_argument("--max_lidar_points",   type=int, default=16_000,
                    help="Max LIDAR points per patch (subsampled if exceeded). "
                         "Also the padding target for batching. Default 16k "
                         "matches Myria3D's training convention.")
parser.add_argument("--valid_patches_file", type=str, default=None,
                    help="Optional precomputed JSON listing valid patch IDs "
                         "per split (filters out patches with <1000 points). "
                         "If None, the dataset filters in-loop via __getitem__ "
                         "recursion (less efficient but works).")

# Loss / training options
parser.add_argument("--ignore_index", type=int, default=255,
                    help="Class to ignore. Default 255 matches FractalDataset's "
                         "padding label. Set to None to score all positions.")
parser.add_argument("--class_weighting", type=str, default="auto",
                    choices=["auto", "none"],
                    help="'auto' = inverse-freq clipped at 50 (default; "
                         "severely imbalanced dataset). 'none' = unweighted CE.")

# Resume / test
parser.add_argument("--ckpt_path",    type=str, default=None,
                    help="Path to a checkpoint. Two use cases: "
                         "(1) resume FRACTAL training from a previous run, or "
                         "(2) initialize from a FLAIR-HUB checkpoint for "
                         "cross-task transfer (strict_loading=False handles "
                         "the 19->7 class head mismatch).")
parser.add_argument("--wandb_run_id", type=str, default=None)
parser.add_argument("--test_only",    action="store_true")
args = parser.parse_args()


# =============================================================================
# CONFIG + LOOKUP
# =============================================================================
config_model         = read_yaml(f"./training/configs/{args.config_model}")
configs_dataset_path = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
configs_dataset      = read_yaml(configs_dataset_path)

# Bands: in-code factory for FRACTAL (NIR/R/G/B at 0.2m).
# We also load FLAIR-HUB's bands_info for compatibility — if the encoder
# was pretrained on FLAIR-HUB, having the same spectral indices registered
# in the lookup table means the encoder's spectral embeddings transfer
# directly.
fractal_bands = create_fractal_bands_info()
flair_bands   = create_flairhub_bands_info()
# Merge: FLAIR-HUB bands first (more entries -> more table slots reserved
# for compatibility), then FRACTAL VHR. Both should resolve to the same
# spectral_idx for the NIR/R/G/B bands if wavelengths match.
bands = {**flair_bands, **fractal_bands}

lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
register_all_resolutions(lookup_table)

# Register the LIDAR ELEV abstract channel. FractalDataset will look up
# spectral_idx for key (-3, -3) -- this registers it.
# (S1 VV/VH use -1/-2; ELEV uses -3 to avoid collision.)
lookup_table.register_abstract_channel("ELEVATION")

# Also register VV/VH/DSM/DTM for FLAIR-HUB checkpoint compatibility.
# (If the FLAIR-HUB encoder was trained with these in its spectral table,
# we need them in our lookup too so the encoder's spectral embeddings
# don't collide.)
lookup_table.register_abstract_channel("VV")
lookup_table.register_abstract_channel("VH")
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
print(f"  FRACTAL Atomizer — Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Modalities:      VHR ortho + LIDAR points (both @ 0.2m)")
print(f"  Max LIDAR pts:   {args.max_lidar_points}")
print(f"  Batch size:      {batch_size}")
print(f"  Epochs:          {args.epochs}")
print(f"  Config:          {args.config_model}")
print(f"  Lookup tbl:      {len(lookup_table.table_wave)} spectral entries")
print(f"  Ignore idx:      {args.ignore_index}")
print(f"  Class weights:   {args.class_weighting}")
if args.ckpt_path is not None:
    if args.test_only:
        print(f"  Mode:            TEST ONLY (ckpt: {args.ckpt_path})")
    else:
        print(f"  Init from ckpt:  {args.ckpt_path}")
        print(f"                   (strict_loading=False; class head reinit if "
              f"size mismatch)")


# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb

    run_name = f"AtomizerFRACTAL_{args.xp_name}"
    wandb_init_kwargs = dict(
        name=run_name,
        project="Atomizer-FRACTAL",
        config={
            **config_model,
            "max_lidar_points": args.max_lidar_points,
            "batch_size":       batch_size,
            "epochs":           args.epochs,
            "ignore_index":     args.ignore_index,
            "class_weighting":  args.class_weighting,
            "init_ckpt":        args.ckpt_path,
        },
    )
    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]     = args.wandb_run_id
        wandb_init_kwargs["resume"] = "must"
        print(f"  W&B:             resuming run {args.wandb_run_id}")
    else:
        print(f"  W&B:             new run {run_name}")
    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="Atomizer-FRACTAL")


# =============================================================================
# DATASETS + DATALOADERS
# =============================================================================

def build_dataset(mode: str):
    """Build a FRACTAL dataset for the given split."""
    return FractalDataset(
        root_path=args.root_path,
        mode=mode,
        dataset_config=bands,
        config_model=config_model,
        look_up=lookup_table,
        max_lidar_points=args.max_lidar_points,
        valid_patches_file=args.valid_patches_file,
    )


def make_loader(dataset, shuffle: bool):
    """DataLoader with manual DistributedSampler when DDP is initialized."""
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


print(f"\n[FRACTAL] Building datasets...")
train_ds = build_dataset("train")
val_ds   = build_dataset("val")
test_ds  = build_dataset("test")
print(f"[FRACTAL] Sizes: "
      f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")


# =============================================================================
# DataModule
# =============================================================================

class FractalDataModule(pl.LightningDataModule):
    """Lightning wrapper around FRACTAL DataLoaders."""

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return make_loader(train_ds, shuffle=True)

    def val_dataloader(self):
        return make_loader(val_ds, shuffle=False)

    def test_dataloader(self):
        return make_loader(test_ds, shuffle=False)


data_module = FractalDataModule()


# =============================================================================
# MODEL
# =============================================================================
# class_weights: "auto" or None depending on flag
class_weights_arg = "auto" if args.class_weighting == "auto" else None

model = Model_Fractal(
    config=config_model,
    wand=wandb_logger is not None,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
    ignore_index=args.ignore_index,
    class_weights=class_weights_arg,
)


# =============================================================================
# CALLBACKS + TRAINER
# =============================================================================
ckpt_dir = "./checkpoints/fractal/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_fractal_{args.xp_name}-{{epoch:02d}}-"
                 f"{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_fractal_{args.xp_name}-last-{{epoch:02d}}",
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
    precision="bf16-mixed",
    logger=wandb_logger,
    accumulate_grad_batches=1,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
)


# =============================================================================
# CHECKPOINT LOADING (init from FLAIR-HUB or resume FRACTAL)
# =============================================================================
# Two cases for --ckpt_path:
#   1. Resume FRACTAL run: pass to trainer.fit(ckpt_path=...) — Lightning
#      handles optimizer/scheduler/epoch restoration.
#   2. Init from FLAIR-HUB checkpoint: load weights manually with
#      strict=False to ignore the 19-class head, let Lightning start
#      a fresh optimizer/scheduler.
#
# We auto-detect by checking the filename: if it contains "flairhub" or
# "flair" we treat it as init-only; otherwise as resume. You can also
# explicitly set behavior by passing the appropriate ckpt path style.

def _is_flairhub_checkpoint(path: str) -> bool:
    return any(tag in os.path.basename(path).lower()
               for tag in ("flairhub", "flair_"))


resume_ckpt_path = None       # passed to trainer.fit() for full resume
if args.ckpt_path is not None and not args.test_only:
    if _is_flairhub_checkpoint(args.ckpt_path):
        # Init mode: load weights only, fresh optimizer/scheduler.
        print(f"\n[FRACTAL] Loading FLAIR-HUB init from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu",
                          weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        result = model.load_state_dict(state, strict=False)
        print(f"[FRACTAL]   missing keys: {len(result.missing_keys)} "
              f"(expected: class head reinit)")
        print(f"[FRACTAL]   unexpected keys: {len(result.unexpected_keys)}")
    else:
        # Resume mode: pass to trainer.fit().
        resume_ckpt_path = args.ckpt_path
        print(f"\n[FRACTAL] Resuming from {args.ckpt_path}")


# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  FRACTAL — TEST ONLY\n  ckpt: {args.ckpt_path}\n"
          f"{'='*70}\n")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[FRACTAL] missing keys: {len(result.missing_keys)}, "
          f"unexpected keys: {len(result.unexpected_keys)}")

    trainer.test(model, datamodule=data_module, verbose=True)

else:
    print(f"\n{'='*70}\n  FRACTAL — TRAINING\n{'='*70}\n")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    print(f"\n{'='*70}\n  FRACTAL — FINAL TEST\n{'='*70}\n")
    trainer.test(model, datamodule=data_module, verbose=True, ckpt_path="best")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/atomiser_fractal_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)
    print(f"WANDB_RUN_ID: {wandb.run.id}")
