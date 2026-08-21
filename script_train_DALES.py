"""
DALES Atomizer Training Script (single-task)
================================================

Single-task LIDAR-only semantic segmentation on DALES. 8-class point cloud
segmentation: ground, vegetation, cars, trucks, power_lines, fences, poles,
buildings.

Input modality (single resolution group):
    - LIDAR points, adaptively tiled (KD-tree, ~256k pts/patch max), sparse
      (x, y, elevation), with intensity riding in column 6 of each token
      (see DalesTokenProcessor / build_sparse_tokens' intensity_override).

Unlike FRACTAL, there is NO VHR/ortho modality — groups has a single
resolution key (PIXEL_RESOLUTION = 0.2) holding LIDAR-only tokens.

REQUIRES: Atomiser_Dales (see trainer_DALES.py's import note) — a model
class analogous to Atomiser_Fractal but using DalesTokenProcessor.

Auto-resume: same mechanism as the FRACTAL script — reruns with the same
--xp_name pick up the latest "last" checkpoint unless --no_auto_resume.

Examples
--------
    # From scratch on DALES (will auto-resume if a prior run exists)
    python script_train_dales.py --xp_name dales_v1

    # Force fresh start
    python script_train_dales.py --xp_name dales_v1 --no_auto_resume

    # Test-only evaluation
    python script_train_dales.py --xp_name dales_v1_test \\
        --ckpt_path ./checkpoints/dales/atomiser_dales_v1-best.ckpt \\
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

# Project — config + lookup
from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder

# Trainer + dataset
from training.trainer_DALES import Model_Dales
from training.utils.datasets.utils_dataset_dales import DalesDataset
from training.utils.datasets.token_grouping import collate_grouped


# =============================================================================
# RESOLUTION REGISTRATION (DALES-specific)
# =============================================================================
# DALES uses a single resolution (0.2m) for its LIDAR-only pixel-equivalent
# frame. No VHR modality, so there's only one entry here (vs FRACTAL's one
# shared entry for both VHR + LIDAR).

ALL_DALES_RESOLUTIONS = {
    0.2: 2048,   # LIDAR points, pixel-equivalent frame
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_DALES_RESOLUTIONS.items():
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


parser = argparse.ArgumentParser(description="DALES Atomizer Training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,
                    default="config_test-DALES.yaml",
                    help="Atomizer config YAML.")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--epochs",       type=int, default=100)
parser.add_argument("--batch_size",   type=int, default=None,
                    help="Override config's batchsize")

# DALES-specific dataset args
parser.add_argument("--root_path",          type=str, default="./data",
                    help="Parent dir containing DALES_tiled/{train,val,test}.")
parser.add_argument("--max_lidar_points",   type=int, default=256_000,
                    help="Max LIDAR points per patch (subsampled if exceeded). "
                         "Also the padding target for batching.")
parser.add_argument("--valid_patches_file", type=str, default=None,
                    help="Optional precomputed JSON listing valid patch IDs "
                         "per split.")

# Loss / training options
parser.add_argument("--ignore_index", type=int, default=255,
                    help="Class to ignore. Default 255 matches DalesDataset's "
                         "padding label.")
parser.add_argument("--class_weighting", type=str, default="auto",
                    choices=["auto", "none"])

# Resume / test
parser.add_argument("--ckpt_path",    type=str, default=None,
                    help="Path to a checkpoint for resume or --test_only. "
                         "If unset and --test_only is unset, auto-resumes "
                         "from the latest 'last' checkpoint matching "
                         "--xp_name (if one exists).")
parser.add_argument("--no_auto_resume", action="store_true")
parser.add_argument("--wandb_run_id", type=str, default=None)
parser.add_argument("--test_only",    action="store_true")
args = parser.parse_args()


# =============================================================================
# CONFIG + LOOKUP
# =============================================================================
config_model = read_yaml(f"./training/configs/{args.config_model}")

# NOTE: DALES has no VHR bands, so unlike FRACTAL's `bands` dict (which
# merges FLAIR-HUB + FRACTAL band info), Lookup_encoding here just needs
# whatever minimal bands/config structure it requires to construct — pass
# an empty dict if your Lookup_encoding tolerates that, or the smallest
# bands dict your project's Lookup_encoding constructor requires. VERIFY
# this against your actual Lookup_encoding signature — it wasn't shown to
# me, so this is inferred from the FRACTAL script's usage pattern.
configs_dataset_path = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
configs_dataset      = read_yaml(configs_dataset_path)
bands = {}

lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
register_all_resolutions(lookup_table)

# ONLY ELEVATION needs registering as an abstract channel — intensity is
# NOT a separate spectral channel (it rides in column 6 of the same
# elevation token, see DalesTokenProcessor), so no
# register_abstract_channel("INTENSITY") call is needed.
lookup_table.register_abstract_channel("ELEVATION")

# ── Resolve batch size ───────────────────────────────────────────────
if args.batch_size is not None:
    batch_size = args.batch_size
else:
    batch_size = int(config_model["trainer"].get(
        "batchsize", config_model.get("dataset", {}).get("batchsize", 4)))

config_model["trainer"]["epochs"] = args.epochs


# =============================================================================
# CHECKPOINT DIR + AUTO-RESUME LOOKUP
# =============================================================================
ckpt_dir = "./checkpoints/dales/"
os.makedirs(ckpt_dir, exist_ok=True)


def _find_latest_last_checkpoint(xp_name: str) -> str:
    pattern = os.path.join(
        ckpt_dir, f"atomiser_dales_{xp_name}-last-*.ckpt"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None

    def _epoch_from_name(path: str) -> int:
        base = os.path.basename(path)
        nums = re.findall(r"\d+", base)
        return int(nums[-1]) if nums else -1

    matches.sort(key=_epoch_from_name)
    return matches[-1]


auto_resume_ckpt = None
if (not args.test_only
        and args.ckpt_path is None
        and not args.no_auto_resume):
    auto_resume_ckpt = _find_latest_last_checkpoint(args.xp_name)
    if auto_resume_ckpt is not None:
        print(f"\n[DALES] Auto-resume: found checkpoint {auto_resume_ckpt}")
    else:
        print(f"\n[DALES] Auto-resume: no prior checkpoint for "
              f"xp_name='{args.xp_name}' — starting fresh")


print(f"\n{'='*70}")
print(f"  DALES Atomizer — Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Modality:        LIDAR points only (@ 0.2m)")
print(f"  Max LIDAR pts:   {args.max_lidar_points}")
print(f"  Batch size:      {batch_size}")
print(f"  Epochs:          {args.epochs}")
print(f"  Config:          {args.config_model}")
print(f"  Ignore idx:      {args.ignore_index}")
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
    run_id_path = f"training/wandb_runs/atomiser_dales_{args.xp_name}.txt"
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            wandb_resume_id = f.read().strip()
        print(f"[DALES] Will resume wandb run id={wandb_resume_id} "
              f"from {run_id_path}")

wandb_logger = WandbLogger(
    project="Atomizer-DALES",
    name=f"AtomizerDALES_{args.xp_name}",
    save_dir=os.environ.get("WANDB_DIR", "./wandb"),
    config={
        **config_model,
        "max_lidar_points": args.max_lidar_points,
        "batch_size":       batch_size,
        "epochs":           args.epochs,
        "ignore_index":     args.ignore_index,
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

def build_dataset(mode: str):
    return DalesDataset(
        root_path=args.root_path,
        mode=mode,
        dataset_config=bands,
        config_model=config_model,
        look_up=lookup_table,
        max_lidar_points=args.max_lidar_points,
        valid_patches_file=args.valid_patches_file,
    )


def make_loader(dataset, shuffle: bool):
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


print(f"\n[DALES] Building datasets...")
train_ds = build_dataset("train")
val_ds   = build_dataset("val")
test_ds  = build_dataset("test")
print(f"[DALES] Sizes: "
      f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")


# =============================================================================
# DataModule
# =============================================================================

class DalesDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return make_loader(train_ds, shuffle=True)

    def val_dataloader(self):
        return make_loader(val_ds, shuffle=False)

    def test_dataloader(self):
        return make_loader(test_ds, shuffle=False)


data_module = DalesDataModule()


# =============================================================================
# MODEL
# =============================================================================
class_weights_arg = "auto" if args.class_weighting == "auto" else None

model = Model_Dales(
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
callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"precision32_{args.xp_name}-{{epoch:02d}}-"
                 f"{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_dales_{args.xp_name}-last-{{epoch:02d}}",
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
    precision="bf16-mixed",#precision="32-true",
    logger=wandb_logger,
    accumulate_grad_batches=1,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================
# Simplified vs FRACTAL script: no cross-task-transfer-from-FLAIR-HUB path
# (DALES has no natural FLAIR-HUB-compatible modality to transfer from).
# Add one back if you have a pretrained checkpoint you want to init from.

resume_ckpt_path = None

if args.ckpt_path is not None and not args.test_only:
    resume_ckpt_path = args.ckpt_path
    print(f"\n[DALES] Resuming from {args.ckpt_path}")
elif auto_resume_ckpt is not None and not args.test_only:
    resume_ckpt_path = auto_resume_ckpt
    print(f"\n[DALES] Auto-resuming from {resume_ckpt_path}")


# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  DALES — TEST ONLY\n  ckpt: {args.ckpt_path}\n"
          f"{'='*70}\n")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[DALES] missing keys: {len(result.missing_keys)}, "
          f"unexpected keys: {len(result.unexpected_keys)}")

    trainer.test(model, datamodule=data_module, verbose=True)

else:
    print(f"\n{'='*70}\n  DALES — TRAINING\n{'='*70}\n")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)

    print(f"\n{'='*70}\n  DALES — FINAL TEST\n{'='*70}\n")
    trainer.test(model, datamodule=data_module, verbose=True, ckpt_path="best")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger is not None and trainer.is_global_zero:
    import wandb
    run = getattr(wandb, "run", None)
    if run is not None:
        os.makedirs("training/wandb_runs", exist_ok=True)
        with open(
            f"training/wandb_runs/atomiser_dales_{args.xp_name}.txt", "w"
        ) as f:
            f.write(run.id)
        print(f"WANDB_RUN_ID: {run.id}")
