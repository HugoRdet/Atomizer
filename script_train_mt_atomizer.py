"""
Atomizer Multi-Task Training Script
====================================

Trains MultiTaskAtomizer on 5 heterogeneous EO benchmarks with a shared
encoder + per-task seg/cls heads:

    - BurnScars       (seg, 2 classes, 6 bands @ 30m, 512×512)
    - Sen1Floods11    (seg, 2 classes, 13 S2 + 2 S1 @ 10m, 512×512)
    - PASTIS-HD       (seg, 19 classes, 10 S2 + 3 S1 multi-temporal @ 10m)
    - EuroSAT         (cls, 10 classes, 13 S2 @ 10m, 64×64)
    - ForestNet       (cls, 12 classes, ... single-task config)

Round-robin assumption
----------------------
At each micro-step the loader yields a batch from one task. All DDP
ranks see the SAME task at the SAME step (deterministic round-robin in
RoundRobinLoader). Each rank gets a disjoint shard of that task's data
via DistributedSampler.

Pipeline
--------
For each task:
  1. Build single-task Atomizer dataset (BurnScars, Sen1Floods11, ...).
  2. Wrap in TaggedAtomiserDataset(task_name, task_type) — adds
     batch["task"] and batch["task_type"], lifts queries from the
     "tasks" dict if PASTIS-style.
  3. Build a DataLoader with DistributedSampler (manual, since we set
     use_distributed_sampler=False on pl.Trainer).
  4. Build a per-task collate via make_atomiser_mt_collate(task, type).

Then combine all 5 DataLoaders into one RoundRobinLoader with length
= num_tasks × rounds_per_epoch and feed to pl.Trainer.

Test phase
----------
After fit(), we run trainer.test() once per task — the existing
AtomizerMultiTaskTrainer's test_step expects batch["task"] to be set
(round-robin doesn't help here since we want each task's full test set
in isolation). Each test phase uses a fresh pl.Trainer to avoid stale
logging state between phases.

Examples
--------
    # Default: 5-task MT, all GPUs, default config
    python script_train_mt_atomiser.py --xp_name mt_v1

    # Resume from checkpoint, same wandb run
    python script_train_mt_atomiser.py --xp_name mt_v1_resumed \
        --ckpt_path ./checkpoints/atomiser_mt_v1-best.ckpt \
        --wandb_run_id <id_of_aborted_run>

    # Test-only on a saved checkpoint
    python script_train_mt_atomiser.py --xp_name mt_v1_test \
        --ckpt_path ./checkpoints/atomiser_mt_v1-best.ckpt --test_only
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

# Project — config + lookup
from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder

# Trainer + model
from training.trainer_atomiser_mt import AtomizerMultiTaskTrainer

# Single-task Atomiser datasets
from training.utils.datasets.utils_dataset_BURNSCARS import BurnScarsDataset
from training.utils.datasets.utils_dataset_SENFLOOD  import Sen1Floods11Dataset
from training.utils.datasets.utils_dataset_PASTIS    import PastisHDDataset
from training.utils.datasets.utils_dataset_EUROSAT   import EuroSATDataset
from training.utils.datasets.utils_dataset_FORESTNET import ForestNetDataset

# MT pipeline (Atomiser-side wrapper + collate + round-robin)
from training.utils.datasets_mt.mt_pipeline import (
    TaggedAtomiserDataset,
    make_atomiser_mt_collate,
    RoundRobinLoader,
)


# =============================================================================
# RESOLUTION REGISTRATION (matches PASTIS-HD / BurnScars scripts)
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    1.0:  2048,
    2.5:  2048,
    10.0: 2048,
    20.0: 2048,
    30.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


# =============================================================================
# TASK SPECS
# =============================================================================

# Order of this dict determines:
#   - the order of round-robin task selection
#   - the order metrics are logged in
TASK_SPECS = {
    "burnscars": {"type": "seg", "num_classes": 2,  "ignore_index": 255},
    "senflood":  {"type": "seg", "num_classes": 2,  "ignore_index": 255},
    "pastis":    {"type": "seg", "num_classes": 20, "ignore_index": 255},
    "eurosat":   {"type": "cls", "num_classes": 10},
    "forestnet": {"type": "cls", "num_classes": 12},
}


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Atomizer Multi-Task Training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,
                    default="config_test-Atomiser_Atos_One.yaml")
parser.add_argument("--dataset_name", type=str, default="u_regular",
                    help="configs_dataset_<name>.yaml under data/Tiny_BigEarthNet/")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--grad_accum",   type=int, default=None,
                    help="Gradient accumulation. Default: number of tasks "
                         "(so one optimizer step = one full round of all tasks).")
# Resume / test
parser.add_argument("--ckpt_path",    type=str, default=None)
parser.add_argument("--wandb_run_id", type=str, default=None)
parser.add_argument("--test_only",    action="store_true")
# Per-task data roots (override defaults if needed)
parser.add_argument("--burnscars_root", type=str, default="./data/hls_burn_scars")
parser.add_argument("--senflood_root",  type=str, default="./data/SENFLOOD")
parser.add_argument("--pastis_root",    type=str, default="./data/PASTIS-HD")
parser.add_argument("--eurosat_root",   type=str,
                    default="./data/geo-bench-1.0/classification_v1.0/m-eurosat")
parser.add_argument("--forestnet_root", type=str,
                    default="./data/geo-bench-1.0/classification_v1.0/m-forestnet")
args = parser.parse_args()


# =============================================================================
# CONFIG + LOOKUP
# =============================================================================
config_model         = read_yaml(f"./training/configs/{args.config_model}")
configs_dataset_path = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml_path      = "./data/bands_info/bands.yaml"

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

# Register VV-VH abstract channel for PASTIS S1 (3rd band)
lookup_table.register_abstract_channel("VV_VH")

# Shared bands dict (passed through to each dataset's `dataset_config`)
bands_dict = read_yaml(bands_yaml_path)

# Batch size — uniform across tasks (per user spec)
batch_size = int(config_model["trainer"].get("batchsize",
              config_model.get("dataset", {}).get("batchsize", 4)))
num_tasks  = len(TASK_SPECS)
grad_accum = args.grad_accum if args.grad_accum is not None else num_tasks


print(f"\n{'='*70}")
print(f"  Atomizer MT — Experiment: {args.xp_name}")
print(f"{'='*70}")
print(f"  Tasks:       {list(TASK_SPECS.keys())}")
print(f"  Batch size:  {batch_size} per task (uniform)")
print(f"  Grad accum:  {grad_accum} (1 optimizer step per "
      f"{grad_accum} micro-batches)")
print(f"  Config:      {args.config_model}")
print(f"  Lookup tbl:  {len(lookup_table.table_wave)} entries")


# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb

    run_name = f"AtomizerMT_{args.xp_name}"
    wandb_init_kwargs = dict(
        name=run_name,
        project="Atomizer-MT",
        config={**config_model, "task_specs": TASK_SPECS,
                "batch_size": batch_size, "grad_accum": grad_accum},
    )
    if args.wandb_run_id is not None:
        wandb_init_kwargs["id"]     = args.wandb_run_id
        wandb_init_kwargs["resume"] = "must"
        print(f"  W&B:         resuming run {args.wandb_run_id}")
    else:
        print(f"  W&B:         new run {run_name}")
    wandb.init(**wandb_init_kwargs)
    wandb_logger = WandbLogger(project="Atomizer-MT")


# =============================================================================
# PER-TASK DATASET BUILDERS
# =============================================================================
# Inline construction of each single-task dataset, then wrap with
# TaggedAtomiserDataset. Per-task differences (root_path, use_s1, etc.)
# are explicit here rather than buried in builder functions.

def build_burnscars(mode):
    base = BurnScarsDataset(
        root_path=args.burnscars_root,
        mode=mode,
        config_model=config_model,
        dataset_config=bands_dict,
        look_up=lookup_table,
    )
    return TaggedAtomiserDataset(base, task_name="burnscars", task_type="seg")


def build_senflood(mode):
    base = Sen1Floods11Dataset(
        root_path=args.senflood_root,
        mode=mode,
        config_model=config_model,
        dataset_config=bands_dict,
        look_up=lookup_table,
    )
    return TaggedAtomiserDataset(base, task_name="senflood", task_type="seg")


def build_pastis(mode):
    base = PastisHDDataset(
        root_path=args.pastis_root,
        mode=mode,
        config_model=config_model,
        look_up=lookup_table,
        use_s1=True,
        use_spot=False,           # SPOT off for MT (single-resolution comparison)
    )
    return TaggedAtomiserDataset(base, task_name="pastis", task_type="seg")


def build_eurosat(mode):
    base = EuroSATDataset(
        root_path=args.eurosat_root,
        mode=mode,
        config_model=config_model,
        dataset_config=bands_dict,
        look_up=lookup_table,
    )
    return TaggedAtomiserDataset(base, task_name="eurosat", task_type="cls")


def build_forestnet(mode):
    base = ForestNetDataset(
        root_path=args.forestnet_root,
        mode=mode,
        config_model=config_model,
        dataset_config=bands_dict,
        look_up=lookup_table,
    )
    return TaggedAtomiserDataset(base, task_name="forestnet", task_type="cls")


TASK_BUILDERS = {
    "burnscars": build_burnscars,
    "senflood":  build_senflood,
    "pastis":    build_pastis,
    "eurosat":   build_eurosat,
    "forestnet": build_forestnet,
}


# =============================================================================
# DATALOADER FACTORY (DDP-aware)
# =============================================================================

def make_loader(dataset, task_name: str, task_type: str, shuffle: bool):
    """
    DataLoader with manual DistributedSampler (since we set
    use_distributed_sampler=False on pl.Trainer).

    Each rank gets a disjoint shard of this task's data; all ranks
    select the same task at the same step (via RoundRobinLoader).
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
        collate_fn=make_atomiser_mt_collate(task_name, task_type),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=shuffle,
    )


# =============================================================================
# DATAMODULE — round-robin train/val, per-task test
# =============================================================================

class AtomizerMTDataModule(pl.LightningDataModule):
    """
    Wraps the 5-task setup in a Lightning-friendly module.

    Train/val use RoundRobinLoader so all ranks see the same task at
    each step. Test uses per-task DataLoaders separately (called by the
    launch script after fit() — see below).
    """

    def __init__(self):
        super().__init__()
        self._train_datasets = None
        self._val_datasets   = None
        self._test_datasets  = None
        self._setup_done     = False

    def setup(self, stage=None):
        if self._setup_done:
            return
        self._setup_done = True

        self._train_datasets = {t: TASK_BUILDERS[t]("train")
                                for t in TASK_SPECS}
        self._val_datasets   = {t: TASK_BUILDERS[t]("validation")
                                for t in TASK_SPECS}
        self._test_datasets  = {t: TASK_BUILDERS[t]("test")
                                for t in TASK_SPECS}

        print(f"\n[MT-DM] Per-task dataset sizes:")
        for t in TASK_SPECS:
            print(f"  {t:<14} train={len(self._train_datasets[t]):>6}  "
                  f"val={len(self._val_datasets[t]):>6}  "
                  f"test={len(self._test_datasets[t]):>6}")

    # Train and val use round-robin
    def train_dataloader(self):
        return self._make_round_robin("train", self._train_datasets,
                                      shuffle=True)

    def val_dataloader(self):
        return self._make_round_robin("val", self._val_datasets,
                                      shuffle=False)

    def _make_round_robin(self, stage: str, ds_dict: dict, shuffle: bool):
        # Build per-task loaders
        loaders = {
            t: make_loader(ds, t, TASK_SPECS[t]["type"], shuffle=shuffle)
            for t, ds in ds_dict.items()
        }

        # Length: cycle through all per-task loaders in proportion to
        # their size. Convention: sum of per-task lengths (each rank
        # consumes batches at the same rate, so this is a per-rank count).
        total = sum(len(l) for l in loaders.values())
        # Round to a multiple of num_tasks so every task gets the same
        # number of round-robin slots in an epoch.
        total = (total // num_tasks) * num_tasks
        total = max(total, num_tasks)
        print(f"[MT-DM] {stage} round-robin length: {total} micro-batches "
              f"({total // num_tasks} rounds × {num_tasks} tasks)")
        return RoundRobinLoader(loaders, length=total)

    def test_dataloader_for(self, task_name: str):
        """Return a single-task test DataLoader (used by launch script)."""
        ds = self._test_datasets[task_name]
        return make_loader(ds, task_name, TASK_SPECS[task_name]["type"],
                           shuffle=False)


# =============================================================================
# BUILD MODEL + DM
# =============================================================================
data_module = AtomizerMTDataModule()
data_module.setup()

model = AtomizerMultiTaskTrainer(
    config=config_model,
    wand=wandb_logger is not None,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
    task_specs=TASK_SPECS,
)


# =============================================================================
# CALLBACKS + TRAINER
# =============================================================================
ckpt_dir = "./checkpoints/atomiser_mt/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_mt_{args.xp_name}-{{epoch:02d}}-"
                 f"{{val_mean_primary:.4f}}",
        monitor="val_mean_primary",
        mode="max",
        save_top_k=1,
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"atomiser_mt_{args.xp_name}-last-{{epoch:02d}}",
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
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    accumulate_grad_batches=grad_accum,
)


# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  Atomizer MT — TEST ONLY\n  ckpt: {args.ckpt_path}\n"
          f"{'='*70}\n")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[MT] missing: {len(result.missing_keys)}, "
          f"unexpected: {len(result.unexpected_keys)}")
else:
    print(f"\n{'='*70}\n  Atomizer MT — TRAINING\n{'='*70}\n")
    if args.ckpt_path is not None:
        print(f"  Resuming from: {args.ckpt_path}")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)


# =============================================================================
# PER-TASK TEST PHASE
# =============================================================================
# Each task gets its own test phase with a fresh pl.Trainer. This avoids
# metric collisions and stale logging state between phases. We collect
# per-task results into a single dict that we log at the end.

print(f"\n{'='*70}\n  Atomizer MT — PER-TASK TEST\n{'='*70}\n")

per_task_results = {}
for task_name in TASK_SPECS:
    print(f"\n--- Test: {task_name} ---")

    # Fresh trainer per task. Same callbacks/logger but new state.
    test_trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        use_distributed_sampler=False,
        devices=-1,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        default_root_dir=ckpt_dir,
    )

    test_loader = data_module.test_dataloader_for(task_name)
    out = test_trainer.test(model, dataloaders=test_loader, verbose=True)
    if out:
        # `out` is a list of dicts (one per dataloader). Take the first.
        per_task_results[task_name] = out[0]


# =============================================================================
# RESULTS SUMMARY
# =============================================================================
if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"\n{'='*70}\n  Atomizer MT — FINAL RESULTS\n{'='*70}")
    primary_values = []
    for task_name, res_dict in per_task_results.items():
        spec = TASK_SPECS[task_name]
        primary = "mIoU" if spec["type"] == "seg" else "macro_f1"
        # Find the per-task primary metric in the test logs.
        primary_key = f"test_{primary}/{task_name}"
        val = res_dict.get(primary_key)
        if val is not None:
            primary_values.append(val)
            print(f"  {task_name:<14} {primary:<10}  {val:.4f}")
        else:
            print(f"  {task_name:<14} {primary:<10}  (not found in test logs)")

    if primary_values:
        mean_primary = sum(primary_values) / len(primary_values)
        print(f"  {'='*40}")
        print(f"  mean_primary across {len(primary_values)} tasks: {mean_primary:.4f}")
        print(f"{'='*70}\n")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/atomiser_mt_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)
    print(f"WANDB_RUN_ID: {wandb.run.id}")