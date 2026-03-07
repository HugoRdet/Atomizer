"""
Pre-training Script — Encode-Once Multi-Task (MMEarth + FLAIR-HUB + Combined)
===============================================================================
Single encode, multiple decodes per sample (~2× speedup over interleaved).

Examples:
    # MMEarth — all 3 tasks (default)
    python train_pretrain_v2.py --xp_name test --config_model atomiser.yaml --dataset_name MMEarth

    # MMEarth — seg only
    python train_pretrain_v2.py --xp_name seg --config_model atomiser.yaml --dataset_name MMEarth \
        --tasks esa_worldcover dynamic_world

    # FLAIR-HUB — all 3 tasks (default)
    python train_pretrain_v2.py --xp_name flair --config_model atomiser.yaml --dataset_name FlairHub \
        --flairhub_path /path/to/FLAIR-HUB

    # FLAIR-HUB — COSIA + reconstruction only, cap at 50k samples
    python train_pretrain_v2.py --xp_name flair_cosia --config_model atomiser.yaml --dataset_name FlairHub \
        --tasks flairhub_cosia reconstruction --max_samples 50000

    # Combined — all 5 tasks from both datasets
    python train_pretrain_v2.py --xp_name combined --config_model atomiser.yaml --dataset_name Combined \
        --tasks esa_worldcover dynamic_world flairhub_cosia flairhub_lpis reconstruction

    # Resume from checkpoint
    python train_pretrain_v2.py --xp_name test --config_model atomiser.yaml --dataset_name MMEarth \
        --ckpt_path ./checkpoints/last.ckpt
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import argparse
import torch
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_MM_Earth_pretrain import MMEarthMultiTask
from training.utils.datasets.utils_dataset_FLAIRHUB import FlairHubMultiTask

# CRITICAL UPDATE: Using the newly built Round-Robin Sampler instead of dummy padding
from training.utils.datasets.token_grouping import collate_multitask, RoundRobinDistributedBatchSampler
from training.utils.callbacks.pre_training_vizcallbacks import (
    PretrainSegVizCallback,
    PretrainReconVizCallback,
    COSIA_CLASS_NAMES,
    LPIS_CLASS_NAMES,
    ESA_CLASS_NAMES,
    DW_CLASS_NAMES,
)

# FlairHub reconstruction viz is now a standalone script:
# python visualize_flairhub_recon.py --ckpt_path ... --config_model ...


# =============================================================================
# DATAMODULE — MMEarth
# =============================================================================

class MMEarthMultiTaskDataModule(pl.LightningDataModule):
    """
    Encode-once multi-task DataModule for MMEarth.

    Every sample contains encoder tokens + query sets for all enabled tasks.
    Standard DistributedSampler — no chunking, no interleaving needed.
    """

    def __init__(
        self,
        mmearth_path: str,
        bands_yaml_path: str,
        config_model: dict,
        look_up,
        batch_size: int = 1,
        num_workers: int = 4,
        subset: str = "MMEarth",
        tasks: list = None,
        max_queries_seg: int = 100_000,
        max_queries_recon: int = 200_000,
        max_samples: int = None,
        val_fraction: float = 0.01,
    ):
        super().__init__()
        self.mmearth_path = mmearth_path
        self.bands_yaml_path = bands_yaml_path
        self.config_model = config_model
        self.look_up = look_up
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset = subset
        self.tasks = tasks or ["esa_worldcover", "dynamic_world", "reconstruction"]
        self.max_queries_seg = max_queries_seg
        self.max_queries_recon = max_queries_recon
        self.max_samples = max_samples
        self.val_fraction = val_fraction
        self.dataset_config = read_yaml(bands_yaml_path)

    def setup(self, stage=None):
        common_kwargs = dict(
            root_path=self.mmearth_path,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up,
            subset=self.subset,
            tasks=self.tasks,
            max_queries_seg=self.max_queries_seg,
            max_queries_recon=self.max_queries_recon,
            max_samples=self.max_samples,
        )

        self.train_dataset = MMEarthMultiTask(mode="train", **common_kwargs)
        self.val_dataset = MMEarthMultiTask(mode="train", **common_kwargs)

        # Deterministic train/val split
        full_len = len(self.train_dataset)
        val_len = max(8, int(full_len * self.val_fraction))
        train_len = full_len - val_len

        generator = torch.Generator().manual_seed(42)
        all_indices = torch.randperm(full_len, generator=generator).tolist()

        self.train_dataset.tile_indices = [
            self.train_dataset.tile_indices[i] for i in all_indices[:train_len]
        ]
        self.val_dataset.tile_indices = [
            self.val_dataset.tile_indices[i] for i in all_indices[train_len:]
        ]

        print(f"[MultiTaskDM] tasks={self.tasks}")
        print(f"[MultiTaskDM] train={train_len}, val={val_len}")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_multitask,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


# =============================================================================
# DATAMODULE — FLAIR-HUB
# =============================================================================

class FlairHubMultiTaskDataModule(pl.LightningDataModule):
    """
    Encode-once multi-task DataModule for FLAIR-HUB.

    Uses CSV-based splits (FLAIR-HUB_TRAIN.csv, FLAIR-HUB_VALID.csv).
    Multi-resolution, multi-temporal tokens + per-task queries.
    """

    def __init__(
        self,
        flairhub_path: str,
        bands_yaml_path: str,
        config_model: dict,
        look_up,
        batch_size: int = 1,
        num_workers: int = 4,
        tasks: list = None,
        max_queries_seg: int = 100_000,
        max_queries_recon: int = 200_000,
        max_samples: int = None,
        csv_dir: str = None,
    ):
        super().__init__()
        self.flairhub_path = flairhub_path
        self.bands_yaml_path = bands_yaml_path
        self.config_model = config_model
        self.look_up = look_up
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tasks = tasks or ["flairhub_cosia", "flairhub_lpis", "reconstruction"]
        self.max_queries_seg = max_queries_seg
        self.max_queries_recon = max_queries_recon
        self.max_samples = max_samples
        self.csv_dir = csv_dir
        self.dataset_config = read_yaml(bands_yaml_path)

    def setup(self, stage=None):
        common_kwargs = dict(
            root_path=self.flairhub_path,
            dataset_config=self.dataset_config,
            config_model=self.config_model,
            look_up=self.look_up,
            tasks=self.tasks,
            max_queries_seg=self.max_queries_seg,
            max_queries_recon=self.max_queries_recon,
            max_samples=self.max_samples,
            csv_dir=self.csv_dir,
        )

        self.train_dataset = FlairHubMultiTask(mode="train", **common_kwargs)
        self.val_dataset = FlairHubMultiTask(mode="validation", **common_kwargs)

        print(f"[FlairHubDM] tasks={self.tasks}")
        print(f"[FlairHubDM] train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_multitask,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


# =============================================================================
# DATAMODULE — COMBINED (MMEarth + FLAIR-HUB)
# =============================================================================

class CombinedMultiTaskDataModule(pl.LightningDataModule):
    """
    Merges MMEarth and FLAIR-HUB into a single training loop.

    Uses RoundRobinDistributedBatchSampler to alternate batches 
    between datasets. Guarantees all ranks process the same dataset 
    at the same step, preventing DDP deadlocks entirely.
    """

    def __init__(
        self,
        mmearth_dm: MMEarthMultiTaskDataModule,
        flairhub_dm: FlairHubMultiTaskDataModule,
        all_tasks: list,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.mmearth_dm = mmearth_dm
        self.flairhub_dm = flairhub_dm
        self.all_tasks = all_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.mmearth_dm.setup(stage)
        self.flairhub_dm.setup(stage)

        # Raw datasets (no TaskPaddingWrapper needed anymore!)
        self.mm_train = self.mmearth_dm.train_dataset
        self.mm_val = self.mmearth_dm.val_dataset
        self.fh_train = self.flairhub_dm.train_dataset
        self.fh_val = self.flairhub_dm.val_dataset

        self.train_dataset = ConcatDataset([self.mm_train, self.fh_train])
        self.val_dataset = ConcatDataset([self.mm_val, self.fh_val])
        
        self.train_lengths = [len(self.mm_train), len(self.fh_train)]
        self.val_lengths = [len(self.mm_val), len(self.fh_val)]

        print(f"[CombinedDM] tasks={self.all_tasks}")
        print(f"[CombinedDM] train: MMEarth={len(self.mm_train)}, FlairHub={len(self.fh_train)}")
        print(f"[CombinedDM] val: MMEarth={len(self.mm_val)}, FlairHub={len(self.fh_val)}")

    def _make_round_robin_loader(self, dataset, lengths, shuffle):
        sampler = RoundRobinDistributedBatchSampler(
            dataset_lengths=lengths,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler, # Important: pass via batch_sampler
            num_workers=self.num_workers,
            collate_fn=collate_multitask,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_round_robin_loader(self.train_dataset, self.train_lengths, shuffle=True)

    def val_dataloader(self):
        return self._make_round_robin_loader(self.val_dataset, self.val_lengths, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


# =============================================================================
# VALID TASKS
# =============================================================================
MMEARTH_TASKS = ["esa_worldcover", "dynamic_world", "reconstruction"]
FLAIRHUB_TASKS = ["flairhub_cosia", "flairhub_lpis", "reconstruction"]
COMBINED_TASKS = ["esa_worldcover", "dynamic_world", "flairhub_cosia", "flairhub_lpis", "reconstruction"]

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Atomizer Pre-training (Encode-Once)")
parser.add_argument("--xp_name",        type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",   type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",   type=str, required=True,
                    choices=["MMEarth", "MMEarth100k", "MMEarth64", "FlairHub", "Combined"],
                    help="Dataset to use")
parser.add_argument("--tasks",          type=str, nargs="+", default=None,
                    help="Tasks to train on (default: all for chosen dataset)")
# Paths
parser.add_argument("--mmearth_path",   type=str, default="./data/MM-Earth",
                    help="Path to MMEarth data")
parser.add_argument("--flairhub_path",  type=str, default="./data/FLAIR-HUB/extracted",
                    help="Path to FLAIR-HUB data")
parser.add_argument("--flairhub_csv_dir", type=str, default="./data/FLAIR-HUB",
                    help="Path to FLAIR-HUB CSV split files (default: flairhub_path)")
# Training
parser.add_argument("--ckpt_path",      type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--max_queries_seg",   type=int, default=100_000)
parser.add_argument("--max_queries_recon", type=int, default=200_000)
parser.add_argument("--max_samples",       type=int, default=None,
                    help="Cap training set size per dataset (e.g. 100000). Default: use all.")
# Viz
parser.add_argument("--viz_every_n_epochs", type=int, default=1,
                    help="Log visualization every N epochs (default: 1)")

args = parser.parse_args()

# Resolve default tasks based on dataset
is_flairhub = args.dataset_name == "FlairHub"
is_combined = args.dataset_name == "Combined"

if args.tasks is None:
    if is_combined:
        args.tasks = COMBINED_TASKS
    elif is_flairhub:
        args.tasks = FLAIRHUB_TASKS
    else:
        args.tasks = MMEARTH_TASKS

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

task_label = "_".join(sorted(args.tasks))

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"{config_model['encoder']}_{args.dataset_name}_{task_label}",
        project="Atomizer_Pretrain",
        config={**config_model, "tasks": args.tasks, "dataset": args.dataset_name},
    )
    wandb_logger = WandbLogger(project="Atomizer_Pretrain")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
model = Model_Pretrain(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
)

# =============================================================================
# DATA MODULE
# =============================================================================
if is_combined:
    # Split tasks by dataset
    mm_tasks = [t for t in args.tasks if t in MMEARTH_TASKS]
    fh_tasks = [t for t in args.tasks if t in FLAIRHUB_TASKS]

    # Ensure reconstruction is in both if requested
    if "reconstruction" in args.tasks:
        if "reconstruction" not in mm_tasks:
            mm_tasks.append("reconstruction")
        if "reconstruction" not in fh_tasks:
            fh_tasks.append("reconstruction")

    mmearth_dm = MMEarthMultiTaskDataModule(
        mmearth_path=args.mmearth_path,
        bands_yaml_path=bands_yaml,
        config_model=config_model,
        look_up=lookup_table,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=args.num_workers,
        subset="MMEarth",
        tasks=mm_tasks,
        max_queries_seg=args.max_queries_seg,
        max_queries_recon=args.max_queries_recon,
        max_samples=args.max_samples,
    )

    flairhub_dm = FlairHubMultiTaskDataModule(
        flairhub_path=args.flairhub_path,
        bands_yaml_path=bands_yaml,
        config_model=config_model,
        look_up=lookup_table,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=args.num_workers,
        tasks=fh_tasks,
        max_queries_seg=args.max_queries_seg,
        max_queries_recon=args.max_queries_recon,
        max_samples=args.max_samples,
        csv_dir=args.flairhub_csv_dir,
    )

    data_module = CombinedMultiTaskDataModule(
        mmearth_dm=mmearth_dm,
        flairhub_dm=flairhub_dm,
        all_tasks=args.tasks,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=args.num_workers,
    )

elif is_flairhub:
    data_module = FlairHubMultiTaskDataModule(
        flairhub_path=args.flairhub_path,
        bands_yaml_path=bands_yaml,
        config_model=config_model,
        look_up=lookup_table,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=args.num_workers,
        tasks=args.tasks,
        max_queries_seg=args.max_queries_seg,
        max_queries_recon=args.max_queries_recon,
        max_samples=args.max_samples,
        csv_dir=args.flairhub_csv_dir,
    )

else:
    data_module = MMEarthMultiTaskDataModule(
        mmearth_path=args.mmearth_path,
        bands_yaml_path=bands_yaml,
        config_model=config_model,
        look_up=lookup_table,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=args.num_workers,
        subset=args.dataset_name,
        tasks=args.tasks,
        max_queries_seg=args.max_queries_seg,
        max_queries_recon=args.max_queries_recon,
        max_samples=args.max_samples,
    )

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")
accumulator = GradientAccumulationScheduler(scheduling={0: 16})

# ── Checkpointing ────────────────────────────────────────────────────────────
ALL_SEG_TASKS = ["esa_worldcover", "dynamic_world", "flairhub_cosia", "flairhub_lpis"]
has_seg = any(t in args.tasks for t in ALL_SEG_TASKS)

if has_seg:
    ckpt_monitor = "val_avg_mIoU"
    ckpt_mode = "max"
    ckpt_fmt = f"{config_model['encoder']}_{xp_name}-val_mIoU-{{epoch:02d}}-{{val_avg_mIoU:.4f}}"
else:
    ckpt_monitor = "val_recon_mse"
    ckpt_mode = "min"
    ckpt_fmt = f"{config_model['encoder']}_{xp_name}-val_mse-{{epoch:02d}}-{{val_recon_mse:.4f}}"

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=ckpt_fmt,
    monitor=ckpt_monitor,
    mode=ckpt_mode,
    save_top_k=1,
    verbose=True,
)

checkpoint_resume = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}_{xp_name}-last-{{epoch:02d}}",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
    verbose=True,
)

# ── Visualization callbacks ──────────────────────────────────────────────────
# Task → (class_names, rgb_indices)
VIZ_TASK_CONFIG = {
    "flairhub_cosia":  (COSIA_CLASS_NAMES, [0, 1, 2]),  # FLAIR aerial: R=0, G=1, B=2
    "flairhub_lpis":   (LPIS_CLASS_NAMES,  [0, 1, 2]),
    "esa_worldcover":  (ESA_CLASS_NAMES,   [2, 1, 0]),  # MMEarth S2: B04=2, B03=1, B02=0
    "dynamic_world":   (DW_CLASS_NAMES,    [2, 1, 0]),
}

viz_callbacks = []

# ── Segmentation viz (per-task) ──────────────────────────────────────────────
for task_name in args.tasks:
    if task_name in VIZ_TASK_CONFIG:
        class_names, rgb_idx = VIZ_TASK_CONFIG[task_name]
        viz_callbacks.append(
            PretrainSegVizCallback(
                task_name=task_name,
                class_names=class_names,
                rgb_indices=rgb_idx,
                sample_indices=(0, 1, 2),
                log_every_n_epochs=args.viz_every_n_epochs,
            )
        )

# ── Reconstruction viz ───────────────────────────────────────────────────────
if "reconstruction" in args.tasks:
    # MMEarth single-resolution recon (when MMEarth is involved)
    if not is_flairhub:  # i.e. MMEarth-only or Combined
        recon_rgb = [2, 1, 0]  # S2: B04, B03, B02
        viz_callbacks.append(
            PretrainReconVizCallback(
                rgb_indices=recon_rgb,
                sample_indices=(0, 1),
                log_every_n_epochs=args.viz_every_n_epochs,
            )
        )

    # FlairHub multi-modal recon: use standalone visualize_flairhub_recon.py
    # after training (no callback needed — avoids blocking DDP)

callbacks = [accumulator, checkpoint_val, checkpoint_resume, lr_monitor] #+ viz_callbacks
print(f"[Callbacks] {len(viz_callbacks)} viz callbacks: "
      f"{[type(c).__name__ for c in viz_callbacks]}")

# =============================================================================
# SETUP DATA MODULE
# =============================================================================
data_module.setup()

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False, 
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir="./checkpoints/",
    gradient_clip_val=1.0,
    limit_val_batches=10000,
    val_check_interval=0.05
)

# =============================================================================
# TRAIN & TEST
# =============================================================================
trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
trainer.test(model, datamodule=data_module)

# =============================================================================
# MEASURE COMPLEXITY (rank 0 only)
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


if os.environ.get("LOCAL_RANK", "0") == "0":
    from fvcore.nn import FlopCountAnalysis

    print("\n" + "=" * 80)
    print("MEASURING MODEL COMPLEXITY")
    print("=" * 80 + "\n")

    data_module.setup("test")
    test_dataset = data_module.val_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_samples = min(25, len(test_dataset))
    samples = [test_dataset[i] for i in range(num_samples)]

    results = []

    for input_size in [128]:
        print(f"\nTesting input size: {input_size}x{input_size}")

        batch_0 = collate_multitask([samples[0]])
        batch_0 = _batch_to_device(batch_0, device)

        # Warmup
        with torch.no_grad():
            _ = model.forward_multitask(batch_0, training=False)

        # FLOPs
        batch_1 = collate_multitask([samples[1]])
        batch_1 = _batch_to_device(batch_1, device)

        try:
            with torch.no_grad():
                flops = FlopCountAnalysis(model, (batch_1,))
                gflops = flops.total() / 1e9
        except Exception as e:
            print(f"  FLOPs measurement failed: {e}")
            gflops = -1

        # Inference time
        num_warmup, num_runs = 3, 20
        with torch.no_grad():
            for i in range(num_warmup):
                b = collate_multitask([samples[i % num_samples]])
                b = _batch_to_device(b, device)
                _ = model.forward_multitask(b, training=False)

            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_runs):
                idx = (i + num_warmup) % num_samples
                b = collate_multitask([samples[idx]])
                b = _batch_to_device(b, device)
                _ = model.forward_multitask(b, training=False)
                torch.cuda.synchronize()
            end = time.time()

        avg_time_ms = (end - start) / num_runs * 1000

        first_res = next(iter(batch_0["groups"]))
        num_tokens = batch_0["groups"][first_res]["tokens"].shape[1]

        results.append({
            "input_size": input_size,
            "num_tokens": num_tokens,
            "gflops": gflops,
            "inference_time_ms": avg_time_ms,
        })

        print(f"  Tokens: {num_tokens}")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Inference time: {avg_time_ms:.2f} ms/tile")

    print("\n" + "=" * 80)
    print(f"COMPLEXITY SUMMARY ({config_model['encoder']})")
    print("=" * 80)
    print(f"{'Input':<10} {'Tokens':<12} {'GFLOPs':<12} {'Time (ms)':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['input_size']:<10} {r['num_tokens']:<12} {r['gflops']:<12.2f} {r['inference_time_ms']:<12.2f}")
    print("=" * 80)

# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)