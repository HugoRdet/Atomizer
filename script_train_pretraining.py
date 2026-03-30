"""
Pre-training Script — Encode-Once Multi-Task (MMEarth + FLAIR-HUB)
===================================================================
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

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

# --- Project imports ---
from training.utils import read_yaml
from training.utils import Lookup_encoding
from training.trainer_pretraining_v2 import Model_Pretrain
from training.utils.datasets.mmearth_multitask import MMEarthMultiTask
from training.utils.datasets.flairhub_multitask import FlairHubMultiTask
from training.utils.datasets.collate_multitask import collate_multitask
from training.viz_callbacks_pretrain import (
    PretrainSegVizCallback,
    PretrainReconVizCallback,
)


# =============================================================================
# DATAMODULE
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
        subset: str = "MMEarth100k",
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
# VALID TASKS
# =============================================================================
MMEARTH_TASKS = ["esa_worldcover", "dynamic_world", "reconstruction"]
FLAIRHUB_TASKS = ["flairhub_cosia", "flairhub_lpis", "reconstruction"]
ALL_TASKS = MMEARTH_TASKS + ["flairhub_cosia", "flairhub_lpis"]  # reconstruction shared

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Atomizer Pre-training (Encode-Once)")
parser.add_argument("--xp_name",        type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",   type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",   type=str, required=True,
                    choices=["MMEarth", "MMEarth100k", "MMEarth64", "FlairHub"],
                    help="Dataset to use")
parser.add_argument("--tasks",          type=str, nargs="+", default=None,
                    help="Tasks to train on (default: all for chosen dataset)")
# Paths
parser.add_argument("--mmearth_path",   type=str, default="./data/MM-Earth",
                    help="Path to MMEarth data")
parser.add_argument("--flairhub_path",  type=str, default="./data/FLAIR-HUB/FLAIR-HUB",
                    help="Path to FLAIR-HUB data")
parser.add_argument("--flairhub_csv_dir", type=str, default=None,
                    help="Path to FLAIR-HUB CSV split files (default: flairhub_path)")
# Training
parser.add_argument("--ckpt_path",      type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--max_queries_seg",   type=int, default=100_000)
parser.add_argument("--max_queries_recon", type=int, default=200_000)
parser.add_argument("--max_samples",       type=int, default=None,
                    help="Cap training set size (e.g. 100000). Default: use all.")

args = parser.parse_args()

# Resolve default tasks based on dataset
is_flairhub = args.dataset_name == "FlairHub"
if args.tasks is None:
    args.tasks = FLAIRHUB_TASKS if is_flairhub else MMEARTH_TASKS

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
if is_flairhub:
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
accumulator = GradientAccumulationScheduler(scheduling={0: 10})

# Checkpoint — monitor seg metric if any seg task present, else recon
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

# Viz callbacks — auto-select based on which tasks are enabled
viz_callbacks = []

seg_tasks_in_use = [t for t in args.tasks if t != "reconstruction"]
for task_name in seg_tasks_in_use:
    viz_callbacks.append(
        PretrainSegVizCallback(
            task_name=task_name,
            sample_indices=(0, 1, 2),
            log_every_n_epochs=1,
        )
    )

if "reconstruction" in args.tasks:
    viz_callbacks.append(
        PretrainReconVizCallback(
            sample_indices=(0,),
            log_every_n_epochs=1,
        )
    )

callbacks = [accumulator, checkpoint_val, checkpoint_resume, lr_monitor] + viz_callbacks

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    strategy="ddp_find_unused_parameters_true",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir="./checkpoints/",
    gradient_clip_val=1.0,
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