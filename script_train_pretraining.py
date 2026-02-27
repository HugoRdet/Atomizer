"""
Pre-training Script — MMEarth + FLAIR-HUB
==========================================
Supports single-task validation runs (one dataset class at a time)
and multi-task training (ChunkedInterleaved with all tasks).

Uses grouped-token batch format:
    batch = {
        "groups": {res: {"tokens": [B,N,8], "mask": [B,N], "shape": (C,H,W)}},
        "queries":      [B, M, 8],
        "queries_mask":  [B, M],
        "task":          str,
    }

Examples:
    # MMEarth multi-task (original)
    python train_pretrain.py --xp_name test --config_model atomiser.yaml --dataset_name MMEarth --task all

    # FLAIR-HUB reconstruction only (toy dataset test)
    python train_pretrain.py --xp_name flair_test --config_model atomiser.yaml --dataset_name FLAIR-HUB \
        --task flairhub_recon --flairhub_path ./data/FLAIR-HUB/toy/FLAIR-HUB_TOY

    # FLAIR-HUB all tasks
    python train_pretrain.py --xp_name flair_all --config_model atomiser.yaml --dataset_name FLAIR-HUB \
        --task all_flairhub --flairhub_path ./data/FLAIR-HUB/toy/FLAIR-HUB_TOY

    # Combined MMEarth + FLAIR-HUB
    python train_pretrain.py --xp_name combined --config_model atomiser.yaml --dataset_name combined \
        --task all_combined --flairhub_path ./data/FLAIR-HUB/toy/FLAIR-HUB_TOY
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import argparse
import torch
import numpy as np
from collections import defaultdict

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)

seed_everything(42, workers=True)

# --- Project imports ---
from training.utils import read_yaml
from training.utils import Lookup_encoding

from training.trainer_pretraining import Model_Pretrain

from training.utils.datasets.utils_dataset_MM_Earth_pretrain import (
    MMEarthReconstruction, MMEarthSegDW, MMEarthSegESA,
    ESA_NUM_CLASSES, DW_NUM_CLASSES,
)

from training.utils.datasets.utils_dataset_FLAIRHUB import (
    FlairHubSegCOSIA, FlairHubSegLPIS, FlairHubReconstruction,
)

from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.token_grouping import collate_grouped
from training.utils.datasets.dataloaders_MTP import PretrainDataModule

from training.utils.callbacks.pre_training_vizcallbacks import (
    PretrainSegVizCallback, PretrainReconVizCallback,
    ESA_CLASS_NAMES, DW_CLASS_NAMES,
)

# =============================================================================
# DATASET CLASS REGISTRY
# =============================================================================
DATASET_CLASSES = {
    # MMEarth
    "esa_worldcover":  MMEarthSegESA,
    "dynamic_world":   MMEarthSegDW,
    "reconstruction":  MMEarthReconstruction,
    # FLAIR-HUB
    "flairhub_cosia":  FlairHubSegCOSIA,
    "flairhub_lpis":   FlairHubSegLPIS,
    "flairhub_recon":  FlairHubReconstruction,
}

# Task groupings for --task argument
TASK_GROUPS = {
    "all":              {"mmearth": True,  "flairhub": False},
    "all_flairhub":     {"mmearth": False, "flairhub": True},
    "all_combined":     {"mmearth": True,  "flairhub": True},
}

VALID_TASKS = list(DATASET_CLASSES.keys()) + list(TASK_GROUPS.keys())

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Atomizer Pre-training (MMEarth + FLAIR-HUB)")
parser.add_argument("--xp_name",        type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",   type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",   type=str, required=True, help="Name of the dataset used")
parser.add_argument("--task",           type=str, default="all",
                    choices=VALID_TASKS,
                    help="Which task to run (default: all)")
# Paths
parser.add_argument("--mmearth_path",   type=str, default="./data/MM-Earth",
                    help="Path to MMEarth data")
parser.add_argument("--flairhub_path",  type=str, default="./data/FLAIR-HUB/FLAIR-HUB_TOY",
                    help="Path to FLAIR-HUB data")
# FLAIR-HUB options
parser.add_argument("--flairhub_tasks", type=str, nargs="+",
                    default=["cosia", "lpis", "recon"],
                    help="FLAIR-HUB tasks to include (cosia, lpis, recon)")
parser.add_argument("--flairhub_max_timestamps", type=int, default=10,
                    help="Max timestamps per temporal modality")
parser.add_argument("--flairhub_temporal_dropout", type=float, default=0.3,
                    help="Fraction of timestamps to drop during training")
parser.add_argument("--flairhub_csv_dir", type=str, default=None,
                    help="Directory containing FLAIR-HUB split CSVs (defaults to flairhub_path)")

parser.add_argument("--ckpt_path", type=str, default=None, help="Resume from checkpoint")


args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

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
        name=f"{config_model['encoder']}_{args.task}",
        project="Atomizer_Pretrain",
        config={**config_model, "task": args.task,
                "flairhub_path": args.flairhub_path,
                "flairhub_tasks": args.flairhub_tasks},
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
task = args.task

# Determine enable flags
if task in TASK_GROUPS:
    # Multi-task mode
    enable_mmearth = TASK_GROUPS[task]["mmearth"]
    enable_flairhub = TASK_GROUPS[task]["flairhub"]
    use_multi_task = True
elif task.startswith("flairhub_"):
    # Single FLAIR-HUB task
    enable_mmearth = False
    enable_flairhub = True
    use_multi_task = True  # Use PretrainDataModule even for single FLAIR-HUB task
    # Map task name to flairhub_tasks list
    flair_task_map = {
        "flairhub_cosia": ["cosia"],
        "flairhub_lpis":  ["lpis"],
        "flairhub_recon": ["recon"],
    }
    args.flairhub_tasks = flair_task_map[task]
else:
    # Single MMEarth task — use original UnifiedDataModule
    enable_mmearth = True
    enable_flairhub = False
    use_multi_task = False

if use_multi_task:
    data_module = PretrainDataModule(
        mmearth_path=args.mmearth_path,
        flairhub_path=args.flairhub_path,
        bands_yaml_path=bands_yaml,
        config_model=config_model,
        look_up=lookup_table,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=4,
        enable_mmearth=enable_mmearth,
        enable_flairhub=enable_flairhub,
        flairhub_tasks=args.flairhub_tasks,
        flairhub_max_timestamps=args.flairhub_max_timestamps,
        flairhub_temporal_dropout=args.flairhub_temporal_dropout,
        flairhub_csv_dir=args.flairhub_csv_dir,
    )
else:
    dataset_class = DATASET_CLASSES[task]
    data_module = UnifiedDataModule(
        path=args.mmearth_path,
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=4,
        trans_modalities=None,
        trans_tokens=None,
        model=config_model["encoder"],
        dataset_config=read_yaml(bands_yaml),
        config_model=config_model,
        look_up=lookup_table,
        dataset_class=dataset_class,
    )

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

# ── Checkpoint: pick the right metric per task ──
seg_tasks = ["esa_worldcover", "dynamic_world", "flairhub_cosia", "flairhub_lpis",
             "all", "all_flairhub", "all_combined"]
recon_tasks = ["reconstruction", "flairhub_recon"]

if task in seg_tasks:
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

# ── Viz callbacks ──
viz_callbacks = []

# COSIA class names (for viz)
COSIA_CLASS_NAMES = [
    "Building", "Greenhouse", "Swimming pool", "Impervious", "Pervious",
    "Bare soil", "Water", "Snow", "Herbaceous veg", "Agricultural",
    "Plowed", "Vineyard", "Deciduous", "Coniferous", "Brushwood",
    "Clear cut", "Ligneous", "Mixed",
]

# MMEarth viz
if enable_mmearth and task in ("esa_worldcover", "all", "all_combined"):
    ds_attr = "train_dataset_esa" if use_multi_task else "train_dataset"
    viz_callbacks.append(PretrainSegVizCallback(
        task_name="esa_worldcover",
        dataset_attr=ds_attr,
        class_names=ESA_CLASS_NAMES,
        sample_indices=[0, 1, 2],
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
    ))

if enable_mmearth and task in ("dynamic_world", "all", "all_combined"):
    ds_attr = "train_dataset_dw" if use_multi_task else "train_dataset"
    viz_callbacks.append(PretrainSegVizCallback(
        task_name="dynamic_world",
        dataset_attr=ds_attr,
        class_names=DW_CLASS_NAMES,
        sample_indices=[0, 1, 2],
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
    ))

if task in ("reconstruction", "all", "all_combined", "flairhub_recon"):
    ds_attr = "train_dataset_recon" if (use_multi_task and enable_mmearth) else \
              "train_dataset_flair_recon" if (use_multi_task and enable_flairhub and not enable_mmearth) else \
              "train_dataset"
    viz_callbacks.append(PretrainReconVizCallback(
        dataset_attr=ds_attr,
        sample_indices=[0, 1, 2],
        log_every_n_epochs=1,
        use_wandb=True,
    ))

# FLAIR-HUB viz
if enable_flairhub and "cosia" in args.flairhub_tasks and task in ("flairhub_cosia", "all_flairhub", "all_combined"):
    viz_callbacks.append(PretrainSegVizCallback(
        task_name="flairhub_cosia",
        dataset_attr="train_dataset_cosia",
        class_names=COSIA_CLASS_NAMES,
        sample_indices=[0, 1, 2],
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
    ))

if enable_flairhub and "lpis" in args.flairhub_tasks and task in ("flairhub_lpis", "all_flairhub", "all_combined"):
    viz_callbacks.append(PretrainSegVizCallback(
        task_name="flairhub_lpis",
        dataset_attr="train_dataset_lpis",
        class_names=None,  # 23 classes, skip names for now
        sample_indices=[0, 1, 2],
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
    ))


checkpoint_resume = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}_{xp_name}-last-{{epoch:02d}}",
    every_n_epochs=1,
    save_top_k=1,       # keep only the latest
    save_last=True,      # also saves "last.ckpt" symlink
    verbose=True,
)

callbacks = [accumulator, checkpoint_val,checkpoint_resume, lr_monitor] + viz_callbacks

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    use_distributed_sampler=False,
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
    ckpt_path=args.ckpt_path
)

# =============================================================================
# TRAIN & TEST
# =============================================================================
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)

# =============================================================================
# MEASURE COMPLEXITY
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    """Recursively move a nested batch dict to device."""
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

    # Pick a dataset for complexity measurement
    if use_multi_task:
        if enable_mmearth:
            test_dataset = data_module.val_dataset_esa
            task_name = "esa_worldcover"
        elif hasattr(data_module, "val_dataset_cosia"):
            test_dataset = data_module.val_dataset_cosia
            task_name = "flairhub_cosia"
        else:
            test_dataset = data_module.val_dataset_flair_recon
            task_name = "reconstruction"
    else:
        test_dataset = data_module.test_dataset
        task_name = test_dataset.TASK_NAME

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_samples = min(25, len(test_dataset))
    samples = [test_dataset[i] for i in range(num_samples)]

    results = []

    for input_size in [128]:
        print(f"\nTesting input size: {input_size}x{input_size}")

        # Collate single sample into batch-of-1
        batch_0 = collate_grouped([samples[0]])
        batch_0 = _batch_to_device(batch_0, device)

        # Warmup — Model_Pretrain.forward needs (batch, task_name)
        with torch.no_grad():
            _ = model(batch_0, task_name, training=False)

        # FLOPs
        batch_1 = collate_grouped([samples[1]])
        batch_1 = _batch_to_device(batch_1, device)

        try:
            with torch.no_grad():
                flops = FlopCountAnalysis(model, (batch_1, task_name))
                gflops = flops.total() / 1e9
        except Exception as e:
            print(f"  FLOPs measurement failed: {e}")
            gflops = -1

        # Inference time
        num_warmup, num_runs = 3, 20
        with torch.no_grad():
            for i in range(num_warmup):
                b = collate_grouped([samples[i % num_samples]])
                b = _batch_to_device(b, device)
                _ = model(b, task_name, training=False)

            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_runs):
                idx = (i + num_warmup) % num_samples
                b = collate_grouped([samples[idx]])
                b = _batch_to_device(b, device)
                _ = model(b, task_name, training=False)
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

    # Summary
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