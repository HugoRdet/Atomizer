"""
MMEarth MAE Training Script
============================
Uses grouped-token batch format:
    batch = {
        "groups": {res: {"tokens": [B,N,8], "mask": [B,N], "shape": (C,H,W)}},
        "queries":      [B, M, 8],
        "queries_mask":  [B, M],
    }
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

from training.trainer_MAE import Model_MMEarth
from training.utils.datasets.utils_dataset_SENFLOOD import Sen1Floods11Dataset
from training.utils.datasets.utils_dataset_MM_Earth import MMEarthMAEDataset
from training.utils.datasets.utils_dataset_PASTIS_SPOT import PastisSpotReconDataset
from training.utils.datasets.utils_dataset_PASTIS_S2 import PastisS2ReconDataset

from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.token_grouping import collate_grouped
from training.utils.callbacks.mae_visualization_callback import ReconstructionVizCallback
from training.utils.callbacks.token_assignement import TokenAssignmentCallbackSenFlood

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="MMEarth MAE Training")
parser.add_argument("--xp_name",       type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
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
        name=config_model["encoder"],
        project="PASTIS_SPOT",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="PASTIS_SPOT")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
model = Model_MMEarth(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
)

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    path="./data/PASTIS-HD",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=PastisS2ReconDataset#MMEarthMAEDataset#PastisSpotReconDataset#,###PastisS2ReconDataset##
)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_mse-{{epoch:02d}}-{{val_mse:.4f}}",
    monitor="val_mse",
    mode="min",
    save_top_k=1,
    verbose=True,
)





callbacks = [accumulator, checkpoint_val, lr_monitor]

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
    #val_check_interval=250,
    limit_val_batches=40
    limit_train_batches=750
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
    test_dataset = data_module.test_dataset

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

        # Warmup
        with torch.no_grad():
            _ = model(batch_0)

        # FLOPs
        batch_1 = collate_grouped([samples[1]])
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
                b = collate_grouped([samples[i % num_samples]])
                b = _batch_to_device(b, device)
                _ = model(b)

            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_runs):
                idx = (i + num_warmup) % num_samples
                b = collate_grouped([samples[idx]])
                b = _batch_to_device(b, device)
                _ = model(b)
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