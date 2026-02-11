"""
MADOS Training Script (v2)
==========================
Uses grouped-token batch format with multi-resolution groups (10m, 20m, 60m):
    batch = {
        "groups": {
            10.0: {"tokens": [B,N_10,8], "mask": [B,N_10], "shape": (4,240,240)},
            20.0: {"tokens": [B,N_20,8], "mask": [B,N_20], "shape": (5,120,120)},
            60.0: {"tokens": [B,N_60,8], "mask": [B,N_60], "shape": (1,40,40)},
        },
        "queries":      [B, M, 8],
        "queries_mask":  [B, M],
        "label":         [B, 240, 240],
        "target_resolution": 10.0,
        "image":         [B, 10, 240, 240],
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

# v2 classes (grouped token format)
from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.datasets.utils_dataset_MADOS import MADOSDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.token_grouping import collate_grouped

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="MADOS Training script")
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
        project="MADOS",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="MADOS")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
model = Model_SenFlood(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,           # encoder creates its own TokenProcessor
    lookup_table=lookup_table,
)




# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    path="./data/MADOS",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=None,        # v2 dataset handles tokenization internally
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=MADOSDataset,
)



# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_mIoU:.4f}}",
    monitor="val_mIoU",
    mode="max",
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
            out[k] = v  # scalars, tuples, strings — keep as-is
    return out


if os.environ.get("LOCAL_RANK", "0") == "0":
    from fvcore.nn import FlopCountAnalysis

    print("\n" + "=" * 80)
    print("MEASURING MODEL COMPLEXITY (MADOS)")
    print("=" * 80 + "\n")

    data_module.setup("test")
    test_dataset = data_module.test_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_samples = min(25, len(test_dataset))
    samples = [test_dataset[i] for i in range(num_samples)]

    results = []

    print(f"\nTesting MADOS (240×240 @ 10m, 120×120 @ 20m, 40×40 @ 60m)")

    # ============= Atomizer: grouped dict batch =============
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

    # Count tokens across all resolution groups
    total_tokens = 0
    tokens_per_res = {}
    for res, group in batch_0["groups"].items():
        n = group["tokens"].shape[1]
        tokens_per_res[res] = n
        total_tokens += n

    results.append({
        "total_tokens": total_tokens,
        "tokens_per_res": tokens_per_res,
        "gflops": gflops,
        "inference_time_ms": avg_time_ms,
    })

    # Summary
    print("\n" + "=" * 80)
    print(f"COMPLEXITY SUMMARY ({config_model['encoder']} — MADOS)")
    print("=" * 80)
    print(f"\n  Token counts per resolution:")
    for res in sorted(tokens_per_res.keys()):
        n = tokens_per_res[res]
        print(f"    {res:5.1f} m/px: {n:>8,} tokens")
    print(f"    {'Total':>10}: {total_tokens:>8,} tokens")
    print(f"\n  GFLOPs:         {gflops:.2f}")
    print(f"  Inference time: {avg_time_ms:.2f} ms/tile")
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