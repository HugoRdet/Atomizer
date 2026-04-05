"""
MNIST Classification — Training Script
========================================
Tests the new multi-res Atomiser on MNIST digit classification.
Uses new-format dataset (8-column tokens, batch dict output).

Usage:
    python train_mnist.py
"""

import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_MNIST import Model_MNIST
from training.utils.datasets.utils_dataset_MNIST import MNISTSparseCanvas
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# COLLATE
# =============================================================================

def mnist_collate(samples):
    """
    Collate MNIST dict samples into a batched dict.

    Each sample:
        {"groups": {0.2: {"tokens": [N,8], "mask": [N], "shape": (H,W)}},
         "queries": [M,8], "queries_mask": [M], "label": scalar}

    Batched output:
        {"groups": {0.2: {"tokens": [B,N,8], "mask": [B,N], "shape": (H,W)}},
         "queries": [B,M,8], "queries_mask": [B,M], "label": [B]}
    """
    # Get all resolution keys
    all_res = sorted(samples[0]["groups"].keys())

    # Stack groups
    groups = {}
    for res in all_res:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in samples]),
            "mask": torch.stack([s["groups"][res]["mask"] for s in samples]),
            "shape": samples[0]["groups"][res]["shape"],
        }

    return {
        "groups": groups,
        "queries": torch.stack([s["queries"] for s in samples]),
        "queries_mask": torch.stack([s["queries_mask"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
    }


# =============================================================================
# CONFIG
# =============================================================================

xp_name = "mnist_classification"

config_model = read_yaml("./training/configs/config_test-Atomiser_Atos_One.yaml")

# MNIST overrides
config_model["trainer"]["max_tokens"] = 784
config_model["trainer"]["max_tokens_reconstruction"] = 784

# Latent grid: 1 latent per token → 784 latents (28×28)
if "latent_grid" not in config_model:
    config_model["latent_grid"] = {}
config_model["latent_grid"]["tokens_per_latent"] = 784
config_model["latent_grid"]["sigma_factor"] = 1.5
config_model["latent_grid"]["max_k"] = 784

# =============================================================================
# LOOKUP TABLE
# =============================================================================

bands_yaml = "./data/bands_info/bands.yaml"
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
lookup_table = Lookup_encoding(
    read_yaml(configs_dataset), read_yaml(bands_yaml), config_model
)

# Register MNIST resolution (0.2 m/px, 28×28 canvas)
TokenBuilder.REFERENCE_SIZES[0.2] = 28
lookup_table.get_or_register_modality(0.2, 28)
lookup_table.get_resolution_idx(0.2)

# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(name=xp_name, project="MNIST_classification", config=config_model)
    wandb_logger = WandbLogger(project="MNIST_classification")

# =============================================================================
# DATASETS
# =============================================================================

train_dataset = MNISTSparseCanvas(
    mode="train", config_model=config_model, look_up=lookup_table,
)
val_dataset = MNISTSparseCanvas(
    mode="val", config_model=config_model, look_up=lookup_table,
)

batch_size = config_model["dataset"]["batchsize"]

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, collate_fn=mnist_collate, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, collate_fn=mnist_collate, pin_memory=True,
)

# =============================================================================
# MODEL
# =============================================================================

model = Model_MNIST(
    config=config_model, wand=True, name=xp_name,
    transform=None, lookup_table=lookup_table,
)

# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================

callbacks = [
    LearningRateMonitor(logging_interval="step"),
    GradientAccumulationScheduler(scheduling={0: 1}),
    ModelCheckpoint(
        dirpath="./checkpoints/mnist/",
        filename=f"{xp_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss", mode="min", save_top_k=1, verbose=True,
    ),
]

trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    logger=wandb_logger,
    strategy="ddp_find_unused_parameters_true",
    max_epochs=config_model["trainer"]["epochs"],
    callbacks=callbacks,
    enable_checkpointing=True,
    log_every_n_steps=10,
)

# =============================================================================
# TRAIN
# =============================================================================

print("=" * 60)
print("MNIST Classification — New Atomiser (RoPE test)")
print("=" * 60)
print(f"  Canvas:  28×28, 1 band")
print(f"  Tokens:  784")
print(f"  tokens_per_latent: {config_model['latent_grid']['tokens_per_latent']}")
print(f"  Latents: ~784 (1:1 mapping)")
print(f"  max_k:   {config_model['latent_grid']['max_k']}")
print(f"  Batch:   {batch_size}")
print(f"  Epochs:  {config_model['trainer']['epochs']}")
print("=" * 60)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# =============================================================================
# SAVE RUN ID
# =============================================================================

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)

print("Training complete!")