#%%
"""
Training script for MNIST Sparse Canvas latent movement experiment.

Tests whether Perceiver latents can learn to "move" toward sparse signal
(a single digit on a large black canvas).
"""
from pytorch_lightning.strategies import DDPStrategy
from training.perceiver import *
from training.utils import *
from training.losses import *
from training.utils.callbacks import *
from training.VIT import *
from training.ResNet import *
from collections import defaultdict
from training import *
import os
from sklearn.metrics import average_precision_score
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import LearningRateFinder
import matplotlib.pyplot as plt
import argparse

# Import MNIST-specific components
from training.utils.utils_dataset_MNIST import MNISTSparseCanvas
from training.utils.dataloaders import UnifiedDataModule
from training.utils.callbacks import *

seed_everything(42, workers=True)

# =============================================================================
# Configuration
# =============================================================================

xp_name = "mnist_latent_movement"

# Load base config and modify for MNIST experiment
config_model = read_yaml("./training/configs/config_test-Atomiser_Atos_One.yaml")

# Override config for MNIST experiment
config_model["dataset"]["batchsize"] = 8  # Smaller batch due to large token count
config_model["trainer"]["max_tokens"] = 262144  # 512*512
config_model["trainer"]["max_tokens_reconstruction"] = 65536  # 256*256 queries
config_model["Atomiser"]["spatial_latents"] = 4  # 4x4 = 16 latents

# Debug/visualization config
config_model["debug"] = {
    "viz_every_n_epochs": 5,
    "idxs_to_viz": [0, 1, 2, 3],  # Visualize first 4 samples
}

# MNIST-specific dataset config
mnist_config = {
    "canvas_size": 64,
    "min_digit_size": 64,
    "max_digit_size": 64,
    "num_samples_train": 10000,  # Use subset for faster iteration
    "num_samples_val": 1000,
    "fixed_position": False,  # Random position (set True for ablation)
}

# =============================================================================
# Lookup table (simplified for MNIST - single band)
# =============================================================================

# For MNIST, we use a simplified lookup or None
# The dataset handles coordinate encoding internally
lookup_table = None

# If you need wavelength lookup for compatibility:
bands_yaml = "./data/bands_info/bands.yaml"
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

# =============================================================================
# Wandb Logger
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=xp_name,
        project="MNIST_latent_movement",
        config={
            **config_model,
            "mnist_config": mnist_config,
        }
    )
    wandb_logger = WandbLogger(project="MNIST_latent_movement")

# =============================================================================
# Model
# =============================================================================

# Transformation config (simplified for MNIST)
test_conf = None  # MNIST dataset handles its own tokenization

model = Model_MNIST(
    config_model,
    wand=True,
    name=xp_name,
    transform=test_conf,
    lookup_table=lookup_table
)

# =============================================================================
# Data Module
# =============================================================================

data_module = UnifiedDataModule(
    dataset_class=MNISTSparseCanvas,
    config_model=config_model,
    look_up=lookup_table,
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,  # MNIST doesn't need many workers
    # MNIST-specific params
)

# =============================================================================
# Callbacks
# =============================================================================

# MNIST visualization callback (reconstruction + trajectory)
visualization_callback = MNISTVisualizationCallback(config=config_model)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval="step")

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints/mnist/",
    filename=f"{xp_name}-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
)

# Gradient accumulation (optional)
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

# =============================================================================
# Trainer
# =============================================================================

trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    max_epochs=10,
    logger=wandb_logger,
    callbacks=[
        visualization_callback,
        lr_monitor,
        checkpoint_callback,
        accumulator,
    ],
    enable_checkpointing=True,
    enable_model_summary=True,
    log_every_n_steps=10,
    val_check_interval=1.0,  # Validate every epoch
    # Debugging options (uncomment for quick testing)
    # fast_dev_run=True,
    #limit_train_batches=10,
    #limit_val_batches=5,
)

# =============================================================================
# Training
# =============================================================================

print("=" * 60)
print("MNIST Sparse Canvas - Latent Movement Experiment")
print("=" * 60)
print(f"Canvas size: {mnist_config['canvas_size']}x{mnist_config['canvas_size']}")
print(f"Digit size range: {mnist_config['min_digit_size']}-{mnist_config['max_digit_size']}")
print(f"Number of latents: {config_model['Atomiser']['spatial_latents']**2}")
print(f"Train samples: {mnist_config['num_samples_train']}")
print(f"Val samples: {mnist_config['num_samples_val']}")
print(f"Batch size: {config_model['dataset']['batchsize']}")
print("=" * 60)

# Fit the model
trainer.fit(model, datamodule=data_module)

# =============================================================================
# Save run info
# =============================================================================

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print(f"WANDB_RUN_ID: {run_id}")
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
    
    # Log final summary
    wandb.log({
        "experiment/canvas_size": mnist_config["canvas_size"],
        "experiment/num_latents": config_model["Atomiser"]["spatial_latents"] ** 2,
        "experiment/digit_size_range": f"{mnist_config['min_digit_size']}-{mnist_config['max_digit_size']}",
    })

print("Training complete!")