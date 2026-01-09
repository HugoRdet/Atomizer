from training.perceiver import *
from training.utils import *
from training.losses import *
from training.utils.callbacks import *
from training.utils.datasets import*
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
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity
from pytorch_lightning.callbacks import LearningRateFinder
import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO

seed_everything(42, workers=True)
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import argparse

# --- NEW IMPORTS ---
# Import the new TokenProcessor from your refactored module
from training.utils.token_building.processor import TokenProcessor

# Create the parser
parser = argparse.ArgumentParser(description="Execute visualization callback")
parser.add_argument("--xp_name",       type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml       = "./data/bands_info/bands.yaml"

# Hardcoded checkpoint path
checkpoint_path = "./checkpoints/Atomiserxp_20260108_094816_o2w4-val_loss-epoch=06-val_loss=0.2000.ckpt"


# 1. Initialize Lookup Table (Kept as is, assuming it handles modality indices)
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

# 2. Modalities Transformations (Data Augmentation stuff)
modalities_trans = modalities_transformations_config(
    configs_dataset, 
    model=config_model["encoder"], 
    name_config=args.dataset_name
)

# 3. Initialize the New Processor (Replaces transformations_config)
# The TokenProcessor handles all encoding logic (Physics + Math)
# It takes the full config and the lookup table.
input_processor = TokenProcessor(config_model, lookup_table)


wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"],
        project="dynamic_sys",
        config=config_model
    )
    wandb_logger = WandbLogger(project="dynamic_sys")

    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")
    

# Load model from checkpoint
print(f"Loading model from checkpoint: {checkpoint_path}")
model = Model_MAE_err.load_from_checkpoint(
    checkpoint_path,
    strict=False,  # Allow missing keys (displacement MLP is new)
    config=config_model,
    wand=True,
    name=xp_name,
    transform=input_processor,
    lookup_table=lookup_table
)
model.eval()
print("✓ Model loaded successfully")


data_module = UnifiedDataModule(
    f"./data/custom_flair/{args.dataset_name}",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=FLAIR_MAE_err
)

reconstruction_callback = MAE_err_CustomVisualizationCallback(
    config=config_model
)

# Trainer - simplified for callback execution only
trainer = Trainer(
    devices=1,  # Use single device for visualization
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    callbacks=[reconstruction_callback],
    default_root_dir="./checkpoints/",
    limit_val_batches=1,
)

# Run validation loop to initialize everything
print("Running validation to execute callback...")
trainer.validate(model, datamodule=data_module)

# Directly call the callback's visualization method to bypass epoch checks
# (callback normally checks epoch >= 2 and epoch % log_every_n_epochs == 0)
print("Executing callback visualization...")
if hasattr(trainer, 'is_global_zero') and trainer.is_global_zero:
    try:
        reconstruction_callback._perform_custom_reconstruction(trainer, model)
        print("✓ Callback visualization completed!")
    except Exception as e:
        print(f"Error executing callback: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ Not on global rank 0, skipping visualization")

print("✓ Callback execution completed!")

# Save wandb run ID
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}_viz.txt", "w") as f:
        f.write(run_id)