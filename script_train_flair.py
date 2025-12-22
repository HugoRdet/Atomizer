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
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--xp_name",       type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml       = "./data/bands_info/bands.yaml"

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
        project="FLAIR_seg_overfitting",
        config=config_model
    )
    wandb_logger = WandbLogger(project="FLAIR_seg_overfitting")

    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")
    

# 4. Instantiate Model
# We pass the input_processor where 'transform' used to go
# Ensure your Model_MAE/__init__ assigns self.input_processor = input_processor
# AND that Atomiser inside Model_MAE uses it.
model = Model_MAE(
    config_model,
    wand=True,
    name=xp_name,
    transform=input_processor, # Pass the new processor here,
    lookup_table=lookup_table
)



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
    dataset_class=FLAIR_MAE
)

reconstruction_callback = MAE_CustomVisualizationCallback(
    config=config_model
)

LR_finder=LearningRateFinder(min_lr=1e-05, max_lr=1, num_training_steps=450, mode='exponential', early_stop_threshold=4.0, update_attr=True, attr_name='')


lr_monitor = LearningRateMonitor(logging_interval="step")

checkpoint_val_mod_train = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
)

accumulator = GradientAccumulationScheduler(scheduling={0:1})

# Trainer
trainer = Trainer(
    strategy="ddp_find_unused_parameters_true",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=[accumulator, reconstruction_callback, checkpoint_val_mod_train],
    default_root_dir="./checkpoints/",
)

# Fit the model
trainer.fit(model, datamodule=data_module)

# Save wandb run ID
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)