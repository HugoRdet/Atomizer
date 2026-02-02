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


def print_memory(label: str):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{label}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Max: {max_allocated:.2f} GB")

def reset_memory_stats():
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        
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
model = Model_MAE_err( #Model_FLAIR
    config_model,
    wand=True,
    name=xp_name,
    transform=input_processor, # Pass the new processor here,
    lookup_table=lookup_table,
)

#checkpoint = torch.load("./checkpoints/Atos_tofine.ckpt", map_location="cpu")
#state_dict = checkpoint["state_dict"]

#mismatched_keys = [
#    "transform.geometry.token_centers_lookup",
#    "transform.geometry.token_gsd_lookup",
#    "encoder.input_processor.geometry.token_centers_lookup",
#    "encoder.input_processor.geometry.token_gsd_lookup"
#]

# 3. Remove them from the state_dict
#for key in mismatched_keys:
#    if key in state_dict:
#        print(f"Removing {key} from checkpoint due to size mismatch.")
#        del state_dict[key]

# 4. Save the modified checkpoint to a temporary file or load directly if the method allows
# Most Lightning-based models prefer a path, so we save it back:
#temp_path = "./checkpoints/modified_checkpoint.ckpt"
#torch.save(checkpoint, temp_path)

checkpoint_path = "./checkpoints/Atomiserxp_20260129_161407_2fj2-val_loss-epoch=52-val_loss=0.0422.ckpt"
# Option 1: Load checkpoint with strict=False (recommended)
#model = Model_MAE_err.load_from_checkpoint(
#    checkpoint_path,
#    strict=False,  # Allow missing keys (displacement MLP is new)
#    config=config_model,
#    wand=True,
#    name=xp_name,
#    transform=input_processor,
#    lookup_table=lookup_table
#)

def load_checkpoint_with_rope_reset(model, checkpoint_path):
    """Load checkpoint but reinitialize RoPE parameters."""
    
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    
    # Find and remove RoPE-related keys
    rope_keys = [k for k in state_dict.keys() if any(x in k for x in [
        'rope.inv_freq',
        'rope.scale_x', 
        'rope.scale_y',
        'rope.log_ref_scale',
        'log_sigma',  # Also reset Gaussian sigma if using combined
    ])]
    
    print(f"Resetting RoPE parameters: {rope_keys}")
    for k in rope_keys:
        del state_dict[k]
    
    # Load remaining weights (strict=False allows missing keys)
    model.load_state_dict(state_dict, strict=False)
    
    # RoPE parameters are now freshly initialized from model.__init__
    return model


#load_checkpoint_with_rope_reset(model, checkpoint_path)

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

reconstruction_callback = MAE_err_CustomVisualizationCallback( #FLAIR_CustomSegmentationCallback
    config=config_model
)



gravity_callback=ErrorLandscapeVisualizationCallback(config=config_model)

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
#gradient_warmup = DisplacementGradientWarmupCallback(start_epoch=10, warmup_epochs=10)

profiler = PyTorchProfiler(
    dirpath="./profiling/", 
    filename="perf_logs", 
    export_to_chrome=True  # This ensures the .json is created
)

# Trainer
trainer = Trainer(
    strategy="ddp_find_unused_parameters_true",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=[accumulator, reconstruction_callback,gravity_callback, checkpoint_val_mod_train],
    default_root_dir="./checkpoints/",
    #profiler=profiler
    #overfit_batches=1
    #limit_train_batches=1,
    #limit_val_batches=1,
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