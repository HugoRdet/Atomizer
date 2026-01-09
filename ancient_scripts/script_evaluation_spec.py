from training.perceiver import *
from training.utils import *
from training.losses import *
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

import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

seed_everything(42, workers=True)
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import argparse
from datetime import datetime

# Create the parser
parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("--run_id", type=str, help="WandB run id from training (optional)")
parser.add_argument("--xp_name", type=str, required=True, help="Experiment name")
parser.add_argument("--config_model", type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset used")
parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
run_id = args.run_id
xp_name = args.xp_name
config_model_path = args.config_model
config_name_dataset_force = args.dataset_name
use_wandb = not args.no_wandb

print("Using WandB Run ID:", run_id if run_id else "None (will create new run)")

# CRITICAL FIX: Monkey patch wandb.log to prevent errors
import wandb
original_wandb_log = wandb.log
original_wandb_init = wandb.init
original_wandb_finish = wandb.finish

# Global state tracking
WANDB_INITIALIZED = False
WANDB_AVAILABLE = False

def safe_wandb_init(*args, **kwargs):
    global WANDB_INITIALIZED, WANDB_AVAILABLE
    try:
        if not use_wandb:
            WANDB_INITIALIZED = False
            WANDB_AVAILABLE = False
            return None
        result = original_wandb_init(*args, **kwargs)
        WANDB_INITIALIZED = True
        WANDB_AVAILABLE = True
        print("✅ WandB initialized successfully")
        return result
    except Exception as e:
        print(f"❌ WandB init failed: {e}")
        WANDB_INITIALIZED = False
        WANDB_AVAILABLE = False
        return None

def safe_wandb_log(data_dict, *args, **kwargs):
    global WANDB_INITIALIZED, WANDB_AVAILABLE
    
    if not use_wandb or not WANDB_AVAILABLE or not WANDB_INITIALIZED:
        print(f"📊 Metrics (local): {data_dict}")
        return
    
    try:
        if wandb.run is not None:
            original_wandb_log(data_dict, *args, **kwargs)
        else:
            print(f"⚠️  WandB run not available: {data_dict}")
            WANDB_AVAILABLE = False
    except Exception as e:
        print(f"❌ WandB log failed: {e}")
        print(f"📊 Metrics (local): {data_dict}")
        WANDB_AVAILABLE = False

def safe_wandb_finish(*args, **kwargs):
    global WANDB_INITIALIZED, WANDB_AVAILABLE
    try:
        if WANDB_INITIALIZED and wandb.run is not None:
            result = original_wandb_finish(*args, **kwargs)
            WANDB_INITIALIZED = False
            WANDB_AVAILABLE = False
            return result
    except Exception as e:
        print(f"Warning: WandB finish error: {e}")
    finally:
        WANDB_INITIALIZED = False
        WANDB_AVAILABLE = False

# Apply the monkey patches
wandb.log = safe_wandb_log
wandb.init = safe_wandb_init
wandb.finish = safe_wandb_finish

# Helper function to handle loading of checkpoints with mismatched architectures
def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    
    # Get the state dictionary
    state_dict = ckpt["state_dict"]
    
    # Create a new state dict that only contains keys that exist in the model
    model_state_dict = model.state_dict()
    
    # Filter out unexpected keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load the filtered state dict
    missing_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # Print information about what was loaded and what was missed
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    
    print(f"Loaded {len(filtered_state_dict)} parameters")
    print(f"Missing {len(missing_keys.missing_keys)} parameters")
    print(f"Ignored {len(unexpected_keys)} unexpected parameters")
    
    if len(unexpected_keys) > 0:
        print("First few unexpected keys:", list(unexpected_keys)[:5])
    if len(missing_keys.missing_keys) > 0:
        print("First few missing keys:", missing_keys.missing_keys[:5])
    
    return model

def test_size_res_(config_model, modalities_trans, test_conf, ckpt, comment_log, modality_name, lookup_table):
    global WANDB_AVAILABLE
    
    #####
    #SIZE TESTED
    #####
    sizes_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model = Model_test_resolutions(config_model, wand=WANDB_AVAILABLE and use_wandb, name=xp_name, transform=test_conf, 
                                   resolutions=sizes_to_test, mode_eval="size", modality_name=modality_name)
    model = load_checkpoint(model, ckpt)
    model = model.float()
    model.comment_log = comment_log

    original_comment_log = model.comment_log
    model.comment_log = comment_log
    
    model.comment_log = "sizes - " + model.comment_log
    
    data_module = Tiny_BigEarthNetDataModule(
        f"./data/Tiny_BigEarthNet/regular",
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=4,
        trans_modalities=modalities_trans,
        trans_tokens=None,
        model=config_model["encoder"],
        dataset_config=read_yaml(bands_yaml),
        config_model=config_model,
        look_up=lookup_table,
        dataset_class=Tiny_BigEarthNet
    )

    # Create trainer with safe logger
    test_trainer = Trainer(
        accelerator="gpu",
        devices=[1],
        logger=wandb_logger if (WANDB_AVAILABLE and wandb_logger) else None,
        precision="bf16-mixed",
    )

    test_results_val_val = run_test(
        test_trainer, 
        model, 
        data_module
    )

    #####
    #RESOLUTION TESTED
    #####
    resolutions_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model = Model_test_resolutions(config_model, wand=WANDB_AVAILABLE and use_wandb, name=xp_name, transform=test_conf,
                                   resolutions=resolutions_to_test, modality_name=modality_name)
    model = load_checkpoint(model, ckpt)
    model = model.float()

    model.comment_log = comment_log
    model.comment_log = "res - " + model.comment_log
    
    data_module = Tiny_BigEarthNetDataModule(
        f"./data/Tiny_BigEarthNet/regular",
        batch_size=config_model["dataset"]["batchsize"],
        num_workers=4,
        trans_modalities=modalities_trans,
        trans_tokens=None,
        model=config_model["encoder"],
        dataset_config=read_yaml(bands_yaml),
        config_model=config_model,
        look_up=lookup_table,
        dataset_class=Tiny_BigEarthNet
    )

    # Create trainer with safe logger
    test_trainer = Trainer(
        accelerator="gpu",
        devices=[1],
        logger=wandb_logger if (WANDB_AVAILABLE and wandb_logger) else None,
        precision="bf16-mixed",
    )

    test_results_val_val = run_test(
        test_trainer, 
        model, 
        data_module
    )

    model.comment_log = original_comment_log

def generate_run_name(xp_name, config_model, config_name_dataset_force):
    """Generate a descriptive run name when no run_id is provided"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_name = config_model.get('encoder', 'unknown')
    dataset_name = config_name_dataset_force or 'default'
    return f"eval_{xp_name}_{encoder_name}_{dataset_name}_{timestamp}"

def setup_wandb(config_model, xp_name, run_id=None, config_name_dataset_force=None):
    """Set up W&B logging with error handling and auto-run creation"""
    global WANDB_INITIALIZED, WANDB_AVAILABLE
    
    if not use_wandb:
        print("W&B logging disabled via --no_wandb flag")
        return None
        
    if os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            # Set environment variables that might help with connection issues
            os.environ["WANDB_CONSOLE"] = "off"
            os.environ["WANDB_RECONNECT_ATTEMPTS"] = "5"
            
            # Determine run configuration
            if run_id:
                # Resume existing run
                print(f"Attempting to resume W&B run: {run_id}")
                run_name = None  # Will use existing run name
                resume_mode = "allow"
                run_id_to_use = run_id
            else:
                # Create new run
                run_name = generate_run_name(xp_name, config_model, config_name_dataset_force)
                print(f"Creating new W&B run: {run_name}")
                resume_mode = None
                run_id_to_use = None
            
            # Initialize wandb with more robust settings
            run = wandb.init(
                id=run_id_to_use,
                resume=resume_mode,
                name=run_name,
                project="Atomizer_BigEarthNet_DDDDDD",
                config=config_model,
                tags=["evaluation", config_model['encoder'], config_name_dataset_force or "default"],
                settings=wandb.Settings(
                    _service_wait=300,
                    _file_stream_buffer=8192,
                ),
                # Add job type to distinguish evaluation runs
                job_type="evaluation"
            )
            
            if run is not None:
                # Log additional metadata for evaluation runs
                wandb.config.update({
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "dataset_name": config_name_dataset_force,
                    "experiment_name": xp_name,
                    "resumed_from_id": run_id if run_id else None
                })
                
                # Create logger with the run
                logger = WandbLogger(project="Atomizer_BigEarthNet_DDDDDD", experiment=run)
                print(f"✅ W&B logging successfully initialized (Run ID: {run.id})")
                return logger
            else:
                print("❌ WandB init returned None")
                return None
            
        except Exception as e:
            print(f"❌ Error initializing wandb: {e}")
            print("Continuing without wandb logging")
            WANDB_INITIALIZED = False
            WANDB_AVAILABLE = False
            return None
    return None

def run_test(trainer, model, datamodule, ckpt_path=None):
    """Run the test with comprehensive error handling"""
    try:
        return trainer.test(
            model=model,
            datamodule=datamodule,
            verbose=True,
            ckpt_path=ckpt_path
        )
    except Exception as e:
        if "wandb" in str(e).lower() or "BrokenPipeError" in str(e):
            print(f"WandB-related error: {e}")
            print("Disabling WandB and retrying...")
            
            global WANDB_AVAILABLE, WANDB_INITIALIZED, use_wandb
            WANDB_AVAILABLE = False
            WANDB_INITIALIZED = False
            use_wandb = False
            
            # Create new trainer without wandb
            new_trainer = Trainer(
                strategy=trainer.strategy if hasattr(trainer, 'strategy') else "auto",
                devices=trainer.num_devices if hasattr(trainer, 'num_devices') else -1,
                accelerator=trainer._accelerator_connector.accelerator_flag if hasattr(trainer, '_accelerator_connector') else "gpu",
                precision=trainer.precision if hasattr(trainer, 'precision') else "bf16-mixed",
                logger=None,
                limit_test_batches=getattr(trainer, 'limit_test_batches', None)
            )
            
            return new_trainer.test(
                model=model,
                datamodule=datamodule,
                verbose=True,
                ckpt_path=ckpt_path
            )
        else:
            raise

seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

# Read configuration files (updated to match training script structure)
config_model = read_yaml("./training/configs/" + config_model_path)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

# Create lookup table (added to match training script)
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml))

# Set up modalities transformations (updated to match training script)
modalities_trans = modalities_transformations_config(configs_dataset, model=config_model["encoder"], 
                                                    name_config="regular",force_modality=config_name_dataset_force)

# Set up test configuration (updated to handle different encoder types like training script)
test_conf = None
if config_model["encoder"] == "Atomiser_tradi":
    test_conf = transformations_config_tradi(bands_yaml, config_model, lookup_table=lookup_table)
else:
    test_conf = transformations_config(bands_yaml, config_model, lookup_table=lookup_table)

# Initialize W&B if enabled
wandb_logger = None
if use_wandb:
    wandb_logger = setup_wandb(config_model, xp_name, run_id, config_name_dataset_force)

checkpoint_dir = "./checkpoints"
all_ckpt_files = [
    os.path.join(checkpoint_dir, f)
    for f in os.listdir(checkpoint_dir)
    if f.endswith(".ckpt")
]
if not all_ckpt_files:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

def latest_ckpt_for(prefix: str):
    # filter to files that start with the prefix,
    # then pick the most recently‐modified one
    matches = [f for f in all_ckpt_files if os.path.basename(f).startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"No checkpoints matching {prefix}* in {checkpoint_dir}")
    return max(matches, key=os.path.getmtime)

# Find the best model checkpoint (updated filename pattern to match training script)
prefix = f"Atomiserxp_20250715_152823_skrk-alone_best_model_val_mod_train"
#Atomiserxp_20250715_152823_skrk-alone_best_model_val_mod_train-epoch=19-val_mod_train_ap=65.0766
if config_model["encoder"] == "ScaleMAE":
    prefix = "ScaleMAExp_20250530_142929"
    
if config_model["encoder"] == "ResNetSuperSmall":
    prefix="ResNetSuperSmall-best_model_val_mod_train-epoch=73-val_mod_train_ap=63.5750.ckpt"
    
if config_model["encoder"] == "ViT":
    prefix="ViT-best_model_val_mod_train-epoch=42-val_mod_train_ap=55.7395.ckpt"

    
ckpt_train = latest_ckpt_for(prefix)
print("→ Testing on ckpt (val_mod_train):", ckpt_train)

# Set up data module for testing (updated to match training script structure)
data_module = Tiny_BigEarthNetDataModule(
    f"./data/Tiny_BigEarthNet/regular",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=Tiny_BigEarthNet
)

# Create trainer (updated to match training script settings)
test_trainer = Trainer(
    strategy="ddp",
    devices=-1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger if (WANDB_AVAILABLE and wandb_logger) else None,
    #limit_test_batches=300
)

# Test with the "train‐best" checkpoint
print("\n===== Testing model from train-best checkpoint on test data =====")
model = Model(config_model, wand=WANDB_AVAILABLE and use_wandb, name=xp_name, transform=test_conf)
model = load_checkpoint(model, ckpt_train)
model = model.float()

# Uncomment if you want to test size/resolution variations
# test_size_res_(config_model, modalities_trans, test_conf, ckpt_train, "train_best", 
#                modality_name=config_name_dataset_force, lookup_table=lookup_table)

model.comment_log = f"{config_name_dataset_force} "

test_results_train = run_test(
    test_trainer, 
    model, 
    data_module
)

# Clean up wandb
print("🧹 Cleaning up WandB...")
try:
    if WANDB_INITIALIZED and wandb.run is not None:
        print(f"Final W&B run ID: {wandb.run.id}")
        wandb.finish()
except Exception as e:
    print(f"Warning during WandB cleanup: {e}")
finally:
    WANDB_INITIALIZED = False
    WANDB_AVAILABLE = False

print("✅ Evaluation completed!")