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
from pytorch_lightning.callbacks import LearningRateMonitor
# ← new imports for the PyTorch profiler
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity

import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

seed_everything(42, workers=True)
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import argparse

from tqdm import tqdm

# instantiate the PyTorchProfiler
#profiler = PyTorchProfiler(
#    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#    record_shapes=True,
#    profile_memory=True,
#    export_to_chrome=True,       # dumps a trace.json for Chrome/TensorBoard
#    dirpath="profiling",         # where to write trace.json
#    filename="trace"             # will produce "profiling/trace.json"
#)


xp_name = "args.xp_name"
config_model = read_yaml("./training/configs/config_test-Atomiser_Atos_One.yaml")
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
bands_yaml       = "./data/bands_info/bands.yaml"
lookup_table=Lookup_encoding(read_yaml(configs_dataset),read_yaml(bands_yaml))
modalities_trans = modalities_transformations_config(configs_dataset,model=config_model["encoder"], name_config="regular")
test_conf=None
if config_model["encoder"] == "Atomiser_tradi":
    test_conf        = transformations_config_tradi(bands_yaml, config_model,lookup_table=lookup_table)
else:
    test_conf        = transformations_config(bands_yaml, config_model,lookup_table=lookup_table)

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"],
        project="MAE_debug",
        config=config_model
    )
    wandb_logger = WandbLogger(project="MAE_debug")
    

model = Model_FLAIR( #MODEL_MAE
    config_model,
    wand=True,
    name=xp_name,
    transform=test_conf
)

data_module = Tiny_BigEarthNetDataModule(
    f"./data/custom_flair/regular",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=FLAIR_SEG
)

# Setup the data module
data_module.setup()
train_dataset = data_module.train_dataset

print(f"Starting to iterate through {len(train_dataset)} training samples...")

# Keep track of errors
error_indices = []
error_details = []
successful_loads = 0

for idx in tqdm(range(50000,len(train_dataset))):
    try:
        # Try to load the sample
        sample = train_dataset[idx]
        successful_loads += 1
    except:
        print(idx)