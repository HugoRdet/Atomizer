from training.perceiver import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ResNet import *
from collections import defaultdict
from training import *
import os
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.loggers import WandbLogger

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





seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

config_dico = read_yaml("./training/configs/config_test-ViT_XS.yaml")
config_name_dataset="loader"

bands_yaml = "./data/Tiny_BigEarthNet/bands.yaml"
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_"+config_name_dataset+".yaml"

test_conf = transformations_config(bands_yaml, configs_dataset, path_imgs_config="./data/Tiny_BigEarthNet/", name_config=config_name_dataset)

# Initialize DataModule but do NOT call setup() manually here.
data_module = Tiny_BigEarthNetDataModule(
    "./data/Tiny_BigEarthNet/"+config_name_dataset,
    batch_size=config_dico['dataset']['batchsize'],
    num_workers=4,
    trans_config=test_conf,
    model=config_dico["encoder"]
)


wand = True
wandb_logger = None
if wand:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            name=get_xp_name(config_dico['encoder']) + " modalities",
            project=config_name_dataset+"_modalities",
            config=config_dico
        )
        wandb_logger = WandbLogger(project="first_xp")
        print("===== ",wandb.run.id," =====")


model = Model(config_dico, wand=wand, name="ATOS_test")

# Configure the trainer for distributed training.
trainer = Trainer(
    use_distributed_sampler=False,  # we use our custom sampler
    strategy="ddp",
    max_epochs=config_dico["trainer"]["epochs"],
    logger=wandb_logger,
    log_every_n_steps=1,
    devices=-1,
    accelerator="gpu"
)

# Fit the model. Pass the data module instead of dataloaders directly.
trainer.fit(model, datamodule=data_module)

test_trainer = Trainer(
    use_distributed_sampler=False,
    max_epochs=config_dico["trainer"]["epochs"],
    logger=wandb_logger,
    log_every_n_steps=1,
    devices=1,
    num_nodes=1,
    accelerator="gpu"
)

if dist.is_initialized():
    dist.destroy_process_group()

data_module = Tiny_BigEarthNetDataModule(
    "./data/Tiny_BigEarthNet/"+config_name_dataset,
    batch_size=config_dico['dataset']['batchsize'],
    num_workers=4,
    trans_config=test_conf,
    model=config_dico["encoder"],
    modality="test"
)
    
model = Model(config_dico, wand=wand, name="ATOS_test")
model.load_model(name="best_val_ap")
model.table=True
model.mode = "best_val_ap"
test_trainer.validate(model, datamodule=data_module)
test_trainer.test(model, datamodule=data_module)

data_module = Tiny_BigEarthNetDataModule(
    "./data/Tiny_BigEarthNet/"+config_name_dataset,
    batch_size=config_dico['dataset']['batchsize'],
    num_workers=4,
    trans_config=test_conf,
    model=config_dico["encoder"],
    modality="validation"
)

# Lightning will call setup() on each process appropriately.
# Create a new model or load the best checkpoint for testing.
model.load_model(name="best_val_ap")
model.mode = "best_val_ap"
test_trainer.validate(model, datamodule=data_module)
test_trainer.test(model, datamodule=data_module)
