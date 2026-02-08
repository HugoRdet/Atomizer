"""
Sweep-compatible training script for Atomizer on SenFlood.
Usage:
  1) Create the sweep:    wandb sweep sweep_config.yaml
  2) Launch agent(s):     wandb agent <SWEEP_ID>
     (launch multiple agents on different GPUs for parallelism)
"""

from training.perceiver import *
from training.utils import *
from training.losses import *
from training.utils.callbacks import *
from training.utils.datasets import *
from training.VIT import *
from training.ResNet import *
from collections import defaultdict
from training import *
import os
import copy
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
import argparse

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO

seed_everything(42, workers=True)
import wandb

from training.utils.token_building.processor import TokenProcessor
from training.utils.callbacks.token_assignement import TokenAssignmentCallbackSenFlood


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Sweep training script")
parser.add_argument("--xp_name",       type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()


# =============================================================================
# LOAD BASE CONFIG & APPLY SWEEP OVERRIDES
# =============================================================================
config_model = read_yaml("./training/configs/config_test-Atomiser_Atos_One.yaml" )
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

xp_name = args.xp_name


def apply_sweep_overrides(config, sweep_params):
    """Override config values with sweep parameters from wandb."""
    c = copy.deepcopy(config)

    if "lr" in sweep_params:
        c["trainer"]["lr"] = sweep_params["lr"]
    if "weight_decay" in sweep_params:
        c["trainer"]["weight_decay"] = sweep_params["weight_decay"]

    # Atomiser architecture
    if "ff_dropout" in sweep_params:
        c["Atomiser"]["ff_dropout"] = sweep_params["ff_dropout"]
    if "attn_dropout" in sweep_params:
        c["Atomiser"]["attn_dropout"] = sweep_params["attn_dropout"]
    if "depth" in sweep_params:
        c["Atomiser"]["depth"] = sweep_params["depth"]
        c["Atomiser"]["stable_depth"] = sweep_params["depth"]  # keep in sync
    if "self_per_cross_attn" in sweep_params:
        c["Atomiser"]["self_per_cross_attn"] = sweep_params["self_per_cross_attn"]
    if "latent_dim" in sweep_params:
        c["Atomiser"]["latent_dim"] = sweep_params["latent_dim"]
    if "global_latents" in sweep_params:
        c["Atomiser"]["global_latents"] = sweep_params["global_latents"]
    if "masking" in sweep_params:
        c["Atomiser"]["masking"] = sweep_params["masking"]
    if "decoder_k_spatial" in sweep_params:
        c["Atomiser"]["decoder_k_spatial"] = sweep_params["decoder_k_spatial"]
    if "lambda_error" in sweep_params:
        c["Atomiser"]["lambda_error"] = sweep_params["lambda_error"]

    if "latents_senflood" in sweep_params:
        c["latent_grids"]["SENFLOOD"]["latents"] = sweep_params["latents_senflood"]
    if "hexagonal" in sweep_params:
        c["latent_grids"]["SENFLOOD"]["hexagonal"] = sweep_params["hexagonal"]

    return c


# =============================================================================
# MAIN TRAINING FUNCTION (called by wandb agent)
# =============================================================================
def train():
    # Init wandb run — sweep agent populates wandb.config
    run = wandb.init(project="Atomiser_BigEarthNet")
    sweep_params = dict(wandb.config)

    # Apply overrides
    config = apply_sweep_overrides(config_model, sweep_params)

    # Update wandb with full config for reproducibility
    wandb.config.update({"full_config": config}, allow_val_change=True)

    wandb_logger = WandbLogger(project="Atomiser_BigEarthNet", experiment=run)

    # ---- Lookup & transforms ----
    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset), read_yaml(bands_yaml), config
    )
    modalities_trans = modalities_transformations_config(
        configs_dataset, model=config["encoder"], name_config=args.dataset_name
    )
    input_processor = TokenProcessor(config, lookup_table)

    # ---- Model ----
    model = Model_SenFlood(
        config,
        wand=True,
        name=xp_name,
        transform=input_processor,
        lookup_table=lookup_table,
    )

    # ---- Data ----
    data_module = UnifiedDataModule(
        f"./data/SENFLOOD",
        batch_size=config["dataset"]["batchsize"],
        num_workers=4,
        trans_modalities=modalities_trans,
        trans_tokens=None,
        model=config["encoder"],
        dataset_config=read_yaml(bands_yaml),
        config_model=config,
        look_up=lookup_table,
        dataset_class=Sen1Floods11Dataset,
    )

    # ---- Callbacks ----
    lr_monitor = LearningRateMonitor(logging_interval="step")
    accumulator = GradientAccumulationScheduler(scheduling={0: 1})



    callbacks = [accumulator, lr_monitor]

    # ---- Trainer ----
    trainer = Trainer(
        strategy="auto",
        devices=1,
        max_epochs=config["trainer"]["epochs"],
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir="./checkpoints/sweeps/",
    )

    # ---- Train ----
    trainer.fit(model, datamodule=data_module)

    # ---- Test & log mIoU ----
    test_results = trainer.test(model, datamodule=data_module)

    if test_results:
        test_dict = test_results[0]
        miou = test_dict.get("test_mIoU")
        if miou is not None:
            wandb.log({"test_mIoU": miou})
            wandb.summary["test_mIoU"] = miou
            print(f"\n{'='*60}")
            print(f"SWEEP RUN RESULT — test_mIoU: {miou:.4f}")
            print(f"{'='*60}\n")
        else:
            print(f"WARNING: Could not find mIoU in test results: {test_dict.keys()}")
            for k, v in test_dict.items():
                wandb.summary[k] = v

    # ---- Measure & log GFLOPs ----
    try:
        from fvcore.nn import FlopCountAnalysis
        import time

        data_module.setup("test")
        test_dataset = data_module.test_dataset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Grab one sample for FLOPs measurement
        image_tokens, attention_mask, queries, queries_mask, label, latent_pos, image = test_dataset[0]
        image_tokens = image_tokens.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        queries = queries.unsqueeze(0).to(device)
        queries_mask = queries_mask.unsqueeze(0).to(device)
        latent_pos_d = latent_pos.unsqueeze(0).to(device)

        with torch.no_grad():
            # Warmup
            _ = model(image_tokens, attention_mask, queries, queries_mask, latent_pos_d)

            # FLOPs
            flops = FlopCountAnalysis(
                model,
                (image_tokens, attention_mask, queries, queries_mask, latent_pos_d)
            )
            gflops = flops.total() / 1e9

            # Inference time (average over 10 runs)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = model(image_tokens, attention_mask, queries, queries_mask, latent_pos_d)
                torch.cuda.synchronize()
            avg_time_ms = (time.time() - start) / 10 * 1000

        wandb.summary["GFLOPs"] = gflops
        wandb.summary["inference_time_ms"] = avg_time_ms
        wandb.summary["num_tokens"] = image_tokens.shape[1]
        print(f"GFLOPs: {gflops:.2f} | Inference: {avg_time_ms:.2f} ms")

    except Exception as e:
        print(f"GFLOPs measurement failed: {e}")
        wandb.summary["GFLOPs"] = -1

    wandb.finish()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    train()