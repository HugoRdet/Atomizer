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
from fvcore.nn import FlopCountAnalysis
import time
from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO

seed_everything(42, workers=True)
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import argparse

from training.utils.token_building.processor import TokenProcessor
from training.utils.callbacks.token_assignement import TokenAssignmentCallbackSenFlood




# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--xp_name",       type=str, required=True, help="Experiment name")
parser.add_argument("--config_model",  type=str, required=True, help="Model config yaml file")
parser.add_argument("--dataset_name",  type=str, required=True, help="Name of the dataset used")
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml       = "./data/bands_info/bands.yaml"

is_unet = config_model["encoder"] == "UNET"

# =============================================================================
# LOOKUP TABLE & TRANSFORMS (needed for both, but processor only for Atomizer)
# =============================================================================
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

modalities_trans = modalities_transformations_config(
    configs_dataset, 
    model=config_model["encoder"], 
    name_config=args.dataset_name
)

input_processor = None
if not is_unet:
    input_processor = TokenProcessor(config_model, lookup_table)

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"],
        project="SenFlood",
        config=config_model
    )
    wandb_logger = WandbLogger(project="SenFlood")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
if is_unet:
    from training.unet.model_unet_senflood import Model_UNet_SenFlood
    model = Model_UNet_SenFlood(
        config=config_model,
        wand=True,
        name=xp_name,
    )
else:
    checkpoint_path = "./checkpoints/Atomiserxp_20260205_010115_lviu-val_loss-epoch=83-val_loss=0.0597.ckpt"
    # Option 1: Load checkpoint with strict=False (recommended)
    #model =  Model_SenFlood.load_from_checkpoint(
    #    checkpoint_path,
    #    strict=False,  # Allow missing keys (displacement MLP is new)
    #    config=config_model,
    #    wand=True,
    #    name=xp_name,
    #    transform=input_processor,
    #    lookup_table=lookup_table
    #)
    model = Model_SenFlood(
        config_model,
        wand=True,
        name=xp_name,
        transform=input_processor,
        lookup_table=lookup_table,
    )

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    f"./data/SENFLOOD",
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=4,
    trans_modalities=modalities_trans,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=Sen1Floods11Dataset,
)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor = LearningRateMonitor(logging_interval="step")
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_val_mod_train = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
)



# Atomizer-specific callbacks (U-Net doesn't need these)
#if not is_unet:
#    reconstruction_callback = MAE_err_CustomVisualizationCallback(config=config_model)
#    gravity_callback = ErrorLandscapeVisualizationCallback(config=config_model)
#    callbacks.extend([reconstruction_callback, gravity_callback])

# =============================================================================
# TRAINER
# =============================================================================

#tokens_ass_viz=TokenAssignmentCallbackSenFlood(
#        log_every_n_epochs=5,
#        sample_indices=[0],
#        save_dir="./viz_token_assignment",
#        use_wandb=True
#    )

#

callbacks = [accumulator, checkpoint_val_mod_train]

trainer = Trainer(
    strategy="ddp_find_unused_parameters_true" if not is_unet else "auto",
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
if os.environ.get("LOCAL_RANK", "0") == "0":
    print("\n" + "="*80)
    print("MEASURING MODEL COMPLEXITY")
    print("="*80 + "\n")
    
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    num_samples = min(25, len(test_dataset))
    samples = [test_dataset[i] for i in range(num_samples)]
    
    results = []
    
    for input_size in [512]:
        print(f"\nTesting input size: {input_size}x{input_size}")
        
        if is_unet:
            # ============= U-Net: simple image input =============
            image_0, label_0 = samples[0]
            image_0 = image_0.unsqueeze(0).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = model(image_0)
            
            # FLOPs
            image_1, _ = samples[1]
            image_1 = image_1.unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    flops = FlopCountAnalysis(model, (image_1,))
                    gflops = flops.total() / 1e9
            except Exception as e:
                print(f"FLOPs measurement failed: {e}")
                gflops = -1
            
            # Inference time
            num_warmup = 3
            num_runs = 20
            
            with torch.no_grad():
                for i in range(num_warmup):
                    img, _ = samples[i % num_samples]
                    _ = model(img.unsqueeze(0).to(device))
                
                torch.cuda.synchronize()
                start = time.time()
                for i in range(num_runs):
                    idx = (i + num_warmup) % num_samples
                    img, _ = samples[idx]
                    _ = model(img.unsqueeze(0).to(device))
                    torch.cuda.synchronize()
                end = time.time()
            
            avg_time_ms = (end - start) / num_runs * 1000
            num_tokens = 15 * input_size * input_size  # for comparison
            
        else:
            # ============= Atomizer: token input =============
            image_tokens, attention_mask, queries, queries_mask, label, latent_pos, image = samples[0]
            
            image_tokens = image_tokens.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            queries = queries.unsqueeze(0).to(device)
            queries_mask = queries_mask.unsqueeze(0).to(device)
            latent_pos_d = latent_pos.unsqueeze(0).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = model(image_tokens, attention_mask, queries, queries_mask, latent_pos_d)
            
            # FLOPs
            image_tokens_2, attention_mask_2, queries_2, queries_mask_2, _, latent_pos_2, _ = samples[1]
            image_tokens_2 = image_tokens_2.unsqueeze(0).to(device)
            attention_mask_2 = attention_mask_2.unsqueeze(0).to(device)
            queries_2 = queries_2.unsqueeze(0).to(device)
            queries_mask_2 = queries_mask_2.unsqueeze(0).to(device)
            latent_pos_d = latent_pos_2.unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    flops = FlopCountAnalysis(
                        model,
                        (image_tokens_2, attention_mask_2, queries_2, queries_mask_2, latent_pos_d)
                    )
                    gflops = flops.total() / 1e9
            except Exception as e:
                print(f"FLOPs measurement failed: {e}")
                gflops = -1
            
            # Inference time
            num_warmup = 3
            num_runs = 20
            
            with torch.no_grad():
                for i in range(num_warmup):
                    img_tok, att_mask, qry, qry_mask, _, lp, _ = samples[i % num_samples]
                    _ = model(
                        img_tok.unsqueeze(0).to(device),
                        att_mask.unsqueeze(0).to(device),
                        qry.unsqueeze(0).to(device),
                        qry_mask.unsqueeze(0).to(device),
                        lp.unsqueeze(0).to(device)
                    )
                
                torch.cuda.synchronize()
                start = time.time()
                for i in range(num_runs):
                    idx = (i + num_warmup) % num_samples
                    img_tok, att_mask, qry, qry_mask, _, lp, _ = samples[idx]
                    _ = model(
                        img_tok.unsqueeze(0).to(device),
                        att_mask.unsqueeze(0).to(device),
                        qry.unsqueeze(0).to(device),
                        qry_mask.unsqueeze(0).to(device),
                        lp.unsqueeze(0).to(device)
                    )
                    torch.cuda.synchronize()
                end = time.time()
            
            avg_time_ms = (end - start) / num_runs * 1000
            num_tokens = image_tokens.shape[1]
        
        results.append({
            "input_size": input_size,
            "gsd_target": 10,
            "num_tokens": num_tokens,
            "gflops": gflops,
            "inference_time_ms": avg_time_ms,
        })
        
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Inference time: {avg_time_ms:.2f} ms/tile")
    
    # Print summary
    print("\n" + "="*80)
    print(f"COMPLEXITY SUMMARY ({config_model['encoder']})")
    print("="*80)
    print(f"{'Input':<10} {'GSD':<8} {'GFLOPs':<12} {'Time (ms)':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['input_size']:<10} {r['gsd_target']:<8} {r['gflops']:<12.2f} {r['inference_time_ms']:<12.2f}")
    print("="*80)

# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)