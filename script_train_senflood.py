"""
Sen1Floods11 Training Script (v2)
=================================
Uses grouped-token batch format:
    batch = {
        "groups": {res: {"tokens": [B,N,6], "mask": [B,N], "shape": (C,H,W)}},
        "queries":      [B, M, 6],
        "queries_mask":  [B, M],
        "label":         [B, H, W],
        "target_resolution": float,
        "image":         [B, C, H, W],
    }
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import argparse
import torch
import numpy as np
from collections import defaultdict


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)

seed_everything(42, workers=True)

# --- Project imports ---
from training.utils import read_yaml
from training.utils import Lookup_encoding

# v2 classes (grouped token format)
from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.datasets.utils_dataset_SENFLOOD import Sen1Floods11Dataset
from training.utils.datasets.utils_dataset_SENFLOOD_MULTIRES import Sen1Floods11MultiResDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.token_grouping import collate_grouped

from training.utils.callbacks.token_assignement import TokenAssignmentCallbackSenFlood
from training.utils.callbacks.segmentation_viz_callback import SegmentationVizCallback

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--xp_name",      type=str,            required=True,  help="Experiment name")
parser.add_argument("--config_model", type=str,            required=True,  help="Model config yaml file")
parser.add_argument("--dataset_name", type=str,            required=True,  help="Name of the dataset used")
parser.add_argument("--clipping",     action="store_true",                 help="Enable gradient clipping at 1.0")
args = parser.parse_args()

xp_name = args.xp_name
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
bands_yaml = "./data/bands_info/bands.yaml"

if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"[Train] Gradient clipping: {'ON (val=1.0)' if args.clipping else 'OFF'}")

# =============================================================================
# LOOKUP TABLE
# =============================================================================
lookup_table = Lookup_encoding(read_yaml(configs_dataset), read_yaml(bands_yaml), config_model)

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"],
        project="SenFlood",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="SenFlood")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss", step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
is_unet = False
if is_unet:
    pass

else:
    model = Model_SenFlood(
        config=config_model,
        wand=True,
        name=xp_name,
        transform=None,
        lookup_table=lookup_table,
    )

    #checkpoint_path = "./checkpoints/ATOMIZER_recon-val_mIoU-epoch=16-val_avg_mIoU=0.4233.ckpt"
    #model = Model_SenFlood.load_from_checkpoint(
    #    checkpoint_path,
    #    strict=False,
    #    config=config_model,
    #    wand=True,
    #    name=xp_name,
    #    transform=None,
    #    lookup_table=lookup_table
    #)

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = UnifiedDataModule(
    path="./data/SENFLOOD",
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
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
lr_monitor   = LearningRateMonitor(logging_interval="step")
accumulator  = GradientAccumulationScheduler(scheduling={0: 4})

checkpoint_val = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"{config_model['encoder']}{xp_name}-val_loss-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)

token_assign = TokenAssignmentCallbackSenFlood(
    log_every_n_epochs=1,
    sample_indices=[0],
    save_dir="./viz_token_assignment",
    use_wandb=True,
)

viz_callback = SegmentationVizCallback(
    sample_indices=[0, 1, 2],
    log_every_n_epochs=1,
    use_wandb=True,
)

callbacks = [accumulator, checkpoint_val, lr_monitor, viz_callback]

# =============================================================================
# TRAINER
# =============================================================================
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
    gradient_clip_val=1.0 if args.clipping else None,
)

# =============================================================================
# TRAIN & TEST
# =============================================================================
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)


# =============================================================================
# MEASURE COMPLEXITY
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    """Recursively move a nested batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


if os.environ.get("LOCAL_RANK", "0") == "0":
    from fvcore.nn import FlopCountAnalysis

    print("\n" + "=" * 80)
    print("MEASURING MODEL COMPLEXITY")
    print("=" * 80 + "\n")

    data_module.setup("fit")
    train_dataset = data_module.train_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_samples = min(25, len(train_dataset))
    samples = [train_dataset[i] for i in range(num_samples)]

    results = []

    for input_size in [512]:
        print(f"\nTesting input size: {input_size}x{input_size}")

        if is_unet:
            image_0, label_0 = samples[0]
            image_0 = image_0.unsqueeze(0).to(device)

            with torch.no_grad():
                _ = model(image_0)

            image_1, _ = samples[1]
            image_1 = image_1.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    flops = FlopCountAnalysis(model, (image_1,))
                    gflops = flops.total() / 1e9
            except Exception as e:
                print(f"  FLOPs measurement failed: {e}")
                gflops = -1

            num_warmup, num_runs = 3, 20
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
            num_tokens = 15 * input_size * input_size

        else:
            batch_0 = collate_grouped([samples[0]])
            batch_0 = _batch_to_device(batch_0, device)

            with torch.no_grad():
                _ = model(batch_0)

            batch_1 = collate_grouped([samples[1]])
            batch_1 = _batch_to_device(batch_1, device)

            try:
                with torch.no_grad():
                    flops = FlopCountAnalysis(model, (batch_1,))
                    gflops = flops.total() / 1e9
            except Exception as e:
                print(f"  FLOPs measurement failed: {e}")
                gflops = -1

            num_warmup, num_runs = 3, 20
            with torch.no_grad():
                for i in range(num_warmup):
                    b = collate_grouped([samples[i % num_samples]])
                    b = _batch_to_device(b, device)
                    _ = model(b)

                torch.cuda.synchronize()
                start = time.time()
                for i in range(num_runs):
                    idx = (i + num_warmup) % num_samples
                    b = collate_grouped([samples[idx]])
                    b = _batch_to_device(b, device)
                    _ = model(b)
                    torch.cuda.synchronize()
                end = time.time()

            avg_time_ms = (end - start) / num_runs * 1000

            first_res = next(iter(batch_0["groups"]))
            num_tokens = batch_0["groups"][first_res]["tokens"].shape[1]

        results.append({
            "input_size":       input_size,
            "gsd_target":       10,
            "num_tokens":       num_tokens,
            "gflops":           gflops,
            "inference_time_ms": avg_time_ms,
        })

        print(f"  Tokens: {num_tokens}")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Inference time: {avg_time_ms:.2f} ms/tile")

    print("\n" + "=" * 80)
    print(f"COMPLEXITY SUMMARY ({config_model['encoder']})")
    print("=" * 80)
    print(f"{'Input':<10} {'Tokens':<12} {'GFLOPs':<12} {'Time (ms)':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['input_size']:<10} {r['num_tokens']:<12} {r['gflops']:<12.2f} {r['inference_time_ms']:<12.2f}")
    print("=" * 80)

# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)