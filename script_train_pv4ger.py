"""
Atomiser PV4GER Training Script
=================================

Binary segmentation training on geo-bench m-pv4ger-seg (PV panel detection,
RGB aerial imagery).

Mirror of script_train_senflood.py but using PV4GERDataset and the
multi-task-ready collate (task field defaults to "segmentation").

Required:
    - bands_pv4ger section in ./data/bands_info/bands.yaml
      (Red/Green/Blue with idx, bandwidth, central_wavelength)
    - configs_dataset_pv4ger.yaml under ./data/Tiny_BigEarthNet/
"""

import os
import argparse

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    EarlyStopping,
)

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding

# Use the existing Sen1Floods11 segmentation trainer — same pipeline.
from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.datasets.utils_dataset_pv4ger import PV4GERDataset
from training.utils.datasets.dataloaders import UnifiedDataModule

# Register PV4GERDataset as a grouped dataset (avoids the h5-path-mangling
# branch, same trick we used for ForestNet).
UnifiedDataModule.GROUPED_DATASET_CLASSES = (
    UnifiedDataModule.GROUPED_DATASET_CLASSES | {"PV4GERDataset"}
)


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser PV4GER training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str, required=True,
                    help="Model config yaml (e.g. atomiser_pv4ger.yaml)")
parser.add_argument("--dataset_name", type=str, default="pv4ger",
                    help="Used to find configs_dataset_<n>.yaml")
parser.add_argument("--clipping",     action="store_true")
args = parser.parse_args()

xp_name           = args.xp_name
config_model      = read_yaml("./training/configs/" + args.config_model)
configs_dataset   = f"./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml        = "./data/bands_info/bands.yaml"

if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"[Train] Gradient clipping: {'ON (val=1.0)' if args.clipping else 'OFF'}")


# =============================================================================
# LOOKUP TABLE
# =============================================================================

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset),
    read_yaml(bands_yaml),
    config_model,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=config_model["encoder"] + "_" + xp_name,
        project="Atomizer_PV4GER",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="Atomizer_PV4GER")


# =============================================================================
# DATA MODULE
# =============================================================================

data_module = UnifiedDataModule(
    path="./data/geo-bench-1.0/segmentation_v1.0/m-pv4ger-seg",
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=PV4GERDataset,
)


# =============================================================================
# MODEL
# =============================================================================

model = Model_SenFlood(
    config=config_model,
    wand=True,
    name=xp_name,
    transform=None,
    lookup_table=lookup_table,
)


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/pv4ger/"
os.makedirs(ckpt_dir, exist_ok=True)

lr_monitor   = LearningRateMonitor(logging_interval="step")
accumulator  = GradientAccumulationScheduler(scheduling={0: 4})

checkpoint_val = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=f"{config_model['encoder']}_{xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
    monitor="val_mIoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)

checkpoint_last = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=f"{config_model['encoder']}_{xp_name}-last",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
)

early_stop = EarlyStopping(
    monitor="val_mIoU",
    mode="max",
    patience=int(config_model["trainer"].get("patience", 20)),
    verbose=True,
)

callbacks = [accumulator, checkpoint_val, checkpoint_last, early_stop, lr_monitor]


# =============================================================================
# TRAINER
# =============================================================================

trainer = Trainer(
    strategy="ddp_find_unused_parameters_true",
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    gradient_clip_val=1.0 if args.clipping else None,
)


# =============================================================================
# TRAIN + TEST (test on rank 0 only — avoids DDP-test hang)
# =============================================================================

trainer.fit(model, datamodule=data_module)

best_ckpt = checkpoint_val.best_model_path

if trainer.is_global_zero:
    print(f"\n{'='*60}")
    print(f"  Testing best checkpoint: {best_ckpt}")
    print(f"{'='*60}\n")

    test_trainer = Trainer(
        devices=1,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        default_root_dir=ckpt_dir,
    )
    test_trainer.test(model, datamodule=data_module, ckpt_path=best_ckpt)

# ─────────────────────────────────────────────────────────────────────
# Coordinated DDP teardown
# ─────────────────────────────────────────────────────────────────────
import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)
    wandb.finish()