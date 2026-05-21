"""
Atomiser xView2 Training Script
================================

5-class damage segmentation on xView2 (PANGAEA setup):
    Input:  pre + post RGB, [T=2, C=3, H=512, W=512]
    Target: 5-class damage mask {0..4}, no IGNORE
    Native resolution: 0.5 m (sub-meter aerial WorldView-3)

The same Model_SenFlood trainer used for Sen1Floods11 — xView2 is just
another multi-temporal segmentation task from its perspective.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training.

Required:
    - bands_xview section in ./data/bands_info/bands.yaml (3 RGB bands)
    - configs_dataset_xview.yaml under ./data/Tiny_BigEarthNet/ (or wherever
      you keep dataset configs)
    - atomiser_xview.yaml under ./training/configs/

Example:
    python script_train_xview.py --xp_name v1 \\
        --config_model atomiser_xview.yaml
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

from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.datasets.utils_dataset_xview import XView2Dataset
from training.utils.datasets.dataloaders import UnifiedDataModule

# Tell UnifiedDataModule that XView2Dataset uses the grouped-tokens collate
UnifiedDataModule.GROUPED_DATASET_CLASSES = (
    UnifiedDataModule.GROUPED_DATASET_CLASSES | {"XView2Dataset"}
)


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Atomiser xView2 training")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,  default="config_test-xview.yaml",
                    help="Model config yaml (e.g. atomiser_xview.yaml)")
parser.add_argument("--clipping",     action="store_true")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

parser.add_argument("--data_dir", type=str, default="./data/xview")

args = parser.parse_args()

xp_name           = args.xp_name
config_model      = read_yaml("./training/configs/" + args.config_model)
configs_dataset   = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml        = "./data/bands_info/bands.yaml"

if os.environ.get("LOCAL_RANK", "0") == "0":
    if args.test_only:
        print(f"[Train] Test-only mode: {args.test_only}")
    else:
        print(f"[Train] Gradient clipping: {'ON' if args.clipping else 'OFF'}")


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
        project="Atomizer_xView2",
        config=config_model,
    )
    wandb_logger = WandbLogger(project="Atomizer_xView2")


# =============================================================================
# DATA MODULE
# =============================================================================

data_module = UnifiedDataModule(
    path=args.data_dir,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=4,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=XView2Dataset,
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
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/xview/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
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
        gradient_clip_val=1.0 ,
    )

    trainer.fit(model, datamodule=data_module)

    best_ckpt = checkpoint_val.best_model_path

    import torch.distributed as dist
    is_rank_zero = trainer.is_global_zero

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if not is_rank_zero:
        if wandb_logger:
            import wandb
            wandb.finish()
        raise SystemExit(0)

else:
    if not os.path.exists(args.test_only):
        raise FileNotFoundError(
            f"--test_only checkpoint not found: {args.test_only}"
        )
    best_ckpt = args.test_only
    print(f"\n[test-only mode] Skipping training, testing: {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================
# Note: For now we use Lightning's test() with the cropped val-style protocol
# (center crop 512×512). Sliding-window full-image evaluation for Atomiser
# requires custom logic since the model takes tokenized queries rather than
# spatial logits — see docstring for future work.

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
if unexpected:
    print(f"[load_state_dict] ignored {len(unexpected)} unexpected keys "
          f"(runtime caches — recreated automatically)")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(model, datamodule=data_module)


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