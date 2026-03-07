"""
MAE Pre-training Script — MMEarth
==================================
Masked Autoencoder pretraining with Atomiser on MMEarth.

Uses Model_MAE (training/trainer_MAE.py):
  - Encode once with mask_ratio (default 0.20 for initial testing, 0.75 for full MAE)
  - Build reconstruction queries from masked latent pools
  - MSE loss on masked token reflectances

Examples:
    # Quick test — 20% masking, 10k samples
    python train_mae.py --xp_name mae_test --config_model atomiser.yaml \
        --mask_ratio 0.20 --max_samples 10000

    # Full MAE — 75% masking
    python train_mae.py --xp_name mae_75 --config_model atomiser.yaml \
        --mask_ratio 0.75

    # Resume from checkpoint
    python train_mae.py --xp_name mae_75 --config_model atomiser.yaml \
        --mask_ratio 0.75 --ckpt_path ./checkpoints/last.ckpt
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    GradientAccumulationScheduler,
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_MAE import Model_MAE
from training.utils.datasets.utils_dataset_MM_Earth_pretrain import MMEarthMultiTask
from training.utils.datasets.token_grouping import collate_multitask


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Atomiser MAE Pre-training (MMEarth)")
parser.add_argument("--xp_name",      type=str, required=True, help="Experiment name")
parser.add_argument("--config_model", type=str, required=True, help="Model config yaml")
parser.add_argument("--mask_ratio",   type=float, default=0.20,
                    help="Fraction of latents to mask (0.20 for testing, 0.75 for full MAE)")
parser.add_argument("--dataset_name", type=str, default="MMEarth",
                    choices=["MMEarth", "MMEarth100k", "MMEarth64"],
                    help="MMEarth subset")
parser.add_argument("--mmearth_path", type=str, default="./data/MM-Earth")
parser.add_argument("--max_samples",  type=int, default=None,
                    help="Cap dataset size for quick tests (e.g. 10000)")
parser.add_argument("--max_queries_recon", type=int, default=200_000)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--ckpt_path",    type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--val_fraction", type=float, default=0.01)
args = parser.parse_args()

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model = read_yaml("./training/configs/" + args.config_model)
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
bands_yaml      = "./data/bands_info/bands.yaml"

# Inject mask_ratio into config so Model_MAE picks it up
config_model.setdefault("pretrain", {})
config_model["pretrain"]["mask_ratio"] = args.mask_ratio

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset), read_yaml(bands_yaml), config_model
)

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"MAE_{args.dataset_name}_mask{int(args.mask_ratio*100)}_{args.xp_name}",
        project="Atomizer_Pretrain",
        config={
            **config_model,
            "mask_ratio": args.mask_ratio,
            "dataset": args.dataset_name,
            "max_samples": args.max_samples,
        },
    )
    wandb_logger = WandbLogger(project="Atomizer_Pretrain")
    wandb.define_metric("train_loss", step_metric="trainer/global_step")
    wandb.define_metric("val_loss",   step_metric="trainer/global_step")

# =============================================================================
# MODEL
# =============================================================================
model = Model_MAE(
    config=config_model,
    wand=True,
    name=args.xp_name,
    lookup_table=lookup_table,
)

# =============================================================================
# DATASET
# =============================================================================
dataset_config = read_yaml(bands_yaml)

common_kwargs = dict(
    root_path=args.mmearth_path,
    dataset_config=dataset_config,
    config_model=config_model,
    look_up=lookup_table,
    subset=args.dataset_name,
    # MAE only needs reconstruction queries — no seg queries
    tasks=["reconstruction"],
    max_queries_seg=0,
    max_queries_recon=args.max_queries_recon,
    max_samples=args.max_samples,
)

full_dataset = MMEarthMultiTask(mode="train", **common_kwargs)

# Deterministic train/val split
full_len  = len(full_dataset)
val_len   = max(8, int(full_len * args.val_fraction))
train_len = full_len - val_len

generator   = torch.Generator().manual_seed(42)
all_indices = torch.randperm(full_len, generator=generator).tolist()

train_dataset = MMEarthMultiTask(mode="train", **common_kwargs)
val_dataset   = MMEarthMultiTask(mode="train", **common_kwargs)

train_dataset.tile_indices = [full_dataset.tile_indices[i] for i in all_indices[:train_len]]
val_dataset.tile_indices   = [full_dataset.tile_indices[i] for i in all_indices[train_len:]]

print(f"[MAE] mask_ratio={args.mask_ratio}")
print(f"[MAE] dataset={args.dataset_name}  train={train_len}  val={val_len}")

# =============================================================================
# DATALOADERS
# =============================================================================
batch_size = config_model["dataset"]["batchsize"]

def make_loader(dataset, shuffle):
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_multitask,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

train_loader = make_loader(train_dataset, shuffle=True)
val_loader   = make_loader(val_dataset,   shuffle=False)

# =============================================================================
# CALLBACKS
# =============================================================================
lr_monitor   = LearningRateMonitor(logging_interval="step")
accumulator  = GradientAccumulationScheduler(scheduling={0: 16})

checkpoint_best = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=(
        f"mae_{args.xp_name}"
        f"_mask{int(args.mask_ratio*100)}"
        "-val_mse-{epoch:02d}-{val_mse:.4f}"
    ),
    monitor="val_mse",
    mode="min",
    save_top_k=1,
    verbose=True,
)

checkpoint_last = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=f"mae_{args.xp_name}_mask{int(args.mask_ratio*100)}-last-{{epoch:02d}}",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
    verbose=True,
)

# =============================================================================
# TRAINER
# =============================================================================
trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1,
    max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=[accumulator, checkpoint_best, checkpoint_last, lr_monitor],
    default_root_dir="./checkpoints/",
    gradient_clip_val=1.0,
    limit_val_batches=500,       # ~500 batches for a fast val loop
    #val_check_interval=0.05,     # validate every 5% of an epoch
)

# =============================================================================
# TRAIN
# =============================================================================
trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=args.ckpt_path,
)

# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb as _wandb
    run_id = _wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/mae_{args.xp_name}.txt", "w") as f:
        f.write(run_id)