"""
MNIST Classification — Training Script
========================================
Tests the new multi-res Atomiser on MNIST digit classification.
Uses new-format dataset (8-column tokens, batch dict output).

All model / latent-grid / token-budget settings come from
`./training/configs/config_test-MNIST.yaml`. The only override
applied here is `num_classes = 10`, which is a property of the
task rather than the model.

CHANGES vs. original:
    - val_dataset now draws from a genuine held-out slice of the TRAIN set
      (see the patched MNISTSparseCanvas / _load_mnist_with_val_split),
      not the test set. ModelCheckpoint(monitor="val_loss") therefore
      selects the best checkpoint without ever touching test data.
    - A separate test_dataset/test_loader (mode="test") is now built and
      used for the final trainer.test() call, instead of reusing
      val_loader. This is the actual, untouched 10k MNIST test set.

Usage:
    # Train + auto-test on best checkpoint
    python train_mnist.py

    # Test an existing checkpoint, fresh wandb run
    python train_mnist.py --test-ckpt path/to/ckpt.ckpt

    # Test an existing checkpoint, append results to an existing wandb run
    python train_mnist.py --test-ckpt path/to/ckpt.ckpt \\
                          --wandb-run-id abc123def
"""

import argparse
import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MNIST Classification — Atomiser train / test")
    p.add_argument(
        "--test-ckpt", type=str, default=None,
        help="Path to a Lightning checkpoint (.ckpt). If set, skips training "
             "and runs only trainer.test() on the MNIST test set.",
    )
    p.add_argument(
        "--wandb-run-id", type=str, default=None,
        help="If set, resume this wandb run instead of creating a new one. "
             "Useful for appending test metrics to an existing training run.",
    )
    return p.parse_args()


args = parse_args()
test_only = args.test_ckpt is not None

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_MNIST import Model_MNIST
from training.utils.datasets.utils_dataset_MNIST import MNISTSparseCanvas
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# COLLATE
# =============================================================================

def mnist_collate(samples):
    """
    Collate MNIST dict samples into a batched dict.

    Each sample (from MNISTSparseCanvas):
        {"groups": {0.2: {"tokens": [N,8], "mask": [N], "shape": (C,H,W)}},
         "queries": [M,8], "queries_mask": [M],
         "target_resolution": 0.2, "image": [C,H,W],
         "label": scalar}

    Batched output:
        {"groups": {0.2: {"tokens": [B,N,8], "mask": [B,N], "shape": (C,H,W)}},
         "queries": [B,M,8], "queries_mask": [B,M],
         "target_resolution": 0.2,
         "label": [B]}
    """
    all_res = sorted(samples[0]["groups"].keys())

    groups = {}
    for res in all_res:
        groups[res] = {
            "tokens": torch.stack([s["groups"][res]["tokens"] for s in samples]),
            "mask":   torch.stack([s["groups"][res]["mask"]   for s in samples]),
            "shape":  samples[0]["groups"][res]["shape"],
        }

    return {
        "groups":            groups,
        "queries":           torch.stack([s["queries"]      for s in samples]),
        "queries_mask":      torch.stack([s["queries_mask"] for s in samples]),
        "target_resolution": samples[0]["target_resolution"],
        "latent_layout":     samples[0].get("latent_layout", "grid"),
        "label":             torch.stack([s["label"]        for s in samples]),
    }


# =============================================================================
# CONFIG
# =============================================================================

xp_name = "mnist_classification"

config_model = read_yaml("./training/configs/config_test-MNIST.yaml")

# Only task-level override: number of classes (digits 0..9).
config_model["trainer"]["num_classes"] = 10

# =============================================================================
# LOOKUP TABLE
# =============================================================================

bands_yaml = "./data/bands_info/bands.yaml"
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
lookup_table = Lookup_encoding(
    None, read_yaml(bands_yaml), config_model
)

# Register MNIST resolution (0.2 m/px, 28×28 canvas)
TokenBuilder.REFERENCE_SIZES[0.2] = 28
lookup_table.get_or_register_modality(0.2, 28)
lookup_table.get_resolution_idx(0.2)

# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb_kwargs = dict(
        name=xp_name, project="MNIST_classification", config=config_model,
    )
    if args.wandb_run_id is not None:
        # Resume existing run — test metrics will be appended to it.
        # resume="must" makes wandb error if the ID doesn't exist, so we
        # fail loudly rather than silently creating a fresh run.
        wandb_kwargs["id"]     = args.wandb_run_id
        wandb_kwargs["resume"] = "must"
        print(f"[wandb] Resuming run id={args.wandb_run_id}")
    wandb.init(**wandb_kwargs)
    wandb_logger = WandbLogger(project="MNIST_classification")

# =============================================================================
# DATASETS
# =============================================================================
# train: 55k (60k train minus held-out val), FORCED subsample_keep_rate=1.0
#        by MNISTSparseCanvas regardless of config.
# val:   5k, stratified held-out slice of the TRAIN set (used for checkpoint
#        selection only -- never touches test data).
# test:  the real, untouched 10k MNIST test set. Only used once, at the end,
#        for the final reported numbers.

train_dataset = MNISTSparseCanvas(
    mode="train", config_model=config_model, look_up=lookup_table,
)
val_dataset = MNISTSparseCanvas(
    mode="val", config_model=config_model, look_up=lookup_table,
)
test_dataset = MNISTSparseCanvas(
    mode="test", config_model=config_model, look_up=lookup_table,
)

batch_size = config_model["trainer"]["batchsize"]

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, collate_fn=mnist_collate, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, collate_fn=mnist_collate, pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, collate_fn=mnist_collate, pin_memory=True,
)

# =============================================================================
# MODEL
# =============================================================================

model = Model_MNIST(
    config=config_model, wand=True, name=xp_name,
    transform=None, lookup_table=lookup_table,
)

# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================

callbacks = [
    LearningRateMonitor(logging_interval="step"),
    GradientAccumulationScheduler(scheduling={0: 1}),
    ModelCheckpoint(
        dirpath="./checkpoints/mnist/",
        filename=f"{xp_name}-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val_accuracy", mode="max", save_top_k=1, verbose=True,
    ),
]

trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    # Test-only mode: force single GPU. DDP teardown without a preceding
    # fit() lifecycle is fragile (rank 0 can stall on wandb flush while
    # other ranks wait at a barrier). MNIST test is 10k samples — one GPU
    # is plenty.
    devices=1 if test_only else -1,
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    logger=wandb_logger,
    strategy="auto" if test_only else "ddp_find_unused_parameters_true",
    max_epochs=config_model["trainer"]["epochs"],
    callbacks=callbacks,
    enable_checkpointing=True,
    log_every_n_steps=10,
)

# =============================================================================
# TRAIN
# =============================================================================

if test_only:
    print("=" * 60)
    print("TEST-ONLY MODE")
    print("=" * 60)
    print(f"  Checkpoint: {args.test_ckpt}")
    if args.wandb_run_id:
        print(f"  Wandb run:  {args.wandb_run_id} (resumed)")
    print("=" * 60)
    # Real test set, not val_loader.
    trainer.test(model, dataloaders=test_loader, ckpt_path=args.test_ckpt)

else:
    print("=" * 60)
    print("MNIST Classification — New Atomiser")
    print("=" * 60)
    print(f"  Canvas:           28×28, 1 band")
    print(f"  Tokens:           784")
    print(f"  max_tokens:       {config_model['trainer']['max_tokens']}")
    print(f"  max_tokens_recon: {config_model['trainer']['max_tokens_reconstruction']}")
    print(f"  sigma_factor:     {config_model['latent_grid']['sigma_factor']}")
    print(f"  hexagonal:        {config_model['latent_grid'].get('hexagonal', False)}")
    print(f"  train_sampling:   {config_model['latent_grid']['train_sampling']}")
    print(f"  val_sampling:     {config_model['latent_grid']['val_sampling']}")
    print(f"  num_classes:      {config_model['trainer']['num_classes']}")
    print(f"  mode:             {config_model['trainer'].get('mode', 'segmentation')}")
    print(f"  Batch:            {batch_size}")
    print(f"  Epochs:           {config_model['trainer']['epochs']}")
    print(f"  Train samples:    {len(train_dataset)}")
    print(f"  Val samples:      {len(val_dataset)}  (held-out from train, stratified)")
    print(f"  Test samples:     {len(test_dataset)}  (real MNIST test set)")
    print("=" * 60)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # =========================================================================
    # TEST (auto, on best checkpoint)
    # =========================================================================
    # Best checkpoint is selected via val_loss on the held-out train slice
    # (never touches test data). Final reported metrics come from a single
    # pass over the real, untouched MNIST test set.
    print("\n" + "=" * 60)
    print("Running final test on best checkpoint...")
    print("=" * 60)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")

# =============================================================================
# SAVE RUN ID
# =============================================================================
# Only save the run-id file in training mode. In test-only mode the wandb
# id either belongs to a resumed training run (already on disk) or to a
# fresh ad-hoc test run we don't want to track as the canonical run.

if (not test_only) and wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/{xp_name}.txt", "w") as f:
        f.write(run_id)

# Flush wandb explicitly. Lightning's logger teardown normally handles this
# during fit(), but in test-only mode the timing is different and rank 0
# can stall on upload while the process never exits.
if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        wandb.finish()
    except Exception as e:
        print(f"[wandb] finish() raised: {e!r} (ignored)")

print("Done.")
