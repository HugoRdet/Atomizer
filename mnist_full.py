"""
MNIST Classification — Training Script — CONTROLLED VARIANT
==============================================================
Isolates whether the drop from the old (leaky) numbers -- 99.43 / 85.49 --
to the new (correct) numbers -- 98.77 / 63.41 -- was driven by:
    (a) training-set size (60k vs. 55k after the val carve-out), or
    (b) checkpoint selection (old: monitor="val_loss" computed on the literal
        test set, i.e. searching across all 50 epochs for whichever one
        scored best ON the test set -- vs. new: genuinely held-out val).

This variant controls for (b) while reverting (a) to the old regime:
    - val_size=0 -> train on the FULL 60k images, no held-out slice at all.
    - No val_dataloaders passed to trainer.fit() at all (there's nothing
      genuinely held-out to validate against in this mode).
    - No monitor-based ModelCheckpoint. We use save_last=True and always
      evaluate/report from the LAST epoch's checkpoint, chosen by a fixed
      rule (end of training) rather than searched for.

If this run's token-removal numbers land close to the OLD 85.49, that
points to checkpoint-selection-on-test-set as the dominant cause of the
original (inflated) result. If they land close to the NEW 63.41, training-
set size was doing more of the work than expected.

Usage:
    python train_mnist_full_lastepoch.py
"""

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

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_MNIST import Model_MNIST
from training.utils.datasets.utils_dataset_MNIST import MNISTSparseCanvas
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# COLLATE (identical to train_mnist.py's mnist_collate)
# =============================================================================

def mnist_collate(samples):
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

xp_name = "mnist_full_lastepoch_control"

config_model = read_yaml("./training/configs/config_test-MNIST.yaml")
config_model["trainer"]["num_classes"] = 10

# THE key change for this variant: no held-out val slice.
config_model["trainer"]["val_size"] = 0

# =============================================================================
# LOOKUP TABLE
# =============================================================================

bands_yaml = "./data/bands_info/bands.yaml"
configs_dataset = "./data/Tiny_BigEarthNet/configs_dataset_regular.yaml"
lookup_table = Lookup_encoding(
    None, read_yaml(bands_yaml), config_model
)
TokenBuilder.REFERENCE_SIZES[0.2] = 28
lookup_table.get_or_register_modality(0.2, 28)
lookup_table.get_resolution_idx(0.2)

# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(name=xp_name, project="MNIST_classification", config=config_model)
    wandb_logger = WandbLogger(project="MNIST_classification")

# =============================================================================
# DATASETS
# =============================================================================
# train: FULL 60k (val_size=0 -> no carve-out), forced subsample_keep_rate=1.0
#        by MNISTSparseCanvas regardless of config.
# test:  the real, untouched 10k MNIST test set. Used ONLY for the final
#        reported numbers, from the LAST epoch's checkpoint (not searched
#        for via any monitored metric).
#
# NOTE: no val_dataset/val_loader in this variant -- there's nothing
# genuinely held-out to validate against when val_size=0, and we're
# deliberately not doing monitor-based checkpoint selection here.

train_dataset = MNISTSparseCanvas(
    mode="train", config_model=config_model, look_up=lookup_table,
)
test_dataset = MNISTSparseCanvas(
    mode="test", config_model=config_model, look_up=lookup_table,
)

batch_size = config_model["trainer"]["batchsize"]

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
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
# No monitor-based ModelCheckpoint. save_last=True writes a `last.ckpt`
# unconditionally at the end of every epoch (overwriting the previous one),
# so at the end of training it points at epoch 49/50 -- a FIXED rule, not
# a metric search across all epochs.

callbacks = [
    LearningRateMonitor(logging_interval="step"),
    GradientAccumulationScheduler(scheduling={0: 1}),
    ModelCheckpoint(
        dirpath="./checkpoints/mnist_full_lastepoch_control/",
        filename=f"{xp_name}-{{epoch:02d}}",
        save_top_k=0,      # do NOT keep a "best" checkpoint by any metric
        save_last=True,    # always keep the most recent epoch
        verbose=True,
    ),
]

trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    logger=wandb_logger,
    strategy="ddp_find_unused_parameters_true",
    max_epochs=config_model["trainer"]["epochs"],
    callbacks=callbacks,
    enable_checkpointing=True,
    log_every_n_steps=10,
    # No validation loop at all in this variant.
    num_sanity_val_steps=0,
)

# =============================================================================
# TRAIN
# =============================================================================

print("=" * 60)
print("MNIST Classification — CONTROLLED VARIANT (full 60k, last-epoch ckpt)")
print("=" * 60)
print(f"  Train samples:    {len(train_dataset)}  (full 60k, no held-out slice)")
print(f"  Test samples:     {len(test_dataset)}  (real MNIST test set)")
print(f"  Epochs:           {config_model['trainer']['epochs']}")
print(f"  Checkpoint rule:  LAST epoch (fixed), no monitor-based selection")
print("=" * 60)

# No val_dataloaders passed -- there is nothing genuinely held-out in this
# mode, and we are not doing monitor-based selection.
trainer.fit(model, train_dataloaders=train_loader)

print("\n" + "=" * 60)
print("Running final test on LAST-EPOCH checkpoint (fixed rule, not searched)...")
print("=" * 60)
trainer.test(model, dataloaders=test_loader, ckpt_path="last")

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        wandb.finish()
    except Exception as e:
        print(f"[wandb] finish() raised: {e!r} (ignored)")

print("Done.")
print("\nNext step: run test_mnist_sweep.py with --ckpt pointing at the "
      "'last.ckpt' file saved under ./checkpoints/mnist_full_lastepoch_control/, "
      "and compare its rate x mode table against both the old (85.49-style) "
      "and new (63.41-style) numbers.")
