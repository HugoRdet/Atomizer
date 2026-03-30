"""
Launch BigEarthNet pretraining.

Usage:
    python train_ben.py

Loads the base config from the existing YAML, then adds/overrides
BEN-specific settings on top.
"""

import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.utils_dataset_Ben import (
    BigEarthNetAtomizer, collate_ben, register_ben_bands,
)
from training.trainer_Ben import BENPretrainTrainer
from training.utils.datasets.token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# PATHS (matching your C2Seg script)
# ═══════════════════════════════════════════════════════════════════════

CONFIG_MODEL_PATH = "./training/configs/config_test-Atomiser_Atos_One.yaml"
BANDS_YAML_PATH = "./data/bands_info/bands.yaml"
CONFIGS_DATASET_PATH = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"


# ═══════════════════════════════════════════════════════════════════════
# BEN-SPECIFIC OVERRIDES
# ═══════════════════════════════════════════════════════════════════════

BEN_OVERRIDES = {
    "trainer": {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "max_epochs": 40,
        "batch_size": 32,
        "grad_accum": 1,
        "num_workers": 8,
        "precision": "bf16-mixed",
        "num_classes": 19,
    },
    "data": {
        "images_lmdb": "data/Encoded-BigEarthNet",
        "metadata_parquet": "data/Encoded-BigEarthNet/metadata.parquet",
        "metadata_snow_cloud_parquet": "data/Encoded-BigEarthNet/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    },
    "ben_pretrain": {
        "tpl_min": 64,
        "tpl_max": 1024,
        "tpl_step": 32,
        "tpl_val": 768,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# KNOWN RESOLUTIONS (same as C2Seg)
# ═══════════════════════════════════════════════════════════════════════

ALL_KNOWN_RESOLUTIONS = {
    2.2: 2048, 2.5: 2048, 4.78: 2048, 5.0: 2048,
    10.0: 2048, 20.0: 2048, 30.0: 2048, 60.0: 2048,
}


def register_all_resolutions(lookup_table):
    """Pre-register all known resolutions to avoid runtime registration."""
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


def main():
    # ── Load config (same pattern as C2Seg script) ──
    config_model = read_yaml(CONFIG_MODEL_PATH)

    # Apply BEN overrides
    for key, value in BEN_OVERRIDES.items():
        if key in config_model and isinstance(config_model[key], dict) and isinstance(value, dict):
            config_model[key].update(value)
        else:
            config_model[key] = value

    # ── Lookup table (same constructor as C2Seg) ──
    lookup_table = Lookup_encoding(
        read_yaml(CONFIGS_DATASET_PATH),
        read_yaml(BANDS_YAML_PATH),
        config_model,
    )

    # Pre-register resolutions
    register_all_resolutions(lookup_table)

    # ── CRITICAL: Register BEN bands BEFORE creating datasets ──
    register_ben_bands(lookup_table)

    # ── Datasets ──
    data_dirs = config_model["data"]
    stats_path = config_model["data"].get(
        "norm_stats", "data/Encoded-BigEarthNet/ben_norm_stats.json"
    )

    train_ds = BigEarthNetAtomizer(
        data_dirs=data_dirs,
        split="train",
        look_up=lookup_table,
        stats_path=stats_path,
    )

    val_ds = BigEarthNetAtomizer(
        data_dirs=data_dirs,
        split="val",
        look_up=lookup_table,
        stats_path=stats_path,
    )

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_ds,
        batch_size=config_model["trainer"]["batch_size"],
        shuffle=True,
        num_workers=config_model["trainer"]["num_workers"],
        collate_fn=collate_ben,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config_model["trainer"]["batch_size"],
        shuffle=False,
        num_workers=config_model["trainer"]["num_workers"],
        collate_fn=collate_ben,
        pin_memory=True,
    )

    # ── Model ──
    model = BENPretrainTrainer(
        config=config_model,
        wand=None,
        name="ben_s2_pretrain",
        transform=None,
        lookup_table=lookup_table,
    )

    # ── Logger ──
    wandb_logger = None
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            name="ben_pretrain_12bands",
            project="Atomizer_BigEarthNet",
            config=config_model,
        )
        wandb_logger = WandbLogger(project="Atomizer_BigEarthNet")

    # ── Callbacks ──
    ckpt_dir = "./checkpoints/ben/"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_mAP",
        mode="max",
        save_top_k=3,
        filename="ben-{epoch:02d}-{val_mAP:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── Trainer ──
    trainer = pl.Trainer(
        max_epochs=config_model["trainer"]["max_epochs"],
        accumulate_grad_batches=config_model["trainer"]["grad_accum"],
        precision=config_model["trainer"]["precision"],
        accelerator="gpu",
        devices=-1,  # all available GPUs
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        #val_check_interval=0.25,
    )

    # ── Train ──
    trainer.fit(model, train_loader, val_loader)

    # ── Save encoder for downstream ──
    model.save_encoder_only("final")
    print("\n[Done] Encoder checkpoint ready for C2Seg fine-tuning.")


if __name__ == "__main__":
    main()