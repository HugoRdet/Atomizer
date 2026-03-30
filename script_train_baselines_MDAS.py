"""
MDAS Baseline Training Script — UNet Segmentation
===================================================

Train baseline models (UNet, ViT, etc.) on MDAS for cross-sensor experiments.

Examples:
    # Train UNet on HySpex (native, no aug)
    python train_baseline_mdas.py --xp_name unet_hyspex \
        --model unet --sensor hyspex --batch_size 8

    # Train UNet on Sentinel-2 (native, no aug)
    python train_baseline_mdas.py --xp_name unet_s2 \
        --model unet --sensor sentinel2 --batch_size 8

    # Train UNet on HySpex with augmentation (same format as Atomizer)
    python train_baseline_mdas.py --xp_name unet_hyspex_aug \
        --model unet --sensor hyspex --batch_size 8 \
        --res_augment 1 2 3 4 5 \
        --spectral_configs 12 32 128 256 368

    # Resume from checkpoint
    python train_baseline_mdas.py --xp_name unet_hyspex \
        --model unet --sensor hyspex --ckpt_path checkpoints/baselines/last.ckpt
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
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_MDAS import MDASBaselineDataset
from training.utils.datasets_baselines.collate import (
    get_collate_fn,
    get_augmented_collate_fn,
)
from training.trainer_baselines import BaselineTrainer
from training.unet.model_unet import UNet

# =============================================================================
# MODEL FACTORY
# =============================================================================

def build_model(model_name: str, in_channels: int, num_classes: int = 6) -> torch.nn.Module:
    """Instantiate a baseline model by name."""
    if model_name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes, base_dim=64)
    # Future: vit_upernet, scalemae, perceiver
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: unet")


# =============================================================================
# SENSOR CONFIG
# =============================================================================

SENSOR_CHANNELS = {
    "hyspex": 368,
    "enmap_10m": 242,
    "enmap_30m": 242,
    "sentinel2": 12,
}


# =============================================================================
# DATAMODULE
# =============================================================================

class MDASBaselineDataModule(pl.LightningDataModule):
    """
    DataModule for MDAS baseline experiments.

    Train on SA1, validate on SA2.
    Augmentation (spectral merge, resolution blur) lives in the collate.
    """

    def __init__(
        self,
        root: str,
        sensor: str,
        crop_index_path: str,
        stats_path: str,
        spectral_meta_path: str,
        in_channels: int,
        batch_size: int = 8,
        num_workers: int = 4,
        res_augment: list = None,
        spectral_configs: list = None,
    ):
        super().__init__()
        self.root = root
        self.sensor = sensor
        self.crop_index_path = crop_index_path
        self.stats_path = stats_path
        self.spectral_meta_path = spectral_meta_path
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Resolution augmentation: list of factors (include 1 for identity)
        # e.g. [1, 2, 3, 4, 5] → randomly pick one per sample
        # Factors > 1 produce blur; factor 1 = no change
        self.res_augment = res_augment

        # Spectral configs: list of target band counts (include native for identity)
        # e.g. [12, 32, 128, 256, 368] → randomly pick one per sample
        # Values >= in_channels = no merging
        self.spectral_configs = spectral_configs

    def setup(self, stage=None):
        common = dict(
            root=self.root,
            sensor=self.sensor,
            crop_index_path=self.crop_index_path,
            stats_path=self.stats_path,
            spectral_meta_path=self.spectral_meta_path,
        )

        self.train_dataset = MDASBaselineDataset(
            sub_areas=[1], mode="train", augment=True, **common,
        )
        self.val_dataset = MDASBaselineDataset(
            sub_areas=[2], mode="test", augment=False, **common,
        )

        print(f"[MDAS DM] Train: {len(self.train_dataset)} crops (SA1)")
        print(f"[MDAS DM] Val:   {len(self.val_dataset)} crops (SA2)")

    def _get_collate(self, train: bool):
        modalities = [self.sensor]
        has_aug = self.res_augment or self.spectral_configs
        if train and has_aug:
            # Filter resolution factors > 1 for actual blur
            # (factor 1 = identity, handled by the augmentation function)
            return get_augmented_collate_fn(
                modalities=modalities,
                spectral_aug_prob=0.5 if self.spectral_configs else 0.0,
                spectral_groups=self.spectral_configs,
                resolution_aug_prob=0.5 if self.res_augment else 0.0,
                resolution_factors=self.res_augment,
            )
        return get_collate_fn(modalities)

    def _make_loader(self, dataset, shuffle: bool, train: bool):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._get_collate(train=train),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=train,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True, train=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False, train=False)


# =============================================================================
# ARGS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MDAS Baseline Training")
    parser.add_argument("--xp_name", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet"],
                        help="Baseline model to train")
    parser.add_argument("--sensor", type=str, default="hyspex",
                        choices=["hyspex", "sentinel2", "enmap_10m", "enmap_30m"])

    # Paths
    parser.add_argument("--data_root", type=str,
                        default="./data/MDAS/Augsburg_data_4_publication")
    parser.add_argument("--crop_index", type=str, default=None)
    parser.add_argument("--stats_json", type=str, default=None)
    parser.add_argument("--spectral_meta", type=str, default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume from checkpoint")

    # Augmentation (same format as Atomizer)
    parser.add_argument("--res_augment", type=int, nargs="+", default=None,
                        help="Resolution augment factors (e.g. --res_augment 1 2 3 4 5). "
                             "Include 1 for identity. None = no aug.")
    parser.add_argument("--spectral_configs", type=int, nargs="+", default=None,
                        help="Spectral merge configs (e.g. --spectral_configs 12 32 128 256 368). "
                             "Include native band count for identity. None = no aug.")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="Atomizer_MDAS_Baselines")
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Resolve default paths
    if args.crop_index is None:
        args.crop_index = os.path.join(args.data_root, "mdas_crop_index.csv")
    if args.stats_json is None:
        args.stats_json = os.path.join(args.data_root, "mdas_norm_stats.json")
    if args.spectral_meta is None:
        args.spectral_meta = os.path.join(args.data_root, "mdas_spectral_meta.json")

    in_channels = SENSOR_CHANNELS[args.sensor]

    print(f"[MDAS Baseline] Model: {args.model}")
    print(f"[MDAS Baseline] Sensor: {args.sensor} ({in_channels} channels)")
    print(f"[MDAS Baseline] Augmentation: res_augment={args.res_augment}, "
          f"spectral_configs={args.spectral_configs}")

    # ── Model ───────────────────────────────────────────────────────
    model = build_model(args.model, in_channels=in_channels, num_classes=6)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MDAS Baseline] Parameters: {param_count:,}")

    # ── Trainer module ──────────────────────────────────────────────
    trainer_module = BaselineTrainer(
        model=model,
        modality=args.sensor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_classes=6,
    )

    # ── Data module ─────────────────────────────────────────────────
    data_module = MDASBaselineDataModule(
        root=args.data_root,
        sensor=args.sensor,
        crop_index_path=args.crop_index,
        stats_path=args.stats_json,
        spectral_meta_path=args.spectral_meta,
        in_channels=in_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        res_augment=args.res_augment,
        spectral_configs=args.spectral_configs,
    )

    # ── WandB ───────────────────────────────────────────────────────
    wandb_logger = None
    if not args.no_wandb and os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            name=f"{args.model}_{args.sensor}_{args.xp_name}",
            project=args.wandb_project,
            config=vars(args),
        )
        wandb_logger = WandbLogger(project=args.wandb_project)

    # ── Callbacks ───────────────────────────────────────────────────
    ckpt_dir = f"./checkpoints/baselines/{args.model}_{args.sensor}"

    checkpoint_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_best, checkpoint_last, lr_monitor]

    # ── Setup data ──────────────────────────────────────────────────
    data_module.setup()

    # ── PL Trainer ──────────────────────────────────────────────────
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        use_distributed_sampler=False,
        devices=-1,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        precision=args.precision,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
    )

    # ── Train ───────────────────────────────────────────────────────
    trainer.fit(trainer_module, datamodule=data_module, ckpt_path=args.ckpt_path)

    # ── Save WandB run ID ───────────────────────────────────────────
    if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        run_id = wandb.run.id
        print(f"WANDB_RUN_ID: {run_id}")
        os.makedirs("training/wandb_runs", exist_ok=True)
        with open(f"training/wandb_runs/baseline_{args.xp_name}.txt", "w") as f:
            f.write(run_id)

    print("[MDAS Baseline] Training complete.")


if __name__ == "__main__":
    main()