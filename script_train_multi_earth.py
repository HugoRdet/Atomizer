#!/usr/bin/env python3
"""
MultiEarth Training Script — Atomizer Cross-Sensor Deforestation
==================================================================

Train Atomizer on MultiEarth deforestation segmentation.
Cross-sensor transfer: train on one sensor, test on both.

Experiments:
  1. --sensor s2                  (train S2, test S2 + L8)
  2. --sensor l8                  (train L8, test L8 + S2)

Usage:
    # Train on S2, test cross-sensor on L8
    python train_multiearth.py --xp_name atomizer_s2 --sensor s2

    # Train on L8, test cross-sensor on S2
    python train_multiearth.py --xp_name atomizer_l8 --sensor l8

    # With pretrained encoder
    python train_multiearth.py --xp_name atomizer_s2_pretrained \
        --sensor s2 --pretrained_encoder ./pth_files/mae_encoder.pth
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
from torch.utils.data import DataLoader
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_pretraining import Model_Pretrain
from training.utils.datasets.utils_dataset_multiearth import (
    MultiEarthDataset,
    S2_BANDS_INFO, L8_BANDS_INFO,
    TASK_NAME, NUM_CLASSES,
)
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder


# =============================================================================
# SENSOR CONFIG
# =============================================================================

SENSOR_CONFIG = {
    "s2": {
        "bands_info": S2_BANDS_INFO,
        "gsd": 10.0,
        "n_bands": 12,
    },
    "l8": {
        "bands_info": L8_BANDS_INFO,
        "gsd": 30.0,
        "n_bands": 7,
    },
}

ALL_KNOWN_RESOLUTIONS = {
    10.0: 2048, 30.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


def register_sensor_bands(lookup_table, dataset_config, sensor_name):
    """Register all spectral bands for a sensor in the lookup table."""
    bands_info = SENSOR_CONFIG[sensor_name]["bands_info"]
    for band_name, info in bands_info.items():
        wl = info["central_wavelength"]
        bw = info["bandwidth"]
        key = (bw, wl)
        if key not in lookup_table.table_wave:
            lookup_table.table_wave[key] = len(lookup_table.table_wave)
            print(f"  Registered {band_name}: λ={wl}nm, Δλ={bw}nm → idx={lookup_table.table_wave[key]}")


# =============================================================================
# PRETRAINED ENCODER LOADING
# =============================================================================

def load_pretrained_encoder(model, ckpt_path):
    print(f"\n{'='*60}")
    print(f"  Loading pretrained encoder from: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        full_state = ckpt["state_dict"]
        encoder_state = {k[len("encoder."):]: v
                         for k, v in full_state.items() if k.startswith("encoder.")}
    elif "encoder" in ckpt:
        encoder_state = ckpt["encoder"]
    else:
        raise ValueError(f"Checkpoint keys: {list(ckpt.keys())}")

    model_state = model.encoder.state_dict()
    compatible = {}
    skipped = []
    for k, v in encoder_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v
        elif k in model_state:
            skipped.append((k, v.shape, model_state[k].shape))

    result = model.encoder.load_state_dict(compatible, strict=False)
    print(f"  Loaded: {len(compatible)}, Skipped: {len(skipped)}, "
          f"Fresh: {len(result.missing_keys) - len(skipped)}")
    for k, s, d in skipped:
        print(f"    - {k}: {s} ≠ {d}")
    print(f"{'='*60}\n")
    return model


# =============================================================================
# DATA MODULE
# =============================================================================

class MultiEarthDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        train_sensor: str,
        test_sensor: str,
        look_up,
        config_model: dict,
        n_timesteps: int = 3,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.train_sensor = train_sensor
        self.test_sensor = test_sensor
        self.look_up = look_up
        self.config_model = config_model
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if hasattr(self, "_setup_done") and self._setup_done:
            return
        self._setup_done = True

        common = dict(
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            look_up=self.look_up,
            config_model=self.config_model,
            n_timesteps=self.n_timesteps,
        )

        # ── Training (one sensor) ─────────────────────────
        self.train_dataset = MultiEarthDataset(
            split="train", sensor=self.train_sensor,
            augment=True, **common)

        # ── Validation (same sensor as training) ──────────
        self.val_dataset = MultiEarthDataset(
            split="val", sensor=self.train_sensor,
            augment=False, **common)

        # ── Test: same sensor ─────────────────────────────
        self.test_same = MultiEarthDataset(
            split="test", sensor=self.train_sensor,
            augment=False, **common)

        # ── Test: cross sensor ────────────────────────────
        if self.test_sensor != self.train_sensor:
            self.test_cross = MultiEarthDataset(
                split="test", sensor=self.test_sensor,
                augment=False, **common)
        else:
            self.test_cross = None

        print(f"\n[MultiEarth-DM] Summary:")
        print(f"  Train: {len(self.train_dataset)} samples ({self.train_sensor})")
        print(f"  Val:   {len(self.val_dataset)} samples ({self.train_sensor})")
        print(f"  Test same:  {len(self.test_same)} samples ({self.train_sensor})")
        if self.test_cross:
            print(f"  Test cross: {len(self.test_cross)} samples ({self.test_sensor})")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None), sampler=sampler,
            num_workers=self.num_workers, collate_fn=collate_multitask,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_same, shuffle=False)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="MultiEarth Atomizer Training")

# Data
parser.add_argument("--data_dir", type=str, default="./data/multi_earth/")
parser.add_argument("--csv_path", type=str, default="multiearth_split.csv")
parser.add_argument("--sensor", type=str, default="s2", choices=["s2", "l8"])
parser.add_argument("--eval_sensor", type=str, default=None, choices=["s2", "l8", None])
parser.add_argument("--n_timesteps", type=int, default=3)

# Model
parser.add_argument("--config_model", type=str,
                    default="config_test-Atomiser_Atos_One.yaml")
parser.add_argument("--pretrained_encoder", type=str, default=None)
parser.add_argument("--ckpt_path", type=str, default=None,
                    help="Resume from Lightning checkpoint")

# Training
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=1)

# Experiment
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--project", type=str, default="multiearth")
parser.add_argument("--precision", type=str, default="bf16-mixed")

args = parser.parse_args()

if args.eval_sensor is None:
    args.eval_sensor = "l8" if args.sensor == "s2" else "s2"


# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

# Register bands for BOTH sensors (so cross-sensor test works)
print("\n[MultiEarth] Registering spectral bands:")
dataset_config = read_yaml(bands_yaml_path)
register_sensor_bands(lookup_table, dataset_config, "s2")
register_sensor_bands(lookup_table, dataset_config, "l8")

print(f"\n[MultiEarth] Lookup table: {len(lookup_table.table_wave)} spectral entries")


# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    pretrain_tag = "pretrained" if args.pretrained_encoder else "scratch"
    run_name = f"ME_{args.xp_name}_{args.sensor}→{args.eval_sensor}_{pretrain_tag}"
    wandb.init(
        name=run_name, project=args.project,
        config={
            **config_model,
            "sensor": args.sensor,
            "eval_sensor": args.eval_sensor,
            "n_timesteps": args.n_timesteps,
            "pretrained": args.pretrained_encoder is not None,
        })
    wandb_logger = WandbLogger(project=args.project)


# =============================================================================
# DATA MODULE
# =============================================================================
data_module = MultiEarthDataModule(
    data_dir=args.data_dir,
    csv_path=args.csv_path,
    train_sensor=args.sensor,
    test_sensor=args.eval_sensor,
    look_up=lookup_table,
    config_model=config_model,
    n_timesteps=args.n_timesteps,
    batch_size=config_model["dataset"]["batchsize"],
    num_workers=args.num_workers,
)
data_module.setup()


# =============================================================================
# MODEL
# =============================================================================
model = Model_Pretrain(
    config=config_model, wand=True, name=args.xp_name,
    transform=None, lookup_table=lookup_table)

if args.pretrained_encoder:
    model = load_pretrained_encoder(model, args.pretrained_encoder)


# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================
ckpt_dir = f"./checkpoints/multi_earth/{args.xp_name}/"
os.makedirs(ckpt_dir, exist_ok=True)

MONITOR_METRIC = f"val_{TASK_NAME}_mIoU"

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"multiearth_{args.xp_name}-{{epoch:02d}}-{{{MONITOR_METRIC}:.4f}}",
        monitor=MONITOR_METRIC, mode="max", save_top_k=1, verbose=True),
    LearningRateMonitor(logging_interval="step"),
    pl.callbacks.EarlyStopping(
        monitor=MONITOR_METRIC, mode="max", patience=15, verbose=True),
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1, max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu", precision=args.precision,
    logger=wandb_logger, log_every_n_steps=10,
    callbacks=callbacks, default_root_dir=ckpt_dir,
    gradient_clip_val=1.0, accumulate_grad_batches=args.grad_accum,
)


# =============================================================================
# TRAIN
# =============================================================================
print(f"\n{'='*60}")
print(f"  Atomizer MultiEarth Training")
print(f"{'='*60}")
print(f"  Train sensor: {args.sensor}")
print(f"  Eval sensor:  {args.eval_sensor}")
print(f"  Timesteps:    {args.n_timesteps}")
print(f"  Pretrained:   {args.pretrained_encoder or 'no'}")
print(f"{'='*60}\n")

trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)


# =============================================================================
# TEST — SAME SENSOR
# =============================================================================
print(f"\n{'='*60}")
print(f"Testing: {args.sensor}→{args.sensor} (same sensor)")
print(f"{'='*60}")
results_same = trainer.test(model, datamodule=data_module, ckpt_path="best")

if results_same and wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    for k, v in results_same[0].items():
        wandb_logger.experiment.summary[f"same_sensor/{k}"] = v


# =============================================================================
# TEST — CROSS SENSOR
# =============================================================================
if data_module.test_cross is not None:
    cross_loader = data_module._make_loader(data_module.test_cross, shuffle=False)

    print(f"\n{'='*60}")
    print(f"Testing: {args.eval_sensor}→{args.sensor} encoder (cross-sensor, NO interpolation)")
    print(f"{'='*60}")
    results_cross = trainer.test(model, dataloaders=cross_loader, ckpt_path="best")

    if results_cross and wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
        for k, v in results_cross[0].items():
            wandb_logger.experiment.summary[f"cross_sensor/{k}"] = v


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
if results_same:
    miou = results_same[0].get(f"test_{TASK_NAME}_mIoU", "?")
    print(f"  {args.sensor}→{args.sensor}: mIoU={miou:.4f}"
          if isinstance(miou, float) else f"  {args.sensor}→{args.sensor}: mIoU={miou}")
if data_module.test_cross is not None and results_cross:
    miou = results_cross[0].get(f"test_{TASK_NAME}_mIoU", "?")
    print(f"  {args.eval_sensor}→{args.sensor} (cross): mIoU={miou:.4f}"
          if isinstance(miou, float) else f"  cross: mIoU={miou}")
print(f"{'='*60}")

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/multiearth_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)