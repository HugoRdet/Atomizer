"""
PASTIS-HD Training Script — Multi-temporal Crop Segmentation
==============================================================

Train Atomizer-IO on PASTIS-HD for crop type segmentation.
  - S2 (10 bands, multi-temporal) — always enabled
  - S1A (2 bands, multi-temporal) — optional via --use_s1
  - SPOT6 (3 bands, single frame) — optional via --use_spot

Splits (fold-based):
  - Train: folds 1, 2, 3
  - Val:   fold 4
  - Test:  fold 5

Examples:
    # S2-only, from scratch
    python train_pastis.py --xp_name pastis_s2only

    # S2 + S1 + SPOT, full temporal
    python train_pastis.py --xp_name pastis_full \
        --use_s1 --use_spot --multi_temporal 10
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

from training.utils import read_yaml, Lookup_encoding
from training.trainer_PASTIS import PASTISTrainer
from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
from training.utils.datasets.token_grouping import collate_multitask
from training.utils.datasets.token_builder import TokenBuilder

from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.callbacks.segmentation_viz_callback import SegmentationVizCallback

# =============================================================================
# KNOWN RESOLUTIONS
# =============================================================================

ALL_KNOWN_RESOLUTIONS = {
    1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048,
}


def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


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
        print(f"  Extracted {len(encoder_state)} encoder keys from Lightning checkpoint")
    elif "encoder" in ckpt:
        encoder_state = ckpt["encoder"]
        print(f"  Loaded {len(encoder_state)} encoder keys from raw checkpoint")
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
# COLLATE
# =============================================================================

def pastis_collate(samples):
    batch = collate_multitask(samples)
    if "queries" not in batch and "tasks" in batch:
        task_data = next(iter(batch["tasks"].values()))
        batch["queries"] = task_data["queries"]
        batch["queries_mask"] = task_data["queries_mask"]
    return batch


# =============================================================================
# DATAMODULE
# =============================================================================

class PastisDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root_path: str,
        config_model: dict,
        look_up,
        batch_size: int = 4,
        num_workers: int = 4,
        use_s1: bool = True,
        use_spot: bool = True,
    ):
        super().__init__()
        self.root_path    = root_path
        self.config_model = config_model
        self.look_up      = look_up
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.use_s1       = use_s1
        self.use_spot     = use_spot

    def _make_dataset(self, mode: str):
        return PastisHDDataset(
            root_path=self.root_path,
            mode=mode,
            config_model=self.config_model,
            look_up=self.look_up,
            use_s1=self.use_s1,
            use_spot=self.use_spot,
        )

    def setup(self, stage=None):
        if hasattr(self, "_setup_done") and self._setup_done:
            return
        self._setup_done = True

        self.train_dataset = self._make_dataset("train")
        self.val_dataset   = self._make_dataset("validation")
        self.test_dataset  = self._make_dataset("test")

        modalities = ["S2"]
        if self.use_s1:
            modalities.append("S1")
        if self.use_spot:
            modalities.append("SPOT")

        print(f"\n[PASTIS-DM] Summary:")
        print(f"  Modalities: {' + '.join(modalities)}")
        print(f"  Train: {len(self.train_dataset)} patches (folds 1,2,3)")
        print(f"  Val:   {len(self.val_dataset)} patches (fold 4)")
        print(f"  Test:  {len(self.test_dataset)} patches (fold 5)")

    def _make_loader(self, dataset, shuffle=False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None), sampler=sampler,
            num_workers=self.num_workers, collate_fn=pastis_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)


# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="PASTIS-HD Training")
parser.add_argument("--xp_name",        type=str, required=True)
parser.add_argument("--config_model",   type=str,
                    default="config_test-Atomiser_Atos_One.yaml")
parser.add_argument("--data_dir",       type=str, default="./data/PASTIS-HD")
parser.add_argument("--ckpt_path",      type=str, default=None,
                    help="Resume training from Lightning checkpoint")
parser.add_argument("--pretrained_encoder", type=str, default=None,
                    help="Load pretrained encoder weights (no head)")
parser.add_argument("--num_workers",    type=int, default=4)
parser.add_argument("--grad_accum",     type=int, default=1)

# Test-only mode
parser.add_argument("--test_only",      action="store_true",
                    help="Skip training, load checkpoint from config trainer.checkpoint_path "
                         "(or --ckpt_path) and run test split only")

# Modality toggles
parser.add_argument("--use_s1",         action="store_true",
                    help="Enable S1A SAR data (default: S2-only)")
parser.add_argument("--use_spot",       action="store_true",
                    help="Enable SPOT6 RGB data (default: S2-only)")

# Temporal config
parser.add_argument("--multi_temporal", type=int, default=None,
                    help="Number of temporal frames (overrides config)")

args = parser.parse_args()

# =============================================================================
# CONFIG & LOOKUP
# =============================================================================
config_model         = read_yaml("./training/configs/" + args.config_model)
bands_yaml_path      = "./data/bands_info/bands.yaml"
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

# Override multi_temporal in config if specified via CLI
if args.multi_temporal is not None:
    if "dataset" not in config_model:
        config_model["dataset"] = {}
    config_model["dataset"]["multi_temporal"] = args.multi_temporal

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), read_yaml(bands_yaml_path), config_model)
register_all_resolutions(lookup_table)

# Register VV-VH SAR channel (not in bands.yaml, needed for S1 3rd band)
if args.use_s1:
    lookup_table.register_abstract_channel("VV_VH")

# Build modality description for logging
modalities = ["S2"]
if args.use_s1:
    modalities.append("S1")
if args.use_spot:
    modalities.append("SPOT")
modality_str = "+".join(modalities)

multi_temporal = config_model.get("dataset", {}).get("multi_temporal", 10)

print(f"\n[PASTIS] Experiment:   {args.xp_name}")
print(f"[PASTIS] Data dir:     {args.data_dir}")
print(f"[PASTIS] Modalities:   {modality_str}")
print(f"[PASTIS] Temporal:     {multi_temporal} frames (uniform via linspace)")

# =============================================================================
# WANDB
# =============================================================================
wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    pretrain_tag = "pretrained" if args.pretrained_encoder else "scratch"
    run_name = f"PASTIS_{args.xp_name}_{modality_str}_{pretrain_tag}"
    wandb.init(
        name=run_name, project="PASTIS",
        config={
            **config_model,
            "modalities":     modalities,
            "use_s1":         args.use_s1,
            "use_spot":       args.use_spot,
            "multi_temporal": multi_temporal,
        },
    )
    wandb_logger = WandbLogger(project="PASTIS")

# =============================================================================
# DATA MODULE
# =============================================================================
data_module = PastisDataModule(
    root_path=args.data_dir,
    config_model=config_model,
    look_up=lookup_table,
    batch_size=config_model["trainer"]["batchsize"],
    num_workers=args.num_workers,
    use_s1=args.use_s1,
    use_spot=args.use_spot,
)
data_module.setup()
print(f"[PASTIS] Lookup table: {len(lookup_table.table_wave)} entries")

# =============================================================================
# MODEL
# =============================================================================
model = PASTISTrainer(
    config=config_model, wand=True, name=args.xp_name,
    transform=None, lookup_table=lookup_table,
)

if args.pretrained_encoder:
    model = load_pretrained_encoder(model, args.pretrained_encoder)

# =============================================================================
# CALLBACKS & TRAINER
# =============================================================================
ckpt_dir = f"./checkpoints/pastis/"
os.makedirs(ckpt_dir, exist_ok=True)

viz_callback = SegmentationVizCallback(
    dataset_preset="pastis",
    sample_indices=[0, 1, 2],
    log_every_n_epochs=1,
    use_wandb=True,
)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"pastis_{args.xp_name}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU", mode="max",
        save_top_k=1, verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"pastis_{args.xp_name}-last-{{epoch:02d}}",
        every_n_epochs=1, save_top_k=1, save_last=True, verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
    viz_callback,
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    use_distributed_sampler=False,
    devices=-1, max_epochs=config_model["trainer"]["epochs"],
    accelerator="gpu", precision="bf16-mixed",
    logger=wandb_logger, log_every_n_steps=5,
    callbacks=callbacks, default_root_dir=ckpt_dir,
    accumulate_grad_batches=args.grad_accum,
)

# =============================================================================
# TRAIN / TEST
# =============================================================================
if args.test_only:
    # ── Test-only mode: load checkpoint and run test split ──────────────
    ckpt_to_load = (
        args.ckpt_path
        or config_model.get("trainer", {}).get("checkpoint_path")
    )
    if ckpt_to_load is None:
        raise ValueError(
            "--test_only requires a checkpoint. Either pass --ckpt_path "
            "or set trainer.checkpoint_path in the config YAML."
        )

    print(f"\n{'='*60}")
    print(f"  PASTIS-HD — TEST ONLY")
    print(f"  Checkpoint: {ckpt_to_load}")
    print(f"  Modalities: {modality_str}")
    print(f"  Temporal:   {multi_temporal} frames")
    print(f"{'='*60}\n")

    # Load weights into the already-constructed model
    ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[PASTIS] Loaded checkpoint — "
          f"missing: {len(result.missing_keys)}, "
          f"unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"[PASTIS] First 5 missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[PASTIS] First 5 unexpected: {result.unexpected_keys[:5]}")

    # Run validation + test
    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
else:
    # ── Normal training ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PASTIS-HD — {modality_str}")
    print(f"  Temporal: {multi_temporal} frames (linspace)")
    print(f"  Train: folds 1,2,3 → Val: fold 4 → Test: fold 5")
    print(f"{'='*60}\n")

    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    with open(f"training/wandb_runs/pastis_{args.xp_name}.txt", "w") as f:
        f.write(wandb.run.id)