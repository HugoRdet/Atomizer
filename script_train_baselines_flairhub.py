"""
FLAIR-HUB Baseline Training Script (ResNet/ViT/UNet)
======================================================

Train segmentation baselines on FLAIR-HUB with the same multi-modal
multi-resolution input that Atomizer sees, fused via upsampling +
channel concatenation (the standard "zero-padding" baseline).

All modalities are bilinearly upsampled to 0.2m (512×512) and concatenated
into a single channel dimension. With default flags (VHR + DEM + S2 + S1),
the input is [B, 90, 512, 512].

Cross-sensor transfer (matches Atomizer setup):
    Train: --use_vhr --no_use_spot
    Test:  --no_use_vhr --use_spot --test_only --ckpt_path <best>
    Same ResNet weights, SPOT replaces VHR in the leading 4 channels.

Examples
--------
    # ResNet50 on the same FLAIR-HUB subset Atomizer used
    python script_train_flairhub_baselines.py --xp_name resnet50_v1 \\
        --model resnet --resnet_variant resnet50 \\
        --subset_indices ./data/FLAIR-HUB/subset_indices.json \\
        --epochs 30

    # Cross-sensor transfer test
    python script_train_flairhub_baselines.py --xp_name resnet50_v1_spot_test \\
        --model resnet --resnet_variant resnet50 \\
        --subset_indices ./data/FLAIR-HUB/subset_indices.json \\
        --ckpt_path ./checkpoints/flairhub_baselines/bl_resnet50_v1_resnet-best.ckpt \\
        --test_only --no_use_vhr --use_spot
"""

import os
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping,
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

seed_everything(42, workers=True)

# Project imports
from training.utils.datasets_baselines.utils_dataset_flairhub_baselines import (
    FlairHubBaselineDataset,
)


from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.ResNet.model_resnet_fuse     import build_resnet_upernet_per_modality
from training.VIT.model_vit_upernet     import ViTUPerNet
from training.VIT.model_vit_fuse        import build_vit_upernet_per_modality
from training.unet.model_unet           import UNet
from training.trainer_baselines        import (
    BaselineTrainer, TASK_CLASS_NAMES,
)


# =============================================================================
# Register FLAIR-HUB class names in the trainer's task registry.
# (Modifies module-level dict — safe because we own the import.)
# =============================================================================
TASK_CLASS_NAMES["flairhub"] = {
    0:  "building",
    1:  "greenhouse",
    2:  "swimming_pool",
    3:  "impervious",
    4:  "pervious",
    5:  "bare_soil",
    6:  "water",
    7:  "snow",
    8:  "herbaceous",
    9:  "agricultural",
    10: "plowed",
    11: "vineyard",
    12: "deciduous",
    13: "coniferous",
    14: "brushwood",
    15: "clear_cut",
    16: "ligneous",
    17: "mixed",
    18: "undefined",
}


# =============================================================================
# COLLATE
# =============================================================================

def flairhub_collate(batch):
    """
    Stack image and target. Supports both:
      - Concat mode:       batch[i]["image"] = {"flairhub": tensor}
      - Per-modality mode: batch[i]["image"] = {"flairhub_pm": dict_of_tensors}

    For per-modality mode, the inner dict is collated key-by-key
    (each branch tensor stacked across batch).
    """
    image_keys = list(batch[0]["image"].keys())
    images = {}
    for k in image_keys:
        v0 = batch[0]["image"][k]
        if isinstance(v0, dict):
            # Per-modality: inner dict of branch tensors
            images[k] = {
                bk: torch.stack([s["image"][k][bk] for s in batch])
                for bk in v0.keys()
            }
        else:
            # Concat: single tensor per sample
            images[k] = torch.stack([s["image"][k] for s in batch])

    targets  = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# ---------------------------------------------------------------------------
# Per-modality dataset wrapper
# ---------------------------------------------------------------------------
# `BaselineTrainer` accesses `batch["image"][self.modality]` to get its input.
# For per-modality runs, we need the model to receive a dict of branch
# tensors. We put that dict under a single synthetic modality key
# ("flairhub_pm") so the trainer's existing access pattern still works:
#   batch["image"]["flairhub_pm"] == {"optical": ..., "dem": ..., ...}
# The model's forward then takes that dict directly.

class _PerModalityWrap(torch.utils.data.Dataset):
    """Wraps a FlairHubBaselineDataset(per_modality=True) so its 'image'
    field holds a SINGLE key 'flairhub_pm' whose value is the inner
    branch dict. This keeps BaselineTrainer's access pattern intact."""

    def __init__(self, ds):
        self.ds = ds
        # Forward attributes used elsewhere (e.g. patch_rows for subset filter).
        self.patch_rows = ds.patch_rows
        self.NUM_CLASSES = ds.NUM_CLASSES
        self.IGNORE_INDEX = ds.IGNORE_INDEX
        self.num_channels = getattr(ds, "num_channels", 0)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # sample["image"] is currently a dict like {"optical": ..., "dem": ..., ...}
        sample["image"] = {"flairhub_pm": sample["image"]}
        return sample


# =============================================================================
# MODEL BUILDER
# =============================================================================

class _PerModalityModelWrapper(nn.Module):
    """Adapter: BaselineTrainer calls `model(image)`. For per-modality we
    want `model(image_dict)`. The dataset puts the dict under a single
    'flairhub_pm' key, so trainer passes that dict here directly."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, image_dict):
        return self.inner(image_dict)


def build_model(model_name: str, in_channels: int, num_classes: int, args):
    if model_name == "resnet":
        return build_resnet_upernet(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=args.decoder_channels,
        )
    elif model_name == "resnet_pm":
        # Per-modality fusion (FLAIR-HUB style): 4 branches + concat fusion.
        inner = build_resnet_upernet_per_modality(
            num_classes=num_classes,
            use_vhr_or_spot=(args.use_vhr or args.use_spot),
            use_dem=args.use_dem,
            use_s2=args.use_s2,
            use_s1=args.use_s1,
            num_frames=args.multi_temporal,
            resnet_variant=args.resnet_variant,
            branch_target_size=512,
            decoder_channels=args.decoder_channels,
        )
        return _PerModalityModelWrapper(inner)
    elif model_name == "vit_pm":
        # Per-modality ViT: optical/DEM at 512×512 (patch 16), satellite
        # branches at native 10×10 (patch 2 → 5×5 patches per frame), with
        # per-FPN-LTAE for temporal aggregation.
        inner = build_vit_upernet_per_modality(
            num_classes=num_classes,
            use_vhr_or_spot=(args.use_vhr or args.use_spot),
            use_dem=args.use_dem,
            use_s2=args.use_s2,
            use_s1=args.use_s1,
            num_frames=args.multi_temporal,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.decoder_channels,
            optical_dem_img_size=args.img_size,
            optical_dem_patch=args.vit_patch_size,
            sat_img_size=10,
            sat_patch=2,
        )
        return _PerModalityModelWrapper(inner)
    elif model_name == "vit":
        return ViTUPerNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.decoder_channels,
        )
    elif model_name == "unet":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            topology=tuple(args.unet_topology),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# ARGS
# =============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="FLAIR-HUB Baseline Training")
parser.add_argument("--xp_name",  type=str, required=True)
parser.add_argument("--model",    type=str, default="resnet",
                    choices=["resnet", "resnet_pm", "vit", "vit_pm", "unet"])
parser.add_argument("--root_path", type=str, default="./data/FLAIR-HUB")

# Modality flags (must match the Atomizer setup for fair comparison)
parser.add_argument("--use_vhr",  type=str2bool, default=True)
parser.add_argument("--use_spot", type=str2bool, default=False)
parser.add_argument("--spot_norm_as_vhr", type=str2bool, default=False,
                    help="When use_spot=True, normalize SPOT pixel values "
                         "using VHR (aerial) statistics instead of SPOT's own. "
                         "Diagnostic for whether cross-sensor degradation is "
                         "driven by pixel-value distribution shift.")
parser.add_argument("--use_dem",  type=str2bool, default=True)
parser.add_argument("--use_s2",   type=str2bool, default=True)
parser.add_argument("--use_s1",   type=str2bool, default=True)
parser.add_argument("--multi_temporal", type=int, default=6)

# Training
parser.add_argument("--batch_size",   type=int, default=2)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=30)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# UNet
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])

# ViT
parser.add_argument("--img_size",          type=int, default=512)
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])

# Decoder (shared by ViT and ResNet)
parser.add_argument("--decoder_channels", type=int, default=256)

# Subset
parser.add_argument("--subset_indices", type=str, default=None,
                    help="Path to subset_indices.json from select_flair_subset.py.")

# Resume / test
parser.add_argument("--ckpt_path",    type=str, default=None)
parser.add_argument("--wandb_run_id", type=str, default=None)
parser.add_argument("--test_only",    action="store_true")

args = parser.parse_args()


# =============================================================================
# DATASETS
# =============================================================================

print(f"\n{'='*70}")
print(f"  FLAIR-HUB Baseline — {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
_spot_norm_label = ""
if args.use_spot and args.spot_norm_as_vhr:
    _spot_norm_label = " [VHR-norm]"
print(f"  Modalities:  VHR={args.use_vhr}  SPOT={args.use_spot}{_spot_norm_label}  "
      f"DEM={args.use_dem}  S2={args.use_s2}  S1={args.use_s1}")
print(f"  Temporal:    {args.multi_temporal} timesteps (linspace)")
print(f"  Batch size:  {args.batch_size}")
print(f"  Epochs:      {args.epochs}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"{'='*70}\n")

print("[Datasets] Building...")
_per_mod = args.model in ("resnet_pm", "vit_pm")

train_ds = FlairHubBaselineDataset(
    root_path=args.root_path, mode="train",
    use_vhr=args.use_vhr, use_spot=args.use_spot,
    spot_norm_as_vhr=args.spot_norm_as_vhr,
    use_dem=args.use_dem, use_s2=args.use_s2, use_s1=args.use_s1,
    multi_temporal=args.multi_temporal,
    per_modality=_per_mod,
)
val_ds = FlairHubBaselineDataset(
    root_path=args.root_path, mode="validation",
    use_vhr=args.use_vhr, use_spot=args.use_spot,
    spot_norm_as_vhr=args.spot_norm_as_vhr,
    use_dem=args.use_dem, use_s2=args.use_s2, use_s1=args.use_s1,
    multi_temporal=args.multi_temporal,
    per_modality=_per_mod,
)
test_ds = FlairHubBaselineDataset(
    root_path=args.root_path, mode="test",
    use_vhr=args.use_vhr, use_spot=args.use_spot,
    spot_norm_as_vhr=args.spot_norm_as_vhr,
    use_dem=args.use_dem, use_s2=args.use_s2, use_s1=args.use_s1,
    multi_temporal=args.multi_temporal,
    per_modality=_per_mod,
)
print(f"[Datasets] Pre-subset: train={len(train_ds)}  "
      f"val={len(val_ds)}  test={len(test_ds)}")

# Channel count from one sample (concat mode only — per-modality has no
# single channel count since each branch has its own).
NUM_CHANNELS = train_ds.num_channels
if not _per_mod:
    print(f"[Datasets] Channels per sample: {NUM_CHANNELS}")


# ── Subset filtering (matches Atomizer pipeline) ───────────────────
if args.subset_indices is not None:
    import json as _json
    print(f"[Datasets] Loading subset from: {args.subset_indices}")
    with open(args.subset_indices) as f:
        subset = _json.load(f)
    split_key = {
        "train": "train_patch_ids",
        "validation": "val_patch_ids",
        "test": "test_patch_ids",
    }
    for ds, name in [(train_ds, "train"), (val_ds, "validation"),
                     (test_ds, "test")]:
        wanted = set(subset.get(split_key[name], []))
        if not wanted:
            continue
        before = len(ds.patch_rows)
        ds.patch_rows = [r for r in ds.patch_rows
                         if r["patch_id"] in wanted]
        print(f"[Datasets]   {name}: {before} → {len(ds.patch_rows)}")

print(f"[Datasets] Final: train={len(train_ds)}  "
      f"val={len(val_ds)}  test={len(test_ds)}")

# ── Wrap per-modality datasets so the inner branch dict appears under
#   a single key ("flairhub_pm") — keeps BaselineTrainer's modality
#   lookup intact (it does batch["image"][modality]).
if _per_mod:
    train_ds = _PerModalityWrap(train_ds)
    val_ds   = _PerModalityWrap(val_ds)
    test_ds  = _PerModalityWrap(test_ds)
    _MODALITY_KEY = "flairhub_pm"
else:
    _MODALITY_KEY = "flairhub"


# =============================================================================
# DATALOADERS (DDP-aware)
# =============================================================================

def make_loader(dataset, shuffle: bool, batch_size: int):
    """Build a DataLoader. Lightning will wrap the sampler with a
    DistributedSampler automatically at fit-time when DDP is active."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=flairhub_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=shuffle,
    )


train_loader = make_loader(train_ds, shuffle=True,  batch_size=args.batch_size)
val_loader   = make_loader(val_ds,   shuffle=False, batch_size=args.batch_size)
test_loader  = make_loader(test_ds,  shuffle=False, batch_size=args.batch_size)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS,
                    FlairHubBaselineDataset.NUM_CLASSES, args)

trainer_module = BaselineTrainer(
    model=model,
    modality=_MODALITY_KEY,
    temporal=False,                  # we already flatten T into channels (concat mode)
                                     # or handle T inside model (per-modality mode)
    task="flairhub",
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_classes=FlairHubBaselineDataset.NUM_CLASSES,
    ignore_index=FlairHubBaselineDataset.IGNORE_INDEX,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        run_name = f"BL_FLAIR_{args.xp_name}_{args.model}"
        if args.model == "resnet":
            run_name += f"_{args.resnet_variant}"
        init_kwargs = dict(
            name=run_name,
            project="Atomizer-FLAIR-Baselines",
            config={**vars(args), "num_channels": NUM_CHANNELS},
        )
        if args.wandb_run_id is not None:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "must"
            print(f"[WandB] Resuming run {args.wandb_run_id}")
        wandb.init(**init_kwargs)
        wandb_logger = WandbLogger(project="Atomizer-FLAIR-Baselines")
    except Exception as e:
        print(f"[WandB] not available: {e}")


# =============================================================================
# CALLBACKS + TRAINER
# =============================================================================

ckpt_dir = "./checkpoints/flairhub_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU", mode="max",
        save_top_k=1, verbose=True,
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}-last",
        every_n_epochs=1, save_last=True,
    ),
    EarlyStopping(
        monitor="val_mIoU", mode="max",
        patience=args.patience, verbose=True,
    ),
    LearningRateMonitor(logging_interval="step"),
]

trainer = Trainer(
    strategy=DDPStrategy(find_unused_parameters=True),
    devices=-1,
    max_epochs=args.epochs,
    accelerator="gpu",
    precision="bf16-mixed",
    sync_batchnorm=True,                  # safe with DDP + small per-rank batches
    logger=wandb_logger,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=ckpt_dir,
    gradient_clip_val=1.0,
    accumulate_grad_batches=args.grad_accum,
)


# =============================================================================
# TRAIN / TEST
# =============================================================================

if args.test_only:
    if args.ckpt_path is None:
        raise ValueError("--test_only requires --ckpt_path.")

    print(f"\n{'='*70}\n  FLAIR-HUB Baseline — TEST ONLY\n"
          f"  ckpt: {args.ckpt_path}\n{'='*70}\n")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = trainer_module.load_state_dict(state, strict=False)
    print(f"[FLAIR-HUB-BL] missing: {len(result.missing_keys)}  "
          f"unexpected: {len(result.unexpected_keys)}")
    trainer.test(trainer_module, test_loader, verbose=True)
else:
    print(f"\n{'='*70}\n  FLAIR-HUB Baseline — TRAINING\n{'='*70}\n")
    if args.ckpt_path is not None:
        print(f"  Resuming from: {args.ckpt_path}")
    trainer.fit(trainer_module, train_loader, val_loader,
                ckpt_path=args.ckpt_path)

    print(f"\n{'='*70}\n  FLAIR-HUB Baseline — FINAL TEST\n{'='*70}\n")
    trainer.test(trainer_module, test_loader, verbose=True, ckpt_path="best")


# =============================================================================
# SAVE WANDB RUN ID
# =============================================================================

if wandb_logger and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    os.makedirs("training/wandb_runs", exist_ok=True)
    run_id = wandb.run.id
    out = f"training/wandb_runs/bl_flairhub_{args.xp_name}_{args.model}.txt"
    with open(out, "w") as f:
        f.write(run_id)
    print(f"[WandB] run_id={run_id}  saved to {out}")