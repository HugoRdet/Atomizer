"""
EuroSAT-SAR Baseline Training Script
=======================================

Train classification baselines on EuroSAT-SAR (10-class, S2 optical 13
bands + S1 SAR VV/VH fused into 15 channels, 64×64).

Mirror of script_train_eurosat_baselines.py — same models, same trainer.
Differences: 15 input channels (fused), MODALITY_KEY="fused", dataset
reads the raw class-folder EuroSAT_MS/EuroSAT-SAR trees (paired by
filename) instead of geo-bench HDF5, and checkpoint selection uses
val_top1 (not val_macro_f1) to match the Atomizer EuroSAT-SAR script.

Models:
  - resnet : ResNetClassifier (variant via --resnet_variant)
  - vit    : ViTClassifier
  - perceiver : PerceiverCls
  - ramen  : RAMENClassifier — multi-modal spectral tokenization + CLS
             token classification head. Unlike Sen1Floods11's RAMENUPerNet,
             no sliding-window tiling is needed: 64×64 tiles are small
             enough to fit in one forward pass even at native resolution.

RAMEN and modality dropping:
    EuroSATSARBaselineDataset already zeroes dropped/non-kept bands at
    the DATASET level (--bands_keep/--bands_drop), before RAMEN ever sees
    the tensor. So RAMEN needs no drop-aware wrapper — just a plain
    adapter that reshapes the dataset's merged "fused" tensor into
    RAMEN's {"optical","sar"} dict (see RAMENInputAdapter below). Existing
    --bands_keep/--bands_drop ablation workflows work unchanged for RAMEN.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training.

Modality ablation:
    --bands_keep / --bands_drop take band names from
    EuroSATSARBaselineDataset.ALL_BAND_NAMES (Blue, Green, Red, NIR,
    RedEdge1-4, SWIR1, SWIR2, CoastalAerosol, WaterVapour, Cirrus, VV, VH).
    Dropped/non-kept channels are zeroed (fixed 15-channel input shape),
    matching the Atomizer modality-drop script's semantics — so a single
    trained checkpoint can be re-tested under different --test_only +
    --bands_drop combinations without retraining.

Examples:
    python script_train_eurosat_sar_baselines.py --xp_name resnet50_fused \
        --model resnet --resnet_variant resnet50 \
        --batch_size 32 --lr 1e-4 --epochs 80

    python script_train_eurosat_sar_baselines.py --xp_name ramen_fused \
        --model ramen \
        --batch_size 32 --lr 1e-4 --epochs 80

    # SAR-only ablation on an existing RAMEN checkpoint:
    python script_train_eurosat_sar_baselines.py --xp_name ramen_sar_only \
        --model ramen \
        --test_only ./checkpoints/eurosat_sar_baselines/bl_ramen_fused_ramen-last.ckpt \
        --bands_drop Blue Green Red NIR RedEdge1 RedEdge2 RedEdge3 RedEdge4 \
                     SWIR1 SWIR2 CoastalAerosol WaterVapour Cirrus
"""

import os
import argparse

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_eurosat_sar_baseline import (
    EuroSATSARBaselineDataset,
)
from training.ResNet.model_resnet_upernet import build_resnet_classifier
from training.VIT.model_vit_upernet import ViTClassifier
from training.perceiverIO.perceiver_cls import PerceiverCls
from training.RAMEN.ramen_classifier import build_ramen_classifier  # adjust import path
from training.trainer_baselines_classification import ClassificationBaselineTrainer


NUM_CLASSES  = EuroSATSARBaselineDataset.NUM_CLASSES
NUM_CHANNELS = EuroSATSARBaselineDataset.NUM_S2_CHANNELS + EuroSATSARBaselineDataset.NUM_S1_CHANNELS  # 15
MODALITY_KEY = "fused"


# =============================================================================
# RAMEN band metadata — derived from the dataset's OWN band naming/order,
# not a hardcoded duplicate, so it can't drift out of sync.
# =============================================================================

# Physical central wavelength (nm) per Sentinel-2 band code. EuroSAT's
# NAME_TO_S2CODE maps its own naming (Blue, RedEdge1, ...) to these codes.
_S2_CODE_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B08A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
    "B12": 2202.4,
}

RAMEN_S2_WAVELENGTHS = {
    name: _S2_CODE_WAVELENGTHS_NM[EuroSATSARBaselineDataset.NAME_TO_S2CODE[name]]
    for name in EuroSATSARBaselineDataset.S2_NAME_ORDER
}

# RadarProjector's pol_map only has ascending/descending-tagged keys
# (e.g. "asc_vv"); EuroSAT-SAR doesn't expose pass direction, so this
# defaults to "asc_*", same convention as the Sen1Floods11 scripts.
RAMEN_S1_POLARIZATIONS = {"VV": "asc_vv", "VH": "asc_vh"}

RAMEN_INPUT_BANDS = {
    "optical": EuroSATSARBaselineDataset.S2_NAME_ORDER,
    "sar": ["VV", "VH"],
}
RAMEN_WAVELENGTHS = {
    "optical": RAMEN_S2_WAVELENGTHS,
    "sar": RAMEN_S1_POLARIZATIONS,
}


# =============================================================================
# RAMEN INPUT ADAPTER
# =============================================================================

class RAMENInputAdapter(nn.Module):
    """
    Splits the dataset's merged image["fused"] : [B,15,H,W] tensor into
    RAMEN's expected {"optical": [B,13,H,W], "sar": [B,2,H,W]}.

    No modality-drop logic here: EuroSATSARBaselineDataset already zeroes
    dropped/non-kept bands at the DATASET level (see --bands_keep/
    --bands_drop in this script and `active_mask` in the dataset), so by
    the time this adapter sees "fused" any ablation is already applied.
    This adapter's only job is reshaping for RAMEN's per-modality input.
    """
    expects_full_image_dict = True

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: dict, **kwargs):
        merged = x[MODALITY_KEY]  # [B, 15, H, W]
        optical = merged[:, :EuroSATSARBaselineDataset.NUM_S2_CHANNELS]
        sar = merged[:, EuroSATSARBaselineDataset.NUM_S2_CHANNELS:
                        EuroSATSARBaselineDataset.NUM_S2_CHANNELS
                        + EuroSATSARBaselineDataset.NUM_S1_CHANNELS]
        return self.model({"optical": optical, "sar": sar}, **kwargs)


# =============================================================================
# COLLATE  (unchanged from EuroSAT — already generic over sensor keys, and
# this dataset only ever emits one key: "fused")
# =============================================================================

def eurosat_sar_collate(batch):
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, args):
    if model_name == "resnet":
        return build_resnet_classifier(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    elif model_name == "vit":
        return ViTClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            dropout=args.dropout,
        )
    elif model_name == "perceiver":
        return PerceiverCls(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=args.img_size,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            depth=args.perceiver_depth,
            cross_heads=args.cross_heads,
            latent_heads=args.latent_heads,
            cross_dim_head=args.cross_dim_head,
            latent_dim_head=args.latent_dim_head,
            self_per_cross_attn=args.self_per_cross_attn,
            weight_tie_layers=(not args.no_weight_tie),
            num_freq_bands=args.num_freq_bands,
            max_freq=args.max_freq,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
        )
    elif model_name == "ramen":
        base = build_ramen_classifier(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=num_classes,
            input_size=args.img_size,  # 64 — dataset always returns 64x64 tiles
            embed_dim=args.ramen_embed_dim,
            depth=args.ramen_depth,
            num_heads=args.ramen_num_heads,
            input_res=args.ramen_input_res,
            res=args.ramen_res,
            dropout=args.dropout,
        )
        return RAMENInputAdapter(base)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="EuroSAT-SAR Baseline Classification")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["resnet", "vit", "perceiver", "ramen"])
parser.add_argument("--data_dir",  type=str, default="./data",
                    help="Parent dir containing EuroSAT_MS/ and EuroSAT-SAR/")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

parser.add_argument("--resume", action="store_true",
                    help="Resume training from the '-last' checkpoint for this "
                         "xp_name/model, if one exists. Restores full trainer "
                         "state (epoch, optimizer, LR schedule, EarlyStopping/"
                         "ModelCheckpoint state) — not just weights. Safe to "
                         "pass on every submission: if no checkpoint is found "
                         "yet, training just starts fresh. Ignored if "
                         "--test_only is set.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Explicit checkpoint path to resume from (overrides "
                         "--resume auto-detection).")

parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)
parser.add_argument("--dropout",      type=float, default=0.1)
parser.add_argument("--label_smoothing", type=float, default=0.0)

# Modality ablation (zeroes channels, keeps 15-channel input shape)
parser.add_argument("--bands_keep", type=str, nargs="+", default=None,
                    help="Band names to keep (others zeroed). Default: all 15.")
parser.add_argument("--bands_drop", type=str, nargs="+", default=None,
                    help="Band names to zero out (must be within bands_keep, if set).")

# Image size
parser.add_argument("--img_size",  type=int, default=64,
                    help="ViT/RAMEN positional embedding size (must equal 64).")

# ViT
parser.add_argument("--vit_embed_dim",  type=int, default=384)
parser.add_argument("--vit_depth",      type=int, default=12)
parser.add_argument("--vit_num_heads",  type=int, default=6)
parser.add_argument("--vit_patch_size", type=int, default=8,
                    help="64 / patch_size must be int; 8 → 8×8 patches per image")

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN
parser.add_argument("--ramen_embed_dim", type=int, default=384,
                    help="Matches --vit_embed_dim's default for a fair "
                         "parameter-count comparison; independent knob.")
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=6)
parser.add_argument("--ramen_input_res", type=float, default=10.0,
                    help="Native GSD (m/px) of EuroSAT-SAR imagery.")
parser.add_argument("--ramen_res",       type=float, default=40.0,
                    help="Working resolution (m/px). Default equals "
                         "--ramen_input_res (no resampling, full native "
                         "detail) — reasonable at 64x64 since token count "
                         "is already modest. Increase to trade detail for "
                         "speed, e.g. for a resolution sweep.")

# Perceiver-IO
# NOTE: the original standalone EuroSAT Perceiver script defaults to
# --batch_size 4 (not this script's shared default of 32) because
# num_latents x latent_dim x depth cross-attention is memory-heavy at
# larger batch sizes. This shared script does NOT auto-change --batch_size
# based on --model, so pass --batch_size explicitly for Perceiver runs
# (the SLURM script for Perceiver does this).
parser.add_argument("--num_latents",         type=int, default=512)
parser.add_argument("--latent_dim",          type=int, default=768)
parser.add_argument("--perceiver_depth",     type=int, default=1)
parser.add_argument("--cross_heads",         type=int, default=8)
parser.add_argument("--latent_heads",        type=int, default=8)
parser.add_argument("--cross_dim_head",      type=int, default=64)
parser.add_argument("--latent_dim_head",     type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=12)
parser.add_argument("--no_weight_tie",       action="store_true",
                    help="Disable weight-tying across Perceiver encoder blocks.")
parser.add_argument("--num_freq_bands",      type=int, default=16)
parser.add_argument("--max_freq",            type=float, default=16.0)
parser.add_argument("--attn_dropout",        type=float, default=0.0)
parser.add_argument("--ff_dropout",          type=float, default=0.0)

# Band-dropout augmentation (train only) — gives baselines training-time
# exposure to missing modalities/bands, for fairer comparison against
# Atomiser's own token-dropout augmentation at the modality-drop eval.
# See EuroSATSARBaselineDataset's docstring for exact semantics. Applied
# on top of --bands_keep/--bands_drop, not instead of it.
parser.add_argument("--band_dropout", action="store_true", default=True,
                    help="Enable band-dropout augmentation during training "
                         "(default: on). Set the probabilities below to the "
                         "SAME values used on the Atomiser side for a fair "
                         "comparison.")
parser.add_argument("--no_band_dropout", dest="band_dropout", action="store_false",
                    help="Disable band-dropout augmentation (e.g. for an "
                         "ablation isolating its effect).")
parser.add_argument("--p_dropout_applied", type=float, default=0.5,
                    help="Probability a given training sample gets ANY "
                         "band dropout applied (the rest keep all bands).")
parser.add_argument("--p_whole_modality", type=float, default=0.5,
                    help="Given dropout is applied, probability it's a "
                         "whole-modality drop (all S1 or all S2) rather "
                         "than a random per-band subset.")
parser.add_argument("--p_band_drop", type=float, default=0.15,
                    help="Given a per-band (not whole-modality) drop, the "
                         "independent probability each of the 15 bands is "
                         "individually zeroed.")

args = parser.parse_args()

bands_cfg = {"keep": args.bands_keep, "drop": args.bands_drop}


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  EuroSAT-SAR Baseline Classification")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
if args.model == "ramen":
    print(f"  Embed dim:   {args.ramen_embed_dim}, depth={args.ramen_depth}, "
          f"heads={args.ramen_num_heads}")
    print(f"  Resolution:  input_res={args.ramen_input_res}, res={args.ramen_res} "
          f"({'no resampling' if args.ramen_input_res == args.ramen_res else 'resampled'})")
if args.model == "perceiver":
    print(f"  Latents:     {args.num_latents} x {args.latent_dim}, depth={args.perceiver_depth}")
    print(f"  Tokens:      {args.img_size ** 2:,} per sample")
    if args.batch_size > 8:
        print(f"  ⚠ WARN: --batch_size={args.batch_size} for Perceiver — the "
              f"reference EuroSAT Perceiver script defaults to 4 due to "
              f"cross-attention memory cost. Watch for OOM.")
print(f"  Channels:    {NUM_CHANNELS} (13 optical + VV + VH, fused)")
print(f"  Bands keep:  {args.bands_keep or 'ALL'}")
print(f"  Bands drop:  {args.bands_drop or 'none'}")
print(f"  Resume:      {'ON (' + (args.resume_from or 'auto-detect last ckpt') + ')' if (args.resume or args.resume_from) else 'OFF'}")
print(f"  Band drop:   {'ON (p_applied=' + str(args.p_dropout_applied) + ', p_whole_mod=' + str(args.p_whole_modality) + ', p_band=' + str(args.p_band_drop) + ')' if args.band_dropout else 'OFF'}")
print(f"  Classes:     {NUM_CLASSES}")
print(f"  Patch size:  64×64")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = EuroSATSARBaselineDataset(root_path=args.data_dir, mode="train",
                                      augment=True,  bands=bands_cfg,
                                      band_dropout=args.band_dropout,
                                      p_dropout_applied=args.p_dropout_applied,
                                      p_whole_modality=args.p_whole_modality,
                                      p_band_drop=args.p_band_drop)
val_ds   = EuroSATSARBaselineDataset(root_path=args.data_dir, mode="validation",
                                      augment=False, bands=bands_cfg)
# band_dropout intentionally not passed to val_ds/test_ds: the dataset
# gates it to mode=="train" internally regardless of the constructor
# default, so val/test are never augmented either way.
test_ds  = EuroSATSARBaselineDataset(root_path=args.data_dir, mode="test",
                                      augment=False, bands=bands_cfg)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=eurosat_sar_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                           shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                           shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

trainer_module = ClassificationBaselineTrainer(
    model=model,
    # RAMENInputAdapter consumes the full image dict (see
    # ClassificationBaselineTrainer._get_image); `modality` is unused in
    # that case, only shown in logs/hparams.
    modality=MODALITY_KEY,
    num_classes=NUM_CLASSES,
    lr=args.lr,
    weight_decay=args.weight_decay,
    label_smoothing=args.label_smoothing,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0":
    try:
        import wandb
        run_name = f"BL_{args.xp_name}_{args.model}"
        if args.model == "resnet":
            run_name += f"_{args.resnet_variant}"
        wandb.init(
            name=run_name,
            project="Atomizer_EuroSAT_SAR_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_EuroSAT_SAR_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/eurosat_sar_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    # ── Resume detection (full trainer state, not just weights) ──────────
    resume_ckpt_path = None
    if args.resume_from is not None:
        resume_ckpt_path = args.resume_from
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(
                f"--resume_from checkpoint not found: {resume_ckpt_path}"
            )
        print(f"[Resume] Using explicit checkpoint: {resume_ckpt_path}")
    elif args.resume:
        auto_path = os.path.join(ckpt_dir, f"bl_{args.xp_name}_{args.model}-last.ckpt")
        if os.path.exists(auto_path):
            resume_ckpt_path = auto_path
            print(f"[Resume] Found existing checkpoint, resuming: {auto_path}")
        else:
            print(f"[Resume] --resume set but no checkpoint found at "
                  f"{auto_path} — starting fresh.")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_top1:.4f}}",
            monitor="val_top1",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-last",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_top1",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # RAMEN's RadarProjector (reused verbatim from RAMEN/ramen_encoder.py)
    # defines 8 separate polarization parameters (asc_vv, asc_vh, asc_hv,
    # asc_hh, des_vv, des_vh, des_hv, des_hh), but RAMEN_S1_POLARIZATIONS
    # here only ever requests 2 of them ("asc_vv","asc_vh") — the other 6
    # are registered trainable params that never appear in the forward
    # graph, which DDP's default unused-parameter check rejects. Only
    # RAMEN needs find_unused_parameters=True; other models don't pay the
    # (small) runtime cost of DDP's extra graph traversal.
    strategy = (
        DDPStrategy(find_unused_parameters=True) if args.model == "ramen"
        else DDPStrategy(find_unused_parameters=False)
    )

    trainer = Trainer(
        strategy=strategy,
        devices=-1,
        max_epochs=args.epochs,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
    )

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=resume_ckpt_path)

    best_ckpt = trainer.checkpoint_callback.best_model_path

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

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
test_trainer.test(trainer_module, test_loader, ckpt_path=best_ckpt)

if wandb_logger:
    import wandb
    wandb.finish()
