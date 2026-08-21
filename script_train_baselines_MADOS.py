"""
MADOS Baseline Training Script
================================

Train single-frame segmentation baselines on MADOS — 15-class marine debris
segmentation on Sentinel-2, single modality (optical only, no SAR).

MADOS is single-temporal — no LTAE needed. Supports:
  - unet      : classic UNet (PANGAEA-style topology)
  - vit       : ViT encoder + UPerNet decoder
  - resnet    : ResNet encoder + UPerNet decoder (variant via --resnet_variant)
  - ramen     : RAMEN encoder + UPerNet decoder (per-band spectral encoding;
                single "optical" modality)
  - universat : UniverSat encoder (gastruc/UniverSat, AnySat v2) trained
                FROM SCRATCH + linear per-token head, via
                training.Universat.universat_augmenter.UniverSatSegmenter.

                SINGLE-MODALITY path, same as the Cashew script — only
                "optical" is present, so Bi_ACA_in cross-modal fusion is
                structurally inert (expected, not a bug, for a
                single-sensor dataset).

                No window size baked at construction: the latent grid is
                recomputed per input (H / patch_px per side). Default eval
                is a single full-image dense forward (window_size=None);
                --universat_window_size is kept available as a fallback.

                Geometry constraints: every input side must be divisible by
                the patch size in pixels (--universat_patch_m / 10) and by
                --universat_output_stride. MADOS's native size (240) is
                NOT a power of 2 (240 = 2^4 x 3 x 5) — the defaults below
                (80 m patch = 8 px, output_stride=4) divide 240 cleanly
                (240 / 8 = 30), but this is worth double-checking if you
                change either value, since 240's divisor set is smaller
                than Cashew's 256 or Sen1Floods11's 512.

Same conditions as the Atomiser MADOS run:
  - Same train/val/test splits (./data/MADOS/splits/{train|val|test}_X.txt)
  - Same per-band per-resolution normalization (normalization_stats.pt cache)
  - Same bands order (from bands.yaml's bands_mados section)
  - All bands upscaled to 10m -> [C, 240, 240]
  - 15 classes, IGNORE_INDEX=255

D4 augmentation is automatic for training.
240x240 is divisible by 16 -> ViT patch_size=16 works cleanly.

Examples:
    # ResNet50 + UPerNet
    python script_train_mados_baselines.py --xp_name resnet50 \\
        --model resnet --resnet_variant resnet50 \\
        --batch_size 8 --lr 1e-4 --epochs 80

    # UNet baseline
    python script_train_mados_baselines.py --xp_name unet \\
        --model unet --batch_size 8 --lr 1e-3 --epochs 80

    # ViT-S baseline
    python script_train_mados_baselines.py --xp_name vit \\
        --model vit --batch_size 8 --lr 1e-4 --epochs 80

    # RAMEN baseline
    python script_train_mados_baselines.py --xp_name ramen \\
        --model ramen --batch_size 8 --lr 1e-4 --epochs 80

    # UniverSat baseline (from scratch)
    python script_train_mados_baselines.py --xp_name universat \\
        --model universat --batch_size 8 --lr 1e-4 --epochs 80
"""

import os
import argparse

import torch
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

from training.utils import read_yaml
from training.utils.datasets_baselines.utils_dataset_mados_baseline import (
    MADOSBaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.RAMEN.ramen_upernet import build_ramen_upernet
from training.Universat.universat_augmenter import build_universat_segmenter
from training.sliding_window import sliding_window_inference
from training.trainer_baselines_weighted import BaselineTrainerWeighted


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = MADOSBaselineDataset.NUM_CLASSES        # 15
IGNORE_INDEX = MADOSBaselineDataset.IGNORE_INDEX       # 255
MODALITY_KEY = "s2"

# Spatial size of MADOS patches at 10m resolution
NATIVE_H, NATIVE_W = MADOSBaselineDataset.FULL_SIZE_10M  # (240, 240)
MADOS_GSD_M = 10.0  # all bands upscaled to 10m; single scalar used for
                    # RAMEN/UniverSat input_res, same simplification the
                    # Sen1Floods11/Cashew scripts make.


# =============================================================================
# COLLATE
# =============================================================================

def mados_collate(batch):
    """Stack per-modality images, stack targets, keep metadata as list."""
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets  = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image":    images,
        "target":   targets,
        "metadata": metadata,
    }


def mados_collate_ramen(batch):
    """
    RAMEN / UniverSat collate for MADOSBaselineDataset.

    Same pattern as Cashew's collate: MADOS has only one modality, so this
    just wraps the existing image["s2"] tensor as {"optical": ...} to
    match the dict interface RAMENUPerNet / UniverSatSegmenter expect, no
    channel split needed.
    """
    merged = torch.stack([s["image"]["s2"] for s in batch])  # [B, C, H, W]
    images = {"optical": merged}
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name: str, in_channels: int, num_classes: int, args,
                 ramen_input_bands=None, ramen_wavelengths=None,
                 universat_input_bands=None, universat_wavelengths=None):
    if model_name == "unet":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            topology=tuple(args.unet_topology),
        )

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
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "resnet":
        return build_resnet_upernet(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "ramen":
        return build_ramen_upernet(
            input_bands=ramen_input_bands,
            wavelengths=ramen_wavelengths,
            num_classes=num_classes,
            input_size=args.ramen_window_size,
            embed_dim=args.ramen_embed_dim,
            depth=args.ramen_depth,
            num_heads=args.ramen_num_heads,
            input_res=args.ramen_input_res,
            res=args.ramen_res,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "universat":
        return build_universat_segmenter(
            input_bands=universat_input_bands,
            wavelengths=universat_wavelengths,
            num_classes=num_classes,
            input_res={"optical": MADOS_GSD_M},
            patch_size_m=args.universat_patch_m,
            output_stride=args.universat_output_stride,
            size=args.universat_size,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'unet', 'vit', "
            f"'resnet', 'ramen', 'universat'"
        )


# =============================================================================
# GFLOPs MEASUREMENT — FlopCounterMode, same harness as
# script_train_cashew_baselines.py (kept consistent across the baseline
# family; NOT comparable to the Perceiver scripts' torch.profiler numbers).
# =============================================================================

from torch.utils.flop_counter import FlopCounterMode


def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b


@torch.no_grad()
def measure_gflops_forward(forward_fn, batches, device, n_warmup=1):
    for b in batches[:n_warmup]:
        _ = forward_fn(b)
    if device == "cuda":
        torch.cuda.synchronize()

    flops_list = []
    for b in batches[n_warmup:]:
        fc = FlopCounterMode(display=False)
        with fc:
            _ = forward_fn(b)
        flops_list.append(fc.get_total_flops())

    if not flops_list:
        return float("nan")
    return (sum(flops_list) / len(flops_list)) / 1e9


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="MADOS Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet", "ramen", "universat"])
parser.add_argument("--data_dir",  type=str, default="./data/MADOS")
parser.add_argument("--bands_yaml", type=str, default="./data/bands_info/bands.yaml",
                    help="YAML file containing the bands_mados section")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")
parser.add_argument("--resume",    type=str, default=None,
                    help="Path to a .ckpt file to resume training from "
                         "(full trainer state). Passed to "
                         "Trainer.fit(ckpt_path=...). Ignored if "
                         "--test_only is set.")

# Training
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)

# Image size — MADOS is 240x240. ViT patch_size=16 -> 15x15 patches works cleanly.
parser.add_argument("--img_size", type=int, default=NATIVE_H,
                    help=f"Spatial size baked into ViT pos_embed. "
                         f"Default {NATIVE_H} (full MADOS patch).")

# UNet
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024],
                    help="UNet feature widths per level")

# ViT
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256,
                    help="UPerNet decoder channels (also used by ResNet and RAMEN)")

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=10.0,
                    help="Native GSD (m/px) of the input imagery (post-"
                         "upscaling to 10m, matching the dataset).")
parser.add_argument("--ramen_res",       type=float, default=20.0,
                    help="Common working resolution (m/px). Left equal to "
                         "--ramen_input_res by default (no resampling).")
parser.add_argument("--ramen_window_size", type=int, default=240,
                    help="RAMEN tokenizes at the pixel level, so full "
                         "self-attention over 240x240 is heavy. Model is "
                         "built/trained at this smaller size (default 120 "
                         "= exactly half of MADOS's 240, and divides it "
                         "cleanly for non-overlapping sliding-window "
                         "eval); training crops use this value directly "
                         "(overrides --crop_size for --model ramen). Full "
                         "240x240 val/test is handled via sliding-window "
                         "inference.")
parser.add_argument("--ramen_stride", type=int, default=96,
                    help="Sliding-window stride for RAMEN eval. Must be "
                         "<= --ramen_window_size. Ignored during training.")
parser.add_argument("--ramen_config", type=str, default=None,
                    help="Optional path to a RAMEN YAML config whose keys "
                         "override the --ramen_* CLI defaults above.")

# UniverSat (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"])
parser.add_argument("--universat_patch_m", type=float, default=80.0,
                    help="Patch size in METRES. 80 m = 8 px at MADOS's "
                         "10 m working GSD; 240 / 8 = 30, divides cleanly. "
                         "Must be an integer number of pixels, and every "
                         "input side must be divisible by that pixel "
                         "count.")
parser.add_argument("--universat_output_stride", type=int, default=4,
                    help="Logits at H/stride per side (BaselineTrainer "
                         "bilinearly upsamples to the target).")
parser.add_argument("--universat_window_size", type=int, default=None,
                    help="None (default) = full-image eval in ONE dense "
                         "forward. Set to a valid divisor-compatible size "
                         "to fall back to sliding-window eval if the 240 "
                         "full-image forward OOMs.")
parser.add_argument("--universat_stride", type=int, default=96,
                    help="Sliding-window stride for UniverSat eval; only "
                         "used when --universat_window_size is set.")

# GFLOPs
parser.add_argument("--flops", action="store_true", default=True)
parser.add_argument("--no_flops", dest="flops", action="store_false")
parser.add_argument("--flops_n", type=int, default=3)

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")


# =============================================================================
# RAMEN CONFIG OVERRIDES
# =============================================================================

_RAMEN_CONFIG_KEY_MAP = {
    "res": "ramen_res",
    "input_size": "ramen_window_size",
    "input_res": "ramen_input_res",
    "embed_dim": "ramen_embed_dim",
    "depth": "ramen_depth",
    "num_heads": "ramen_num_heads",
    "stride": "ramen_stride",
}

import sys as _sys
_explicit_cli_args = {
    tok[2:].split("=")[0].replace("-", "_")
    for tok in _sys.argv[1:]
    if tok.startswith("--")
}

if args.ramen_config is not None:
    with open(args.ramen_config, "r") as f:
        import yaml as _yaml
        _ramen_cfg = _yaml.safe_load(f) or {}
    for _key, _val in _ramen_cfg.items():
        _dest = _RAMEN_CONFIG_KEY_MAP.get(_key)
        if _dest is None:
            print(f"[WARNING] Unrecognized key '{_key}' in {args.ramen_config} "
                  f"— ignoring. Known keys: {sorted(_RAMEN_CONFIG_KEY_MAP)}")
            continue
        if _dest in _explicit_cli_args:
            print(f"[INFO] '{_key}' in {args.ramen_config} ignored — "
                  f"--{_dest} was explicitly set on the command line "
                  f"({getattr(args, _dest)}).")
            continue
        print(f"[INFO] {args.ramen_config}: {_key}={_val} -> --{_dest}")
        setattr(args, _dest, _val)


# =============================================================================
# SANITY CHECK FOR VIT
# =============================================================================

if args.model == "vit":
    if args.img_size % args.vit_patch_size != 0:
        raise ValueError(
            f"For ViT: --img_size ({args.img_size}) must be divisible by "
            f"--vit_patch_size ({args.vit_patch_size}). "
            f"Default img_size={NATIVE_H} works with patch_size 16, 12, 8."
        )

if args.model == "ramen":
    if args.ramen_stride > args.ramen_window_size:
        if "ramen_stride" in _explicit_cli_args:
            raise ValueError(
                f"--ramen_stride ({args.ramen_stride}) must be <= "
                f"--ramen_window_size ({args.ramen_window_size})."
            )
        print(f"[WARNING] --ramen_stride ({args.ramen_stride}) exceeds "
              f"--ramen_window_size ({args.ramen_window_size}); clamping "
              f"stride to window_size (non-overlapping tiling).")
        args.ramen_stride = args.ramen_window_size
    if NATIVE_H % args.ramen_window_size != 0:
        print(f"[WARNING] MADOS native size ({NATIVE_H}) is not evenly "
              f"divisible by --ramen_window_size ({args.ramen_window_size}); "
              f"sliding-window eval will have a partial edge tile. "
              f"Divisors of 240 include: 1,2,3,4,5,6,8,10,12,15,16,20,24,"
              f"30,40,48,60,80,120,240.")

if args.model == "universat":
    universat_patch_px = args.universat_patch_m / MADOS_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {MADOS_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px). Use a multiple of "
            f"{MADOS_GSD_M}."
        )
    universat_patch_px = int(round(universat_patch_px))

    import math as _math
    _lcm = _math.lcm(universat_patch_px, args.universat_output_stride)

    if NATIVE_H % _lcm:
        raise ValueError(
            f"The full {NATIVE_H}x{NATIVE_W} eval image is not divisible "
            f"by lcm(patch_px={universat_patch_px}, "
            f"output_stride={args.universat_output_stride})={_lcm} — "
            f"pick --universat_patch_m / --universat_output_stride so "
            f"that {NATIVE_H} is a valid input side (or set "
            f"--universat_window_size to a valid size). MADOS's 240 has "
            f"a smaller divisor set than Cashew's 256 or Sen1Floods11's "
            f"512 — double check this combination explicitly."
        )

    if args.universat_window_size is not None:
        if args.universat_window_size % _lcm:
            raise ValueError(
                f"--universat_window_size ({args.universat_window_size}) "
                f"must be divisible by lcm(patch_px={universat_patch_px}, "
                f"output_stride={args.universat_output_stride})={_lcm}."
            )
        if args.universat_stride > args.universat_window_size:
            print(f"[WARNING] --universat_stride "
                  f"({args.universat_stride}) exceeds "
                  f"--universat_window_size "
                  f"({args.universat_window_size}); clamping to "
                  f"window_size.")
            args.universat_stride = args.universat_window_size


# =============================================================================
# LOAD BANDS METADATA
# =============================================================================

bands_yaml = read_yaml(args.bands_yaml)
if "bands_mados" not in bands_yaml:
    raise KeyError(
        f"[MADOS-BL] bands_yaml ({args.bands_yaml}) must contain a "
        f"'bands_mados' section."
    )
bands_info = bands_yaml["bands_mados"]

# Build RAMEN/UniverSat band-name + wavelength tables DYNAMICALLY from
# bands_info, rather than hardcoding a band list as the Cashew script does
# (Cashew's exact S2 band subset was known in advance from
# CashewBaselineDataset.BAND_PREFIXES; MADOS's isn't hardcoded anywhere in
# this script, so we derive it from the same bands_info dict the dataset
# itself uses, in idx order — this guarantees channel-order and
# wavelength/bandwidth values can never drift out of sync between the
# dataset and the RAMEN/UniverSat model construction below).
if args.model in ("ramen", "universat"):
    _entries = []
    for _name, _data in bands_info.items():
        if "idx" in _data and "central_wavelength" in _data:
            _entries.append((_data["idx"], _name, _data))
    _entries.sort(key=lambda t: t[0])

    if not _entries:
        raise KeyError(
            f"[MADOS-BL] No usable band entries (need 'idx' and "
            f"'central_wavelength' keys) found in bands_mados — cannot "
            f"build RAMEN/UniverSat band metadata."
        )

    _MADOS_BAND_NAMES = [name for _, name, _ in _entries]
    _MADOS_WAVELENGTHS_NM = {
        name: float(data["central_wavelength"]) for _, name, data in _entries
    }

    RAMEN_INPUT_BANDS  = {"optical": _MADOS_BAND_NAMES}
    RAMEN_WAVELENGTHS  = {"optical": _MADOS_WAVELENGTHS_NM}
    UNIVERSAT_INPUT_BANDS = {"optical": _MADOS_BAND_NAMES}
    UNIVERSAT_WAVELENGTHS = {"optical": _MADOS_WAVELENGTHS_NM}

    print(f"[MADOS-BL] RAMEN/UniverSat band order (from bands_mados, by idx): "
          f"{_MADOS_BAND_NAMES}")
else:
    RAMEN_INPUT_BANDS = RAMEN_WAVELENGTHS = None
    UNIVERSAT_INPUT_BANDS = UNIVERSAT_WAVELENGTHS = None


# =============================================================================
# DATASETS
# =============================================================================

train_ds = MADOSBaselineDataset(
    root_path=args.data_dir, mode="train",
    bands_info=bands_info,
)
val_ds = MADOSBaselineDataset(
    root_path=args.data_dir, mode="validation",
    bands_info=bands_info,
)
test_ds = MADOSBaselineDataset(
    root_path=args.data_dir, mode="test",
    bands_info=bands_info,
)

NUM_CHANNELS = train_ds.num_channels

if args.model in ("ramen", "universat") and len(_MADOS_BAND_NAMES) != NUM_CHANNELS:
    raise ValueError(
        f"[MADOS-BL] Band count mismatch: bands_mados yielded "
        f"{len(_MADOS_BAND_NAMES)} usable entries (idx + "
        f"central_wavelength present), but the dataset reports "
        f"NUM_CHANNELS={NUM_CHANNELS}. Some band(s) in bands_mados are "
        f"missing 'idx' or 'central_wavelength' — RAMEN/UniverSat need "
        f"metadata for EVERY channel the dataset actually returns. Check "
        f"bands_mados against the dataset's own band list."
    )


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  MADOS Baseline Training")
print(f"  Model:        {args.model}")
if args.model == "resnet":
    print(f"  Variant:      {args.resnet_variant}")
if args.model == "ramen":
    print(f"  Window size:  {args.ramen_window_size}x{args.ramen_window_size} "
          f"(model built/trained at this size)")
    print(f"  Eval stride:  {args.ramen_stride} "
          f"(sliding-window inference over full {NATIVE_H}x{NATIVE_W})")
if args.model == "universat":
    print(f"  Size:         {args.universat_size} (from scratch, random init)")
    print(f"  Patch:        {args.universat_patch_m:.0f} m "
          f"({int(args.universat_patch_m / MADOS_GSD_M)} px @ "
          f"{MADOS_GSD_M:.0f} m)")
    print(f"  Out stride:   {args.universat_output_stride}")
    if args.universat_window_size is not None:
        print(f"  Eval:         sliding-window (window="
              f"{args.universat_window_size}, stride={args.universat_stride})")
    else:
        print(f"  Eval:         full {NATIVE_H}x{NATIVE_W}, single dense forward")
print(f"  Channels:     {NUM_CHANNELS} bands (all upscaled to 10m)")
print(f"  Patch size:   {NATIVE_H}x{NATIVE_W}")
print(f"  Classes:      {NUM_CLASSES}")
print(f"  Ignore index: {IGNORE_INDEX}")
print(f"  Epochs:       {args.epochs}")
print(f"  Batch size:   {args.batch_size}")
print(f"  LR:           {args.lr}")
print(f"  Grad accum:   {args.grad_accum}")
print(f"  GPUs:         {torch.cuda.device_count()}")
if args.resume is not None:
    print(f"  Resuming from: {args.resume}")
print(f"{'='*60}\n")

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

collate_fn = (mados_collate_ramen if args.model in ("ramen", "universat")
              else mados_collate)

loader_kwargs = dict(
    num_workers=args.num_workers,
    collate_fn=collate_fn,
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
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(
    args.model, NUM_CHANNELS, NUM_CLASSES, args,
    ramen_input_bands=RAMEN_INPUT_BANDS,
    ramen_wavelengths=RAMEN_WAVELENGTHS,
    universat_input_bands=UNIVERSAT_INPUT_BANDS,
    universat_wavelengths=UNIVERSAT_WAVELENGTHS,
)

trainer_module = BaselineTrainerWeighted(
    model=model,
    modality=MODALITY_KEY if args.model not in ("ramen", "universat")
             else "optical",
    temporal=False,
    task="mados",                     # registered in TASK_CLASS_NAMES
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
    window_size=(args.ramen_window_size if args.model == "ramen"
                 else args.universat_window_size if args.model == "universat"
                 else None),
    window_stride=(args.ramen_stride if args.model == "ramen"
                   else args.universat_stride if args.model == "universat"
                   else None),
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
        if args.model == "universat":
            run_name += (f"_{args.universat_size}"
                         f"_os{args.universat_output_stride}")
        wandb.init(
            name=run_name,
            project="Atomizer_MADOS_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_MADOS_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in --test_only mode)
# =============================================================================

ckpt_dir = "./checkpoints/mados_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_mIoU_weighted:.4f}}",
            monitor="val_mIoU_weighted",
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
            monitor="val_mIoU",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # find_unused_parameters=True: RAMEN's RadarProjector/DemProjector-style
    # unused-branch pattern and UniverSat's inert S1/T-axis blocks and
    # unused sensor channel codes both need this. Harmless for the other
    # models — using it unconditionally matches the Cashew script.
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
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

    print(f"\n{'='*60}")
    print(f"  Starting: {args.model} on MADOS")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=args.resume)

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


# =============================================================================
# GFLOPs (rank-zero only)
# =============================================================================

if args.flops and os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"\n{'='*60}")
    print(f"  GFLOPs measurement — {args.model}")
    print(f"{'='*60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = trainer_module.model.to(device).eval()

    flops_raw = []
    for b in test_loader:
        flops_raw.append(_to_device(b, device))
        if len(flops_raw) >= args.flops_n + 1:
            break

    if not flops_raw:
        print("[FLOPs] No test batches available; skipping GFLOPs measurement.")
    else:
        if args.model == "ramen":
            def fwd(b, m=eval_model):
                return sliding_window_inference(
                    m, b["image"],
                    window_size=args.ramen_window_size,
                    stride=args.ramen_stride,
                    num_classes=NUM_CLASSES,
                )
            size_note = f"full {NATIVE_H}x{NATIVE_W}, sliding-window (all tiles)"
        elif args.model == "universat":
            if args.universat_window_size is not None:
                def fwd(b, m=eval_model):
                    return sliding_window_inference(
                        m, b["image"],
                        window_size=args.universat_window_size,
                        stride=args.universat_stride,
                        num_classes=NUM_CLASSES,
                    )
                size_note = f"full {NATIVE_H}x{NATIVE_W}, sliding-window (all tiles)"
            else:
                def fwd(b, m=eval_model):
                    return m(b["image"])
                size_note = f"full {NATIVE_H}x{NATIVE_W}, single dense forward"
        else:
            def fwd(b, m=eval_model):
                return m(b["image"][MODALITY_KEY])
            size_note = f"full {NATIVE_H}x{NATIVE_W}, single dense forward"

        gflops = measure_gflops_forward(fwd, flops_raw, device, n_warmup=1)
        print(f"  GFLOPs/forward (bs=1, {size_note}): {gflops:.2f}"
              f"  (mean of {len(flops_raw) - 1} passes)")

        if wandb_logger:
            import wandb
            wandb.log({"test_gflops": gflops})

if wandb_logger:
    import wandb
    wandb.finish()
