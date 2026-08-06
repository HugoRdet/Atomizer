"""
Sen1Floods11 Baseline Training Script
======================================

Train single-frame segmentation baselines on Sen1Floods11 (binary flood / no-flood).

Sen1Floods11 is single-temporal — no LTAE needed. Supports:
  - unet   : classic UNet (PANGAEA-style topology [64, 128, 256, 512, 1024])
  - vit    : ViT encoder + UPerNet decoder
  - resnet : ResNet encoder + UPerNet decoder (variant via --resnet_variant)
  - ramen  : RAMEN multi-modal encoder + UPerNet decoder (separate S2/S1
             modality tensors, per-band spectral encoding)

Same conditions as Atomiser:
  - Same train/val/test splits
  - Same normalization (per-band z-score, normalization_stats.pt)
  - Same NaN cleanup, ignore_index=255
  - Same D4 augmentation
  - 15 input channels (13 S2 + 2 S1) — merged for unet/vit/resnet,
    kept as separate {"optical","sar"} tensors for ramen

GFLOPs: measured once after testing completes, with the SAME harness used
across the other baseline scripts (torch.profiler, with_flops=True, mean
over --flops_n passes, bs=1, full 512x512 image, one discarded warmup).
UNet/ViT/ResNet all do a single dense forward on the full image (ViT
already trains/evals at native 512x512 here, no tiling). RAMEN's number
is the full sliding-window pass over the whole image (all tiles),
directly comparable to the others' single dense forward. Rank-zero only
(avoids redundant profiling across DDP ranks). Disable with --no_flops.

Examples:
    # ResNet50 + UPerNet on S2+S1
    python script_train_senflood_baseline.py --xp_name resnet50_s2s1 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 8 --lr 1e-4 --epochs 80

    # UNet baseline (matches PANGAEA's UNet setup)
    python script_train_senflood_baseline.py --xp_name unet_s2s1 \
        --model unet \
        --batch_size 8 --lr 1e-3 --epochs 80

    # ViT-S baseline
    python script_train_senflood_baseline.py --xp_name vit_s2s1 \
        --model vit \
        --batch_size 8 --lr 1e-4 --epochs 80

    # RAMEN baseline
    python script_train_senflood_baseline.py --xp_name ramen_s2s1 \
        --model ramen \
        --batch_size 8 --lr 1e-4 --epochs 80
"""

import os
import argparse

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.sliding_window import sliding_window_inference  # adjust import path
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS (from the dataset)
# =============================================================================

NUM_CLASSES = Sen1Floods11BaselineDataset.NUM_CLASSES        # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX      # 255
NUM_CHANNELS = Sen1Floods11BaselineDataset.NUM_CHANNELS      # 15
NUM_S2_BANDS = Sen1Floods11BaselineDataset.NUM_S2_BANDS      # 13
NUM_S1_BANDS = Sen1Floods11BaselineDataset.NUM_S1_BANDS      # 2
MODALITY_KEY = "s2s1"  # dataset returns image[{MODALITY_KEY}]


# =============================================================================
# RAMEN band metadata
# =============================================================================
#
# IMPORTANT: confirm this band order matches how your S2Hand GeoTIFFs are
# actually stacked on disk. This follows the standard Sentinel-2 L1C
# 13-band order (B10 included) — update if your preprocessing differs.

S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

S2_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B8A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
    "B12": 2202.4,
}

S1_BAND_NAMES = ["VV", "VH"]

# RadarProjector's pol_map only has ascending/descending-tagged keys
# (e.g. "asc_vv"); Sen1Floods11 doesn't expose pass direction, so this
# defaults to "asc_*". Update if pass direction becomes available.
S1_POLARIZATIONS = {"VV": "asc_vv", "VH": "asc_vh"}

RAMEN_INPUT_BANDS = {
    "optical": S2_BAND_NAMES,
    "sar": S1_BAND_NAMES,
}

RAMEN_WAVELENGTHS = {
    "optical": S2_WAVELENGTHS_NM,
    "sar": S1_POLARIZATIONS,
}


# =============================================================================
# COLLATE — stacks per-modality images, stacks targets, keeps metadata as list
# =============================================================================

def senflood_collate(batch):
    """
    Collate for Sen1Floods11BaselineDataset.

    Each sample is a dict with image[modality_key]: [C, H, W], target: [H, W].
    Stack into batch tensors.
    """
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])

    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "target": targets,
        "metadata": metadata,
    }


def senflood_collate_ramen(batch):
    """
    RAMEN collate for Sen1Floods11BaselineDataset.

    The dataset still returns the merged image["s2s1"] : [15, H, W]
    tensor (no dataset changes needed) — this splits it into separate
    "optical" (first NUM_S2_BANDS channels) and "sar" (remaining
    NUM_S1_BANDS channels) tensors at batch-collation time, since
    RAMENUPerNet needs each modality separately to look up its own
    spectral projector and band wavelengths.
    """
    merged = torch.stack([s["image"]["s2s1"] for s in batch])  # [B, 15, H, W]

    images = {
        "optical": merged[:, :NUM_S2_BANDS],                              # [B, 13, H, W]
        "sar": merged[:, NUM_S2_BANDS : NUM_S2_BANDS + NUM_S1_BANDS],     # [B, 2, H, W]
    }

    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "target": targets,
        "metadata": metadata,
    }


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name: str, in_channels: int, num_classes: int, args):
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
        # Built at the small window size, NOT the full 512 img_size — see
        # --ramen_window_size. Full-resolution eval is handled separately
        # via sliding_window_inference (BaselineTrainer.window_size).
        return build_ramen_upernet(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
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

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'unet', 'vit', 'resnet', 'ramen'"
        )


# =============================================================================
# GFLOPs MEASUREMENT — same harness as the other baseline scripts
# =============================================================================

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
    """
    One warmup pass discarded; each measured pass profiled separately
    with with_flops=True; per-pass total summed over key_averages();
    report mean / 1e9. For RAMEN, forward_fn internally loops over
    sliding-window tiles, so this captures the TOTAL cost of one
    full-image forward — directly comparable to the other models' single
    dense forward pass.
    """
    for b in batches[:n_warmup]:
        _ = forward_fn(b)
    if device == "cuda":
        torch.cuda.synchronize()

    flops_list = []
    for b in batches[n_warmup:]:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True, record_shapes=True, profile_memory=True) as prof:
            _ = forward_fn(b)
            if device == "cuda":
                torch.cuda.synchronize()
        total = sum(evt.flops for evt in prof.key_averages()
                    if getattr(evt, "flops", None))
        flops_list.append(total)

    if not flops_list:
        return float("nan")
    return (sum(flops_list) / len(flops_list)) / 1e9


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="Sen1Floods11 Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet", "ramen"])
parser.add_argument("--data_dir",  type=str, default="./data/SENFLOOD")
parser.add_argument("--resume",    type=str, default=None,
                    help="Path to a .ckpt file to resume training from "
                         "(full trainer state: model, optimizer, LR "
                         "scheduler, epoch/step counters, callback state "
                         "e.g. EarlyStopping/ModelCheckpoint's best-score "
                         "tracking). Passed to Trainer.fit(ckpt_path=...). "
                         "If omitted, training starts from scratch.")

# Training
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=150)
parser.add_argument("--grad_accum",   type=int, default=1)

# Crop / image size
parser.add_argument("--crop_size", type=int, default=512,
                    help="Random crop size for training. Default 512 = no crop "
                         "(use full image). For ViT/RAMEN, must match --img_size.")
parser.add_argument("--img_size",  type=int, default=512,
                    help="Spatial size baked into ViT/RAMEN positional embeddings. "
                         "MUST equal --crop_size for ViT/RAMEN (and equal eval size). "
                         "UNet/ResNet ignore this.")

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
                    help="Native GSD (m/px) of the input imagery.")
parser.add_argument("--ramen_res",       type=float, default=40.0,
                    help="Common working resolution (m/px) all modalities are "
                         "resampled to before the shared ViT stack. Since "
                         "input_res is the same across modalities here, this "
                         "can be left equal to --ramen_input_res (no resampling).")
parser.add_argument("--ramen_window_size", type=int, default=128,
                    help="RAMEN tokenizes at the pixel level (no patch "
                         "embedding), so full self-attention over a "
                         "512x512 image is intractable. The model is "
                         "built and trained at this smaller spatial size "
                         "instead; training crops use this value directly "
                         "(overrides --crop_size for --model ramen). Full "
                         "512x512 val/test images are handled via "
                         "sliding-window inference (see --ramen_stride).")
parser.add_argument("--ramen_stride", type=int, default=96,
                    help="Step between windows for sliding-window "
                         "inference at val/test time. Must be <= "
                         "--ramen_window_size; smaller values give more "
                         "overlap (averaged in the output) at the cost of "
                         "more forward passes per image. Ignored during "
                         "training.")
parser.add_argument("--ramen_config", type=str, default=None,
                    help="Optional path to a RAMEN YAML config (e.g. "
                         "training/RAMEN/config_SENFLOOD.yaml) whose keys "
                         "override the --ramen_* CLI defaults above. Each "
                         "top-level key `foo` in the YAML maps to "
                         "--ramen_foo (e.g. `res: 40` -> args.ramen_res, "
                         "`input_size: 64` -> args.ramen_window_size). "
                         "Explicit CLI flags still take precedence over "
                         "config values for keys passed on the command "
                         "line — see the override logic below.")

# GFLOPs
parser.add_argument("--flops", action="store_true", default=True,
                    help="Measure GFLOPs/forward on the final model after "
                         "testing (default: on).")
parser.add_argument("--no_flops", dest="flops", action="store_false",
                    help="Disable GFLOPs measurement.")
parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of profiled forward passes to average.")

# Band-dropout augmentation (train only) — gives baselines training-time
# exposure to missing modalities/bands, for fairer comparison against
# Atomiser's native padding-token robustness at the modality-drop eval
# (script_test_senflood_baseline_modality_drop.py). See
# Sen1Floods11BaselineDataset's docstring for exact semantics.
parser.add_argument("--band_dropout", action="store_true", default=True,
                    help="Enable band-dropout augmentation during training "
                         "(default: on). Matches the intent of Atomiser's "
                         "own token-dropout augmentation — set the "
                         "probabilities below to the SAME values used on "
                         "the Atomiser side for a fair comparison.")
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

if args.resume is not None and not os.path.isfile(args.resume):
    raise FileNotFoundError(
        f"--resume checkpoint not found: {args.resume}"
    )


# =============================================================================
# RAMEN CONFIG OVERRIDES
# =============================================================================
#
# YAML keys map to --ramen_* CLI args as follows. Extend this table if
# config_SENFLOOD.yaml grows new keys.
_RAMEN_CONFIG_KEY_MAP = {
    "res": "ramen_res",
    "input_size": "ramen_window_size",
    "input_res": "ramen_input_res",
    "embed_dim": "ramen_embed_dim",
    "depth": "ramen_depth",
    "num_heads": "ramen_num_heads",
    "stride": "ramen_stride",
}

# Track which --ramen_* flags were explicitly passed on the command line,
# so config values only fill in *unset* args rather than silently
# overriding something the user typed.
import sys as _sys
_explicit_cli_args = {
    tok[2:].split("=")[0].replace("-", "_")
    for tok in _sys.argv[1:]
    if tok.startswith("--")
}

if args.ramen_config is not None:
    with open(args.ramen_config, "r") as f:
        _ramen_cfg = yaml.safe_load(f) or {}

    for _key, _val in _ramen_cfg.items():
        _dest = _RAMEN_CONFIG_KEY_MAP.get(_key)
        if _dest is None:
            print(f"[WARNING] Unrecognized key '{_key}' in {args.ramen_config} "
                  f"— no matching --ramen_* arg, ignoring. Known keys: "
                  f"{sorted(_RAMEN_CONFIG_KEY_MAP)}")
            continue
        if _dest in _explicit_cli_args:
            print(f"[INFO] '{_key}' in {args.ramen_config} ignored — "
                  f"--{_dest} was explicitly set on the command line "
                  f"({getattr(args, _dest)}).")
            continue
        print(f"[INFO] {args.ramen_config}: {_key}={_val} -> --{_dest}")
        setattr(args, _dest, _val)


# =============================================================================
# SANITY CHECKS
# =============================================================================

# ViT bakes spatial size into pos_embed → train and eval must match exactly
# (no sliding-window support for ViT here). Sen1Floods11 native size is 512.
if args.model == "vit":
    if args.crop_size != args.img_size:
        raise ValueError(
            f"For vit: --crop_size ({args.crop_size}) must equal "
            f"--img_size ({args.img_size}). Positional embedding is baked at "
            f"construction; train and eval must use the same spatial size. "
            f"Sen1Floods11 native size is 512; recommended: both 512."
        )
    if args.img_size != 512:
        print(f"[WARNING] vit trained at {args.img_size}×{args.img_size}, "
              f"but Sen1Floods11 native size is 512×512. Eval at 512 will fail; "
              f"the dataset would need explicit cropping/resizing.")

# RAMEN tokenizes at the pixel level — full self-attention over 512x512 is
# intractable. Train on small windows (--ramen_window_size), eval on full
# 512x512 via sliding-window inference. --crop_size is overridden here so
# the dataset crops to the window size, not the full image.
if args.model == "ramen":
    if args.ramen_stride > args.ramen_window_size:
        if "ramen_stride" in _explicit_cli_args:
            raise ValueError(
                f"--ramen_stride ({args.ramen_stride}) must be <= "
                f"--ramen_window_size ({args.ramen_window_size})."
            )
        print(f"[WARNING] --ramen_stride ({args.ramen_stride}) exceeds "
              f"--ramen_window_size ({args.ramen_window_size}) after "
              f"config overrides; clamping stride to window_size "
              f"(non-overlapping tiling).")
        args.ramen_stride = args.ramen_window_size
    if args.crop_size != args.ramen_window_size:
        print(f"[INFO] --model ramen: overriding --crop_size "
              f"({args.crop_size}) with --ramen_window_size "
              f"({args.ramen_window_size}) for training crops.")
        args.crop_size = args.ramen_window_size


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  Sen1Floods11 Baseline Training")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
if args.model == "ramen":
    print(f"  Window size: {args.ramen_window_size}×{args.ramen_window_size} "
          f"(model built/trained at this size)")
    print(f"  Eval stride: {args.ramen_stride} "
          f"(sliding-window inference over full 512×512)")
print(f"  Channels:    {NUM_CHANNELS} (13 S2 + 2 S1)")
print(f"  Crop (train):{args.crop_size}×{args.crop_size}")
print(f"  Eval size:   512×512 (full)")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  Grad acc:    {args.grad_accum}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"  GFLOPs:      {'ON (n=' + str(args.flops_n) + ')' if args.flops else 'OFF'}")
print(f"  Band drop:   {'ON (p_applied=' + str(args.p_dropout_applied) + ', p_whole_mod=' + str(args.p_whole_modality) + ', p_band=' + str(args.p_band_drop) + ')' if args.band_dropout else 'OFF'}")
if args.resume is not None:
    print(f"  Resuming from: {args.resume}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
    band_dropout=args.band_dropout,
    p_dropout_applied=args.p_dropout_applied,
    p_whole_modality=args.p_whole_modality,
    p_band_drop=args.p_band_drop,
)
val_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=None, augment=False,
    # band_dropout intentionally not passed: the dataset gates it to
    # mode=="train" internally regardless of the constructor default,
    # so val/test are never augmented either way.
)
test_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

collate_fn = senflood_collate_ramen if args.model == "ramen" else senflood_collate

# Val/test use full 512×512 images → bigger memory footprint per sample.
# Use batch_size=1 for eval to be safe; train uses cropped 256×256 at full BS.
loader_kwargs_train = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

loader_kwargs_eval = dict(
    batch_size=1,                       # full 512 — memory-conservative
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs_train)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs_eval)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs_eval)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

trainer_module = BaselineTrainer(
    model=model,
    # RAMENUPerNet consumes the full image dict (see BaselineTrainer._get_image);
    # `modality` is unused in that case, only shown in the startup print.
    modality=MODALITY_KEY if args.model != "ramen" else "optical+sar",
    temporal=False,                   # single-frame
    task="senflood",                  # registered in TASK_CLASS_NAMES (or falls back)
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
    # Sliding-window inference at val/test — ignored by every model except
    # RAMEN (see BaselineTrainer._shared_step / sliding_window_inference).
    window_size=args.ramen_window_size if args.model == "ramen" else None,
    window_stride=args.ramen_stride if args.model == "ramen" else None,
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
            project="Atomizer_SenFlood_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_SenFlood_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/senflood_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
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


# =============================================================================
# TRAINER
# =============================================================================

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


# =============================================================================
# TRAIN
# =============================================================================

print(f"\n{'='*60}")
if args.resume is not None:
    print(f"  Resuming: {args.model} on Sen1Floods11 from {args.resume}")
else:
    print(f"  Starting: {args.model} on Sen1Floods11")
print(f"{'='*60}\n")

trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=args.resume)


# =============================================================================
# TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing best checkpoint")
print(f"{'='*60}\n")

trainer.test(trainer_module, test_loader, ckpt_path="best")


# =============================================================================
# GFLOPs (rank-zero only, after best-checkpoint weights are loaded above)
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
        if len(flops_raw) >= args.flops_n + 1:  # +1 warmup
            break

    if not flops_raw:
        print("[FLOPs] No test batches available; skipping GFLOPs measurement.")
    else:
        if args.model == "ramen":
            # senflood_collate_ramen already splits the batch into
            # {"optical","sar"} at collate time, so no adapter is needed
            # here — sliding_window_inference crops that dict generically
            # and RAMENUPerNet consumes it directly.
            def fwd(b, m=eval_model):
                return sliding_window_inference(
                    m, b["image"],
                    window_size=args.ramen_window_size,
                    stride=args.ramen_stride,
                    num_classes=NUM_CLASSES,
                )
            size_note = "full 512x512, sliding-window (all tiles)"
        else:
            # UNet/ViT/ResNet all consume the merged [B,15,H,W] tensor
            # directly. ViT already trains/evals at native 512x512 here
            # (no tiling), so a single dense forward is the correct,
            # directly-comparable measurement for all three.
            def fwd(b, m=eval_model):
                return m(b["image"][MODALITY_KEY])
            size_note = "full 512x512, single dense forward"

        gflops = measure_gflops_forward(fwd, flops_raw, device, n_warmup=1)
        print(f"  GFLOPs/forward (bs=1, {size_note}): {gflops:.2f}"
              f"  (mean of {len(flops_raw) - 1} passes)")

        if wandb_logger:
            import wandb
            wandb.log({"test_gflops": gflops})


if wandb_logger:
    import wandb
    wandb.finish()
