"""
HLS BurnScars Baseline Training Script
========================================

Train single-frame segmentation baselines on HLS BurnScars (binary burn scar).

Single-temporal — no LTAE needed. Supports:
  - unet   : classic UNet (PANGAEA-style topology)
  - vit    : ViT encoder + UPerNet decoder
  - resnet : ResNet encoder + UPerNet decoder (variant via --resnet_variant)
  - ramen  : RAMEN encoder + UPerNet decoder. Unlike Sen1Floods11/EuroSAT-SAR,
             this dataset is SINGLE-modality (one "hls" key, 6 bands, no SAR)
             and already returns a per-modality dict — so RAMEN needs no
             split/rename adapter at all. RAMEN's modality name is simply
             set to "hls" to match the dataset's own key directly, and
             batch["image"] = {"hls": tensor} passes straight through via
             BaselineTrainer's expects_full_image_dict duck-typing.

             Like Sen1Floods11, RAMEN tokenizes at the pixel level, so a
             full 512x512 eval image needs sliding_window_inference — the
             model is built at --ramen_window_size and eval tiles the full
             image at --ramen_stride, same mechanism as Sen1Floods11.

             IMPORTANT: HLS is standardized to 30m GSD (Harmonized Landsat
             Sentinel — resampled to 30m regardless of source sensor),
             NOT Sentinel-2's native 10m. --ramen_input_res defaults to
             30.0 here accordingly (differs from the Sen1Floods11/EuroSAT
             scripts, which default to 10.0).
  - universat : UniverSat encoder (gastruc/UniverSat, AnySat v2) trained
             FROM SCRATCH (random init — no pretrained weights anywhere on
             this path) + linear per-token head, via
             training.Universat.universat_augmenter.UniverSatSegmenter.
             Like RAMEN it consumes the full modality dict
             (expects_full_image_dict), so batch["image"] = {"hls": tensor}
             passes straight through unchanged.

             Unlike RAMEN, NO window size is baked at construction: the
             latent grid is recomputed per input (H / patch_px per side),
             so the same weights handle the 256 train crops and the full
             512 eval image. Default eval is therefore a SINGLE full-image
             dense forward (window_size=None). If the full-image forward
             OOMs (most likely at --universat_output_stride 1, where the
             CA_Sub skip decodes per-pixel), set --universat_window_size
             to fall back to the same sliding-window machinery as RAMEN.

             Geometry constraints: every input side (train crop AND eval
             size / window) must be divisible by the patch size in pixels
             (--universat_patch_m / 30 = 8 px at the 240 m default) and by
             --universat_output_stride. The script auto-rounds --crop_size
             up to the nearest valid multiple (224 -> 256 with defaults).
             512 is divisible by 8 and 4, so full-image eval needs no
             adjustment. Logits come out at H/output_stride per side;
             BaselineTrainer bilinearly upsamples them to the target.

Same protocol as Sen1Floods11 baseline:
  - Same splits as PANGAEA (90/10 stratified train/val from training/, validation/ for test)
  - Same normalization (loaded from normalization_stats.pt or computed on train)
  - Same D4 augmentation
  - Train: 256×256 random crops; Eval: full 512×512 (UNet/ResNet/UniverSat via
    a single dense forward; RAMEN/ViT via sliding window)
  - ViT: train+eval at the same fixed size (--crop_size == --img_size)

# >>> FLOPS_METHOD: GFLOPs is now measured with
# torch.utils.flop_counter.FlopCounterMode (SDPA attention counted),
# matching script_universat_sweep_burnscars.py / _senflood.py and the
# RAMEN resolution-sweep scripts exactly — replacing the previous
# torch.profiler(with_flops=True) harness. This matters beyond
# consistency: torch.profiler's with_flops=True has no formulas for fused
# scaled_dot_product_attention kernels and silently drops ALL attention
# FLOPs — a large undercount for any transformer backbone here (ViT,
# RAMEN, UniverSat). Do NOT mix GFLOPs numbers produced by this script
# before this change with numbers produced after it, and do not mix them
# with any other torch.profiler-harness numbers anywhere in the paper.
# Measured once after testing completes: mean over --flops_n passes, bs=1,
# full image (RAMEN/ViT via the full sliding-window pass over all tiles —
# directly comparable to the other models' single dense forward;
# UniverSat via a single full-image dense forward by default, or the full
# sliding-window pass if --universat_window_size is set — matching
# whatever eval mode the reported mIoU used), one discarded warmup.
# Rank-zero only (avoids redundant profiling across DDP ranks). Disable
# with --no_flops.

Examples:
    # ResNet50 + UPerNet (matches Sen1Floods11 setup)
    python script_train_burnscars_baselines.py --xp_name resnet50 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 2 --lr 1e-4 --epochs 100

    # UNet baseline
    python script_train_burnscars_baselines.py --xp_name unet \
        --model unet \
        --batch_size 2 --lr 1e-3 --epochs 100

    # ViT-S at 512 (Sen1Floods11-native size)
    python script_train_burnscars_baselines.py --xp_name vit_512 \
        --model vit \
        --crop_size 512
        \
        --batch_size 2 --grad_accum 1 --lr 1e-4 --epochs 100

    # RAMEN
    python script_train_burnscars_baselines.py --xp_name ramen \
        --model ramen \
        --ramen_window_size 128 --ramen_stride 96 \
        --batch_size 2 --lr 1e-4 --epochs 100

    # UniverSat-S from scratch (~34.9M effective, parameter-matched):
    # 256 crops, full-image eval in one forward, logits at H/4
    python script_train_burnscars_baselines.py --xp_name universat \
        --model universat \
        --batch_size 2 --lr 1e-4 --epochs 100

    # UniverSat per-pixel decoding ablation (expensive CA_Sub setting);
    # windowed eval fallback shown in case full-image per-pixel OOMs
    python script_train_burnscars_baselines.py --xp_name universat_px \
        --model universat --universat_output_stride 1 \
        --universat_window_size 256 --universat_stride 192 \
        --batch_size 1 --grad_accum 2 --lr 1e-4 --epochs 100
"""

import os
import argparse

import torch
import torch.nn as nn
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
from torch.utils.flop_counter import FlopCounterMode   # >>> FLOPS_METHOD

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_burnscars_baselines import (
    BurnScarsBaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.Universat.universat_augmenter import build_universat_segmenter
from training.sliding_window import sliding_window_inference  # adjust import path
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES = BurnScarsBaselineDataset.NUM_CLASSES        # 2
IGNORE_INDEX = BurnScarsBaselineDataset.IGNORE_INDEX      # 255
NUM_CHANNELS = BurnScarsBaselineDataset.NUM_CHANNELS      # 6
MODALITY_KEY = "hls"
HLS_GSD_M = 30.0   # HLS common grid — shared by RAMEN (--ramen_input_res
                   # default) and UniverSat (patch_px = patch_m / GSD)


# =============================================================================
# RAMEN / UniverSat band metadata — single modality, keyed as "hls" to match
# the dataset's own dict key exactly (no adapter needed). Wavelengths keyed
# to HLS's OWN band naming (note "B8A", not "B08A"). UniverSat's adapter
# reuses these verbatim: it accepts the nm dict and converts to µm
# internally (UniverSat's registry convention), in HLS_BANDS order.
# =============================================================================

HLS_WAVELENGTHS_NM = {
    "B02": 492.4,   # Blue
    "B03": 559.8,   # Green
    "B04": 664.6,   # Red
    "B8A": 864.7,   # NIR narrow
    "B11": 1613.7,  # SWIR1
    "B12": 2202.4,  # SWIR2
}
assert set(HLS_WAVELENGTHS_NM.keys()) == set(BurnScarsBaselineDataset.HLS_BANDS), (
    "HLS_WAVELENGTHS_NM keys must exactly match BurnScarsBaselineDataset.HLS_BANDS "
    "— check for naming drift (e.g. 'B8A' vs 'B08A')."
)

RAMEN_INPUT_BANDS = {MODALITY_KEY: BurnScarsBaselineDataset.HLS_BANDS}
RAMEN_WAVELENGTHS = {MODALITY_KEY: HLS_WAVELENGTHS_NM}


# =============================================================================
# ViT FULL-IMAGE (SLIDING-WINDOW) ADAPTER
# =============================================================================

class ViTFullImageAdapter(nn.Module):
    """
    Wraps ViTUPerNet so it can go through the same sliding-window eval
    machinery as RAMEN (BaselineTrainer's window_size/window_stride path),
    to faithfully reproduce PANGAEA's protocol: a ViT built at a FIXED,
    small input_size (matching the pretrained-encoder convention — 224 in
    pangaea-bench's vit_scratch.yaml), tiled over the full native image
    via overlap-averaged sliding-window inference, rather than either a
    single center crop or a single oversized full-image forward.

    ViTUPerNet takes a plain [B,C,H,W] tensor, not a dict — this adapter
    unwraps the dict form sliding_window_inference crops generically, so
    the exact same windowing code used for RAMEN works unmodified here.
    """
    expects_full_image_dict = True

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: dict, **kwargs):
        return self.model(x[MODALITY_KEY], **kwargs)


# =============================================================================
# COLLATE
# =============================================================================

def burnscars_collate(batch):
    """Stack per-modality images, stack targets, keep metadata as list."""
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
        base = ViTUPerNet(
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
        return ViTFullImageAdapter(base)

    elif model_name == "resnet":
        return build_resnet_upernet(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "ramen":
        # Built at the small window size, NOT the full native image size —
        # RAMEN tokenizes at the pixel level. Full-resolution eval goes
        # through sliding_window_inference instead (handled inside
        # BaselineTrainer when window_size is set).
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

    elif model_name == "universat":
        # Random init, no HF download anywhere on this path. Unlike RAMEN,
        # no window size is baked at construction — the latent grid is
        # recomputed per input, so the same weights handle 256 crops and
        # the 512 full image. Reuses the RAMEN band metadata verbatim (the
        # adapter converts nm -> µm internally, in HLS_BANDS order).
        return build_universat_segmenter(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=num_classes,
            input_res={MODALITY_KEY: HLS_GSD_M},
            patch_size_m=args.universat_patch_m,
            output_stride=args.universat_output_stride,
            size=args.universat_size,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'unet', 'vit', "
            f"'resnet', 'ramen', 'universat'"
        )


def apply_ramen_config(args):
    """
    Same YAML-override logic as script_train_senflood_baseline.py: keys
    in --ramen_config override --ramen_* CLI defaults, unless that flag
    was explicitly passed on the command line.
    """
    if args.ramen_config is None:
        return args

    key_map = {
        "res": "ramen_res", "input_size": "ramen_window_size",
        "input_res": "ramen_input_res", "embed_dim": "ramen_embed_dim",
        "depth": "ramen_depth", "num_heads": "ramen_num_heads",
        "stride": "ramen_stride",
    }

    import sys
    explicit = {
        tok[2:].split("=")[0].replace("-", "_")
        for tok in sys.argv[1:] if tok.startswith("--")
    }

    with open(args.ramen_config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    for key, val in cfg.items():
        dest = key_map.get(key)
        if dest is None:
            print(f"[WARNING] Unrecognized key '{key}' in {args.ramen_config} "
                  f"— ignoring. Known keys: {sorted(key_map)}")
            continue
        if dest in explicit:
            print(f"[INFO] '{key}' in {args.ramen_config} ignored — "
                  f"--{dest} was explicitly set on the command line "
                  f"({getattr(args, dest)}).")
            continue
        print(f"[INFO] {args.ramen_config}: {key}={val} -> --{dest}")
        setattr(args, dest, val)

    if args.ramen_stride > args.ramen_window_size:
        if "ramen_stride" in explicit:
            raise ValueError(
                f"--ramen_stride ({args.ramen_stride}) must be <= "
                f"--ramen_window_size ({args.ramen_window_size})."
            )
        print(f"[WARNING] --ramen_stride ({args.ramen_stride}) exceeds "
              f"--ramen_window_size ({args.ramen_window_size}); clamping "
              f"to window_size (non-overlapping tiling).")
        args.ramen_stride = args.ramen_window_size

    return args


# =============================================================================
# >>> FLOPS_METHOD: GFLOPs MEASUREMENT — FlopCounterMode (SDPA attention
# counted), matching script_universat_sweep_burnscars.py / _senflood.py and
# the RAMEN resolution-sweep scripts. Replaces the previous
# torch.profiler(with_flops=True) harness — see the module docstring's
# FLOPS_METHOD note for why (silent attention-FLOPs undercount otherwise).
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
    One warmup pass discarded; each measured pass counted with
    FlopCounterMode; report mean / 1e9 (analytic and deterministic per
    shape — the mean is a sanity check). For RAMEN, forward_fn internally
    loops over sliding-window tiles, so this captures the TOTAL cost of
    one full-image forward — directly comparable to the other models'
    single dense forward pass.
    """
    for b in batches[:n_warmup]:
        _ = forward_fn(b)
    if device == "cuda":
        torch.cuda.synchronize()

    flops_list = []
    for b in batches[n_warmup:]:
        fc = FlopCounterMode(display=False)
        with fc:
            _ = forward_fn(b)
            if device == "cuda":
                torch.cuda.synchronize()
        flops_list.append(fc.get_total_flops())

    if not flops_list:
        return float("nan")
    return (sum(flops_list) / len(flops_list)) / 1e9


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="HLS BurnScars Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet", "ramen", "universat"])
parser.add_argument("--data_dir",  type=str, default="./data/hls_burn_scars")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training entirely and "
                         "test directly with this checkpoint's weights "
                         "(also used for the GFLOPs measurement below). "
                         "You must still pass the same --model / "
                         "--resnet_variant / --vit_* / --ramen_* / "
                         "--universat_* flags used when the checkpoint was "
                         "trained, since the architecture is rebuilt from "
                         "these flags, not from the checkpoint.")

# Training
parser.add_argument("--batch_size",   type=int, default=2)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=100)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=40)
parser.add_argument("--grad_accum",   type=int, default=1)

# Crop / image size
parser.add_argument("--crop_size", type=int, default=224,
                    help="Random crop for train. UNet/ResNet eval at full image "
                         "(direct forward); ViT/RAMEN eval via sliding-window "
                         "tiling over the full image, so --eval_size is left "
                         "None for them regardless of this default. ViT "
                         "requires --crop_size == --img_size == the model's "
                         "window size (positional embedding is baked at "
                         "construction). 224 is a standard ViT window choice "
                         "(14x14 patch grid at patch_size=16); independent of "
                         "--vit_embed_dim/--vit_num_heads, which control "
                         "parameter count/capacity, not window size. "
                         "UniverSat needs the crop divisible by its patch "
                         "size in px AND by --universat_output_stride; the "
                         "default 224 is auto-rounded up to 256 for it.")
parser.add_argument("--img_size",  type=int, default=224,
                    help="ViT positional embedding size (ignored by UNet/ResNet). "
                         "Tiled via sliding-window over the full native image at "
                         "eval — NOT the size ViT trains AND evaluates on "
                         "directly (that was this repo's earlier, buggy behavior).")
parser.add_argument("--eval_size", type=int, default=None,
                    help="Eval crop size. None = full image (UNet/ResNet/RAMEN/"
                         "UniverSat — RAMEN tiles the full image via sliding-"
                         "window inference; UniverSat runs a single dense "
                         "forward unless --universat_window_size is set). "
                         "ViT auto-forces this to --img_size.")

# UNet
parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])

# ViT — parameter-matched to Atomizer's ~34M budget: ViT-Small width
# (embed_dim=384, correct num_heads=6 for the standard ViT-S convention
# — num_heads barely affects param count, ~30.05M either way, but 6 is
# the actual named-variant convention, not the previous 8). This is a
# DIFFERENT goal from reproducing PANGAEA's literature number (81.58%),
# which used a much bigger ViT-Base (768/12/12, ~96M params) — the two
# aren't simultaneously satisfiable by one config; see script history/
# notes if you need the PANGAEA-reproduction settings instead
# (--vit_embed_dim 768 --vit_num_heads 12 --vit_output_layers 3 5 7 11
# --img_size 224). The sliding-window eval fix (ViTFullImageAdapter,
# below) is independent of model size and applies regardless of which
# variant you pick.
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256,
                    help="UPerNet decoder channels (also used by ResNet/RAMEN)")
parser.add_argument("--vit_eval_stride", type=int, default=None,
                    help="Sliding-window stride for full-image ViT eval. "
                         "Default: equal to --img_size (non-overlapping "
                         "tiling except a forced overlap at the flush "
                         "final window) — matches PANGAEA's 'fewest "
                         "patches that still cover the whole image'.")

# ResNet
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN
parser.add_argument("--ramen_embed_dim",   type=int, default=384)
parser.add_argument("--ramen_depth",       type=int, default=12)
parser.add_argument("--ramen_num_heads",   type=int, default=8)
parser.add_argument("--ramen_input_res",   type=float, default=30.0,
                    help="HLS is standardized to 30m GSD (harmonized Landsat/"
                         "Sentinel), NOT Sentinel-2's native 10m — differs "
                         "from the Sen1Floods11/EuroSAT scripts' default of 10.0.")
parser.add_argument("--ramen_res",         type=float, default=120,
                    help="Working resolution (m/px). Default equals "
                         "--ramen_input_res (no resampling).")
parser.add_argument("--ramen_window_size", type=int, default=512,
                    help="Spatial size RAMEN is built/trained at. --crop_size "
                         "is auto-overridden to this value for RAMEN.")
parser.add_argument("--ramen_stride",      type=int, default=96,
                    help="Sliding-window stride for full-image eval.")
parser.add_argument("--ramen_config",      type=str, default=None,
                    help="Optional YAML overriding --ramen_* args, same "
                         "format as script_train_senflood_baseline.py's "
                         "--ramen_config.")

# UniverSat (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"],
                    help="'small' (384-d, 12 SA blocks, heads=6) is ~36.1M "
                         "total / ~34.9M effective — parameter-matched to "
                         "the ViT-S / RAMEN ~34M budget. 'tiny' ~6.2M "
                         "replicates the repo's Tiny recipe; 'base' ~201M "
                         "replicates the released config.")
parser.add_argument("--universat_patch_m", type=float, default=480,
                    help="Patch size in METRES (UniverSat convention; scale "
                         "= patch_m/10 internally). 240 m = 8 px at HLS's "
                         "30 m GSD; 480 m = 16 px (ViT patch-size parity). "
                         "Must be an integer number of pixels, and every "
                         "input side must be divisible by that pixel count.")
parser.add_argument("--universat_output_stride", type=int, default=1,
                    help="Logits at H/stride per side (BaselineTrainer "
                         "bilinearly upsamples to the target). 1 = per-pixel "
                         "decoding via the CA_Sub sub-patch skip — the "
                         "paper's headline setting, and the expensive one.")
parser.add_argument("--universat_window_size", type=int, default=None,
                    help="None (default) = full-image eval in ONE dense "
                         "forward — UniverSat's grids scale natively, no "
                         "construction-time window exists. Set (e.g. 256) "
                         "to fall back to RAMEN-style sliding-window eval "
                         "if the 512 full-image forward OOMs (most likely "
                         "at --universat_output_stride 1). Must satisfy the "
                         "same divisibility constraints as --crop_size.")
parser.add_argument("--universat_stride", type=int, default=192,
                    help="Sliding-window stride for UniverSat eval; only "
                         "used when --universat_window_size is set.")

# GFLOPs
parser.add_argument("--flops", action="store_true", default=True,
                    help="Measure GFLOPs/forward on the final model after "
                         "testing (default: on).")
parser.add_argument("--no_flops", dest="flops", action="store_false",
                    help="Disable GFLOPs measurement.")
parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of profiled forward passes to average.")

args = parser.parse_args()
args = apply_ramen_config(args)

if args.test_only is not None and not os.path.exists(args.test_only):
    raise FileNotFoundError(f"--test_only checkpoint not found: {args.test_only}")


# =============================================================================
# SANITY CHECKS & SIZE RESOLUTION
# =============================================================================

if args.model == "vit":
    if args.crop_size != args.img_size:
        raise ValueError(
            f"For ViT: --crop_size ({args.crop_size}) must equal "
            f"--img_size ({args.img_size}). ViT positional embedding is "
            f"baked at construction; train and eval must use the same size."
        )
    if args.vit_eval_stride is None:
        args.vit_eval_stride = args.img_size
    if args.vit_eval_stride > args.img_size:
        print(f"[WARNING] --vit_eval_stride ({args.vit_eval_stride}) exceeds "
              f"--img_size ({args.img_size}); clamping to img_size "
              f"(non-overlapping tiling).")
        args.vit_eval_stride = args.img_size
    # Deliberately do NOT force args.eval_size here (unlike this repo's
    # earlier behavior, which set eval_size = img_size and evaluated ViT
    # on a single CENTER CROP). PANGAEA evaluates ViT on the FULL image
    # via overlap-averaged sliding-window tiling at input_size, not a
    # crop — leaving eval_size None makes val/test datasets return full
    # native images, which BaselineTrainer's window_size/window_stride
    # path (wired below) tiles exactly the same way it does for RAMEN.

if args.model == "ramen":
    if args.crop_size != args.ramen_window_size:
        print(f"[INFO] RAMEN: overriding --crop_size ({args.crop_size}) -> "
              f"--ramen_window_size ({args.ramen_window_size}). RAMEN trains "
              f"at its window size; --eval_size stays independent (None = "
              f"full image via sliding-window inference).")
        args.crop_size = args.ramen_window_size
    # Deliberately do NOT force args.eval_size here — leaving it None means
    # val/test datasets return full native images, which is what
    # sliding_window_inference (triggered inside BaselineTrainer via
    # window_size) needs to tile over.

if args.model == "universat":
    # Patch size must be an integer number of pixels at HLS's 30 m GSD.
    universat_patch_px = args.universat_patch_m / HLS_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {HLS_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px). Use a multiple of {HLS_GSD_M}."
        )
    universat_patch_px = int(round(universat_patch_px))

    # Every input side must be divisible by patch_px AND output_stride
    # (the adapter asserts this at forward time; better to catch it here).
    import math as _math
    _lcm = _math.lcm(universat_patch_px, args.universat_output_stride)

    if args.crop_size % _lcm:
        new_crop = ((args.crop_size + _lcm - 1) // _lcm) * _lcm
        print(f"[INFO] UniverSat: --crop_size ({args.crop_size}) not divisible "
              f"by lcm(patch_px={universat_patch_px}, "
              f"output_stride={args.universat_output_stride})={_lcm}; "
              f"rounding up -> {new_crop}.")
        args.crop_size = new_crop

    if args.universat_window_size is not None:
        if args.universat_window_size % _lcm:
            raise ValueError(
                f"--universat_window_size ({args.universat_window_size}) must "
                f"be divisible by lcm(patch_px={universat_patch_px}, "
                f"output_stride={args.universat_output_stride})={_lcm}."
            )
        if args.universat_stride > args.universat_window_size:
            print(f"[WARNING] --universat_stride ({args.universat_stride}) "
                  f"exceeds --universat_window_size "
                  f"({args.universat_window_size}); clamping to window_size "
                  f"(non-overlapping tiling).")
            args.universat_stride = args.universat_window_size

    if args.eval_size is not None and args.eval_size % _lcm:
        raise ValueError(
            f"--eval_size ({args.eval_size}) must be divisible by "
            f"lcm(patch_px, output_stride)={_lcm} for UniverSat. "
            f"(The default full 512 image is fine at the default settings.)"
        )
    # Deliberately do NOT force args.eval_size — None means val/test return
    # full native 512 images. Unlike RAMEN there's no baked-in window, so
    # by default UniverSat runs ONE dense forward over the full image
    # (BaselineTrainer skips tiling when window_size=None); setting
    # --universat_window_size opts back into the sliding-window path.


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  HLS BurnScars Baseline Training")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
if args.model == "ramen":
    print(f"  Embed dim:   {args.ramen_embed_dim}, depth={args.ramen_depth}, "
          f"heads={args.ramen_num_heads}")
    print(f"  Resolution:  input_res={args.ramen_input_res}, res={args.ramen_res} "
          f"({'no resampling' if args.ramen_input_res == args.ramen_res else 'resampled'})")
    print(f"  Window:      {args.ramen_window_size}x{args.ramen_window_size}, "
          f"eval stride={args.ramen_stride}")
if args.model == "universat":
    print(f"  Size:        {args.universat_size} (from scratch, random init)")
    print(f"  Patch:       {args.universat_patch_m:.0f} m "
          f"({int(args.universat_patch_m / HLS_GSD_M)} px @ {HLS_GSD_M:.0f} m)")
    print(f"  Out stride:  {args.universat_output_stride} "
          f"(logits at H/{args.universat_output_stride}, "
          f"upsampled by trainer)")
print(f"  Channels:    {NUM_CHANNELS} (HLS optical)")
print(f"  Train crop:  {args.crop_size}×{args.crop_size}")
if args.model in ("vit", "ramen"):
    eval_str = f"full image, sliding-window (window={args.img_size if args.model=='vit' else args.ramen_window_size}, stride={args.vit_eval_stride if args.model=='vit' else args.ramen_stride})"
elif args.model == "universat" and args.universat_window_size is not None:
    eval_str = (f"full image, sliding-window (window={args.universat_window_size}, "
                f"stride={args.universat_stride})")
elif args.eval_size:
    eval_str = f"{args.eval_size}×{args.eval_size} (center crop)"
else:
    eval_str = "full image, single dense forward"
print(f"  Eval size:   {eval_str}")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  Grad acc:    {args.grad_accum}")
print(f"  GPUs:        {torch.cuda.device_count()}")
print(f"  GFLOPs:      {'ON (n=' + str(args.flops_n) + ', FlopCounterMode)' if args.flops else 'OFF'}")
print(f"  Mode:        {'TEST ONLY (' + args.test_only + ')' if args.test_only else 'train + test'}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

if args.test_only is None:
    train_ds = BurnScarsBaselineDataset(
        root_path=args.data_dir, mode="train",
        crop_size=args.crop_size, augment=True,
    )
    val_ds = BurnScarsBaselineDataset(
        root_path=args.data_dir, mode="validation",
        crop_size=args.eval_size, augment=False,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
else:
    print(f"  [test-only mode] Skipping train/val dataset construction — "
          f"normalization_stats.pt must already exist at {args.data_dir}.")

test_ds = BurnScarsBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.eval_size, augment=False,
)
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs_eval = dict(
    batch_size=1,                       # full image — memory-conservative
    num_workers=args.num_workers,
    collate_fn=burnscars_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs_eval)

if args.test_only is None:
    loader_kwargs_train = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=burnscars_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs_train)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs_eval)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    temporal=False,                   # single-frame
    task="burnscars",
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
    window_size=(args.ramen_window_size if args.model == "ramen"
                 else args.img_size if args.model == "vit"
                 else args.universat_window_size if args.model == "universat"
                 else None),
    window_stride=(args.ramen_stride if args.model == "ramen"
                   else args.vit_eval_stride if args.model == "vit"
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
            project="Atomizer_BurnScars_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_BurnScars_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# CALLBACKS
# =============================================================================

ckpt_dir = "./checkpoints/burnscars_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = []
if args.test_only is None:
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

# find_unused_parameters=True already covers RAMEN's RadarProjector/
# DemProjector pattern (only relevant with SAR/DEM modalities, which
# BurnScars doesn't have — so RAMEN has no unused params here — but this
# was already set unconditionally for all models, so no change needed).
# For UniverSat it is REQUIRED, not incidental: in this single-modality
# snapshot setting several parameter tensors are structurally inert
# (Bi_ACA_in fusion attention with one modality, the UPE's S1-axis block
# at subpatch_px=1, the T-axis block at T=1, SAR/DEM channel codes —
# ~1.28M of the ~36.1M total) and never receive gradients.
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
# TRAIN (skipped in test-only mode)
# =============================================================================

if args.test_only is None:
    print(f"\n{'='*60}")
    print(f"  Starting: {args.model} on HLS BurnScars")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader)

    test_ckpt_path = "best"
else:
    print(f"\n{'='*60}")
    print(f"  [test-only mode] Skipping training")
    print(f"{'='*60}\n")

    test_ckpt_path = args.test_only


# =============================================================================
# TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {test_ckpt_path}")
print(f"{'='*60}\n")

trainer.test(trainer_module, test_loader, ckpt_path=test_ckpt_path)


# =============================================================================
# GFLOPs (rank-zero only, after best-checkpoint weights are loaded above)
# =============================================================================

if args.flops and os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"\n{'='*60}")
    print(f"  GFLOPs measurement — {args.model} (FlopCounterMode, on test set)")
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
            def fwd(b, m=eval_model):
                return sliding_window_inference(
                    m, b["image"],
                    window_size=args.ramen_window_size,
                    stride=args.ramen_stride,
                    num_classes=NUM_CLASSES,
                )
            size_note = "full image, sliding-window (all tiles)"
        elif args.model == "vit":
            def fwd(b, m=eval_model):
                return sliding_window_inference(
                    m, b["image"],
                    window_size=args.img_size,
                    stride=args.vit_eval_stride,
                    num_classes=NUM_CLASSES,
                )
            size_note = "full image, sliding-window (all tiles)"
        elif args.model == "universat":
            # Match whatever eval mode produced the reported mIoU: a single
            # full-image dense forward by default, or the full sliding-
            # window pass if --universat_window_size was set. UniverSat
            # consumes the modality dict directly (expects_full_image_dict).
            if args.universat_window_size is not None:
                def fwd(b, m=eval_model):
                    return sliding_window_inference(
                        m, b["image"],
                        window_size=args.universat_window_size,
                        stride=args.universat_stride,
                        num_classes=NUM_CLASSES,
                    )
                size_note = "full image, sliding-window (all tiles)"
            else:
                def fwd(b, m=eval_model):
                    return m(b["image"])
                size_note = "full image, single dense forward"
        else:
            def fwd(b, m=eval_model):
                return m(b["image"][MODALITY_KEY])
            size_note = "full image, single dense forward"

        gflops = measure_gflops_forward(fwd, flops_raw, device, n_warmup=1)
        print(f"  GFLOPs/forward (bs=1, {size_note}, FlopCounterMode "
              f"[SDPA counted]): {gflops:.2f}"
              f"  (mean of {len(flops_raw) - 1} passes)")

        if wandb_logger:
            import wandb
            wandb.log({"test_gflops": gflops})


if wandb_logger:
    import wandb
    wandb.finish()
