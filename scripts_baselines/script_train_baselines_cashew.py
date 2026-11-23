"""
Cashew Baseline Training Script
================================

Train baselines on geo-bench m-cashew-plant (7-class segmentation,
12 S2 bands, 256x256, single-temporal, SINGLE modality — optical only,
no SAR).

Models:
  - unet      : classic UNet
  - vit       : ViT + UPerNet
  - resnet    : ResNet + UPerNet
  - ramen     : RAMEN encoder + UPerNet decoder (per-band spectral
                encoding; single "optical" modality)
  - universat : UniverSat encoder (gastruc/UniverSat, AnySat v2) trained
                FROM SCRATCH + linear per-token head, via
                training.Universat.universat_augmenter.UniverSatSegmenter.

                SINGLE-MODALITY path (unlike the Sen1Floods11 script,
                which feeds {"optical","sar"} and thereby exercises the
                Bi_ACA_in cross-modal fusion block). Here only "optical"
                is present, so — same as the BurnScars script noted for
                single-modality inputs — Bi_ACA_in is structurally inert
                (no second modality to fuse against). This is expected
                and not a bug; it's the correct behavior for a
                single-sensor dataset, not a workaround.

                No window size baked at construction: the latent grid is
                recomputed per input (H / patch_px per side), so the same
                weights handle any crop and the full 256 image. Default
                eval is a SINGLE full-image dense forward
                (window_size=None); Cashew's native size (256) is small
                enough that sliding-window eval should rarely be needed,
                but --universat_window_size is kept available as a
                fallback (e.g. if --universat_output_stride 1 OOMs).

                Geometry constraints: every input side must be divisible
                by the patch size in pixels (--universat_patch_m / 10)
                and by --universat_output_stride. Defaults (80 m patch =
                8 px at 10 m, output_stride=4) divide 256 cleanly.

  - ptv3      : Point Transformer V3 (Pointcept) applied to the raster as
                a PIXELS-AS-POINTS point cloud, via
                training.PTv3.model_ptv3_seg.PTv3Segmenter.

                Each pixel becomes one point (x, y, 0) (metres = pixel
                index * GSD) whose feature is its 12 band values. The
                wrapper maps [B, 12, H, W] -> [B, num_classes, H, W], so
                BaselineTrainer / collate / checkpointing work unchanged.
                256x256 = 65,536 points per tile, so NO sliding window
                is needed (window_size=None, single dense forward).

                Uses the standalone `point_transformer_v3` package
                (github.com/Pointcept/PointTransformerV3): pure PyTorch +
                Triton, so NO spconv / torch_scatter / flash-attn needed.
                Make it importable with
                    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH
                Random init, no pretrained weights.

                NOTE on GFLOPs: FlopCounterMode may not see the Triton
                sparse-conv / varlen-attention kernels, so the reported
                GFLOPs for ptv3 can be an UNDER-ESTIMATE and may not be
                comparable to the other models. Treat as indicative only.

All single-frame, no LTAE.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training.

Examples:
    python script_train_cashew_baselines.py --xp_name resnet50 \
        --model resnet --resnet_variant resnet50 \
        --batch_size 16 --lr 1e-4 --epochs 80

    python script_train_cashew_baselines.py --xp_name ramen_optical \
        --model ramen \
        --batch_size 16 --lr 1e-4 --epochs 80

    python script_train_cashew_baselines.py --xp_name universat_optical \
        --model universat \
        --batch_size 16 --lr 1e-4 --epochs 80

    python script_train_cashew_baselines.py --xp_name ptv3_pixels \
        --model ptv3 --ptv3_patch 1024 \
        --ptv3_enc_channels 32 64 128 256 384 \
        --batch_size 16 --lr 1e-4 --epochs 80
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_cashew_baselines import (
    CashewBaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.RAMEN.ramen_upernet import build_ramen_upernet
from training.Universat.universat_augmenter import build_universat_segmenter
from training.PTv3.model_ptv3_seg import PTv3Segmenter
from training.sliding_window import sliding_window_inference
from training.trainer_baselines import BaselineTrainer


NUM_CLASSES  = CashewBaselineDataset.NUM_CLASSES   # 7
IGNORE_INDEX = CashewBaselineDataset.IGNORE_INDEX  # 255
NUM_CHANNELS = CashewBaselineDataset.NUM_CHANNELS  # 12
MODALITY_KEY = "s2"
CASHEW_GSD_M = 10.0  # single scalar used for RAMEN/UniverSat/PTv3 input_res,
                     # same simplification the Sen1Floods11 script makes
                     # for its S2 bands (which also natively vary 10/20/60m).


# =============================================================================
# RAMEN / UniverSat band metadata — optical only, in
# CashewBaselineDataset.BAND_PREFIXES order (fixes channel order):
#   02-Blue, 03-Green, 04-Red, 08-NIR, 05/06/07-RedEdge, 08A-RedEdge,
#   11-SWIR, 12-SWIR, 01-CoastalAerosol, 09-WaterVapour
# =============================================================================

S2_BAND_NAMES = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07",
    "B08A", "B11", "B12", "B01", "B09",
]

S2_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B08A": 864.7, "B09": 945.1, "B11": 1613.7, "B12": 2202.4,
}

RAMEN_INPUT_BANDS  = {"optical": S2_BAND_NAMES}
RAMEN_WAVELENGTHS  = {"optical": S2_WAVELENGTHS_NM}

UNIVERSAT_INPUT_BANDS = {"optical": S2_BAND_NAMES}
UNIVERSAT_WAVELENGTHS = {"optical": S2_WAVELENGTHS_NM}


# =============================================================================
# COLLATE
# =============================================================================

def cashew_collate(batch):
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


def cashew_collate_ramen(batch):
    """
    RAMEN / UniverSat collate for CashewBaselineDataset.

    Unlike Sen1Floods11 (which splits a merged S2+S1 tensor into
    {"optical","sar"}), Cashew has only one modality — this just wraps
    the existing image["s2"] tensor as {"optical": ...} so it matches the
    dict interface RAMENUPerNet / UniverSatSegmenter expect, with no
    channel split needed.
    """
    merged = torch.stack([s["image"]["s2"] for s in batch])  # [B, 12, H, W]
    images = {"optical": merged}
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, num_classes, args):
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
        return build_universat_segmenter(
            input_bands=UNIVERSAT_INPUT_BANDS,
            wavelengths=UNIVERSAT_WAVELENGTHS,
            num_classes=num_classes,
            input_res={"optical": CASHEW_GSD_M},
            patch_size_m=args.universat_patch_m,
            output_stride=args.universat_output_stride,
            size=args.universat_size,
        )
    elif model_name == "ptv3":
        ptv3_kwargs = {}
        if args.ptv3_enc_channels is not None:
            ptv3_kwargs["enc_channels"] = tuple(args.ptv3_enc_channels)
        if args.ptv3_enc_depths is not None:
            ptv3_kwargs["enc_depths"] = tuple(args.ptv3_enc_depths)
        if args.ptv3_dec_channels is not None:
            ptv3_kwargs["dec_channels"] = tuple(args.ptv3_dec_channels)
        if args.ptv3_dec_depths is not None:
            ptv3_kwargs["dec_depths"] = tuple(args.ptv3_dec_depths)
        return PTv3Segmenter(
            in_channels=in_channels,
            num_classes=num_classes,
            gsd=CASHEW_GSD_M,
            patch_size=args.ptv3_patch,
            head_dim=args.ptv3_head_dim,
            drop_path=args.ptv3_drop_path,
            **ptv3_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'unet', 'vit', "
            f"'resnet', 'ramen', 'universat', 'ptv3'"
        )


# =============================================================================
# GFLOPs MEASUREMENT — same harness as the Sen1Floods11 baseline script
# =============================================================================

# =============================================================================
# GFLOPs MEASUREMENT — FlopCounterMode (counts SDPA attention), same
# harness as script_universat_sweep_senflood.py. NOTE: these numbers are
# NOT comparable to any earlier run of this script that used the old
# torch.profiler(with_flops=True) harness — the two methods count
# differently and must not be mixed when reporting.
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
    """One warmup pass discarded; each measured pass counted with
    FlopCounterMode; report mean / 1e9 (analytic and deterministic per
    shape — the mean is a sanity check, not a variance measurement)."""
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

parser = argparse.ArgumentParser(description="Cashew Baseline Training")
parser.add_argument("--xp_name",   type=str, required=True)
parser.add_argument("--model",     type=str, default="resnet",
                    choices=["unet", "vit", "resnet", "ramen", "universat",
                             "ptv3"])
parser.add_argument("--data_dir",  type=str,
                    default="./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant")

parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")
parser.add_argument("--resume",    type=str, default=None,
                    help="Path to a .ckpt file to resume training from "
                         "(full trainer state: model, optimizer, LR "
                         "scheduler, epoch/step counters, callback state "
                         "e.g. EarlyStopping/ModelCheckpoint's best-score "
                         "tracking). Passed to Trainer.fit(ckpt_path=...). "
                         "Ignored if --test_only is set. If omitted, "
                         "training starts from scratch.")

parser.add_argument("--batch_size",   type=int, default=16)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",       type=int, default=80)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--patience",     type=int, default=20)
parser.add_argument("--grad_accum",   type=int, default=1)
parser.add_argument("--precision",    type=str, default="bf16-mixed",
                    help="Lightning precision string for train AND test "
                         "(default bf16-mixed, as for all other baselines). "
                         "If spconv (ptv3) complains about bf16, try "
                         "16-mixed.")

parser.add_argument("--crop_size", type=int, default=None,
                    help="Crop size (None = full 256x256 native). For "
                         "ramen, this is OVERRIDDEN by --ramen_window_size "
                         "(see sanity checks below).")
parser.add_argument("--img_size",  type=int, default=256,
                    help="ViT positional embedding size.")

parser.add_argument("--unet_topology", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])

parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256,
                    help="UPerNet decoder channels (also used by ResNet and RAMEN)")

parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=10.0,
                    help="Native GSD (m/px) of the input imagery.")
parser.add_argument("--ramen_res",       type=float, default=20.0,
                    help="Common working resolution (m/px). Left equal to "
                         "--ramen_input_res by default (no resampling).")
parser.add_argument("--ramen_window_size", type=int, default=256,
                    help="RAMEN tokenizes at the pixel level, so full "
                         "self-attention over 256x256 (Cashew's native "
                         "size) is still heavy. Model is built/trained at "
                         "this smaller size; training crops use this "
                         "value directly (overrides --crop_size for "
                         "--model ramen). Full 256x256 val/test is handled "
                         "via sliding-window inference.")
parser.add_argument("--ramen_stride", type=int, default=96,
                    help="Sliding-window stride for RAMEN eval. Must be "
                         "<= --ramen_window_size. Ignored during training.")
parser.add_argument("--ramen_config", type=str, default=None,
                    help="Optional path to a RAMEN YAML config whose keys "
                         "override the --ramen_* CLI defaults above.")

# UniverSat (from scratch)
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"])
parser.add_argument("--universat_patch_m", type=float, default=40.0,
                    help="Patch size in METRES. 80 m = 8 px at Cashew's "
                         "10 m working GSD. Must be an integer number of "
                         "pixels, and every input side must be divisible "
                         "by that pixel count.")
parser.add_argument("--universat_output_stride", type=int, default=1,
                    help="Logits at H/stride per side (BaselineTrainer "
                         "bilinearly upsamples to the target).")
parser.add_argument("--universat_window_size", type=int, default=None,
                    help="None (default) = full-image eval in ONE dense "
                         "forward. Set (e.g. 128) to fall back to "
                         "sliding-window eval if the 256 full-image "
                         "forward OOMs (most likely at "
                         "--universat_output_stride 1).")
parser.add_argument("--universat_stride", type=int, default=96,
                    help="Sliding-window stride for UniverSat eval; only "
                         "used when --universat_window_size is set.")

# PTv3 (pixels-as-points, from scratch)
parser.add_argument("--ptv3_patch", type=int, default=1024,
                    help="PTv3 serialized-attention patch size (points per "
                         "attention window), used at every stage. Attention "
                         "uses torch's varlen kernel, so the default 1024 "
                         "should be fine; if you hit memory errors, try "
                         "256-512.")
parser.add_argument("--ptv3_enc_channels", type=int, nargs=5, default=None,
                    help="Override PTv3 encoder channels (5 values), e.g. "
                         "32 64 128 256 384 to shrink the model. Default: "
                         "32 64 128 256 512 (~46M params). Each value must "
                         "be divisible by --ptv3_head_dim.")
parser.add_argument("--ptv3_enc_depths", type=int, nargs=5, default=None,
                    help="Override PTv3 encoder depths (5 values). "
                         "Default: 2 2 2 6 2.")
parser.add_argument("--ptv3_dec_channels", type=int, nargs=4, default=None,
                    help="Override PTv3 decoder channels (4 values). The "
                         "first value sets the linear head's input width. "
                         "Default: 64 64 128 256.")
parser.add_argument("--ptv3_dec_depths", type=int, nargs=4, default=None,
                    help="Override PTv3 decoder depths (4 values). "
                         "Default: 2 2 2 2.")
parser.add_argument("--ptv3_head_dim", type=int, default=16,
                    help="Channels per attention head (number of heads = "
                         "channels / head_dim at every stage). Default 16, "
                         "as in the reference recipe.")
parser.add_argument("--ptv3_drop_path", type=float, default=0.3,
                    help="Largest DropPath rate (reference recipe: 0.3).")

# GFLOPs
parser.add_argument("--flops", action="store_true", default=True)
parser.add_argument("--no_flops", dest="flops", action="store_false")
parser.add_argument("--flops_n", type=int, default=3)

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise FileNotFoundError(
        f"--resume checkpoint not found: {args.resume}"
    )


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
        _ramen_cfg = yaml.safe_load(f) or {}
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
# SANITY CHECKS
# =============================================================================

if args.model == "vit":
    eff_size = args.crop_size if args.crop_size is not None else 256
    if eff_size != args.img_size:
        raise ValueError(
            f"For ViT: input size ({eff_size}) must equal --img_size "
            f"({args.img_size}). Default: both 256."
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
    if args.crop_size != args.ramen_window_size:
        print(f"[INFO] --model ramen: overriding --crop_size "
              f"({args.crop_size}) with --ramen_window_size "
              f"({args.ramen_window_size}) for training crops.")
        args.crop_size = args.ramen_window_size

if args.model == "universat":
    universat_patch_px = args.universat_patch_m / CASHEW_GSD_M
    if abs(universat_patch_px - round(universat_patch_px)) > 1e-6:
        raise ValueError(
            f"--universat_patch_m ({args.universat_patch_m}) is not an "
            f"integer number of pixels at {CASHEW_GSD_M} m GSD "
            f"({universat_patch_px:.3f} px). Use a multiple of "
            f"{CASHEW_GSD_M}."
        )
    universat_patch_px = int(round(universat_patch_px))

    import math as _math
    _lcm = _math.lcm(universat_patch_px, args.universat_output_stride)

    _base_crop = args.crop_size if args.crop_size is not None else 256
    if _base_crop % _lcm:
        new_crop = ((_base_crop + _lcm - 1) // _lcm) * _lcm
        print(f"[INFO] UniverSat: crop size ({_base_crop}) not divisible "
              f"by lcm(patch_px={universat_patch_px}, "
              f"output_stride={args.universat_output_stride})={_lcm}; "
              f"rounding up -> {new_crop}.")
        args.crop_size = new_crop

    if 256 % _lcm:
        raise ValueError(
            f"The full 256x256 eval image is not divisible by "
            f"lcm(patch_px={universat_patch_px}, "
            f"output_stride={args.universat_output_stride})={_lcm} — "
            f"pick --universat_patch_m / --universat_output_stride so "
            f"that 256 is a valid input side (or set "
            f"--universat_window_size to a valid size)."
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

if args.model == "ptv3":
    _enc = args.ptv3_enc_channels or [32, 64, 128, 256, 512]
    _dec = args.ptv3_dec_channels or [64, 64, 128, 256]
    _bad = [c for c in (*_enc, *_dec) if c % args.ptv3_head_dim]
    if _bad:
        raise ValueError(
            f"PTv3 channel widths {_bad} are not divisible by "
            f"--ptv3_head_dim ({args.ptv3_head_dim})."
        )
    # No geometric constraint on the crop: each pixel is a point, and
    # PTv3 handles arbitrary point counts per sample (via `offset`).


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  Cashew Baseline Training")
print(f"  Model:       {args.model}")
if args.model == "resnet":
    print(f"  Variant:     {args.resnet_variant}")
if args.model == "ramen":
    print(f"  Window size: {args.ramen_window_size}x{args.ramen_window_size} "
          f"(model built/trained at this size)")
    print(f"  Eval stride: {args.ramen_stride} "
          f"(sliding-window inference over full 256x256)")
if args.model == "universat":
    print(f"  Size:        {args.universat_size} (from scratch, random init)")
    print(f"  Patch:       {args.universat_patch_m:.0f} m "
          f"({int(args.universat_patch_m / CASHEW_GSD_M)} px @ "
          f"{CASHEW_GSD_M:.0f} m)")
    print(f"  Out stride:  {args.universat_output_stride}")
    if args.universat_window_size is not None:
        print(f"  Eval:        sliding-window (window="
              f"{args.universat_window_size}, stride={args.universat_stride})")
    else:
        print(f"  Eval:        full 256x256, single dense forward")
if args.model == "ptv3":
    print(f"  Mode:        pixels-as-points (1 pixel = 1 point), from scratch")
    print(f"  Patch size:  {args.ptv3_patch} points")
    print(f"  Head dim:    {args.ptv3_head_dim}")
    print(f"  Drop path:   {args.ptv3_drop_path}")
    if args.ptv3_enc_channels is not None:
        print(f"  Enc ch:      {tuple(args.ptv3_enc_channels)}")
    if args.ptv3_enc_depths is not None:
        print(f"  Enc depths:  {tuple(args.ptv3_enc_depths)}")
    if args.ptv3_dec_channels is not None:
        print(f"  Dec ch:      {tuple(args.ptv3_dec_channels)}")
    if args.ptv3_dec_depths is not None:
        print(f"  Dec depths:  {tuple(args.ptv3_dec_depths)}")
    print(f"  Eval:        full 256x256, single dense forward "
          f"(65,536 points/tile)")
print(f"  Channels:    {NUM_CHANNELS} (S2 12 bands, optical only)")
print(f"  Classes:     {NUM_CLASSES}")
crop_str = f"{args.crop_size}x{args.crop_size}" if args.crop_size else "256x256 (full)"
print(f"  Input size:  {crop_str}")
print(f"  Epochs:      {args.epochs}")
print(f"  BS:          {args.batch_size}")
print(f"  LR:          {args.lr}")
print(f"  Precision:   {args.precision}")
print(f"  GPUs:        {torch.cuda.device_count()}")
if args.resume is not None:
    print(f"  Resuming from: {args.resume}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

train_ds = CashewBaselineDataset(
    root_path=args.data_dir, mode="train",
    crop_size=args.crop_size, augment=True,
)
val_ds = CashewBaselineDataset(
    root_path=args.data_dir, mode="validation",
    crop_size=args.crop_size, augment=False,
)
test_ds = CashewBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=args.crop_size, augment=False,
)

print(f"  Train: {len(train_ds)} samples")
print(f"  Val:   {len(val_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")


# =============================================================================
# DATALOADERS
# =============================================================================

collate_fn = (cashew_collate_ramen if args.model in ("ramen", "universat")
              else cashew_collate)

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

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)

trainer_module = BaselineTrainer(
    model=model,
    modality=MODALITY_KEY if args.model not in ("ramen", "universat")
             else "optical",
    temporal=False,
    task="cashew",
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
        if args.model == "ptv3":
            run_name += f"_p{args.ptv3_patch}"
        wandb.init(
            name=run_name,
            project="Atomizer_Cashew_Baselines",
            config=vars(args),
        )
        wandb_logger = WandbLogger(project="Atomizer_Cashew_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/cashew_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

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

    # find_unused_parameters=True: RAMEN's RadarProjector/DemProjector-style
    # unused-branch pattern and UniverSat's inert S1/T-axis blocks and
    # unused sensor channel codes (see UniverSat docstring above) both need
    # this, same as the Sen1Floods11 script. Harmless for unet/vit/resnet/ptv3.
    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=-1,
        max_epochs=args.epochs,
        accelerator="gpu",
        precision=args.precision,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
    )

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
    print(f"\n[test-only mode] Skipping training, testing checkpoint:")
    print(f"  {best_ckpt}\n")


# =============================================================================
# SINGLE-GPU TEST
# =============================================================================

print(f"\n{'='*60}")
print(f"  Testing checkpoint: {best_ckpt}")
print(f"{'='*60}\n")

test_trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision=args.precision,
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
            size_note = "full 256x256, sliding-window (all tiles)"
        elif args.model == "universat":
            if args.universat_window_size is not None:
                def fwd(b, m=eval_model):
                    return sliding_window_inference(
                        m, b["image"],
                        window_size=args.universat_window_size,
                        stride=args.universat_stride,
                        num_classes=NUM_CLASSES,
                    )
                size_note = "full 256x256, sliding-window (all tiles)"
            else:
                def fwd(b, m=eval_model):
                    return m(b["image"])
                size_note = "full 256x256, single dense forward"
        elif args.model == "ptv3":
            def fwd(b, m=eval_model):
                return m(b["image"][MODALITY_KEY])
            size_note = ("full 256x256, single dense forward; "
                         "possible UNDER-ESTIMATE: Triton sparse-conv / "
                         "varlen-attention kernels may not be counted by "
                         "FlopCounterMode")
            print("[FLOPs] WARNING: PTv3 GFLOPs may exclude Triton "
                  "sparse-conv / attention kernels and may not be "
                  "comparable to the other models.")
        else:
            def fwd(b, m=eval_model):
                return m(b["image"][MODALITY_KEY])
            size_note = "full 256x256, single dense forward"

        gflops = measure_gflops_forward(fwd, flops_raw, device, n_warmup=1)
        print(f"  GFLOPs/forward (bs=1, {size_note}): {gflops:.2f}"
              f"  (mean of {len(flops_raw) - 1} passes)")

        if wandb_logger:
            import wandb
            wandb.log({"test_gflops": gflops})

if wandb_logger:
    import wandb
    wandb.finish()
