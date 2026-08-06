"""
BioMassters Baseline Training Script
=====================================

Train AGB regression baselines on BioMassters.

Supported models (via --model):
  - resnet_upernet_mt : ResNet + channel-concat early fusion (TimeMerge
                        DoubleConv) + UPerNet, regression head (num_classes=1).
                        This is the "ResNet with double-conv in input" baseline.
  - vit_ltae          : ViT per-frame + LTAE temporal aggregation + UPerNet,
                        regression head (num_classes=1). This is the
                        "ViT with LTAE at the end" baseline.
  - ramen             : RAMENUPerNet (multi-modal encoder, per-modality LTAE
                        temporal fusion, shared ViT stack), regression head
                        (num_classes=1). Uses a DIFFERENT collate than the
                        other models (make_ramen_collate, not
                        make_fused_collate) -- modalities are kept as separate
                        "optical"/"sar" dict entries in [B,C,T,H,W] layout,
                        since RAMEN's forward() branches on the modality key
                        name and expects channel-before-time.
  - perceiver         : PerceiverSeg (Perceiver-IO), regression head
                        (num_classes=1). Uses the SAME fused collate as
                        resnet_upernet_mt/vit_ltae -- receives [B, T, C, H, W]
                        directly, DOY from batch["dates"]["s2"] via `doy`
                        kwarg (matches BaselineRegressionTrainer's default,
                        no temporal_kwarg override needed).

Regression via num_classes=1: same trick used throughout this codebase
(Atomizer's reconstruction_head, trainer_biomassters.py) -- these backbones'
final head is sized by num_classes, so passing num_classes=1 turns the
classification head into a single-channel regression head with no
architectural changes needed.

--test_only mode:
    Pass --test_only <path/to/checkpoint.ckpt> to skip training and run
    test on a saved checkpoint (single GPU, no DDP).

--resume_from mode (NEW):
    Pass --resume_from <path/to/checkpoint.ckpt> to resume training
    (full trainer state: model, optimizer, LR scheduler, epoch/step
    counters, callback state) from that checkpoint via
    Trainer.fit(ckpt_path=...). If the file doesn't exist yet, use
    --resume_wait_seconds to poll for it instead of failing immediately
    (useful for chained SLURM jobs where the next job in the chain can
    start before the previous job's checkpoint write has landed on disk).

Examples:
    # ResNet + early fusion, last 3 months, S2+S1
    python script_train_biomassters_baseline.py --xp_name resnet_mt_t3 \
        --model resnet_upernet_mt --resnet_variant resnet50 \
        --multi_temporal 3 --temporal_last --batch_size 8 --lr 1e-4 --epochs 100

    # ViT + LTAE, last 3 months, S2+S1
    python script_train_biomassters_baseline.py --xp_name vit_ltae_t3 \
        --model vit_ltae --multi_temporal 3 --temporal_last \
        --batch_size 8 --lr 1e-4 --epochs 100

    # RAMEN, last 3 months, S2+S1
    python script_train_biomassters_baseline.py --xp_name ramen_t3 \
        --model ramen --multi_temporal 3 --temporal_last \
        --batch_size 8 --lr 1e-4 --epochs 100

    # Perceiver-IO, last 3 months, S2+S1
    python script_train_biomassters_baseline.py --xp_name perceiver_t3 \
        --model perceiver --multi_temporal 3 --temporal_last \
        --batch_size 8 --lr 1e-4 --epochs 100

    # Resume a chained SLURM job, waiting up to 10 minutes for the checkpoint
    python script_train_biomassters_baseline.py --xp_name resnet_mt_t3 \
        --model resnet_upernet_mt \
        --resume_from ./checkpoints/biomassters_baselines/bl_resnet_mt_t3_resnet_upernet_mt-last.ckpt \
        --resume_wait_seconds 600
"""

import os
import time
import argparse

import torch
import torch.nn as nn
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

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_biomassters import (
    BioMasstersBaselineDataset, NUM_S2_BANDS, NUM_S1_BANDS, IGNORE_VALUE,
)
from training.VIT.model_vit_upernet import ViTLTAEUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet_mt
from training.RAMEN.ramen_upernet import build_ramen_upernet
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.trainer_baselines_biomassters import BaselineRegressionTrainer


# =============================================================================
# RESUME HELPER (mirrors train_biomassters.py's wait_for_checkpoint)
# =============================================================================

def wait_for_checkpoint(path: str, wait_seconds: int, poll_interval: int = 15) -> str:
    """
    Polls for `path` to exist, up to `wait_seconds` total, checking every
    `poll_interval` seconds. Useful for chained SLURM jobs where the next
    job in the chain can start before the previous job's checkpoint write
    (and any filesystem sync delay, common on Lustre) has actually landed.

    wait_seconds=0 means "check once, don't wait" -- fails fast if the
    file isn't there.

    Raises FileNotFoundError if the checkpoint never appears within the
    timeout, rather than silently falling back to training from scratch --
    resuming should be an explicit, verified action.
    """
    if os.path.exists(path):
        return path

    if wait_seconds <= 0:
        raise FileNotFoundError(
            f"Checkpoint not found: {path} "
            f"(use --resume_wait_seconds > 0 to poll for it instead of failing immediately)"
        )

    print(f"[BioMassters-BL] Checkpoint not found yet: {path}")
    print(f"[BioMassters-BL] Waiting up to {wait_seconds}s (polling every {poll_interval}s)...")
    waited = 0
    while waited < wait_seconds:
        time.sleep(poll_interval)
        waited += poll_interval
        if os.path.exists(path):
            print(f"[BioMassters-BL] Checkpoint appeared after {waited}s: {path}")
            return path
        print(f"[BioMassters-BL]   ...still waiting ({waited}/{wait_seconds}s)")

    raise FileNotFoundError(
        f"Checkpoint still not found after waiting {wait_seconds}s: {path}"
    )


# =============================================================================
# COLLATE — handles nested dicts (image, dates, target, metadata)
# =============================================================================

def biomassters_collate(batch):
    images = {}
    dates = {}
    sensor_keys = list(batch[0]["image"].keys())

    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
        dates[key] = torch.stack([s["dates"][key] for s in batch])

    targets = torch.stack([s["target"] for s in batch])  # [B, 1, H, W] float
    metadata = [s["metadata"] for s in batch]

    return {
        "image": images,
        "dates": dates,
        "target": targets,
        "metadata": metadata,
    }


def make_fused_collate():
    """
    Returns a collate that fuses S2 and S1 along the channel dim, matching
    PASTIS's make_fused_collate. BioMassters ALWAYS uses both sensors (no
    S2-only mode the way PASTIS has), so this is unconditional here rather
    than gated behind a --use_s1 flag.
    """
    def fused_collate(batch):
        out = biomassters_collate(batch)
        s2 = out["image"]["s2"]  # [B, T, 10, H, W]
        s1 = out["image"]["s1"]  # [B, T,  4, H, W]
        T = min(s2.shape[1], s1.shape[1])
        fused = torch.cat([s2[:, :T], s1[:, :T]], dim=2)  # [B, T, 14, H, W]
        out["image"] = {"s2": fused}
        out["dates"] = {"s2": out["dates"]["s2"][:, :T]}
        return out

    return fused_collate


def make_ramen_collate():
    """
    RAMEN needs a fundamentally different layout than the fused collate:
      1. Modalities kept SEPARATE (not concatenated) -- "optical"/"sar" keys,
         since RAMENBackbone spectrally projects and LTAE-fuses each
         modality independently before the shared ViT stack.
      2. Renamed from "s2"/"s1" to "optical"/"sar" -- RAMEN's forward()
         branches on the literal string "sar" (RadarProjector) vs anything
         else (SpectralProjector), so the key name is load-bearing, not
         cosmetic.
      3. Channel-before-time: [B, C, T, H, W], not [B, T, C, H, W] like
         everything else in this codebase -- RAMENBackbone.forward expects
         x[modality].dim() == 5 with that specific axis order.
    """
    def ramen_collate(batch):
        out = biomassters_collate(batch)
        s2 = out["image"]["s2"]  # [B, T, 10, H, W]
        s1 = out["image"]["s1"]  # [B, T,  4, H, W]
        out["image"] = {
            "optical": s2.permute(0, 2, 1, 3, 4).contiguous(),  # [B, 10, T, H, W]
            "sar":     s1.permute(0, 2, 1, 3, 4).contiguous(),  # [B, 4,  T, H, W]
        }
        out["dates"] = {
            "optical": out["dates"]["s2"],  # [B, T]
            "sar":     out["dates"]["s1"],  # [B, T]
        }
        return out

    return ramen_collate


def make_channel_stack_collate(base_collate):
    """[B, T, C, H, W] → [B, T*C, H, W] for non-temporal models."""
    def stacked_collate(batch):
        out = base_collate(batch)
        for key, img in out["image"].items():
            if img.dim() == 5:
                B, T, C, H, W = img.shape
                out["image"][key] = img.reshape(B, T * C, H, W)
        return out
    return stacked_collate


# =============================================================================
# RAMEN BAND INFO
# =============================================================================
# RAMEN wants input_bands/wavelengths as dict[modality][band] -> value,
# with the modality keys "optical"/"sar" (see make_ramen_collate). Physical
# S2 wavelengths match what's used throughout this codebase (Lookup_encoding's
# create_biomassters_bands_info). SAR band names map to RadarProjector's
# pol_map strings -- we only have ascending/descending VV/VH (no HH/HV), so
# only those 4 pol_map entries are used, in the SAME channel order as
# BioMasstersBaselineDataset produces (VV_asc, VH_asc, VV_desc, VH_desc).

RAMEN_S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
RAMEN_S2_WAVELENGTHS = {
    "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}
# Channel order MUST match BioMasstersBaselineDataset's S1 band order
# (VV_asc, VH_asc, VV_desc, VH_desc) -- see NUM_S1_BANDS in that dataset.
RAMEN_S1_BAND_ORDER = ["VV_asc", "VH_asc", "VV_desc", "VH_desc"]
RAMEN_S1_POLARIZATIONS = {
    "VV_asc": "asc_vv", "VH_asc": "asc_vh",
    "VV_desc": "des_vv", "VH_desc": "des_vh",
}

RAMEN_INPUT_BANDS = {"optical": RAMEN_S2_BAND_ORDER, "sar": RAMEN_S1_BAND_ORDER}
RAMEN_WAVELENGTHS = {"optical": RAMEN_S2_WAVELENGTHS, "sar": RAMEN_S1_POLARIZATIONS}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name, in_channels, args):
    """Dispatch to the requested model architecture. num_classes=1 -> regression."""
    if model_name == "vit_ltae":
        return ViTLTAEUPerNet(
            in_channels=in_channels,
            num_classes=1,
            img_size=args.img_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
            ltae_n_head=args.n_heads,
            ltae_d_k=args.d_k,
            ltae_d_model=args.d_model,
        )

    elif model_name == "resnet_upernet_mt":
        return build_resnet_upernet_mt(
            variant=args.resnet_variant,
            in_channels=in_channels,
            num_classes=1,
            num_frames=args.multi_temporal,
            decoder_channels=args.vit_decoder_channels,
        )

    elif model_name == "ramen":
        # Built at --ramen_window_size (default = full 256, i.e. no tiling
        # needed at BioMassters' native tile size) -- see the sanity-check
        # block below and the module docstring note on sliding-window
        # inference NOT being implemented here (unlike Sen1Floods11's
        # script) for window_size < 256.
        model = build_ramen_upernet(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=1,
            input_size=args.ramen_window_size,
            embed_dim=args.ramen_embed_dim,
            depth=args.ramen_depth,
            num_heads=args.ramen_num_heads,
            input_res=args.ramen_input_res,
            res=args.ramen_res,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )
        # RAMENUPerNet.forward(x, dates=...) uses `dates`, not `doy` --
        # BaselineRegressionTrainer.forward() reads this attribute to pick
        # the right kwarg name (see trainer_baselines_biomassters.py).
        model.temporal_kwarg = "dates"
        return model

    elif model_name == "perceiver":
        # Same fused collate/"s2" key convention as resnet_upernet_mt/vit_ltae
        # -- no temporal_kwarg override needed, PerceiverSeg.forward already
        # uses `doy` (matches BaselineRegressionTrainer.forward's default).
        return PerceiverSeg(
            in_channels=in_channels,
            num_classes=1,
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
            attn_dropout=args.perceiver_attn_dropout,
            ff_dropout=args.perceiver_ff_dropout,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'resnet_upernet_mt', "
            f"'vit_ltae', 'ramen', 'perceiver'"
        )


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="BioMassters Baseline Training")
parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--model",      type=str, default="resnet_upernet_mt",
                    choices=["resnet_upernet_mt", "vit_ltae", "ramen", "perceiver"])
parser.add_argument("--data_dir",   type=str, default="./data/biomassters")

# Test-only mode
parser.add_argument("--test_only", type=str, default=None,
                    help="Path to a .ckpt file. Skip training, test directly.")

# Resume-training mode (NEW)
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint to resume TRAINING from (full "
                         "trainer state via Trainer.fit(ckpt_path=...)). If "
                         "the file doesn't exist yet, use --resume_wait_seconds "
                         "to poll for it instead of failing immediately.")
parser.add_argument("--resume_wait_seconds", type=int, default=0,
                    help="How long to poll for --resume_from (or --test_only, "
                         "reused for both) to appear before giving up "
                         "(0 = check once, fail immediately if missing).")
parser.add_argument("--resume_poll_interval", type=int, default=15,
                    help="Seconds between polls while waiting for the checkpoint.")

# Temporal
parser.add_argument("--multi_temporal", type=int, default=3,
                    help="Number of temporal frames to use")
parser.add_argument("--temporal_last",  action="store_true", default=True,
                    help="Take last N timesteps instead of uniform sampling "
                         "(default True, matching the Atomizer comparison run)")

# Training
parser.add_argument("--batch_size",  type=int, default=8)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--epochs",      type=int, default=150)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--patience",    type=int, default=30)
parser.add_argument("--grad_accum",  type=int, default=1)

# LTAE shared params (used by vit_ltae)
parser.add_argument("--n_heads",    type=int, default=16)
parser.add_argument("--d_k",        type=int, default=4)
parser.add_argument("--d_model",    type=int, default=256)

# ViT-specific
parser.add_argument("--img_size",          type=int, default=256)
parser.add_argument("--vit_embed_dim",     type=int, default=384)
parser.add_argument("--vit_depth",         type=int, default=12)
parser.add_argument("--vit_num_heads",     type=int, default=6)
parser.add_argument("--vit_patch_size",    type=int, default=16)
parser.add_argument("--vit_output_layers", type=int, nargs="+", default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)

# ResNet-specific
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN-specific
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=10.0,
                    help="Native GSD (m/px) of the input imagery.")
parser.add_argument("--ramen_res",       type=float, default=40.0,
                    help="Common working resolution (m/px) all modalities are "
                         "resampled to before the shared ViT stack -- fixes "
                         "effective_size/token count. Native GSD is 10.0; the "
                         "40.0 default matches RAMEN's own default and keeps "
                         "token count tractable.")
parser.add_argument("--ramen_window_size", type=int, default=256,
                    help="Spatial size RAMEN is built/trained at. BioMassters "
                         "tiles are natively 256x256 (vs Sen1Floods11's 512x512), "
                         "so the default here is the FULL tile -- no tiling "
                         "needed, unlike Sen1Floods11's script. If you set this "
                         "below 256, NOTE: sliding-window inference is NOT "
                         "implemented in trainer_baselines_biomassters.py "
                         "(unlike Sen1Floods11's BaselineTrainer), so a mismatch "
                         "between this and the actual 256x256 eval tiles would "
                         "silently break -- that machinery would need to be "
                         "added first.")
parser.add_argument("--ramen_stride", type=int, default=256,
                    help="Step between windows for sliding-window inference at "
                         "eval time. Kept for parity with Sen1Floods11's script "
                         "and future-proofing, but UNUSED currently -- see the "
                         "--ramen_window_size note above.")
parser.add_argument("--ramen_config", type=str, default=None,
                    help="Optional path to a RAMEN YAML config whose keys "
                         "override the --ramen_* CLI defaults above. Each "
                         "top-level key `foo` in the YAML maps to --ramen_foo "
                         "(e.g. `res: 40` -> args.ramen_res). Explicit CLI "
                         "flags still take precedence over config values for "
                         "keys passed on the command line.")

# Perceiver-IO-specific
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=768)
parser.add_argument("--perceiver_depth",    type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=8)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=2)
parser.add_argument("--no_weight_tie",      action="store_true",
                    help="Disable weight-tying across encoder blocks.")
parser.add_argument("--num_freq_bands",     type=int, default=16)
parser.add_argument("--max_freq",           type=float, default=16.0)
parser.add_argument("--perceiver_attn_dropout", type=float, default=0.0)
parser.add_argument("--perceiver_ff_dropout",   type=float, default=0.0)

# Band-dropout augmentation (train only) -- gives baselines training-time
# exposure to missing modalities/bands, matching the intent of Atomiser's
# own token-dropout augmentation. Applied consistently across ALL T
# timesteps for a given sample (see BioMasstersBaselineDataset's
# _band_dropout_augment docstring for why).
parser.add_argument("--band_dropout", action="store_true", default=True,
                    help="Enable band-dropout augmentation during training "
                         "(default: on). Set the probabilities below to the "
                         "SAME values used on the Atomizer side for a fair "
                         "comparison.")
parser.add_argument("--no_band_dropout", dest="band_dropout", action="store_false",
                    help="Disable band-dropout augmentation (e.g. for an "
                         "ablation isolating its effect).")
parser.add_argument("--p_dropout_applied", type=float, default=0.5,
                    help="Probability a given training sample gets ANY "
                         "band dropout applied (the rest keep all bands).")
parser.add_argument("--p_whole_modality", type=float, default=0.5,
                    help="Given dropout is applied, probability it's a "
                         "whole-modality drop (all S1 or all S2, for the "
                         "entire time series) rather than a random per-band "
                         "subset.")
parser.add_argument("--p_band_drop", type=float, default=0.15,
                    help="Given a per-band (not whole-modality) drop, the "
                         "independent probability each of the 14 bands is "
                         "individually zeroed (same set dropped every "
                         "timestep).")

args = parser.parse_args()

if args.test_only is not None and args.resume_from is not None:
    raise ValueError(
        "--test_only and --resume_from are mutually exclusive: --test_only "
        "skips training entirely (loads weights, runs test), --resume_from "
        "continues training from a checkpoint. Pick one."
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
    import yaml as _yaml
    with open(args.ramen_config, "r") as f:
        _ramen_cfg = _yaml.safe_load(f) or {}

    for _key, _val in _ramen_cfg.items():
        _dest = _RAMEN_CONFIG_KEY_MAP.get(_key)
        if _dest is None:
            print(f"[WARNING] Unrecognized key '{_key}' in {args.ramen_config} "
                  f"-- no matching --ramen_* arg, ignoring. Known keys: "
                  f"{sorted(_RAMEN_CONFIG_KEY_MAP)}")
            continue
        if _dest in _explicit_cli_args:
            print(f"[INFO] '{_key}' in {args.ramen_config} ignored -- "
                  f"--{_dest} was explicitly set on the command line "
                  f"({getattr(args, _dest)}).")
            continue
        print(f"[INFO] {args.ramen_config}: {_key}={_val} -> --{_dest}")
        setattr(args, _dest, _val)

# =============================================================================
# RAMEN SANITY CHECKS
# =============================================================================
if args.model == "ramen":
    if args.ramen_window_size != 256:
        print(f"[WARNING] --ramen_window_size={args.ramen_window_size} != 256 "
              f"(BioMassters' native tile size), but sliding-window inference "
              f"is NOT implemented in trainer_baselines_biomassters.py. Eval "
              f"will feed full 256x256 tiles regardless of this setting, which "
              f"RAMEN will silently resample to its internal effective_size -- "
              f"not a crash, but a train/eval spatial-extent mismatch you "
              f"probably don't want. Leave at 256 unless you've added "
              f"sliding-window support to the trainer first.")
    if args.ramen_stride > args.ramen_window_size:
        print(f"[WARNING] --ramen_stride ({args.ramen_stride}) exceeds "
              f"--ramen_window_size ({args.ramen_window_size}); clamping "
              f"(this arg is currently unused anyway -- see note above).")
        args.ramen_stride = args.ramen_window_size



# =============================================================================
# CONFIG
# =============================================================================

per_frame_channels = NUM_S2_BANDS + NUM_S1_BANDS  # 10 + 4 = 14, always fused (CLP excluded)

temporal_str = f"{args.multi_temporal} frames ({'last' if args.temporal_last else 'uniform'})"

# Both models here accept 5D [B, T, C, H, W] input directly (their own
# internal temporal handling — LTAE or TimeMerge DoubleConv). Unlike PASTIS's
# script, there's no non-temporal channel-stack option here since neither
# baseline you specified (ResNet+double-conv, ViT+LTAE) needs it -- add a
# branch here if a non-temporal baseline is added later.
is_temporal_model = True
model_in_channels = per_frame_channels  # model sees [B, T, C, H, W]

if args.test_only:
    print(f"\n[Train] Test-only mode: {args.test_only}\n")
if args.resume_from:
    print(f"\n[Train] Resume-training mode: {args.resume_from} "
          f"(wait up to {args.resume_wait_seconds}s if not found yet)\n")

print(f"\n{'='*60}")
print(f"  BioMassters Baseline Training")
print(f"  Model:      {args.model}")
if args.model == "ramen":
    print(f"  Sensors:    S2 (optical) + S1 (sar), kept SEPARATE "
          f"({NUM_S2_BANDS} + {NUM_S1_BANDS} bands)")
    print(f"  Window:     {args.ramen_window_size}x{args.ramen_window_size} "
          f"(res={args.ramen_res}m/px, input_res={args.ramen_input_res}m/px)")
else:
    print(f"  Sensors:    S2+S1 fused ({per_frame_channels} bands/frame)")
print(f"  Temporal:   {temporal_str}")
print(f"  Epochs:     {args.epochs}")
print(f"  BS:         {args.batch_size}")
print(f"  LR:         {args.lr}")
print(f"  Grad acc:   {args.grad_accum}")
print(f"  GPUs:       {torch.cuda.device_count()}")
print(f"  Band drop:  {'ON (p_applied=' + str(args.p_dropout_applied) + ', p_whole_mod=' + str(args.p_whole_modality) + ', p_band=' + str(args.p_band_drop) + ')' if args.band_dropout else 'OFF'}")
print(f"{'='*60}\n")


# =============================================================================
# DATASETS
# =============================================================================

common = dict(
    root_path=args.data_dir,
    multi_temporal=args.multi_temporal,
    temporal_last=args.temporal_last,
    temporal_mode="sequence",
)

train_ds = BioMasstersBaselineDataset(
    mode="train", augment=True,
    band_dropout=args.band_dropout,
    p_dropout_applied=args.p_dropout_applied,
    p_whole_modality=args.p_whole_modality,
    p_band_drop=args.p_band_drop,
    **common,
)
val_ds = BioMasstersBaselineDataset(
    mode="validation", augment=False,
    # band_dropout intentionally not passed: the dataset gates it to
    # mode=="train" internally regardless of the constructor default,
    # so val/test are never augmented either way.
    **common,
)
test_ds = BioMasstersBaselineDataset(mode="test", augment=False, **common)

print(f"  Train: {len(train_ds)} chips")
print(f"  Val:   {len(val_ds)} chips")
print(f"  Test:  {len(test_ds)} chips")


# =============================================================================
# COLLATE SELECTION
# =============================================================================

collate_fn = make_ramen_collate() if args.model == "ramen" else make_fused_collate()
if args.model == "ramen":
    print("[BioMassters-BL] RAMEN: modalities kept separate (optical/sar), "
          "channel-before-time layout")
else:
    print("[BioMassters-BL] S2+S1 fusion: concatenating bands in collate")


# =============================================================================
# DATALOADERS
# =============================================================================

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)


# =============================================================================
# MODEL + TRAINER MODULE
# =============================================================================

model = build_model(
    args.model,
    in_channels=model_in_channels,
    args=args,
)

# Target normalization stats (plain z-score) -- SAME file/stats as
# BioMasstersSkipDataset, since BioMasstersBaselineDataset loads the exact
# same normalization_stats.pt. Pulled from the already-constructed train
# dataset for consistency with train_biomassters.py's approach.
_agb_mean = train_ds.norm_stats["agb_mean"].item()
_agb_std  = train_ds.norm_stats["agb_std"].item()
print(f"[BioMassters-BL] AGB target normalization: z-score "
      f"(mean={_agb_mean:.4f}, std={_agb_std:.4f})")

trainer_module = BaselineRegressionTrainer(
    model=model,
    modality="s2",  # fused collate merges S2+S1 into the "s2" key
    temporal=is_temporal_model,
    lr=args.lr,
    weight_decay=args.weight_decay,
    ignore_value=IGNORE_VALUE,
    agb_mean=_agb_mean, agb_std=_agb_std,
)


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if os.environ.get("LOCAL_RANK", "0") == "0" and args.test_only is None:
    try:
        import wandb
        run_name = f"BL_{args.xp_name}_{args.model}"
        wandb_init_kwargs = dict(
            name=run_name,
            project="Atomizer_BioMassters_Baselines",
            config=vars(args),
        )
        wandb.init(**wandb_init_kwargs)
        wandb_logger = WandbLogger(project="Atomizer_BioMassters_Baselines")
    except Exception:
        print("  WandB not available, logging to console only.")


# =============================================================================
# TRAIN (skipped in test-only mode)
# =============================================================================

ckpt_dir = "./checkpoints/biomassters_baselines/"
os.makedirs(ckpt_dir, exist_ok=True)

if args.test_only is None:
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"bl_{args.xp_name}_{args.model}-{{epoch:02d}}-{{val_RMSE:.4f}}",
            monitor="val_RMSE",
            mode="min",
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
            monitor="val_RMSE",
            mode="min",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=-1, num_nodes=num_nodes,
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

    # Resolve --resume_from (with polling) BEFORE fit, so a chained-job
    # wait doesn't happen after the (potentially slow) dataset/model setup
    # above has already run -- consistent with train_biomassters.py's order.
    fit_ckpt_path = None
    if args.resume_from is not None:
        fit_ckpt_path = wait_for_checkpoint(
            args.resume_from, args.resume_wait_seconds, args.resume_poll_interval)

    print(f"\n{'='*60}")
    print(f"  Starting: {args.model}")
    print(f"  Temporal: {temporal_str}")
    print(f"  Train/Val: carved from train_features (10% held out) → Test: test_features")
    if fit_ckpt_path is not None:
        print(f"  RESUMING from: {fit_ckpt_path}")
    print(f"{'='*60}\n")

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=fit_ckpt_path)

    best_ckpt = trainer.checkpoint_callback.best_model_path

    # ─────────────────────────────────────────────────────────────────────
    # Destroy DDP process group BEFORE the test trainer is built.
    # Rank 1+ exit cleanly here; only rank 0 proceeds to test.
    # ─────────────────────────────────────────────────────────────────────
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
    ckpt_to_test = wait_for_checkpoint(
        args.test_only, args.resume_wait_seconds, args.resume_poll_interval)
    best_ckpt = ckpt_to_test
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
    precision="bf16-mixed",
    logger=wandb_logger,
    default_root_dir=ckpt_dir,
)
results = test_trainer.test(trainer_module, test_loader, ckpt_path=best_ckpt)
if results:
    metrics = results[0]
    print(f"RESULT ckpt={best_ckpt} "
          f"test_RMSE={metrics.get('test_RMSE', float('nan')):.6f} "
          f"test_MAE={metrics.get('test_MAE', float('nan')):.6f}")

if wandb_logger:
    import wandb
    wandb.finish()
