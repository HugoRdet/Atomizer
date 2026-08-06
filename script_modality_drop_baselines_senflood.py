"""
Sen1Floods11 Baseline — Modality Drop Inference + FLOPs Script
==============================================================

Load trained baseline checkpoints (UNet / ViT / ResNet / PerceiverIO /
RAMEN) and evaluate on the test split under different band configurations.

For UNet/ViT/ResNet/Perceiver, modality dropping = zeroing the dropped
channel indices in the [B, C, H, W] image tensor before it reaches the
model. No mask/attention mechanism — just channel zeroing.

RAMEN is different in two ways that this script accounts for:
  1. Input format: RAMEN needs {"optical": [B,13,H,W], "sar": [B,2,H,W]}
     rather than one stacked [B,15,H,W] tensor, so dropping happens per
     modality tensor (see RAMENChannelDropWrapper).
  2. Cost: RAMEN tokenizes at the pixel level (no patch embedding), so a
     full 512x512 forward is intractable — sliding_window_inference tiles
     the image the same way script_train_senflood_baseline.py does for
     training/eval.

FLOPs: measured with the SAME harness as Atomizer (torch.profiler,
with_flops=True, sum over key_averages(), averaged over N passes, bs=1, one
discarded warmup), on the BASE model under the 'all' config (no channels
dropped). For RAMEN, the profiled forward is the FULL sliding-window
pass over the 512x512 image (all windows), so the resulting GFLOPs number
is directly comparable to the other baselines' full-image dense forward —
not just the cost of a single window.

Channel layout (fixed, matches Sen1Floods11BaselineDataset):
    indices 0–12  : S2 bands (B01–B12, order = idx field in bands_senflood)
    indices 13–14 : S1 bands (VV=13, VH=14)

Usage
-----
    # Single checkpoint (accuracy + FLOPs)
    python script_test_senflood_baseline_modality_drop.py \
        --ckpt ./checkpoints/bl_perceiver.ckpt \
        --model perceiver \
        --xp_name perceiver_drop_eval

    # Multiple checkpoints, including RAMEN
    python script_test_senflood_baseline_modality_drop.py \
        --ckpts unet=./checkpoints/bl_unet.ckpt ramen=./checkpoints/bl_ramen.ckpt \
        --xp_name baseline_drop_eval \
        --ramen_config training/RAMEN/config_SENFLOOD.yaml \
        --ablations all s2_only s1_only rgb_only no_swir no_re

    # FLOPs only (skip the ablation accuracy loop)
    python script_test_senflood_baseline_modality_drop.py \
        --ckpts resnet=./ckpts/rn.ckpt ramen=./ckpts/ramen.ckpt \
        --ramen_config training/RAMEN/config_SENFLOOD.yaml \
        --xp_name flops_only --flops_only
"""

import os
import argparse
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.sliding_window import sliding_window_inference  # adjust import path
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = Sen1Floods11BaselineDataset.NUM_CLASSES   # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX  # 255
NUM_CHANNELS = Sen1Floods11BaselineDataset.NUM_CHANNELS  # 15
NUM_S2_BANDS = Sen1Floods11BaselineDataset.NUM_S2_BANDS  # 13
NUM_S1_BANDS = Sen1Floods11BaselineDataset.NUM_S1_BANDS  # 2
MODALITY_KEY = "s2s1"

# Fixed channel mapping — matches dataset band order (idx field in yaml)
BAND_TO_CHANNEL = {
    "B01": 0,  "B02": 1,  "B03": 2,  "B04": 3,
    "B05": 4,  "B06": 5,  "B07": 6,  "B08": 7,
    "B08A": 8, "B09": 9,  "B10": 10, "B11": 11, "B12": 12,
    "VV": 13,  "VH": 14,
}

ALL_S2    = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
ALL_S1    = ["VV","VH"]
ALL_BANDS = ALL_S2 + ALL_S1

BUILTIN_ABLATIONS = {
    "all":      [],                                                      # nothing zeroed
    "s2_only":  ALL_S1,                                                  # zero S1
    "s1_only":  ALL_S2,                                                  # zero S2
    "rgb_only": [b for b in ALL_BANDS if b not in ["B02","B03","B04"]],      # keep only RGB
    "no_swir":  ["B10","B11","B12"],
    "no_re":    ["B05","B06","B07","B08A"],
}

def parse_ablation(name: str):
    """Returns list of band names to zero out."""
    if name in BUILTIN_ABLATIONS:
        return BUILTIN_ABLATIONS[name]
    # Inline: "drop=VV,VH"
    for part in name.strip().split():
        if part.startswith("drop="):
            return [b.strip() for b in part[5:].split(",") if b.strip()]
    return []


# =============================================================================
# RAMEN band metadata
# =============================================================================
#
# Mirrors script_train_senflood_baseline.py. IMPORTANT: confirm this band
# order matches how your S2Hand GeoTIFFs are stacked on disk.

S2_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B08A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
    "B12": 2202.4,
}

# RadarProjector's pol_map only has ascending/descending-tagged keys
# (e.g. "asc_vv"); Sen1Floods11 doesn't expose pass direction, so this
# defaults to "asc_*". Keep in sync with the training script.
S1_POLARIZATIONS = {"VV": "asc_vv", "VH": "asc_vh"}

RAMEN_INPUT_BANDS = {
    "optical": ALL_S2,
    "sar": ALL_S1,
}

RAMEN_WAVELENGTHS = {
    "optical": S2_WAVELENGTHS_NM,
    "sar": S1_POLARIZATIONS,
}

# band name -> (modality, index within that modality's tensor). Same
# ordering as BAND_TO_CHANNEL, just re-expressed per-modality instead of
# as one flat 0-14 index (VV/VH's "-13" offset made explicit as index 0/1
# within the "sar" tensor).
RAMEN_BAND_TO_MODALITY_CHANNEL = {
    band: ("optical", i) for i, band in enumerate(ALL_S2)
}
RAMEN_BAND_TO_MODALITY_CHANNEL.update({
    band: ("sar", i) for i, band in enumerate(ALL_S1)
})


# =============================================================================
# CHANNEL-ZEROING WRAPPERS
# =============================================================================

class ChannelDropWrapper(nn.Module):
    """
    Wraps a baseline model and zeros specified input channels before forward.
    """
    def __init__(self, model: nn.Module, drop_channels: list):
        super().__init__()
        self.model         = model
        self.drop_channels = drop_channels

    def forward(self, x, **kwargs):
        if self.drop_channels:
            x = x.clone()
            x[:, self.drop_channels, :, :] = 0.0
        return self.model(x, **kwargs)


class RAMENChannelDropWrapper(nn.Module):
    """
    Wraps a RAMENUPerNet for modality-drop ablations.

    Input is the raw merged tensor under key MODALITY_KEY ("s2s1"):
    {"s2s1": [B, 15, h, w]} — exactly what sliding_window_inference
    crops per-window (it crops generically by dict key, regardless of
    the key's name, so this composes correctly: each window crop still
    arrives here as {"s2s1": [B, 15, window, window]}).

    This wrapper splits that merged tensor into RAMEN's expected
    {"optical": [B,13,h,w], "sar": [B,2,h,w]} on the fly, zeros the
    requested (modality, channel) entries, and forwards to the inner
    RAMENUPerNet. Used for every ablation including "all" (drop_specs=[]),
    so the same code path (and same sliding-window composition) is
    exercised whether or not anything is actually zeroed.
    """
    expects_full_image_dict = True

    def __init__(self, model: nn.Module, drop_specs: list):
        super().__init__()
        self.model = model
        self.drop_specs = drop_specs  # list of (modality, channel_idx)

    def forward(self, x: dict, **kwargs):
        merged = x[MODALITY_KEY]  # [B, 15, h, w]
        optical = merged[:, :NUM_S2_BANDS].clone()
        sar = merged[:, NUM_S2_BANDS: NUM_S2_BANDS + NUM_S1_BANDS].clone()

        for modality, idx in self.drop_specs:
            if modality == "optical":
                optical[:, idx, :, :] = 0.0
            elif modality == "sar":
                sar[:, idx, :, :] = 0.0

        return self.model({"optical": optical, "sar": sar}, **kwargs)


# =============================================================================
# COLLATE
# =============================================================================

def senflood_collate(batch):
    images  = {k: torch.stack([s["image"][k] for s in batch])
               for k in batch[0]["image"]}
    targets  = torch.stack([s["target"]   for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(model_name: str, args) -> nn.Module:
    if model_name == "unet":
        return UNet(
            in_channels=NUM_CHANNELS,
            num_classes=NUM_CLASSES,
            topology=tuple(args.unet_topology),
        )
    elif model_name == "vit":
        return ViTUPerNet(
            in_channels=NUM_CHANNELS,
            num_classes=NUM_CLASSES,
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
            in_channels=NUM_CHANNELS,
            num_classes=NUM_CLASSES,
            decoder_channels=args.vit_decoder_channels,
        )
    elif model_name == "perceiver":
        return PerceiverSeg(
            in_channels=NUM_CHANNELS,
            num_classes=NUM_CLASSES,
            img_size=args.img_size,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            depth=args.depth,
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
        # Built at the small window size, NOT the full 512 img_size —
        # RAMEN tokenizes at the pixel level, full self-attention over
        # 512x512 is intractable. Full-resolution eval goes through
        # sliding_window_inference instead (see below).
        return build_ramen_upernet(
            input_bands=RAMEN_INPUT_BANDS,
            wavelengths=RAMEN_WAVELENGTHS,
            num_classes=NUM_CLASSES,
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
        raise ValueError(f"Unknown model: {model_name}")


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
# FLOPs MEASUREMENT — SAME HARNESS AS ATOMIZER
# =============================================================================

def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b


def stack_image_dict(image_dict, device):
    """
    Reproduce how the trainer feeds single-tensor models: the dataset
    returns image as {modality: [B,C,H,W]}; UNet/ViT/ResNet/Perceiver
    consume a single [B, 15, H, W] tensor with fixed channel order
    (S2 0..12, S1 13..14). NOT used for RAMEN — see the "ramen" branch
    in profile_baselines_flops, which keeps the dict form and routes
    through sliding_window_inference instead.
    """
    if MODALITY_KEY in image_dict:                      # single 's2s1' key
        x = image_dict[MODALITY_KEY]
    elif "s2" in image_dict and "s1" in image_dict:     # separate modalities
        x = torch.cat([image_dict["s2"], image_dict["s1"]], dim=1)   # [B,15,H,W]
    else:
        x = next(iter(image_dict.values()))             # single-key fallback
    return x.to(device)


@torch.no_grad()
def measure_gflops_forward(forward_fn, batches, device, n_warmup=1):
    """
    Identical methodology to the Atomizer measurement:
      - one warmup pass discarded
      - each measured pass profiled separately with with_flops=True
      - per-pass total = sum(evt.flops for evt in key_averages() if evt.flops)
      - report mean / 1e9  (LOWER BOUND: profiler-counted ops only)

    For RAMEN, forward_fn internally loops over sliding-window tiles, so
    the profiler context here captures the SUM of FLOPs across all
    windows composing one full image — the number reported is the true
    total cost per full-image forward, directly comparable to the other
    baselines' single dense forward pass.
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


def profile_baselines_flops(model_ckpts, args, test_loader,
                            n_profile=2, n_warmup=1):
    """
    Returns {model_name: gflops}. Reuses the SAME batches for every model so the
    input geometry is identical. Measures the BASE model, 'all' config (no drop).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build the SAME batch list once, reuse for every model
    raw = []
    for i, b in enumerate(test_loader):
        raw.append(_to_device(b, device))
        if len(raw) >= n_profile + n_warmup:
            break
    if not raw:
        print("[FLOPs] No test batches; skipping FLOPs.")
        return {}

    gflops = {}
    for model_name, ckpt_path in model_ckpts:
        base_model = build_model(model_name, args)

        # load real weights via the trainer, then extract the bare model
        load_kwargs = dict(
            strict=True, model=base_model, temporal=False,
            task="senflood", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
        )
        if model_name == "ramen":
            load_kwargs.update(
                modality="optical+sar",
                window_size=args.ramen_window_size,
                window_stride=args.ramen_stride,
            )
        else:
            load_kwargs.update(modality=MODALITY_KEY)

        try:
            tm = BaselineTrainer.load_from_checkpoint(ckpt_path, **load_kwargs)
            base_model = tm.model
        except Exception as e:
            print(f"[FLOPs][{model_name}] weight-load note: {e} "
                  f"(profiling randomly-initialized weights; FLOPs still valid "
                  f"since count is shape-driven, but load if you can)")
        base_model = base_model.to(device).eval()

        if model_name == "ramen":
            wrapped = RAMENChannelDropWrapper(base_model, []).to(device).eval()

            def fwd(b, m=wrapped):
                # b["image"] is already {"s2s1": [1, 15, 512, 512]} — the
                # wrapper splits/forwards per window, sliding_window_inference
                # handles the tiling and stitching.
                return sliding_window_inference(
                    m, b["image"],
                    window_size=args.ramen_window_size,
                    stride=args.ramen_stride,
                    num_classes=NUM_CLASSES,
                )
        else:
            def fwd(b, m=base_model):
                x = stack_image_dict(b["image"], device)      # [1, 15, 512, 512]
                return m(x)

        g = measure_gflops_forward(fwd, raw, device, n_warmup=n_warmup)
        gflops[model_name] = g
        print(f"[FLOPs] {model_name:<12} = {g:.1f} GFLOPs/forward "
              f"(bs=1, 15x512x512, torch.profiler, mean of {len(raw)-n_warmup})")

        del base_model
        if device == "cuda":
            torch.cuda.empty_cache()

    return gflops


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()

# Single-checkpoint mode
parser.add_argument("--ckpt",  type=str, default=None,
                    help="Path to a single checkpoint")
parser.add_argument("--model", type=str, default="resnet",
                    choices=["unet","vit","resnet","perceiver","ramen"],
                    help="Architecture for --ckpt")

# Multi-checkpoint mode: name=path pairs
parser.add_argument("--ckpts", type=str, nargs="+", default=None,
                    help="name=path pairs, e.g. unet=./ckpts/unet.ckpt ramen=./ckpts/ramen.ckpt")

parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--data_dir",   type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers",type=int, default=4)
parser.add_argument("--ablations",  type=str, nargs="+",
                    default=["all","s2_only","s1_only","rgb_only","no_swir","no_re"])
parser.add_argument("--wandb",      action="store_true")

# FLOPs controls
parser.add_argument("--flops", action="store_true", default=True,
                    help="Measure GFLOPs/forward per model (default: on)")
parser.add_argument("--no_flops", dest="flops", action="store_false",
                    help="Disable FLOPs measurement")
parser.add_argument("--flops_only", action="store_true",
                    help="Only measure FLOPs; skip the ablation accuracy loop")
parser.add_argument("--flops_n", type=int, default=2,
                    help="Number of profiled forward passes (mean)")

# Shared Architecture args
parser.add_argument("--img_size",             type=int, default=512)

# UNet args
parser.add_argument("--unet_topology",        type=int, nargs="+", default=[64,128,256,512,1024])

# ViT / ResNet args
parser.add_argument("--vit_embed_dim",        type=int, default=384)
parser.add_argument("--vit_depth",            type=int, default=12)
parser.add_argument("--vit_num_heads",        type=int, default=6)
parser.add_argument("--vit_patch_size",       type=int, default=16)
parser.add_argument("--vit_output_layers",    type=int, nargs="+", default=[2,5,8,11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)
parser.add_argument("--resnet_variant",       type=str, default="resnet50")

# Perceiver IO args
parser.add_argument("--num_latents",        type=int, default=512)
parser.add_argument("--latent_dim",         type=int, default=768)
parser.add_argument("--depth",              type=int, default=1)
parser.add_argument("--cross_heads",        type=int, default=8)
parser.add_argument("--latent_heads",       type=int, default=8)
parser.add_argument("--cross_dim_head",     type=int, default=64)
parser.add_argument("--latent_dim_head",    type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=2)
parser.add_argument("--no_weight_tie",      action="store_true",
                    help="Disable weight-tying across encoder blocks.")
parser.add_argument("--num_freq_bands",     type=int, default=16)
parser.add_argument("--max_freq",           type=float, default=16.0)
parser.add_argument("--attn_dropout",       type=float, default=0.0)
parser.add_argument("--ff_dropout",         type=float, default=0.0)

# RAMEN args (mirrors script_train_senflood_baseline.py)
parser.add_argument("--ramen_embed_dim",   type=int, default=384)
parser.add_argument("--ramen_depth",       type=int, default=12)
parser.add_argument("--ramen_num_heads",   type=int, default=8)
parser.add_argument("--ramen_input_res",   type=float, default=10.0)
parser.add_argument("--ramen_res",         type=float, default=20.0)
parser.add_argument("--ramen_window_size", type=int, default=128,
                    help="Spatial size RAMEN was built/trained at. MUST "
                         "match the checkpoint being loaded.")
parser.add_argument("--ramen_stride",      type=int, default=96,
                    help="Sliding-window stride for full 512x512 eval.")
parser.add_argument("--ramen_config",      type=str, default=None,
                    help="Optional YAML overriding --ramen_* args, same "
                         "format as script_train_senflood_baseline.py's "
                         "--ramen_config. Use the SAME config the "
                         "checkpoint was trained with.")

args = parser.parse_args()
args = apply_ramen_config(args)

# Build (model_name, ckpt_path) list
if args.ckpts:
    model_ckpts = []
    for item in args.ckpts:
        name, path = item.split("=", 1)
        model_ckpts.append((name, path))
elif args.ckpt:
    model_ckpts = [(args.model, args.ckpt)]
else:
    raise ValueError("Provide --ckpt or --ckpts")


# =============================================================================
# TEST DATASET
# =============================================================================

test_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers,
    collate_fn=senflood_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

print(f"[Eval] Test set: {len(test_ds)} samples")


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb.init(
        name=f"{args.xp_name}_baseline_drop",
        project="SenFlood",
        config={"ckpts": str(model_ckpts), "ablations": args.ablations},
    )
    wandb_logger = WandbLogger(project="SenFlood")


# =============================================================================
# FLOPs (measured FIRST, on the base models, same harness as Atomizer)
# =============================================================================

flops_table = {}
if args.flops:
    print(f"\n{'='*60}")
    print(f"  BASELINE FLOPs (same harness as Atomizer)")
    print(f"{'='*60}")
    flops_table = profile_baselines_flops(
        model_ckpts, args, test_loader, n_profile=args.flops_n)
    print(f"\n[FLOPs] Summary (GFLOPs/forward, bs=1, 15x512x512):")
    for name, g in flops_table.items():
        print(f"    {name:<12} {g:.1f}")
    print(f"[FLOPs] NOTE: baselines are FULL dense decode. RAMEN's number is "
          f"the full sliding-window pass (all tiles), so it's directly "
          f"comparable. Compare against the Atomizer-quadtree number, and "
          f"state in the caption that patch-grid baselines cannot use "
          f"coordinate-native adaptive decode.")


# =============================================================================
# RUN — ablation accuracy loop (skipped if --flops_only)
# =============================================================================

all_results = {}

if not args.flops_only:
    for model_name, ckpt_path in model_ckpts:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}   Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # ── Load base model weights ───────────────────────────────────────
        base_model = build_model(model_name, args)

        load_kwargs = dict(
            strict=True, model=base_model, temporal=False,
            task="senflood", num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
        )
        if model_name == "ramen":
            load_kwargs.update(
                modality="optical+sar",
                window_size=args.ramen_window_size,
                window_stride=args.ramen_stride,
            )
        else:
            load_kwargs.update(modality=MODALITY_KEY)

        trainer_module = BaselineTrainer.load_from_checkpoint(ckpt_path, **load_kwargs)
        trainer_module.eval()

        all_results[model_name] = {}

        for ablation_name in args.ablations:
            drop_bands = parse_ablation(ablation_name)
            drop_str   = ",".join(drop_bands) if drop_bands else "none"

            print(f"\n  {'─'*50}")
            print(f"  Ablation : {ablation_name}   Drop : {drop_str}")
            print(f"  {'─'*50}")

            if model_name == "ramen":
                drop_specs = [RAMEN_BAND_TO_MODALITY_CHANNEL[b] for b in drop_bands]
                trainer_module.model = RAMENChannelDropWrapper(base_model, drop_specs)
            else:
                drop_channels = [BAND_TO_CHANNEL[b] for b in drop_bands]
                trainer_module.model = ChannelDropWrapper(base_model, drop_channels)

            trainer = Trainer(
                devices=-1,
                accelerator="gpu",
                precision="bf16-mixed",
                logger=wandb_logger,
                enable_progress_bar=True,
                enable_model_summary=False,
            )

            results     = trainer.test(trainer_module, test_loader, verbose=True)
            metrics     = results[0] if results else {}
            all_results[model_name][ablation_name] = metrics

            if args.wandb and wandb_logger:
                import wandb
                wandb.log({
                    f"{model_name}/{ablation_name}/{k}": v
                    for k, v in metrics.items()
                })

        trainer_module.model = base_model


# =============================================================================
# SUMMARY TABLE
# =============================================================================

if not args.flops_only and all_results:
    print(f"\n\n{'='*80}")
    print(f"  BASELINE MODALITY DROP SUMMARY — {args.xp_name}")
    print(f"{'='*80}")

    sample_metrics = next(
        m for res in all_results.values() for m in res.values() if m
    )
    metric_keys = list(sample_metrics.keys())

    for mkey in metric_keys:
        print(f"\n  Metric: {mkey}")
        header = f"{'Model':<14}" + "".join(f"  {a:<12}" for a in args.ablations)
        print(f"  {header}")
        print(f"  {'─' * len(header)}")
        for model_name in all_results:
            row = f"{model_name:<14}"
            for abl in args.ablations:
                v = all_results[model_name].get(abl, {}).get(mkey, float("nan"))
                row += f"  {v:<12.4f}"
            print(f"  {row}")

    print(f"\n\n  Flat table (Ablation × Model):")
    print(f"  {'Ablation':<14} {'Drop':<40}", end="")
    for model_name in all_results:
        print(f"  {model_name:<14}", end="")
    print()
    print(f"  {'─'*80}")
    for abl in args.ablations:
        drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
        print(f"  {abl:<14} {drop_str:<40}", end="")
        for model_name in all_results:
            v = all_results[model_name].get(abl, {}).get("test_mIoU", float("nan"))
            print(f"  {v:<14.4f}", end="")
        print()

    print(f"\n{'='*80}\n")


# =============================================================================
# WRITE RESULTS (accuracy + FLOPs)
# =============================================================================

out_path = f"./results_{args.xp_name}_baseline_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Experiment: {args.xp_name}\n")
    f.write(f"Checkpoints: {model_ckpts}\n\n")

    if flops_table:
        f.write("FLOPs (GFLOPs/forward, bs=1, 15x512x512, torch.profiler, "
                "same harness as Atomizer; FULL dense decode / full "
                "sliding-window pass for RAMEN):\n")
        for name, g in flops_table.items():
            f.write(f"  {name:<12} {g:.1f}\n")
        f.write("\n")

    if not args.flops_only and all_results:
        f.write(f"{'Ablation':<14} {'Drop':<40}")
        for model_name in all_results:
            f.write(f"  {model_name:<14}")
        f.write("\n" + "─"*80 + "\n")
        for abl in args.ablations:
            drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
            f.write(f"{abl:<14} {drop_str:<40}")
            for model_name in all_results:
                v = all_results[model_name].get(abl, {}).get("test_mIoU", float("nan"))
                f.write(f"  {v:<14.4f}")
            f.write("\n")

print(f"[Eval] Results saved to {out_path}")

if args.wandb:
    import wandb
    wandb.finish()
