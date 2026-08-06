"""
BioMassters Baseline — Modality Drop Inference + FLOPs Script
================================================================

Load trained baseline checkpoints (ResNet+MT / ViT+LTAE / Perceiver-IO /
RAMEN) and evaluate on the test split under different band configurations.

Adapted from script_test_senflood_baseline_modality_drop.py. Key differences,
since BioMassters is multi-temporal regression rather than single-frame
segmentation:

  1. TENSOR RANK: the fused collate produces [B, T, C, H, W] (5D), not
     Sen1Floods11's single-frame [B, C, H, W] (4D) -- ChannelDropWrapper
     zeros along dim=2 (channel), not dim=1.

  2. RAMEN LAYOUT: BioMasstersBaselineDataset's RAMEN collate produces
     per-modality tensors as [B, C, T, H, W] (channel-before-time, per
     RAMENBackbone's expected input rank) -- RAMENChannelDropWrapper zeros
     along dim=1, not the merged-then-split pattern Sen1Floods11 used.

  3. NO SLIDING WINDOW: BioMassters tiles are natively 256x256, and
     --ramen_window_size defaults to the full tile (see
     script_train_biomassters_baseline.py) -- no tiling/stitching needed,
     unlike Sen1Floods11's 512x512 requiring RAMEN windowing.

  4. REGRESSION METRICS: test_RMSE / test_MAE (Mg/ha, lower is better),
     via BaselineRegressionTrainer, not test_mIoU/test_accuracy.
     IGNORE_VALUE = -1.0 (AGB never negative), not ignore_index=255.

  5. BAND SET: BioMassters' S2 is 10 physical bands (B02-B12, CLP
     excluded -- no B01/B09/B10 the way Sen1Floods11 has); S1 is 4 bands
     (VV/VH x ascending/descending), not Sen1Floods11's plain VV/VH.

Channel layout (fixed, matches BioMasstersBaselineDataset / the fused
collate's channel-concat order):
    indices 0-9   : S2 bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
    indices 10-13 : S1 bands (VV_asc, VH_asc, VV_desc, VH_desc)

Usage
-----
    # Single checkpoint (accuracy + FLOPs)
    python script_test_biomassters_baseline_modality_drop.py \
        --ckpt ./checkpoints/biomassters_baselines/bl_resnet.ckpt \
        --model resnet_upernet_mt \
        --xp_name resnet_drop_eval

    # Multiple checkpoints, including RAMEN
    python script_test_biomassters_baseline_modality_drop.py \
        --ckpts resnet_upernet_mt=./ckpts/bl_resnet.ckpt ramen=./ckpts/bl_ramen.ckpt \
        --xp_name baseline_drop_eval \
        --ablations all s2_only s1_only rgb_only no_swir no_re

    # FLOPs only (skip the ablation accuracy loop)
    python script_test_biomassters_baseline_modality_drop.py \
        --ckpts vit_ltae=./ckpts/bl_vit_ltae.ckpt ramen=./ckpts/bl_ramen.ckpt \
        --xp_name flops_only --flops_only
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
from pytorch_lightning import Trainer, seed_everything

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
# CONSTANTS
# =============================================================================

MODALITY_KEY = "s2"  # fused collate keys the concatenated S2+S1 tensor under "s2"

# Fixed channel mapping -- matches BioMasstersBaselineDataset's fused order
BAND_TO_CHANNEL = {
    "B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
    "B07": 5, "B08": 6, "B8A": 7, "B11": 8, "B12": 9,
    "VV_asc": 10, "VH_asc": 11, "VV_desc": 12, "VH_desc": 13,
}

ALL_S2    = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ALL_S1    = ["VV_asc", "VH_asc", "VV_desc", "VH_desc"]
ALL_BANDS = ALL_S2 + ALL_S1

BUILTIN_ABLATIONS = {
    "all":      [],                                                    # nothing zeroed
    "s2_only":  ALL_S1,                                                 # zero S1
    "s1_only":  ALL_S2,                                                 # zero S2
    "rgb_only": [b for b in ALL_BANDS if b not in ["B02", "B03", "B04"]],  # keep only RGB
    "no_swir":  ["B11", "B12"],
    "no_re":    ["B05", "B06", "B07", "B8A"],
}


def parse_ablation(name: str):
    """Returns list of band names to zero out."""
    if name in BUILTIN_ABLATIONS:
        return BUILTIN_ABLATIONS[name]
    # Inline: "drop=VV_asc,VH_asc"
    for part in name.strip().split():
        if part.startswith("drop="):
            return [b.strip() for b in part[5:].split(",") if b.strip()]
    return []


# =============================================================================
# RAMEN band metadata (mirrors script_train_biomassters_baseline.py exactly)
# =============================================================================

RAMEN_S2_WAVELENGTHS = {
    "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}
RAMEN_S1_POLARIZATIONS = {
    "VV_asc": "asc_vv", "VH_asc": "asc_vh",
    "VV_desc": "des_vv", "VH_desc": "des_vh",
}
RAMEN_INPUT_BANDS = {"optical": ALL_S2, "sar": ALL_S1}
RAMEN_WAVELENGTHS = {"optical": RAMEN_S2_WAVELENGTHS, "sar": RAMEN_S1_POLARIZATIONS}

# band name -> (modality, index within that modality's tensor)
RAMEN_BAND_TO_MODALITY_CHANNEL = {band: ("optical", i) for i, band in enumerate(ALL_S2)}
RAMEN_BAND_TO_MODALITY_CHANNEL.update({band: ("sar", i) for i, band in enumerate(ALL_S1)})


# =============================================================================
# CHANNEL-ZEROING WRAPPERS
# =============================================================================

class ChannelDropWrapper(nn.Module):
    """
    Wraps a baseline model and zeros specified input channels before forward.

    Input is [B, T, C, H, W] (5D, multi-temporal) -- channel dim is 2, NOT
    1 as in Sen1Floods11's single-frame [B, C, H, W] version. Drops are
    applied to EVERY timestep (consistent with the training-time band-
    dropout augmentation's convention: a missing band/sensor stays missing
    for the whole time series).
    """
    def __init__(self, model: nn.Module, drop_channels: list):
        super().__init__()
        self.model = model
        self.drop_channels = drop_channels

    def forward(self, x, **kwargs):
        if self.drop_channels:
            x = x.clone()
            x[:, :, self.drop_channels, :, :] = 0.0
        return self.model(x, **kwargs)


class RAMENChannelDropWrapper(nn.Module):
    """
    Wraps a RAMENUPerNet for modality-drop ablations.

    Input is {"optical": [B,10,T,H,W], "sar": [B,4,T,H,W]} -- channel dim
    is 1 (RAMENBackbone's channel-before-time layout), applied to every
    timestep by construction (zeroing dim=1 zeros that channel across all
    of dim=2's T slices automatically, no separate loop needed).

    temporal_kwarg = "dates": BaselineRegressionTrainer.forward() does
    `getattr(self.model, "temporal_kwarg", "doy")` to pick the right kwarg
    name for the wrapped model's forward(). During the ablation loop,
    trainer_module.model gets REASSIGNED to an instance of THIS wrapper
    class (see the ablation loop below), so the attribute lookup happens
    on the wrapper, not the underlying RAMENUPerNet it wraps -- without
    this attribute here, that lookup falls back to the "doy" default and
    RAMENUPerNet.forward() (which only accepts `dates`) raises a
    TypeError. Must match build_model()'s `model.temporal_kwarg = "dates"`
    for the unwrapped case.
    """
    expects_full_image_dict = True
    temporal_kwarg = "dates"

    def __init__(self, model: nn.Module, drop_specs: list):
        super().__init__()
        self.model = model
        self.drop_specs = drop_specs  # list of (modality, channel_idx)

    def forward(self, x: dict, **kwargs):
        optical = x["optical"].clone()
        sar = x["sar"].clone()

        for modality, idx in self.drop_specs:
            if modality == "optical":
                optical[:, idx] = 0.0
            elif modality == "sar":
                sar[:, idx] = 0.0

        return self.model({"optical": optical, "sar": sar}, **kwargs)


# =============================================================================
# COLLATE (mirrors script_train_biomassters_baseline.py's two collate paths)
# =============================================================================

def biomassters_collate(batch):
    images = {}
    dates = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
        dates[key] = torch.stack([s["dates"][key] for s in batch])
    targets = torch.stack([s["target"] for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "dates": dates, "target": targets, "metadata": metadata}


def make_fused_collate():
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
    def ramen_collate(batch):
        out = biomassters_collate(batch)
        s2 = out["image"]["s2"]  # [B, T, 10, H, W]
        s1 = out["image"]["s1"]  # [B, T,  4, H, W]
        out["image"] = {
            "optical": s2.permute(0, 2, 1, 3, 4).contiguous(),  # [B, 10, T, H, W]
            "sar":     s1.permute(0, 2, 1, 3, 4).contiguous(),  # [B, 4,  T, H, W]
        }
        out["dates"] = {"optical": out["dates"]["s2"], "sar": out["dates"]["s1"]}
        return out
    return ramen_collate


# =============================================================================
# MODEL BUILDER (mirrors script_train_biomassters_baseline.py's build_model)
# =============================================================================

def build_model(model_name: str, args) -> nn.Module:
    in_channels = NUM_S2_BANDS + NUM_S1_BANDS  # 14, always fused for non-RAMEN models

    if model_name == "vit_ltae":
        return ViTLTAEUPerNet(
            in_channels=in_channels, num_classes=1, img_size=args.img_size,
            embed_dim=args.vit_embed_dim, depth=args.vit_depth,
            num_heads=args.vit_num_heads, patch_size=args.vit_patch_size,
            output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
            ltae_n_head=args.n_heads, ltae_d_k=args.d_k, ltae_d_model=args.d_model,
        )
    elif model_name == "resnet_upernet_mt":
        return build_resnet_upernet_mt(
            variant=args.resnet_variant, in_channels=in_channels, num_classes=1,
            num_frames=args.multi_temporal, decoder_channels=args.vit_decoder_channels,
        )
    elif model_name == "ramen":
        model = build_ramen_upernet(
            input_bands=RAMEN_INPUT_BANDS, wavelengths=RAMEN_WAVELENGTHS,
            num_classes=1, input_size=args.ramen_window_size,
            embed_dim=args.ramen_embed_dim, depth=args.ramen_depth,
            num_heads=args.ramen_num_heads, input_res=args.ramen_input_res,
            res=args.ramen_res, output_layers=tuple(args.vit_output_layers),
            decoder_channels=args.vit_decoder_channels,
        )
        model.temporal_kwarg = "dates"
        return model
    elif model_name == "perceiver":
        return PerceiverSeg(
            in_channels=in_channels, num_classes=1, img_size=args.img_size,
            num_latents=args.num_latents, latent_dim=args.latent_dim,
            depth=args.perceiver_depth, cross_heads=args.cross_heads,
            latent_heads=args.latent_heads, cross_dim_head=args.cross_dim_head,
            latent_dim_head=args.latent_dim_head,
            self_per_cross_attn=args.self_per_cross_attn,
            weight_tie_layers=(not args.no_weight_tie),
            num_freq_bands=args.num_freq_bands, max_freq=args.max_freq,
            attn_dropout=args.perceiver_attn_dropout, ff_dropout=args.perceiver_ff_dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# FLOPs MEASUREMENT -- same harness as Atomizer / the Sen1Floods11 version
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
        total = sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", None))
        flops_list.append(total)

    if not flops_list:
        return float("nan")
    return (sum(flops_list) / len(flops_list)) / 1e9


def profile_baselines_flops(model_ckpts, args, ramen_loader, fused_loader, n_profile=2, n_warmup=1):
    """
    Returns {model_name: gflops}. RAMEN uses ramen_loader (separate
    optical/sar tensors), everything else uses fused_loader (single fused
    tensor) -- SAME underlying chips/batches either way, just different
    collate, so input geometry is comparable across models.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_fused, raw_ramen = [], []
    for b in fused_loader:
        raw_fused.append(_to_device(b, device))
        if len(raw_fused) >= n_profile + n_warmup:
            break
    for b in ramen_loader:
        raw_ramen.append(_to_device(b, device))
        if len(raw_ramen) >= n_profile + n_warmup:
            break

    gflops = {}
    for model_name, ckpt_path in model_ckpts:
        base_model = build_model(model_name, args)

        load_kwargs = dict(strict=True, model=base_model, ignore_value=IGNORE_VALUE)
        if model_name == "ramen":
            load_kwargs.update(temporal=True, modality="optical+sar")
        else:
            load_kwargs.update(temporal=True, modality=MODALITY_KEY)

        try:
            tm = BaselineRegressionTrainer.load_from_checkpoint(ckpt_path, **load_kwargs)
            base_model = tm.model
        except Exception as e:
            print(f"[FLOPs][{model_name}] weight-load note: {e} "
                  f"(profiling randomly-initialized weights; FLOPs still valid "
                  f"since count is shape-driven, but load if you can)")
        base_model = base_model.to(device).eval()

        if model_name == "ramen":
            wrapped = RAMENChannelDropWrapper(base_model, []).to(device).eval()

            def fwd(b, m=wrapped):
                return m(b["image"], dates=b["dates"])
            raw = raw_ramen
        else:
            wrapped = ChannelDropWrapper(base_model, []).to(device).eval()

            def fwd(b, m=wrapped):
                doy = b["dates"][MODALITY_KEY]
                return m(b["image"][MODALITY_KEY], doy=doy)
            raw = raw_fused

        if not raw:
            print(f"[FLOPs][{model_name}] no batches available; skipping.")
            continue

        g = measure_gflops_forward(fwd, raw, device, n_warmup=n_warmup)
        gflops[model_name] = g
        n_bands = NUM_S2_BANDS + NUM_S1_BANDS
        print(f"[FLOPs] {model_name:<16} = {g:.1f} GFLOPs/forward "
              f"(bs=1, {n_bands}x{args.multi_temporal}x{args.img_size}x{args.img_size}, "
              f"mean of {len(raw) - n_warmup})")

        del base_model
        if device == "cuda":
            torch.cuda.empty_cache()

    return gflops


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument("--ckpt",  type=str, default=None)
parser.add_argument("--model", type=str, default="resnet_upernet_mt",
                    choices=["resnet_upernet_mt", "vit_ltae", "ramen", "perceiver"])
parser.add_argument("--ckpts", type=str, nargs="+", default=None,
                    help="name=path pairs, e.g. resnet_upernet_mt=./ckpts/bl_resnet.ckpt "
                         "ramen=./ckpts/bl_ramen.ckpt")

parser.add_argument("--xp_name",     type=str, required=True)
parser.add_argument("--data_dir",    type=str, default="./data/biomassters")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ablations",   type=str, nargs="+",
                    default=["all", "s2_only", "s1_only", "rgb_only", "no_swir", "no_re"])
parser.add_argument("--wandb",       action="store_true")

# Temporal config -- MUST match what the checkpoint was trained with
parser.add_argument("--multi_temporal", type=int, default=3)
parser.add_argument("--temporal_last",  action="store_true", default=True)

# FLOPs controls
parser.add_argument("--flops", action="store_true", default=True)
parser.add_argument("--no_flops", dest="flops", action="store_false")
parser.add_argument("--flops_only", action="store_true")
parser.add_argument("--flops_n", type=int, default=2)

# Shared architecture args
parser.add_argument("--img_size", type=int, default=256)

# ViT / ResNet / LTAE args
parser.add_argument("--vit_embed_dim",        type=int, default=384)
parser.add_argument("--vit_depth",            type=int, default=12)
parser.add_argument("--vit_num_heads",        type=int, default=6)
parser.add_argument("--vit_patch_size",       type=int, default=16)
parser.add_argument("--vit_output_layers",    type=int, nargs="+", default=[2, 5, 8, 11])
parser.add_argument("--vit_decoder_channels", type=int, default=256)
parser.add_argument("--resnet_variant",       type=str, default="resnet50")
parser.add_argument("--n_heads",              type=int, default=16)
parser.add_argument("--d_k",                  type=int, default=4)
parser.add_argument("--d_model",              type=int, default=256)

# Perceiver-IO args
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

# RAMEN args
parser.add_argument("--ramen_embed_dim",   type=int, default=384)
parser.add_argument("--ramen_depth",       type=int, default=12)
parser.add_argument("--ramen_num_heads",   type=int, default=8)
parser.add_argument("--ramen_input_res",   type=float, default=10.0)
parser.add_argument("--ramen_res",         type=float, default=40.0)
parser.add_argument("--ramen_window_size", type=int, default=256,
                    help="MUST match the checkpoint being loaded.")

args = parser.parse_args()

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
# TEST DATASETS + LOADERS (one dataset instance, two collates)
# =============================================================================

test_ds = BioMasstersBaselineDataset(
    root_path=args.data_dir, mode="test",
    multi_temporal=args.multi_temporal, temporal_last=args.temporal_last,
    temporal_mode="sequence", augment=False, band_dropout=False,
)

fused_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
    collate_fn=make_fused_collate(), pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)
ramen_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
    collate_fn=make_ramen_collate(), pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)

print(f"[Eval] Test set: {len(test_ds)} chips")


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb.init(
        name=f"{args.xp_name}_baseline_drop",
        project="BioMassters",
        config={"ckpts": str(model_ckpts), "ablations": args.ablations},
    )
    wandb_logger = WandbLogger(project="BioMassters")


# =============================================================================
# FLOPs
# =============================================================================

flops_table = {}
if args.flops:
    print(f"\n{'='*60}")
    print(f"  BASELINE FLOPs (same harness as Atomizer)")
    print(f"{'='*60}")
    flops_table = profile_baselines_flops(
        model_ckpts, args, ramen_loader, fused_loader, n_profile=args.flops_n)
    n_bands = NUM_S2_BANDS + NUM_S1_BANDS
    print(f"\n[FLOPs] Summary (GFLOPs/forward, bs=1, "
          f"{n_bands}x{args.multi_temporal}x{args.img_size}x{args.img_size}):")
    for name, g in flops_table.items():
        print(f"    {name:<16} {g:.1f}")


# =============================================================================
# RUN -- ablation accuracy loop
# =============================================================================

all_results = {}

if not args.flops_only:
    for model_name, ckpt_path in model_ckpts:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}   Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        base_model = build_model(model_name, args)

        load_kwargs = dict(strict=True, model=base_model, ignore_value=IGNORE_VALUE, temporal=True)
        if model_name == "ramen":
            load_kwargs.update(modality="optical+sar")
            loader = ramen_loader
        else:
            load_kwargs.update(modality=MODALITY_KEY)
            loader = fused_loader

        trainer_module = BaselineRegressionTrainer.load_from_checkpoint(ckpt_path, **load_kwargs)
        trainer_module.eval()

        all_results[model_name] = {}

        for ablation_name in args.ablations:
            drop_bands = parse_ablation(ablation_name)
            drop_str = ",".join(drop_bands) if drop_bands else "none"

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
                devices=-1, accelerator="gpu", precision="bf16-mixed",
                logger=wandb_logger, enable_progress_bar=True, enable_model_summary=False,
            )

            results = trainer.test(trainer_module, loader, verbose=True)
            metrics = results[0] if results else {}
            all_results[model_name][ablation_name] = metrics

            if args.wandb and wandb_logger:
                import wandb
                wandb.log({f"{model_name}/{ablation_name}/{k}": v for k, v in metrics.items()})

        trainer_module.model = base_model


# =============================================================================
# SUMMARY TABLE
# =============================================================================

if not args.flops_only and all_results:
    print(f"\n\n{'='*80}")
    print(f"  BASELINE MODALITY DROP SUMMARY — {args.xp_name}")
    print(f"{'='*80}")

    sample_metrics = next(m for res in all_results.values() for m in res.values() if m)
    metric_keys = list(sample_metrics.keys())

    for mkey in metric_keys:
        print(f"\n  Metric: {mkey}")
        header = f"{'Model':<18}" + "".join(f"  {a:<12}" for a in args.ablations)
        print(f"  {header}")
        print(f"  {'─' * len(header)}")
        for model_name in all_results:
            row = f"{model_name:<18}"
            for abl in args.ablations:
                v = all_results[model_name].get(abl, {}).get(mkey, float("nan"))
                row += f"  {v:<12.4f}"
            print(f"  {row}")

    print(f"\n\n  Flat table (Ablation × Model, RMSE):")
    print(f"  {'Ablation':<14} {'Drop':<40}", end="")
    for model_name in all_results:
        print(f"  {model_name:<18}", end="")
    print()
    print(f"  {'─'*100}")
    for abl in args.ablations:
        drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
        print(f"  {abl:<14} {drop_str:<40}", end="")
        for model_name in all_results:
            v = all_results[model_name].get(abl, {}).get("test_RMSE", float("nan"))
            print(f"  {v:<18.4f}", end="")
        print()

    print(f"\n{'='*80}\n")


# =============================================================================
# WRITE RESULTS
# =============================================================================

out_path = f"./results_{args.xp_name}_baseline_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Experiment: {args.xp_name}\n")
    f.write(f"Checkpoints: {model_ckpts}\n\n")

    if flops_table:
        n_bands = NUM_S2_BANDS + NUM_S1_BANDS
        f.write(f"FLOPs (GFLOPs/forward, bs=1, "
                f"{n_bands}x{args.multi_temporal}x{args.img_size}x{args.img_size}, "
                f"torch.profiler, same harness as Atomizer):\n")
        for name, g in flops_table.items():
            f.write(f"  {name:<16} {g:.1f}\n")
        f.write("\n")

    if not args.flops_only and all_results:
        f.write(f"{'Ablation':<14} {'Drop':<40}")
        for model_name in all_results:
            f.write(f"  {model_name:<18}")
        f.write("\n" + "─"*100 + "\n")
        for abl in args.ablations:
            drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
            f.write(f"{abl:<14} {drop_str:<40}")
            for model_name in all_results:
                v = all_results[model_name].get(abl, {}).get("test_RMSE", float("nan"))
                f.write(f"  {v:<18.4f}")
            f.write("\n")

print(f"[Eval] Results saved to {out_path}")

if args.wandb:
    import wandb
    wandb.finish()
