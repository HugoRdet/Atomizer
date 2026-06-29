"""
Sen1Floods11 Baseline — Modality Drop Inference Script
=======================================================

Load trained baseline checkpoints (UNet / ViT / ResNet / PerceiverIO) and
evaluate on the test split under different band configurations.

For baselines, modality dropping = zeroing the dropped channel indices
in the [B, C, H, W] image tensor before it reaches the model.
No mask/attention mechanism — just channel zeroing.

Channel layout (fixed, matches Sen1Floods11BaselineDataset):
    indices 0–12  : S2 bands (B01–B12, order = idx field in bands_senflood)
    indices 13–14 : S1 bands (VV=13, VH=14)

Usage
-----
    # Single checkpoint
    python script_test_senflood_baseline_modality_drop.py \
        --ckpt ./checkpoints/bl_perceiver.ckpt \
        --model perceiver \
        --xp_name perceiver_drop_eval

    # Multiple checkpoints in one run
    python script_test_senflood_baseline_modality_drop.py \
        --ckpts unet=./checkpoints/bl_unet.ckpt perceiver=./checkpoints/bl_perceiver.ckpt \
        --xp_name baseline_drop_eval \
        --ablations all s2_only s1_only rgb_only no_swir no_re
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.unet.model_unet import UNet
from training.VIT.model_vit_upernet import ViTUPerNet
from training.ResNet.model_resnet_upernet import build_resnet_upernet
from training.perceiverIO.perceiver_seg import PerceiverSeg
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = Sen1Floods11BaselineDataset.NUM_CLASSES   # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX  # 255
NUM_CHANNELS = Sen1Floods11BaselineDataset.NUM_CHANNELS  # 15
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
# CHANNEL-ZEROING WRAPPER
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
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()

# Single-checkpoint mode
parser.add_argument("--ckpt",  type=str, default=None,
                    help="Path to a single checkpoint")
parser.add_argument("--model", type=str, default="resnet",
                    choices=["unet","vit","resnet","perceiver"],
                    help="Architecture for --ckpt")

# Multi-checkpoint mode: name=path pairs
parser.add_argument("--ckpts", type=str, nargs="+", default=None,
                    help="name=path pairs, e.g. unet=./ckpts/unet.ckpt resnet=./ckpts/rn.ckpt")

parser.add_argument("--xp_name",    type=str, required=True)
parser.add_argument("--data_dir",   type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers",type=int, default=4)
parser.add_argument("--ablations",  type=str, nargs="+",
                    default=["all","s2_only","s1_only","rgb_only","no_swir","no_re"])
parser.add_argument("--wandb",      action="store_true")

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
parser.add_argument("--num_latents",         type=int, default=512)
parser.add_argument("--latent_dim",          type=int, default=768)
parser.add_argument("--depth",               type=int, default=1)
parser.add_argument("--cross_heads",         type=int, default=16)
parser.add_argument("--latent_heads",        type=int, default=8)
parser.add_argument("--cross_dim_head",      type=int, default=64)
parser.add_argument("--latent_dim_head",     type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=6)
parser.add_argument("--no_weight_tie",       action="store_true")
parser.add_argument("--num_freq_bands",      type=int, default=16)
parser.add_argument("--max_freq",            type=float, default=16.0)
parser.add_argument("--attn_dropout",        type=float, default=0.0)
parser.add_argument("--ff_dropout",          type=float, default=0.0)

args = parser.parse_args()

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
# RUN
# =============================================================================

# results[model_name][ablation_name] = metrics dict
all_results = {}

for model_name, ckpt_path in model_ckpts:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}   Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # ── Load base model weights ───────────────────────────────────────
    base_model = build_model(model_name, args)

    trainer_module = BaselineTrainer.load_from_checkpoint(
        ckpt_path,
        strict=True,
        model=base_model,
        modality=MODALITY_KEY,
        temporal=False,
        task="senflood",
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    )
    trainer_module.eval()

    all_results[model_name] = {}

    for ablation_name in args.ablations:
        drop_bands    = parse_ablation(ablation_name)
        drop_channels = [BAND_TO_CHANNEL[b] for b in drop_bands]
        drop_str      = ",".join(drop_bands) if drop_bands else "none"

        print(f"\n  {'─'*50}")
        print(f"  Ablation : {ablation_name}   Drop : {drop_str}")
        print(f"  {'─'*50}")

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

out_path = f"./results_{args.xp_name}_baseline_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Experiment: {args.xp_name}\n")
    f.write(f"Checkpoints: {model_ckpts}\n\n")
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
