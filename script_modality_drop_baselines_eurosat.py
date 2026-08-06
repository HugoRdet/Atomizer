"""
EuroSAT-SAR Baseline — Modality Drop Inference Script
========================================================

Load a trained ResNet/ViT/Perceiver/RAMEN baseline checkpoint
(ClassificationBaselineTrainer on EuroSATSARBaselineDataset) and evaluate
on the test split under different band configurations, without retraining.

IMPORTANT — architecture must match training exactly:
    Unlike Model_ForestNet (which rebuilds its encoder from the saved YAML
    config), a baseline's architecture is NOT reconstructed from the
    checkpoint — you must pass the same --model / --resnet_variant /
    --vit_* / --ramen_* flags used when the checkpoint was trained, or the
    freshly built model's shapes won't match the saved weights. The
    checkpoint is loaded with strict=False and missing/unexpected keys are
    printed, so a real mismatch shows up loudly instead of silently
    loading garbage.

Band dropping: EuroSATSARBaselineDataset zeroes dropped/non-kept channels
rather than removing them (fixed 15-channel input shape across ablations),
so one trained checkpoint (in_channels=15 always, or for RAMEN: 13
optical + 2 sar always) can be re-tested under every ablation without
rebuilding the model.

RAMEN specifics: unlike ResNet/ViT/Perceiver (which consume a single
[B,15,H,W] "fused" tensor), RAMEN needs {"optical","sar"} — handled by
RAMENInputAdapter, which just reshapes the already-zeroed "fused" tensor
(no drop logic of its own needed, since EuroSATSARBaselineDataset already
applies the ablation at the dataset level before RAMEN ever sees it).

Note on --seeds: unlike the Atomizer/Perceiver-style modality-drop script,
a ResNet/ViT/RAMEN baseline in eval() mode (dropout disabled, no
stochastic token subsampling) is deterministic on a fixed test set —
expect std=0 across seeds here. The --seeds machinery is kept for
structural parity with the Atomizer script and in case a model has other
eval-time stochasticity (e.g. MC-dropout variants).

Usage
-----
    python script_test_eurosat_sar_baselines_modality_drop.py \
        --ckpt ./checkpoints/eurosat_sar_baselines/bl_eurosat_sar_bl_resnet50_resnet-last.ckpt \
        --xp_name resnet50_modality_drop \
        --model resnet --resnet_variant resnet50 \
        --ablations "all" "s2_only" "s1_only" "rgb_only" "no_swir" "no_re"

    python script_test_eurosat_sar_baselines_modality_drop.py \
        --ckpt ./checkpoints/eurosat_sar_baselines/bl_ramen_fused_ramen-last.ckpt \
        --xp_name ramen_modality_drop \
        --model ramen \
        --ablations "all" "s2_only" "s1_only" "rgb_only" "no_swir" "no_re"

Built-in ablation names
------------------------
    all        — all 15 bands (baseline, no drop)
    s2_only    — keep all optical, drop VV VH
    s1_only    — keep all bands, drop all 13 optical bands (SAR only)
    rgb_only   — keep only Blue, Green, Red
    no_swir    — drop SWIR1, SWIR2
    no_re      — drop RedEdge1, RedEdge2, RedEdge3, RedEdge4

    Inline:  --ablations "keep=Blue,Green,Red,VV,VH drop=VV,VH"
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


def reseed(seed: int):
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# identical to script_train_eurosat_sar_baselines.py, so it can't drift
# out of sync.
# =============================================================================

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
# RAMEN INPUT ADAPTER (identical to script_train_eurosat_sar_baselines.py)
# =============================================================================

class RAMENInputAdapter(nn.Module):
    """
    Splits the dataset's merged image["fused"] : [B,15,H,W] tensor into
    RAMEN's expected {"optical": [B,13,H,W], "sar": [B,2,H,W]}.

    No modality-drop logic here: EuroSATSARBaselineDataset already zeroes
    dropped/non-kept bands at the DATASET level (see --bands_keep/
    --bands_drop equivalent here — this script's `bands_cfg` built from
    each ablation's keep/drop lists), so by the time this adapter sees
    "fused" any ablation is already applied. This adapter's only job is
    reshaping for RAMEN's per-modality input.
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
# COLLATE (identical to script_train_eurosat_sar_baselines.py)
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
# MODEL BUILDER (identical to script_train_eurosat_sar_baselines.py — must
# be called with the SAME args used at training time)
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
# BUILT-IN ABLATION CONFIGS — same vocabulary as EuroSATSARDataset /
# EuroSATSARBaselineDataset.ALL_BAND_NAMES
# =============================================================================

ALL_OPTICAL = [
    "Blue", "Green", "Red", "NIR", "RedEdge1", "RedEdge2", "RedEdge3", "RedEdge4",
    "SWIR1", "SWIR2", "CoastalAerosol", "WaterVapour", "Cirrus",
]
ALL_SAR   = ["VV", "VH"]
ALL_BANDS = ALL_OPTICAL + ALL_SAR

BUILTIN_ABLATIONS = {
    "all":      (None, None),
    "s2_only":  (None, ALL_SAR),
    "s1_only":  (None, ALL_OPTICAL),
    "rgb_only": (None, [b for b in ALL_BANDS if b not in ["Blue", "Green", "Red"]]),
    "no_swir":  (None, ["SWIR1", "SWIR2"]),
    "no_re":    (None, ["RedEdge1", "RedEdge2", "RedEdge3", "RedEdge4"]),
}


def parse_ablation(name: str):
    if name in BUILTIN_ABLATIONS:
        return BUILTIN_ABLATIONS[name]
    keep, drop = None, None
    for part in name.strip().split():
        if part.startswith("keep="):
            keep = [b.strip() for b in part[5:].split(",") if b.strip()]
        elif part.startswith("drop="):
            drop = [b.strip() for b in part[5:].split(",") if b.strip()]
    return keep, drop


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="EuroSAT-SAR baseline modality drop evaluation")
parser.add_argument("--ckpt",      type=str, required=True,  help="Path to .ckpt file")
parser.add_argument("--xp_name",   type=str, required=True,  help="Experiment name")
parser.add_argument("--model",     type=str, required=True, choices=["resnet", "vit", "perceiver", "ramen"],
                    help="MUST match the checkpoint's training-time architecture.")
parser.add_argument("--data_dir",  type=str, default="./data",
                    help="Parent dir containing EuroSAT_MS/ and EuroSAT-SAR/")
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--dropout",     type=float, default=0.1,
                    help="Irrelevant at eval time (model.eval() disables dropout), "
                         "but kept for architecture-building parity with training.")

parser.add_argument("--ablations", type=str, nargs="+",
                    default=["all", "s2_only", "s1_only"],
                    help="Ablation names or inline 'keep=... drop=...' specs")
parser.add_argument("--wandb",     action="store_true", help="Log results to wandb")
parser.add_argument("--seeds", type=int, default=1,
                    help="NUMBER of random seeds to evaluate (see module docstring "
                         "re: expected std=0 for deterministic baselines).")
parser.add_argument("--seed_base", type=int, default=42)

# Image size (must match training)
parser.add_argument("--img_size",  type=int, default=64)

# ViT (must match training)
parser.add_argument("--vit_embed_dim",  type=int, default=384)
parser.add_argument("--vit_depth",      type=int, default=12)
parser.add_argument("--vit_num_heads",  type=int, default=6)
parser.add_argument("--vit_patch_size", type=int, default=8)

# ResNet (must match training)
parser.add_argument("--resnet_variant", type=str, default="resnet50",
                    choices=["resnet_super_small", "resnet_small",
                             "resnet50", "resnet101", "resnet152"])

# RAMEN (must match training)
parser.add_argument("--ramen_embed_dim", type=int, default=384)
parser.add_argument("--ramen_depth",     type=int, default=12)
parser.add_argument("--ramen_num_heads", type=int, default=8)
parser.add_argument("--ramen_input_res", type=float, default=10.0)
parser.add_argument("--ramen_res",       type=float, default=40.0)

# Perceiver-IO (must match training)
parser.add_argument("--num_latents",         type=int, default=512)
parser.add_argument("--latent_dim",          type=int, default=768)
parser.add_argument("--perceiver_depth",     type=int, default=6)
parser.add_argument("--cross_heads",         type=int, default=1)
parser.add_argument("--latent_heads",        type=int, default=8)
parser.add_argument("--cross_dim_head",      type=int, default=64)
parser.add_argument("--latent_dim_head",     type=int, default=64)
parser.add_argument("--self_per_cross_attn", type=int, default=1)
parser.add_argument("--no_weight_tie",       action="store_true")
parser.add_argument("--num_freq_bands",      type=int, default=16)
parser.add_argument("--max_freq",            type=float, default=16.0)
parser.add_argument("--attn_dropout",        type=float, default=0.0)
parser.add_argument("--ff_dropout",          type=float, default=0.0)

args = parser.parse_args()

SEED_LIST = [args.seed_base + i for i in range(max(1, args.seeds))]


# =============================================================================
# SETUP
# =============================================================================

print(f"\n{'='*60}")
print(f"  EuroSAT-SAR Baseline Modality Drop Evaluation")
print(f"  Checkpoint: {args.ckpt}")
print(f"  Model:      {args.model}"
      f"{' (' + args.resnet_variant + ')' if args.model == 'resnet' else ''}")
if args.model == "ramen":
    print(f"  RAMEN:      embed_dim={args.ramen_embed_dim}, depth={args.ramen_depth}, "
          f"heads={args.ramen_num_heads}, res={args.ramen_res} "
          f"(input_res={args.ramen_input_res})")
print(f"  Ablations:  {args.ablations}")
print(f"{'='*60}\n")


# =============================================================================
# BUILD MODEL + LOAD CHECKPOINT ONCE
# =============================================================================

model = build_model(args.model, NUM_CHANNELS, NUM_CLASSES, args)
trainer_module = ClassificationBaselineTrainer(
    model=model,
    modality=MODALITY_KEY,
    num_classes=NUM_CLASSES,
)

print(f"[Eval] Loading checkpoint: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
missing, unexpected = trainer_module.load_state_dict(ckpt["state_dict"], strict=False)
if missing:
    print(f"[Eval] WARN: {len(missing)} missing keys — likely means --model/"
          f"--resnet_variant/--vit_*/--ramen_* flags don't match the "
          f"checkpoint's training-time architecture:")
    for k in missing[:10]:
        print(f"    {k}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
if unexpected:
    print(f"[Eval] {len(unexpected)} unexpected keys ignored:")
    for k in unexpected[:5]:
        print(f"    {k}")
    if len(unexpected) > 5:
        print(f"    ... and {len(unexpected) - 5} more")
if not missing and not unexpected:
    print(f"[Eval] Checkpoint loaded cleanly (no missing/unexpected keys).")

trainer_module.eval()
print("[Eval] Model ready.\n")


# =============================================================================
# WANDB (optional)
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"{args.xp_name}_modality_drop",
        project="Atomizer_EuroSAT_SAR_Baselines",
        config={"ckpt": args.ckpt, "ablations": args.ablations, "model": args.model},
    )
    wandb_logger = WandbLogger(project="Atomizer_EuroSAT_SAR_Baselines")


# =============================================================================
# RUN ABLATIONS
# =============================================================================

results = []
multi_seed = len(SEED_LIST) > 1
print(f"[Eval] Seeds ({len(SEED_LIST)}): {SEED_LIST}  ({'multi-seed study' if multi_seed else 'single run'})")

for ablation_name in args.ablations:

    keep, drop = parse_ablation(ablation_name)
    keep_str = ",".join(keep) if keep else "ALL"
    drop_str = ",".join(drop) if drop else "none"
    print(f"\n{'─'*60}")
    print(f"  Ablation : {ablation_name}")
    print(f"  Keep     : {keep_str}")
    print(f"  Drop     : {drop_str}")
    print(f"{'─'*60}")

    bands_cfg = {"keep": keep, "drop": drop}

    per_seed_metrics = []
    for seed in SEED_LIST:
        reseed(seed)
        print(f"    · seed={seed}")

        test_ds = EuroSATSARBaselineDataset(
            root_path=args.data_dir, mode="test", augment=False, bands=bands_cfg,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eurosat_sar_collate,
            pin_memory=True,
        )

        trainer = Trainer(
            devices=1,
            accelerator="gpu",
            precision="bf16-mixed",
            logger=wandb_logger,
            enable_progress_bar=False if multi_seed else True,
            enable_model_summary=False,
        )

        test_results = trainer.test(trainer_module, dataloaders=test_loader, verbose=not multi_seed)
        m = test_results[0] if test_results else {}
        per_seed_metrics.append(m)

        if args.wandb and wandb_logger:
            import wandb
            wandb.log({f"{ablation_name}/seed{seed}/{k}": v for k, v in m.items()})

    metric_names = []
    for m in per_seed_metrics:
        for k in m:
            if k not in metric_names:
                metric_names.append(k)

    agg = {}
    for k in metric_names:
        vals = [m[k] for m in per_seed_metrics if k in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        agg[k] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "n":    int(len(arr)),
            "values": vals,
        }

    results.append((ablation_name, keep_str, drop_str, agg))

    if "test_top1" in agg:
        a = agg["test_top1"]
        if multi_seed:
            print(f"    => test_top1  mean={a['mean']:.4f}  std={a['std']:.4f}  "
                  f"min={a['min']:.4f}  max={a['max']:.4f}  (n={a['n']})")
        else:
            print(f"    => test_top1  {a['mean']:.4f}")

    if args.wandb and wandb_logger:
        import wandb
        for k, a in agg.items():
            wandb.log({f"{ablation_name}/{k}_mean": a["mean"],
                       f"{ablation_name}/{k}_std":  a["std"],
                       f"{ablation_name}/{k}_min":  a["min"],
                       f"{ablation_name}/{k}_max":  a["max"]})


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*78}")
print(f"  MODALITY DROP SUMMARY (BASELINE) — {args.xp_name}")
print(f"  Checkpoint: {args.ckpt}")
print(f"  Model: {args.model}"
      f"{' (' + args.resnet_variant + ')' if args.model == 'resnet' else ''}")
print(f"  Seeds ({len(SEED_LIST)}): {SEED_LIST}")
print(f"{'='*78}")

all_keys = []
for _, _, _, agg in results:
    for k in agg:
        if k not in all_keys:
            all_keys.append(k)

multi_seed = len(SEED_LIST) > 1

for mkey in all_keys:
    print(f"\n  Metric: {mkey}")
    if multi_seed:
        header = (f"  {'Ablation':<14} {'Drop':<28} "
                  f"{'mean':>9} {'std':>8} {'min':>9} {'max':>9} {'n':>3}")
    else:
        header = f"  {'Ablation':<14} {'Drop':<28} {'value':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name, keep_str, drop_str, agg in results:
        a = agg.get(mkey)
        if a is None:
            continue
        if multi_seed:
            print(f"  {name:<14} {drop_str:<28} "
                  f"{a['mean']:>9.4f} {a['std']:>8.4f} "
                  f"{a['min']:>9.4f} {a['max']:>9.4f} {a['n']:>3}")
        else:
            print(f"  {name:<14} {drop_str:<28} {a['mean']:>9.4f}")

if "test_top1" in all_keys:
    print(f"\n  test_top1 (mean ± std):")
    print(f"  {'Ablation':<14} {'Drop':<28} {'top1':>22}")
    print("  " + "─" * 66)
    for name, keep_str, drop_str, agg in results:
        a = agg.get("test_top1")
        if a is None:
            continue
        if multi_seed:
            cell = f"{a['mean']:.4f} ± {a['std']:.4f}"
        else:
            cell = f"{a['mean']:.4f}"
        print(f"  {name:<14} {drop_str:<28} {cell:>22}")

print(f"\n{'='*78}\n")


# =============================================================================
# WRITE RESULTS
# =============================================================================

out_path = f"./results_{args.xp_name}_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Checkpoint: {args.ckpt}\n")
    f.write(f"Model: {args.model}"
            f"{' (' + args.resnet_variant + ')' if args.model == 'resnet' else ''}\n")
    f.write(f"Seeds ({len(SEED_LIST)}): {SEED_LIST}\n\n")
    for mkey in all_keys:
        f.write(f"Metric: {mkey}\n")
        if multi_seed:
            f.write(f"{'Ablation':<14} {'Drop':<28} "
                    f"{'mean':>9} {'std':>8} {'min':>9} {'max':>9} {'n':>3}   values\n")
        else:
            f.write(f"{'Ablation':<14} {'Drop':<28} {'value':>9}\n")
        f.write("─" * 96 + "\n")
        for name, keep_str, drop_str, agg in results:
            a = agg.get(mkey)
            if a is None:
                continue
            if multi_seed:
                vals_str = ",".join(f"{v:.4f}" for v in a["values"])
                f.write(f"{name:<14} {drop_str:<28} "
                        f"{a['mean']:>9.4f} {a['std']:>8.4f} "
                        f"{a['min']:>9.4f} {a['max']:>9.4f} {a['n']:>3}   [{vals_str}]\n")
            else:
                f.write(f"{name:<14} {drop_str:<28} {a['mean']:>9.4f}\n")
        f.write("\n")

print(f"[Eval] Results saved to {out_path}")

if args.wandb and wandb_logger:
    import wandb
    wandb.finish()
