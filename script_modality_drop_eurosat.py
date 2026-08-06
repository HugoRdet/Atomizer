"""
EuroSAT-SAR — Modality Drop Inference Script (Atomizer)
============================================================

Load a trained Atomizer EuroSAT-SAR checkpoint (Model_ForestNet) and
evaluate on the test split under different band configurations, without
retraining.

Mirror of script_test_senflood_modality_drop.py (SKIP variant), adapted
for EuroSAT-SAR classification:
    trainer :  Model_ForestNet          (was Model_SenFlood_Skip)
    dataset :  EuroSATSARDataset        (was Sen1Floods11SkipDataset)
    collate :  UnifiedDataModule default (was collate_grouped_skip)

No SKIP stack here: unlike Sen1Floods11SkipDataset, EuroSATSARDataset has
no pixel-skip cascade (no query_token_idx/query_token_valid — it's a
single CLS query per sample, not per-pixel segmentation queries), so
there's no skip-specific collate or checkpoint-loading concern to worry
about, and no need for the "is the skip actually firing" warning the
Sen1Floods11 script carries.

Band dropping: dropped bands are injected via config["trainer"]["bands"]
= {keep, drop}; EuroSATSARDataset masks them as padding tokens (same N
tokens, same grid), so the latent structure is identical across
ablations — same mechanism as Sen1Floods11.

Usage
-----
    python script_test_eurosat_sar_modality_drop.py \
        --ckpt ./checkpoints/eurosat_sar/atomiser_myrun-last.ckpt \
        --xp_name eurosat_modality_drop_eval \
        --ablations "all" "s2_only" "s1_only" "rgb_only" "no_swir" "no_re"

Built-in ablation names
-----------------------
    all        — all 15 bands (baseline, no drop)
    s2_only    — keep all optical, drop VV VH
    s1_only    — keep all bands, drop all 13 optical bands (SAR only)
    rgb_only   — keep all bands, drop everything except Blue Green Red
    no_swir    — drop SWIR1 SWIR2
    no_re      — drop RedEdge1 RedEdge2 RedEdge3 RedEdge4

    Inline:  --ablations "keep=Blue,Green,Red,VV,VH drop=VV,VH"

    (Same band vocabulary as script_test_eurosat_sar_baselines_modality_drop.py,
    for apples-to-apples comparison against the baseline results.)
"""

import os
import random
import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


def reseed(seed: int):
    """Reseed all RNGs so inference-time stochastic token sampling varies.
    Variance across seeds characterizes the model's stability to the random
    geographic token subsampling (torch.randperm in _sample_tokens) and any
    random val_sampling choice. If inference is fully deterministic, std=0,
    which is itself a valid finding."""
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from training.utils import read_yaml, Lookup_encoding

from training.trainer_FORESTNET import Model_ForestNet
from training.utils.datasets.utils_dataset_EUROSAT_SAR import EuroSATSARDataset
from training.utils.datasets.dataloaders import UnifiedDataModule


# =============================================================================
# BUILT-IN ABLATION CONFIGS  —  name → (keep: list|None, drop: list|None)
# Same band vocabulary as EuroSATSARDataset.ALL_BAND_NAMES / the baseline
# EuroSAT-SAR modality-drop script (Blue/Green/.../VV/VH, NOT B01-B12).
# =============================================================================

ALL_S2 = [
    "Blue", "Green", "Red", "NIR", "RedEdge1", "RedEdge2", "RedEdge3", "RedEdge4",
    "SWIR1", "SWIR2", "CoastalAerosol", "WaterVapour", "Cirrus",
]
ALL_S1 = ["VV", "VH"]
ALL_BANDS = ALL_S2 + ALL_S1

BUILTIN_ABLATIONS = {
    "all":      (None, None),
    "s2_only":  (None, ALL_S1),
    "s1_only":  (None, ALL_S2),
    "rgb_only": (None, [b for b in ALL_BANDS if b not in ["Blue", "Green", "Red"]]),
    "no_swir":  (None, ["SWIR1", "SWIR2"]),
    "no_re":    (None, ["RedEdge1", "RedEdge2", "RedEdge3", "RedEdge4"]),
}


def parse_ablation(name: str):
    """Builtin name or inline 'keep=... drop=...' → (keep|None, drop|None)."""
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

parser = argparse.ArgumentParser(description="EuroSAT-SAR modality drop evaluation (Atomizer)")
parser.add_argument("--ckpt",      type=str, required=True,  help="Path to .ckpt file")
parser.add_argument("--xp_name",   type=str, required=True,  help="Experiment name")
parser.add_argument("--data_dir",  type=str, default="./data",
                    help="Parent dir containing EuroSAT_MS/ and EuroSAT-SAR/")
parser.add_argument("--config",    type=str, default="./training/configs/config_test-EUROSAT.yaml")
parser.add_argument("--bands_yaml",type=str, default="./data/bands_info/bands.yaml")
parser.add_argument("--dataset_config", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ablations", type=str, nargs="+",
                    default=["all", "s2_only", "s1_only"],
                    help="Ablation names or inline 'keep=... drop=...' specs")
parser.add_argument("--wandb",     action="store_true", help="Log results to wandb")
parser.add_argument("--seeds", type=int, default=1,
                    help="NUMBER of random seeds to evaluate. --seeds 1 (default) "
                         "runs a single deterministic pass; --seeds 30 runs 30 "
                         "seeds and reports mean/std/min/max per ablation.")
parser.add_argument("--seed_base", type=int, default=42,
                    help="Base seed; the N seeds are seed_base + [0..N-1] "
                         "(reproducible across runs).")
args = parser.parse_args()

# Expand the seed COUNT into concrete, reproducible seed values.
SEED_LIST = [args.seed_base + i for i in range(max(1, args.seeds))]


# =============================================================================
# SETUP
# =============================================================================

config_model   = read_yaml(args.config)
lookup_table   = Lookup_encoding(
    read_yaml(args.dataset_config),
    read_yaml(args.bands_yaml),
    config_model,
)
dataset_config = read_yaml(args.bands_yaml)

print(f"\n{'='*60}")
print(f"  EuroSAT-SAR Modality Drop Evaluation (Atomizer)")
print(f"  Checkpoint: {args.ckpt}")
print(f"  Ablations:  {args.ablations}")
print(f"{'='*60}\n")


# =============================================================================
# LOAD MODEL ONCE
# =============================================================================

print("[Eval] Loading checkpoint into Model_ForestNet...")
model = Model_ForestNet.load_from_checkpoint(
    args.ckpt,
    strict=False,
    config=config_model,
    wand=False,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
)
model.eval()
print("[Eval] Checkpoint loaded.\n")


# =============================================================================
# WANDB (optional)
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"{args.xp_name}_modality_drop",
        project="Atomizer_EuroSAT_SAR",
        config={"ckpt": args.ckpt, "ablations": args.ablations},
    )
    wandb_logger = WandbLogger(project="Atomizer_EuroSAT_SAR")


# =============================================================================
# RUN ABLATIONS
# =============================================================================

# results[i] = (name, keep_str, drop_str, agg_metrics)
# agg_metrics[metric] = {"mean","std","min","max","n","values"}
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

    # ── Inject band config; EuroSATSARDataset masks dropped bands as padding ──
    config_model["trainer"]["bands"] = {"keep": keep, "drop": drop}

    # ── collect per-seed metric dicts ────────────────────────────────────────
    per_seed_metrics = []      # list of dict(metric -> value)
    for seed in SEED_LIST:
        reseed(seed)
        print(f"    · seed={seed}")

        data_module = UnifiedDataModule(
            path=args.data_dir,
            batch_size=1,
            num_workers=args.num_workers,
            trans_modalities=None,
            trans_tokens=None,
            model=config_model["encoder"],
            dataset_config=dataset_config,
            config_model=config_model,
            look_up=lookup_table,
            dataset_class=EuroSATSARDataset,
            # No collate_fn override — matches script_train_eurosat_sar.py,
            # which relies on UnifiedDataModule's default (EuroSATSARDataset
            # has no SKIP cascade, so the default/non-skip collate is correct
            # here, unlike the Sen1Floods11 SKIP eval script).
        )

        trainer = Trainer(
            devices=-1,
            accelerator="gpu",
            precision="bf16-mixed",
            logger=wandb_logger,
            enable_progress_bar=False if multi_seed else True,
            enable_model_summary=False,
        )

        test_results = trainer.test(model, datamodule=data_module, verbose=not multi_seed)
        m = test_results[0] if test_results else {}
        per_seed_metrics.append(m)

        if args.wandb and wandb_logger:
            import wandb
            wandb.log({f"{ablation_name}/seed{seed}/{k}": v for k, v in m.items()})

    # ── aggregate: mean / std / min / max per metric ─────────────────────────
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
            "std":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,  # sample std
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "n":    int(len(arr)),
            "values": vals,
        }

    results.append((ablation_name, keep_str, drop_str, agg))

    # per-ablation console summary
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
# SUMMARY TABLE  (mean ± std, with min/max)
# =============================================================================

print(f"\n\n{'='*78}")
print(f"  MODALITY DROP SUMMARY — {args.xp_name}")
print(f"  Checkpoint: {args.ckpt}")
print(f"  Seeds ({len(SEED_LIST)}): {SEED_LIST}")
print(f"{'='*78}")

# collect metric names across all ablations
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

# compact top1 table (mean ± std) across ablations
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
