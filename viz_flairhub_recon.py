"""
Sen1Floods11 — Modality Drop Inference Script
==============================================

Load a trained Atomizer checkpoint and evaluate on the test split under
different band configurations, without retraining.

How it works
------------
For each ablation config the script:
  1. Instantiates Sen1Floods11Dataset with bands.keep / bands.drop set.
  2. Builds a DataModule and runs trainer.test().
  3. Prints a summary table of mIoU per config.

The model checkpoint is loaded once and reused for all ablations.
The latent grid stays identical across ablations because dropped bands
are replaced by masked padding tokens (same N tokens, same Voronoi cells).

Usage
-----
    python script_test_senflood_modality_drop.py \
        --ckpt ./checkpoints/my_run.ckpt \
        --xp_name modality_drop_eval

    # Custom ablation list (yaml band names):
    python script_test_senflood_modality_drop.py \
        --ckpt ./checkpoints/my_run.ckpt \
        --xp_name modality_drop_eval \
        --ablations "all" "s2_only" "s1_only" "rgb_only"

Available built-in ablation names
----------------------------------
    all        — all 15 bands (baseline, no drop)
    s2_only    — keep all S2 (B01–B12), drop VV VH
    s1_only    — keep all bands, drop B01–B12  (SAR only)
    rgb_only   — keep all bands, drop everything except B02 B03 B04
    no_swir    — keep all bands, drop B10 B11 B12
    no_re      — keep all bands, drop red-edge (B05 B06 B07 B08A)

    Or define your own inline:
        --ablations "keep=B02,B03,B04,VV,VH drop=VV,VH"
    Format: "keep=<comma-list> drop=<comma-list>"
    Either key can be omitted (keep omitted = all bands).
"""

import os
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.trainer_SENFLOOD import Model_SenFlood
from training.utils.datasets.utils_dataset_SENFLOOD import Sen1Floods11Dataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.token_grouping import collate_grouped

# =============================================================================
# BUILT-IN ABLATION CONFIGS
# name → (keep: list|None, drop: list|None)
# =============================================================================

ALL_S2 = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
ALL_S1 = ["VV","VH"]
ALL_BANDS = ALL_S2 + ALL_S1

BUILTIN_ABLATIONS = {
    "all":      (None,      None),
    "s2_only":  (None,      ALL_S1),
    "s1_only":  (None,      ALL_S2),
    "rgb_only": (None,      [b for b in ALL_BANDS if b not in ["B02","B03","B04"]]),
    "no_swir":  (None,      ["B10","B11","B12"]),
    "no_re":    (None,      ["B05","B06","B07","B08A"]),
}


def parse_ablation(name: str):
    """
    Parse an ablation spec. Either a builtin name or inline format:
        "keep=B02,B03,B04 drop=VV,VH"
    Returns (keep: list|None, drop: list|None).
    """
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

parser = argparse.ArgumentParser(description="Sen1Floods11 modality drop evaluation")
parser.add_argument("--ckpt",      type=str, required=True,  help="Path to .ckpt file")
parser.add_argument("--xp_name",   type=str, required=True,  help="Experiment name")
parser.add_argument("--data_dir",  type=str, default="./data/SENFLOOD")
parser.add_argument("--config",    type=str, default="./training/configs/config_test-SENFLOOD.yaml")
parser.add_argument("--bands_yaml",type=str, default="./data/bands_info/bands.yaml")
parser.add_argument("--dataset_config", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ablations", type=str, nargs="+",
                    default=["all", "s2_only", "s1_only"],
                    help="Ablation names or inline 'keep=... drop=...' specs")
parser.add_argument("--wandb",     action="store_true", help="Log results to wandb")
args = parser.parse_args()


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
print(f"  Sen1Floods11 Modality Drop Evaluation")
print(f"  Checkpoint: {args.ckpt}")
print(f"  Ablations:  {args.ablations}")
print(f"{'='*60}\n")


# =============================================================================
# LOAD MODEL ONCE
# =============================================================================

print("[Eval] Loading checkpoint...")
model = Model_SenFlood.load_from_checkpoint(
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
        project="SenFlood",
        config={"ckpt": args.ckpt, "ablations": args.ablations},
    )
    wandb_logger = WandbLogger(project="SenFlood")


# =============================================================================
# RUN ABLATIONS
# =============================================================================

results = []   # list of (name, metrics_dict)

for ablation_name in args.ablations:

    keep, drop = parse_ablation(ablation_name)

    keep_str = ",".join(keep) if keep else "ALL"
    drop_str = ",".join(drop) if drop else "none"
    print(f"\n{'─'*60}")
    print(f"  Ablation : {ablation_name}")
    print(f"  Keep     : {keep_str}")
    print(f"  Drop     : {drop_str}")
    print(f"{'─'*60}")

    # ── Inject band config into model config ─────────────────────────────
    # Sen1Floods11Dataset reads bands.keep / bands.drop from config_model.
    # We patch it per-ablation; the model weights are untouched.
    config_model["trainer"]["bands"] = {
        "keep": keep,
        "drop": drop,
    }

    # ── Build test datamodule ─────────────────────────────────────────────
    data_module = UnifiedDataModule(
        path=args.data_dir,
        batch_size=1,          # full 512×512 tiles, keep memory safe
        num_workers=args.num_workers,
        trans_modalities=None,
        trans_tokens=None,
        model=config_model["encoder"],
        dataset_config=dataset_config,
        config_model=config_model,
        look_up=lookup_table,
        dataset_class=Sen1Floods11Dataset,
    )

    # ── Trainer (test only, no fitting) ──────────────────────────────────
    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # trainer.test returns a list of dicts, one per dataloader
    test_results = trainer.test(model, datamodule=data_module, verbose=True)
    metrics = test_results[0] if test_results else {}

    results.append((ablation_name, keep_str, drop_str, metrics))

    if args.wandb and wandb_logger:
        import wandb
        wandb.log({
            f"{ablation_name}/{k}": v
            for k, v in metrics.items()
        })


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*70}")
print(f"  MODALITY DROP SUMMARY — {args.xp_name}")
print(f"  Checkpoint: {args.ckpt}")
print(f"{'='*70}")

# Collect all metric keys that appeared
all_keys = []
for _, _, _, m in results:
    for k in m:
        if k not in all_keys:
            all_keys.append(k)

# Print header
col_w = 22
print(f"\n{'Ablation':<18} {'Keep':<30} {'Drop':<30}", end="")
for k in all_keys:
    print(f"  {k:<14}", end="")
print()
print("─" * (18 + 30 + 30 + len(all_keys) * 16))

# Print rows
for name, keep_str, drop_str, metrics in results:
    print(f"{name:<18} {keep_str:<30} {drop_str:<30}", end="")
    for k in all_keys:
        v = metrics.get(k, float("nan"))
        print(f"  {v:<14.4f}", end="")
    print()

print(f"{'='*70}\n")

# Save to txt
out_path = f"./results_{args.xp_name}_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Checkpoint: {args.ckpt}\n\n")
    f.write(f"{'Ablation':<18} {'Keep':<30} {'Drop':<30}")
    for k in all_keys:
        f.write(f"  {k:<14}")
    f.write("\n" + "─" * (18 + 30 + 30 + len(all_keys) * 16) + "\n")
    for name, keep_str, drop_str, metrics in results:
        f.write(f"{name:<18} {keep_str:<30} {drop_str:<30}")
        for k in all_keys:
            v = metrics.get(k, float("nan"))
            f.write(f"  {v:<14.4f}")
        f.write("\n")

print(f"[Eval] Results saved to {out_path}")

if args.wandb and wandb_logger:
    import wandb
    wandb.finish()
