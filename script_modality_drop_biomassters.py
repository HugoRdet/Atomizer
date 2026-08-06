"""
BioMassters Atomizer — Modality Drop Inference Script
========================================================

Load a trained Atomizer checkpoint and evaluate on the test split under
different band configurations (S1-only, S2-only, RGB-only, no-SWIR,
no-red-edge, etc.), mirroring
script_test_biomassters_baseline_modality_drop.py's ablation set and
output format for direct comparison.

KEY STRUCTURAL DIFFERENCE from the baselines' version: Atomizer's band
drop is DATASET-driven (BioMasstersSkipDataset reads config
trainer.bands.drop and masks matching tokens at __getitem__ time via
padding-token semantics -- zero value + mask=1.0), not a model-forward
wrapper that zeros input channels. This means:

  - The MODEL is loaded ONCE per checkpoint and reused across every
    ablation -- no ChannelDropWrapper/RAMENChannelDropWrapper equivalent
    needed, since nothing about the model's forward() changes between
    ablations.
  - What changes per ablation is the TEST DATASET/DATALOADER: each
    ablation gets its own BioMasstersSkipDataset instance, constructed
    with config_model["trainer"]["bands"]["drop"] set to that ablation's
    band list, since BioMasstersSkipDataset reads that config once at
    __init__ time (see BioMasstersSkipDataset._resolve_drop_indices).
  - Dropped bands become padding tokens (zeroed value + mask=1.0,
    ignored by cross-attention) rather than zeroed input channels --
    this is Atomizer's native missing-modality representation, the
    architectural point of comparison against the baselines' channel-
    zeroing approach.

Channel/band names: SAME as the baselines' script (BioMasstersSkipDataset's
ALL_BAND_NAMES): B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 (S2, CLP excluded)
and VV_asc,VH_asc,VV_desc,VH_desc (S1).

Usage
-----
    # Single checkpoint, default ablation set
    python script_test_biomassters_atomizer_modality_drop.py \
        --ckpt ./checkpoints/biomassters/biomassters_run1-last.ckpt \
        --xp_name atomizer_drop_eval

    # Custom ablation subset
    python script_test_biomassters_atomizer_modality_drop.py \
        --ckpt ./checkpoints/biomassters/biomassters_run1-last.ckpt \
        --xp_name atomizer_drop_eval --ablations all s2_only s1_only
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.utils_dataset_biomasters import BioMasstersSkipDataset
from training.utils.datasets.collate_biomassters_skip import collate_biomassters_skip
from training.trainer_biomassters import Model_BioMassters_Skip
from training.utils.lookup_positional import create_biomassters_bands_info


# =============================================================================
# ABLATIONS -- SAME set/names as script_test_biomassters_baseline_modality_drop.py
# =============================================================================

ALL_S2 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ALL_S1 = ["VV_asc", "VH_asc", "VV_desc", "VH_desc"]
ALL_BANDS = ALL_S2 + ALL_S1

BUILTIN_ABLATIONS = {
    "all":      [],                                                    # nothing dropped
    "s2_only":  ALL_S1,                                                 # drop S1
    "s1_only":  ALL_S2,                                                 # drop S2
    "rgb_only": [b for b in ALL_BANDS if b not in ["B02", "B03", "B04"]],  # keep only RGB
    "no_swir":  ["B11", "B12"],
    "no_re":    ["B05", "B06", "B07", "B8A"],
}


def parse_ablation(name: str):
    """Returns list of band names to drop."""
    if name in BUILTIN_ABLATIONS:
        return BUILTIN_ABLATIONS[name]
    # Inline: "drop=VV_asc,VH_asc"
    for part in name.strip().split():
        if part.startswith("drop="):
            return [b.strip() for b in part[5:].split(",") if b.strip()]
    return []


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="BioMassters Atomizer Modality Drop Eval")
parser.add_argument("--ckpt",         type=str, required=True,
                    help="Path to the trained Atomizer checkpoint (.ckpt)")
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str, default="config_test-Biomassters.yaml")
parser.add_argument("--data_dir",     type=str, default="./data/biomassters")
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--ablations",    type=str, nargs="+",
                    default=["all", "s2_only", "s1_only", "rgb_only", "no_swir", "no_re"])
parser.add_argument("--wandb",        action="store_true")
parser.add_argument("--num_timesteps", type=int, default=None,
                    help="Fixed timesteps per sensor (overrides config). MUST match "
                         "what the checkpoint was trained with.")

args = parser.parse_args()


# =============================================================================
# CONFIG & LOOKUP (built ONCE, model loaded ONCE)
# =============================================================================

config_model         = read_yaml("./training/configs/" + args.config_model)
configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

if args.num_timesteps is not None:
    config_model.setdefault("dataset", {})["num_timesteps"] = args.num_timesteps

fixed_T = config_model.get("dataset", {}).get("num_timesteps", BioMasstersSkipDataset.N_MONTHS)

lookup_table = Lookup_encoding(
    read_yaml(configs_dataset_path), create_biomassters_bands_info(), config_model)

print(f"\n{'='*60}")
print(f"  BioMassters Atomizer — Modality Drop Eval")
print(f"  Checkpoint: {args.ckpt}")
print(f"  Timesteps:  {fixed_T} (fixed -- MUST match training)")
print(f"  Ablations:  {args.ablations}")
print(f"{'='*60}\n")


# =============================================================================
# MODEL (loaded ONCE -- reused unchanged across every ablation, since band
# drop lives in the dataset, not the model). Target normalization stats
# (agb_mean/std) must match what the checkpoint was trained with -- pulled
# from a throwaway "all bands" train dataset instance (cheap: reads the
# cached normalization_stats.pt, doesn't recompute).
# =============================================================================

_stats_probe_config = {**config_model}
_stats_probe_config["trainer"] = {**config_model["trainer"], "bands": {"keep": None, "drop": None}}
_stats_probe_ds = BioMasstersSkipDataset(
    root_path=args.data_dir, mode="train",
    config_model=_stats_probe_config, look_up=lookup_table,
)
_agb_mean = _stats_probe_ds.norm_stats["agb_mean"].item()
_agb_std  = _stats_probe_ds.norm_stats["agb_std"].item()
print(f"[Atomizer-Eval] AGB target normalization: z-score "
      f"(mean={_agb_mean:.4f}, std={_agb_std:.4f})")
del _stats_probe_ds  # only needed the stats, not the dataset itself

model = Model_BioMassters_Skip(
    config=config_model, wand=False, name=args.xp_name,
    transform=None, lookup_table=lookup_table,
    agb_mean=_agb_mean, agb_std=_agb_std,
)

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
result = model.load_state_dict(state, strict=False)
print(f"[Atomizer-Eval] Loaded checkpoint — "
      f"missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")
if result.missing_keys:
    print(f"[Atomizer-Eval] First 5 missing: {result.missing_keys[:5]}")
if result.unexpected_keys:
    print(f"[Atomizer-Eval] First 5 unexpected: {result.unexpected_keys[:5]}")
model.eval()


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb.init(
        name=f"{args.xp_name}_atomizer_drop",
        project="BioMassters",
        config={"ckpt": args.ckpt, "ablations": args.ablations},
    )
    wandb_logger = WandbLogger(project="BioMassters")


# =============================================================================
# ABLATION LOOP -- rebuild the TEST DATASET per ablation (band drop is
# baked into dataset construction), NOT the model.
# =============================================================================

def make_test_loader(drop_bands: list):
    """
    Builds a fresh BioMasstersSkipDataset(mode="test") with
    trainer.bands.drop set to `drop_bands`, and its DataLoader. A new
    config dict is used per call (shallow-copied at the "trainer" level)
    so ablations never leak into each other via shared mutable config.
    """
    ablation_config = {**config_model}
    ablation_config["trainer"] = {
        **config_model["trainer"],
        "bands": {"keep": None, "drop": drop_bands if drop_bands else None},
    }
    ds = BioMasstersSkipDataset(
        root_path=args.data_dir, mode="test",
        config_model=ablation_config, look_up=lookup_table,
    )
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(ds, shuffle=False)
    loader = DataLoader(
        ds, batch_size=config_model["trainer"]["batchsize"],
        shuffle=False, sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_biomassters_skip,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return loader


all_results = {}

for ablation_name in args.ablations:
    drop_bands = parse_ablation(ablation_name)
    drop_str = ",".join(drop_bands) if drop_bands else "none"

    print(f"\n  {'─'*50}")
    print(f"  Ablation : {ablation_name}   Drop : {drop_str}")
    print(f"  {'─'*50}")

    test_loader = make_test_loader(drop_bands)

    trainer = Trainer(
        devices=-1, accelerator="gpu", precision="bf16-mixed",
        logger=wandb_logger, enable_progress_bar=True, enable_model_summary=False,
    )

    results = trainer.test(model, test_loader, verbose=True)
    metrics = results[0] if results else {}
    all_results[ablation_name] = metrics

    if args.wandb and wandb_logger:
        import wandb
        wandb.log({f"atomizer/{ablation_name}/{k}": v for k, v in metrics.items()})


# =============================================================================
# SUMMARY TABLE
# =============================================================================

if all_results:
    print(f"\n\n{'='*80}")
    print(f"  ATOMIZER MODALITY DROP SUMMARY — {args.xp_name}")
    print(f"{'='*80}")

    sample_metrics = next(m for m in all_results.values() if m)
    metric_keys = list(sample_metrics.keys())

    for mkey in metric_keys:
        print(f"\n  Metric: {mkey}")
        for abl in args.ablations:
            v = all_results.get(abl, {}).get(mkey, float("nan"))
            print(f"    {abl:<14} {v:.4f}")

    print(f"\n\n  Flat table (RMSE):")
    print(f"  {'Ablation':<14} {'Drop':<40} {'test_RMSE':<14} {'test_MAE':<14}")
    print(f"  {'─'*82}")
    for abl in args.ablations:
        drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
        rmse = all_results.get(abl, {}).get("test_RMSE", float("nan"))
        mae  = all_results.get(abl, {}).get("test_MAE", float("nan"))
        print(f"  {abl:<14} {drop_str:<40} {rmse:<14.4f} {mae:<14.4f}")

    print(f"\n{'='*80}\n")


# =============================================================================
# WRITE RESULTS -- same filename convention as the baselines' script, so a
# downstream table-builder can glob results_*_modality_drop.txt uniformly.
# =============================================================================

out_path = f"./results_{args.xp_name}_atomizer_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Experiment: {args.xp_name}\n")
    f.write(f"Checkpoint: {args.ckpt}\n\n")
    f.write(f"{'Ablation':<14} {'Drop':<40} {'test_RMSE':<14} {'test_MAE':<14}\n")
    f.write("─" * 82 + "\n")
    for abl in args.ablations:
        drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
        rmse = all_results.get(abl, {}).get("test_RMSE", float("nan"))
        mae  = all_results.get(abl, {}).get("test_MAE", float("nan"))
        f.write(f"{abl:<14} {drop_str:<40} {rmse:<14.4f} {mae:<14.4f}\n")

print(f"[Atomizer-Eval] Results saved to {out_path}")

if args.wandb and wandb_logger:
    import wandb
    wandb.finish()
