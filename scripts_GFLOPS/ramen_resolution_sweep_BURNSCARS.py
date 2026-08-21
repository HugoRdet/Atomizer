"""
RAMEN Resolution Sweep — HLS BurnScars
==========================================

Evaluates a SINGLE trained RAMEN checkpoint across a range of working
resolutions (`res`), measuring GFLOPs/forward and test mIoU at each.
Single modality ("hls", 6 bands), no modality dropping.

Why one checkpoint works across every resolution: RAMENBackbone's
pos_embed is a non-persistent buffer, recomputed at construction time
from `effective_size = input_size * (input_res/res)` — it is NEVER
saved into the checkpoint's state_dict. Every learnable weight
(SpectralProjector, ScaleResampler, ViT blocks) is resolution-agnostic.
So rebuilding the model at a different `res` and loading the SAME
checkpoint's weights is always shape-safe — no retraining needed to
sweep resolution at eval time.

Differences from the Sen1Floods11 version of this script:
  - Single modality ("hls"), so NO RAMENInputAdapter is needed —
    RAMENUPerNet's modality name is set to "hls" to match the dataset's
    own dict key directly, so trainer_module.model stays as the loaded
    model itself (no split/wrap step).
  - HLS is standardized to 30m GSD (harmonized Landsat/Sentinel), NOT
    Sentinel-2's native 10m — --ramen_input_res defaults to 30.0.
  - Band naming uses "B8A" (no leading zero), not "B08A".

Sliding-window mechanics are unchanged: --ramen_window_size (pixel-space
crop of the *input* image) stays fixed across the sweep, so the number
of windows tiling the full native image stays fixed too. Only the
number of TOKENS per window changes with `res`.

# >>> PARETO_HULL: after the sweep, this script also computes the Pareto
# front (mIoU maximize, GFLOPs minimize) over ALL swept resolutions and
# reduces it to its convex hull (the efficient frontier — points where
# marginal mIoU gain per extra GFLOP is diminishing; a non-dominated
# point that still sits below the line connecting two better trade-off
# points is dropped). Both are printed and saved. Point identity here is
# a single axis (`res`, the working resolution in m/px), unlike the
# UniverSat sweep scripts which key on (patch_size_m, output_stride[,
# subpatch_px]). Every point already has a REAL measured gflops (this
# sweep never skips GFLOPs), so the hull is a genuine GFLOPs-vs-mIoU
# curve directly, no fallback needed.
#
# >>> ADDED: this script previously only wrote a .txt summary. A JSON
# output (./scripts_GFLOPS/results/<xp_name>_resolution_sweep.json) is
# now also written, for consistency with the other GFLOPs sweep scripts
# and so recompute_pareto_hull.py can be pointed at it directly.

Usage
-----
    python script_ramen_resolution_sweep_burnscars.py \
        --ckpt ./checkpoints/burnscars_baselines/bl_ramen_ramen-last.ckpt \
        --xp_name ramen_res_sweep_burnscars \
        --ramen_window_size 128 --ramen_stride 96

    # Custom resolution list
    python script_ramen_resolution_sweep_burnscars.py \
        --ckpt ./checkpoints/.../ramen-last.ckpt \
        --xp_name ramen_res_sweep_burnscars_coarse \
        --resolutions 30 60 90 150 300
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import argparse
import json

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode   # >>> FLOPS_METHOD
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_burnscars_baselines import (
    BurnScarsBaselineDataset,
)
from training.RAMEN.ramen_upernet import build_ramen_upernet  # adjust import path
from training.sliding_window import sliding_window_inference  # adjust import path
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = BurnScarsBaselineDataset.NUM_CLASSES    # 2
IGNORE_INDEX = BurnScarsBaselineDataset.IGNORE_INDEX    # 255
MODALITY_KEY = "hls"

# Wavelength table keyed to HLS's OWN band naming (note "B8A", not "B08A").
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

# Single modality, keyed as "hls" to match the dataset's own dict key
# exactly — no adapter needed, unlike Sen1Floods11's optical+sar split.
RAMEN_INPUT_BANDS = {MODALITY_KEY: BurnScarsBaselineDataset.HLS_BANDS}
RAMEN_WAVELENGTHS = {MODALITY_KEY: HLS_WAVELENGTHS_NM}


# =============================================================================
# COLLATE
# =============================================================================

def burnscars_collate(batch):
    images = {}
    sensor_keys = list(batch[0]["image"].keys())
    for key in sensor_keys:
        images[key] = torch.stack([s["image"][key] for s in batch])
    targets  = torch.stack([s["target"]   for s in batch])
    metadata = [s["metadata"] for s in batch]
    return {"image": images, "target": targets, "metadata": metadata}


# =============================================================================
# >>> FLOPS_METHOD: FlopCounterMode (SDPA attention counted) — matches
# script_universat_sweep_burnscars.py / _senflood.py exactly, replacing the
# previous torch.profiler(with_flops=True) harness. This matters beyond
# consistency: torch.profiler's with_flops=True has no formulas for fused
# scaled_dot_product_attention kernels and silently drops ALL attention
# FLOPs — a large undercount for a transformer backbone like RAMEN's ViT
# blocks. Mixing the two methodologies in the same table would make RAMEN
# look artificially cheap relative to UniverSat. Do not mix FlopCounterMode
# numbers with any older torch.profiler-harness numbers you may already
# have on disk from a previous run of this script — re-run to get numbers
# on the same footing.
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
    shape — the mean is a sanity check). forward_fn internally loops over
    sliding-window tiles, so this captures the TOTAL cost of one
    full-image forward.
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
# >>> PARETO_HULL: Pareto front + convex hull over (gflops, miou)
# =============================================================================

def _pareto_front(points, cost_key="gflops", score_key="miou"):
    """
    point A dominates point B iff A.score >= B.score AND A.cost <= B.cost,
    with at least one strict inequality. NaN score/cost are excluded.
    """
    valid = [p for p in points
             if p.get(score_key) == p.get(score_key)
             and p.get(cost_key) == p.get(cost_key)]
    front = []
    for p in valid:
        dominated = False
        for q in valid:
            if q is p:
                continue
            better_or_equal = (q[score_key] >= p[score_key]) and (q[cost_key] <= p[cost_key])
            strictly_better = (q[score_key] > p[score_key]) or (q[cost_key] < p[cost_key])
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p[cost_key])
    return front


def _cross(o, a, b):
    """2D cross product of (a-o) and (b-o); >0 = left/CCW turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _upper_convex_hull(points, cost_key="gflops", score_key="miou"):
    """
    Upper convex hull in (cost, score) space — the concave, diminishing-
    returns boundary. Points below the segment through their cost-
    neighbors on the front are dropped even if individually non-dominated.
    Monotone-chain (Andrew's algorithm), upper half only.
    """
    pts = sorted(points, key=lambda p: (p[cost_key], -p[score_key]))
    hull = []
    for p in pts:
        xy = (p[cost_key], p[score_key])
        while len(hull) >= 2:
            o_xy = (hull[-2][cost_key], hull[-2][score_key])
            a_xy = (hull[-1][cost_key], hull[-1][score_key])
            if _cross(o_xy, a_xy, xy) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _fmt(v):
    if v is None or v != v:
        return "nan"
    return f"{v:.4f}"


def _print_pareto_table(title, pts):
    print(f"\n{title}")
    if not pts:
        print("  (no valid points — all miou or gflops values were NaN)")
        return
    print(f"  {'resolution':>12} {'tokens/mod':>12} {'gflops':>12} {'miou':>8}")
    print("  " + "-" * 48)
    for p in pts:
        print(f"  {str(int(p['res']))+' m':>12} {p['tokens_per_modality']:>12} "
              f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}")


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="RAMEN resolution sweep on HLS BurnScars")
parser.add_argument("--ckpt", type=str, required=True,
                    help="RAMEN checkpoint to evaluate at every resolution.")
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/hls_burn_scars")
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--resolutions", type=float, nargs="+", default=None,
                    help="Explicit list of `res` values (m/px) to sweep. "
                         "Default: 30, 60, ..., 300 (step 30, 10 values) — "
                         "kept in multiples of HLS's native 30m GSD.")

parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of profiled forward passes per resolution (mean).")

# RAMEN architecture args — fixed across the sweep, MUST match the
# checkpoint's training config EXCEPT `res` itself, which is swept.
# Defaults here mirror script_train_burnscars_baselines.py's own RAMEN
# defaults, not copied from the Sen1Floods11 sweep script (which used
# different, mismatched defaults) — always double check against your
# actual training run regardless.
parser.add_argument("--ramen_embed_dim",   type=int, default=384)
parser.add_argument("--ramen_depth",       type=int, default=12)
parser.add_argument("--ramen_num_heads",   type=int, default=8)
parser.add_argument("--ramen_input_res",   type=float, default=30.0,
                    help="HLS is standardized to 30m GSD (harmonized "
                         "Landsat/Sentinel), NOT Sentinel-2's native 10m.")
parser.add_argument("--ramen_window_size", type=int, default=128,
                    help="MUST match the checkpoint's training window size.")
parser.add_argument("--ramen_stride",      type=int, default=96,
                    help="Sliding-window stride for full-image eval.")
parser.add_argument("--ramen_output_layers", type=int, nargs="+",
                    default=[2, 5, 8, 11])
parser.add_argument("--ramen_decoder_channels", type=int, default=256)

parser.add_argument("--ramen_config", type=str, default=None,
                    help="Optional YAML setting window_size/embed_dim/etc "
                         "(same format as script_train_burnscars_baselines.py's "
                         "--ramen_config). A `res` key in this file is "
                         "IGNORED — resolution is controlled by "
                         "--resolutions, since sweeping it is this "
                         "script's entire point.")

args = parser.parse_args()

if args.resolutions is None:
    args.resolutions = [float(r) for r in range(30, 301, 30)]  # 30..300 step 30

# ── apply RAMEN config (everything except `res`) ───────────────────────────
if args.ramen_config is not None:
    key_map = {
        "input_size": "ramen_window_size", "input_res": "ramen_input_res",
        "embed_dim": "ramen_embed_dim", "depth": "ramen_depth",
        "num_heads": "ramen_num_heads", "stride": "ramen_stride",
    }
    import sys
    explicit = {
        tok[2:].split("=")[0].replace("-", "_")
        for tok in sys.argv[1:] if tok.startswith("--")
    }
    with open(args.ramen_config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, val in cfg.items():
        if key == "res":
            print(f"[INFO] Ignoring 'res' in {args.ramen_config} — this "
                  f"script sweeps resolution via --resolutions instead.")
            continue
        dest = key_map.get(key)
        if dest is None:
            print(f"[WARNING] Unrecognized key '{key}' in {args.ramen_config} "
                  f"— ignoring. Known keys: {sorted(key_map)} (+ 'res', ignored here).")
            continue
        if dest in explicit:
            print(f"[INFO] '{key}' in {args.ramen_config} ignored — "
                  f"--{dest} was explicitly set on the command line.")
            continue
        print(f"[INFO] {args.ramen_config}: {key}={val} -> --{dest}")
        setattr(args, dest, val)

if args.ramen_stride > args.ramen_window_size:
    print(f"[WARNING] --ramen_stride ({args.ramen_stride}) exceeds "
          f"--ramen_window_size ({args.ramen_window_size}); clamping to "
          f"window_size (non-overlapping tiling).")
    args.ramen_stride = args.ramen_window_size


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  RAMEN Resolution Sweep — HLS BurnScars")
print(f"  Checkpoint:   {args.ckpt}")
print(f"  Window size:  {args.ramen_window_size}x{args.ramen_window_size}")
print(f"  Stride:       {args.ramen_stride}")
print(f"  Resolutions:  {args.resolutions}")
print(f"{'='*60}\n")


# =============================================================================
# TEST DATASET (built once, reused across every resolution)
# =============================================================================

test_ds = BurnScarsBaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers,
    collate_fn=burnscars_collate,
    pin_memory=True,
    persistent_workers=args.num_workers > 0,
    prefetch_factor=2 if args.num_workers > 0 else None,
)
print(f"[Eval] Test set: {len(test_ds)} samples\n")

# Fixed batch pool for FLOPs profiling — same input geometry reused
# across every resolution so the comparison is apples-to-apples.
device = "cuda" if torch.cuda.is_available() else "cpu"
_flops_raw = []
for b in test_loader:
    _flops_raw.append(_to_device(b, device))
    if len(_flops_raw) >= args.flops_n + 1:  # +1 warmup
        break


# =============================================================================
# SWEEP
# =============================================================================

results = []  # list of dicts: {res, effective_size, tokens_per_modality, gflops, miou}

for res in args.resolutions:
    effective_size = int(args.ramen_window_size * (args.ramen_input_res / res))
    effective_size = max(effective_size, 1)
    tokens_per_modality = effective_size ** 2

    print(f"\n{'─'*60}")
    print(f"  res = {res:.0f} m/px  "
          f"(effective grid {effective_size}x{effective_size} = "
          f"{tokens_per_modality} tokens/modality)")
    print(f"{'─'*60}")

    # ── Build model at this resolution ──────────────────────────────────
    model = build_ramen_upernet(
        input_bands=RAMEN_INPUT_BANDS,
        wavelengths=RAMEN_WAVELENGTHS,
        num_classes=NUM_CLASSES,
        input_size=args.ramen_window_size,
        embed_dim=args.ramen_embed_dim,
        depth=args.ramen_depth,
        num_heads=args.ramen_num_heads,
        input_res=args.ramen_input_res,
        res=res,
        output_layers=tuple(args.ramen_output_layers),
        decoder_channels=args.ramen_decoder_channels,
    )

    # ── Load the SAME checkpoint's weights — safe at any res, see module
    #    docstring: pos_embed is a non-persistent buffer, never saved ───
    trainer_module = BaselineTrainer.load_from_checkpoint(
        args.ckpt,
        strict=True,
        model=model,
        modality=MODALITY_KEY,   # "hls" — single modality, matches the
                                   # dataset's own key directly
        temporal=False,
        task="burnscars",
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        window_size=args.ramen_window_size,
        window_stride=args.ramen_stride,
    )
    # No RAMENInputAdapter needed: RAMENUPerNet's modality is "hls",
    # matching batch["image"]["hls"] directly — trainer_module.model
    # stays as the loaded RAMENUPerNet itself.
    trainer_module.eval()

    # ── FLOPs (full sliding-window pass over the full native image) ─────
    eval_model = trainer_module.model.to(device).eval()

    def fwd(b, m=eval_model):
        return sliding_window_inference(
            m, b["image"],
            window_size=args.ramen_window_size,
            stride=args.ramen_stride,
            num_classes=NUM_CLASSES,
        )

    gflops = measure_gflops_forward(fwd, _flops_raw, device, n_warmup=1)
    print(f"  GFLOPs/forward (bs=1, full image, sliding-window): {gflops:.2f}")

    # ── mIoU (full test split, single modality, no dropping) ────────────
    pl_trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        precision="bf16-mixed",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    test_results = pl_trainer.test(trainer_module, test_loader, verbose=False)
    metrics = test_results[0] if test_results else {}
    miou = metrics.get("test_mIoU", float("nan"))
    print(f"  test mIoU: {miou:.4f}")

    results.append({
        "res": res,
        "effective_size": effective_size,
        "tokens_per_modality": tokens_per_modality,
        "gflops": gflops,
        "miou": miou,
    })

    del model, trainer_module, eval_model
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*70}")
print(f"  RAMEN RESOLUTION SWEEP SUMMARY — {args.xp_name}")
print(f"{'='*70}")
print(f"  {'Resolution':<14}{'Tokens/mod':<14}{'GFLOPs':<12}{'mIoU':<10}")
print(f"  {'─'*50}")
for r in results:
    print(f"  {str(int(r['res']))+' m':<14}"
          f"{r['tokens_per_modality']:<14}"
          f"{r['gflops']:<12.2f}"
          f"{r['miou']:<10.4f}")
print(f"{'='*70}\n")

# Also print in the exact "resolution XX: GFLOPS: YY mIoU: ZZ" phrasing
print("Compact summary:")
for r in results:
    print(f"resolution {int(r['res'])}: GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}")


# =============================================================================
# >>> PARETO_HULL: Pareto front + convex hull over ALL swept resolutions
# =============================================================================
# Every point already has a real measured GFLOPs (this sweep never skips
# it), so this is a genuine GFLOPs-vs-mIoU curve, no proxy/fallback needed.

pareto_front = _pareto_front(results, cost_key="gflops", score_key="miou")
convex_hull = _upper_convex_hull(pareto_front, cost_key="gflops", score_key="miou")
convex_hull.sort(key=lambda p: p["gflops"])

print(f"\n{'='*78}")
print(f"PARETO FRONT + CONVEX HULL — GFLOPs x mIoU  ({args.xp_name})")
print(f"{'='*78}")
_print_pareto_table(
    f"Pareto front ({len(pareto_front)} points, all non-dominated):",
    pareto_front)
_print_pareto_table(
    f"Convex hull / efficient frontier ({len(convex_hull)} points, "
    f"the ones actually worth plotting/using):",
    convex_hull)


# =============================================================================
# WRITE RESULTS
# =============================================================================

def _jf(v):
    v = float(v)
    return None if v != v else v

json_results = [
    {**r, "gflops": _jf(r["gflops"]), "miou": _jf(r["miou"])}
    for r in results
]
json_pareto_front = [
    {**p, "gflops": _jf(p["gflops"]), "miou": _jf(p["miou"])}
    for p in pareto_front
]
json_convex_hull = [
    {**p, "gflops": _jf(p["gflops"]), "miou": _jf(p["miou"])}
    for p in convex_hull
]

# >>> ADDED: this script previously wrote only a .txt file. A JSON output
# is now also written (matching the other GFLOPs sweep scripts) so
# recompute_pareto_hull.py and any downstream tooling can consume it
# directly via its "pareto_front" auto-detection path.
json_dir = "./scripts_GFLOPS/results/"
os.makedirs(json_dir, exist_ok=True)
json_path = os.path.join(json_dir, f"{args.xp_name}_resolution_sweep.json")
with open(json_path, "w") as f:
    json.dump(
        {
            "experiment": args.xp_name,
            "model": "ramen",
            "dataset": "burnscars",
            "checkpoint": args.ckpt,
            "ramen_window_size": args.ramen_window_size,
            "ramen_stride": args.ramen_stride,
            "ramen_input_res": args.ramen_input_res,
            "resolutions": args.resolutions,
            "flops_method": "torch.utils.flop_counter.FlopCounterMode",
            "flops_n": args.flops_n,
            "results": json_results,
            # >>> PARETO_HULL
            "pareto_front": json_pareto_front,
            "convex_hull": json_convex_hull,
        },
        f, indent=2,
    )
print(f"[Sweep] JSON results saved to {json_path}")

out_path = f"./results_{args.xp_name}_resolution_sweep.txt"
with open(out_path, "w") as f:
    f.write(f"RAMEN Resolution Sweep — HLS BurnScars — {args.xp_name}\n")
    f.write(f"Checkpoint: {args.ckpt}\n")
    f.write(f"Window size: {args.ramen_window_size}, stride: {args.ramen_stride}\n\n")
    f.write(f"{'Resolution':<14}{'Tokens/mod':<14}{'GFLOPs':<12}{'mIoU':<10}\n")
    f.write(f"{'─'*50}\n")
    for r in results:
        f.write(f"{str(int(r['res']))+' m':<14}"
                f"{r['tokens_per_modality']:<14}"
                f"{r['gflops']:<12.2f}"
                f"{r['miou']:<10.4f}\n")
    f.write("\nCompact summary:\n")
    for r in results:
        f.write(f"resolution {int(r['res'])}: GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}\n")

    # >>> PARETO_HULL
    f.write(f"\n\n{'='*78}\n")
    f.write(f"PARETO FRONT + CONVEX HULL — GFLOPs x mIoU\n")
    f.write(f"{'='*78}\n")
    f.write(f"\nPareto front ({len(pareto_front)} points, all non-dominated):\n")
    f.write(f"  {'resolution':>12} {'tokens/mod':>12} {'gflops':>12} {'miou':>8}\n")
    f.write("  " + "-" * 48 + "\n")
    for p in pareto_front:
        f.write(f"  {str(int(p['res']))+' m':>12} {p['tokens_per_modality']:>12} "
                f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}\n")
    f.write(f"\nConvex hull / efficient frontier ({len(convex_hull)} points, "
            f"the ones actually worth plotting/using):\n")
    f.write(f"  {'resolution':>12} {'tokens/mod':>12} {'gflops':>12} {'miou':>8}\n")
    f.write("  " + "-" * 48 + "\n")
    for p in convex_hull:
        f.write(f"  {str(int(p['res']))+' m':>12} {p['tokens_per_modality']:>12} "
                f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}\n")

print(f"\n[Sweep] Results saved to {out_path}")
