"""
UniverSat Patch-Size / Output-Stride Sweep — Sen1Floods11
===========================================================

Sen1Floods11 twin of script_universat_sweep_burnscars.py: evaluates a
SINGLE trained UniverSat checkpoint over a set of (patch_size_m,
output_stride) points, measuring GFLOPs/forward (FlopCounterMode —
SDPA attention counted) and test mIoU at each. TRUE multimodal path:
the collate splits the dataset's merged "s2s1" [15,H,W] tensor into
{"optical": 13ch, "sar": 2ch}, and SAR channels use UniverSat's string
codes "VV"/"VH" (Encoding_<code> embeddings — NOT RAMEN's "asc_vv"
pol_map keys). Eval is always a single dense forward per image,
center-cropped per point to the largest side <= 512 divisible by
lcm(patch_px, output_stride).

The sweep is defined by the lists right below this docstring:

  PATCH_SIZES_M          -> each evaluated at DEFAULT_OUTPUT_STRIDE
  PATCH_STRIDE_COUPLES   -> (patch_size_m, [strides]) pairs

The union of both, deduplicated in order, is the run plan (printed at
startup before anything heavy runs).

GSD is 10 m here (vs 30 m for BurnScars), so the SAME pixel ladder
maps to different metre values: 80 m = 8 px is the training default,
the analogue of BurnScars' 240 m. Every multiple of 10 m is a valid
patch size (per-point center crops handle divisibility).

DEFAULT_OUTPUT_STRIDE MUST match the checkpoint's training stride so
the patch-size arm varies exactly one thing. NOTE: the universat_px
training config used output_stride=1 — if that is your checkpoint,
keep DEFAULT_OUTPUT_STRIDE = 1 below and expect a HEAVY patch arm
(every point pays the ~262k-query CA_Sub bill, ~110 TFLOP/forward,
and the mIoU pass over the test split dominates runtime). The
BurnScars grid showed strong eval-time stride-insensitivity, so
running the patch arm at a coarser stride is a defensible shortcut,
but then the patch curve is measured off the training anchor — say so
when reporting. For a stride-4-trained checkpoint set 4.

All other caveats from the BurnScars sweep apply verbatim (one-
checkpoint strict-safety on both knobs; off-training values measure
generalization around the training point; coarse strides genuinely
predict at coarser ground granularity, os * 10 m here; per-point eval
sides 512/504/500, recorded in the JSON; FlopCounterMode numbers must
never be mixed with torch.profiler-harness numbers). See that
script's docstring for the full discussion.

# >>> PARETO_HULL: after the sweep, this script also computes the Pareto
# front (mIoU maximize, GFLOPs minimize) over ALL swept points and reduces
# it to its convex hull (the efficient frontier — points where marginal
# mIoU gain per extra GFLOP is diminishing; a non-dominated point that
# still sits below the line connecting two better trade-off points is
# dropped). Both are printed, and saved into the same JSON/txt outputs
# the sweep already writes, under "pareto_front" / "convex_hull" keys.
# Every point already has a REAL measured gflops (this sweep never skips
# GFLOPs), so the hull is a genuine GFLOPs-vs-mIoU curve directly, no
# fallback needed. Point identity here is (patch_m, output_stride) —
# one axis fewer than the BurnScars twin, which also sweeps subpatch_px.

Usage
-----
    python script_universat_sweep_senflood.py \
        --ckpt ./checkpoints/senflood_baselines/bl_universat_px_universat-last.ckpt \
        --xp_name universat_sweep_senflood \
        --universat_size small
"""

# =============================================================================
# SWEEP DEFINITION — edit these lists
# =============================================================================

DEFAULT_OUTPUT_STRIDE = 4

# Sub-patch size (px) used for every entry of PATCH_SIZES_M and
# PATCH_STRIDE_COUPLES. MUST match the checkpoint's training
# subpatch_px so those two sweeps still vary exactly one thing each.
# 1 = pixel-level CA_Sub keys (matches the pre-sub-patching training
# convention used on BurnScars / the other coarse-GSD datasets).
DEFAULT_SUBPATCH_PX = 1

# Patch-size sweep (encoder density), each at DEFAULT_OUTPUT_STRIDE and
# DEFAULT_SUBPATCH_PX. Every multiple of 30 m works (per-point center
# crop handles divisibility); 30/60 m excluded as intractable/very heavy.
PATCH_SIZES_M = [
     10,20,40,80.0, 160.0, 320.0,640,1280
]

# Output-stride sweep(s) (decoder density): (patch_size_m, [strides]),
# each at DEFAULT_SUBPATCH_PX. Typically one couple anchored at the
# training patch size. os=1 is the ~105 TFLOP point — drop it here if
# runtime is a concern.
PATCH_STRIDE_COUPLES = [
    (40, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (20, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (80, [1, 2, 4, 8, 16,32,64,128,256,512]),

]

# Sub-patch sweep(s) (CA_Sub key density): (patch_size_m, output_stride,
# [subpatch_px]). Typically anchored at the training (patch, stride)
# point. subpatch_px must evenly divide patch_px at that patch size —
# invalid combinations are skipped (with a message) at run-plan build
# time, not silently dropped.
PATCH_SUBPATCH_COUPLES = [
    (40, DEFAULT_OUTPUT_STRIDE, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (80.0, DEFAULT_OUTPUT_STRIDE, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (20.0, DEFAULT_OUTPUT_STRIDE, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (40.0, 4, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (80.0, 4, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (160.0, 4, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (80.0, 8, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (80.0, 8, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (80.0, 8, [1, 2, 4, 8, 16,32,64,128,256,512]),
]

# =============================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils.datasets_baselines.utils_dataset_senflood_baselines import (
    Sen1Floods11BaselineDataset,
)
from training.Universat.universat_augmenter import build_universat_segmenter
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = Sen1Floods11BaselineDataset.NUM_CLASSES     # 2
IGNORE_INDEX = Sen1Floods11BaselineDataset.IGNORE_INDEX    # 255
NUM_S2_BANDS = Sen1Floods11BaselineDataset.NUM_S2_BANDS    # 13
NUM_S1_BANDS = Sen1Floods11BaselineDataset.NUM_S1_BANDS    # 2
SENFLOOD_GSD_M = 10.0
NATIVE_SIDE_PX = 512

S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

S2_WAVELENGTHS_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B8A": 864.7, "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
    "B12": 2202.4,
}

S1_BAND_NAMES = ["VV", "VH"]

# UniverSat metadata: optical embedded by continuous wavelength (adapter
# converts nm -> µm); SAR uses UniverSat's OWN string codes "VV"/"VH"
# (Encoding_<code> attribute names — NOT RAMEN's "asc_vv" pol_map keys).
UNIVERSAT_INPUT_BANDS = {
    "optical": S2_BAND_NAMES,
    "sar": S1_BAND_NAMES,
}
UNIVERSAT_WAVELENGTHS = {
    "optical": S2_WAVELENGTHS_NM,
    "sar": S1_BAND_NAMES,          # the names ARE the codes: ["VV", "VH"]
}


# =============================================================================
# COLLATE — per-point center crop of image dict AND target
# =============================================================================

def make_crop_collate(eval_side: int):
    """Split the dataset's merged "s2s1" [15,H,W] tensor into
    {"optical","sar"} (what UniverSatSegmenter consumes) AND center-crop
    image + target to eval_side — deterministic; degenerates to a plain
    split+stack at the native side. mIoU at that sweep point is computed
    over the crop. Band order: 13 S2 first, then VV/VH, matching the
    dataset's fixed merge order."""

    def _crop(t: torch.Tensor) -> torch.Tensor:
        H, W = t.shape[-2], t.shape[-1]
        top, left = (H - eval_side) // 2, (W - eval_side) // 2
        return t[..., top:top + eval_side, left:left + eval_side]

    def collate(batch):
        merged = torch.stack([_crop(s["image"]["s2s1"]) for s in batch])
        images = {
            "optical": merged[:, :NUM_S2_BANDS],
            "sar": merged[:, NUM_S2_BANDS:NUM_S2_BANDS + NUM_S1_BANDS],
        }
        targets  = torch.stack([_crop(s["target"]) for s in batch])
        metadata = [s["metadata"] for s in batch]
        return {"image": images, "target": targets, "metadata": metadata}

    return collate


# =============================================================================
# FLOPs MEASUREMENT — FlopCounterMode (counts SDPA attention)
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
    shape — the mean is a sanity check). Every pass is one dense
    forward at bs=1."""
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
    print(f"  {'patch':>7} {'os':>4} {'eval':>5} "
          f"{'gflops':>12} {'miou':>8}")
    print("  " + "-" * 44)
    for p in pts:
        print(f"  {int(p['patch_m']):>5} m {p['output_stride']:>4} "
              f"{p['eval_side']:>5} "
              f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}")


# =============================================================================
# ARGS (sweep definition lives in the lists at the top of the file)
# =============================================================================

parser = argparse.ArgumentParser(
    description="UniverSat (patch size x output stride) sweep on Sen1Floods11")
parser.add_argument("--ckpt", type=str, required=True,
                    help="UniverSat checkpoint to evaluate at every point.")
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--flops_n", type=int, default=3,
                    help="Number of counted forward passes per point (mean).")
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"],
                    help="MUST match the checkpoint's training size.")

args = parser.parse_args()

if not os.path.exists(args.ckpt):
    raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")


# =============================================================================
# BUILD THE RUN PLAN — union of the two lists, deduped in order
# =============================================================================

raw_points = [(pm, DEFAULT_OUTPUT_STRIDE) for pm in PATCH_SIZES_M]
for pm, strides in PATCH_STRIDE_COUPLES:
    raw_points.extend((pm, s) for s in strides)

seen = set()
valid_points = []   # (patch_m, patch_px, output_stride, eval_side)
for pm, os_ in raw_points:
    key = (float(pm), int(os_))
    if key in seen:
        continue
    seen.add(key)

    px = pm / SENFLOOD_GSD_M
    if abs(px - round(px)) > 1e-6:
        print(f"[SKIP] patch {pm} m is not an integer pixel count at "
              f"{SENFLOOD_GSD_M} m GSD ({px:.3f} px).")
        continue
    px = int(round(px))
    if os_ < 1:
        print(f"[SKIP] (patch {pm} m, os {os_}): output_stride < 1.")
        continue
    lcm = math.lcm(px, os_)
    eval_side = (NATIVE_SIDE_PX // lcm) * lcm
    if eval_side <= 0:
        print(f"[SKIP] (patch {pm} m, os {os_}): lcm(px={px}, os)={lcm} "
              f"exceeds the native side {NATIVE_SIDE_PX}.")
        continue
    valid_points.append((pm, px, os_, eval_side))

if not valid_points:
    raise ValueError("No valid sweep points after geometry checks.")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*76}")
print(f"  UniverSat (Patch x Stride) Sweep — Sen1Floods11")
print(f"  Checkpoint:    {args.ckpt}")
print(f"  Size:          {args.universat_size}")
print(f"  Modalities:    optical (13ch S2) + sar (VV/VH)")
print(f"  Eval:          single dense forward per image, per-point center crop")
print(f"  FLOPs:         FlopCounterMode (SDPA attention counted)")
print(f"  Run plan ({len(valid_points)} points):")
print(f"    {'patch':>7} {'px':>4} {'os':>4} {'eval':>5} "
      f"{'trunk tokens':>13} {'CA_Sub queries':>15} {'pred gran.':>11}")
for pm, px, os_, es in valid_points:
    print(f"    {int(pm):>5} m {px:>4} {os_:>4} {es:>5} "
          f"{(es // px) ** 2:>13} {(es // os_) ** 2:>15} "
          f"{int(os_ * SENFLOOD_GSD_M):>9} m")
print(f"{'='*76}\n")


# =============================================================================
# TEST DATASET (built once; per-point loaders re-wrap it with a crop collate)
# =============================================================================

test_ds = Sen1Floods11BaselineDataset(
    root_path=args.data_dir, mode="test",
    crop_size=None, augment=False,
)
print(f"[Eval] Test set: {len(test_ds)} samples\n")

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(eval_side: int) -> DataLoader:
    return DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_crop_collate(eval_side),
        pin_memory=True,
        persistent_workers=False,   # loaders are rebuilt per sweep point
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


# =============================================================================
# SWEEP
# =============================================================================

results = []  # dicts: {patch_m, patch_px, output_stride, eval_side,
              #         latent_side, latent_tokens, out_side, out_queries,
              #         pred_granularity_m, gflops, miou}

for patch_m, patch_px, os_, eval_side in valid_points:
    latent_side = eval_side // patch_px
    latent_tokens = latent_side ** 2
    out_side = eval_side // os_
    out_queries = out_side ** 2

    print(f"\n{'─'*76}")
    print(f"  patch = {patch_m:.0f} m ({patch_px} px), output_stride = {os_}, "
          f"eval side {eval_side}")
    print(f"  latent {latent_side}x{latent_side} = {latent_tokens} trunk "
          f"tokens; {out_side}x{out_side} = {out_queries} CA_Sub queries")
    print(f"{'─'*76}")

    # ── Per-point loader with matching center-crop collate ──────────────
    test_loader = make_loader(eval_side)

    _flops_raw = []
    for b in test_loader:
        _flops_raw.append(_to_device(b, device))
        if len(_flops_raw) >= args.flops_n + 1:  # +1 warmup
            break

    # ── Build the adapter at this (patch, stride) ───────────────────────
    # No learnable tensor is shaped by either knob — the same checkpoint
    # loads strict at every point (see module docstring).
    model = build_universat_segmenter(
        input_bands=UNIVERSAT_INPUT_BANDS,
        wavelengths=UNIVERSAT_WAVELENGTHS,
        num_classes=NUM_CLASSES,
        input_res={"optical": SENFLOOD_GSD_M, "sar": SENFLOOD_GSD_M},
        patch_size_m=patch_m,
        output_stride=os_,
        size=args.universat_size,
    )

    trainer_module = BaselineTrainer.load_from_checkpoint(
        args.ckpt,
        strict=True,
        model=model,
        modality="optical+sar",
        temporal=False,
        task="senflood",
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        window_size=None,          # always full (cropped) dense forward
        window_stride=None,
    )
    trainer_module.eval()

    # ── FLOPs — one dense forward over the cropped image ────────────────
    eval_model = trainer_module.model.to(device).eval()

    def fwd(b, m=eval_model):
        return m(b["image"])

    gflops = measure_gflops_forward(fwd, _flops_raw, device, n_warmup=1)
    print(f"  GFLOPs/forward (bs=1, dense, side {eval_side}): {gflops:.2f}")

    # ── mIoU (full test split on the same crop, no dropping) ────────────
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
    print(f"  test mIoU (on {eval_side}x{eval_side} center crop): {miou:.4f}")

    results.append({
        "patch_m": patch_m,
        "patch_px": patch_px,
        "output_stride": os_,
        "eval_side": eval_side,
        "latent_side": latent_side,
        "latent_tokens": latent_tokens,
        "out_side": out_side,
        "out_queries": out_queries,
        "pred_granularity_m": os_ * SENFLOOD_GSD_M,
        "gflops": gflops,
        "miou": miou,
    })

    del model, trainer_module, eval_model, test_loader, _flops_raw
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*84}")
print(f"  UNIVERSAT (PATCH x STRIDE) SWEEP SUMMARY — {args.xp_name}")
print(f"{'='*84}")
print(f"  {'Patch':<9}{'(px)':<6}{'os':<5}{'Eval':<7}{'Trunk tok':<11}"
      f"{'Queries':<10}{'GFLOPs':<12}{'mIoU':<10}")
print(f"  {'─'*72}")
for r in results:
    print(f"  {str(int(r['patch_m']))+' m':<9}"
          f"{r['patch_px']:<6}"
          f"{r['output_stride']:<5}"
          f"{r['eval_side']:<7}"
          f"{r['latent_tokens']:<11}"
          f"{r['out_queries']:<10}"
          f"{r['gflops']:<12.2f}"
          f"{r['miou']:<10.4f}")
print(f"{'='*84}\n")

print("Compact summary:")
for r in results:
    print(f"patch {int(r['patch_m'])} os {r['output_stride']}: "
          f"GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}")


# =============================================================================
# >>> PARETO_HULL: Pareto front + convex hull over ALL swept points
# =============================================================================
# Every point already has a real measured GFLOPs (this sweep never skips
# or light-samples it), so this is a genuine GFLOPs-vs-mIoU curve, no
# proxy/fallback needed like in the density-eval scripts.

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

json_dir = "./scripts_GFLOPS/results/"
os.makedirs(json_dir, exist_ok=True)
json_path = os.path.join(json_dir, f"{args.xp_name}_patch_stride_sweep.json")
with open(json_path, "w") as f:
    json.dump(
        {
            "experiment": args.xp_name,
            "model": "universat",
            "dataset": "senflood",
            "checkpoint": args.ckpt,
            "universat_size": args.universat_size,
            "default_output_stride": DEFAULT_OUTPUT_STRIDE,
            "patch_sizes_m": PATCH_SIZES_M,
            "patch_stride_couples": [
                {"patch_m": pm, "strides": strides}
                for pm, strides in PATCH_STRIDE_COUPLES
            ],
            "eval_mode": {"mode": "dense_center_crop",
                          "native_side": NATIVE_SIDE_PX,
                          "note": "per-point eval_side in results; mIoU and "
                                  "GFLOPs are computed on that crop"},
            "flops_method": "torch.utils.flop_counter.FlopCounterMode",
            "gsd_m": SENFLOOD_GSD_M,
            "flops_n": args.flops_n,
            "results": json_results,
            # >>> PARETO_HULL
            "pareto_front": json_pareto_front,
            "convex_hull": json_convex_hull,
        },
        f, indent=2,
    )
print(f"[Sweep] JSON results saved to {json_path}")

out_path = f"./results_{args.xp_name}_patch_stride_sweep.txt"
with open(out_path, "w") as f:
    f.write(f"UniverSat (Patch x Stride) Sweep — Sen1Floods11 — {args.xp_name}\n")
    f.write(f"Checkpoint: {args.ckpt}\n")
    f.write(f"Size: {args.universat_size}; "
            f"default output_stride: {DEFAULT_OUTPUT_STRIDE}\n")
    f.write(f"Eval: single dense forward, per-point center crop "
            f"(eval side in table)\n")
    f.write(f"FLOPs: FlopCounterMode (SDPA attention counted)\n\n")
    f.write(f"{'Patch':<9}{'(px)':<6}{'os':<5}{'Eval':<7}{'Trunk tok':<11}"
            f"{'Queries':<10}{'GFLOPs':<12}{'mIoU':<10}\n")
    f.write(f"{'─'*72}\n")
    for r in results:
        f.write(f"{str(int(r['patch_m']))+' m':<9}"
                f"{r['patch_px']:<6}"
                f"{r['output_stride']:<5}"
                f"{r['eval_side']:<7}"
                f"{r['latent_tokens']:<11}"
                f"{r['out_queries']:<10}"
                f"{r['gflops']:<12.2f}"
                f"{r['miou']:<10.4f}\n")
    f.write("\nCompact summary:\n")
    for r in results:
        f.write(f"patch {int(r['patch_m'])} os {r['output_stride']}: "
                f"GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}\n")

    # >>> PARETO_HULL
    f.write(f"\n\n{'='*78}\n")
    f.write(f"PARETO FRONT + CONVEX HULL — GFLOPs x mIoU\n")
    f.write(f"{'='*78}\n")
    f.write(f"\nPareto front ({len(pareto_front)} points, all non-dominated):\n")
    f.write(f"  {'patch':>7} {'os':>4} {'eval':>5} "
            f"{'gflops':>12} {'miou':>8}\n")
    f.write("  " + "-" * 44 + "\n")
    for p in pareto_front:
        f.write(f"  {int(p['patch_m']):>5} m {p['output_stride']:>4} "
                f"{p['eval_side']:>5} "
                f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}\n")
    f.write(f"\nConvex hull / efficient frontier ({len(convex_hull)} points, "
            f"the ones actually worth plotting/using):\n")
    f.write(f"  {'patch':>7} {'os':>4} {'eval':>5} "
            f"{'gflops':>12} {'miou':>8}\n")
    f.write("  " + "-" * 44 + "\n")
    for p in convex_hull:
        f.write(f"  {int(p['patch_m']):>5} m {p['output_stride']:>4} "
                f"{p['eval_side']:>5} "
                f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}\n")

print(f"\n[Sweep] Results saved to {out_path}")
