"""
UniverSat Patch-Size / Output-Stride / Sub-Patch Sweep — HLS BurnScars
========================================================================

Merged sweep: evaluates a SINGLE trained UniverSat checkpoint over a set
of (patch_size_m, output_stride, subpatch_px) points, measuring
GFLOPs/forward (FlopCounterMode — SDPA attention counted) and test mIoU
at each. Eval is always a single dense forward per image, center-cropped
per point to the largest side <= 512 divisible by
lcm(patch_px, output_stride, subpatch_px).

The sweep is defined by the three groups of lists right below this
docstring:

  PATCH_SIZES_M          -> each evaluated at DEFAULT_OUTPUT_STRIDE and
                            DEFAULT_SUBPATCH_PX (the encoder-density /
                            patch-size sweep)
  PATCH_STRIDE_COUPLES   -> (patch_size_m, [strides]) pairs, each stride
                            evaluated at that patch size and
                            DEFAULT_SUBPATCH_PX (the decoder-density /
                            output-stride sweep — typically anchored at
                            the training patch size)
  PATCH_SUBPATCH_COUPLES -> (patch_size_m, output_stride, [subpatch_px])
                            triples, each subpatch_px evaluated at that
                            patch size and stride (the CA_Sub-key-density
                            sweep — typically anchored at the training
                            patch size / output_stride)

The union of all three, deduplicated in order, is the run plan (printed
at startup before anything heavy happens).

The three knobs and what they cost:
  - patch_size_m  -> latent grid (side/patch_px)^2 = SA-trunk tokens.
                     Quadratic trunk cost; minority of the bill at
                     coarse patches.
  - output_stride -> CA_Sub queries (side/os)^2 against sub-patch tokens
                     (dense cross-attention). The DOMINANT cost term for
                     most of the sweep, ~1/os^2 — the knob that moves
                     UniverSat's compute by orders of magnitude.
  - subpatch_px   -> CA_Sub KEYS. Pixels are grouped into subpatch_px x
                     subpatch_px sub-patches before the S1 axis; the
                     CA_Sub keys are (side/subpatch_px)^2 sub-patch
                     tokens rather than (side)^2 raw pixels. Quadratic
                     in 1/subpatch_px on the key side of CA_Sub — at
                     subpatch_px=1 (pixel-level keys, the setting used
                     on the coarser-GSD datasets) this term is largest;
                     each doubling of subpatch_px cuts CA_Sub key count
                     4x. At HLS's 30 m GSD this axis matters far less
                     than at VHR (0.5 m) -- included here mainly for
                     completeness / cross-dataset comparability, not
                     because BurnScars is expected to be cost-bound by
                     it the way xView2 is.

Why one checkpoint covers every point: nothing learnable is shaped by
any of the three knobs (positions/scale are forward-time inputs; the
head is a per-token 1x1 conv), so rebuilding at any (patch, stride,
subpatch) and loading the same weights is strict-safe. CAVEATS for
reporting: the checkpoint was trained at ONE grid point, so off-training
values on any axis mix architecture capability with generalization from
training conditioning; and coarse strides genuinely predict at coarser
ground granularity (os * GSD metres) before the trainer's bilinear
upsample, so their mIoU drop conflates generalization with legitimate
resolution loss. Say all of this in the caption.

PROTOCOL NOTE: per-point eval sides may be 512, 504, or 500 (largest
valid center crop; recorded per point in the JSON). Up to ~4.6% of
image area differs between points — fine for trend curves; quote
headline numbers from exact-512 points.

FLOPs: torch.utils.flop_counter.FlopCounterMode, NOT
torch.profiler(with_flops=True) — the profiler has no formulas for
fused scaled_dot_product_attention kernels and silently drops every
attention term (both the trunk N^2 and the dominant CA_Sub cost). Do
not mix these numbers with profiler-harness results anywhere in the
paper; the JSON records flops_method. The same undercount affects any
SDPA-based model measured with the profiler harness elsewhere
(ViT / RAMEN / the training scripts' GFLOPs) — re-measure those with
FlopCounterMode before putting them in the same table.

Heavy points to watch: fine patches (90 m -> 28k trunk tokens), fine
strides (os=1 -> 262k dense CA_Sub queries, ~105 TFLOP/forward at patch
240), and subpatch_px=1 combined with a fine stride (largest CA_Sub key
count). Both should fit an eval-only bf16 bs=1 forward, but they
dominate runtime; remove them from the lists if needed — there is no
windowed fallback by design.

# >>> PARETO_HULL: after the sweep, this script also computes the Pareto
# front (mIoU maximize, GFLOPs minimize) over ALL swept points and reduces
# it to its convex hull (the efficient frontier — points where marginal
# mIoU gain per extra GFLOP is diminishing; a non-dominated point that
# still sits below the line connecting two better trade-off points is
# dropped). Both are printed, and saved into the same JSON/txt outputs
# the sweep already writes, under "pareto_front" / "convex_hull" keys.
# Every point already has a REAL measured gflops (this sweep never skips
# GFLOPs — unlike the Sen1Floods11/BurnScars density-eval scripts, there
# is no light/proxy mode here), so the hull is a genuine GFLOPs-vs-mIoU
# curve directly, no fallback needed.

Usage
-----
    python script_universat_sweep_burnscars.py \
        --ckpt ./checkpoints/burnscars_baselines/bl_universat_universat-last.ckpt \
        --xp_name universat_sweep_burnscars \
        --universat_size small
"""

# =============================================================================
# SWEEP DEFINITION — edit these lists
# =============================================================================

# Stride used for every entry of PATCH_SIZES_M. MUST match the
# checkpoint's training output_stride so the patch-size sweep varies
# exactly one thing.
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
     120.0,240.0, 420.0, 480.0,960.0, 1920.0
]

# Output-stride sweep(s) (decoder density): (patch_size_m, [strides]),
# each at DEFAULT_SUBPATCH_PX. Typically one couple anchored at the
# training patch size. os=1 is the ~105 TFLOP point — drop it here if
# runtime is a concern.
PATCH_STRIDE_COUPLES = [
    (240.0, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (480.0, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (960.0, [1, 2, 4, 8, 16,32,64,128,256,512]),

]

# Sub-patch sweep(s) (CA_Sub key density): (patch_size_m, output_stride,
# [subpatch_px]). Typically anchored at the training (patch, stride)
# point. subpatch_px must evenly divide patch_px at that patch size —
# invalid combinations are skipped (with a message) at run-plan build
# time, not silently dropped.
PATCH_SUBPATCH_COUPLES = [
    (120.0, DEFAULT_OUTPUT_STRIDE, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (240.0, DEFAULT_OUTPUT_STRIDE, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (480.0, DEFAULT_OUTPUT_STRIDE, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (120.0, 4, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (240.0, 4, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (480.0, 4, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (120.0, 8, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (240.0, 8, [1, 2, 4, 8, 16,32,64,128,256,512]),
    (480.0, 8, [1, 2, 4, 8, 16,32,64,128,256,512]),
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

from training.utils.datasets_baselines.utils_dataset_burnscars_baselines import (
    BurnScarsBaselineDataset,
)
from training.Universat.universat_augmenter import build_universat_segmenter
from training.trainer_baselines import BaselineTrainer


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_CLASSES  = BurnScarsBaselineDataset.NUM_CLASSES    # 2
IGNORE_INDEX = BurnScarsBaselineDataset.IGNORE_INDEX   # 255
MODALITY_KEY = "hls"
HLS_GSD_M    = 30.0
NATIVE_SIDE_PX = 512

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

UNIVERSAT_INPUT_BANDS = {MODALITY_KEY: BurnScarsBaselineDataset.HLS_BANDS}
UNIVERSAT_WAVELENGTHS = {MODALITY_KEY: HLS_WAVELENGTHS_NM}


# =============================================================================
# COLLATE — per-point center crop of image dict AND target
# =============================================================================

def make_crop_collate(eval_side: int):
    """Center-crop every modality tensor and the target to eval_side —
    deterministic; degenerates to a plain stack at the native side.
    mIoU at that sweep point is computed over the crop."""

    def _crop(t: torch.Tensor) -> torch.Tensor:
        H, W = t.shape[-2], t.shape[-1]
        top, left = (H - eval_side) // 2, (W - eval_side) // 2
        return t[..., top:top + eval_side, left:left + eval_side]

    def collate(batch):
        images = {}
        sensor_keys = list(batch[0]["image"].keys())
        for key in sensor_keys:
            images[key] = torch.stack([_crop(s["image"][key]) for s in batch])
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
    print(f"  {'patch':>7} {'os':>4} {'sp':>4} {'eval':>5} "
          f"{'gflops':>12} {'miou':>8}")
    print("  " + "-" * 48)
    for p in pts:
        print(f"  {int(p['patch_m']):>5} m {p['output_stride']:>4} "
              f"{p['subpatch_px']:>4} {p['eval_side']:>5} "
              f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}")


# =============================================================================
# ARGS (sweep definition lives in the lists at the top of the file)
# =============================================================================

parser = argparse.ArgumentParser(
    description="UniverSat (patch size x output stride x subpatch) sweep on HLS BurnScars")
parser.add_argument("--ckpt", type=str, required=True,
                    help="UniverSat checkpoint to evaluate at every point.")
parser.add_argument("--xp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/hls_burn_scars")
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
# BUILD THE RUN PLAN — union of the three lists, deduped in order
# =============================================================================

raw_points = [(pm, DEFAULT_OUTPUT_STRIDE, DEFAULT_SUBPATCH_PX) for pm in PATCH_SIZES_M]
for pm, strides in PATCH_STRIDE_COUPLES:
    raw_points.extend((pm, s, DEFAULT_SUBPATCH_PX) for s in strides)
for pm, os_, subpatches in PATCH_SUBPATCH_COUPLES:
    raw_points.extend((pm, os_, sp) for sp in subpatches)

seen = set()
valid_points = []   # (patch_m, patch_px, output_stride, subpatch_px, eval_side)
for pm, os_, sp in raw_points:
    key = (float(pm), int(os_), int(sp))
    if key in seen:
        continue
    seen.add(key)

    px = pm / HLS_GSD_M
    if abs(px - round(px)) > 1e-6:
        print(f"[SKIP] patch {pm} m is not an integer pixel count at "
              f"{HLS_GSD_M} m GSD ({px:.3f} px).")
        continue
    px = int(round(px))
    if os_ < 1:
        print(f"[SKIP] (patch {pm} m, os {os_}, sp {sp}): output_stride < 1.")
        continue
    if sp < 1:
        print(f"[SKIP] (patch {pm} m, os {os_}, sp {sp}): subpatch_px < 1.")
        continue
    if px % sp:
        print(f"[SKIP] (patch {pm} m -> {px} px, os {os_}, sp {sp}): patch_px "
              f"must be a multiple of subpatch_px (S1 axis groups whole "
              f"sub-patches into each patch).")
        continue
    lcm = math.lcm(px, os_)
    lcm = math.lcm(lcm, sp)
    eval_side = (NATIVE_SIDE_PX // lcm) * lcm
    if eval_side <= 0:
        print(f"[SKIP] (patch {pm} m, os {os_}, sp {sp}): "
              f"lcm(px={px}, os={os_}, sp={sp})={lcm} exceeds the native "
              f"side {NATIVE_SIDE_PX}.")
        continue
    valid_points.append((pm, px, os_, sp, eval_side))

if not valid_points:
    raise ValueError("No valid sweep points after geometry checks.")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*92}")
print(f"  UniverSat (Patch x Stride x Sub-patch) Sweep — HLS BurnScars")
print(f"  Checkpoint:    {args.ckpt}")
print(f"  Size:          {args.universat_size}")
print(f"  Eval:          single dense forward per image, per-point center crop")
print(f"  FLOPs:         FlopCounterMode (SDPA attention counted)")
print(f"  Run plan ({len(valid_points)} points):")
print(f"    {'patch':>7} {'px':>4} {'os':>4} {'sp':>4} {'eval':>5} "
      f"{'trunk tokens':>13} {'CA_Sub queries':>15} {'CA_Sub keys':>12} "
      f"{'pred gran.':>11}")
for pm, px, os_, sp, es in valid_points:
    print(f"    {int(pm):>5} m {px:>4} {os_:>4} {sp:>4} {es:>5} "
          f"{(es // px) ** 2:>13} {(es // os_) ** 2:>15} "
          f"{(es // sp) ** 2:>12} "
          f"{int(os_ * HLS_GSD_M):>9} m")
print(f"{'='*92}\n")


# =============================================================================
# TEST DATASET (built once; per-point loaders re-wrap it with a crop collate)
# =============================================================================

test_ds = BurnScarsBaselineDataset(
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

results = []  # dicts: {patch_m, patch_px, output_stride, subpatch_px,
              #         eval_side, latent_side, latent_tokens, out_side,
              #         out_queries, subpatch_side, subpatch_keys,
              #         pred_granularity_m, gflops, miou}

for patch_m, patch_px, os_, sp, eval_side in valid_points:
    latent_side = eval_side // patch_px
    latent_tokens = latent_side ** 2
    out_side = eval_side // os_
    out_queries = out_side ** 2
    subpatch_side = eval_side // sp
    subpatch_keys = subpatch_side ** 2

    print(f"\n{'─'*92}")
    print(f"  patch = {patch_m:.0f} m ({patch_px} px), output_stride = {os_}, "
          f"subpatch_px = {sp}, eval side {eval_side}")
    print(f"  latent {latent_side}x{latent_side} = {latent_tokens} trunk "
          f"tokens; {out_side}x{out_side} = {out_queries} CA_Sub queries; "
          f"{subpatch_side}x{subpatch_side} = {subpatch_keys} CA_Sub keys")
    print(f"{'─'*92}")

    # ── Per-point loader with matching center-crop collate ──────────────
    test_loader = make_loader(eval_side)

    _flops_raw = []
    for b in test_loader:
        _flops_raw.append(_to_device(b, device))
        if len(_flops_raw) >= args.flops_n + 1:  # +1 warmup
            break

    # ── Build the adapter at this (patch, stride, subpatch) ─────────────
    # No learnable tensor is shaped by any of the three knobs — the same
    # checkpoint loads strict at every point (see module docstring).
    model = build_universat_segmenter(
        input_bands=UNIVERSAT_INPUT_BANDS,
        wavelengths=UNIVERSAT_WAVELENGTHS,
        num_classes=NUM_CLASSES,
        input_res={MODALITY_KEY: HLS_GSD_M},
        patch_size_m=patch_m,
        output_stride=os_,
        size=args.universat_size,
        subpatch_px=sp,
    )

    trainer_module = BaselineTrainer.load_from_checkpoint(
        args.ckpt,
        strict=True,
        model=model,
        modality="hls",
        temporal=False,
        task="burnscars",
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
        "subpatch_px": sp,
        "eval_side": eval_side,
        "latent_side": latent_side,
        "latent_tokens": latent_tokens,
        "out_side": out_side,
        "out_queries": out_queries,
        "subpatch_side": subpatch_side,
        "subpatch_keys": subpatch_keys,
        "pred_granularity_m": os_ * HLS_GSD_M,
        "gflops": gflops,
        "miou": miou,
    })

    del model, trainer_module, eval_model, test_loader, _flops_raw
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*100}")
print(f"  UNIVERSAT (PATCH x STRIDE x SUBPATCH) SWEEP SUMMARY — {args.xp_name}")
print(f"{'='*100}")
print(f"  {'Patch':<9}{'(px)':<6}{'os':<5}{'sp':<5}{'Eval':<7}{'Trunk tok':<11}"
      f"{'Queries':<10}{'Keys':<10}{'GFLOPs':<12}{'mIoU':<10}")
print(f"  {'─'*88}")
for r in results:
    print(f"  {str(int(r['patch_m']))+' m':<9}"
          f"{r['patch_px']:<6}"
          f"{r['output_stride']:<5}"
          f"{r['subpatch_px']:<5}"
          f"{r['eval_side']:<7}"
          f"{r['latent_tokens']:<11}"
          f"{r['out_queries']:<10}"
          f"{r['subpatch_keys']:<10}"
          f"{r['gflops']:<12.2f}"
          f"{r['miou']:<10.4f}")
print(f"{'='*100}\n")

print("Compact summary:")
for r in results:
    print(f"patch {int(r['patch_m'])} os {r['output_stride']} sp {r['subpatch_px']}: "
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
json_path = os.path.join(json_dir, f"{args.xp_name}_patch_stride_subpatch_sweep.json")
with open(json_path, "w") as f:
    json.dump(
        {
            "experiment": args.xp_name,
            "model": "universat",
            "dataset": "burnscars",
            "checkpoint": args.ckpt,
            "universat_size": args.universat_size,
            "default_output_stride": DEFAULT_OUTPUT_STRIDE,
            "default_subpatch_px": DEFAULT_SUBPATCH_PX,
            "patch_sizes_m": PATCH_SIZES_M,
            "patch_stride_couples": [
                {"patch_m": pm, "strides": strides}
                for pm, strides in PATCH_STRIDE_COUPLES
            ],
            "patch_subpatch_couples": [
                {"patch_m": pm, "output_stride": os_, "subpatches": subpatches}
                for pm, os_, subpatches in PATCH_SUBPATCH_COUPLES
            ],
            "eval_mode": {"mode": "dense_center_crop",
                          "native_side": NATIVE_SIDE_PX,
                          "note": "per-point eval_side in results; mIoU and "
                                  "GFLOPs are computed on that crop"},
            "flops_method": "torch.utils.flop_counter.FlopCounterMode",
            "gsd_m": HLS_GSD_M,
            "flops_n": args.flops_n,
            "results": json_results,
            # >>> PARETO_HULL
            "pareto_front": json_pareto_front,
            "convex_hull": json_convex_hull,
        },
        f, indent=2,
    )
print(f"[Sweep] JSON results saved to {json_path}")

out_path = f"./results_{args.xp_name}_patch_stride_subpatch_sweep.txt"
with open(out_path, "w") as f:
    f.write(f"UniverSat (Patch x Stride x Subpatch) Sweep — HLS BurnScars — {args.xp_name}\n")
    f.write(f"Checkpoint: {args.ckpt}\n")
    f.write(f"Size: {args.universat_size}; "
            f"default output_stride: {DEFAULT_OUTPUT_STRIDE}; "
            f"default subpatch_px: {DEFAULT_SUBPATCH_PX}\n")
    f.write(f"Eval: single dense forward, per-point center crop "
            f"(eval side in table)\n")
    f.write(f"FLOPs: FlopCounterMode (SDPA attention counted)\n\n")
    f.write(f"{'Patch':<9}{'(px)':<6}{'os':<5}{'sp':<5}{'Eval':<7}{'Trunk tok':<11}"
            f"{'Queries':<10}{'Keys':<10}{'GFLOPs':<12}{'mIoU':<10}\n")
    f.write(f"{'─'*88}\n")
    for r in results:
        f.write(f"{str(int(r['patch_m']))+' m':<9}"
                f"{r['patch_px']:<6}"
                f"{r['output_stride']:<5}"
                f"{r['subpatch_px']:<5}"
                f"{r['eval_side']:<7}"
                f"{r['latent_tokens']:<11}"
                f"{r['out_queries']:<10}"
                f"{r['subpatch_keys']:<10}"
                f"{r['gflops']:<12.2f}"
                f"{r['miou']:<10.4f}\n")
    f.write("\nCompact summary:\n")
    for r in results:
        f.write(f"patch {int(r['patch_m'])} os {r['output_stride']} "
                f"sp {r['subpatch_px']}: "
                f"GFLOPS: {r['gflops']:.2f} mIoU: {r['miou']:.4f}\n")

    # >>> PARETO_HULL
    f.write(f"\n\n{'='*78}\n")
    f.write(f"PARETO FRONT + CONVEX HULL — GFLOPs x mIoU\n")
    f.write(f"{'='*78}\n")
    f.write(f"\nPareto front ({len(pareto_front)} points, all non-dominated):\n")
    f.write(f"  {'patch':>7} {'os':>4} {'sp':>4} {'eval':>5} "
            f"{'gflops':>12} {'miou':>8}\n")
    f.write("  " + "-" * 48 + "\n")
    for p in pareto_front:
        f.write(f"  {int(p['patch_m']):>5} m {p['output_stride']:>4} "
                f"{p['subpatch_px']:>4} {p['eval_side']:>5} "
                f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}\n")
    f.write(f"\nConvex hull / efficient frontier ({len(convex_hull)} points, "
            f"the ones actually worth plotting/using):\n")
    f.write(f"  {'patch':>7} {'os':>4} {'sp':>4} {'eval':>5} "
            f"{'gflops':>12} {'miou':>8}\n")
    f.write("  " + "-" * 48 + "\n")
    for p in convex_hull:
        f.write(f"  {int(p['patch_m']):>5} m {p['output_stride']:>4} "
                f"{p['subpatch_px']:>4} {p['eval_side']:>5} "
                f"{p['gflops']:>12.2f} {_fmt(p['miou']):>8}\n")

print(f"\n[Sweep] Results saved to {out_path}")
