"""
UniverSat GFLOPs Probe — pick a training config by compute budget
===================================================================

Builds UniverSatSegmenter at every (patch_px, output_stride) combination
in the grids below and measures TOTAL GFLOPs of one forward pass with
torch.utils.flop_counter.FlopCounterMode (counts fused-SDPA attention on
CUDA -- unlike torch.profiler, which silently drops it). Random init, no
checkpoint, no dataset: FLOP counts are shape-driven, so this runs in
minutes and is exact for the given geometry.

Purpose: the baselines are parameter-matched (~30-37M) but NOT natively
compute-matched -- RAMEN and Atomizer are both run at reduced working
resolution to keep training feasible, and UniverSat's native fine-grid
config is far more expensive than either. This probe lets you choose
UniverSat's (patch, stride) so its per-forward compute lands in the same
envelope, the same way RAMEN's `res` was chosen. Pass --target_gflops
(e.g. Atomizer's or RAMEN's per-forward number from the same harness) to
get ratio columns and a highlighted nearest config.

Defaults are the xView2 geometry (side 512 train crop, T=2, 3ch BGR VHR
at 0.5 m). Use the flags for other datasets, e.g.:

    # xView2 (default)
    python script_universat_flops_probe.py

    # Sen1Floods11 geometry
    python script_universat_flops_probe.py --preset senflood

    # BioMassters geometry
    python script_universat_flops_probe.py --preset biomassters

    # With a compute target (GFLOPs/forward from the SAME FlopCounterMode
    # harness -- never mix with torch.profiler numbers)
    python script_universat_flops_probe.py --target_gflops 950

Estimated TRAINING cost per sample is also printed as ~3x the forward
(fwd + bwd ~= 2x fwd for attention/linear-dominated nets) -- a standard
rule of thumb, labeled as such.

NOTE on hardware: run this ON GPU. FlopCounterMode has formulas for the
CUDA SDPA kernels (flash/efficient/cudnn) but not for the CPU flash
variant, so CPU runs undercount attention -- the script warns if CUDA is
unavailable.
"""

# =============================================================================
# CANDIDATE GRIDS — edit these
# =============================================================================

PATCH_PX_LIST = [4, 8, 16, 32]        # patch size in PIXELS (metres = px * GSD)
OUTPUT_STRIDES = [2, 4, 8, 16, 32]    # CA_Sub query stride

# =============================================================================

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.flop_counter import FlopCounterMode

from training.Universat.universat_augmenter import build_universat_segmenter


PRESETS = {
    # side, T, gsd_m, modalities: {name: (n_channels, wavelengths_spec)}
    "xview": dict(
        side=512, T=2, gsd=0.5,
        input_bands={"vhr": ["Blue", "Green", "Red"]},
        wavelengths={"vhr": [490.0, 560.0, 665.0]},     # BGR order (cv2)
        temporal=True,
    ),
    "senflood": dict(
        side=512, T=1, gsd=10.0,
        input_bands={"optical": [f"B{i:02d}" for i in range(1, 13)] + ["B8A"],
                     "sar": ["VV", "VH"]},
        wavelengths={"optical": {"B01": 442.7, "B02": 492.4, "B03": 559.8,
                                 "B04": 664.6, "B05": 704.1, "B06": 740.5,
                                 "B07": 782.8, "B08": 832.8, "B8A": 864.7,
                                 "B09": 945.1, "B10": 1373.5, "B11": 1613.7,
                                 "B12": 2202.4},
                     "sar": ["VV", "VH"]},
        temporal=False,
    ),
    "biomassters": dict(
        side=256, T=3, gsd=10.0,
        input_bands={"optical": ["B02", "B03", "B04", "B05", "B06",
                                 "B07", "B08", "B8A", "B11", "B12"],
                     "sar": ["VV_asc", "VH_asc", "VV_desc", "VH_desc"]},
        wavelengths={"optical": {"B02": 490, "B03": 560, "B04": 665,
                                 "B05": 705, "B06": 740, "B07": 783,
                                 "B08": 842, "B8A": 865, "B11": 1610,
                                 "B12": 2190},
                     "sar": ["VV", "VH", "HH", "HV"]},
        temporal=True,
    ),
    "burnscars": dict(
        side=512, T=1, gsd=30.0,
        input_bands={"hls": ["B02", "B03", "B04", "B8A", "B11", "B12"]},
        wavelengths={"hls": {"B02": 492.4, "B03": 559.8, "B04": 664.6,
                             "B8A": 864.7, "B11": 1613.7, "B12": 2202.4}},
        temporal=False,
    ),
}


parser = argparse.ArgumentParser(
    description="UniverSat GFLOPs probe over (patch, stride) configs")
parser.add_argument("--preset", type=str, default="xview",
                    choices=list(PRESETS.keys()))
parser.add_argument("--side", type=int, default=None,
                    help="Override the preset's input side (train crop size).")
parser.add_argument("--universat_size", type=str, default="small",
                    choices=["tiny", "small", "base"])
parser.add_argument("--target_gflops", type=float, default=None,
                    help="Reference GFLOPs/forward (Atomizer / RAMEN, measured "
                         "with FlopCounterMode on the SAME input geometry). "
                         "Adds a ratio column and highlights the nearest config.")
parser.add_argument("--bf16", action="store_true", default=True,
                    help="Run the counted forward under bf16 autocast "
                         "(matches training; FLOP counts are dtype-agnostic).")
args = parser.parse_args()

cfg = PRESETS[args.preset]
side = args.side or cfg["side"]
T, gsd = cfg["T"], cfg["gsd"]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("[WARN] No CUDA: FlopCounterMode lacks a formula for the CPU SDPA "
          "kernel, so attention FLOPs will be UNDERCOUNTED. Run on GPU for "
          "usable numbers.\n")

# Synthetic batch at bs=1 (numbers are per-image; batch scales linearly)
def make_batch():
    x = {}
    for mod, bands in cfg["input_bands"].items():
        C = len(bands)
        if cfg["temporal"]:
            x[mod] = torch.randn(1, T, C, side, side, device=device)
            x[f"{mod}_dates"] = torch.linspace(0, 180, T).long()\
                                     .unsqueeze(0).to(device)
        else:
            x[mod] = torch.randn(1, C, side, side, device=device)
    return x


print(f"\n{'='*84}")
print(f"  UniverSat GFLOPs probe — preset={args.preset}, size={args.universat_size}")
print(f"  Input: side {side}, T={T}, GSD {gsd} m, "
      f"modalities {list(cfg['input_bands'].keys())}")
print(f"  Harness: FlopCounterMode (SDPA counted on CUDA); bs=1 forward")
if args.target_gflops:
    print(f"  Target: {args.target_gflops:.1f} GFLOPs/forward")
print(f"{'='*84}\n")

rows = []
params_printed = False
for patch_px in PATCH_PX_LIST:
    for os_ in OUTPUT_STRIDES:
        lcm = math.lcm(patch_px, os_)
        if side % lcm:
            print(f"[skip] patch {patch_px}px / os {os_}: side {side} not "
                  f"divisible by lcm={lcm}")
            continue

        model = build_universat_segmenter(
            input_bands=cfg["input_bands"],
            wavelengths=cfg["wavelengths"],
            num_classes=2,     # head cost is negligible; class count irrelevant
            input_res={m: gsd for m in cfg["input_bands"]},
            patch_size_m=patch_px * gsd,
            output_stride=os_,
            size=args.universat_size,
        ).to(device).eval()

        if not params_printed:
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  [params] {n_params:.2f}M (constant across all configs)\n")
            params_printed = True

        x = make_batch()
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            if args.bf16 and device == "cuda":
                with fc, torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x)
            else:
                with fc:
                    model(x)
            gflops = fc.get_total_flops() / 1e9

        latent = (side // patch_px) ** 2
        queries = (side // os_) ** 2
        rows.append(dict(patch_px=patch_px, patch_m=patch_px * gsd,
                         os=os_, latent=latent, queries=queries,
                         gran_m=os_ * gsd, gflops=gflops))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

# ── Table ──────────────────────────────────────────────────────────────────
hdr = (f"  {'patch':>7} {'(px)':>5} {'os':>4} {'trunk tok':>10} "
       f"{'queries':>9} {'pred gran':>10} {'GFLOPs/fwd':>12} "
       f"{'~train/sample':>14}")
if args.target_gflops:
    hdr += f" {'x target':>9}"
print(hdr)
print("  " + "─" * (len(hdr) - 2))

best = None
for r in sorted(rows, key=lambda r: r["gflops"]):
    line = (f"  {r['patch_m']:>6.1f}m {r['patch_px']:>5} {r['os']:>4} "
            f"{r['latent']:>10} {r['queries']:>9} {r['gran_m']:>9.1f}m "
            f"{r['gflops']:>12.1f} {3 * r['gflops']:>14.1f}")
    if args.target_gflops:
        ratio = r["gflops"] / args.target_gflops
        line += f" {ratio:>8.2f}x"
        if best is None or abs(math.log(ratio)) < abs(math.log(best[0])):
            best = (ratio, r)
    print(line)

if args.target_gflops and best:
    r = best[1]
    print(f"\n  [nearest to target] patch {r['patch_m']:.1f} m "
          f"({r['patch_px']} px), os={r['os']}: {r['gflops']:.1f} GFLOPs "
          f"({best[0]:.2f}x target)")

print(f"\n  Notes: '~train/sample' = 3x forward (fwd+bwd rule of thumb). "
      f"Numbers are per bs=1 image at side {side}; scale linearly with "
      f"batch. Sliding-window eval at tile=side costs "
      f"(image_area/side^2) x GFLOPs/fwd per test image.")
