"""
BurnScars Density Generalization Driver
=========================================

Loads ONE trained checkpoint and evaluates it under all
(tokens_per_latent, cross_k, decoder_k_spatial) configs. Each config runs
as a fresh subprocess (clean ckpt reload).

# >>> PORTED_FROM_SENFLOOD: this is a rename+feature-port of
# run_senflood_density_eval.py onto BurnScars — same machinery, same tags:
#
#   --split val  (default, recommended for wide sweeps): scores every
#     config on the VALIDATION set. GFLOPs is measured with the 'light'
#     scope by default (small sampled forward-pass budget,
#     --gflops_light_n, default 30 total across all GPUs) rather than a
#     full-split pass — GFLOPs is ~content-independent for a fixed
#     (tpl, cross_k, decoder_k_spatial), unlike mIoU, so this still gives
#     a real number per config. Result: a genuine mIoU-vs-GFLOPs Pareto
#     front (reduced to its convex hull / efficient frontier) for every
#     config in the grid, not a proxy.
#
#   --split test (final numbers): scores on the held-out test set, GFLOPs
#     measured with the 'full' scope by default (precise, slow) — meant
#     for the handful of configs the val-split hull shortlisted. The
#     printed "suggested next step" command reuses the SAME CONFIGS
#     indices, so `--only <hull indices>` from a val run maps 1:1 onto
#     the identical (tpl, ck, dks) tuples in a test run, as long as
#     TABLE1/TABLE2/TABLE3 below aren't edited between the two runs.
#
# >>> TYPE: unlike the original three near-identical BurnScars drivers
# (run_burnscars_ablation.py / _quad.py / _zone.py), this single driver
# takes --type explicitly (regular | quadtree | zoneprobe, default
# "regular") and folds it into xp names / tmp_results filenames, the same
# way the Sen1Floods11 driver does — so you no longer need three near-
# duplicate files to cover the three decode methods.

# >>> FLAG: TEST_SCRIPT below points at
# "./scripts_GFLOPS/script_test_burnscars_density.py" — the companion
# worker script (see its own module docstring for the FlopCounterMode /
# sharding / split implementation).

# >>> RESULTS_TRACKING: the worker script writes a small per-run JSON file
# to ./scripts_GFLOPS/tmp_results/ once it finishes scoring a config
# (filename includes split AND gflops_scope, so val/test/light/full scores
# for the same config never collide in the resume cache). Before launching
# each subprocess, this driver checks for that file and, if found, reuses
# its scores instead of re-running.

Usage:
    # Wide, fast val-split sweep + Pareto front/hull (recommended entry point):
    python run_burnscars_ablation.py --ckpt ./checkpoints/best.ckpt

    # Final numbers on a shortlist (indices from the val sweep's hull):
    python run_burnscars_ablation.py --ckpt ... --split test --only 0 5 11

    python run_burnscars_ablation.py --ckpt ... --dry_run

Resuming after an interrupted run: just re-run the same command. Configs
that already have a matching file in ./scripts_GFLOPS/tmp_results/ (same
ckpt, config, split, gflops_scope, and type) are skipped automatically.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import re
import time
import json       # >>> RESULTS_TRACKING
import hashlib     # >>> RESULTS_TRACKING
import argparse
import subprocess
from datetime import datetime

# >>> RENAME: BurnScars default config, kept as a default (still overridable
# via --config, same as the Sen1Floods11 driver) rather than hardcoded, so
# this single driver can be pointed at a different config if needed.
DEFAULT_CONFIG_PATH = "./training/configs/config_test-BURNSCARS.yaml"

# >>> DECODER_K: fixed decoder_k_spatial used for tables that don't sweep it.
# Set this to your checkpoint's trained/default value.
DEFAULT_DKS = 4

TABLE1 = [(tpl, 1000, DEFAULT_DKS) for tpl in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,7000,8000,12000,16000]]

# TABLE2: vary cross_k, tokens_per_latent held fixed at the paper's value (2000).
TABLE2 = [(2000, ck, DEFAULT_DKS) for ck in [1,25,50,75,100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000]]

# >>> DECODER_K: new sweep axis. tpl and cross_k held fixed; adjust the dks
# list to whatever range you want to probe.
TABLE3 = [(2000, 500, dks) for dks in
          [1,2,3,4,5,6,7,8,9]]
TABLE3 += [(2000, 500, dks) for dks in
          [1,2,3,4,5,6,7,8,9]]

TABLE3 += [(2000,500,2),(2000,300,2),(4000,300,2),(8000,300,2),(4000,500,2),(8000,500,2),
    (2000,1000,2),(2000,1000,2),(4000,300,2),(8000,1000,4),(4000,1000,4),(8000,1000,4),
    (2000,1000,4),(2000,1000,4),(4000,300,4),(8000,1000,4),(4000,1000,4),(8000,500,5),
    (2000,100,1),(7000,100,1),(8000,100,4),(8000,200,2),(16000,500,1),(32000,2,1)]

CONFIGS = TABLE1 + TABLE2 + TABLE3


TEST_SCRIPT = "./scripts_GFLOPS/script_test_burnscars_density.py"
TASK = "burnscars"

# >>> RESULT line format matches the updated worker script (split=..., mIoU=,
# accuracy=... instead of test_mIoU=/test_accuracy=).
RESULT_RE = re.compile(
    r"RESULT split=(\w+) tpl=(\d+) cross_k=(\d+) decoder_k_spatial=(\d+) "
    r"mIoU=([-\d.nan]+) accuracy=([-\d.nan]+) gflops=([-\d.nan]+)")


# =============================================================================
# >>> RESULTS_TRACKING: resume support + final results file
# =============================================================================
TMP_RESULTS_DIR = "./scripts_GFLOPS/tmp_results"
RESULTS_DIR = "./scripts_GFLOPS/results"


def _ckpt_tag(ckpt_path):
    """MUST stay in sync with the worker script's own _ckpt_tag."""
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    h = hashlib.md5(os.path.abspath(ckpt_path).encode()).hexdigest()[:8]
    return f"{safe}_{h}"


def tmp_result_path(ckpt_path, tpl, ck, dks, decode_type, split, gflops_scope):
    return os.path.join(
        TMP_RESULTS_DIR,
        f"{TASK}_{_ckpt_tag(ckpt_path)}_{decode_type}_{split}_{gflops_scope}_"
        f"tpl{tpl}_ck{ck}_dks{dks}.json",
    )


def load_existing_result(ckpt_path, tpl, ck, dks, decode_type, split, gflops_scope):
    p = tmp_result_path(ckpt_path, tpl, ck, dks, decode_type, split, gflops_scope)
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[driver][WARN] could not read existing result {p}: {e}")
    return None


parser = argparse.ArgumentParser(description="Drive BurnScars density-generalization eval sweep")
parser.add_argument("--ckpt", required=True, help="Single trained checkpoint to evaluate")
parser.add_argument("--only", type=int, nargs="+", default=None)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--python", type=str, default=sys.executable)
parser.add_argument("--continue_on_fail", action="store_true", default=True,
                    help="Keep going if a run fails (default True for eval).")
parser.add_argument("--type", type=str, default="regular",
                    choices=("regular", "quadtree", "zoneprobe"),
                    help="Decode method label: regular | quadtree | zoneprobe.")
parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                    help="Config passed through to the worker script.")
parser.add_argument("--no_resume", action="store_true",
                    help="Ignore any existing files in ./scripts_GFLOPS/tmp_results/ "
                         "and re-run every config from scratch.")
parser.add_argument("--split", type=str, default="val", choices=("val", "test"),
                    help="'val' (default): fast wide-grid scoring for config "
                         "SELECTION, GFLOPs measured LIGHT by default (real "
                         "GFLOPs numbers for every config, see "
                         "--gflops_light_n) so the front/hull is a genuine "
                         "mIoU-vs-GFLOPs curve. 'test': final scoring on "
                         "the held-out set, GFLOPs measured FULL by default.")
parser.add_argument("--gflops_scope", type=str, default=None,
                    choices=("full", "light", "skip"),
                    help="Override the GFLOPs measurement scope. Default "
                         "if unset: 'light' for --split val, 'full' for "
                         "--split test. 'skip' disables GFLOPs entirely.")
parser.add_argument("--gflops_light_n", type=int, default=30,
                    help="Total samples (summed across all GPUs) to "
                         "profile per config when GFLOPs scope is 'light'.")
args = parser.parse_args()

indices = args.only if args.only is not None else list(range(len(CONFIGS)))
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DECODE_TYPE = args.type
SPLIT = args.split
GFLOPS_SCOPE = args.gflops_scope or ("light" if SPLIT == "val" else "full")
RUN_GFLOPS = GFLOPS_SCOPE != "skip"

print(f"[driver] ckpt: {args.ckpt}")
print(f"[driver] split: {SPLIT}  gflops: "
      f"{'OFF (skipped)' if not RUN_GFLOPS else GFLOPS_SCOPE}"
      + (f" (n={args.gflops_light_n})" if GFLOPS_SCOPE == "light" else ""))
print(f"[driver] type: {DECODE_TYPE}  config: {args.config}")
print(f"[driver] {len(indices)} configs, start {stamp}")
print(f"[driver] resume: {not args.no_resume} (tmp_results dir: {TMP_RESULTS_DIR})\n")

# idx -> (tpl, ck, dks, miou, acc, gflops, rc, use_quadtree_decode, use_adaptive_decode, resumed)
parsed = {}

for run_n, idx in enumerate(indices, 1):
    tpl, ck, dks = CONFIGS[idx]
    if idx < len(TABLE1):
        table = 1
    elif idx < len(TABLE1) + len(TABLE2):
        table = 2
    else:
        table = 3

    existing = None
    if not args.dry_run and not args.no_resume:
        existing = load_existing_result(args.ckpt, tpl, ck, dks, DECODE_TYPE,
                                        SPLIT, GFLOPS_SCOPE)

    if existing is not None:
        print("=" * 78)
        print(f"[driver] ({run_n}/{len(indices)}) idx={idx} table={table} "
              f"tpl={tpl} cross_k={ck} decoder_k_spatial={dks} "
              f"type={DECODE_TYPE} split={SPLIT} gflops={GFLOPS_SCOPE} "
              f"-> FOUND existing result, skipping run")
        print("=" * 78, flush=True)
        parsed[idx] = (
            tpl, ck, dks,
            existing.get("mIoU", existing.get("test_mIoU", float("nan"))),
            existing.get("accuracy", existing.get("test_accuracy", float("nan"))),
            existing.get("gflops", float("nan")),
            0,
            existing.get("use_quadtree_decode"),
            existing.get("use_adaptive_decode"),
            True,
        )
        continue

    xp = f"denseval_burnscars_{DECODE_TYPE}_{SPLIT}_t{table}_tpl{tpl}_ck{ck}_dks{dks}_{stamp}"

    cmd = [
        args.python, TEST_SCRIPT,
        "--ckpt", args.ckpt,
        "--xp_name", xp,
        "--tokens_per_latent", str(tpl),
        "--cross_k", str(ck),
        "--decoder_k_spatial", str(dks),
        "--type", DECODE_TYPE,
        "--split", SPLIT,
        "--config", args.config,
    ]
    if GFLOPS_SCOPE == "skip":
        cmd += ["--skip_gflops"]
    else:
        cmd += ["--gflops_scope", GFLOPS_SCOPE]
        if GFLOPS_SCOPE == "light":
            cmd += ["--gflops_light_n", str(args.gflops_light_n)]

    print("=" * 78)
    print(f"[driver] ({run_n}/{len(indices)}) idx={idx} table={table} "
          f"tpl={tpl} cross_k={ck} decoder_k_spatial={dks} type={DECODE_TYPE} "
          f"split={SPLIT} gflops={GFLOPS_SCOPE}")
    print(f"[driver] cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)

    if args.dry_run:
        parsed[idx] = (tpl, ck, dks, float("nan"), float("nan"), float("nan"),
                       "DRY", None, None, False)
        continue

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    miou = acc = gflops = float("nan")
    m = None
    for line in proc.stdout.splitlines():
        mm = RESULT_RE.search(line)
        if mm:
            m = mm
    if m:
        # groups: split, tpl, cross_k, decoder_k_spatial, mIoU, accuracy, gflops
        miou   = float(m.group(5))
        acc    = float(m.group(6))
        gflops = float(m.group(7))

    use_quadtree_decode = use_adaptive_decode = None
    written = load_existing_result(args.ckpt, tpl, ck, dks, DECODE_TYPE, SPLIT, GFLOPS_SCOPE)
    if written is not None:
        use_quadtree_decode = written.get("use_quadtree_decode")
        use_adaptive_decode = written.get("use_adaptive_decode")
        if written.get("type") != DECODE_TYPE:
            print(f"[driver][WARN] idx={idx}: worker's own --type "
                  f"({written.get('type')}) doesn't match the driver's "
                  f"({DECODE_TYPE}) — check --config/--type wiring.")
    elif proc.returncode == 0:
        print(f"[driver][WARN] idx={idx} finished rc=0 but no per-run result "
              f"file found at "
              f"{tmp_result_path(args.ckpt, tpl, ck, dks, DECODE_TYPE, SPLIT, GFLOPS_SCOPE)} — "
              f"worker script may have changed or failed to write it.")

    parsed[idx] = (tpl, ck, dks, miou, acc, gflops, proc.returncode,
                   use_quadtree_decode, use_adaptive_decode, False)

    print(f"[driver] idx={idx} rc={proc.returncode} "
          f"mIoU={miou:.4f} gflops={gflops:.2f} "
          f"quadtree={use_quadtree_decode} adaptive={use_adaptive_decode} "
          f"({dt/60:.1f} min)\n",
          flush=True)

    if proc.returncode != 0 and not args.continue_on_fail:
        print(f"[driver] idx={idx} failed; stopping.")
        break

# =============================================================================
# TABLES
# =============================================================================
def fmt(v):
    if v is None:
        return "  nan  "
    return "  nan  " if v != v else f"{v:.4f}"


def fmt_flag(v):
    if v is None:
        return "  ?  "
    return " Y " if v else " N "


print("\n" + "=" * 78)
print(f"DENSITY GENERALIZATION — BurnScars {SPLIT.upper()} mIoU + GFLOPs "
      f"(single checkpoint, type={DECODE_TYPE})")
print("=" * 78)

def _print_table(title, lo, hi):
    print(f"\n{title}  [type={DECODE_TYPE} split={SPLIT}]")
    print(f"  {'tpl':>6} {'cross_k':>8} {'dec_k_sp':>9} "
          f"{'mIoU':>10} {'accuracy':>10} {'GFLOPs':>10} "
          f"{'quadtree':>9} {'adaptive':>9} {'resumed':>8}")
    print("  " + "-" * 92)
    for idx in range(lo, hi):
        if idx in parsed:
            tpl, ck, dks, miou, acc, gflops, rc, uqd, uad, resumed = parsed[idx]
            print(f"  {tpl:>6} {ck:>8} {dks:>9} "
                  f"{fmt(miou):>10} {fmt(acc):>10} {fmt(gflops):>10} "
                  f"{fmt_flag(uqd):>9} {fmt_flag(uad):>9} "
                  f"{('Y' if resumed else 'N'):>8}")

_print_table("Table 1 — vary tokens_per_latent (cross_k=1000, "
             f"decoder_k_spatial={DEFAULT_DKS}):",
             0, len(TABLE1))
_print_table(f"Table 2 — tpl=2000, vary cross_k (decoder_k_spatial={DEFAULT_DKS}):",
             len(TABLE1), len(TABLE1) + len(TABLE2))
_print_table("Table 3 — tpl=2000, cross_k=2000, vary decoder_k_spatial:",
             len(TABLE1) + len(TABLE2), len(CONFIGS))

os.makedirs("training/ablation_runs", exist_ok=True)
csv_path = f"training/ablation_runs/density_eval_burnscars_{DECODE_TYPE}_{SPLIT}_{stamp}.csv"
with open(csv_path, "w") as f:
    f.write("table,idx,tokens_per_latent,cross_k,decoder_k_spatial,type,split,"
             "mIoU,accuracy,gflops,rc,use_quadtree_decode,"
             "use_adaptive_decode,resumed\n")
    for idx in sorted(parsed.keys()):
        tpl, ck, dks, miou, acc, gflops, rc, uqd, uad, resumed = parsed[idx]
        if idx < len(TABLE1):
            table = 1
        elif idx < len(TABLE1) + len(TABLE2):
            table = 2
        else:
            table = 3
        f.write(f"{table},{idx},{tpl},{ck},{dks},{DECODE_TYPE},{SPLIT},{miou},{acc},"
                 f"{gflops},{rc},{uqd},{uad},{resumed}\n")
print(f"\n[driver] CSV written to {csv_path}")

# =============================================================================
# >>> RESULTS_TRACKING: full sweep summary -> ./scripts_GFLOPS/results/
# =============================================================================
os.makedirs(RESULTS_DIR, exist_ok=True)
results_json_path = os.path.join(
    RESULTS_DIR, f"density_eval_{TASK}_{DECODE_TYPE}_{SPLIT}_{stamp}.json")
results_txt_path = os.path.join(
    RESULTS_DIR, f"density_eval_{TASK}_{DECODE_TYPE}_{SPLIT}_{stamp}.txt")

sweep_record = {
    "task": TASK,
    "ckpt": os.path.abspath(args.ckpt),
    "config": args.config,
    "type": DECODE_TYPE,
    "split": SPLIT,
    "stamp": stamp,
    "default_dks": DEFAULT_DKS,
    "runs": [],
}
for idx in sorted(parsed.keys()):
    tpl, ck, dks, miou, acc, gflops, rc, uqd, uad, resumed = parsed[idx]
    if idx < len(TABLE1):
        table = 1
    elif idx < len(TABLE1) + len(TABLE2):
        table = 2
    else:
        table = 3
    sweep_record["runs"].append({
        "idx": idx, "table": table,
        "tokens_per_latent": tpl, "cross_k": ck, "decoder_k_spatial": dks,
        "type": DECODE_TYPE, "split": SPLIT,
        "mIoU": miou, "accuracy": acc, "gflops": gflops, "rc": rc,
        "use_quadtree_decode": uqd, "use_adaptive_decode": uad,
        "resumed_from_tmp_results": resumed,
    })

with open(results_json_path, "w") as f:
    json.dump(sweep_record, f, indent=2)

with open(results_txt_path, "w") as f:
    f.write(f"BurnScars density sweep — ckpt: {args.ckpt}\n")
    f.write(f"config: {args.config}\n")
    f.write(f"type: {DECODE_TYPE}  split: {SPLIT}\n")
    f.write(f"stamp: {stamp}\n\n")
    f.write(f"{'idx':>5} {'table':>5} {'tpl':>6} {'cross_k':>8} {'dec_k_sp':>9} "
            f"{'mIoU':>10} {'accuracy':>10} {'GFLOPs':>10} "
            f"{'quadtree':>9} {'adaptive':>9} {'resumed':>8}\n")
    for r in sweep_record["runs"]:
        f.write(f"{r['idx']:>5} {r['table']:>5} {r['tokens_per_latent']:>6} "
                f"{r['cross_k']:>8} {r['decoder_k_spatial']:>9} "
                f"{fmt(r['mIoU']):>10} {fmt(r['accuracy']):>10} "
                f"{fmt(r['gflops']):>10} {fmt_flag(r['use_quadtree_decode']):>9} "
                f"{fmt_flag(r['use_adaptive_decode']):>9} "
                f"{('Y' if r['resumed_from_tmp_results'] else 'N'):>8}\n")

print(f"[driver] Full sweep results written to {results_json_path} and {results_txt_path}")


# =============================================================================
# >>> PARETO_SELECTION + CONVEX_HULL: Pareto front, reduced to its convex
# hull (the "efficient frontier") over (cost=GFLOPs, score=mIoU)
# =============================================================================

def _pareto_front(points, cost_key):
    """point A dominates point B iff A.mIoU >= B.mIoU AND A.cost <= B.cost,
    with at least one strict inequality. NaN mIoU/cost are excluded."""
    valid = [p for p in points
             if p["mIoU"] == p["mIoU"] and p[cost_key] == p[cost_key]]
    front = []
    for p in valid:
        dominated = False
        for q in valid:
            if q is p:
                continue
            better_or_equal = (q["mIoU"] >= p["mIoU"]) and (q[cost_key] <= p[cost_key])
            strictly_better = (q["mIoU"] > p["mIoU"]) or (q[cost_key] < p[cost_key])
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


def _upper_convex_hull(points, cost_key):
    """Upper convex hull in (cost, mIoU) space — the concave,
    diminishing-returns boundary. Points below the segment through their
    cost-neighbors on the front are dropped even if individually
    non-dominated. Monotone-chain (Andrew's algorithm), upper half only."""
    pts = sorted(points, key=lambda p: (p[cost_key], -p["mIoU"]))
    hull = []
    for p in pts:
        xy = (p[cost_key], p["mIoU"])
        while len(hull) >= 2:
            o_xy = (hull[-2][cost_key], hull[-2]["mIoU"])
            a_xy = (hull[-1][cost_key], hull[-1]["mIoU"])
            if _cross(o_xy, a_xy, xy) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


pareto_points = []
for idx in sorted(parsed.keys()):
    tpl, ck, dks, miou, acc, gflops, rc, uqd, uad, resumed = parsed[idx]
    pareto_points.append({
        "idx": idx, "tpl": tpl, "cross_k": ck, "decoder_k_spatial": dks,
        "mIoU": miou, "accuracy": acc, "gflops": gflops,
        "cost_proxy": tpl * ck,
        "type": DECODE_TYPE,
        "use_quadtree_decode": uqd,
        "use_adaptive_decode": uad,
    })

have_real_gflops = any(p["gflops"] == p["gflops"] for p in pareto_points)
cost_key = "gflops" if have_real_gflops else "cost_proxy"
cost_label = "gflops" if have_real_gflops else "tpl*cross_k"

front = _pareto_front(pareto_points, cost_key)
hull = _upper_convex_hull(front, cost_key)
hull.sort(key=lambda p: p[cost_key])

def _print_front(title, front, cost_key, cost_label):
    print(f"\n{title}")
    if not front:
        print("  (no valid points — all mIoU or cost values were NaN)")
        return
    print(f"  {'idx':>5} {'tpl':>6} {'cross_k':>8} {'dec_k_sp':>9} "
          f"{'mIoU':>10} {cost_label:>14} {'type':>10}")
    print("  " + "-" * 72)
    for p in front:
        print(f"  {p['idx']:>5} {p['tpl']:>6} {p['cross_k']:>8} "
              f"{p['decoder_k_spatial']:>9} {fmt(p['mIoU']):>10} "
              f"{p[cost_key]:>14.4g} {p['type']:>10}")

print("\n" + "=" * 78)
print(f"EFFICIENT FRONTIER — convex hull of the mIoU-vs-{cost_label} Pareto "
      f"front  (split={SPLIT}, gflops_scope={GFLOPS_SCOPE})")
print("=" * 78)
if not have_real_gflops:
    print("\n(No configs in this sweep have a measured GFLOPs value "
          "-- probably run with --gflops_scope skip. Falling back to the "
          "tpl*cross_k proxy; re-run with the default 'light' scope for a "
          "real GFLOPS x mIoU curve.)")
_print_front(f"Pareto front ({len(front)} points, all non-dominated):",
             front, cost_key, cost_label)
_print_front(f"Convex hull / efficient frontier ({len(hull)} points, "
             f"the ones actually worth plotting/using):",
             hull, cost_key, cost_label)

pareto_path = os.path.join(
    RESULTS_DIR, f"pareto_front_{TASK}_{DECODE_TYPE}_{SPLIT}_{stamp}.json")
with open(pareto_path, "w") as f:
    json.dump({
        "ckpt": os.path.abspath(args.ckpt),
        "split": SPLIT,
        "type": DECODE_TYPE,
        "gflops_scope": GFLOPS_SCOPE,
        "cost_key": cost_key,
        "stamp": stamp,
        "pareto_front": front,
        "convex_hull": hull,
    }, f, indent=2)
print(f"\n[driver] Pareto front + convex hull written to {pareto_path}")

if hull:
    shortlist = " ".join(str(p["idx"]) for p in hull)
    print(f"[driver] Suggested next step — re-run the hull's shortlist "
          f"on the test set with full-precision GFLOPs for final numbers:\n"
          f"  python {os.path.basename(__file__)} --ckpt {args.ckpt} "
          f"--split test --only {shortlist}")
