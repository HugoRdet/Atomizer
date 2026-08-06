"""
Sen1Floods11 Density Generalization Driver (TEST-ONLY)
======================================================

Loads ONE trained checkpoint and evaluates it on the test split under all 22
(tokens_per_latent, cross_k) configs — inference-time density generalization.
No training. Each config runs as a fresh subprocess (clean ckpt reload).

Parses each run's "RESULT ..." line and assembles two tables:
  Table 1: vary tpl (1000..6000 step 500), cross_k=1000
  Table 2: tpl=2000, vary cross_k (100,300,...,2000)

Usage:
    python run_senflood_density_eval.py --ckpt ./checkpoints/best.ckpt
    python run_senflood_density_eval.py --ckpt ... --only 0 5 11
    python run_senflood_density_eval.py --ckpt ... --dry_run
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import re
import time
import argparse
import subprocess
from datetime import datetime

TABLE1 = [(tpl, 500) for tpl in range(1000, 6001, 500)]          # idx 0..10
TABLE1 += [(8000,500),(16000,500),(32000,500)]
#TABLE1=[]
#TABLE2 = [(4000, ck) for ck in
#          [1, 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2000]]  # idx 11..21
TABLE2=[]
#(4000,350),(4000,500),(4000,1000),(2000,350),(2000,500),(2000,750),(2000,1000),(4000,1000)
#TABLE1=[(4000,350),(4000,500),(4000,1000),(2000,350),(2000,500),(2000,750),(2000,1000),(4000,1000),(8000,350),(8000,500),(8000,250),(2500,350),(16000,350),(16000,500),(16000,1000),(32000,350),(32000,500),(32000,1000)]
#TABLE1=[(2000,1000),(1000,500)]
CONFIGS = TABLE1 + TABLE2




TEST_SCRIPT = "./scripts_GFLOPS/script_test_senflood_density_skip.py"
RESULT_RE = re.compile(
    r"RESULT tpl=(\d+) cross_k=(\d+) test_mIoU=([-\d.nan]+) "
    r"test_accuracy=([-\d.nan]+) gflops=([-\d.nan]+)")

parser = argparse.ArgumentParser(description="Drive density-generalization test sweep")
parser.add_argument("--ckpt", required=True, help="Single trained checkpoint to evaluate")
parser.add_argument("--only", type=int, nargs="+", default=None)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--python", type=str, default=sys.executable)
parser.add_argument("--continue_on_fail", action="store_true", default=True,
                    help="Keep going if a run fails (default True for eval).")
args = parser.parse_args()

indices = args.only if args.only is not None else list(range(len(CONFIGS)))
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"[driver] ckpt: {args.ckpt}")
print(f"[driver] {len(indices)} configs, start {stamp}\n")

# idx -> (tpl, ck, miou, acc, rc)
parsed = {}

for run_n, idx in enumerate(indices, 1):
    tpl, ck = CONFIGS[idx]
    table = 1 if idx < len(TABLE1) else 2
    xp = f"denseval_t{table}_tpl{tpl}_ck{ck}_{stamp}"

    cmd = [
        args.python, TEST_SCRIPT,
        "--ckpt", args.ckpt,
        "--xp_name", xp,
        "--tokens_per_latent", str(tpl),
        "--cross_k", str(ck),
    ]
    print("=" * 78)
    print(f"[driver] ({run_n}/{len(indices)}) idx={idx} table={table} "
          f"tpl={tpl} cross_k={ck}")
    print(f"[driver] cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)

    if args.dry_run:
        parsed[idx] = (tpl, ck, float("nan"), float("nan"), float("nan"), "DRY")
        continue

    # capture stdout so we can parse RESULT, but also stream it through
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
        miou   = float(m.group(3))
        acc    = float(m.group(4))
        gflops = float(m.group(5))
    parsed[idx] = (tpl, ck, miou, acc, gflops, proc.returncode)

    print(f"[driver] idx={idx} rc={proc.returncode} "
          f"test_mIoU={miou:.4f} gflops={gflops:.2f} ({dt/60:.1f} min)\n",
          flush=True)

    if proc.returncode != 0 and not args.continue_on_fail:
        print(f"[driver] idx={idx} failed; stopping.")
        break

# =============================================================================
# TABLES
# =============================================================================
def fmt(v):
    return "  nan  " if v != v else f"{v:.4f}"

print("\n" + "=" * 78)
print("DENSITY GENERALIZATION — TEST mIoU + GFLOPs (single checkpoint)")
print("=" * 78)

print("\nTable 1 — vary tokens_per_latent (cross_k=1000):")
print(f"  {'tpl':>6} {'cross_k':>8} {'test_mIoU':>10} {'test_acc':>10} {'GFLOPs':>10}")
print("  " + "-" * 50)
for idx in range(len(TABLE1)):
    if idx in parsed:
        tpl, ck, miou, acc, gflops, rc = parsed[idx]
        print(f"  {tpl:>6} {ck:>8} {fmt(miou):>10} {fmt(acc):>10} {fmt(gflops):>10}")

print("\nTable 2 — tpl=2000, vary cross_k:")
print(f"  {'tpl':>6} {'cross_k':>8} {'test_mIoU':>10} {'test_acc':>10} {'GFLOPs':>10}")
print("  " + "-" * 50)
for idx in range(len(TABLE1), len(CONFIGS)):
    if idx in parsed:
        tpl, ck, miou, acc, gflops, rc = parsed[idx]
        print(f"  {tpl:>6} {ck:>8} {fmt(miou):>10} {fmt(acc):>10} {fmt(gflops):>10}")

# CSV for plotting
os.makedirs("training/ablation_runs", exist_ok=True)
csv_path = f"training/ablation_runs/density_eval_{stamp}.csv"
with open(csv_path, "w") as f:
    f.write("table,idx,tokens_per_latent,cross_k,test_mIoU,test_accuracy,gflops,rc\n")
    for idx in sorted(parsed.keys()):
        tpl, ck, miou, acc, gflops, rc = parsed[idx]
        table = 1 if idx < len(TABLE1) else 2
        f.write(f"{table},{idx},{tpl},{ck},{miou},{acc},{gflops},{rc}\n")
print(f"\n[driver] CSV written to {csv_path}")
