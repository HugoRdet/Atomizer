"""
Recompute Pareto Front + Convex Hull from a saved sweep JSON
==============================================================

Standalone utility — no training/checkpoint needed. Point it at a JSON
file produced by run_senflood_density_eval.py / run_burnscars_ablation.py
and it recomputes the Pareto front (mIoU maximize, GFLOPs minimize) and
its convex hull (efficient frontier) from scratch, then writes both to a
plain-text file.

Why "recompute" rather than just re-printing what's already in the file:
useful when you want to re-derive the hull under a DIFFERENT cost metric
than whatever the original run used (e.g. force the tpl*cross_k proxy
even though real GFLOPs is available, or vice versa), or when you've
manually merged/edited points from multiple JSON files into one and want
a fresh front+hull over the combined set.

Accepts two JSON shapes (auto-detected):

  1. A pareto_front_*.json file (has "pareto_front" and/or "convex_hull"
     keys) — the "pareto_front" point list is used as the source of
     truth; the hull is recomputed from it (not just copied from
     "convex_hull", since the whole point of this script is fresh
     recomputation, e.g. under a different --cost_key).

  2. A density_eval_*.json full sweep results file (has a "runs" list,
     the format written by the driver's own RESULTS_TRACKING section) —
     points are rebuilt from "runs" directly, so this works even if you
     never got as far as computing a front/hull on the original run
     (e.g. an older run predating the Pareto/hull feature).

Usage:
    python recompute_pareto_hull.py --json ./scripts_GFLOPS/results/pareto_front_senflood_regular_val_20260815_120000.json
    python recompute_pareto_hull.py --json ./scripts_GFLOPS/results/density_eval_burnscars_regular_test_20260815_130000.json --out my_hull.txt
    python recompute_pareto_hull.py --json ... --cost_key cost_proxy   # force the tpl*cross_k proxy instead of gflops
"""

import argparse
import json
import os


# =============================================================================
# Pareto front + convex hull (identical logic to the drivers)
# =============================================================================

def _pareto_front(points, cost_key):
    """
    point A dominates point B iff A.mIoU >= B.mIoU AND A.cost <= B.cost,
    with at least one strict inequality. NaN mIoU/cost are excluded.
    """
    valid = [p for p in points
             if p.get("mIoU") == p.get("mIoU")  # drop NaN
             and p.get(cost_key) == p.get(cost_key)]
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
    """
    Upper convex hull in (cost, mIoU) space — the concave,
    diminishing-returns boundary. Points below the segment through their
    cost-neighbors on the front are dropped even if individually
    non-dominated. Monotone-chain (Andrew's algorithm), upper half only.
    """
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


# =============================================================================
# Loading points from either JSON shape
# =============================================================================

def _normalize_point(p, idx_fallback):
    """Fill in a 'cost_proxy' if missing but tpl/cross_k are present, and
    make sure every point has the keys the front/hull functions expect."""
    out = dict(p)
    if "idx" not in out:
        out["idx"] = idx_fallback
    if "cost_proxy" not in out and "tokens_per_latent" in out and "cross_k" in out:
        out["cost_proxy"] = out["tokens_per_latent"] * out["cross_k"]
    elif "cost_proxy" not in out and "tpl" in out and "cross_k" in out:
        out["cost_proxy"] = out["tpl"] * out["cross_k"]
    # unify tpl/tokens_per_latent naming so printing works regardless of
    # which JSON shape this came from
    if "tpl" not in out and "tokens_per_latent" in out:
        out["tpl"] = out["tokens_per_latent"]
    if "decoder_k_spatial" not in out and "dks" in out:
        out["decoder_k_spatial"] = out["dks"]
    return out


def load_points(data):
    """
    Returns (points, source_desc) where points is a list of point dicts
    with at least: idx, tpl, cross_k, decoder_k_spatial, mIoU, gflops,
    cost_proxy, type (when available).
    """
    if "pareto_front" in data and data["pareto_front"]:
        pts = [_normalize_point(p, i) for i, p in enumerate(data["pareto_front"])]
        return pts, "'pareto_front' key"

    if "runs" in data and data["runs"]:
        pts = [_normalize_point(p, i) for i, p in enumerate(data["runs"])]
        return pts, "'runs' key (full sweep results file)"

    # last resort: maybe the file IS just a flat list of points
    if isinstance(data, list):
        pts = [_normalize_point(p, i) for i, p in enumerate(data)]
        return pts, "flat JSON list"

    raise ValueError(
        "Could not find a point list in this JSON. Expected a "
        "'pareto_front' key (from pareto_front_*.json), a 'runs' key "
        "(from density_eval_*.json), or a flat JSON list of point dicts."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recompute the Pareto front + convex hull from a saved sweep JSON.")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to a pareto_front_*.json or density_eval_*.json file.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .txt path. Default: same name as --json "
                             "with '_recomputed_hull.txt' suffix, same directory.")
    parser.add_argument("--cost_key", type=str, default=None,
                        choices=("gflops", "cost_proxy"),
                        help="Force which cost metric to use. Default: "
                             "'gflops' if any point has a real (non-NaN) "
                             "gflops value, else falls back to "
                             "'cost_proxy' (tokens_per_latent * cross_k).")
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        raise FileNotFoundError(f"No such file: {args.json}")

    with open(args.json, "r") as f:
        data = json.load(f)

    points, source_desc = load_points(data)
    print(f"[recompute] Loaded {len(points)} points from {args.json} ({source_desc})")

    # Filter out points with NaN mIoU up front so the "have_real_gflops"
    # check below only looks at points that could plausibly appear on a
    # front at all.
    valid_points = [p for p in points if p.get("mIoU") == p.get("mIoU")]
    if len(valid_points) < len(points):
        print(f"[recompute] Dropped {len(points) - len(valid_points)} "
              f"point(s) with NaN mIoU.")

    have_real_gflops = any(
        p.get("gflops") == p.get("gflops") and p.get("gflops") is not None
        for p in valid_points
    )

    if args.cost_key is not None:
        cost_key = args.cost_key
        if cost_key == "gflops" and not have_real_gflops:
            print("[recompute][WARN] --cost_key gflops requested but no "
                  "point has a real GFLOPs value — front/hull will end up "
                  "empty. Consider --cost_key cost_proxy instead.")
    else:
        cost_key = "gflops" if have_real_gflops else "cost_proxy"

    cost_label = "gflops" if cost_key == "gflops" else "tpl*cross_k"
    print(f"[recompute] Using cost_key='{cost_key}' ({cost_label})")

    front = _pareto_front(valid_points, cost_key)
    hull = _upper_convex_hull(front, cost_key)
    hull.sort(key=lambda p: p[cost_key])

    print(f"[recompute] Pareto front: {len(front)} points "
          f"(from {len(valid_points)} valid input points)")
    print(f"[recompute] Convex hull:  {len(hull)} points")

    # =========================================================================
    # OUTPUT
    # =========================================================================
    if args.out is not None:
        out_path = args.out
    else:
        base, _ = os.path.splitext(args.json)
        out_path = f"{base}_recomputed_hull.txt"

    def _fmt(v):
        if v is None or v != v:  # None or NaN
            return "nan"
        return f"{v:.4f}"

    def _row(p):
        typ = p.get("type", "?")
        return (f"{p.get('idx', '?'):>5} {p.get('tpl', '?'):>6} "
                f"{p.get('cross_k', '?'):>8} {p.get('decoder_k_spatial', '?'):>9} "
                f"{_fmt(p.get('mIoU')):>10} {p.get(cost_key, float('nan')):>14.4g} "
                f"{typ:>10}")

    header = (f"{'idx':>5} {'tpl':>6} {'cross_k':>8} {'dec_k_sp':>9} "
              f"{'mIoU':>10} {cost_label:>14} {'type':>10}")
    sep = "-" * len(header)

    with open(out_path, "w") as f:
        f.write(f"Recomputed Pareto front + convex hull\n")
        f.write(f"Source JSON: {os.path.abspath(args.json)} ({source_desc})\n")
        f.write(f"Cost metric: {cost_key} ({cost_label})\n")
        f.write(f"Input points: {len(points)} total, {len(valid_points)} valid "
                f"(non-NaN mIoU)\n\n")

        f.write(f"PARETO FRONT ({len(front)} points, all non-dominated)\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for p in front:
            f.write(_row(p) + "\n")

        f.write(f"\nCONVEX HULL / EFFICIENT FRONTIER ({len(hull)} points, "
                f"the ones actually worth plotting/using)\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for p in hull:
            f.write(_row(p) + "\n")

        if hull:
            shortlist = " ".join(str(p.get("idx", "?")) for p in hull)
            f.write(f"\nHull indices (for --only): {shortlist}\n")

    print(f"[recompute] Written to {out_path}")


if __name__ == "__main__":
    main()
