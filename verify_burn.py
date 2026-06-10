"""
Atomizer Checkpoint Health Diagnostic (v2)
============================================

Loads multiple checkpoints from a training run and examines the parameters
most likely to cause attention entropy collapse, with improved self-attention
detection and per-layer reporting.

Updates from v1:
    - Better substring matching for self-attention (handles 'latent_attn',
      'latent_self_attn', 'latent_to_latent', and nested encoder_layers
      naming conventions)
    - Per-layer max_abs growth: see WHICH specific parameter inflated, not
      just the category aggregate
    - Concentration analysis: what fraction of total category growth is in
      the top-K most-changing parameters (concentrated = one runaway layer;
      diffuse = systemic growth)
    - Always reports unclassified parameter count (with sample names) so
      you can fix the classifier if needed

Usage:
    python check_qk_health.py \\
        --ckpts checkpoints/fractal/*.ckpt \\
        --csv qk_health.csv
"""

import argparse
import os
import re
from collections import defaultdict

import torch
import pandas as pd


def epoch_from_name(path: str) -> int:
    """Pull the epoch number out of a checkpoint filename."""
    base = os.path.basename(path)
    m = re.search(r"epoch=(\d+)", base)
    return int(m.group(1)) if m else -1


def load_state(path: str) -> dict:
    """Load a Lightning checkpoint's state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def classify_param(key: str) -> str:
    """
    Classify a parameter by inspecting its name.

    Returns one of:
        'cross_q', 'cross_k', 'cross_v',
        'self_q',  'self_k',  'self_v',
        'rope_scale', 'norm', 'other'

    Strategy:
    1. Check for RoPE / norm patterns first (most specific)
    2. Then look for self-attention patterns (multiple naming conventions)
    3. Then cross-attention patterns
    4. Within each attention type, identify Q/K/V via projection name patterns

    Self-attention is harder to detect because in lucidrains-style code, it's
    often named generically (e.g. "latent_attn", "to_latent_attn") rather
    than explicitly "self_attn". We use multiple fallback patterns.
    """
    k_lower = key.lower()

    # ── RoPE scales (your config has rope_learnable_scale=true) ───────
    if "rope" in k_lower and ("scale" in k_lower or "freq" in k_lower):
        return "rope_scale"

    # ── Norms (LayerNorm / RMSNorm) ─────────────────────────────────
    if any(p in k_lower for p in ("layernorm", "rmsnorm", "layer_norm",
                                    "rms_norm")):
        return "norm"
    # Be careful with bare "norm": match only when it's clearly a norm
    # parameter (typically .norm.weight or .norm.bias)
    if (".norm." in k_lower or k_lower.endswith(".norm.weight")
            or k_lower.endswith(".norm.bias")):
        return "norm"

    # ── Attention type detection ─────────────────────────────────────
    # Order matters: more specific patterns first
    is_self  = any(p in k_lower for p in (
        "self_attn",         # explicit self-attention
        "self_attention",
        "latent_self_attn",
        "latent_self_attention",
        "latent_attn",       # lucidrains convention: latent-to-latent attn
        "latent_attention",
        "latent_to_latent",
    ))
    is_cross = any(p in k_lower for p in (
        "cross_attn",
        "cross_attention",
        "decoder_cross_attn",
        "encoder_cross_attn",
    ))

    # If neither pattern matched, try to disambiguate from context. In
    # some codebases attention layers are nested under "encoder_layers"
    # without explicit "cross"/"self" tags. We can't reliably tell which
    # without more context, so return 'other' rather than guessing wrong.
    if not (is_self or is_cross):
        return "other"

    # Both flagged? Prefer cross since that's more specific phrasing
    # (e.g. "latent_cross_attn" contains both "latent" and "cross").
    if is_cross:
        attn_type = "cross"
    else:
        attn_type = "self"

    # ── Q / K / V detection within the attention block ──────────────
    q_patterns = ("to_q", ".q_proj", ".q.weight", ".q.bias",
                  "wq.weight", "query_proj", "query.weight",
                  "q_linear")
    k_patterns = ("to_k", ".k_proj", ".k.weight", ".k.bias",
                  "wk.weight", "key_proj", "key.weight",
                  "k_linear")
    v_patterns = ("to_v", ".v_proj", ".v.weight", ".v.bias",
                  "wv.weight", "value_proj", "value.weight",
                  "v_linear")

    if any(p in k_lower for p in q_patterns):
        return f"{attn_type}_q"
    if any(p in k_lower for p in k_patterns):
        return f"{attn_type}_k"
    if any(p in k_lower for p in v_patterns):
        return f"{attn_type}_v"

    return "other"


def analyze_checkpoint(path: str) -> dict:
    """
    Analyze a single checkpoint and return a dict of statistics.

    Returns dict with: epoch, path, param_stats (per-param dict),
                       category_aggregates, unclassified_sample
    """
    state = load_state(path)
    epoch = epoch_from_name(path)

    param_stats = {}
    category_values = defaultdict(list)
    unclassified_names = []

    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        if tensor.numel() == 0:
            continue

        tensor_f32 = tensor.float()
        category = classify_param(key)

        # std() warns on single-element tensors; guard it
        std_val = (tensor_f32.std().item() if tensor.numel() > 1 else 0.0)

        stats = {
            "category": category,
            "shape":    list(tensor.shape),
            "norm":     tensor_f32.norm().item(),
            "max_abs":  tensor_f32.abs().max().item(),
            "mean_abs": tensor_f32.abs().mean().item(),
            "std":      std_val,
            "numel":    tensor.numel(),
        }
        param_stats[key] = stats

        if category != "other":
            category_values[category].append((key, stats))
        elif len(unclassified_names) < 20:
            # Keep a sample of unclassified params for diagnostic
            unclassified_names.append(key)

    # ── Category aggregates ─────────────────────────────────────────
    category_aggregates = {}
    for cat, key_stats_list in category_values.items():
        stats_list = [s for _, s in key_stats_list]
        if not stats_list:
            continue
        category_aggregates[cat] = {
            "num_params":    len(stats_list),
            "mean_norm":     sum(s["norm"] for s in stats_list) / len(stats_list),
            "max_norm":      max(s["norm"] for s in stats_list),
            "max_abs_value": max(s["max_abs"] for s in stats_list),
            "mean_mean_abs": sum(s["mean_abs"] for s in stats_list) / len(stats_list),
        }

    return {
        "epoch":               epoch,
        "path":                path,
        "param_stats":         param_stats,
        "category_aggregates": category_aggregates,
        "unclassified_sample": unclassified_names,
        "total_params":        len(param_stats),
    }


def print_category_comparison(analyses: list):
    """Print category-level aggregate comparison across checkpoints."""
    analyses_sorted = sorted(analyses, key=lambda a: a["epoch"])

    print("\n" + "=" * 88)
    print("CATEGORY-LEVEL AGGREGATES")
    print("=" * 88)

    all_categories = set()
    for a in analyses_sorted:
        all_categories.update(a["category_aggregates"].keys())

    # Order categories for readable output: cross then self then rope/norm
    cat_order = [
        "cross_q", "cross_k", "cross_v",
        "self_q",  "self_k",  "self_v",
        "rope_scale", "norm",
    ]
    all_categories = [c for c in cat_order if c in all_categories]

    if not all_categories:
        print("\nNo Q/K/V/RoPE parameters detected.")
        print("Check the UNCLASSIFIED PARAMETERS section for actual names.")
        return

    header_epochs = [f"ep{a['epoch']:02d}" for a in analyses_sorted]
    print(f"\n{'Category':<14} {'Metric':<15} " +
          " ".join(f"{h:>12}" for h in header_epochs) +
          "    " + "Δ (last/first)")
    print("-" * 88)

    for cat in all_categories:
        for metric in ("mean_norm", "max_norm", "max_abs_value", "mean_mean_abs"):
            values = []
            for a in analyses_sorted:
                agg = a["category_aggregates"].get(cat, {})
                values.append(agg.get(metric))

            valid = [v for v in values if v is not None]
            if len(valid) < 2:
                continue

            value_strs = [
                f"{v:>12.4f}" if v is not None else f"{'—':>12}"
                for v in values
            ]
            ratio_str = ""
            if values[0] and values[-1]:
                ratio = values[-1] / values[0]
                if ratio > 2.0:
                    ratio_str = f"  ⚠ {ratio:.2f}x"
                elif ratio > 1.5:
                    ratio_str = f"   ◔ {ratio:.2f}x"
                else:
                    ratio_str = f"     {ratio:.2f}x"

            print(f"{cat:<14} {metric:<15} " + " ".join(value_strs) + ratio_str)
        print()

    print("=" * 88)
    print("Interpretation:")
    print("  ratio < 1.5x        : healthy, weights stable")
    print("  ratio 1.5x - 2.0x   : ◔  watch but probably OK")
    print("  ratio > 2.0x        : ⚠  suspicious, possible attention sharpening")
    print()
    print("  Q+K growth WITHOUT V growth → classic entropy-collapse signature.")
    print("  CROSS Q/K growing, SELF Q/K stable → cross-attention is the issue.")
    print("=" * 88)


def print_per_param_growth(analyses: list, top_n: int = 15):
    """
    For each Q/K parameter, compute max_abs growth from first to last
    checkpoint and print the top-N most-changed parameters.

    This tells you if the inflation is concentrated in a single layer/head
    or spread across many.
    """
    analyses_sorted = sorted(analyses, key=lambda a: a["epoch"])
    if len(analyses_sorted) < 2:
        return

    first = analyses_sorted[0]
    last  = analyses_sorted[-1]

    # Collect parameters present in both checkpoints
    common_keys = set(first["param_stats"].keys()) & set(last["param_stats"].keys())

    growth_rows = []
    for k in common_keys:
        cat = first["param_stats"][k]["category"]
        if cat not in ("cross_q", "cross_k", "cross_v",
                       "self_q",  "self_k",  "self_v"):
            continue
        v0 = first["param_stats"][k]["max_abs"]
        v1 = last ["param_stats"][k]["max_abs"]
        if v0 < 1e-9:
            continue
        ratio = v1 / v0
        growth_rows.append({
            "param": k,
            "category": cat,
            f"ep{first['epoch']:02d}_max_abs": v0,
            f"ep{last['epoch']:02d}_max_abs":  v1,
            "ratio": ratio,
        })

    if not growth_rows:
        return

    # Sort by ratio descending (largest inflation first)
    growth_rows.sort(key=lambda r: r["ratio"], reverse=True)

    print("\n" + "=" * 88)
    print(f"TOP-{top_n} PARAMETERS BY max_abs GROWTH (ep{first['epoch']:02d} → "
          f"ep{last['epoch']:02d})")
    print("=" * 88)
    print(f"{'Category':<12} {'Ratio':>8}   "
          f"{'Before':>10}   {'After':>10}   Parameter")
    print("-" * 88)
    for row in growth_rows[:top_n]:
        before_key = [k for k in row if k.startswith("ep") and "_max_abs" in k][0]
        after_key  = [k for k in row if k.startswith("ep") and "_max_abs" in k][-1]
        flag = "⚠" if row["ratio"] > 2.0 else ("◔" if row["ratio"] > 1.5 else " ")
        print(f"{row['category']:<12} {row['ratio']:>7.2f}x {flag} "
              f"{row[before_key]:>10.4f}   {row[after_key]:>10.4f}   "
              f"{row['param']}")
    print("=" * 88)


def print_concentration_analysis(analyses: list):
    """
    For each Q/K category, compute what fraction of total growth comes from
    the top-1, top-3, and top-5 parameters.

    Concentrated growth (top-1 accounts for >50% of growth) suggests a
    specific layer is the problem and targeted fixes (or layer-specific
    QK-norm) might work.

    Diffuse growth (no single param dominates) suggests systemic issue
    requiring architectural fix.
    """
    analyses_sorted = sorted(analyses, key=lambda a: a["epoch"])
    if len(analyses_sorted) < 2:
        return

    first = analyses_sorted[0]
    last  = analyses_sorted[-1]

    common_keys = set(first["param_stats"].keys()) & set(last["param_stats"].keys())

    # Group by category
    growth_by_cat = defaultdict(list)
    for k in common_keys:
        cat = first["param_stats"][k]["category"]
        if cat not in ("cross_q", "cross_k", "cross_v",
                       "self_q",  "self_k",  "self_v"):
            continue
        v0 = first["param_stats"][k]["max_abs"]
        v1 = last ["param_stats"][k]["max_abs"]
        delta = v1 - v0  # absolute growth
        if delta > 0:
            growth_by_cat[cat].append(delta)

    if not growth_by_cat:
        return

    print("\n" + "=" * 88)
    print("GROWTH CONCENTRATION (where in each category is the inflation?)")
    print("=" * 88)
    print(f"{'Category':<12} {'#params':>8} {'total Δ':>10}   "
          f"{'top-1 share':>12} {'top-3 share':>12} {'top-5 share':>12}")
    print("-" * 88)

    for cat in ("cross_q", "cross_k", "cross_v",
                "self_q",  "self_k",  "self_v"):
        deltas = growth_by_cat.get(cat, [])
        if not deltas:
            continue
        deltas_sorted = sorted(deltas, reverse=True)
        total = sum(deltas_sorted)
        if total == 0:
            continue
        top1 = deltas_sorted[0] / total
        top3 = sum(deltas_sorted[:3]) / total
        top5 = sum(deltas_sorted[:5]) / total
        print(f"{cat:<12} {len(deltas):>8} {total:>10.4f}   "
              f"{top1:>11.1%}  {top3:>11.1%}  {top5:>11.1%}")

    print("-" * 88)
    print("Reading the concentration:")
    print("  top-1 > 50%   : highly concentrated → one specific layer is the issue")
    print("  top-1 20-50%  : moderately concentrated → a few layers driving it")
    print("  top-1 < 20%   : diffuse → systemic, multiple layers all growing")
    print()
    print("  Concentrated → layer-specific QK-norm might suffice")
    print("  Diffuse      → full QK-norm needed for robustness")
    print("=" * 88)


def print_rope_detail(analyses: list):
    """Print every individual RoPE scale value across checkpoints."""
    analyses_sorted = sorted(analyses, key=lambda a: a["epoch"])

    rope_keys = set()
    for a in analyses_sorted:
        for k, s in a["param_stats"].items():
            if s["category"] == "rope_scale":
                rope_keys.add(k)

    if not rope_keys:
        return

    print("\n" + "=" * 88)
    print("ROPE LEARNABLE SCALE DETAIL")
    print("=" * 88)
    print(f"{'Parameter':<60} " +
          " ".join(f"{f'ep{a['epoch']:02d}':>10}" for a in analyses_sorted))
    print("-" * 88)
    for k in sorted(rope_keys):
        values = []
        for a in analyses_sorted:
            s = a["param_stats"].get(k)
            values.append(s["max_abs"] if s else None)
        value_strs = [
            f"{v:>10.4f}" if v is not None else f"{'—':>10}"
            for v in values
        ]
        print(f"{k[:60]:<60} " + " ".join(value_strs))
    print("=" * 88)


def print_unclassified_summary(analyses: list, show_full: bool = False):
    """
    Show how many params couldn't be classified, with a sample.

    If many params are 'other' but they look like attention modules, the
    classifier needs better substrings.
    """
    analyses_sorted = sorted(analyses, key=lambda a: a["epoch"])
    first = analyses_sorted[0]

    other_count = sum(1 for s in first["param_stats"].values()
                      if s["category"] == "other")
    total = first["total_params"]

    print("\n" + "=" * 88)
    print(f"UNCLASSIFIED PARAMETERS: {other_count} / {total} "
          f"({other_count / total:.1%})")
    print("=" * 88)

    if other_count == 0:
        print("All parameters classified.")
        return

    sample = first["unclassified_sample"]
    print(f"\nSample of unclassified params (showing up to {len(sample)}):")
    for k in sample:
        s = first["param_stats"][k]
        print(f"  {k}  shape={s['shape']}")

    if show_full:
        print("\nFull list of unclassified params:")
        for k, s in sorted(first["param_stats"].items()):
            if s["category"] == "other":
                print(f"  {k}  shape={s['shape']}")
    else:
        print("\nUse --show-unclassified to see the full list.")
        print("If you see attention-related params here (e.g. containing")
        print("'attn', 'attention', 'to_q', 'to_k'), update classify_param()")
        print("to match your codebase's naming.")
    print("=" * 88)


def export_csv(analyses: list, path: str):
    """Export per-parameter stats to a CSV for plotting/analysis."""
    rows = []
    for a in analyses:
        for k, s in a["param_stats"].items():
            rows.append({
                "epoch":    a["epoch"],
                "param":    k,
                "category": s["category"],
                "norm":     s["norm"],
                "max_abs":  s["max_abs"],
                "mean_abs": s["mean_abs"],
                "std":      s["std"],
                "numel":    s["numel"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nFull per-parameter stats written to: {path}")
    print(f"  {len(rows)} rows ({df['param'].nunique()} unique params, "
          f"{df['epoch'].nunique()} checkpoints)")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Diagnose Atomizer attention health")
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Paths to checkpoints (glob-expanded by shell)")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional CSV output for plotting")
    p.add_argument("--show-unclassified", action="store_true",
                   help="Print FULL list of unclassified parameters")
    p.add_argument("--top-n", type=int, default=15,
                   help="Number of top-changed parameters to show "
                        "(default: 15)")
    args = p.parse_args()

    paths = sorted(set(args.ckpts), key=epoch_from_name)
    if not paths:
        print("No checkpoints given.")
        return

    print("\nAnalyzing checkpoints:")
    for p_ in paths:
        print(f"  ep{epoch_from_name(p_):02d}  {p_}")

    analyses = []
    for path in paths:
        print(f"\nLoading {path} ...")
        analyses.append(analyze_checkpoint(path))

    # Always show unclassified summary first (cheapest signal of whether
    # the classifier is working)
    print_unclassified_summary(analyses, show_full=args.show_unclassified)

    print_category_comparison(analyses)
    print_per_param_growth(analyses, top_n=args.top_n)
    print_concentration_analysis(analyses)
    print_rope_detail(analyses)

    if args.csv:
        export_csv(analyses, args.csv)


if __name__ == "__main__":
    main()
