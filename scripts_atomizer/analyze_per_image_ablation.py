"""
Automated analysis of per-image ablation results (zero_message_ablation.py --per_image)
=====================================================================================

Reads one or more CSVs (one row per image and condition) and reports, per dataset and
condition, how the effect of removing a part of the model is distributed across images.

    python scripts_atomizer/analyze_per_image_ablation.py \\
        --csv ablation_per_image_burnscars_test.csv ablation_per_image_pastis_test.csv \\
        --names burnscars pastis --class_of_interest 1 0 --out_dir ablation_analysis

For every dataset x condition:
  * changes per image (condition - baseline): accuracy, mIoU over present classes, and the
    fraction of pixels whose prediction changes: mean, std, median, 5/25/75/95% quantiles,
    bootstrap 95% interval of the mean
  * spread of the effect: images unaffected (|accuracy change| < --small pt), losing more
    than 2 / 10 pts, share of the total loss due to the 10% most affected images, Gini
    coefficient of the loss (0 = every image loses the same, 1 = one image takes it all),
    coefficient of variation
  * what it depends on: Pearson and Spearman correlations of the accuracy change with each
    class's share of the image and with the baseline accuracy; mean +- std of the change by
    quantile bins of the share of --class_of_interest
  * a verdict: does the need for the removed part vary between images?
Figures (PNG, --out_dir): histograms of the changes, change vs class share, Lorenz curves.
A markdown report (report.md) collects the tables.
"""
import argparse
import os

import numpy as np
import pandas as pd


# =============================================================================
# Statistics
# =============================================================================
def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def gini(loss):
    """Gini coefficient of non-negative losses (0 = equal, 1 = concentrated)."""
    x = np.sort(np.clip(np.asarray(loss, float), 0, None))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return np.nan
    return float(a[ok].rank().corr(b[ok].rank()))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def describe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return {k: np.nan for k in ("mean", "std", "median", "q05", "q25", "q75", "q95", "ci_lo", "ci_hi")}
    lo, hi = bootstrap_ci(x)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"mean": x.mean(), "std": x.std(ddof=1) if len(x) > 1 else 0.0, "median": q[2],
            "q05": q[0], "q25": q[1], "q75": q[3], "q95": q[4], "ci_lo": lo, "ci_hi": hi}


def analyze(df, small, coi, n_bins):
    """df: rows of one dataset and condition."""
    d_acc = (df["acc_cond"] - df["acc_base"]).to_numpy()
    d_miou = (df["miou_cond"] - df["miou_base"]).to_numpy()
    changed = df["pixels_changed"].to_numpy()
    loss = np.clip(-d_acc, 0, None)
    top = np.sort(loss)[::-1]
    k10 = max(1, len(top) // 10)
    res = {
        "n": len(df),
        "acc": describe(d_acc), "miou": describe(d_miou), "changed": describe(changed),
        "frac_unaffected": float(np.mean(np.abs(d_acc) < small)),
        "frac_lose2": float(np.mean(d_acc < -2)), "frac_lose10": float(np.mean(d_acc < -10)),
        "top10_share": float(top[:k10].sum() / top.sum()) if top.sum() > 0 else 0.0,
        "gini": gini(loss),
        "cv": float(np.std(d_acc, ddof=1) / abs(np.mean(d_acc))) if len(d_acc) > 1 and np.mean(d_acc) != 0 else np.nan,
    }
    shares = [c for c in df.columns if c.startswith("share_class")]
    res["corr"] = {c: (pearson(df[c], d_acc), spearman(df[c], d_acc)) for c in shares
                   if df[c].std() > 0}
    res["corr"]["acc_base"] = (pearson(df["acc_base"], d_acc), spearman(df["acc_base"], d_acc))
    col = f"share_class{coi}"
    bins = []
    if col in df.columns and df[col].nunique() > 1:
        try:
            cut = pd.qcut(df[col], q=n_bins, duplicates="drop")
        except ValueError:
            cut = None
        if cut is not None:
            for interval, g in df.groupby(cut, observed=True):
                da = (g["acc_cond"] - g["acc_base"]).to_numpy()
                bins.append((f"{interval.left:.1f}-{interval.right:.1f}%", len(g), da.mean(),
                             da.std(ddof=1) if len(da) > 1 else 0.0))
    res["bins"], res["coi"] = bins, coi
    return res, d_acc, d_miou, changed


def verdict(r, small):
    a = r["acc"]
    if abs(a["mean"]) < small and r["frac_unaffected"] > 0.9:
        return ("NO DEPENDENCE: almost every image is unaffected; the removed part is not used "
                "for this dataset")
    if r["frac_unaffected"] >= 0.5 and r["top10_share"] >= 0.5:
        return ("VARIES BETWEEN IMAGES: most images are unaffected and a minority carries most of the "
                "loss; a per-image decision (adaptive depth) has something to exploit")
    if r["gini"] < 0.35 and r["frac_unaffected"] < 0.25:
        return ("UNIFORM: nearly every image depends on the removed part to a similar degree; a "
                "fixed per-dataset choice is enough")
    return ("MIXED: the dependence varies but is spread over many images; inspect the "
            "correlations and bins to see what drives it")


# =============================================================================
# Figures
# =============================================================================
def figures(per_ds, out_dir, coi_of):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analysis] matplotlib not available: no figures")
        return []
    files = []
    keys = list(per_ds)
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        d_acc = per_ds[key]["d_acc"]
        ax.hist(d_acc, bins=40, color="#4C72B0", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(np.mean(d_acc), color="#C44E52", lw=1.2, ls="--", label=f"mean {np.mean(d_acc):+.2f}")
        ax.set_title(f"{key[0]} - {key[1]}", fontsize=10)
        ax.set_xlabel("accuracy change per image (pts)")
        ax.set_ylabel("images")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "hist_accuracy_change.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        df = per_ds[key]["df"]
        col = f"share_class{coi_of[key[0]]}"
        if col in df.columns:
            ax.scatter(df[col], per_ds[key]["d_acc"], s=10, alpha=0.6, color="#55A868")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel(f"share of class {coi_of[key[0]]} in the image (%)")
        ax.set_ylabel("accuracy change (pts)")
        ax.set_title(f"{key[0]} - {key[1]}", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "change_vs_class_share.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for key in keys:
        loss = np.sort(np.clip(-per_ds[key]["d_acc"], 0, None))
        if loss.sum() <= 0:
            continue
        cum = np.concatenate([[0], np.cumsum(loss) / loss.sum()])
        ax.plot(np.linspace(0, 1, len(cum)), cum, label=f"{key[0]} - {key[1]}")
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":", label="equal loss")
    ax.set_xlabel("fraction of images (least to most affected)")
    ax.set_ylabel("fraction of the total accuracy loss")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "lorenz_loss.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    return files


# =============================================================================
# Report
# =============================================================================
def fmt_desc(d):
    return (f"{d['mean']:+.2f} ± {d['std']:.2f} (95% CI {d['ci_lo']:+.2f} … {d['ci_hi']:+.2f}) | "
            f"median {d['median']:+.2f} | 5/25/75/95%: {d['q05']:+.2f} {d['q25']:+.2f} "
            f"{d['q75']:+.2f} {d['q95']:+.2f}")


def main():
    ap = argparse.ArgumentParser(description="Analyse per-image ablation CSVs")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None, help="one name per CSV (default: file names)")
    ap.add_argument("--class_of_interest", nargs="+", type=int, default=None,
                    help="per CSV: class whose image share is binned (default: 1)")
    ap.add_argument("--conditions", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--small", type=float, default=0.5, help="|accuracy change| below this = unaffected (pts)")
    ap.add_argument("--bins", type=int, default=4, help="quantile bins of the class share")
    ap.add_argument("--out_dir", default="ablation_analysis")
    args = ap.parse_args()
    names = args.names or [os.path.splitext(os.path.basename(c))[0] for c in args.csv]
    if len(names) != len(args.csv):
        raise SystemExit("--names needs one name per CSV")
    cois = args.class_of_interest or [1] * len(args.csv)
    if len(cois) == 1 and len(args.csv) > 1:
        cois = cois * len(args.csv)
    coi_of = dict(zip(names, cois))
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = None if args.conditions is None else {c.strip() for c in args.conditions.split(",")}

    per_ds, lines = {}, ["# Per-image ablation analysis", ""]
    overview = []
    for path, name in zip(args.csv, names):
        df_all = pd.read_csv(path)
        for cond, df in df_all.groupby("condition", sort=False):
            if wanted and cond not in wanted:
                continue
            r, d_acc, d_miou, changed = analyze(df.reset_index(drop=True), args.small, coi_of[name], args.bins)
            per_ds[(name, cond)] = {"df": df.reset_index(drop=True), "d_acc": d_acc}
            v = verdict(r, args.small)
            overview.append((name, cond, r, v))
            block = [f"## {name} — {cond} ({r['n']} images)", "",
                     f"- **accuracy change (pts):** {fmt_desc(r['acc'])}",
                     f"- **mIoU change (pts):** {fmt_desc(r['miou'])}",
                     f"- **pixels changed (%):** {fmt_desc(r['changed'])}",
                     f"- **spread:** {100 * r['frac_unaffected']:.1f}% unaffected (|change| < {args.small} pt), "
                     f"{100 * r['frac_lose2']:.1f}% lose > 2 pts, {100 * r['frac_lose10']:.1f}% lose > 10 pts; "
                     f"the 10% most affected carry {100 * r['top10_share']:.1f}% of the loss; "
                     f"Gini {r['gini']:.2f}; coefficient of variation {r['cv']:.2f}",
                     "", "| correlate | Pearson r | Spearman ρ |", "|---|---|---|"]
            for k, (pr, sr) in r["corr"].items():
                block.append(f"| {k} | {pr:+.2f} | {sr:+.2f} |")
            if r["bins"]:
                block += ["", f"Accuracy change by share of class {r['coi']}:", "",
                          "| share bin | images | mean change | std |", "|---|---|---|---|"]
                for b, n, m, s in r["bins"]:
                    block.append(f"| {b} | {n} | {m:+.2f} | {s:.2f} |")
            block += ["", f"**Verdict:** {v}", ""]
            lines += block
            print("\n".join(block).replace("**", ""))

    lines.insert(2, "| dataset | condition | images | mean ± std (acc, pts) | unaffected | top-10% share | Gini | verdict |")
    lines.insert(3, "|---|---|---|---|---|---|---|---|")
    for i, (name, cond, r, v) in enumerate(overview):
        lines.insert(4 + i, f"| {name} | {cond} | {r['n']} | {r['acc']['mean']:+.2f} ± {r['acc']['std']:.2f} | "
                            f"{100 * r['frac_unaffected']:.0f}% | {100 * r['top10_share']:.0f}% | "
                            f"{r['gini']:.2f} | {v.split(':')[0]} |")
    lines.insert(4 + len(overview), "")
    files = figures(per_ds, args.out_dir, coi_of)
    if files:
        lines += ["## Figures", ""] + [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in files]
    report = os.path.join(args.out_dir, "report.md")
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[analysis] report: {report}" + (f" | figures: {', '.join(files)}" if files else ""))


if __name__ == "__main__":
    main()
