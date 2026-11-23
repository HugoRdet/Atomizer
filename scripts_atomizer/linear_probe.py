"""
Linear probe on frozen Atomiser U-Net features
===============================================

Measures how much information about a dataset's labels a FROZEN model's
features carry: only a linear layer is trained, on per-pixel features read
from the model's own maps. Run it twice — with a trained checkpoint and with
--random_init (same architecture, untrained) — and compare: the difference is
what training contributed.

Per-pixel features (--features):
    raw          the pixel's own raw values: every date x band at the pixel (no model
                 features; the input baseline the learned features should beat)
    raw_cell     raw values averaged over the pixel's LEAF CELL (the aggregation unit
                 of d0 / e0): controls for spatial aggregation alone
    raw_window   raw values averaged over a k x k pixel window centred on the pixel
                 (--window, default 5): model-independent smoothing
    d0           decoder level 0, the state of the leaf cell owning the pixel (default)
    e0           encoder level 0 (leaf cell)
    hypercolumn  [E_0; E_1; ...; E_{S-1}] of the cells owning the pixel at every level

A pixel's owner at level l is its nearest level-l anchor (the Voronoi
assignment the hierarchy is built from). Features come from the same code
path as viz_unet_pca.py.

Protocol: up to --pix_per_image labelled pixels per training image are pooled,
standardised, and a linear classifier is trained with AdamW. Validation and
test are evaluated on ALL labelled pixels. mIoU follows the dataset's metric:
    planteur  mean over all 8 classes (background included; absent class = 0)
    pastis    mean over crop classes 1..18
Results are printed and saved to --out as JSON.

Usage (PASTIS joint-encoding checkpoint probed on PLANTEUR, and the control):
    python scripts_atomizer/linear_probe.py \
        --checkpoint ./checkpoints/pastis/<pastis-unet-legacy>.ckpt \
        --config ./training/configs/config_test-PASTIS-unet-legacy.yaml \
        --trained_on pastis --use_s1 --dataset planteur

    python scripts_atomizer/linear_probe.py --random_init \
        --config ./training/configs/config_test-PASTIS-unet-legacy.yaml \
        --trained_on pastis --use_s1 --dataset planteur
"""
import os
import sys
import json
import time
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from viz_unet_pca import (DATASETS, load_model, level_features, prepare_lookup_pastis,  # noqa: E402
                          to_batch, flat_to_per_date, site_grid)


# =============================================================================
# Per-pixel features
# =============================================================================
def site_owners(geom, level):
    """Nearest level-l anchor of every site (cached on the geometry)."""
    cache = getattr(geom, "_probe_owner", None)
    if cache is None:
        cache = geom._probe_owner = {}
    if level not in cache:
        xy, c = geom.site_xy.float(), geom.centers[level].float()
        cache[level] = torch.cat([torch.cdist(xy[s:s + 8192], c).argmin(-1)
                                  for s in range(0, xy.shape[0], 8192)], 0).cpu()
    return cache[level]


def site_labels(tokens, geom):
    """Label of every site (all atoms of a site share it)."""
    from training.atomiser.Atomiser_unet import QueryDecoder
    _, site_atoms, _ = QueryDecoder.site_tables(geom)
    return tokens[site_atoms[:, 0], 4].long().cpu()


@torch.no_grad()
def raw_pixel_values(model, batch, geom):
    """[n_sites, A]: the values of every atom at each pixel, in token order
    (date, band), i.e. the same ordering at every pixel of a regular raster.
    Masked atoms (e.g. S1 no-data) count as 0."""
    from training.atomiser.Atomiser_unet import QueryDecoder
    tokens, mask = model._gather_inputs(batch)
    tokens, mask = tokens[0], mask[0]
    if tokens.dim() == 3:                                        # per date [T, N, 8]
        T, N, _ = tokens.shape
        site_atoms, valid = model.decoder.temporal_site_tables(geom, T, N)
        vals, msk = tokens.reshape(T * N, -1)[:, 0], mask.reshape(T * N)
    else:
        _, site_atoms, valid = QueryDecoder.site_tables(geom)
        vals, msk = tokens[:, 0], mask
    out = torch.where(valid & ~msk[site_atoms], vals[site_atoms], torch.zeros_like(vals[site_atoms]))
    return out.float().cpu()


def cell_mean(x, owner):
    """Average the rows of x [n_sites, A] within each owner cell, then give every
    site its cell's mean."""
    n = int(owner.max()) + 1
    s = torch.zeros(n, x.shape[1]).index_add_(0, owner, x)
    c = torch.bincount(owner, minlength=n).clamp_min(1).unsqueeze(1).float()
    return (s / c)[owner]


def window_mean(x, geom, k):
    """Average x [n_sites, A] over a k x k window of pixels on the site grid
    (borders: mean over the pixels inside the image)."""
    r, c, H, W = site_grid(geom)
    img = torch.zeros(x.shape[1], H, W)
    img[:, r, c] = x.T
    occ = torch.zeros(1, H, W)
    occ[:, r, c] = 1.0
    pool = lambda t: torch.nn.functional.avg_pool2d(t.unsqueeze(0), k, stride=1, padding=k // 2,
                                                    count_include_pad=True)[0]
    sm = pool(img) / pool(occ).clamp_min(1e-6)
    return sm[:, r, c].T


@torch.no_grad()
def pixel_features(model, batch, mode, window=5):
    """-> features [n_sites, F] (float32, cpu), labels [n_sites]."""
    tokens, geom, enc, dec = level_features(model, batch)
    if mode == "raw":
        feats = raw_pixel_values(model, batch, geom)
    elif mode == "raw_cell":
        feats = cell_mean(raw_pixel_values(model, batch, geom), site_owners(geom, 0))
    elif mode == "raw_window":
        feats = window_mean(raw_pixel_values(model, batch, geom), geom, window)
    elif mode == "d0":
        if dec is None:
            raise ValueError("--features d0 needs a segmentation model (decoder)")
        feats = dec[0][site_owners(geom, 0)]
    elif mode == "e0":
        feats = enc[0][site_owners(geom, 0)]
    elif mode == "hypercolumn":
        feats = torch.cat([enc[l][site_owners(geom, l)] for l in range(len(enc))], -1)
    else:
        raise ValueError(mode)
    return feats.float(), site_labels(tokens, geom)


# =============================================================================
# Metric
# =============================================================================
def miou_from_confusion(conf, dataset):
    """conf [C, C] (rows = truth). PLANTEUR: all classes, absent -> 0.
    PASTIS: classes 1..18."""
    tp = conf.diag().double()
    union = conf.sum(0).double() + conf.sum(1).double() - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.zeros_like(tp))
    if dataset == "pastis":
        return float(iou[1:19].mean()), iou
    return float(iou.mean()), iou


# =============================================================================
# Main
# =============================================================================
def build_dataset(args, config, lookup, split):
    mod_name, cls_name, default_root = DATASETS[args.dataset]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    root = args.data_root or default_root
    if args.dataset == "planteur":
        return cls(root_path=root, mode=split, config_model=config, look_up=lookup,
                   use_s1=args.use_s1, split_protocol=args.split_protocol,
                   temporal_window=args.temporal_window)
    if args.dataset == "pastis":
        return cls(root_path=root, mode=split, config_model=config, look_up=lookup,
                   use_s1=args.use_s1, use_spot=False)
    raise ValueError(f"--dataset {args.dataset}: only planteur and pastis are supported here")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="planteur", choices=("planteur", "pastis"))
    ap.add_argument("--trained_on", default="pastis", choices=("pastis", "planteur"))
    ap.add_argument("--use_s1", action="store_true")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split_protocol", default="default")
    ap.add_argument("--temporal_window", default="2019", choices=("2019", "agri", "all"))
    ap.add_argument("--features", default="d0",
                    choices=("raw", "raw_cell", "raw_window", "d0", "e0", "hypercolumn"))
    ap.add_argument("--window", type=int, default=5, help="raw_window: odd window size in pixels")
    ap.add_argument("--pix_per_image", type=int, default=1024)
    ap.add_argument("--train_num", type=int, default=0, help="training images used (0 = all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--class_weight", default="inverse", choices=("none", "inverse", "sqrt"),
                    help="loss weights per class from training pixel counts: inverse "
                         "frequency (default), its square root, or none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--out", default="./figures")
    args = ap.parse_args()
    if args.checkpoint is None and not args.random_init:
        raise SystemExit("--checkpoint is required (or use --random_init)")
    if args.window % 2 == 0:
        raise SystemExit("--window must be odd (the window is centred on the pixel)")

    from training.utils import read_yaml, Lookup_encoding
    config = read_yaml(args.config)
    per_date = config.get("Atomiser", {}).get("unet_temporal", "none") == "per_level"
    if per_date and args.dataset != "planteur":
        raise SystemExit("per-level temporal checkpoints: only --dataset planteur is "
                         "supported here (flat samples reshaped per date)")
    T_cfg = int(config.get("dataset", {}).get("multi_temporal", 10))
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config)
    prepare_lookup_pastis(lookup, args.use_s1)                   # before the model, as in training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, lookup, args.checkpoint, device,
                       random_init=args.random_init, seed=args.seed)

    datasets = {s: build_dataset(args, config, lookup, s) for s in ("train", "validation", "test")}

    def batch_of(ds, i):
        sample = ds[i]
        if per_date:
            sample = flat_to_per_date(sample, T_cfg, ds.NUM_S2_BANDS,
                                      ds.NUM_S1_BANDS if args.use_s1 else 0, lookup)
        return to_batch(sample, device)
    n_cls = int(datasets["train"].NUM_CLASSES)
    rng = np.random.default_rng(args.seed)
    tag = "random_init" if args.random_init else os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"[probe] {tag} | features={args.features} | dataset={args.dataset} "
          f"({args.split_protocol}) | classes={n_cls}")

    # ---- 1. training pixels -------------------------------------------------
    ds = datasets["train"]
    idx = list(range(len(ds)))
    if args.train_num and args.train_num < len(idx):
        idx = sorted(rng.choice(len(ds), size=args.train_num, replace=False).tolist())
    X, Y, t0 = [], [], time.time()
    for n, i in enumerate(idx):
        f, y = pixel_features(model, batch_of(ds, i), args.features, args.window)
        ok = torch.nonzero(y != 255).squeeze(1)
        if ok.numel() == 0:
            continue
        sel = ok[torch.from_numpy(rng.permutation(ok.numel())[:args.pix_per_image])]
        X.append(f[sel].half()); Y.append(y[sel])
        if (n + 1) % 100 == 0:
            print(f"[probe] train features {n + 1}/{len(idx)} ({time.time() - t0:.0f}s)")
    X, Y = torch.cat(X).float(), torch.cat(Y)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd
    print(f"[probe] {X.shape[0]} training pixels, {X.shape[1]}-d features; class counts "
          f"{torch.bincount(Y, minlength=n_cls).tolist()}")

    # ---- 2. linear classifier ---------------------------------------------
    torch.manual_seed(args.seed)
    counts = torch.bincount(Y, minlength=n_cls).float()
    if args.class_weight == "none":
        weight = None
    else:
        w = 1.0 / counts.clamp_min(1)
        if args.class_weight == "sqrt":
            w = w.sqrt()
        w = torch.where(counts > 0, w, torch.zeros_like(w))
        weight = (w / (w * counts).sum() * counts.sum()).to(device)  # mean weight per pixel = 1
        print(f"[probe] class weights ({args.class_weight}): "
              f"{[round(float(v), 3) for v in weight]}")
    probe = nn.Linear(X.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Xd, Yd = X.to(device), Y.to(device)
    for ep in range(args.epochs):
        perm = torch.randperm(Xd.shape[0], device=device)
        tot = 0.0
        for s in range(0, Xd.shape[0], args.batch):
            b = perm[s:s + args.batch]
            loss = nn.functional.cross_entropy(probe(Xd[b]), Yd[b], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * b.numel()
        sched.step()
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"[probe] epoch {ep + 1}/{args.epochs}: train loss {tot / Xd.shape[0]:.4f}")
    del Xd, Yd

    # ---- 3. evaluation on all labelled pixels ------------------------------
    results = {"checkpoint": tag, "features": args.features, "dataset": args.dataset,
               "split_protocol": args.split_protocol, "train_pixels": int(X.shape[0])}
    names = getattr(datasets["train"], "CLASS_NAMES", [f"class_{c}" for c in range(n_cls)])
    mu_d, sd_d = mu.to(device), sd.to(device)
    for split in ("validation", "test"):
        ds = datasets[split]
        conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
        with torch.no_grad():
            for i in range(len(ds)):
                f, y = pixel_features(model, batch_of(ds, i), args.features, args.window)
                ok = y != 255
                if not ok.any():
                    continue
                pred = probe((f[ok].to(device) - mu_d) / sd_d).argmax(-1).cpu()
                conf += torch.bincount(y[ok] * n_cls + pred, minlength=n_cls * n_cls).view(n_cls, n_cls)
        miou, iou = miou_from_confusion(conf, args.dataset)
        acc = float(conf.diag().sum() / conf.sum().clamp_min(1))
        present = conf.sum(1) > 0
        recall = conf.diag().double() / conf.sum(1).double().clamp_min(1)
        mrec = float(recall[present].mean())                   # mean per-class recall
        results[split] = {"mIoU": miou, "pixel_accuracy": acc, "mean_class_recall": mrec,
                          "per_class_IoU": {names[c]: float(iou[c]) for c in range(n_cls)},
                          "per_class_recall": {names[c]: float(recall[c]) for c in range(n_cls)},
                          "pixels_per_class": {names[c]: int(conf.sum(1)[c]) for c in range(n_cls)}}
        print(f"\n[probe] {split}: mIoU {100 * miou:.2f} | mean class recall {100 * mrec:.2f} "
              f"| pixel accuracy {100 * acc:.2f}")
        for c in range(n_cls):
            print(f"         {names[c]:>20s}: IoU {100 * float(iou[c]):6.2f} | "
                  f"recall {100 * float(recall[c]):6.2f} | {int(conf.sum(1)[c])} px")

    os.makedirs(args.out, exist_ok=True)
    results["class_weight"] = args.class_weight
    feat_tag = args.features + (f"{args.window}" if args.features == "raw_window" else "")
    results["window"] = args.window if args.features == "raw_window" else None
    path = os.path.join(args.out, f"probe_{args.dataset}_{feat_tag}_{args.class_weight}_{tag}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[probe] saved {path}")


if __name__ == "__main__":
    main()
