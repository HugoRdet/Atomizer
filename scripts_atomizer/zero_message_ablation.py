"""
Zero-message ablation of a trained Atomiser_Unet (overlap decoder)
==================================================================
Datasets: pastis (default), burnscars, cashew, and the other single-dataset
datasets of viz_unet_pca.py (--dataset); e.g. --dataset burnscars with the
BurnScars checkpoint and its config.

Evaluates ONE checkpoint, without retraining, with parts of the decoder removed:

    baseline        the model as trained
    no_coarse       the message into level 0 removed: nothing from levels 1..S-1 reaches the
                    output (D0 = refined E0). (Earlier versions called this cut_L{l} for every l:
                    zeroing the message into level 0 alone already removes all coarse context,
                    so those conditions were identical.)
    keep_L0-{l}     levels above l removed: only the message into level l is zeroed, the finer
                    stages run as usual (keep_L0-{S-2}: no bottleneck). Earlier name: only_L{l}
    const_gate      every gate replaced by its per-channel mean (measured on the same split
                    in a first pass): same average scaling, no variation across cells.
                    No change -> the gate only scales; a drop -> it selects
    drop_skip_L{l}  intermediate level l (0 < l < S-1): the encoder skip E_l removed from the
                    decoder, which keeps relaying the coarse message: D_l = g_mean * m_l (the gate,
                    normally computed FROM the skip, replaced by its per-channel mean). The
                    encoder's upward path through E_l is unchanged
    drop_skips      the same at every intermediate level at once. No change -> the intermediate
                    levels only relay the bottleneck's context; a drop -> they add their own
    no_obs          the query decoder's direct read of each pixel's raw atoms removed

At a fusion, D_l = s_l + gate * m_l; "removing the message" sets m_l = 0 (D_l = s_l, then the
level's FFN / refinement blocks run as usual).

mIoU: PASTIS as in its trainer, over crop classes 1..18 (background 0 excluded, void 19
ignored); other datasets over all classes (trainer.num_classes). Labels outside the valid
classes (e.g. 255) are ignored. Also reported: mIoU over all classes, pixel accuracy.

Per-image statistics (--per_image): for the conditions in --per_image_conds (default
no_coarse), every image's pixel accuracy, mIoU over the classes present (labels or
predictions), the fraction of pixels whose prediction changes, and its class shares are
recorded; the distribution of the changes across images is printed and written to --csv.
Does the need for the coarse levels vary between images of one dataset?

Caveat: a model never trained without these parts may degrade more than a model trained
without them would, so drops measure what THIS model relies on (an upper bound on how
much the parts are needed).

    python scripts_atomizer/zero_message_ablation.py \\
        --checkpoint ./checkpoints/pastis/<ckpt>.ckpt \\
        --config ./training/configs/config_test-PASTIS-unet-legacy-temporal.yaml \\
        --use_s1 --split validation
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import argparse
import importlib
import types

import torch

from viz_unet_pca import DATASETS, PASTIS_FAMILY, prepare_lookup_pastis, load_model, to_batch

N_CLASSES = 19                     # set in main() from the config
VOID = 19                          # PASTIS void label (ignored); None for other datasets
EXCLUDE = (0,)                     # classes left out of the main mIoU (PASTIS: background)


# =============================================================================
# Ablation hooks on the decoder's fusions and the query decoder
# =============================================================================
def install_hooks(model):
    fusions = list(model.unet.fuse)
    if not fusions:
        raise SystemExit("this model has no fusion stages (use the overlap decoder)")
    for f in fusions:
        f._mode, f._const, f._gsum, f._gcnt = "on", None, None, 0

        def fwd(self, skip, msg):
            if self._mode == "zero":
                return skip
            if self._mode == "dropskip":                             # no E_l: relay the message only
                if self.mode == "add":
                    return msg
                return self._const.to(msg.dtype).view(1, 1, -1) * msg
            if self.mode == "add":
                return skip + msg
            inp = torch.cat([skip, msg], -1) if self.mode == "gate_sm" else skip
            g = torch.sigmoid(self.lin(self.ln(inp)))
            if self._mode == "collect":                              # per-channel mean of the gate
                s = g.detach().float().reshape(-1, g.shape[-1]).sum(0)
                self._gsum = s if self._gsum is None else self._gsum + s
                self._gcnt += g.shape[0] * g.shape[1]
            elif self._mode == "const":
                g = self._const.to(g.dtype).view(1, 1, -1).expand_as(g)
            self.last_gate_mean = g.detach().float().mean()
            return skip + g * msg

        f.forward = types.MethodType(fwd, f)
    dec = model.decoder
    dec._obs_w = (dec.obs_o.weight.data.clone(), dec.obs_o.bias.data.clone())
    return fusions


def set_condition(model, fusions, zero_levels=(), gate=None, no_obs=False, drop_skip=()):
    """zero_levels: fusion indices whose message is removed; drop_skip: fusion indices whose
    skip is removed (message relayed with the mean gate); gate: None | 'collect' | 'const'."""
    for l, f in enumerate(fusions):
        if l in drop_skip:
            f._mode = "dropskip"
        elif l in zero_levels:
            f._mode = "zero"
        elif gate == "collect" and f.mode != "add":
            f._mode = "collect"
        elif gate == "const" and f.mode != "add":
            f._mode = "const"
        else:
            f._mode = "on"
    w, b = model.decoder._obs_w
    if no_obs:
        model.decoder.obs_o.weight.data.zero_()
        model.decoder.obs_o.bias.data.zero_()
    else:
        model.decoder.obs_o.weight.data.copy_(w)
        model.decoder.obs_o.bias.data.copy_(b)


# =============================================================================
# Evaluation
# =============================================================================
def confusion(pred, lab, mask):
    v = (~mask) & (lab >= 0) & (lab < N_CLASSES)
    if VOID is not None:
        v &= lab != VOID
    return torch.bincount(lab[v] * N_CLASSES + pred[v], minlength=N_CLASSES ** 2).view(
        N_CLASSES, N_CLASSES).cpu()


def metrics(cm):
    tp = cm.diag().double()
    union = cm.sum(0).double() + cm.sum(1).double() - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.zeros_like(tp))
    keep = torch.ones(len(iou), dtype=torch.bool)
    for c in EXCLUDE:
        keep[c] = False
    return {"mIoU": float(iou[keep].mean()), "mIoU_all": float(iou.mean()), "iou": iou,
            "acc": float(tp.sum() / cm.sum().clamp_min(1))}


def image_stats(pred, lab, mask):
    """One image: dict of accuracy, mIoU over present classes, class shares, valid count."""
    v = (~mask) & (lab >= 0) & (lab < N_CLASSES)
    if VOID is not None:
        v &= lab != VOID
    p, t = pred[v], lab[v]
    n = int(v.sum())
    cm = torch.bincount(t * N_CLASSES + p, minlength=N_CLASSES ** 2).view(N_CLASSES, N_CLASSES).double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    present = union > 0
    for c in EXCLUDE:
        present[c] = False
    miou = float((tp[present] / union[present]).mean()) if present.any() else float("nan")
    share = (cm.sum(1) / max(1, n)).tolist()
    return {"n": n, "acc": float(tp.sum() / max(1, n)), "miou": miou, "share": share, "valid": v}


@torch.no_grad()
def predict_per_image(model, batches, device):
    """[(pred [M], lab [M], mask [M])] for every image, in order."""
    model.eval()
    out = []
    for b in batches():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(b)["predictions"].float().argmax(-1)
        lab = b["queries"][:, :, 4].long()
        msk = b["queries_mask"].bool()
        for i in range(pred.shape[0]):
            out.append((pred[i].to(torch.uint8).cpu(), lab[i].cpu(), msk[i].cpu()))
    return out


def per_image_report(name, base, cond, csv_rows):
    """Distribution, across images, of the change caused by one condition."""
    d_acc, d_miou, changed = [], [], []
    for i, ((pb, lb, mb), (pc, _, _)) in enumerate(zip(base, cond)):
        sb = image_stats(pb.long(), lb, mb)
        sc = image_stats(pc.long(), lb, mb)
        v = sb["valid"]
        ch = float((pb[v] != pc[v]).float().mean()) if v.any() else 0.0
        d_acc.append(100 * (sc["acc"] - sb["acc"]))
        d_miou.append(100 * (sc["miou"] - sb["miou"]))
        changed.append(100 * ch)
        csv_rows.append([name, i, sb["n"], f"{100 * sb['acc']:.3f}", f"{100 * sc['acc']:.3f}",
                         f"{100 * sb['miou']:.3f}", f"{100 * sc['miou']:.3f}", f"{100 * ch:.3f}"]
                        + [f"{100 * x:.3f}" for x in sb["share"]])
    a = torch.tensor(d_acc)
    m = torch.tensor([x for x in d_miou if x == x])                  # drop NaN (no valid class)
    c = torch.tensor(changed)
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    loss = (-a).clamp(min=0)
    top = torch.sort(loss, descending=True).values
    k10 = max(1, len(top) // 10)
    share_top = float(top[:k10].sum() / top.sum().clamp_min(1e-9)) if float(top.sum()) > 0 else 0.0
    fmt = lambda t: " ".join(f"{float(x):+6.2f}" for x in torch.quantile(t, q))
    print(f"\n[per-image] {name}: {len(a)} images")
    print(f"    accuracy change (pts)   quantiles 5/25/50/75/95%: {fmt(a)}")
    if len(m):
        print(f"    mIoU change (pts)       quantiles 5/25/50/75/95%: {fmt(m)}")
    print(f"    pixels changed (%)      quantiles 5/25/50/75/95%: "
          + " ".join(f"{float(x):6.2f}" for x in torch.quantile(c, q)))
    print(f"    images with |accuracy change| < 0.5 pt: {100 * float((a.abs() < 0.5).float().mean()):5.1f}%"
          f" | losing > 2 pts: {100 * float((a < -2).float().mean()):5.1f}%"
          f" | losing > 10 pts: {100 * float((a < -10).float().mean()):5.1f}%")
    print(f"    the 10% most affected images account for {100 * share_top:.1f}% of the total accuracy loss")
    worst = torch.argsort(a)[:5].tolist()
    print(f"    most affected images (index: accuracy change): "
          + ", ".join(f"{i}: {float(a[i]):+.1f}" for i in worst))


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
    for b in batches():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(b)["predictions"].float().argmax(-1)
        lab = b["queries"][:, :, 4].long()
        cm += confusion(pred, lab, b["queries_mask"].bool())
    return cm


def main():
    ap = argparse.ArgumentParser(description="Zero-message ablation of a trained Atomiser_Unet on PASTIS")
    ap.add_argument("--dataset", default="pastis", choices=sorted(DATASETS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_s1", action="store_true", help="as in training (required for per-level temporal)")
    ap.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--num", type=int, default=0, help="images evaluated (0 = the whole split)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--skip", default="", help="comma-separated conditions to skip, e.g. only,const_gate")
    ap.add_argument("--per_image", action="store_true",
                    help="per-image statistics for the conditions in --per_image_conds")
    ap.add_argument("--per_image_conds", default="no_coarse")
    ap.add_argument("--csv", default=None, help="per-image CSV (default: ablation_per_image_<dataset>_<split>.csv)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from training.utils import read_yaml, Lookup_encoding
    config = read_yaml(args.config)
    global N_CLASSES, VOID, EXCLUDE
    pastis = args.dataset == "pastis"
    N_CLASSES = int(config["trainer"]["num_classes"])
    VOID, EXCLUDE = (19, (0,)) if pastis else (None, ())
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config)
    if args.dataset in PASTIS_FAMILY:
        prepare_lookup_pastis(lookup, args.use_s1)        # as the PASTIS-family training scripts
    model = load_model(config, lookup, args.checkpoint, device)
    fusions = install_hooks(model)
    S = len(fusions) + 1

    mod, cls, root = DATASETS[args.dataset]
    temporal = config.get("Atomiser", {}).get("unet_temporal", "none") == "per_level"
    if temporal and not pastis:
        raise SystemExit("per-level temporal models: only --dataset pastis is supported here")
    if temporal:
        if not args.use_s1:
            raise SystemExit("the per-level temporal model uses PastisHDTemporalDataset: add --use_s1")
        tm = importlib.import_module("training.utils.datasets.pastis_temporal_dataset")
        DS, collate = tm.PastisHDTemporalDataset, tm.pastis_temporal_collate_fn
    else:
        DS, collate = getattr(importlib.import_module(mod), cls), None
    if pastis:
        ds = DS(root_path=args.data_root or root, mode=args.split, config_model=config, look_up=lookup,
                use_s1=args.use_s1, use_spot=False)
    else:                                                  # generic constructor (as viz_unet_pca.py)
        ds = DS(root_path=args.data_root or root, mode=args.split,
                dataset_config=read_yaml(args.bands_yaml), config_model=config, look_up=lookup)
    if args.num:
        ds = torch.utils.data.Subset(ds, list(range(min(args.num, len(ds)))))
    print(f"[ablation] {args.dataset} {args.split}: {len(ds)} images | "
          f"{'per-level temporal' if temporal else 'single-date / joint'} model, {S} levels, "
          f"fusion={fusions[0].mode} | {N_CLASSES} classes, main mIoU "
          f"{'over classes 1-' + str(N_CLASSES - 1) if pastis else 'over all classes'}")

    def mv(v):
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        if isinstance(v, dict):
            return {k: mv(x) for k, x in v.items()}
        return v

    if collate is not None:
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                         collate_fn=collate)
        batches = lambda: (mv(b) for b in dl)
    else:
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                                         collate_fn=lambda x: x[0])
        batches = lambda: (to_batch(s, device) for s in dl)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    conds = [("baseline", {})]
    if "keep" not in skip and "only" not in skip:
        conds += [(f"keep_L0-{l}", {"zero_levels": (l,)}) for l in reversed(range(1, S - 1))]
    conds.append(("no_coarse", {"zero_levels": (0,)}))
    if "const_gate" not in skip and fusions[0].mode != "add":
        conds.append(("const_gate", {"gate": "const"}))
    inter = list(range(1, S - 1))                                      # intermediate levels
    if "drop_skip" not in skip and inter:
        conds += [(f"drop_skip_L{l}", {"drop_skip": (l,)}) for l in reversed(inter)]
        if len(inter) > 1:
            conds.append(("drop_skips", {"drop_skip": tuple(inter)}))
    if "no_obs" not in skip:
        conds.append(("no_obs", {"no_obs": True}))

    results, base = {}, None
    need_means = any(kw.get("gate") == "const" or kw.get("drop_skip") for _, kw in conds)
    means_done = False
    for name, kw in conds:
        if need_means and not means_done and (kw.get("gate") == "const" or kw.get("drop_skip")) \
                and fusions[0].mode != "add":                          # first pass: per-channel means
            means_done = True
            set_condition(model, fusions, gate="collect")
            evaluate(model, batches, device)
            for f in fusions:
                if f.mode != "add":
                    f._const = (f._gsum / max(1, f._gcnt)).to(device)
            print("[ablation] gate means per level (mean over channels): "
                  + " ".join(f"L{l}={float(f._const.mean()):.3f}" for l, f in enumerate(fusions) if f._const is not None))
        set_condition(model, fusions, **kw)
        r = metrics(evaluate(model, batches, device))
        results[name] = r
        if base is None:
            base = r
        d = r["mIoU"] - base["mIoU"]
        print(f"[ablation] {name:12s} mIoU {100 * r['mIoU']:6.2f} ({100 * d:+6.2f}) | "
              f"mIoU(all) {100 * r['mIoU_all']:6.2f} | acc {100 * r['acc']:6.2f} | per class "
              + " ".join(f"{100 * float(x):.1f}" for x in r["iou"]), flush=True)

    print("\n[ablation] summary (main mIoU, change vs baseline):")
    for name, r in results.items():
        print(f"    {name:12s} {100 * r['mIoU']:6.2f}   {100 * (r['mIoU'] - base['mIoU']):+6.2f}")

    if args.per_image:
        wanted = [c.strip() for c in args.per_image_conds.split(",") if c.strip()]
        kws = dict(conds)
        missing = [c for c in wanted if c not in kws]
        if missing:
            raise SystemExit(f"--per_image_conds: unknown conditions {missing}; available: {list(kws)}")
        set_condition(model, fusions)
        base_pred = predict_per_image(model, batches, device)
        rows = []
        for name in wanted:
            set_condition(model, fusions, **kws[name])
            per_image_report(name, base_pred, predict_per_image(model, batches, device), rows)
        set_condition(model, fusions)
        path = args.csv or f"ablation_per_image_{args.dataset}_{args.split}.csv"
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["condition", "image", "valid_pixels", "acc_base", "acc_cond", "miou_base",
                        "miou_cond", "pixels_changed"] + [f"share_class{c}" for c in range(N_CLASSES)])
            w.writerows(rows)
        print(f"\n[per-image] written {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
