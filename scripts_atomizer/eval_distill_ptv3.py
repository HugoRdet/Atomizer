"""
eval_distill_ptv3.py -- downstream evaluation of a distilled Atomizer-PTv3 encoder (PANGAEA / OlmoEarth-style).

Where it goes: scripts_atomizer/eval_distill_ptv3.py (modules in training/eval_ptv3/)

Protocols
    linear    frozen encoder, decoder level-0 features cached once, 1x1 linear probe        (OlmoEarth linear probing)
    decoder   frozen encoder, small multi-dilation conv decoder (+ L-TAE-like temporal head)  (PANGAEA frozen + decoder)
    knn       frozen encoder, mean-pooled features, weighted kNN                             (EuroSAT)
    finetune  whole encoder trained (lr x enc_lr_mult) + decoder                              (upper bound)
Controls with the same command: --ckpt none (same architecture, random init) and a from-scratch run (finetune + none).

Datasets: burnscars, sen1floods11, pastis (multi-temporal S2 + S1), eurosat -- the repo's own dataset classes
(PANGAEA splits), with full token grids, no augmentation, no band dropout.

Features are cached ONCE per (checkpoint, dataset, bn_mode) as float16 memmaps; seeds and label fractions only change
the head and the subset of train tiles, so a full grid of seeds x fractions x probes costs one encoder pass.

Usage
    export PYTHONPATH=/path/to/PointTransformerV3/src:.:$PYTHONPATH
    python scripts_atomizer/eval_distill_ptv3.py --dataset sen1floods11 --protocol linear \
        --distill_config training/configs/config_distill_ptv3.yaml --bands_yaml data/bands_info/bands.yaml \
        --ckpt checkpoints/distill/<xp>/encoder_step18000.pth --seeds 0 1 2 --label_fracs 0.1 1.0
    # control
    python scripts_atomizer/eval_distill_ptv3.py ... --ckpt none
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.utils import read_yaml                                                    # noqa: E402
from training.eval_ptv3.data import SPECS, build_dataset, collate_eval, label_subset    # noqa: E402
from training.eval_ptv3.features import (DenseExtractor, build_lookup, build_net,       # noqa: E402
                                         check_lookup, to_device)
from training.eval_ptv3.metrics import SegMetrics, cls_accuracy                         # noqa: E402
from training.eval_ptv3.probes import (LightSegDecoder, LinearClsProbe, LinearSegProbe,  # noqa: E402
                                       TemporalAggregator, knn_classify)

SPLITS = ("train", "validation", "test")


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------------------------------------ kNN descriptors
def pool_all(f, names, chunk=256):
    """Cached features f [N, D, H, W] (or [N, D] already pooled) -> {name: [N, D'] float32 CPU}, in chunks."""
    out = {n: [] for n in names}
    for i in range(0, f.shape[0], chunk):
        x = torch.from_numpy(np.asarray(f[i:i + chunk], np.float32))
        if x.ndim == 2:                                                    # enc_last: one vector per image
            for n in names:
                out[n].append(x)
            continue
        x = x.flatten(2)                                                   # [n, D, HW]
        mean, mx = x.mean(2), x.amax(2)
        d = {"mean": mean, "max": mx, "meanmax": torch.cat([mean, mx], 1),
             "meanstd": torch.cat([mean, x.std(2)], 1)}
        for n in names:
            out[n].append(d[n])
    return {n: torch.cat(v) for n, v in out.items()}


# ------------------------------------------------------------------------------------------------ feature cache
@torch.no_grad()
def cache_split(ext, loader, path: Path, dev):
    """Writes path/feat.npy (memmap float16 [N, (T,) D, H, W]), label.pt, doy.pt, stats.json."""
    if (path / "done").exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    mm, labels, doys, i0 = None, [], [], 0
    torch.cuda.reset_peak_memory_stats(dev) if dev.type == "cuda" else None
    t0 = time.time()
    for batch in loader:
        batch = to_device(batch, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            f = ext(batch)
        f = f.float().cpu().numpy().astype(np.float16)
        if mm is None:
            mm = np.lib.format.open_memmap(path / "feat.npy", mode="w+", dtype=np.float16,
                                           shape=(len(loader.dataset),) + f.shape[1:])
        mm[i0:i0 + len(f)] = f
        i0 += len(f)
        labels.append(batch["label"].cpu())
        if "doy" in batch:
            doys.append(batch["doy"].cpu())
    mm.flush()
    torch.save(torch.cat(labels), path / "label.pt")
    if doys:
        torch.save(torch.cat(doys), path / "doy.pt")
    stats = {"n": i0, "seconds": time.time() - t0,
             "peak_mem_GB": torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None}
    json.dump(stats, open(path / "stats.json", "w"))
    (path / "done").touch()
    print(f"[cache] {path}: {stats}")


class Cached:
    def __init__(self, path: Path, idx=None):
        self.f = np.load(path / "feat.npy", mmap_mode="r")
        self.y = torch.load(path / "label.pt")
        self.doy = torch.load(path / "doy.pt") if (path / "doy.pt").exists() else None
        self.idx = torch.arange(len(self.y)) if idx is None else idx

    def __len__(self):
        return len(self.idx)

    def get(self, rows, dev):
        r = self.idx[rows].numpy()
        f = torch.from_numpy(np.ascontiguousarray(self.f[r])).to(dev).float()
        doy = None if self.doy is None else self.doy[r].to(dev)
        return f, self.y[r].to(dev), doy


# ------------------------------------------------------------------------------------------------ heads
def make_head(task, protocol, D, K):
    if task == "cls":
        head = LinearClsProbe(D, K)
    elif protocol == "linear":
        head = LinearSegProbe(D, K)
    else:
        head = LightSegDecoder(D, K)
    agg = TemporalAggregator(D, "mean" if protocol == "linear" else "attn") if task == "seg_temporal" else None
    return head, agg


def fit(out, y, task):
    """Bilinear upsampling of coarse (patch-grid) logits to the label size."""
    if task != "cls" and out.shape[-2:] != y.shape[-2:]:
        out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return out


def loss_fn(out, y, task, ignore):
    out = fit(out, y, task)
    return F.cross_entropy(out, y) if task == "cls" else F.cross_entropy(out, y, ignore_index=ignore)


@torch.no_grad()
def eval_cached(head, agg, data: Cached, task, K, ignore, dev, bs):
    head.eval(); agg is not None and agg.eval()
    m, preds = SegMetrics(K, ignore if ignore is not None else -1), []
    for rows in torch.arange(len(data)).split(bs):
        f, y, doy = data.get(rows, dev)
        if agg is not None:
            f = agg(f, doy)
        out = fit(head(f), y, task)
        preds.append(out.argmax(1).cpu()) if task == "cls" else m.update(out, y)
    if task == "cls":
        acc = cls_accuracy(torch.cat(preds), data.y[data.idx]); return {"main": acc, "top1": acc}
    r = m.compute(); r["main"] = r["mIoU"]; return r


def train_cached(head, agg, tr: Cached, va: Cached, a, task, K, ignore, dev):
    params = list(head.parameters()) + ([] if agg is None else list(agg.parameters()))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * max(1, (len(tr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    best, state = -1.0, None
    for ep in range(a.epochs):
        head.train(); agg is not None and agg.train()
        for rows in torch.randperm(len(tr)).split(a.bs):
            f, y, doy = tr.get(rows, dev)
            if agg is not None:
                f = agg(f, doy)
            loss = loss_fn(head(f), y, task, ignore)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.epochs:
            s = eval_cached(head, agg, va, task, K, ignore, dev, a.bs)["main"]
            if s > best:
                best = s
                state = ({k: v.clone() for k, v in head.state_dict().items()},
                         None if agg is None else {k: v.clone() for k, v in agg.state_dict().items()})
    head.load_state_dict(state[0]); agg is not None and agg.load_state_dict(state[1])
    return best


# ------------------------------------------------------------------------------------------------ fine-tuning
def run_finetune(ext, head, agg, loaders, a, task, K, ignore, dev):
    enc_params = [p for p in ext.parameters() if p.requires_grad]
    other = list(head.parameters()) + ([] if agg is None else list(agg.parameters()))
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr * a.enc_lr_mult}, {"params": other, "lr": a.lr}],
                            weight_decay=a.wd)
    steps = a.epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr * a.enc_lr_mult, a.lr], total_steps=steps, pct_start=0.05)

    def evaluate(loader):
        ext.set_mode(False); head.eval(); agg is not None and agg.eval()
        m, preds, ys = SegMetrics(K, ignore if ignore is not None else -1), [], []
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for batch in loader:
                batch = to_device(batch, dev)
                f = ext(batch)
                if agg is not None:
                    f = agg(f.float(), batch.get("doy"))
                out = fit(head(f.float()), batch["label"], task)
                if task == "cls":
                    preds.append(out.argmax(1).cpu()); ys.append(batch["label"].cpu())
                else:
                    m.update(out, batch["label"])
        if task == "cls":
            acc = cls_accuracy(torch.cat(preds), torch.cat(ys)); return {"main": acc, "top1": acc}
        r = m.compute(); r["main"] = r["mIoU"]; return r

    best, state = -1.0, None
    for ep in range(a.epochs):
        ext.set_mode(True); head.train(); agg is not None and agg.train()
        for batch in loaders["train"]:
            batch = to_device(batch, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                f = ext(batch, training=True)
                if agg is not None:
                    f = agg(f.float(), batch.get("doy"))
                loss = loss_fn(head(f.float()), batch["label"], task, ignore)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.epochs:
            s = evaluate(loaders["validation"])["main"]
            print(f"[finetune] epoch {ep + 1}: val {s:.2f}")
            if s > best:
                best = s
                state = [{k: v.detach().clone() for k, v in mod.state_dict().items()} for mod in (ext, head)] + \
                        [None if agg is None else {k: v.detach().clone() for k, v in agg.state_dict().items()}]
    ext.load_state_dict(state[0]); head.load_state_dict(state[1]); agg is not None and agg.load_state_dict(state[2])
    return {"val": evaluate(loaders["validation"]), "test": evaluate(loaders["test"])}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SPECS), required=True)
    ap.add_argument("--protocol", choices=["linear", "decoder", "knn", "finetune"], default="linear")
    ap.add_argument("--distill_config", required=True, help="config of the distillation run (same PTv3/Atomiser keys)")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    ap.add_argument("--ref_table_wave", default=None, help="optional torch file with the distillation lookup.table_wave")
    ap.add_argument("--ds_config", default=None, help="the dataset's own training config (optional, for its keys)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", default="none", help="encoder_*.pth / *.ckpt, or 'none' for the random-init control")
    ap.add_argument("--model", choices=["atomizer", "olmoearth"], default="atomizer",
                    help="olmoearth: frozen OlmoEarth (teacher) features, same splits / heads / metrics")
    ap.add_argument("--olmo_ckpt", default=None, help="default: distill.olmo_ckpt of --distill_config")
    ap.add_argument("--olmo_patch", type=int, default=None, help="default: distill.olmo_patch (2)")
    ap.add_argument("--olmo_ts", type=int, nargs=3, default=[15, 6, 2018],
                    help="timestamp triple for datasets without dates (same format as the OlmoEarth h5 'timestamps')")
    ap.add_argument("--bn_mode", choices=["batch", "eval", "recal"], default="batch")
    ap.add_argument("--multi_temporal", type=int, default=6, help="PASTIS frames per sample")
    ap.add_argument("--feature", choices=["dec0", "enc_last"], default="dec0",
                    help="dec0: decoder level-0 per pixel; enc_last: coarsest encoder level, pooled per image (cls only)")
    ap.add_argument("--keep_time_idx", action="store_true", help="feed the datasets' time_idx (distillation used -1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--label_fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc_lr_mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=16, help="head batch size (cached features)")
    ap.add_argument("--enc_bs", type=int, default=4, help="encoder batch size (512x512 tiles: 1-2)")
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--knn_k", type=int, nargs="+", default=[20], help="kNN neighbours (several values: one result each)")
    ap.add_argument("--knn_T", type=float, default=0.07, help="kNN vote temperature: weight exp(cos / T) (DINO)")
    ap.add_argument("--knn_pool", nargs="+", default=["mean"], choices=["mean", "max", "meanmax", "meanstd"],
                    help="image descriptor from the per-pixel features: mean (protocol), max, concat(mean, max), "
                         "concat(mean, std)")
    ap.add_argument("--knn_norm", nargs="+", default=["none"], choices=["none", "center", "standardize"],
                    help="before the cosine: none (protocol), subtract the train mean, or per-dimension z-score "
                         "with train statistics")
    ap.add_argument("--out", default="./eval_results/distill_ptv3")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SPECS[a.dataset]
    task, K, ignore = spec["task"], spec["num_classes"], spec["ignore"]
    dcfg = read_yaml(a.distill_config)
    ds_cfg = read_yaml(a.ds_config) if a.ds_config else None
    ckpt = None if a.ckpt == "none" else a.ckpt
    ck_tag = "random" if ckpt is None else f"{Path(ckpt).parent.name}_{Path(ckpt).stem}"
    if a.feature != "dec0":
        if task != "cls" or a.model != "atomizer" or a.protocol == "finetune":
            raise ValueError("--feature enc_last: classification, frozen Atomizer-PTv3 only")
        ck_tag += f"_{a.feature}"
    if a.model == "olmoearth":
        dd = dcfg.get("distill", {})
        a.olmo_ckpt = a.olmo_ckpt or dd.get("olmo_ckpt")
        a.olmo_patch = int(a.olmo_patch or dd.get("olmo_patch", 2))
        if a.olmo_ckpt is None:
            raise ValueError("--olmo_ckpt needed (not found in distill config)")
        if a.protocol == "finetune":
            raise NotImplementedError("OlmoEarth is evaluated frozen only")
        ck_tag = f"olmoearth_p{a.olmo_patch}"
        a.bn_mode = "eval"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    lookup = build_lookup(dcfg, a.bands_yaml)
    check_lookup(lookup, torch.load(a.ref_table_wave) if a.ref_table_wave else None)
    bands = read_yaml(a.bands_yaml)
    dsets = {s: build_dataset(a.dataset, s, bands, lookup, a.root, ds_cfg, a.multi_temporal) for s in SPLITS}

    def loader(s, shuffle=False, idx=None, bs=a.enc_bs):
        ds = dsets[s] if idx is None else torch.utils.data.Subset(dsets[s], idx.tolist())
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=a.workers, collate_fn=collate_eval,
                          pin_memory=dev.type == "cuda", drop_last=shuffle)

    results = []
    if a.protocol != "finetune":
        seed_all(0)                                                     # random-init control: fixed init
        if a.model == "olmoearth":
            from training.eval_ptv3.features_olmo import OlmoEarthExtractor
            net = None
            ext = OlmoEarthExtractor(a.olmo_ckpt, a.dataset, dsets["train"], a.olmo_patch, a.olmo_ts).to(dev)
        else:
            net = build_net(dcfg, lookup, K, ckpt)
            ext = DenseExtractor(net, a.bn_mode, None if a.keep_time_idx else -1.0, a.feature).to(dev)
        for p in ext.parameters():
            p.requires_grad_(False)
        ext.set_mode(False)
        if a.bn_mode == "recal":
            ext.recalibrate_bn(loader("train", shuffle=True), dev)
        cdir = out / "cache" / f"{a.dataset}_{ck_tag}_bn{a.bn_mode}_T{a.multi_temporal}"
        for s in SPLITS:
            cache_split(ext, loader(s), cdir / s, dev)
        del ext, net; torch.cuda.empty_cache()
        enc_stats = json.load(open(cdir / "train" / "stats.json"))

        pooled = {}
        if a.protocol == "knn":
            for sp in ("train", "test"):
                full = Cached(cdir / sp)
                pooled[sp] = pool_all(full.f, a.knn_pool)

        for frac in a.label_fracs:
            for seed in a.seeds:
                seed_all(seed)
                tr = Cached(cdir / "train", label_subset(len(dsets["train"]), frac, seed))
                va, te = Cached(cdir / "validation"), Cached(cdir / "test")
                if a.protocol == "knn":
                    if task != "cls":
                        raise ValueError("knn is for classification")
                    ftr_all, fte = pooled["train"], pooled["test"]
                    ytr = tr.y[tr.idx].to(dev)
                    for pname in a.knn_pool:
                        xtr, xte = ftr_all[pname][tr.idx].to(dev), fte[pname][te.idx].to(dev)
                        for norm in a.knn_norm:
                            if norm == "none":
                                ntr, nte = xtr, xte
                            else:
                                mu = xtr.mean(0, keepdim=True)
                                sd = xtr.std(0, keepdim=True).clamp_min(1e-6) if norm == "standardize" else 1.0
                                ntr, nte = (xtr - mu) / sd, (xte - mu) / sd
                            for k in a.knn_k:
                                pred = knn_classify(ntr, ytr, nte, k=min(k, len(ntr)), T=a.knn_T, n_cls=K)
                                var = f"pool={pname} norm={norm} k={k}"
                                rv = {"test": {"top1": cls_accuracy(pred.cpu(), te.y[te.idx]), "main": None},
                                      "variant": var, "seed": seed, "label_frac": frac, "n_train": len(tr)}
                                rv["test"]["main"] = rv["test"]["top1"]
                                results.append(rv)
                                print(f"[{a.dataset}|knn|{ck_tag}|frac {frac}|seed {seed}] {var:38s} "
                                      f"test top-1 {rv['test']['top1']:.2f}")
                    continue
                else:
                    D = tr.f.shape[2] if task == "seg_temporal" else tr.f.shape[1]
                    head, agg = make_head(task, a.protocol, D, K)
                    head = head.to(dev); agg = None if agg is None else agg.to(dev)
                    train_cached(head, agg, tr, va, a, task, K, ignore, dev)
                    r = {"val": eval_cached(head, agg, va, task, K, ignore, dev, a.bs),
                         "test": eval_cached(head, agg, te, task, K, ignore, dev, a.bs)}
                r.update(seed=seed, label_frac=frac, n_train=len(tr))
                results.append(r)
                print(f"[{a.dataset}|{a.protocol}|{ck_tag}|frac {frac}|seed {seed}] "
                      f"val {r.get('val', {}).get('main', float('nan')):.2f}  test {r['test'].get('main', r['test'].get('top1')):.2f}")
    else:
        enc_stats = None
        for frac in a.label_fracs:
            for seed in a.seeds:
                seed_all(seed)
                net = build_net(dcfg, lookup, K, ckpt)
                ext = DenseExtractor(net, "eval" if a.bn_mode != "batch" else "batch",
                                     None if a.keep_time_idx else -1.0).to(dev)
                head, agg = make_head(task, "decoder", ext.dim, K)
                head = head.to(dev); agg = None if agg is None else agg.to(dev)
                idx = label_subset(len(dsets["train"]), frac, seed)
                loaders = {"train": loader("train", True, idx), "validation": loader("validation"), "test": loader("test")}
                r = run_finetune(ext, head, agg, loaders, a, task, K, ignore, dev)
                r.update(seed=seed, label_frac=frac, n_train=len(idx))
                results.append(r)
                print(f"[{a.dataset}|finetune|{ck_tag}|frac {frac}|seed {seed}] val {r['val']['main']:.2f} test {r['test']['main']:.2f}")
                del ext, net, head, agg; torch.cuda.empty_cache()

    # summary: mean +- std over seeds per label fraction
    summary = {}
    variants = sorted({r.get("variant") for r in results}, key=lambda v: (v is None, str(v)))
    for frac in a.label_fracs:
        for var in variants:
            rs = [r for r in results if r["label_frac"] == frac and r.get("variant") == var]
            for split in ("val", "test"):
                v = [r[split].get("main", r[split].get("top1")) for r in rs if split in r]
                if v:
                    key = f"{split}@{frac}" + (f" | {var}" if var else "")
                    summary[key] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
    tag = f"{a.dataset}_{a.protocol}_{ck_tag}_bn{a.bn_mode}" + (f"_{a.feature}" if a.protocol == "knn" else "")
    json.dump({"args": vars(a), "encoder_pass": enc_stats, "summary": summary, "results": results},
              open(out / f"{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
