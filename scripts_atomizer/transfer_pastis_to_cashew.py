"""
PASTIS -> Cashew transfer: linear probe and decoder-only fine-tuning
====================================================================

A PASTIS-trained Atomiser_Unet (joint or per-level temporal) is applied to Cashew
(single-date Sentinel-2, 7 classes) with its encoder FROZEN.

    --mode linear    everything frozen; per-pixel features from the nearest level-0 cell
                     (D0 = decoder output, or E0 = encoder level 0); a linear classifier
                     is trained on sampled training pixels and evaluated on all test pixels
    --mode decoder   encoder frozen (atom embedding, leaf reader, encoder levels, merges,
                     temporal transformers; kept in eval mode); decoder stages + query
                     decoder trained with a NEW 7-class head; best validation epoch tested

Cashew inputs are adapted to the PASTIS model (as viz_unet_pca.py does):
  * bands PASTIS never saw in training (e.g. B01, B09) are masked (untrained embeddings);
  * date-less atoms get day --doy (PASTIS always had dates: its "no date" embedding is
    untrained); time_positions = that day for temporal models;
  * per-level temporal models get each image as a time series of ONE date;
  * leaves keep PASTIS's ground size: k = m = --pixels_per_leaf x atoms per pixel kept.

Controls (essential to interpret the numbers):
  --random_init   same architecture, random weights: what is transfer, what is the probe
  --no_obs        disable the query decoder's direct read of each pixel's raw atoms
                  (that path alone can learn much of Cashew without the encoder)

Examples
    python scripts_atomizer/transfer_pastis_to_cashew.py --mode linear \\
        --checkpoint <pastis ckpt> --config ./training/configs/<pastis config>.yaml --use_s1
    python scripts_atomizer/transfer_pastis_to_cashew.py --mode decoder --no_obs \\
        --checkpoint <pastis ckpt> --config ./training/configs/<pastis config>.yaml --use_s1
    (add --random_init, without --checkpoint, for the control)

--use_s1 must match how the PASTIS model was trained (it fixes the lookup indices).
mIoU is reported two ways: over all 7 classes (absent = 0) and over classes present in
the test labels (the macro average of torchmetrics' JaccardIndex).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import argparse
import copy
import importlib
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from viz_unet_pca import (DATASETS, prepare_lookup_pastis, load_model, level_features,
                          adapt_batch)

NUM_CLASSES = 7


# =============================================================================
# Cashew data, adapted to the PASTIS model
# =============================================================================
def pastis_seen_spectral_ids(lookup, use_s1):
    P = getattr(importlib.import_module(DATASETS["pastis"][0]), DATASETS["pastis"][1])
    infos = dict(P.S2_BANDS_INFO, **(P.S1_BANDS_INFO if use_s1 else {}))
    return {int(lookup.table_wave[(i["bandwidth"], i["central_wavelength"])])
            for i in infos.values() if (i["bandwidth"], i["central_wavelength"]) in lookup.table_wave}


class CashewAdapter:
    """Turns collated Cashew batches into inputs of the PASTIS model."""

    def __init__(self, lookup, seen_ids, doy, temporal, pixels_per_leaf, device):
        self.seen, self.temporal, self.device = seen_ids, temporal, device
        self.doy = int(doy)
        self.time_idx = lookup.get_or_register_time_idx(self.doy)
        self.inv_time = {v: k for k, v in lookup.table_time.items()}
        self.pixels_per_leaf = int(pixels_per_leaf)
        self.drop_ids, self.atoms_per_pixel = None, None

    def inspect(self, batch):
        """From one batch: which spectral ids to mask, atoms per pixel kept, k."""
        tok = next(iter(batch["groups"].values()))["tokens"]
        ids = sorted({int(x) for x in tok[0, :, 3].long().unique().tolist()})
        self.drop_ids = [i for i in ids if i not in self.seen]
        self.atoms_per_pixel = len(ids) - len(self.drop_ids)
        k = self.pixels_per_leaf * self.atoms_per_pixel
        print(f"[transfer] Cashew spectral ids {ids}: masked (unseen in PASTIS training) "
              f"{self.drop_ids or 'none'}; {self.atoms_per_pixel} atoms per pixel kept -> "
              f"k = m = {k} ({self.pixels_per_leaf} pixels per leaf)")
        return k

    def __call__(self, batch):
        def mv(v):
            if isinstance(v, torch.Tensor):
                return v.to(self.device)
            if isinstance(v, dict):
                return {k2: mv(v2) for k2, v2 in v.items()}
            return v
        b = mv(batch)
        b = adapt_batch(b, self.drop_ids or (), self.time_idx)        # mask unseen bands, fill dates
        if self.temporal:                                             # a time series of one date
            res, g = next(iter(b["groups"].items()))
            tok = g["tokens"]
            b["groups"] = {res: dict(g, tokens=tok.unsqueeze(1), mask=g["mask"].bool().unsqueeze(1))}
            t = tok[:, :, 7].long()
            doy = [float(self.inv_time.get(int(t[i][t[i] >= 0][0]) if (t[i] >= 0).any() else -1, self.doy))
                   for i in range(t.shape[0])]
            b["time_positions"] = torch.tensor(doy, device=self.device).view(-1, 1)
        return b


def build_cashew(split, cashew_config, lookup, bands_yaml, root):
    mod, cls, default_root = DATASETS["cashew"]
    C = getattr(importlib.import_module(mod), cls)
    cfg = copy.deepcopy(cashew_config)
    aug = cfg.setdefault("trainer", {}).setdefault("band_dropout_augmentation", {})
    aug["enabled"] = False                                    # transfer: no band dropout
    from training.utils import read_yaml
    return C(root_path=root or default_root, mode=split, dataset_config=read_yaml(bands_yaml),
             config_model=cfg, look_up=lookup)


def loader(ds, batch, shuffle, workers):
    from training.utils.datasets.multitask_data import TaggedView, collate_multitask_batch
    return torch.utils.data.DataLoader(TaggedView(ds, "cashew"), batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, collate_fn=collate_multitask_batch,
                                       persistent_workers=workers > 0)


# =============================================================================
# Model surgery
# =============================================================================
def encoder_modules(model):
    mods = [model.embed, model.leaf, model.unet.enc]
    for name in ("merge", "region", "group_seeds", "group_proj", "group_final"):
        m = getattr(model.unet, name, None)
        if m is not None:
            mods.append(m)
    if getattr(model, "temporal", None) is not None:
        mods.append(model.temporal)
    return mods


def freeze_encoder(model):
    params = [p for m in encoder_modules(model) for p in m.parameters()]
    params += [model.empty_emb] + list(model.unet.seeds)
    for p in params:
        p.requires_grad_(False)
    return params


def set_train_mode(model):
    """Train mode for the decoder side; the frozen encoder stays in eval mode."""
    model.train()
    for m in encoder_modules(model):
        m.eval()


def replace_head(model, num_classes, seed=0):
    torch.manual_seed(seed)
    qdim = model.decoder.q_init.numel()
    model.decoder.head = nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, num_classes))


def disable_obs(model):
    """Zero (and freeze) the output of the query decoder's raw-observation read."""
    o = model.decoder.obs_o
    nn.init.zeros_(o.weight)
    nn.init.zeros_(o.bias)
    o.weight.requires_grad_(False)
    o.bias.requires_grad_(False)


def reset_decoder(model, seed=0):
    """Re-initialise the decoder side (decoder stages + query decoder), as at training
    start: every layer reset, fusion gates zero-initialised again. The atom embedding
    (shared with the encoder, frozen) is left untouched."""
    torch.manual_seed(seed)
    skip = {id(m) for m in model.embed.modules()}
    parts = [getattr(model.unet, n, None) for n in ("odec", "dec", "up", "fuse")] + [model.decoder]
    subs = [sub for part in parts if part is not None for sub in part.modules() if id(sub) not in skip]
    for sub in subs:                                           # 1) reset every layer
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()
    for sub in subs:                                           # 2) THEN zero the gates (a gate's own
        if hasattr(sub, "lin") and getattr(sub, "mode", None) in ("gate", "gate_sm"):   # Linear is reset in 1)
            nn.init.zeros_(sub.lin.weight)                     # gates open at 0.5, as in training
            nn.init.zeros_(sub.lin.bias)
    model.decoder.q_init.data.normal_(0, 0.02)


# =============================================================================
# Metrics
# =============================================================================
def confusion(pred, target, n=NUM_CLASSES, ignore=255):
    v = target != ignore
    return torch.bincount(target[v].long() * n + pred[v].long().clamp(0, n - 1),
                          minlength=n * n).view(n, n).cpu()


def report(cm, name):
    tp = cm.diag().double()
    union = cm.sum(0).double() + cm.sum(1).double() - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.zeros_like(tp))
    present = cm.sum(1) > 0
    out = {"mIoU_all": float(iou.mean()), "mIoU_present": float(iou[present].mean()),
           "acc": float(tp.sum() / cm.sum().clamp_min(1))}
    print(f"[transfer] {name}: mIoU (all 7 classes) {out['mIoU_all']:.4f} | mIoU (present) "
          f"{out['mIoU_present']:.4f} | acc {out['acc']:.4f} | per class "
          + " ".join(f"{float(x):.3f}" for x in iou))
    return out


# =============================================================================
# Linear probe
# =============================================================================
def pixel_features(model, b, km, feature):
    """[M, D] features of the batch's query pixels (first image) + labels [M]."""
    tokens, geom, enc, dec = level_features(model, b, km)
    f = (dec[0] if feature == "d0" else enc[0]).float()               # [L0, D]
    pq = model.embed.centers(b["queries"][0]).float()                 # [M, 2]
    nn_cell = torch.cdist(pq, geom.centers[0].float().to(pq.device)).argmin(1)
    lab = b["queries"][0, :, 4].long()
    lab = torch.where(b["queries_mask"][0].bool(), torch.full_like(lab, 255), lab)
    return f.to(pq.device)[nn_cell], lab


def run_linear(model, adapt, km, args, dl_train, dl_test):
    feats, labs = [], []
    g = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for i, batch in enumerate(dl_train):
            if args.train_num and i >= args.train_num:
                break
            b = adapt(batch)
            for j in range(b["queries"].shape[0]):
                bj = _one(b, j)
                f, l = pixel_features(model, bj, km, args.features)
                keep = (l != 255).nonzero().squeeze(1)
                keep = keep[torch.randperm(len(keep), generator=g)[:args.pix_per_image]]
                feats.append(f[keep].cpu()), labs.append(l[keep].cpu())
    X, Y = torch.cat(feats), torch.cat(labs)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    X = (X - mu) / sd
    print(f"[transfer] linear probe: {len(Y)} training pixels, {X.shape[1]}-d features ({args.features})")
    clf = nn.Linear(X.shape[1], NUM_CLASSES)
    counts = torch.bincount(Y, minlength=NUM_CLASSES).float()
    w = (counts.sum() / counts.clamp_min(1)) if args.class_weight == "inverse" else torch.ones(NUM_CLASSES)
    w = w / w.mean()
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for ep in range(args.epochs):
        perm = torch.randperm(len(Y), generator=g)
        for s in range(0, len(Y), args.batch_pixels):
            idx = perm[s:s + args.batch_pixels]
            loss = F.cross_entropy(clf(X[idx]), Y[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for batch in dl_test:
            b = adapt(batch)
            for j in range(b["queries"].shape[0]):
                f, l = pixel_features(model, _one(b, j), km, args.features)
                pred = clf(((f.cpu() - mu) / sd)).argmax(-1)
                cm += confusion(pred, l.cpu())
    return report(cm, f"TEST linear probe ({args.features})")


def _one(b, j):
    """Image j of a batch, as a batch of one."""
    out = {}
    for k, v in b.items():
        if k == "groups":
            out[k] = {r: {kk: (vv[j:j + 1] if isinstance(vv, torch.Tensor) else vv)
                          for kk, vv in g.items()} for r, g in v.items()}
        elif isinstance(v, torch.Tensor):
            out[k] = v[j:j + 1]
        else:
            out[k] = v
    return out


# =============================================================================
# Decoder-only fine-tuning
# =============================================================================
def evaluate(model, adapt, dl):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad(), torch.autocast(device_type=adapt.device.type, dtype=torch.bfloat16,
                                         enabled=adapt.device.type == "cuda"):
        for batch in dl:
            b = adapt(batch)
            pred = model(b)["predictions"].float().argmax(-1)
            lab = b["queries"][:, :, 4].long()
            lab = torch.where(b["queries_mask"].bool(), torch.full_like(lab, 255), lab)
            cm += confusion(pred, lab)
    return cm


def run_decoder(model, adapt, args, dl_train, dl_val, dl_test):
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params) / 1e6
    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[transfer] decoder-only fine-tuning: {n_tr:.2f}M trainable / {n_all:.1f}M params")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(dl_train)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        set_train_mode(model)
        tot, n = 0.0, 0
        for batch in dl_train:
            b = adapt(batch)
            with torch.autocast(device_type=adapt.device.type, dtype=torch.bfloat16,
                                enabled=adapt.device.type == "cuda"):
                logits = model(b, training=True)["predictions"]
            lab = b["queries"][:, :, 4].long()
            lab = torch.where(b["queries_mask"].bool(), torch.full_like(lab, 255), lab)
            loss = F.cross_entropy(logits.float().reshape(-1, NUM_CLASSES), lab.reshape(-1), ignore_index=255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot, n = tot + float(loss), n + 1
        r = report(evaluate(model, adapt, dl_val), f"epoch {ep} val (train loss {tot / max(1, n):.4f})")
        if r["mIoU_all"] > best:
            best = r["mIoU_all"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return report(evaluate(model, adapt, dl_test), f"TEST decoder-only (best val mIoU {best:.4f})")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="PASTIS -> Cashew: linear probe / decoder-only fine-tuning")
    ap.add_argument("--mode", required=True, choices=("linear", "decoder"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--random_init", action="store_true", help="control: random weights, no checkpoint")
    ap.add_argument("--config", required=True, help="the PASTIS model's config")
    ap.add_argument("--cashew_config", default="./training/configs/config_test-CASHEW-unet-legacy.yaml",
                    help="config used to build the Cashew dataset (trainer keys)")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--use_s1", action="store_true", help="as in the PASTIS model's training")
    ap.add_argument("--doy", type=int, default=180, help="day of year given to date-less atoms")
    ap.add_argument("--pixels_per_leaf", type=int, default=4, help="PASTIS model's leaf size in pixels")
    ap.add_argument("--features", default="d0", choices=("d0", "e0"), help="linear probe features")
    ap.add_argument("--pix_per_image", type=int, default=2048)
    ap.add_argument("--class_weight", default="inverse", choices=("none", "inverse"))
    ap.add_argument("--batch_pixels", type=int, default=8192)
    ap.add_argument("--train_num", type=int, default=0, help="linear: training images used (0 = all)")
    ap.add_argument("--no_obs", action="store_true", help="disable the raw-observation read")
    ap.add_argument("--decoder_init", default="pastis", choices=("pastis", "random"),
                    help="decoder mode: start from the PASTIS decoder or re-initialise it")
    ap.add_argument("--epochs", type=int, default=None, help="default: 30 (linear) / 50 (decoder)")
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 (linear) / 1e-4 (decoder)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    args = ap.parse_args()
    if not args.random_init and not args.checkpoint:
        raise SystemExit("--checkpoint is required (or --random_init for the control)")
    args.epochs = args.epochs or (30 if args.mode == "linear" else 50)
    args.lr = args.lr or (1e-3 if args.mode == "linear" else 1e-4)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from training.utils import read_yaml, Lookup_encoding
    config = read_yaml(args.config)
    cashew_config = read_yaml(args.cashew_config)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config)
    prepare_lookup_pastis(lookup, args.use_s1)                    # same registration as training
    model = load_model(config, lookup, args.checkpoint, device, args.random_init, args.seed)
    temporal = getattr(model, "temporal_mode", "none") == "per_level"

    ds = {s: build_cashew(s, cashew_config, lookup, args.bands_yaml, args.data_root)
          for s in ("train", "validation", "test")}
    print("[transfer] Cashew: " + ", ".join(f"{s}={len(d)}" for s, d in ds.items()))
    dl_train = loader(ds["train"], args.batch, True, args.workers)
    dl_val = loader(ds["validation"], args.batch, False, args.workers)
    dl_test = loader(ds["test"], args.batch, False, args.workers)

    adapt = CashewAdapter(lookup, pastis_seen_spectral_ids(lookup, args.use_s1), args.doy,
                          temporal, args.pixels_per_leaf, device)
    k = adapt.inspect(next(iter(dl_test)))
    model.pruning.k_choices = None                               # any k at inference
    model.km_train, model.km_eval = [(k, k)], (k, k)
    if args.no_obs:
        disable_obs(model)
        print("[transfer] raw-observation read DISABLED (--no_obs)")
    print(f"[transfer] model: {'per-level temporal (1 date)' if temporal else 'joint / single date'}, "
          f"{'RANDOM INIT' if args.random_init else args.checkpoint}")

    if args.mode == "linear":
        model.eval()
        run_linear(model, adapt, (k, k), args, dl_train, dl_test)
    else:
        freeze_encoder(model)
        if args.decoder_init == "random":
            reset_decoder(model, args.seed)
            if args.no_obs:
                disable_obs(model)
        replace_head(model, NUM_CLASSES, args.seed)
        model.to(device)
        run_decoder(model, adapt, args, dl_train, dl_val, dl_test)


if __name__ == "__main__":
    main()
