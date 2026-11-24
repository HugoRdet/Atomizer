"""
viz_distill_pca.py -- standalone PCA figures and mimicry scores for a distillation checkpoint.

Where it goes: scripts_atomizer/viz_distill_pca.py

Loads a Lightning checkpoint of Model_Distill (Atomiser_Unet or AtomizerPTv3 student), runs the student ONCE per batch and
EVERY teacher of each dataset (distill.olmo_teachers, e.g. olmo + smarties; distill.flair_teachers, e.g. dino + pano),
each on its own grid (AnyUp-gridded teachers use teacher_patch: pano / smarties), on the same validation tiles. One
figure per (dataset, teacher), one score line each.

    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH
    python scripts_atomizer/viz_distill_pca.py --config training/configs/config_distill_ptv3_base_elev.yaml \
        --bands_yaml ./data/bands_info/bands_elev.yaml \
        --ckpt ./checkpoints/distill/ptv3_base_elev_s42/last.ckpt --n 6 --tiles 32 --batch 8 --out ./viz_pca/base_elev
Options: --view CROP UP (FLAIR view, default flair_crop 1), --teachers dino pano (subset),
         --keep_flair 3 4 / --keep_olmo ... (the student sees only these channels, the teachers everything: e.g. the
         cross-modal NIR + elevation case), --datasets olmo flair.
--bands_yaml must be the file used for training (bands_elev.yaml with flair_elevation), or the band indices differ.

Figure columns, one row per tile:
    input | teacher PCA | student PCA (teacher scale) | student PCA (own scale) | 1 - cos
          [| student pixels PCA (shared) | student pixels PCA (per image)]
  * the PCA basis is fitted on the (standardized) teacher tokens of the figure and applied to the student's predictions
    for that teacher (same space): same axes, so the hues are comparable;
  * "teacher scale" stretches the student with the teacher's 1-99 % range, "own scale" with its own range;
  * 1 - cos per teacher token (0 = identical direction, 1 = unrelated);
  * PTv3 student: per-pixel level-0 features (before pooling), rank-equalized colours; (shared) one basis over the
    shown tiles; (per image) one basis per tile (shows position-tied patterns such as edge lines), with the fraction
    of that tile's feature variance carried by its 3 components.

Scores (printed, and APPENDED to <out>/scores.txt with the checkpoint and step), per (dataset, teacher), over --tiles:
    cos     student vs teacher token          global  cosine of the per-image mean tokens
    local   cosine of the deviations from the image mean (within-image structure)
    oracle  the teacher's own image mean vs its tokens (a reference, not a baseline to beat)
    spread  std of the student / teacher along the teacher's top-3 PCA axes (1 = full amplitude)
Scores are comparable across checkpoints of one run, not across teacher sets (each teacher has its own scale).

BatchNorm (norm "bn" only): --bn batch (default) normalizes each batch with its own statistics; no effect with LN.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.distill.distill_heads import token_centers_from_tokens
from training.distill.trainer_distill import DistillDataModule, Model_Distill
from training.utils import Lookup_encoding, read_yaml
from training.utils.datasets.utils_dataset_distill import collate_distill


def to_device(x, dev):
    if isinstance(x, torch.Tensor):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    return x


def set_bn(model, mode):
    """mode 'batch': BatchNorm uses per-batch statistics without updating the running ones."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            n += 1
            if mode == "batch":
                m.train()
                m.momentum = 0.0
    return n


def pca3(X, n_fit=100000):
    """X [P, D] -> mean, top-3 principal axes [D, 3] (fitted on a subsample)."""
    sub = X[torch.randperm(X.shape[0])[:n_fit]]
    mu = sub.mean(0)
    _, V = torch.linalg.eigh((sub - mu).T @ (sub - mu))
    return mu, V[:, -3:].flip(-1)


def pca3_var(X):
    """X [P, D] -> mean, top-3 axes [D, 3], fraction of the total variance they carry."""
    mu = X.mean(0)
    ev, V = torch.linalg.eigh((X - mu).T @ (X - mu))
    return mu, V[:, -3:].flip(-1), float(ev[-3:].sum() / ev.clamp_min(0).sum().clamp_min(1e-12))


def stretch(z, ref):
    lo = torch.quantile(ref.reshape(-1, 3), 0.01, dim=0)
    hi = torch.quantile(ref.reshape(-1, 3), 0.99, dim=0)
    return ((z - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def equalize(z):
    """Rank-based colours: each of the 3 channels spread over [0, 1]."""
    r = torch.empty_like(z)
    r.scatter_(0, z.argsort(dim=0), torch.arange(z.shape[0], dtype=z.dtype).unsqueeze(1).expand(-1, 3))
    return r / max(z.shape[0] - 1, 1)


def input_rgb(name, teacher_in, n):
    if name == "flair":
        img = teacher_in["rgb"][:n].float().div(255).permute(0, 2, 3, 1)
    else:
        img = teacher_in["s2n"][:n, :, :, 0, :][..., [2, 1, 0]].float()        # B04, B03, B02
    out = []
    for im in img:
        lo, hi = torch.quantile(im.reshape(-1), 0.02), torch.quantile(im.reshape(-1), 0.98)
        out.append(((im - lo) / (hi - lo + 1e-6)).clamp(0, 1))
    return torch.stack(out).cpu()


def mask_channels(batch, keep):
    """Student-side band subset (as the band sets of training): atoms of the channels NOT in `keep` are masked."""
    if keep is None:
        return batch
    grp = batch["groups"][next(iter(batch["groups"]))]
    C, H, W = (int(x) for x in grp["shape"])
    B = grp["tokens"].shape[0]
    drop = [c for c in range(C) if c not in keep]
    if not drop:
        return batch
    m = grp["mask"].clone().bool().view(B, C, H * W)
    m[:, drop] = True
    grp["mask"] = m.view(B, -1)
    grp["tokens"] = grp["tokens"].clone()
    grp["tokens"][..., 0] = grp["tokens"][..., 0].masked_fill(grp["mask"], 0.0)
    return batch


def teachers_of(model, name):
    if name == "olmo":
        return list(getattr(model, "olmo_teachers", [model.layout["olmo"]["teacher"]]))
    return list(getattr(model, "flair_teachers", [model.layout["flair"]["teacher"]]))


@torch.no_grad()
def run_chunk(model, name, ds, idx, dev, view, keep, teachers, tnorm="batch"):
    """One forward pass of the student on the tiles `idx`, then every teacher of the dataset on its own grid."""
    L = model.layout[name]
    tp = getattr(model, "teacher_patch", {})                   # AnyUp-gridded teachers: tile px per token
    if name == "flair":
        crop, up = view
        items = [ds[(i, crop, up)] for i in idx]
        hw, patch_default = crop, 16 // up
    else:
        items = [ds[i] for i in idx]
        hw, patch_default, up = L["hw"], L["patch"], 1
    batch = to_device(mask_channels(collate_distill(items), keep), dev)
    tokens = batch["groups"][next(iter(batch["groups"]))]["tokens"]
    tin = dict(batch["teacher"], up=float(up)) if name == "flair" else batch["teacher"]
    amp = torch.autocast(dev.type, dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    out = {}
    with amp:
        batch["patch"] = patch_default
        m = int(float(model.dc.get("m_factor", 1.3)) * L["k"])
        d0, geom = model.student.encode(batch, L["k"], m)                       # once per batch
        for t in teachers:
            pt = tp.get(t, patch_default)
            g = hw // pt
            tgt = model._teacher(t)(dict(tin, out_side=g) if t in tp else tin)  # [B, g, g, D]
            assert tgt.shape[1] == g and tgt.shape[2] == g, f"{t}: teacher grid {tuple(tgt.shape[1:3])} != {g}"
            tgt = tgt.flatten(1, 2).float()
            if tnorm == "batch":                 # as in training: this batch's own per-channel statistics
                tgt = (tgt - tgt.mean(dim=(0, 1))) / tgt.std(dim=(0, 1)).clamp(min=model.tnorm[t].eps)
            else:                                # running (EMA) statistics stored in the checkpoint
                tgt = model.tnorm[t](tgt, 0.0)
            q = token_centers_from_tokens(tokens, hw, hw, pt)
            pred = model.student.predict(t, d0, geom, q)
            out[t] = dict(tgt=tgt.float().cpu(), pred=pred.float().cpu(), g=g)
    dense = getattr(model.student, "last_dense", None)
    dense = None if dense is None else dense[:len(idx)].float().cpu()
    return out, input_rgb(name, batch["teacher"], len(idx)), dense


def run_dataset(model, name, ds, n_tiles, batch, seed, dev, keep_n, view, keep, teachers, tnorm="batch"):
    """n_tiles validation tiles in passes of `batch`. Targets / predictions of all tiles kept (CPU) for the scores;
    images and per-pixel features only for the first `keep_n` tiles (the figures)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_tiles, len(ds)), replace=False).tolist()
    acc = {t: dict(tgt=[], pred=[], g=None) for t in teachers}
    imgs, denses = [], []
    for s0 in range(0, len(idx), batch):
        out, img, dense = run_chunk(model, name, ds, idx[s0:s0 + batch], dev, view, keep, teachers, tnorm)
        for t, r in out.items():
            acc[t]["tgt"].append(r["tgt"]); acc[t]["pred"].append(r["pred"]); acc[t]["g"] = r["g"]
        need = keep_n - sum(x.shape[0] for x in imgs)
        if need > 0:
            imgs.append(img[:need])
            if dense is not None:
                denses.append(dense[:need])
        print(f"[viz] {name}: {min(s0 + batch, len(idx))}/{len(idx)} tiles")
    res = {t: dict(idx=idx, tgt=torch.cat(a["tgt"]), pred=torch.cat(a["pred"]), g=a["g"]) for t, a in acc.items()}
    return res, torch.cat(imgs), (torch.cat(denses) if denses else None)


def scores(tgt, pred):
    """cos: student vs teacher token | global: per-image mean tokens | local: deviations from the image mean (the
    within-image structure) | oracle: the teacher's own image mean (a reference, not a baseline to beat) | spread:
    std of the student / teacher along the teacher's top-3 PCA axes."""
    cos_s = F.cosine_similarity(pred, tgt, dim=-1)                                  # [B, M]
    mt, mp = tgt.mean(dim=1, keepdim=True), pred.mean(dim=1, keepdim=True)
    cos_g = F.cosine_similarity(mp[:, 0], mt[:, 0], dim=-1)                         # [B]
    cos_l = F.cosine_similarity(pred - mp, tgt - mt, dim=-1)                        # [B, M]
    cos_o = F.cosine_similarity(mt.expand_as(tgt), tgt, dim=-1)
    D = tgt.shape[-1]
    mu, V = pca3(tgt.reshape(-1, D))
    zt, zp = (tgt.reshape(-1, D) - mu) @ V, (pred.reshape(-1, D) - mu) @ V
    spread = float(zp.std(0).mean() / (zt.std(0).mean() + 1e-12))
    return dict(cos=float(cos_s.mean()), glob=float(cos_g.mean()), local=float(cos_l.mean()), oracle=float(cos_o.mean()),
                spread=spread, cos_per_image=cos_s.mean(1).tolist(), local_per_image=cos_l.mean(1).tolist())


def render(label, r, sc, img, dense, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt, pred, g = r["tgt"], r["pred"], r["g"]
    n, M, D = tgt.shape
    mu, V = pca3(tgt.reshape(-1, D))
    zt, zp = (tgt - mu) @ V, (pred - mu) @ V
    ct = stretch(zt, zt).reshape(n, g, g, 3)
    cp_t = stretch(zp, zt).reshape(n, g, g, 3)                                     # teacher's scale
    cp_o = stretch(zp, zp).reshape(n, g, g, 3)                                     # student's own scale
    err = (1 - F.cosine_similarity(pred, tgt, dim=-1)).reshape(n, g, g)
    cd = None
    if dense is not None:
        Xd = dense.reshape(-1, dense.shape[-1])
        mud, Vd = pca3(Xd)
        cd = equalize((Xd - mud) @ Vd).reshape(*dense.shape[:3], 3)
        cdi, fvi = [], []                                                          # per-image PCA
        for x in dense:
            Xi = x.reshape(-1, x.shape[-1])
            mui, Vi, fv = pca3_var(Xi)
            cdi.append(equalize((Xi - mui) @ Vi).reshape(*x.shape[:2], 3))
            fvi.append(fv)

    cols = ["input", f"teacher PCA\n{g}x{g}", "student PCA\n(teacher scale)", "student PCA\n(own scale)", "1 - cos"]
    if cd is not None:
        cols += ["student pixels PCA\n(shared)", "student pixels PCA\n(per image)"]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.3 * n + 0.6), squeeze=False)
    for i in range(n):
        ax[i, 0].imshow(img[i].numpy())
        ax[i, 1].imshow(ct[i].numpy(), interpolation="nearest")
        ax[i, 2].imshow(cp_t[i].numpy(), interpolation="nearest")
        ax[i, 3].imshow(cp_o[i].numpy(), interpolation="nearest")
        im = ax[i, 4].imshow(err[i].numpy(), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        if cd is not None:
            ax[i, 5].imshow(cd[i].numpy(), interpolation="nearest")
            ax[i, 6].imshow(cdi[i].numpy(), interpolation="nearest")
            ax[i, 6].set_xlabel(f"top-3 var {fvi[i]:.0%}", fontsize=7)
        ax[i, 0].set_ylabel(f"#{r['idx'][i]}\ncos {sc['cos_per_image'][i]:.2f} | local {sc['local_per_image'][i]:.2f}",
                            fontsize=7)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=9)
    fig.colorbar(im, ax=ax[:, 4], fraction=0.05, pad=0.02, label="1 - cos")
    fig.suptitle(f"{title} | {label}: cos {sc['cos']:.3f} | global {sc['glob']:.3f} | local {sc['local']:.3f} | "
                 f"oracle {sc['oracle']:.3f} | spread x{sc['spread']:.2f}", fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="Lightning .ckpt of the distillation (not encoder_*.pth)")
    ap.add_argument("--n", type=int, default=6, help="tiles shown per dataset")
    ap.add_argument("--batch", type=int, default=None,
                    help="tiles per forward pass (default: distill.batch_per_dataset, the training batch)")
    ap.add_argument("--tiles", type=int, default=None,
                    help="tiles per dataset for the scores, run in forward passes of --batch (default: --batch)")
    ap.add_argument("--seed", type=int, default=0, help="which validation tiles")
    ap.add_argument("--datasets", nargs="+", default=["olmo", "flair"], choices=["olmo", "flair"])
    ap.add_argument("--teachers", nargs="+", default=None, help="subset of the teachers (default: all of each dataset)")
    ap.add_argument("--view", type=int, nargs=2, default=None, metavar=("CROP", "UP"),
                    help="FLAIR view (default: flair_crop 1); must be one of distill.flair_views")
    ap.add_argument("--keep_olmo", type=int, nargs="+", default=None,
                    help="student sees only these Sentinel channels (OLMO_S2 order + VV 12, VH 13); teachers see all")
    ap.add_argument("--keep_flair", type=int, nargs="+", default=None,
                    help="student sees only these FLAIR channels (0 R, 1 G, 2 B, 3 NIR, 4 ELEV), e.g. 3 4")
    ap.add_argument("--tnorm", default="batch", choices=["batch", "ema"],
                    help="teacher target standardization: batch = each batch's own statistics, as in training "
                         "(default; scores do not depend on which FLAIR view preceded the checkpoint); ema = the "
                         "running statistics stored in the checkpoint (what validation uses)")
    ap.add_argument("--out", default="./viz_pca")
    ap.add_argument("--bn", default="batch", choices=["batch", "eval"],
                    help="BatchNorm statistics: per batch (as in training) or running (eval mode); no effect with LN")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml",
                    help="the SAME bands file as training (bands_elev.yaml for configs with flair_elevation)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    config = read_yaml(args.config)
    config["Atomiser"]["use_error_predictor"] = False
    config["trainer"]["slide"] = False
    bands = read_yaml(args.bands_yaml)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), bands, config)

    model = Model_Distill(config, lookup, name="viz")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        raise ValueError(f"{args.ckpt} is not a Lightning checkpoint (use last.ckpt / step*.ckpt / timelimit.ckpt)")
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    missing = [k for k in missing if not k.startswith("_teachers")]
    print(f"[viz] loaded {args.ckpt} (step {ck.get('global_step', '?')}): "
          f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        raise RuntimeError(f"checkpoint does not match the model built from --config, e.g. {missing[:5]}")
    model.to(dev).eval()
    n_bn = set_bn(model.student, args.bn)
    batch = int(args.batch or config["distill"]["batch_per_dataset"])
    tiles = max(int(args.tiles or batch), args.n)
    if hasattr(model.student, "dense_keep"):
        model.student.dense_keep = args.n
    view = tuple(args.view) if args.view else (int(config["distill"]["flair_crop"]), 1)
    print(f"[viz] {tiles} tiles per dataset in forward passes of {batch}, figure shows {args.n}; FLAIR view {view}")
    print(f"[viz] {n_bn} BatchNorm layers, statistics: {args.bn if n_bn else 'n/a (no BatchNorm)'}")

    dm = DistillDataModule(config, lookup, bands)
    dm.setup()
    os.makedirs(args.out, exist_ok=True)
    title = f"{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))} step {ck.get('global_step', '?')}"
    lines = []
    for name in args.datasets:
        teachers = [t for t in teachers_of(model, name) if args.teachers is None or t in args.teachers]
        if not teachers:
            continue
        keep = args.keep_olmo if name == "olmo" else args.keep_flair
        res, img, dense = run_dataset(model, name, getattr(dm, f"{name}_val"), tiles, batch, args.seed, dev, args.n,
                                      view, keep, teachers, args.tnorm)
        tag = (name if name == "olmo" else f"flair_c{view[0]}_x{view[1]}") + (f"_keep{''.join(map(str, keep))}" if keep else "")
        for t, r in res.items():
            sc = scores(r["tgt"], r["pred"])                                       # all tiles
            k = min(args.n, len(r["idx"]))
            shown = dict(idx=r["idx"][:k], tgt=r["tgt"][:k], pred=r["pred"][:k], g=r["g"])
            sc_shown = dict(sc, cos_per_image=sc["cos_per_image"][:k], local_per_image=sc["local_per_image"][:k])
            path = os.path.join(args.out, f"pca_{tag}_{t}.png")
            render(f"{tag} / {t}", shown, sc_shown, img[:k], None if dense is None else dense[:k], path, title)
            line = (f"{tag:22s} {t:9s} [{args.tnorm}] cos {sc['cos']:.3f} | global {sc['glob']:.3f} | local {sc['local']:.3f} | "
                    f"oracle {sc['oracle']:.3f} | spread x{sc['spread']:.2f} | grid {r['g']} | {len(r['idx'])} tiles -> {path}")
            print(line)
            lines.append(line)
    with open(os.path.join(args.out, "scores.txt"), "a") as f:
        f.write(f"{args.ckpt} (step {ck.get('global_step', '?')})\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
