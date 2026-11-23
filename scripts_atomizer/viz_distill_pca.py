"""
viz_distill_pca.py -- standalone PCA figures and mimicry scores for a distillation checkpoint.

Where it goes: scripts_atomizer/viz_distill_pca.py

Loads a Lightning checkpoint of Model_Distill (any student: Atomiser_Unet or AtomizerPTv3), runs both teachers and the
student on a few validation tiles of each dataset, and writes one figure per dataset plus a score table.

    export PYTHONPATH=/path/to/PointTransformerV3/src:$PYTHONPATH      # PTv3 student
    python scripts_atomizer/viz_distill_pca.py --config training/configs/config_distill_ptv3.yaml \
        --ckpt ./checkpoints/distill/ptv3_distill_s42/last.ckpt --n 6 --out ./viz_pca/ptv3_s42

Figure columns, one row per tile:
    input | teacher PCA | student PCA (teacher scale) | student PCA (own scale) | 1 - cos [| student pixels PCA]
  * the PCA basis is fitted on the (standardized) teacher tokens of the figure and applied to the student's predictions,
    which live in the same space: same axes, so the hues are comparable;
  * "teacher scale" stretches the student with the teacher's 1-99 % range (shows how much of the teacher's amplitude it
    reproduces), "own scale" with its own range (shows its pattern even when the amplitude is still small);
  * 1 - cos per teacher token, with a colour bar (0 = identical direction, 1 = unrelated);
  * PTv3 student: its per-pixel level-0 features (before the average over each teacher patch), own PCA basis,
    rank-equalized colours.

Scores (printed and saved to scores.txt), per dataset, over the figure's tiles:
    cos(student, teacher)     mean cosine between student prediction and teacher token
    cos(image mean, teacher)  same for a trivial predictor that outputs each image's average teacher token
    spread                    std of the student along the teacher's top-3 PCA axes / std of the teacher
A student that mimics the teacher has cos(student) clearly above the image-mean baseline and a spread near 1.

BatchNorm (PTv3 with norm "bn"): the running statistics mix the two datasets, so by default (--bn batch) BatchNorm
layers normalize each (single-dataset) batch with its own statistics, as in training. --bn eval uses the running
statistics instead. No effect on LayerNorm models.

Batch: the student runs on --batch tiles at once (default: distill.batch_per_dataset, the training batch), so
per-batch BatchNorm statistics come from as many tiles as in training. --tiles N scores N tiles per dataset in several
forward passes (e.g. --tiles 128 --batch 32: 4 passes). The figure shows the first --n tiles.
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


@torch.no_grad()
def run_chunk(model, name, ds, idx, dev):
    """One forward pass on the tiles `idx` (one batch)."""
    L = model.layout[name]
    t, hw, patch = L["teacher"], L["hw"], L["patch"]
    batch = to_device(collate_distill([ds[i] for i in idx]), dev)
    tokens = batch["groups"][next(iter(batch["groups"]))]["tokens"]
    g = hw // patch
    amp = torch.autocast(dev.type, dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    with amp:
        tgt = model._teacher(t)(batch["teacher"])                                  # [B, g, g, D]
        assert tgt.shape[1] == g and tgt.shape[2] == g, f"teacher grid {tuple(tgt.shape[1:3])} != {g}"
        tgt = model.tnorm[t](tgt.flatten(1, 2), 0.0)                                # standardized, no stats update
        m = int(float(model.dc.get("m_factor", 1.3)) * L["k"])
        d0, geom = model.student.encode(batch, L["k"], m)
        q = token_centers_from_tokens(tokens, hw, hw, patch)
        pred = model.student.predict(t, d0, geom, q)
    tgt, pred = tgt.float().cpu(), pred.float().cpu()
    dense = getattr(model.student, "last_dense", None)
    dense = None if dense is None else dense[:len(idx)].float().cpu()
    return dict(idx=idx, tgt=tgt, pred=pred, g=g, img=input_rgb(name, batch["teacher"], len(idx)), dense=dense)


def run_dataset(model, name, ds, n_tiles, batch, seed, dev, keep):
    """n_tiles validation tiles in forward passes of `batch` tiles. Teacher targets and student predictions of all tiles
    are kept (CPU) for the scores; images and per-pixel features only for the first `keep` tiles (the figure)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_tiles, len(ds)), replace=False).tolist()
    tgts, preds, imgs, denses, g = [], [], [], [], None
    for s0 in range(0, len(idx), batch):
        chunk = idx[s0:s0 + batch]
        r = run_chunk(model, name, ds, chunk, dev)
        tgts.append(r["tgt"]); preds.append(r["pred"]); g = r["g"]
        need = keep - sum(x.shape[0] for x in imgs)
        if need > 0:
            imgs.append(r["img"][:need])
            if r["dense"] is not None:
                denses.append(r["dense"][:need])
        print(f"[viz] {name}: {min(s0 + batch, len(idx))}/{len(idx)} tiles")
    return dict(idx=idx, tgt=torch.cat(tgts), pred=torch.cat(preds), g=g, img=torch.cat(imgs),
                dense=torch.cat(denses) if denses else None)


def scores(tgt, pred):
    cos_s = F.cosine_similarity(pred, tgt, dim=-1)                                  # [B, M]
    base = tgt.mean(dim=1, keepdim=True).expand_as(tgt)                             # image-mean predictor
    cos_b = F.cosine_similarity(base, tgt, dim=-1)
    D = tgt.shape[-1]
    mu, V = pca3(tgt.reshape(-1, D))
    zt, zp = (tgt.reshape(-1, D) - mu) @ V, (pred.reshape(-1, D) - mu) @ V
    spread = float(zp.std(0).mean() / (zt.std(0).mean() + 1e-12))
    return dict(cos_student=float(cos_s.mean()), cos_image_mean=float(cos_b.mean()), spread=spread,
                cos_per_image=cos_s.mean(1).tolist(), base_per_image=cos_b.mean(1).tolist())


def render(name, r, sc, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt, pred, g, img, dense = r["tgt"], r["pred"], r["g"], r["img"], r["dense"]
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

    cols = ["input", "teacher PCA", "student PCA\n(teacher scale)", "student PCA\n(own scale)", "1 - cos"]
    if cd is not None:
        cols.append("student pixels PCA")
    fig, ax = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.3 * n + 0.6), squeeze=False)
    for i in range(n):
        ax[i, 0].imshow(img[i].numpy())
        ax[i, 1].imshow(ct[i].numpy(), interpolation="nearest")
        ax[i, 2].imshow(cp_t[i].numpy(), interpolation="nearest")
        ax[i, 3].imshow(cp_o[i].numpy(), interpolation="nearest")
        im = ax[i, 4].imshow(err[i].numpy(), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        if cd is not None:
            ax[i, 5].imshow(cd[i].numpy(), interpolation="nearest")
        ax[i, 0].set_ylabel(f"#{r['idx'][i]}\ncos {sc['cos_per_image'][i]:.2f} | base {sc['base_per_image'][i]:.2f}",
                            fontsize=7)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=9)
    fig.colorbar(im, ax=ax[:, 4], fraction=0.05, pad=0.02, label="1 - cos")
    fig.suptitle(f"{title} | {name}: cos(student) {sc['cos_student']:.3f} vs cos(image mean) "
                 f"{sc['cos_image_mean']:.3f} | spread x{sc['spread']:.2f}", fontsize=9)
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
    ap.add_argument("--out", default="./viz_pca")
    ap.add_argument("--bn", default="batch", choices=["batch", "eval"],
                    help="BatchNorm statistics: per batch (as in training) or running (eval mode)")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
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
    print(f"[viz] {tiles} tiles per dataset in forward passes of {batch}, figure shows {args.n}")
    print(f"[viz] {n_bn} BatchNorm layers, statistics: {args.bn if n_bn else 'n/a (no BatchNorm)'}")

    dm = DistillDataModule(config, lookup, bands)
    dm.setup()
    os.makedirs(args.out, exist_ok=True)
    title = f"{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))} step {ck.get('global_step', '?')}"
    lines = []
    for name in args.datasets:
        r = run_dataset(model, name, getattr(dm, f"{name}_val"), tiles, batch, args.seed, dev, args.n)
        sc = scores(r["tgt"], r["pred"])                                  # all tiles
        k = min(args.n, len(r["idx"]))
        shown = dict(idx=r["idx"][:k], tgt=r["tgt"][:k], pred=r["pred"][:k], g=r["g"], img=r["img"][:k],
                     dense=None if r["dense"] is None else r["dense"][:k])
        sc_shown = dict(sc, cos_per_image=sc["cos_per_image"][:k], base_per_image=sc["base_per_image"][:k])
        path = os.path.join(args.out, f"pca_{name}.png")
        render(name, shown, sc_shown, path, title)
        line = (f"{name:6s} cos(student, teacher) {sc['cos_student']:.3f} | cos(image mean, teacher) "
                f"{sc['cos_image_mean']:.3f} | spread x{sc['spread']:.2f} | {len(r['idx'])} tiles -> {path}")
        print(line)
        lines.append(line)
    with open(os.path.join(args.out, "scores.txt"), "w") as f:
        f.write(f"{args.ckpt}\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
