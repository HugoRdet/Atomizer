"""
viz_dino_upsample.py -- denser DINOv3 targets on FLAIR: native grid vs input upsampling vs AnyUp.

Where it goes: scripts_atomizer/viz_dino_upsample.py
Teacher only (no student, no checkpoint): uses the distillation config for the DINOv3 checkpoint, normalization and the
FLAIR validation tiles (distill.flair_crop).

Variants (one figure column each):
    native          DINOv3 on the crop as in training with flair_patch 16      -> (crop/16)^2 tokens of 16 x 16 px
    input xF        crop upsampled F x (bicubic) before DINOv3, --input_up      -> (F crop/16)^2 tokens of 16/F px
                    (identical to training with flair_patch = 16/F: same DinoV3Teacher code path)
    AnyUp from X    AnyUp (Wimmer et al., ICLR 2026) applied to variant X, guided by the crop's RGB, to --anyup_size

Colours: ONE PCA basis fitted on the native tokens of all shown tiles, applied to every variant (all live in the same
DINOv3 feature space), stretched with the native 1-99 % range, so hues are comparable between columns.

Scores (printed, saved to dino_upsample_scores.txt), each variant average-pooled back to the native grid:
    cos to native        does the denser map still agree with the teacher's own tokens on average?
    ... border / inside  the same, split into the outer ring of native tokens and the rest (edge effects)
    detail               std of the variant inside each native cell / std across native cells (0 = no sub-patch detail)

Setup (Jean Zay compute nodes have no internet: run the first line once on a login node, from the repo root):
    python -c "import torch; torch.hub.set_dir('./checkpoints/anyup'); torch.hub.load('wimmerth/anyup', 'anyup')"
    # -> code in ./checkpoints/anyup/wimmerth_anyup_main/, weights in ./checkpoints/anyup/checkpoints/
    python scripts_atomizer/viz_dino_upsample.py --config training/configs/config_distill_ptv3.yaml \
        --n 6 --tiles 32 --input_up 2 4 --anyup_from native input_x2 --anyup_size 128 --out ./viz_pca/dino_upsample
--anyup_src may also be a local clone of the repo (loaded with source="local").
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from training.distill.teachers import DinoV3Teacher
from training.distill.trainer_distill import DistillDataModule
from training.utils import Lookup_encoding, read_yaml
from training.utils.datasets.utils_dataset_distill import collate_distill

IMAGENET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


# ----------------------------------------------------------------------------------------------- models
def load_anyup(src, name, hub_dir):
    """Code and weights are cached in hub_dir (weights in hub_dir/checkpoints); downloaded on the first call only."""
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(os.path.abspath(hub_dir))
    if os.path.isdir(src):
        m = torch.hub.load(src, name, source="local")
    else:
        m = torch.hub.load(src, name)
    return m.eval()


@torch.no_grad()
def run_anyup(up, rgb, feat, size, q_chunk):
    """rgb [B, 3, H, W] uint8, feat [B, g, g, D] -> [B, size, size, D]. Guide image ImageNet-normalized (AnyUp)."""
    guide = rgb.float() / 255.0
    if guide.shape[-1] != size:
        guide = F.interpolate(guide, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    guide = (guide - IMAGENET[0].to(guide)) / IMAGENET[1].to(guide)
    lr = feat.permute(0, 3, 1, 2).float().contiguous()                                  # [B, D, g, g]
    try:
        hr = up(guide, lr, output_size=(size, size), q_chunk_size=q_chunk)
    except TypeError:                                                                   # older signature
        hr = up(guide, lr, output_size=(size, size))
    if hasattr(hr, "last_hidden_state"):
        hr = hr.last_hidden_state
    assert hr.shape[-2:] == (size, size) and hr.shape[1] == lr.shape[1], f"AnyUp output {tuple(hr.shape)}"
    return hr.permute(0, 2, 3, 1).float()


@torch.no_grad()
def run_dino(teacher, rgb, factor, dev):
    """Same code path as training: DinoV3Teacher with up = factor (flair_patch = 16 / factor)."""
    teacher.up = float(factor)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    with amp:
        out = teacher({"rgb": rgb})
    return out.float()                                                                   # [B, g, g, D]


# ----------------------------------------------------------------------------------------------- analysis
def pool_to(x, g):
    """[B, s, s, D] -> [B, g, g, D] (mean over s/g x s/g cells)."""
    B, s, _, D = x.shape
    if s == g:
        return x
    f = s // g
    assert f * g == s, f"{s} is not a multiple of {g}"
    return x.view(B, g, f, g, f, D).mean(dim=(2, 4))


def detail_ratio(x, g):
    """std inside each native cell / std of the cell means (0 for a piecewise-constant map)."""
    B, s, _, D = x.shape
    if s == g:
        return 0.0
    f = s // g
    c = x.view(B, g, f, g, f, D)
    within = (c - c.mean(dim=(2, 4), keepdim=True)).pow(2).mean().sqrt()
    cm = c.mean(dim=(2, 4))                                                             # [B, g, g, D]
    across = (cm - cm.mean(dim=(1, 2), keepdim=True)).pow(2).mean().sqrt()
    return float(within / (across + 1e-12))


def scores(native, var):
    g = native.shape[1]
    cos = F.cosine_similarity(pool_to(var, g), native, dim=-1)                          # [B, g, g]
    ring = torch.zeros(g, g, dtype=torch.bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    return dict(cos=float(cos.mean()), border=float(cos[:, ring].mean()), inside=float(cos[:, ~ring].mean()),
                detail=detail_ratio(var, g))


def pca3(X, n_fit=200000):
    sub = X[torch.randperm(X.shape[0])[:n_fit]]
    mu = sub.mean(0)
    _, V = torch.linalg.eigh((sub - mu).T @ (sub - mu))
    return mu, V[:, -3:].flip(-1)


def to_rgb(x, mu, V, lo, hi):
    return (((x - mu) @ V - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def show_rgb(rgb):
    out = []
    for im in rgb.float().div(255).permute(0, 2, 3, 1):
        lo, hi = torch.quantile(im.reshape(-1), 0.02), torch.quantile(im.reshape(-1), 0.98)
        out.append(((im - lo) / (hi - lo + 1e-6)).clamp(0, 1))
    return torch.stack(out)


def render(img, maps, names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    native = maps[names[0]]
    D = native.shape[-1]
    mu, V = pca3(native.reshape(-1, D))
    z = (native.reshape(-1, D) - mu) @ V
    lo, hi = torch.quantile(z, 0.01, dim=0), torch.quantile(z, 0.99, dim=0)
    n = img.shape[0]
    cols = ["input"] + [f"{k}\n{maps[k].shape[1]}x{maps[k].shape[2]}" for k in names]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.3 * n + 0.5), squeeze=False)
    for i in range(n):
        ax[i, 0].imshow(img[i].numpy())
        for j, k in enumerate(names, start=1):
            ax[i, j].imshow(to_rgb(maps[k][i], mu, V, lo, hi).numpy(), interpolation="nearest")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=6, help="tiles in the figure")
    ap.add_argument("--tiles", type=int, default=32, help="tiles for the scores (>= n)")
    ap.add_argument("--batch", type=int, default=4, help="tiles per teacher pass (input x4 = 1024 px: keep small)")
    ap.add_argument("--input_up", type=int, nargs="*", default=[2, 4], help="input upsampling factors (16 must divide)")
    ap.add_argument("--anyup_from", nargs="*", default=["native", "input_x2"],
                    help="variants AnyUp upsamples ('native', 'input_x2', ...); empty list = no AnyUp")
    ap.add_argument("--anyup_size", type=int, default=128, help="AnyUp output side (<= crop)")
    ap.add_argument("--anyup_src", default="wimmerth/anyup", help="torch.hub repo or local clone of it")
    ap.add_argument("--anyup_dir", default="./checkpoints/anyup",
                    help="cache for AnyUp code + weights (fill it once on a node with internet, see the docstring)")
    ap.add_argument("--anyup_model", default="anyup", help="hub entry point (e.g. anyup_multi_backbone)")
    ap.add_argument("--q_chunk", type=int, default=256, help="AnyUp query chunk (memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./viz_pca/dino_upsample")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    config = read_yaml(args.config)
    dc = config["distill"]
    bands = read_yaml(args.bands_yaml)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), bands, config)
    dm = DistillDataModule(config, lookup, bands)
    dm.setup()
    ds = dm.flair_val
    crop = int(dc["flair_crop"])

    teacher = DinoV3Teacher(dc["dino_ckpt"], dc.get("dino_norm", "auto"), student_patch=16).to(dev).eval()
    for f in args.input_up:
        if 16 % f:
            raise ValueError(f"--input_up {f}: 16 / {f} is not an integer patch")
    up = load_anyup(args.anyup_src, args.anyup_model, args.anyup_dir).to(dev) if args.anyup_from else None
    if up is not None and (args.anyup_size > crop or args.anyup_size % (crop // 16)):
        raise ValueError(f"--anyup_size must be <= {crop} and a multiple of the native grid {crop // 16}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(max(args.tiles, args.n), len(ds)), replace=False).tolist()
    keep, acc, imgs, names = {}, {}, [], None
    for s0 in range(0, len(idx), args.batch):
        batch = collate_distill([ds[i] for i in idx[s0:s0 + args.batch]])
        rgb = batch["teacher"]["rgb"].to(dev)
        maps = {"native": run_dino(teacher, rgb, 1, dev)}
        for f in args.input_up:
            maps[f"input_x{f}"] = run_dino(teacher, rgb, f, dev)
        for src in args.anyup_from:
            if src not in maps:
                raise ValueError(f"--anyup_from {src}: not a computed variant ({list(maps)})")
            maps[f"anyup_from_{src}"] = run_anyup(up, rgb, maps[src], args.anyup_size, args.q_chunk)
        names = list(maps)
        for k in names[1:]:
            for key, v in scores(maps["native"], maps[k]).items():
                acc.setdefault(k, {}).setdefault(key, []).append((v, rgb.shape[0]))
        need = args.n - sum(x.shape[0] for x in imgs)
        if need > 0:
            imgs.append(show_rgb(rgb[:need].cpu()))
            for k in names:
                keep.setdefault(k, []).append(maps[k][:need].cpu())
        print(f"[dino_upsample] {min(s0 + args.batch, len(idx))}/{len(idx)} tiles")

    os.makedirs(args.out, exist_ok=True)
    img = torch.cat(imgs)
    keep = {k: torch.cat(v) for k, v in keep.items()}
    path = os.path.join(args.out, "dino_upsample_flair.png")
    render(img, keep, names, path, f"DINOv3 on FLAIR {crop}x{crop} | PCA basis: native tokens")
    lines = [f"{'variant':22s} {'grid':>9s} {'cos to native':>14s} {'border':>8s} {'inside':>8s} {'detail':>8s}"]
    for k in names[1:]:
        m = {key: sum(v * w for v, w in vals) / sum(w for _, w in vals) for key, vals in acc[k].items()}
        s = keep[k].shape[1]
        lines.append(f"{k:22s} {f'{s}x{s}':>9s} {m['cos']:14.3f} {m['border']:8.3f} {m['inside']:8.3f} {m['detail']:8.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "dino_upsample_scores.txt"), "w") as f:
        f.write(f"{len(idx)} FLAIR val tiles, crop {crop}, seed {args.seed}\n{txt}\n")
    print(f"[dino_upsample] figure -> {path}")


if __name__ == "__main__":
    main()
