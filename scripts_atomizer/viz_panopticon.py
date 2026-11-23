"""
viz_panopticon.py -- Panopticon (any-sensor ViT-B/14) as a candidate teacher, on Sentinel (OlmoEarth tiles) and FLAIR VHR.

Where it goes: scripts_atomizer/viz_panopticon.py   (imports load_anyup / run_anyup from viz_dino_upsample.py)

For each dataset, on the same validation tiles as the distillation, the figure shows (one row per tile):
    input RGB | current teacher (OlmoEarth / DINOv3 SAT) | Panopticon all bands @ s1 | Panopticon RGB only @ s1
    | Panopticon all bands @ s2 | AnyUp(Panopticon all bands @ s1) -> --anyup_size
Each model has its own PCA basis (fitted on all shown tiles of that model; different feature spaces), so compare
structure, not colours. Panopticon all bands, RGB only and AnyUp(@s1) share one basis: there the colours ARE comparable.

Scores (printed, saved to panopticon_scores.txt), over --tiles tiles:
    CKA(model, current teacher)   linear CKA after average-pooling both maps to their common (coarser) grid
    cos / CKA(all bands, RGB)     how much Panopticon's features change when the non-RGB bands are removed
                                  (1 = it ignores them; DINOv3 / OlmoEarth cannot be tested this way)
    detail(AnyUp)                 std inside each native cell / std across cells (0 = no sub-patch detail)

Panopticon input: the student's own per-band values (dataset z-score: ~standard normal per channel, as Panopticon
expects), resized to the input side (multiple of 14) with bilinear interpolation; channel ids = wavelengths in nm.
Sentinel-1 (VV, VH) is left out unless --sar_ids is given (Panopticon codes SAR channels with negative ids, see its repo).
FLAIR: R, G, B, NIR (elevation, if present in the config, is not given to Panopticon).

Setup (once, on a node with internet), then run on a GPU node:
    python -c "import torch; torch.hub.set_dir('./checkpoints/panopticon'); torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14')"
    python scripts_atomizer/viz_panopticon.py --config training/configs/config_distill_ptv3.yaml \
        --sizes_olmo 224 448 --sizes_flair 224 448 --anyup_size 128 --n 6 --tiles 32 --out ./viz_pca/panopticon
(for a config with flair_elevation, add --bands_yaml ./data/bands_info/bands_elev.yaml)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from training.distill.teachers import DinoV3Teacher, OlmoEarthTeacher
from training.distill.trainer_distill import DistillDataModule
from training.utils import Lookup_encoding, read_yaml
from training.utils.datasets.utils_dataset_distill import OLMO_S1, OLMO_S2, collate_distill
from viz_dino_upsample import load_anyup, pca3, run_anyup

S2_NM = {"B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740, "B07": 783, "B08": 842,
         "B8A": 865, "B09": 945, "B11": 1610, "B12": 2190}
AERIAL_NM = [664, 559, 493, 842]                       # R, G, B, NIR (Sentinel-2-like centres)


# ----------------------------------------------------------------------------------------------- model
def load_panopticon(hub_dir):
    """Offline when possible: Panopticon's hubconf.panopticon_vitb14 re-downloads its weights on EVERY call (no
    existence check), which fails on nodes without internet. If the cached repo and the weights are both in hub_dir
    (filled once by torch.hub.load on a login node), build the model with the repo's own constructor and load them."""
    repo = os.path.join(hub_dir, "Panopticon-FM_panopticon_main")
    ckpt = os.path.join(hub_dir, "panopticon_vitb14_teacher.pth")
    if os.path.isdir(repo) and os.path.isfile(ckpt):
        sys.path.insert(0, repo)                         # hubconf imports dinov2.* from the repo root
        import hubconf
        model = hubconf._panopticon_vitb14()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        print(f"[panopticon] loaded offline: {ckpt}")
        return model
    print(f"[panopticon] {repo} or {ckpt} missing: downloading through torch.hub (needs internet)")
    torch.hub.set_dir(hub_dir)
    return torch.hub.load("Panopticon-FM/panopticon", "panopticon_vitb14", trust_repo=True)


# ----------------------------------------------------------------------------------------------- inputs
def band_values(batch):
    """Student values -> [B, C, H, W] (z-scored per band; channel-major tokens)."""
    grp = batch["groups"][next(iter(batch["groups"]))]
    C, H, W = (int(x) for x in grp["shape"])
    return grp["tokens"][..., 0].reshape(-1, C, H, W).float()


def rgb_uint8(x, rgb_idx):
    """[B, C, H, W] z-scored -> uint8 RGB [B, 3, H, W] (2-98 % stretch per image), for figures and the AnyUp guide."""
    v = x[:, rgb_idx]
    lo = torch.quantile(v.flatten(1), 0.02, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(v.flatten(1), 0.98, dim=1).view(-1, 1, 1, 1)
    return ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1).mul(255).round().to(torch.uint8)


@torch.no_grad()
def run_panopticon(model, x, nm, size, dev):
    """x [B, C, H, W] z-scored, nm [C] channel ids -> [B, g, g, 768], g = size / 14."""
    if size % 14:
        raise ValueError(f"Panopticon input side {size} is not a multiple of 14")
    xi = F.interpolate(x.to(dev), size=(size, size), mode="bilinear", align_corners=False)
    ids = torch.tensor(nm, device=dev, dtype=torch.long if all(float(v).is_integer() for v in nm) else torch.float)
    ids = ids.unsqueeze(0).expand(x.shape[0], -1)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    with amp:
        out = model.get_intermediate_layers({"imgs": xi, "chn_ids": ids}, n=1, return_class_token=False)
    tok = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(tok, (list, tuple)):
        tok = tok[0]
    g = size // 14
    assert tok.shape[1] == g * g, f"Panopticon returned {tok.shape[1]} tokens, expected {g}x{g}"
    return tok.float().reshape(x.shape[0], g, g, -1)


# ----------------------------------------------------------------------------------------------- scores
def pool(x, g):
    return F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), g).permute(0, 2, 3, 1)


def cka(a, b):
    """Linear CKA of two token maps [B, g, g, D1], [B, g', g', D2] on their common grid."""
    g = min(a.shape[1], b.shape[1])
    X, Y = pool(a, g).reshape(-1, a.shape[-1]).double(), pool(b, g).reshape(-1, b.shape[-1]).double()
    X, Y = X - X.mean(0), Y - Y.mean(0)
    hsic = (X.T @ Y).pow(2).sum()
    return float(hsic / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt() + 1e-12))


def detail(x, g):
    f = x.shape[1] // g
    c = x.reshape(x.shape[0], g, f, g, f, -1)
    cm = c.mean(dim=(2, 4))
    within = (c - cm[:, :, None, :, None]).pow(2).mean().sqrt()
    return float(within / ((cm - cm.mean(dim=(1, 2), keepdim=True)).pow(2).mean().sqrt() + 1e-12))


# ----------------------------------------------------------------------------------------------- figure
def render(img, cols, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bases = {}
    for name, (fmap, basis) in cols.items():                 # one PCA basis per model (key `basis`)
        if basis not in bases:
            X = fmap.reshape(-1, fmap.shape[-1])
            mu, V = pca3(X)
            z = (X - mu) @ V
            bases[basis] = (mu, V, torch.quantile(z, 0.01, dim=0), torch.quantile(z, 0.99, dim=0))
    n = img.shape[0]
    names = ["input"] + list(cols)
    fig, ax = plt.subplots(n, len(names), figsize=(2.3 * len(names), 2.3 * n + 0.5), squeeze=False)
    for i in range(n):
        ax[i, 0].imshow(img[i].permute(1, 2, 0).numpy())
        for j, (name, (fmap, basis)) in enumerate(cols.items(), start=1):
            mu, V, lo, hi = bases[basis]
            ax[i, j].imshow((((fmap[i] - mu) @ V - lo) / (hi - lo + 1e-6)).clamp(0, 1).numpy(), interpolation="nearest")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    for j, name in enumerate(names):
        sub = "" if j == 0 else f"\n{cols[name][0].shape[1]}x{cols[name][0].shape[2]}"
        ax[0, j].set_title(name + sub, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="+", default=["olmo", "flair"], choices=["olmo", "flair"])
    ap.add_argument("--sizes_olmo", type=int, nargs="+", default=[224, 448], help="Panopticon input sides (x14)")
    ap.add_argument("--sizes_flair", type=int, nargs="+", default=[224, 448])
    ap.add_argument("--sar_ids", type=float, nargs=2, default=None,
                    help="Panopticon channel ids for VV, VH (negative SAR codes from its repo); default: S1 left out")
    ap.add_argument("--anyup_size", type=int, default=128, help="AnyUp output side (0: no AnyUp)")
    ap.add_argument("--anyup_dir", default="./checkpoints/anyup")
    ap.add_argument("--q_chunk", type=int, default=256)
    ap.add_argument("--hub_dir", default="./checkpoints/panopticon")
    ap.add_argument("--pca_olmo", default="column", choices=["column", "model"],
                    help="column: one PCA basis per column (best to see whether each map has structure); "
                         "model: Panopticon all/RGB/AnyUp share one basis (colours comparable between them)")
    ap.add_argument("--pca_flair", default="model", choices=["column", "model"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./viz_pca/panopticon")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    config = read_yaml(args.config)
    dc = config["distill"]
    dc.pop("band_drop", None)                                # validation tiles anyway; no student-side drops
    bands = read_yaml(args.bands_yaml)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), bands, config)
    dm = DistillDataModule(config, lookup, bands)
    dm.setup()

    os.makedirs(args.hub_dir, exist_ok=True)
    pano = load_panopticon(os.path.abspath(args.hub_dir)).to(dev).eval()
    up = load_anyup("wimmerth/anyup", "anyup", args.anyup_dir).to(dev) if args.anyup_size else None
    os.makedirs(args.out, exist_ok=True)
    lines = []

    for name in args.datasets:
        ds = getattr(dm, f"{name}_val")
        if name == "olmo":
            teacher = OlmoEarthTeacher(dc["olmo_ckpt"], int(dc["olmo_patch"])).to(dev).eval()
            tname, sizes = "OlmoEarth", args.sizes_olmo
            chans = list(range(12)) + ([12, 13] if args.sar_ids else [])
            nm = [S2_NM[b] for b in OLMO_S2] + (list(args.sar_ids) if args.sar_ids else [])
            rgb_idx = [OLMO_S2.index("B04"), OLMO_S2.index("B03"), OLMO_S2.index("B02")]
        else:
            teacher = DinoV3Teacher(dc["dino_ckpt"], dc.get("dino_norm", "auto"), student_patch=16).to(dev).eval()
            tname, sizes = "DINOv3 SAT", args.sizes_flair
            chans, nm, rgb_idx = [0, 1, 2, 3], AERIAL_NM, [0, 1, 2]
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(ds), size=min(max(args.tiles, args.n), len(ds)), replace=False).tolist()
        keep, acc, imgs = {}, {}, []
        for s0 in range(0, len(idx), args.batch):
            batch = collate_distill([ds[i] for i in idx[s0:s0 + args.batch]])
            tin = {k: v.to(dev) for k, v in batch["teacher"].items()}
            x = band_values(batch)
            x_all, x_rgb = x[:, chans], x[:, rgb_idx]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
            with torch.no_grad(), amp:
                ref = teacher(tin).float()
            maps = {f"{tname}": ref}
            s1 = sizes[0]
            maps[f"Panopticon all @{s1}"] = run_panopticon(pano, x_all, nm, s1, dev)
            maps[f"Panopticon RGB @{s1}"] = run_panopticon(pano, x_rgb, [nm[chans.index(c)] for c in rgb_idx], s1, dev)
            for s in sizes[1:]:
                maps[f"Panopticon all @{s}"] = run_panopticon(pano, x_all, nm, s, dev)
            guide = rgb_uint8(x, rgb_idx).to(dev)
            if up is not None:
                maps[f"AnyUp(Pano @{s1})"] = run_anyup(up, guide, maps[f"Panopticon all @{s1}"], args.anyup_size, args.q_chunk)
            # scores
            for k, v in maps.items():
                if k != tname:
                    acc.setdefault(f"CKA({k}, {tname})", []).append(cka(v, ref))
            a, r = maps[f"Panopticon all @{s1}"], maps[f"Panopticon RGB @{s1}"]
            acc.setdefault("cos(Pano all, Pano RGB)", []).append(float(F.cosine_similarity(a, r, dim=-1).mean()))
            acc.setdefault("CKA(Pano all, Pano RGB)", []).append(cka(a, r))
            if up is not None:
                acc.setdefault("detail AnyUp", []).append(detail(maps[f"AnyUp(Pano @{s1})"], a.shape[1]))
            need = args.n - sum(t.shape[0] for t in imgs)
            if need > 0:
                imgs.append(guide[:need].cpu())
                for k, v in maps.items():
                    keep.setdefault(k, []).append(v[:need].cpu())
            print(f"[panopticon] {name}: {min(s0 + args.batch, len(idx))}/{len(idx)} tiles")
        keep = {k: torch.cat(v) for k, v in keep.items()}
        s1 = sizes[0]
        cols = {k: (v, "teacher" if k == tname else (f"pano{s1}" if k.startswith("AnyUp") else f"pano{k.split('@')[-1]}"))
                for k, v in keep.items()}               # AnyUp output lives in Panopticon's space: same basis as @s1
        if getattr(args, f"pca_{name}") == "column":
            cols = {k: (v, k) for k, (v, _) in cols.items()}                      # one basis per column
        path = os.path.join(args.out, f"panopticon_{name}.png")
        mode = getattr(args, f"pca_{name}")
        render(torch.cat(imgs), cols, path, f"{name}: {tname} vs Panopticon (PCA: "
               + ("one basis per column" if mode == "column" else "one basis per model / input size") + ")")
        lines.append(f"== {name} ({len(idx)} val tiles; Panopticon channels {len(chans)}: ids {nm}) -> {path}")
        lines += [f"  {k:45s} {np.mean(v):.3f}" for k, v in acc.items()]
        del teacher
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "panopticon_scores.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
