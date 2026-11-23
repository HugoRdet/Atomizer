#!/usr/bin/env python3
"""
pca_dinov3.py -- PCA visualizations of DINOv3 patch features on FLAIR-HUB aerial (VHR) patches.

Samples random patches from the index built by build_flair_index.py (reads the TIFFs straight from the zips),
runs one or several DINOv3 variants, and saves one row per patch:

    [ RGB ] ([ NIR-R-G ]) [ variant 1 PCA ] [ variant 2 PCA ] ...

Variants are given as NAME=PATH[@SIZE]:  PATH is a local DINOv3 folder (transformers format), SIZE the input side in
pixels (multiple of 16; default = native 512). Examples:
    --variant lvd=checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m
    --variant sat=checkpoints/dinov3/dinov3-vitl16-pretrain-sat493m
    --variant sat176=checkpoints/dinov3/dinov3-vitl16-pretrain-sat493m@176
FLAIR patches are 512 px at 20 cm; SAT-493M was trained on 512 px crops at 0.6 m, so @176 gives ~0.58 m/px
(the scale the satellite model saw). The effective ground resolution and grid size are printed per variant.

--pca-scope tile : one PCA per patch and variant (shows within-patch structure; colors not comparable across rows)
--pca-scope set  : one PCA per variant over all displayed patches (colors comparable across rows)
"""
import argparse
import io
import os
import time
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

p = argparse.ArgumentParser()
p.add_argument("--index", default="idx_flair/flair_images.csv")
p.add_argument("--root", required=True, help="folder containing the FLAIR-HUB zips (e.g. .../flair_hub/data)")
p.add_argument("--variant", action="append", required=True, help="NAME=PATH[@SIZE], repeatable")
p.add_argument("--n", type=int, default=16, help="number of patches (rows)")
p.add_argument("--split", choices=["train", "test", "all"], default="train")
p.add_argument("--pca-scope", choices=["tile", "set"], default="tile")
p.add_argument("--show-irc", action="store_true", help="add a NIR-R-G false-color column")
p.add_argument("--bs", type=int, default=4)
p.add_argument("--bf16", action="store_true", help="run the model under bfloat16 autocast (cuda only)")
p.add_argument("--device", default="auto")
p.add_argument("--cell", type=float, default=2.4, help="inches per image cell")
p.add_argument("--out", default="pca_dinov3.png")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                      ("cpu" if args.device == "auto" else args.device))
print("device:", device)
torch.set_grad_enabled(False)

# normalization statistics (the SAT-493M ones are the values used in public DINOv3 demos, not from the model card)
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SAT = dict(mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143])
GSD_NATIVE, SIDE_NATIVE = 0.2, 512  # FLAIR-HUB aerial: 20 cm, 512 px


# --- helpers begin
def pca_rgb(tok):
    """tok: (m, D) -> (m, 3) colors in [0, 1] from the top-3 principal components; also explained variance."""
    X = tok.astype(np.float64)
    X = X - X.mean(0)
    w, V = np.linalg.eigh(X.T @ X)
    top = V[:, ::-1][:, :3]
    Z = X @ top
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    return np.clip((Z - lo) / (hi - lo + 1e-9), 0, 1), w[::-1][:3] / w.sum()


def parse_variant(s):
    name, rest = s.split("=", 1)
    path, _, size = rest.partition("@")
    return name, path, (int(size) if size else None)
# --- helpers end


def read_tif(data):
    """bytes -> (H, W, C) numpy array."""
    try:
        import tifffile
        a = tifffile.imread(io.BytesIO(data))
    except ImportError:
        try:
            from rasterio.io import MemoryFile
            with MemoryFile(data) as mf, mf.open() as ds:
                a = ds.read()
        except ImportError:
            raise SystemExit("no TIFF reader: pip install tifffile imagecodecs   (or rasterio)")
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[0] <= 8 and a.shape[-1] > 8:  # (C, H, W) -> (H, W, C)
        a = np.moveaxis(a, 0, -1)
    return a


# ---------------------------------------------------------------- data
df = pd.read_csv(args.index, dtype=str)
if args.split != "all":
    df = df[df["split"] == args.split]
df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
zcache, imgs = {}, []
for r in df.itertuples():
    if r.zip not in zcache:
        zcache[r.zip] = zipfile.ZipFile(os.path.join(args.root, r.zip))
    imgs.append(read_tif(zcache[r.zip].read(r.member)))
shapes = {a.shape for a in imgs}
assert len(shapes) == 1, f"patches have different shapes: {shapes}"
imgs = np.stack(imgs)  # (N, H, W, C)
N, H, W, C = imgs.shape
print(f"{N} patches, shape {imgs.shape[1:]}, dtype {imgs.dtype}, departments {sorted(set(df.dept))}")
assert H == W and H % 16 == 0 and C >= 3, "expected square patches with >= 3 bands and a side divisible by 16"
print("per-band mean (expect R,G,B,NIR order; NIR usually > R on vegetated land):",
      imgs.reshape(-1, C).mean(0).round(1))
rgb_u8 = imgs[..., :3]
if args.show_irc:
    assert C >= 4, "need a NIR band for --show-irc"
    irc = imgs[..., [3, 0, 1]]

# ---------------------------------------------------------------- models
from transformers import AutoModel  # noqa: E402

variants = [parse_variant(v) for v in args.variant]
results = {}  # name -> (N, g, g, D)
for name, path, size in variants:
    t0 = time.time()
    size = size or H
    assert size % 16 == 0, "SIZE must be a multiple of 16 (patch size)"
    stats = SAT if ("sat493m" in path.lower() or name.lower().startswith("sat")) else IMAGENET
    model = AutoModel.from_pretrained(path).to(device).eval()
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    mean = torch.tensor(stats["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 3, 1, 1)
    chunks = []
    for i in range(0, N, args.bs):
        x = torch.from_numpy(rgb_u8[i:i + args.bs]).to(device).permute(0, 3, 1, 2).float() / 255.0
        if size != H:
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        x = (x - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
            out = model(pixel_values=x)
        h = out.last_hidden_state.float().cpu()          # (b, 1 + registers + patches, D)
        tok = h[:, 1 + n_reg:]
        g = size // 16
        assert tok.shape[1] == g * g, f"{name}: expected {g * g} patch tokens, got {tok.shape[1]} (registers={n_reg}, seq={h.shape[1]})"
        chunks.append(tok.reshape(tok.shape[0], g, g, tok.shape[2]))
    results[name] = torch.cat(chunks).numpy()
    print(f"[{name}] {path}  input {size}px (~{GSD_NATIVE * SIDE_NATIVE / size:.2f} m/px), grid {g}x{g}, "
          f"dim {results[name].shape[-1]}, registers {n_reg}, norm {'SAT' if stats is SAT else 'ImageNet'}  ({time.time() - t0:.0f}s)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- PCA colors
cols = {}
for name, F in results.items():
    n, g, _, D = F.shape
    if args.pca_scope == "tile":
        c = np.empty((n, g, g, 3))
        ev = []
        for i in range(n):
            ci, e = pca_rgb(F[i].reshape(-1, D))
            c[i] = ci.reshape(g, g, 3)
            ev.append(e)
        print(f"[{name}] mean variance explained by PC1-3 per patch: {np.mean(ev, 0).round(3)}")
    else:
        ci, e = pca_rgb(F.reshape(-1, D))
        c = ci.reshape(n, g, g, 3)
        print(f"[{name}] variance explained by PC1-3 over all patches: {e.round(3)}")
    cols[name] = c

# ---------------------------------------------------------------- figure
panels = [("RGB", rgb_u8)]
if args.show_irc:
    panels.append(("NIR-R-G", irc))
panels += [(f"{name}\n{(size or H)}px, {cols[name].shape[1]}x{cols[name].shape[1]} tokens", cols[name]) for name, _, size in variants]
fig, ax = plt.subplots(N, len(panels), figsize=(len(panels) * args.cell, N * args.cell), squeeze=False)
for i in range(N):
    for j, (title, arr) in enumerate(panels):
        ax[i, j].imshow(arr[i], interpolation="nearest")
        ax[i, j].axis("off")
        if i == 0:
            ax[0, j].set_title(title, fontsize=8)
plt.tight_layout(pad=0.3)
plt.savefig(args.out, dpi=100)
print("saved", args.out)
