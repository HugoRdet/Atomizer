#!/usr/bin/env python3
"""
pca_olmo.py -- compare the patch embeddings of OlmoEarth v1.2-Base and v1.3-Base.

For random tiles of the OlmoEarth pretraining set it picks one random date where both
Sentinel-2 and Sentinel-1 are present, runs both encoders, and saves a figure with one row
per tile:

    [ S2 RGB ]  [ v1.2 PCA ]  [ v1.3 PCA ]  ( + [ disagreement map ] in --color-mode aligned )

Color modes (--color-mode):
  separate : one PCA per model (colors NOT comparable between the two model columns).
  joint    : ONE PCA fitted on the pooled tokens of both models, same basis for both. Same
             basis does not mean same meaning: channel i of v1.2 and channel i of v1.3 are
             unrelated unless the models happen to share a channel layout, so colors can differ
             only because the coordinate systems differ.
  aligned  : (default) v1.3 tokens are first mapped into v1.2's space with a ridge regression,
             then ONE PCA is fitted on [v1.2, mapped v1.3] and applied to both. Equal colors then
             mean "v1.3 linearly predicts what v1.2 encodes there". The 4th column shows where it
             does not (normalized squared prediction error). Run with --n >= --nvis + 8 so the
             mapping is fitted on tiles that are not displayed.

Run from anywhere (see --help for paths):
    python3 pca_olmo.py --n 60 --nvis 25
Reuse saved features without rerunning the models (same --n, --seed, --patch):
    python3 pca_olmo.py --n 60 --nvis 25 --load-feats pca_olmo_feats.npz
"""
import argparse
import io
import time

import h5py
import hdf5plugin  # noqa: F401  (registers the zstd filter used by the h5 files)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import olmoearth_pretrain_minimal as oep
import pandas as pd
import torch
from olmoearth_pretrain_minimal import Normalizer
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=25, help="number of tiles run through the models (>= --nvis)")
p.add_argument("--nvis", type=int, default=25, help="number of rows in the figure")
p.add_argument("--color-mode", choices=["separate", "joint", "aligned"], default="aligned")
p.add_argument("--ridge-lam", type=float, default=1e-3, help="ridge strength (relative) for the alignment / metrics")
p.add_argument("--patch", type=int, default=4, help="OlmoEarth patch size (token grid = 128/patch per side)")
p.add_argument("--dim", type=int, default=768, help="encoder token dimension")
p.add_argument("--bs", type=int, default=4, help="batch size for the encoders")
p.add_argument("--index", default="data/olmo_uncompressed/idx/all.csv")
p.add_argument("--tar-dir", default="data/olmo_uncompressed/tars")
p.add_argument("--ckpt", default="checkpoints/olmo")
p.add_argument("--out", default="pca_olmo.png")
p.add_argument("--feats-out", default="pca_olmo_feats.npz", help="where to save the token grids (empty to skip)")
p.add_argument("--load-feats", default="", help="load token grids from this npz instead of running the models")
p.add_argument("--cell", type=float, default=2.2, help="size in inches of one image cell")
p.add_argument("--device", default="auto", help="auto | cpu | cuda")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
args.nvis = min(args.nvis, args.n)

torch.set_grad_enabled(False)
rng = np.random.default_rng(args.seed)
device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                      ("cpu" if args.device == "auto" else args.device))
print("device:", device)

MODELS = {
    "v1.2": ("OlmoEarth-v1_2-Base", oep.ModelID.OLMOEARTH_V1_2_BASE),
    "v1.3": ("OlmoEarth-v1_3-Base", oep.ModelID.OLMOEARTH_V1_3_BASE),
}
HP = 128 // args.patch  # tokens per side


# --------------------------------------------------------------------------------------
# data: random tiles, one random date where S2 and S1 are both present
# --------------------------------------------------------------------------------------
def load_samples():
    df = pd.read_csv(args.index, usecols=["tar", "offset", "size", "s2_mask", "s1_mask"], dtype=str).fillna("")
    df = df.sample(n=min(len(df), max(4000, 10 * args.n)), random_state=args.seed)
    out = []
    for r in df.itertuples():
        if len(out) == args.n:
            break
        a, b = r.s2_mask, r.s1_mask
        common = [t for t in range(min(len(a), len(b))) if a[t] == "1" and b[t] == "1"]
        if not common:
            continue
        t = int(rng.choice(common))
        try:
            with open(f"{args.tar_dir}/{r.tar}", "rb") as f:
                f.seek(int(r.offset))
                data = f.read(int(r.size))
            with h5py.File(io.BytesIO(data), "r") as h:
                d = {}
                for name in ("sentinel2_l2a", "sentinel1"):
                    m = h[f"missing_timesteps_masks/{name}"][()]
                    assert m[t] and m.sum() == h[name].shape[2], f"{name}: mask/array mismatch"
                    k = int(m[:t].sum())  # stored arrays only hold the present timesteps
                    d[name] = h[name][:, :, k:k + 1, :].astype(np.float32)  # (H, W, 1, C)
                d["ts"] = h["timestamps"][()][t:t + 1].astype(np.int64)  # (1, 3)
            out.append(d)
        except Exception as e:  # noqa: BLE001
            print("skip", r.tar, repr(e))
    return out


samples = load_samples()
N = len(samples)
assert N > 0, "no samples loaded"
args.nvis = min(args.nvis, N)
print("samples loaded:", N)

s2 = np.stack([d["sentinel2_l2a"] for d in samples])  # (N, H, W, 1, 12)
s1 = np.stack([d["sentinel1"] for d in samples])      # (N, H, W, 1, 2)
ts = torch.from_numpy(np.stack([d["ts"] for d in samples]))  # (N, 1, 3)
print("timestamps of the first 4 tiles (raw triples):", ts[:4, 0].tolist())

norm = Normalizer(std_multiplier=2.0)
s2n = norm.normalize(Modality.SENTINEL2_L2A, s2)
s1n = norm.normalize(Modality.SENTINEL1, s1)
for nm, a in (("S2", s2n), ("S1", s1n)):
    print(f"normalized {nm} percentiles 1/50/99:", np.percentile(a, [1, 50, 99]).round(2))


# --------------------------------------------------------------------------------------
# encoders (or load saved features)
# --------------------------------------------------------------------------------------
def find_tokens(out):
    """All tensors shaped (B, HP, HP, ..., dim) anywhere in the encoder output."""
    found = []

    def walk(o, path):
        if torch.is_tensor(o):
            if o.ndim >= 4 and o.shape[-1] == args.dim and o.shape[1] == HP and o.shape[2] == HP:
                found.append((path, o))
        elif hasattr(o, "_asdict"):
            walk(o._asdict(), path)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif hasattr(o, "__dict__"):
            walk(vars(o), path)

    walk(out, "out")
    return found


def dump(o, path="out"):
    if torch.is_tensor(o):
        print(f"  {path}: {tuple(o.shape)} {o.dtype}")
    elif hasattr(o, "_asdict"):
        dump(o._asdict(), path)
    elif isinstance(o, dict):
        for k, v in o.items():
            dump(v, f"{path}.{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            dump(v, f"{path}[{i}]")
    elif hasattr(o, "__dict__"):
        dump(vars(o), path)


feats = {}
if args.load_feats:
    z = np.load(args.load_feats)
    for key in MODELS:
        feats[key] = z[key].astype(np.float32)
        assert feats[key].shape[0] == N and feats[key].shape[1] == HP, (
            f"{key}: saved features {feats[key].shape} do not match N={N}, grid={HP}; "
            "use the same --n/--seed/--patch as when they were saved")
    print("loaded features from", args.load_feats, {k: v.shape for k, v in feats.items()})
else:
    for key, (folder, model_id) in MODELS.items():
        t0 = time.time()
        try:
            model = oep.load_model_from_path(f"{args.ckpt}/{folder}")
        except Exception as e:  # noqa: BLE001
            print(f"[{key}] load_model_from_path failed ({e!r}); falling back to load_model_from_id (needs internet)")
            model = oep.load_model_from_id(model_id, load_weights=True)
        model.to(device).eval()
        chunks = []
        for i in range(0, N, args.bs):
            sl = slice(i, i + args.bs)
            b = ts[sl].shape[0]
            sample = MaskedOlmoEarthSample(
                timestamps=ts[sl].to(device),
                sentinel2_l2a=torch.from_numpy(s2n[sl]).float().to(device),
                sentinel2_l2a_mask=torch.zeros(b, 128, 128, 1, dtype=torch.long, device=device),
                sentinel1=torch.from_numpy(s1n[sl]).float().to(device),
                sentinel1_mask=torch.zeros(b, 128, 128, 1, dtype=torch.long, device=device),
            )
            out = model.encoder(sample, patch_size=args.patch, input_res=10, fast_pass=True)
            found = find_tokens(out)
            if not found:
                print(f"[{key}] no token tensor shaped (B,{HP},{HP},...,{args.dim}) in the output. Structure:")
                dump(out)
                raise SystemExit(1)
            if i == 0:
                print(f"[{key}] token tensors found:", [(pth, tuple(t.shape)) for pth, t in found])
            # collapse any extra dims (time / bandset) -> (b, HP, HP, D), then average over modalities
            red = [(t.float().flatten(3, -2).mean(3) if t.ndim > 4 else t.float()) for _, t in found]
            chunks.append(torch.stack(red).mean(0).cpu())
        feats[key] = torch.cat(chunks).numpy()  # (N, HP, HP, D)
        print(f"[{key}] features {feats[key].shape}  ({time.time() - t0:.0f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.feats_out:
        np.savez_compressed(args.feats_out, **{k: v.astype(np.float16) for k, v in feats.items()})
        print("saved token grids to", args.feats_out)


# --------------------------------------------------------------------------------------
# helpers: PCA colors, ridge alignment, similarity metrics
# --- helpers begin
# --------------------------------------------------------------------------------------
def flat(M):
    return M.reshape(-1, M.shape[-1]).astype(np.float64)


def shared_pca_rgb(Fs):
    """One PCA basis fitted on the pooled tokens of every array in Fs, applied to each.

    Fs: list of (n, hp, hp, D). Returns (list of (n, hp, hp, 3) colors in [0, 1], explained var of PC1-3).
    The same centering, basis and color scaling are used for all arrays, so equal colors mean equal
    coordinates in the shared basis.
    """
    X = np.concatenate([flat(F) for F in Fs])
    mu = X.mean(0)
    Xc = X - mu
    w, V = np.linalg.eigh(Xc.T @ Xc)          # ascending eigenvalues
    top = V[:, ::-1][:, :3]
    expl = w[::-1][:3] / w.sum()
    Zs = [(flat(F) - mu) @ top for F in Fs]
    Zall = np.concatenate(Zs)
    lo, hi = np.percentile(Zall, 1, axis=0), np.percentile(Zall, 99, axis=0)
    cols = [np.clip((Z - lo) / (hi - lo + 1e-9), 0, 1).reshape(*F.shape[:3], 3) for Z, F in zip(Zs, Fs)]
    return cols, expl


def fit_ridge(X, Y, lam):
    """Ridge map X -> Y (X standardized, Y centered). Returns a predict function."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))


def r2_per_channel(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
# --- helpers end


def rgb_view(tile):
    """tile: (H, W, 1, 12) raw S2 DN, band order B02,B03,B04,... -> displayable RGB (B04,B03,B02)."""
    x = tile[:, :, 0, [2, 1, 0]]
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


# --------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------
nv = args.nvis
A, B = feats["v1.2"], feats["v1.3"]
err_map = None

if args.color_mode == "separate":
    (c12,), e12 = shared_pca_rgb([A[:nv]])
    (c13,), e13 = shared_pca_rgb([B[:nv]])
    print("variance explained by PC1-3  v1.2:", e12.round(3), " v1.3:", e13.round(3))
    t12, t13 = "v1.2 (own PCA)", "v1.3 (own PCA)"
elif args.color_mode == "joint":
    (c12, c13), ev = shared_pca_rgb([A[:nv], B[:nv]])
    print("joint PCA variance explained by PC1-3:", ev.round(3))
    t12, t13 = "v1.2 (joint PCA)", "v1.3 (joint PCA)"
else:  # aligned
    fit_idx = np.arange(nv, N) if N - nv >= 8 else np.arange(N)
    if N - nv < 8:
        print("WARNING: fewer than 8 tiles outside the figure, so the alignment is fitted on the displayed "
              "tiles (in-sample, optimistic). Use --n >= --nvis + 8.")
    f_b2a = fit_ridge(flat(B[fit_idx]), flat(A[fit_idx]), args.ridge_lam)      # v1.3 -> v1.2 space
    A_disp = A[:nv].astype(np.float64)
    B_in_A = f_b2a(flat(B[:nv])).reshape(A_disp.shape)
    (c12, c13), ev = shared_pca_rgb([A_disp, B_in_A])
    print("aligned PCA variance explained by PC1-3:", ev.round(3))
    resid = ((flat(A_disp) - flat(B_in_A)) ** 2).sum(1)
    denom = ((flat(A_disp) - flat(A_disp).mean(0)) ** 2).sum(1).mean()
    err_map = (resid / denom).reshape(nv, HP, HP)
    print(f"mean normalized disagreement (1 - R2) on the displayed tiles: {err_map.mean():.3f}  "
          f"({'held-out' if N - nv >= 8 else 'in-sample'} alignment)")
    t12, t13 = "v1.2 (shared PCA)", "v1.3 -> v1.2 space (shared PCA)"

ncol = 4 if err_map is not None else 3
fig, ax = plt.subplots(nv, ncol, figsize=(ncol * args.cell, nv * args.cell), squeeze=False)
vmax = np.percentile(err_map, 99) if err_map is not None else None
for i in range(nv):
    panels = [rgb_view(s2[i]), c12[i], c13[i]]
    for j, im in enumerate(panels):
        ax[i, j].imshow(im, interpolation="nearest")
        ax[i, j].axis("off")
    if err_map is not None:
        ax[i, 3].imshow(err_map[i], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax[i, 3].axis("off")
titles = ["S2 RGB", t12, t13] + (["disagreement\n||v1.2 - f(v1.3)||^2 / var"] if err_map is not None else [])
for j, title in enumerate(titles):
    ax[0, j].set_title(title, fontsize=9)
plt.tight_layout(pad=0.3)
plt.savefig(args.out, dpi=100)
print("saved figure to", args.out)


# --------------------------------------------------------------------------------------
# how different are the two embeddings?
# --------------------------------------------------------------------------------------
ntr = max(1, int(0.75 * N))
print(f"\nlinear CKA(v1.2, v1.3) over all patches: {cka(flat(A), flat(B)):.3f}")
if N - ntr >= 1:
    for a, b, nm in ((A, B, "v1.2 -> v1.3"), (B, A, "v1.3 -> v1.2")):
        f = fit_ridge(flat(a[:ntr]), flat(b[:ntr]), args.ridge_lam)
        r2 = r2_per_channel(flat(b[ntr:]), f(flat(a[ntr:])))
        print(f"ridge {nm}: held-out R2 mean {r2.mean():.3f}, median {np.median(r2):.3f}  "
              f"(train {ntr} tiles, test {N - ntr} tiles)")
else:
    print("not enough tiles for a held-out ridge test; raise --n")
