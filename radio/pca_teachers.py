#!/usr/bin/env python3
"""
pca_teachers.py -- compare the dense patch embeddings of several teachers on the same tiles.

Teachers (--teachers, comma separated, in this order; the figure uses the first two):
    olmo_v1.2, olmo_v1.3     OlmoEarth-Base checkpoints (S2 + S1, one date)
    croma_base, croma_large  CROMA (S2 + S1, one date), joint radar-optical patch encodings
    smarties_b               SMARTIES ViT-B (S2 and/or S1), patch size 16 on 224x224 (14x14 grid)

For random tiles of the OlmoEarth pretraining set it picks one random date where S2 and S1 are
both present, runs each teacher, and saves a figure with one row per tile:

    [ S2 RGB ]  [ teacher A PCA ]  [ teacher B PCA ]  ( + [ disagreement map ] in --color-mode aligned )

Color modes (--color-mode):
  separate : one PCA per teacher (colors NOT comparable between the two teacher columns).
  joint    : ONE PCA on the pooled tokens of both teachers (needs equal feature dims). Same basis
             does not mean same meaning: channel i of A and channel i of B are unrelated.
  aligned  : (default) B's tokens are mapped into A's space with a ridge regression, then ONE PCA
             is fitted on [A, mapped B] and applied to both. Equal colors mean "B linearly predicts
             what A encodes here". The 4th column shows where it does not. Use --n >= --nvis + 8
             so the mapping is fitted on tiles that are not displayed.

It also prints, for every pair of teachers, linear CKA and held-out ridge R^2 in both directions.

Example (from the radio folder):
    python3 pca_teachers.py --teachers olmo_v1.2,croma_base --n 60 --nvis 25 --patch 8 \
        --index ../data/olmo_uncompressed/idx/all.csv --tar-dir ../data/olmo_uncompressed/tars \
        --ckpt ../checkpoints/olmo --croma-dir ../checkpoints/croma
Reuse saved features (same --teachers/--n/--seed/--patch):  add  --load-feats pca_teachers_feats.npz
"""
import argparse
import io
import itertools
import math
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
p.add_argument("--teachers", default="olmo_v1.2,croma_base")
p.add_argument("--n", type=int, default=25, help="number of tiles run through the teachers (>= --nvis)")
p.add_argument("--nvis", type=int, default=25, help="number of rows in the figure")
p.add_argument("--color-mode", choices=["separate", "joint", "aligned"], default="aligned")
p.add_argument("--ridge-lam", type=float, default=1e-3, help="relative ridge strength (alignment / metrics)")
p.add_argument("--patch", type=int, default=8, help="OlmoEarth patch size (grid = 128/patch). 8 matches CROMA's 16x16 grid")
p.add_argument("--olmo-dim", type=int, default=768, help="OlmoEarth encoder token dimension")
p.add_argument("--bs", type=int, default=4, help="batch size for the teachers")
p.add_argument("--index", default="data/olmo_uncompressed/idx/all.csv")
p.add_argument("--tar-dir", default="data/olmo_uncompressed/tars")
p.add_argument("--ckpt", default="checkpoints/olmo", help="folder containing the OlmoEarth-v1_x-Base folders")
p.add_argument("--croma-dir", default="checkpoints/croma",
               help="folder containing croma-base-patch8-120/ (BiliSakura/CROMA-transformers layout)")
p.add_argument("--croma-out", default="joint_hidden_states",
               choices=["joint_hidden_states", "optical_hidden_states", "sar_hidden_states"])
p.add_argument("--croma-norm", choices=["image", "set"], default="image",
               help="image: per-image, per-channel mean+-2std (the HF port's processor); "
                    "set: statistics over all loaded tiles (the original CROMA README)")
p.add_argument("--croma-8bit", action="store_true", help="quantize the normalized inputs to 8 bits (original README default)")
p.add_argument("--smarties-dir", default="checkpoints/smarties/SMARTIES-v1-ViT-B",
               help="local folder of gsumbul/SMARTIES-v1-ViT-B (config.json + smarties_model.py + weights)")
p.add_argument("--smarties-mod", choices=["s2", "s1", "both"], default="s2",
               help="which sensor(s) to feed SMARTIES; 'both' concatenates the per-sensor tokens (2x768 dims)")
p.add_argument("--no-figure", action="store_true", help="skip the PCA figure, only print the embedding metrics")
p.add_argument("--knn-k", type=int, default=10, help="neighbours for the mutual-kNN metric")
p.add_argument("--knn-queries", type=int, default=2000, help="query tokens for the mutual-kNN metric")
p.add_argument("--knn-pool", type=int, default=50000, help="max tokens in the neighbour pool for the mutual-kNN metric")
p.add_argument("--max-tokens", type=int, default=200000,
               help="random patch tokens used for CKA / ridge / raw cosine (keeps memory bounded)")
p.add_argument("--olmo-include-location", action="store_true",
               help="also average OlmoEarth's 'location_tokens' (v1.3 outputs them) into the embedding; off by default")
p.add_argument("--out", default="pca_teachers.png")
p.add_argument("--feats-out", default="pca_teachers_feats.npz", help="where to save the token grids (empty to skip)")
p.add_argument("--load-feats", default="", help="load token grids from this npz instead of running the teachers")
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

TEACHERS = {
    "olmo_v1.2": dict(kind="olmo", folder="OlmoEarth-v1_2-Base", mid=oep.ModelID.OLMOEARTH_V1_2_BASE, label="OlmoEarth v1.2"),
    "olmo_v1.3": dict(kind="olmo", folder="OlmoEarth-v1_3-Base", mid=oep.ModelID.OLMOEARTH_V1_3_BASE, label="OlmoEarth v1.3"),
    "croma_base": dict(kind="croma", folder="croma-base-patch8-120", label="CROMA-B"),
    "croma_large": dict(kind="croma", folder="croma-large-patch8-120", label="CROMA-L"),
    "smarties_b": dict(kind="smarties", label="SMARTIES-B"),
}
names = [t.strip() for t in args.teachers.split(",") if t.strip()]
assert len(names) >= 2 and all(n in TEACHERS for n in names), f"--teachers: pick >= 2 of {list(TEACHERS)}"
HP_OLMO = 128 // args.patch


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

s2 = np.stack([d["sentinel2_l2a"] for d in samples])  # (N, H, W, 1, 12), raw DN, OlmoEarth band order
s1 = np.stack([d["sentinel1"] for d in samples])      # (N, H, W, 1, 2)   (vv, vh)
ts = torch.from_numpy(np.stack([d["ts"] for d in samples]))  # (N, 1, 3)
print("timestamps of the first 4 tiles (raw triples):", ts[:4, 0].tolist())

# --- band bookkeeping (check the printed means: B08 should exceed B04 on vegetated land) ---
# --- teacher input helpers begin
OLMO_S2 = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]  # OlmoEarth order
CROMA_S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]  # assumed CROMA order
CROMA_IDX = [OLMO_S2.index(b) for b in CROMA_S2]


def croma_normalize(x, per_image=True, eight_bit=False):
    """Per-channel mean +- 2 std normalization of x: (N, C, H, W) to [0, 1].

    per_image=True : statistics of each image and channel (what the HF port's processor does, population std).
    per_image=False: statistics over all N images (what the original CROMA README does).
    eight_bit=True : quantize to uint8 and rescale to [0, 1] (original README default).
    """
    dims = (2, 3) if per_image else (0, 2, 3)
    m = x.mean(dims, keepdim=True)
    s = x.std(dims, keepdim=True, correction=0 if per_image else 1)
    lo, hi = m - 2 * s, m + 2 * s
    den = hi - lo
    den = torch.where(den == 0, torch.ones_like(den), den)
    y = (x - lo) / den
    if eight_bit:
        return torch.clip(y * 255.0, 0, 255).to(torch.uint8).float() / 255.0
    return torch.clip(y, 0, 1)


def alibi_2d(num_heads, num_patches):
    """Copy of get_2dalibi from the HF port's modeling_croma.py: (1, heads, L, L) bias for a g x g grid."""
    g = int(math.sqrt(num_patches))
    points = [(i, j) for i in range(g) for j in range(g)]

    def get_slopes(n):
        def pow2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            return [start * start ** i for i in range(n)]
        if math.log2(n).is_integer():
            return pow2(n)
        c = 2 ** math.floor(math.log2(n))
        return pow2(c) + get_slopes(2 * c)[0::2][: n - c]

    slopes = torch.tensor(get_slopes(num_heads), dtype=torch.float32).unsqueeze(1)
    idxs = [-math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * slopes for p1 in points for p2 in points]
    return torch.cat(idxs, dim=1).view(1, num_heads, num_patches, num_patches)

# SMARTIES names its projection layers by spectral range (config spectrum_specs). Map each OlmoEarth S2 band.
OLMO_TO_SMARTIES = {
    "B01": "aerosol", "B02": "blue_1", "B03": "green_2", "B04": "red_2", "B05": "red_edge_1",
    "B06": "red_edge_2", "B07": "near_infrared_2", "B08": "near_infrared_1", "B8A": "near_infrared_3",
    "B09": "short_wave_infrared_1", "B11": "short_wave_infrared_3", "B12": "short_wave_infrared_4",
}
SMARTIES_S2_BANDS = [OLMO_TO_SMARTIES[b] for b in OLMO_S2]   # in the channel order of our arrays
SMARTIES_S1_BANDS = ["microwave_1", "microwave_2"]            # VV, VH


def smarties_normalize(x):
    """x: numpy (N, C, H, W). Per band, over the whole set: clip to the 1st-99th percentile, min-max to [0, 1],
    then z-score with the set mean/std (the paper: percentile min-max, then standardization with dataset stats).
    Whether the authors clip is not stated; their per-dataset statistics live in the repo's config/*.yaml."""
    out = np.empty(x.shape, dtype=np.float32)
    for c in range(x.shape[1]):
        v = x[:, c].astype(np.float64)
        lo, hi = np.percentile(v, [1, 99])
        u = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        out[:, c] = (u - u.mean()) / max(u.std(), 1e-6)
    return out
# --- teacher input helpers end

print("raw S2 per-band mean (OlmoEarth order assumed):",
      dict(zip(OLMO_S2, s2.mean((0, 1, 2, 3)).round(0))))
print(f"mean B08/B04 ratio: {s2[..., 3].mean() / s2[..., 2].mean():.2f}  (index map h5 -> CROMA order: {CROMA_IDX})")

norm = Normalizer(std_multiplier=2.0)
s2n = norm.normalize(Modality.SENTINEL2_L2A, s2)
s1n = norm.normalize(Modality.SENTINEL1, s1)
for nm, a in (("S2", s2n), ("S1", s1n)):
    print(f"OlmoEarth-normalized {nm} percentiles 1/50/99:", np.percentile(a, [1, 50, 99]).round(2))


# --------------------------------------------------------------------------------------
# teachers
# --------------------------------------------------------------------------------------
def find_tokens(out):
    """All tensors shaped (B, HP, HP, ..., dim) anywhere in the OlmoEarth encoder output."""
    found = []

    def walk(o, path):
        if torch.is_tensor(o):
            if o.ndim >= 4 and o.shape[-1] == args.olmo_dim and o.shape[1] == HP_OLMO and o.shape[2] == HP_OLMO:
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


def run_olmo(cfg):
    try:
        model = oep.load_model_from_path(f"{args.ckpt}/{cfg['folder']}")
    except Exception as e:  # noqa: BLE001
        print(f"  load_model_from_path failed ({e!r}); falling back to load_model_from_id (needs internet)")
        model = oep.load_model_from_id(cfg["mid"], load_weights=True)
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
        if not args.olmo_include_location:
            found = [(pth, t) for pth, t in found if "location" not in pth]
        if not found:
            print(f"  no token tensor shaped (B,{HP_OLMO},{HP_OLMO},...,{args.olmo_dim}) in the output. Structure:")
            dump(out)
            raise SystemExit(1)
        if i == 0:
            print("  token tensors averaged into the embedding:", [(pth, tuple(t.shape)) for pth, t in found])
        red = [(t.float().flatten(3, -2).mean(3) if t.ndim > 4 else t.float()) for _, t in found]
        chunks.append(torch.stack(red).mean(0).cpu())
    return torch.cat(chunks).numpy()  # (N, HP, HP, D)


def run_croma(cfg):
    from transformers import AutoModel

    x2 = torch.from_numpy(s2[:, :, :, 0, :][..., CROMA_IDX]).permute(0, 3, 1, 2).float()  # (N, 12, H, W)
    x1 = torch.from_numpy(s1[:, :, :, 0, :]).permute(0, 3, 1, 2).float()                  # (N, 2, H, W) (vv, vh)
    per_image = args.croma_norm == "image"
    x2 = croma_normalize(x2, per_image, args.croma_8bit)
    x1 = croma_normalize(x1, per_image, args.croma_8bit)
    print(f"  CROMA inputs ({args.croma_norm}-normalized) percentiles 1/50/99  S2:",
          np.percentile(x2.numpy(), [1, 50, 99]).round(2), " S1:", np.percentile(x1.numpy(), [1, 50, 99]).round(2))

    path = f"{args.croma_dir}/{cfg['folder']}"
    model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()  # runs the folder's own .py files
    heads, n_tok = model.config.num_attention_heads, (128 // model.config.patch_size) ** 2

    # The config bakes in a 15x15 grid (120 px). The buffer is only ALiBi, so rebuild it for 16x16 (128 px).
    old = model.attn_bias
    mine = alibi_2d(heads, old.shape[-1])
    print(f"  my ALiBi copy matches the checkpoint's buffer for {old.shape[-1]} patches:",
          bool(torch.allclose(mine.to(old.device), old)))
    model.attn_bias = alibi_2d(heads, n_tok).to(old.device)

    chunks = []
    for i in range(0, N, args.bs):
        sl = slice(i, i + args.bs)
        out = model(sar_pixel_values=x1[sl].to(device), optical_pixel_values=x2[sl].to(device))
        if i == 0:
            print("  CROMA output tensors:", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
        enc = getattr(out, args.croma_out).float().cpu()  # (b, L, D), patches in row-major order
        assert enc.shape[1] == n_tok, f"expected {n_tok} tokens, got {enc.shape[1]}"
        assert torch.isfinite(enc).all(), "NaN/inf in CROMA output"
        g = int(round(n_tok ** 0.5))
        chunks.append(enc.reshape(enc.shape[0], g, g, enc.shape[2]))
    return torch.cat(chunks).numpy()


def run_smarties(cfg):
    from transformers import AutoModel

    from safetensors.torch import load_file
    from transformers import AutoConfig

    # AutoModel.from_pretrained crashes on transformers 5.x (the repo's class never calls post_init()), so build the
    # model from its config and load the weights ourselves, checking that the keys match.
    cfg_ = AutoConfig.from_pretrained(args.smarties_dir, trust_remote_code=True)
    model = AutoModel.from_config(cfg_, trust_remote_code=True)
    sd = load_file(f"{args.smarties_dir}/model.safetensors")
    best = None
    for prefix in ("", "model.", "module."):
        cand = {(k[len(prefix):] if prefix and k.startswith(prefix) else k): v for k, v in sd.items()}
        res = model.load_state_dict(cand, strict=False)
        if best is None or len(res.missing_keys) < len(best[1].missing_keys):
            best = (cand, res)
    cand, res = best
    model.load_state_dict(cand, strict=False)
    enc_missing = [k for k in res.missing_keys if not k.startswith(("decoder", "mask_token"))]
    print(f"  SMARTIES weights: {len(cand)} tensors in file, {len(res.missing_keys)} missing in model "
          f"({len(enc_missing)} encoder-side), {len(res.unexpected_keys)} unexpected")
    if enc_missing or res.unexpected_keys:
        print("   missing (encoder-side):", enc_missing[:8], " unexpected:", res.unexpected_keys[:8])
    assert not enc_missing, "encoder weights did not load; do not trust the features"
    model = model.to(device).eval()
    size = int(cfg_.img_size)  # 224: fixed sin-cos position embeddings and 16-px projection layers
    known = getattr(model, "projection_conversion", None)
    for names_ in (SMARTIES_S2_BANDS, SMARTIES_S1_BANDS):  # forward() silently drops unknown band names
        if known is not None:
            missing = [b for b in names_ if b not in known]
            assert not missing, f"band names not known to SMARTIES: {missing}"

    def prep(arr):  # (N, H, W, 1, C) raw -> (N, C, size, size) normalized, bicubic-resized like the paper
        x = smarties_normalize(np.transpose(arr[:, :, :, 0, :], (0, 3, 1, 2)))
        x = torch.from_numpy(x)
        return torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)

    def tokens(x, bands):
        chunks = []
        for i in range(0, N, args.bs):
            out = model(x[i:i + args.bs].to(device), is_patchify=True, bands=bands, all_tokens=True)  # (b, 1+L, D)
            if i == 0:
                print("  SMARTIES output (CLS + patch tokens):", tuple(out.shape))
            chunks.append(out[:, 1:].float().cpu())  # drop CLS; patches are row-major
        t = torch.cat(chunks)
        g = int(round(t.shape[1] ** 0.5))
        assert g * g == t.shape[1] and torch.isfinite(t).all()
        return t.reshape(N, g, g, t.shape[2])

    parts = []
    if args.smarties_mod in ("s2", "both"):
        x2 = prep(s2)
        print("  SMARTIES S2 input percentiles 1/50/99:", np.percentile(x2.numpy(), [1, 50, 99]).round(2))
        parts.append(tokens(x2, SMARTIES_S2_BANDS))
    if args.smarties_mod in ("s1", "both"):
        x1 = prep(s1)
        print("  SMARTIES S1 input percentiles 1/50/99:", np.percentile(x1.numpy(), [1, 50, 99]).round(2))
        parts.append(tokens(x1, SMARTIES_S1_BANDS))
    return torch.cat(parts, dim=-1).numpy()


feats = {}
if args.load_feats:
    z = np.load(args.load_feats)
    for nm in names:
        feats[nm] = z[nm].astype(np.float32)
        assert feats[nm].shape[0] == N, f"{nm}: saved {feats[nm].shape} vs N={N}; use the same --n/--seed"
    print("loaded features from", args.load_feats, {k: v.shape for k, v in feats.items()})
else:
    for nm in names:
        cfg = TEACHERS[nm]
        t0 = time.time()
        print(f"[{nm}] running ...")
        feats[nm] = {"olmo": run_olmo, "croma": run_croma, "smarties": run_smarties}[cfg["kind"]](cfg)
        print(f"[{nm}] features {feats[nm].shape}  ({time.time() - t0:.0f}s)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.feats_out:
        np.savez_compressed(args.feats_out, **{k: v.astype(np.float16) for k, v in feats.items()})
        print("saved token grids to", args.feats_out)


def to_grid(F, g):
    """Bilinearly resize a (n, h, w, D) token grid to (n, g, g, D)."""
    if F.shape[1] == g:
        return F
    t = torch.from_numpy(F).permute(0, 3, 1, 2)
    t = torch.nn.functional.interpolate(t, size=(g, g), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy()


G = max(F.shape[1] for F in feats.values())
if any(F.shape[1] != G for F in feats.values()):
    print(f"grids differ {({k: v.shape[1] for k, v in feats.items()})}; resizing all to {G}x{G} (bilinear)")
feats = {k: to_grid(v, G) for k, v in feats.items()}
HP = G


# --------------------------------------------------------------------------------------
# helpers: PCA colors, ridge alignment, similarity metrics
# --- helpers begin
# --------------------------------------------------------------------------------------
def flat(M):
    return M.reshape(-1, M.shape[-1]).astype(np.float64)


def shared_pca_rgb(Fs):
    """One PCA basis fitted on the pooled tokens of every array in Fs, applied to each.

    Fs: list of (n, hp, hp, D) with the same D. Returns (list of (n, hp, hp, 3) colors in [0, 1],
    explained variance of PC1-3). Same centering, basis and color scaling for all arrays.
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
def rsa_per_tile(Fa, Fb):
    """Alignment-free comparison of the two spaces. For each tile, the cosine-similarity matrix between its tokens is
    computed in each space (after centering the tile); returns the Pearson correlation of the two matrices' upper
    triangles, per tile. High = both models agree on which patches of a tile look alike."""
    out = []
    for a, b in zip(Fa, Fb):
        A = a.reshape(-1, a.shape[-1]).astype(np.float64)
        B = b.reshape(-1, b.shape[-1]).astype(np.float64)
        A = A - A.mean(0)
        B = B - B.mean(0)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
        iu = np.triu_indices(len(A), 1)
        out.append(np.corrcoef((A @ A.T)[iu], (B @ B.T)[iu])[0, 1])
    return np.array(out)


def mutual_knn(Fa, Fb, k=10, n_query=2000, pool=50000, seed=0):
    """Alignment-free. Pick random query tokens; find their k nearest neighbours (cosine, among tokens of OTHER tiles)
    in each space; return the mean fraction of neighbours shared by the two spaces, and the chance level."""
    rng = np.random.default_rng(seed)
    n, h, w = Fa.shape[:3]
    tid = np.repeat(np.arange(n), h * w)
    A = Fa.reshape(n * h * w, -1)
    B = Fb.reshape(n * h * w, -1)
    sel = rng.choice(len(A), size=min(pool, len(A)), replace=False)
    A, B, tid = A[sel].astype(np.float32), B[sel].astype(np.float32), tid[sel]
    A -= A.mean(0)
    B -= B.mean(0)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    q = rng.choice(len(A), size=min(n_query, len(A)), replace=False)
    ov = []
    for s in range(0, len(q), 256):
        qi = q[s:s + 256]
        same = tid[qi][:, None] == tid[None, :]
        nbrs = []
        for X in (A, B):
            sim = X[qi] @ X.T
            sim[same] = -np.inf
            nbrs.append(np.argpartition(-sim, k, axis=1)[:, :k])
        ov += [len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(*nbrs)]
    chance = k / max(1.0, len(A) * (1 - 1.0 / n))
    return float(np.mean(ov)), chance


def raw_cosine(Fa, Fb, seed=0):
    """Cosine between the SAME patch's vectors in the two models, with no alignment at all, and between random
    different patches (baseline). Only meaningful if the two models share a channel basis. Needs equal dims."""
    rng = np.random.default_rng(seed)
    A = Fa.reshape(-1, Fa.shape[-1]).astype(np.float64)
    B = Fb.reshape(-1, Fb.shape[-1]).astype(np.float64)
    nrm = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)  # noqa: E731
    same = (nrm(A) * nrm(B)).sum(1).mean()
    perm = rng.permutation(len(B))
    diff = (nrm(A) * nrm(B[perm])).sum(1).mean()
    return float(same), float(diff)

def sample_pairs(Fa, Fb, tiles, m, rng):
    """m random (token_a, token_b) pairs at the SAME tile/position, drawn from the given tiles. float32 (m, D)."""
    n, h, w = len(tiles), Fa.shape[1], Fa.shape[2]
    ti, yi, xi = rng.integers(n, size=m), rng.integers(h, size=m), rng.integers(w, size=m)
    t = np.asarray(tiles)[ti]
    return Fa[t, yi, xi].astype(np.float32), Fb[t, yi, xi].astype(np.float32)


def cka_chunked(X, Y, chunk=50000):
    """Linear CKA between paired token matrices (m, D1), (m, D2); float64 accumulation in chunks."""
    mx, my = X.mean(0, dtype=np.float64), Y.mean(0, dtype=np.float64)
    xtx = np.zeros((X.shape[1],) * 2); yty = np.zeros((Y.shape[1],) * 2); xty = np.zeros((X.shape[1], Y.shape[1]))
    for s in range(0, len(X), chunk):
        a = X[s:s + chunk].astype(np.float64) - mx
        b = Y[s:s + chunk].astype(np.float64) - my
        xtx += a.T @ a
        yty += b.T @ b
        xty += a.T @ b
    return float(np.linalg.norm(xty) ** 2 / (np.linalg.norm(xtx) * np.linalg.norm(yty)))


def ridge_r2_chunked(Xtr, Ytr, Xte, Yte, lam, chunk=50000):
    """Ridge X -> Y fitted on the train pairs, per-channel R^2 on the test pairs. float64 accumulation in chunks."""
    mu, sd = Xtr.mean(0, dtype=np.float64), Xtr.std(0, dtype=np.float64) + 1e-6
    ym = Ytr.mean(0, dtype=np.float64)
    xtx = np.zeros((Xtr.shape[1],) * 2); xty = np.zeros((Xtr.shape[1], Ytr.shape[1]))
    for s in range(0, len(Xtr), chunk):
        a = (Xtr[s:s + chunk].astype(np.float64) - mu) / sd
        xtx += a.T @ a
        xty += a.T @ (Ytr[s:s + chunk].astype(np.float64) - ym)
    W = np.linalg.solve(xtx + lam * len(Xtr) * np.eye(len(xtx)), xty)
    yme = Yte.mean(0, dtype=np.float64)
    sse = np.zeros(Yte.shape[1]); sst = np.zeros(Yte.shape[1])
    for s in range(0, len(Xte), chunk):
        a = (Xte[s:s + chunk].astype(np.float64) - mu) / sd
        y = Yte[s:s + chunk].astype(np.float64)
        sse += ((y - (a @ W + ym)) ** 2).sum(0)
        sst += ((y - yme) ** 2).sum(0)
    return 1 - sse / sst


def raw_cosine_pairs(X, Y, seed=0):
    """Cosine between paired token vectors (same patch) vs. between a token and a different random token."""
    rng = np.random.default_rng(seed)
    nrm = lambda Z: Z.astype(np.float32) / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)  # noqa: E731
    Xn, Yn = nrm(X), nrm(Y)
    return float((Xn * Yn).sum(1).mean()), float((Xn * Yn[rng.permutation(len(Yn))]).sum(1).mean())
# --- helpers end


def rgb_view(tile):
    """tile: (H, W, 1, 12) raw S2 DN in OlmoEarth band order -> displayable RGB (B04, B03, B02)."""
    x = tile[:, :, 0, [2, 1, 0]]
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


# --------------------------------------------------------------------------------------
# how different are the teachers? embedding-level metrics for every pair (no PCA involved)
# --------------------------------------------------------------------------------------
ntr = max(1, int(0.75 * N))
mrng = np.random.default_rng(args.seed + 1)
m_tr, m_te, m_all = int(args.max_tokens * 0.75), int(args.max_tokens * 0.25), args.max_tokens
print(f"\n(CKA / raw cosine use {m_all} random paired tokens from all {N} tiles; ridge is fitted on {m_tr} tokens from the first "
      f"{ntr} tiles and tested on {m_te} tokens from the other {N - ntr}; RSA and kNN use all tiles)")
for a_nm, b_nm in itertools.combinations(names, 2):
    Fa, Fb = feats[a_nm], feats[b_nm]
    print(f"\n=== {a_nm} vs {b_nm}   (grid {Fa.shape[1]}x{Fa.shape[2]}, dims {Fa.shape[-1]} / {Fb.shape[-1]}) ===")
    Xa, Xb = sample_pairs(Fa, Fb, np.arange(N), m_all, mrng)
    print(f"linear CKA                          : {cka_chunked(Xa, Xb):.3f}   (1 = identical up to rotation/scale)")
    if N - ntr >= 1:
        Ta, Tb = sample_pairs(Fa, Fb, np.arange(ntr), m_tr, mrng)
        Ea, Eb = sample_pairs(Fa, Fb, np.arange(ntr, N), m_te, mrng)
        for x_nm, (Xt, Xe), y_nm, (Yt, Ye) in ((a_nm, (Ta, Ea), b_nm, (Tb, Eb)), (b_nm, (Tb, Eb), a_nm, (Ta, Ea))):
            r2 = ridge_r2_chunked(Xt, Yt, Xe, Ye, args.ridge_lam)
            print(f"ridge {x_nm} -> {y_nm:<10}: held-out R2 mean {r2.mean():.3f}, median {np.median(r2):.3f}")
    r = rsa_per_tile(Fa, Fb)
    print(f"within-tile similarity structure    : Pearson r mean {r.mean():.3f}  (min {r.min():.3f}, max {r.max():.3f}) over {N} tiles")
    ov, chance = mutual_knn(Fa, Fb, args.knn_k, args.knn_queries, args.knn_pool, args.seed)
    print(f"mutual {args.knn_k}-NN overlap across tiles      : {ov:.3f}   (chance ~ {chance:.5f})")
    if Fa.shape[-1] == Fb.shape[-1]:
        same, diff = raw_cosine_pairs(Xa, Xb, args.seed)
        print(f"raw cosine, same patch / random patch: {same:.3f} / {diff:.3f}   (only meaningful if channels correspond)")
    del Xa, Xb

if args.no_figure:
    raise SystemExit(0)

# --------------------------------------------------------------------------------------
# figure (first two teachers)
# --------------------------------------------------------------------------------------
nA, nB = names[0], names[1]
lA, lB = TEACHERS[nA]["label"], TEACHERS[nB]["label"]
A, B = feats[nA], feats[nB]
nv = args.nvis
mode = args.color_mode
if mode == "joint" and A.shape[-1] != B.shape[-1]:
    print(f"joint PCA needs equal dims ({A.shape[-1]} vs {B.shape[-1]}); using 'separate'")
    mode = "separate"
err_map = None

if mode == "separate":
    (cA,), eA = shared_pca_rgb([A[:nv]])
    (cB,), eB = shared_pca_rgb([B[:nv]])
    print(f"variance explained by PC1-3  {lA}: {eA.round(3)}   {lB}: {eB.round(3)}")
    tA, tB = f"{lA} (own PCA)", f"{lB} (own PCA)"
elif mode == "joint":
    (cA, cB), ev = shared_pca_rgb([A[:nv], B[:nv]])
    print("joint PCA variance explained by PC1-3:", ev.round(3))
    tA, tB = f"{lA} (joint PCA)", f"{lB} (joint PCA)"
else:  # aligned: map B -> A's space
    held_out = N - nv >= 8
    fit_idx = np.arange(nv, N) if held_out else np.arange(N)
    if not held_out:
        print("WARNING: fewer than 8 tiles outside the figure, so the alignment is fitted on the displayed "
              "tiles (in-sample, optimistic). Use --n >= --nvis + 8.")
    _xa, _xb = sample_pairs(A, B, fit_idx, min(args.max_tokens, len(fit_idx) * A.shape[1] * A.shape[2]), np.random.default_rng(args.seed + 2))
    f_b2a = fit_ridge(_xb.astype(np.float64), _xa.astype(np.float64), args.ridge_lam)
    A_disp = A[:nv].astype(np.float64)
    B_in_A = f_b2a(flat(B[:nv])).reshape(A_disp.shape)
    (cA, cB), ev = shared_pca_rgb([A_disp, B_in_A])
    print("aligned PCA variance explained by PC1-3:", ev.round(3))
    resid = ((flat(A_disp) - flat(B_in_A)) ** 2).sum(1)
    denom = ((flat(A_disp) - flat(A_disp).mean(0)) ** 2).sum(1).mean()
    err_map = (resid / denom).reshape(nv, HP, HP)
    print(f"mean normalized disagreement (1 - R2) on the displayed tiles: {err_map.mean():.3f}  "
          f"({'held-out' if held_out else 'in-sample'} alignment {lB} -> {lA})")
    tA, tB = f"{lA} (shared PCA)", f"{lB} -> {lA} space (shared PCA)"

ncol = 4 if err_map is not None else 3
fig, ax = plt.subplots(nv, ncol, figsize=(ncol * args.cell, nv * args.cell), squeeze=False)
vmax = np.percentile(err_map, 99) if err_map is not None else None
for i in range(nv):
    for j, im in enumerate([rgb_view(s2[i]), cA[i], cB[i]]):
        ax[i, j].imshow(im, interpolation="nearest")
        ax[i, j].axis("off")
    if err_map is not None:
        ax[i, 3].imshow(err_map[i], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax[i, 3].axis("off")
titles = ["S2 RGB", tA, tB] + ([f"disagreement\n||A - f(B)||^2 / var"] if err_map is not None else [])
for j, title in enumerate(titles):
    ax[0, j].set_title(title, fontsize=8)
plt.tight_layout(pad=0.3)
plt.savefig(args.out, dpi=100)
print("saved figure to", args.out)
