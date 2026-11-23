"""
viz_smarties.py -- SMARTIES (spectrum-aware multi-sensor MAE, ICCV 2025) as a candidate teacher on Sentinel tiles,
next to OlmoEarth (current teacher) and Panopticon.

Where it goes: scripts_atomizer/viz_smarties.py  (reuses helpers of viz_panopticon.py / viz_dino_upsample.py)

Figure (one row per OlmoEarth validation tile, one PCA basis PER COLUMN by default):
    input RGB | OlmoEarth | Panopticon all bands @--pano_size | SMARTIES all bands | SMARTIES RGB only
    [| AnyUp(SMARTIES all) -> --anyup_size]
Scores (panopticon-style, saved to smarties_scores.txt): CKA of each map with OlmoEarth (common grid),
cos / CKA(SMARTIES all bands, RGB only) (1 = it ignores the non-RGB bands), AnyUp detail.

SMARTIES input: the student's own per-band values (dataset z-score) resized to the model's img_size; the bands are
passed by NAME (`bands=` of SMARTIESHF.forward), matched to the model's spectrum_specs (printed at start). Sentinel-1
is left out unless --with_s1 (only if the spec has VV / VH). Output: all tokens, CLS dropped.

Setup (login node, once; the HF model uses trust_remote_code):
    pip install -U transformers                     # if AutoModel fails on the remote code
    huggingface-cli download gsumbul/SMARTIES-v1-ViT-B --local-dir ./checkpoints/smarties/SMARTIES-v1-ViT-B
Run (GPU node):
    python scripts_atomizer/viz_smarties.py --config training/configs/config_distill_ptv3.yaml \
        --smarties ./checkpoints/smarties/SMARTIES-v1-ViT-B --n 6 --tiles 32 --out ./viz_pca/smarties
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from training.distill.teachers import OlmoEarthTeacher
from training.distill.trainer_distill import DistillDataModule
from training.utils import Lookup_encoding, read_yaml
from training.utils.datasets.utils_dataset_distill import OLMO_S1, OLMO_S2, collate_distill
from viz_dino_upsample import load_anyup, run_anyup
from viz_panopticon import S2_NM, band_values, cka, detail, load_panopticon, render, rgb_uint8, run_panopticon


def _norm_name(n):
    """'B02' / 'B2' / 'b02' / 'S2_B02' -> 'B2'; 'B8A' -> 'B8A'; VV / VH unchanged."""
    n = str(n).upper().split("_")[-1]
    m = re.fullmatch(r"B0*(\d+)(A?)", n)
    return f"B{m.group(1)}{m.group(2)}" if m else n


def _build_smarties(path):
    """SMARTIESHF built from its own remote class + weights loaded by hand. AutoModel.from_pretrained fails with recent
    transformers ('SMARTIESHF' object has no attribute 'all_tied_weights_keys': the remote code predates that API)."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    ref = (getattr(cfg, "auto_map", None) or {}).get("AutoModel")
    if ref is None:
        raise RuntimeError(f"{path}: config has no auto_map['AutoModel'] (cannot find the SMARTIES class)")
    cls = get_class_from_dynamic_module(ref, path)
    model = cls(cfg)
    st = os.path.join(path, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[smarties] built {ref} from {path}: {len(sd)} tensors, missing {len(missing)} {missing[:5]}, "
          f"unexpected {len(unexpected)} {unexpected[:5]}")
    if len(missing) > 0.05 * max(1, len(list(model.state_dict()))):
        raise RuntimeError("too many missing SMARTIES weights: check the checkpoint folder")
    return model


def load_smarties(path, dev):
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
    except AttributeError as e:                       # recent transformers vs the model's remote code
        print(f"[smarties] AutoModel.from_pretrained failed ({e}); building the model by hand")
        model = _build_smarties(path)
    model = model.to(dev).eval()
    cfg = model.config
    specs = getattr(cfg, "spectrum_specs", None) or getattr(model, "spectrum_specs", None) or {}
    img = int(getattr(cfg, "img_size", 224))
    patch = int(getattr(cfg, "patch_size", getattr(model, "patch_size", 16)))
    print(f"[smarties] {path}: img_size {img}, patch {patch} | spectrum specs: {list(specs)}")
    groups = getattr(cfg, "sensor_band_specs", None) or getattr(model, "sensor_band_specs", None)
    if groups:
        print(f"[smarties] sensor band groups: {groups}")
    return model, img, patch, list(specs)


S2_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]   # Sentinel-2 without B10


def s2_table(model):
    """SMARTIES names its bands by spectral region (blue_1, red_2, ...). Its sensor group 'S2' lists 12 of them: Sentinel-2
    in the standard order without B10 (cirrus). Map our band names to them by position."""
    groups = getattr(model.config, "sensor_band_specs", None) or getattr(model, "sensor_band_specs", None) or {}
    s2 = list(groups.get("S2", []))
    if len(s2) != len(S2_ORDER):
        print(f"[smarties] S2 group has {len(s2)} bands, expected 12: positional mapping skipped")
        return {}
    table = dict(zip(S2_ORDER, s2))
    s1 = list(groups.get("S1", []))
    if len(s1) == 2:
        table.update({"VV": s1[0], "VH": s1[1]})
    print("[smarties] band mapping: " + ", ".join(f"{k}->{v}" for k, v in table.items()))
    return table


def match_bands(ours, spec_names, s2map=None):
    """Our channel names -> SMARTIES spec names (same order); channels without a match are left out."""
    table = {_norm_name(s): s for s in spec_names}
    table.update(s2map or {})
    keep, names = [], []
    for c, n in enumerate(ours):
        s = table.get(_norm_name(n))
        if s is not None:
            keep.append(c)
            names.append(s)
    missing = [n for n in ours if _norm_name(n) not in table]
    if missing:
        print(f"[smarties] no spectrum spec for {missing}: left out")
    return keep, names


@torch.no_grad()
def run_smarties(model, x, names, img, patch, dev):
    """x [B, C, H, W] z-scored -> [B, g, g, D], g = img / patch."""
    xi = F.interpolate(x.to(dev), size=(img, img), mode="bilinear", align_corners=False)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    with amp:
        out = model(xi, is_patchify=True, bands=list(names), all_tokens=True)
    tok = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    if isinstance(tok, (list, tuple)):
        tok = tok[0]
    g = img // patch
    if tok.shape[1] == g * g + 1:
        tok = tok[:, 1:]
    assert tok.shape[1] == g * g, f"SMARTIES returned {tok.shape[1]} tokens, expected {g}x{g} (+ CLS)"
    return tok.float().reshape(x.shape[0], g, g, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smarties", default="./checkpoints/smarties/SMARTIES-v1-ViT-B", help="local HF folder or hub id")
    ap.add_argument("--with_s1", action="store_true", help="also give VV, VH (if the spectrum specs have them)")
    ap.add_argument("--pano_size", type=int, default=224, help="Panopticon input side (0: no Panopticon column)")
    ap.add_argument("--pano_dir", default="./checkpoints/panopticon")
    ap.add_argument("--anyup_size", type=int, default=0, help="AnyUp(SMARTIES) output side (0: none)")
    ap.add_argument("--anyup_dir", default="./checkpoints/anyup")
    ap.add_argument("--q_chunk", type=int, default=256)
    ap.add_argument("--pca", default="column", choices=["column", "model"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./viz_pca/smarties")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    config = read_yaml(args.config)
    dc = config["distill"]
    dc.pop("band_drop", None)
    bands = read_yaml(args.bands_yaml)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), bands, config)
    dm = DistillDataModule(config, lookup, bands)
    dm.setup()
    ds = dm.olmo_val

    sm, img, patch, spec_names = load_smarties(args.smarties, dev)
    ours = OLMO_S2 + (OLMO_S1 if args.with_s1 else [])
    s2map = s2_table(sm)
    keep_all, names_all = match_bands(ours, spec_names, s2map)
    rgb_idx = [OLMO_S2.index("B04"), OLMO_S2.index("B03"), OLMO_S2.index("B02")]
    keep_rgb, names_rgb = match_bands([ours[c] for c in rgb_idx], spec_names, s2map)
    keep_rgb = [rgb_idx[k] for k in keep_rgb]
    if len(keep_rgb) != 3:
        raise SystemExit(f"[smarties] RGB bands not all found in the spectrum specs {spec_names}")
    print(f"[smarties] all bands: {names_all} | RGB: {names_rgb}")
    teacher = OlmoEarthTeacher(dc["olmo_ckpt"], int(dc["olmo_patch"])).to(dev).eval()
    pano = load_panopticon(os.path.abspath(args.pano_dir)).to(dev).eval() if args.pano_size else None
    up = load_anyup("wimmerth/anyup", "anyup", args.anyup_dir).to(dev) if args.anyup_size else None

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(max(args.tiles, args.n), len(ds)), replace=False).tolist()
    keep, acc, imgs = {}, {}, []
    for s0 in range(0, len(idx), args.batch):
        batch = collate_distill([ds[i] for i in idx[s0:s0 + args.batch]])
        tin = {k: v.to(dev) for k, v in batch["teacher"].items()}
        x = band_values(batch)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
        with torch.no_grad(), amp:
            ref = teacher(tin).float()
        maps = {"OlmoEarth": ref}
        if pano is not None:
            maps[f"Panopticon @{args.pano_size}"] = run_panopticon(
                pano, x[:, :12], [S2_NM[b] for b in OLMO_S2], args.pano_size, dev)
        maps["SMARTIES all"] = run_smarties(sm, x[:, keep_all], names_all, img, patch, dev)
        maps["SMARTIES RGB"] = run_smarties(sm, x[:, keep_rgb], names_rgb, img, patch, dev)
        guide = rgb_uint8(x, rgb_idx).to(dev)
        if up is not None:
            maps["AnyUp(SMARTIES)"] = run_anyup(up, guide, maps["SMARTIES all"], args.anyup_size, args.q_chunk)
        for k, v in maps.items():
            if k != "OlmoEarth":
                acc.setdefault(f"CKA({k}, OlmoEarth)", []).append(cka(v, ref))
        a, r = maps["SMARTIES all"], maps["SMARTIES RGB"]
        acc.setdefault("cos(SMARTIES all, RGB)", []).append(float(F.cosine_similarity(a, r, dim=-1).mean()))
        acc.setdefault("CKA(SMARTIES all, RGB)", []).append(cka(a, r))
        if up is not None:
            acc.setdefault("detail AnyUp(SMARTIES)", []).append(detail(maps["AnyUp(SMARTIES)"], a.shape[1]))
        need = args.n - sum(t.shape[0] for t in imgs)
        if need > 0:
            imgs.append(guide[:need].cpu())
            for k, v in maps.items():
                keep.setdefault(k, []).append(v[:need].cpu())
        print(f"[smarties] {min(s0 + args.batch, len(idx))}/{len(idx)} tiles")

    keep = {k: torch.cat(v) for k, v in keep.items()}
    if args.pca == "column":
        cols = {k: (v, k) for k, v in keep.items()}
    else:
        cols = {k: (v, "smarties" if ("SMARTIES" in k) else k) for k, v in keep.items()}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "smarties_olmo.png")
    render(torch.cat(imgs), cols, path, f"Sentinel: OlmoEarth vs Panopticon vs SMARTIES (PCA per {args.pca})")
    lines = [f"== Sentinel ({len(idx)} val tiles) | SMARTIES {args.smarties} img {img} patch {patch} "
             f"bands {names_all} -> {path}"] + [f"  {k:40s} {np.mean(v):.3f}" for k, v in acc.items()]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "smarties_scores.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
