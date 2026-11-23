"""
check_distill_data.py -- sanity checks for the two distillation datasets (and optionally the teachers), before any training.

Where it goes: training/distill/check_distill_data.py. From the repo root:
    PYTHONPATH=. python training/distill/check_distill_data.py --config training/configs/config_distill.yaml --compute-stats
    PYTHONPATH=. python training/distill/check_distill_data.py --config training/configs/config_distill.yaml --teachers

--compute-stats  computes the per-band z-score statistics (500 random train samples per dataset) and saves them. Run it ONCE,
                 before the DDP training (the datasets refuse to start without them).
--teachers       also loads OlmoEarth and DINOv3 (GPU) and runs them on a 2-sample batch of each dataset: checks that the teacher
                 token grid matches the student's query grid (token centres) and prints the target statistics.
"""
import argparse
import os
import sys
import time

import torch

# make `import training...` work wherever the script is launched from (repo root = 3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.distill.distill_heads import token_centers_from_tokens
from training.utils import Lookup_encoding, read_yaml
from training.utils.datasets.utils_dataset_distill import FlairAtoms, OlmoSentinelAtoms, collate_distill

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
p.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
p.add_argument("--compute-stats", action="store_true")
p.add_argument("--teachers", action="store_true")
p.add_argument("--value-stats", action="store_true", help="per-band percentiles of the z-scored values and the fraction beyond the +-10 clamp (200 random augmented samples per dataset)")
p.add_argument("--n-stats", type=int, default=500)
args = p.parse_args()

config = read_yaml(args.config)
dc = config["distill"]
bands = read_yaml(args.bands_yaml)
lookup = Lookup_encoding(read_yaml(args.configs_dataset), bands, config)
common = dict(look_up=lookup, dataset_config=bands, stats_required=False)
olmo = OlmoSentinelAtoms(dc["olmo_index"], dc["olmo_tar_dir"], "train", aug=True, stats_path=dc["olmo_stats"], **common)
flair = FlairAtoms(dc["flair_index"], dc["flair_root"], "train", int(dc["flair_crop"]), aug=True, stats_path=dc["flair_stats"],
                   aerial_section=dc.get("aerial_section", "bands_flairhub_aerial"), share_s2=bool(dc.get("share_s2", False)), **common)

if args.compute_stats:
    for ds in (olmo, flair):
        t0 = time.time()
        ds.compute_and_save_stats(n=args.n_stats)
        print(f"  ({time.time() - t0:.0f}s)")

if args.value_stats:
    import numpy as np
    from training.utils.datasets.utils_dataset_distill import OLMO_S1, OLMO_S2
    names = {"olmo": OLMO_S2 + OLMO_S1, "flair": ["R", "G", "B", "NIR"]}
    for name, ds in (("olmo", olmo), ("flair", flair)):
        idx = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)
        z = torch.stack([(ds.raw_image(int(i), aug=True) - ds.mean) / ds.std for i in idx])      # [n, C, H, W], before the clamp
        flat = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1)
        q = torch.quantile(flat[:, ::37].float(), torch.tensor([0.01, 0.5, 0.99]), dim=1)
        clip = (flat.abs() > 10).float().mean(1)
        print(f"\n== z-scored values, {name} ({len(idx)} samples, before the +-10 clamp) ==")
        for c, b in enumerate(names[name]):
            flag = "   <-- clipped" if clip[c] > 0.005 else ""
            print(f"  {b:5s} p1 {q[0, c]:+6.2f}  median {q[1, c]:+6.2f}  p99 {q[2, c]:+6.2f}  fraction |z|>10: {100 * clip[c]:.2f}%{flag}")

specs = {"olmo": dict(ds=olmo, patch=int(dc["olmo_patch"]), hw=128), "flair": dict(ds=flair, patch=int(dc.get("flair_patch", 16)), hw=int(dc["flair_crop"]))}
batches = {}
for name, s in specs.items():
    ds = s["ds"]
    t0 = time.time()
    samples = [ds[i] for i in range(4)]
    dt = (time.time() - t0) / 4
    b = collate_distill(samples[:2])
    batches[name] = b
    res = next(iter(b["groups"]))
    tok = b["groups"][res]["tokens"]
    same = torch.equal(tok[0, :, 1:3], tok[1, :, 1:3])
    print(f"\n== {name} ==  {dt:.2f} s/sample | tokens {tuple(tok.shape)} | positions identical across samples: {same}")
    assert same, "token positions differ between samples: the shared hierarchy geometry would be wrong"
    print("  value column: mean %.3f std %.3f min %.2f max %.2f (z-scored; ~0/1 once stats exist)" % (
        tok[..., 0].mean(), tok[..., 0].std(), tok[..., 0].min(), tok[..., 0].max()))
    print("  columns: x[%.0f..%.0f] y[%.0f..%.0f] spectral %s resolution_idx %s time %s" % (
        tok[0, :, 1].min(), tok[0, :, 1].max(), tok[0, :, 2].min(), tok[0, :, 2].max(),
        sorted(set(tok[0, :, 3].long().tolist())), sorted(set(tok[0, :, 6].long().tolist())), sorted(set(tok[0, :, 7].long().tolist()))))
    q = token_centers_from_tokens(tok, s["hw"], s["hw"], s["patch"])
    g = s["hw"] // s["patch"]
    print(f"  teacher token centres {tuple(q.shape)} (grid {g}x{g}); teacher inputs: "
          f"{ {k: (tuple(v.shape), str(v.dtype)) for k, v in b['teacher'].items()} }")

if args.teachers:
    from training.distill.teachers import DinoV3Teacher, OlmoEarthTeacher
    dev = "cuda"
    for name, make in (("olmo", lambda: OlmoEarthTeacher(dc["olmo_ckpt"], int(dc["olmo_patch"]))),
                       ("flair", lambda: DinoV3Teacher(dc["dino_ckpt"], dc.get("dino_norm", "auto"), student_patch=int(dc.get("flair_patch", 16))))):
        t = make().to(dev).eval()
        inp = {k: v.to(dev) for k, v in batches[name]["teacher"].items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = t(inp)
        g = specs[name]["hw"] // specs[name]["patch"]
        print(f"\n[{name}] teacher output {tuple(out.shape)}  (expected [2, {g}, {g}, D]) | mean {out.float().mean():.3f} std {out.float().std():.3f}")
        assert out.shape[1] == g and out.shape[2] == g, "teacher grid != student query grid"

        # --- orientation check: does teacher token (row a, col b) sit where the student's query (row a, col b) is? ---
        # Correlate the top-3 principal-component maps of the teacher features with the pooled pixel brightness at the same
        # resolution, in 5 arrangements. With the right convention 'identity' should score highest (heuristic: needs spatial structure).
        def pooled(img):                                  # [H, W] -> [g, g]
            p = img.shape[-1] // g
            return img.reshape(g, p, g, p).mean((1, 3))

        def corr(a, b):
            a, b = a - a.mean(), b - b.mean()
            return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))

        arr = {"identity": lambda m: m, "transpose": lambda m: m.T, "flip_rows": lambda m: m.flip(0),
               "flip_cols": lambda m: m.flip(1), "rot180": lambda m: m.flip(0).flip(1)}
        tin = batches[name]["teacher"]
        img = (tin["s2n"][..., 0, 3] if name == "olmo" else tin["rgb"].float().mean(1)).float().cpu()   # NIR | mean RGB, [B, H, W]
        score = {k: 0.0 for k in arr}
        for b_ in range(out.shape[0]):
            X = out[b_].float().reshape(g * g, -1)
            X = X - X.mean(0)
            U, S, _ = torch.linalg.svd(X, full_matrices=False)
            pcs = (U[:, :3] * S[:3]).reshape(g, g, 3).cpu()
            m = pooled(img[b_])
            for k, f in arr.items():
                score[k] += max(abs(corr(pcs[..., j].flatten(), f(m).flatten())) for j in range(3)) / out.shape[0]
        best = max(score, key=score.get)
        print(f"  orientation check |corr| " + "  ".join(f"{k} {v:.2f}" for k, v in score.items())
              + f"  -> best: {best}" + ("" if best == "identity" else "   <-- WARNING: not 'identity'; check the grid convention")
              + ("   (weak structure, inconclusive)" if score[best] < 0.3 else ""))
        del t
        torch.cuda.empty_cache()
print("\ncheck finished")
