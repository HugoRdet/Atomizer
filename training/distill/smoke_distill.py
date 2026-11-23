"""
smoke_distill.py -- end-to-end smoke test of the distillation student on SYNTHETIC atoms.

Where it goes: training/distill/smoke_distill.py. Run from the repo root (needs a GPU):
    PYTHONPATH=. python training/distill/smoke_distill.py --config training/configs/config_test-BURNSCARS-legacy-shared-enc-scale.yaml

What it checks, with the real model code, config and lookup table:
  1. atoms for TWO layouts built with TokenBuilder (Sentinel-like 14 x 128 x 128 @ 10 m, aerial-like 4 x HW x HW @ 0.2 m);
  2. the pruning geometry is built once per layout and cached (alternating layouts must not rebuild it);
  3. encode -> teacher heads at the teacher's token centres -> DUNE loss (0.5 cos + 0.5 smooth-L1) -> backward ->
     per-parameter clipping -> AdamW step, in bf16;
  4. memory and time per step for each layout, atoms per leaf cell, number of latents per level;
  5. an overfit run on one fixed batch per layout with content-dependent targets (avg-pooled pixels through a fixed random
     map): the loss must go down. Random atoms and teachers: this tests the wiring, not the data.
Spectral indices are arbitrary here (the real datasets take them from look_up.table_wave).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# make `import training...` work wherever the script is launched from (repo root = 3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.utils import Lookup_encoding, read_yaml
from training.utils.datasets.token_builder import TokenBuilder
from training.distill.distill_heads import DistillStudent, token_centers_from_tokens
from training.distill.distill_losses import PatchStandardizer, distill_loss_per_image
from training.distill.distill_optim import clip_per_parameter, make_optimizer, set_lr_wd

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--configs-dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
p.add_argument("--bands-yaml", default="./data/bands_info/bands.yaml")
p.add_argument("--bs", type=int, default=2)
p.add_argument("--rounds", type=int, default=3, help="alternate the two layouts this many times (cache check)")
p.add_argument("--overfit-steps", type=int, default=40)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--print-every", type=int, default=0, help="overfit log interval (default: 10 for <=100 steps, else 50)")
p.add_argument("--m-factor", type=float, default=1.3, help="max atoms read per leaf = m_factor * k")
p.add_argument("--k-olmo", type=int, default=224, help="atoms per leaf, Sentinel-like layout (14 atoms/pixel)")
p.add_argument("--k-flair", type=int, default=1024, help="atoms per leaf, aerial-like layout (4 atoms/pixel)")
p.add_argument("--flair-hw", type=int, default=256)
p.add_argument("--olmo-patch", type=int, default=4)
p.add_argument("--flair-patch", type=int, default=16)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "the real model needs a GPU (1M-atom layouts)"
torch.manual_seed(0)

config = read_yaml(args.config)
lookup = Lookup_encoding(read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config)
tb = TokenBuilder(lookup)
print("atom features:", config["Atomiser"].get("unet_atom_features", "atomizer"),
      "| km_train:", config["Atomiser"].get("unet_km_train"), "| decoder:", config["Atomiser"].get("unet_decoder"))

SPEC = {
    "olmo": dict(C=14, HW=128, res=10.0, k=args.k_olmo, patch=args.olmo_patch, D=768, teacher="olmo"),
    "flair": dict(C=4, HW=args.flair_hw, res=0.2, k=args.k_flair, patch=args.flair_patch, D=1024, teacher="dino"),
}
student = DistillStudent(config, lookup, {"olmo": 768, "dino": 1024}).to(device)
tnorm = torch.nn.ModuleDict({"olmo": PatchStandardizer(768), "dino": PatchStandardizer(1024)}).to(device)
params = [q for q in student.parameters() if q.requires_grad]
opt = make_optimizer(student, fused=True)
student.train()
tnorm.train()


def to_dev(x):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v) for k, v in x.items()}
    return x


def make_data(name):
    s = SPEC[name]
    C, HW, res, patch, D = s["C"], s["HW"], s["res"], s["patch"], s["D"]
    imgs = torch.randn(args.bs, C, HW, HW).clamp(-3, 3)
    label = torch.full((HW, HW), 255, dtype=torch.long)
    sidx = torch.arange(C).long()
    ridx = lookup.get_resolution_idx(res)
    toks = torch.stack([tb.build_tokens(im, label, res, sidx, ridx, -1) for im in imgs])         # [B, C*H*W, 8]
    batch = {"groups": {res: {"tokens": toks, "mask": torch.zeros(toks.shape[:2], dtype=torch.bool), "shape": (C, HW, HW)}}}
    q = token_centers_from_tokens(toks, HW, HW, patch)                                              # [B, g*g, 2]
    g = HW // patch
    pooled = F.avg_pool2d(imgs, patch).permute(0, 2, 3, 1).reshape(args.bs, g * g, C)               # content per token
    W = torch.randn(C, D) / C ** 0.5
    target = torch.tanh(pooled @ W * 2.0)                                                            # [B, g*g, D]
    return to_dev(batch), to_dev(q), to_dev(target), g


DATA = {n: make_data(n) for n in SPEC}
print({n: (tuple(v[0]["groups"][SPEC[n]["res"]]["tokens"].shape), f"{v[3]}x{v[3]} teacher tokens") for n, v in DATA.items()})


def step(name, lr, momentum=1.0):
    s = SPEC[name]
    batch, q, target, g = DATA[name]
    m = int(args.m_factor * s["k"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d0, geom = student.encode(batch, s["k"], m)
        pred = student.predict(s["teacher"], d0, geom, q)
    tgt = tnorm[s["teacher"]](target, momentum)
    loss_i, mets = distill_loss_per_image(pred, tgt)
    loss = loss_i.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norms = clip_per_parameter(params, 1.0)
    set_lr_wd(opt, lr, 0.03)
    opt.step()
    return float(loss.detach()), mets, geom, norms


# ---- 1) alternate the layouts: geometry built once per layout, memory/time per step ----
print("\n== alternating layouts ==")
for r in range(args.rounds):
    for name in SPEC:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        loss, mets, geom, norms = step(name, lr=1e-5)
        torch.cuda.synchronize()
        print(f"round {r} {name:6s} loss {loss:.4f} cos {float(mets['cos']):.3f} fid {float(mets['fidelity']):+.3f} | "
              f"{time.time() - t0:5.2f}s | peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB | "
              f"latents/level {geom.sizes} | leaf atoms/cell mean {float(geom.leaf_counts.float().mean()):.0f} "
              f"(min {int(geom.leaf_counts.min())}, max {int(geom.leaf_counts.max())}) | grad-norm max {float(norms.max()):.2f}")
n_cached = len(student.net.pruning._cache)
print(f"geometries cached: {n_cached} (expected {len(SPEC)})")
assert n_cached == len(SPEC), "the pruning cache did not hold both layouts"

# ---- 2) overfit one fixed batch per layout ----
print("\n== overfit (fixed batches, content-dependent targets) ==")
pe = args.print_every or (10 if args.overfit_steps <= 100 else 50)
t_start = time.time()
first, first_fid, last, last_fid = {}, {}, {}, {}
for i in range(args.overfit_steps):
    for name in SPEC:
        loss, mets, _, _ = step(name, lr=args.lr, momentum=0.0 if i > 0 else 1.0)
        fid = float(mets["fidelity"])
        first.setdefault(name, loss)
        first_fid.setdefault(name, fid)
        last[name], last_fid[name] = loss, fid
        if i % pe == 0 or i == args.overfit_steps - 1:
            print(f"  step {i:4d} {name:6s} loss {loss:.4f}  cos {float(mets['cos']):.3f}  fidelity {fid:+.3f}  ({time.time() - t_start:.0f}s)")
for n in SPEC:
    ok = last_fid[n] > 0.3 or last[n] < 0.5 * first[n]
    print(f"{n}: loss {first[n]:.4f} -> {last[n]:.4f} | fidelity {first_fid[n]:+.3f} -> {last_fid[n]:+.3f}  "
          f"({'OK, fitting' if ok else 'NOT fitting yet: check the wiring or raise --lr'})")
print("smoke test finished")
