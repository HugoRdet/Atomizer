"""
GFLOPs of an Atomiser_Unet forward pass, depending on the levels kept
=====================================================================

For every depth l = S-1 (full model) down to 0, runs ONLY what a prediction from levels
0..l needs, and counts its FLOPs (torch.utils.flop_counter: matrix products, attention,
convolutions):

    leaf read                atoms -> level-0 cells
    encoder level j <= l     local blocks (global blocks at level S-1 in the full model)
    merges j -> j+1, j < l
    temporal level j <= l    per-level temporal transformers (per-level temporal models)
    decoder                  stages l-1 .. 0 (D_l = projected skip: nothing comes from above)
    query decoder            observation read + latent read + head, every pixel

Keeping levels 0..l is exactly what the ablation's keep_L0-{l} (and, for l = 0, no_coarse)
conditions evaluate, so each row's accuracy is the ablation's for that condition: the
levels above l are not computed at all.

PyTorch's fused transformer "fast path" is disabled while counting (its fused kernels are
invisible to the counter). Not counted: building the hierarchy geometry (cached, once per token layout), nearest-
neighbour searches and gathers (no matrix products). Weights are random: FLOPs depend only
on the shapes. The input is a synthetic sample with the dataset's layout.

    python scripts_atomizer/gflops_by_depth.py \\
        --config ./training/configs/config_test-PASTIS-unet-legacy-temporal.yaml
    python scripts_atomizer/gflops_by_depth.py \\
        --config ./training/configs/config_test-BURNSCARS-legacy.yaml --H 512 --W 512 --dates 1 --atoms_per_pixel 6
    add --time (GPU) to also measure the latency of each depth
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import argparse
import contextlib
import io
import time

import torch
from torch.utils.flop_counter import FlopCounterMode


# =============================================================================
# Synthetic input with the dataset's layout
# =============================================================================
def synthetic_batch(H, W, C, T, temporal, device, offset=7680.0):
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1) + offset

    def frame(t):
        f = torch.zeros(C * H * W, 8)
        f[:, 0] = torch.randn(C * H * W)
        f[:, 1:3] = xy.repeat(C, 1)
        f[:, 3] = torch.arange(C).repeat_interleave(H * W).float()
        f[:, 6] = 1.0
        f[:, 7] = float(10 + 50 * t)
        return f

    if temporal:
        tok = torch.stack([frame(t) for t in range(T)]).unsqueeze(0)         # [1, T, N, 8]
        mask = torch.zeros(1, T, tok.shape[2], dtype=torch.bool)
    else:                                                                    # joint: all dates as atoms
        tok = torch.cat([frame(t) for t in range(T)]).unsqueeze(0)           # [1, T*N, 8]
        mask = torch.zeros(1, tok.shape[1], dtype=torch.bool)
    q = torch.zeros(1, H * W, 8)
    q[0, :, 1:3] = xy
    b = {"groups": {10.0: {"tokens": tok.to(device), "mask": mask.to(device), "shape": (H, W)}},
         "queries": q.to(device), "queries_mask": torch.zeros(1, H * W, dtype=torch.bool, device=device)}
    if temporal:
        b["time_positions"] = torch.tensor([[10.0 + 50 * t for t in range(T)]], device=device)
    return b


# =============================================================================
# Forward pass truncated at level l, with FLOPs per component
# =============================================================================
class Counter:
    def __init__(self, enabled=True):
        self.parts, self.enabled = {}, enabled
        if enabled and hasattr(torch.backends, "mha"):
            # nn.TransformerEncoder / MultiheadAttention's fused inference "fast path" is invisible
            # to the FLOP counter: disable it while counting (same arithmetic, counted ops)
            torch.backends.mha.set_fastpath_enabled(False)

    @contextlib.contextmanager
    def part(self, name):
        if not self.enabled:
            yield
            return
        with FlopCounterMode(display=False) as fc:
            yield
        self.parts[name] = self.parts.get(name, 0) + fc.get_total_flops()


@torch.no_grad()
def forward_to_depth(model, batch, l, counter):
    """Prediction from levels 0..l only (l = S-1: the full model)."""
    unet, S = model.unet, model.unet.S
    tokens, mask = model._gather_inputs(batch)
    k, m = model.km_eval
    temporal = tokens.dim() == 4
    if temporal:
        B, T, N, C = tokens.shape
        tok_in, msk_in = tokens.reshape(B * T, N, C), mask.reshape(B * T, N)
    else:
        B, T = tokens.shape[0], 1
        tok_in, msk_in = tokens, mask
    tok_sel, _, bias = model.pruning(tok_in, msk_in, None, geo_k=m, atoms_per_latent=k)   # not counted
    geom = model.pruning.last_geometry
    with counter.part("leaf read"):
        z0 = model.leaf(tok_sel, bias, geom)
        z0 = z0 + model.empty_emb * model.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
    x, skips = z0, []
    for j in range(l + 1):
        c = unet.cell_coords(geom, j, j)
        with counter.part(f"encoder L{j}"):
            if j < S - 1:
                for blk in unet.enc[j]:
                    x = blk(x, geom.local_idx[j], c)
            else:
                for blk in unet.enc[j]:
                    x = blk(x, c)
        skips.append(x)
        if j < l:
            with counter.part(f"merge L{j}->L{j + 1}"):
                q = unet.region_queries(geom, j + 1).expand(x.shape[0], -1, -1)
                x = q + unet.merge[j](q, x, geom.merge_idx[j], geom.merge_valid[j],
                                      unet.cell_coords(geom, j + 1, j + 1), unet.cell_coords(geom, j, j + 1))
    if temporal:
        skips_t = [s.view(B, T, s.shape[1], s.shape[2]) for s in skips]
        tp = batch.get("time_positions")
        tp = None if tp is None else tp.float()
        maps = []
        for j, st in enumerate(skips_t):
            with counter.part(f"temporal L{j}"):
                maps.append(_temporal_level(model.temporal, j, st, tp, mask.all(-1)))
    else:
        maps = skips
    from training.atomiser.Atomiser_unet import overlap_weights
    od = unet.odec
    with counter.part("decoder"):
        d = od.proj[l](maps[l])
        if l < S - 1:
            # level l's own decoder stage still runs (its fusion receives nothing from above:
            # D_l = s_l, then the FFN and refinement blocks), as in the trained model / ablation
            if od.ffn is not None:
                d = d + od.ffn[l](d)
            cl = unet.cell_coords(geom, l, l)
            for blk in od.refine[l]:
                d = blk(d, geom.local_idx[l], cl)
        for j in reversed(range(l)):
            s = od.proj[j](maps[j])
            t = od.transport[j](d)
            w = overlap_weights(geom, j).to(t.dtype)
            msg = (t[:, geom.up_idx[j]] * w[None, :, :, None]).sum(2)
            xx = od.fuse[j](s, msg)
            if od.ffn is not None:
                xx = xx + od.ffn[j](xx)
            cc = unet.cell_coords(geom, j, j)
            for blk in od.refine[j]:
                xx = blk(xx, geom.local_idx[j], cc)
            d = xx
    with counter.part("query decoder"):
        if temporal:
            preds = model.decoder(batch["queries"], tokens.reshape(B, T * N, C), mask.reshape(B, T * N),
                                  d, geom, n_dates=T)
        else:
            preds = model.decoder(batch["queries"], tokens, mask, d, geom)
    return preds, geom


def _temporal_level(temporal, j, x, time_pos, pad):
    """The temporal transformer of level j on [B, T, L, D] -> [B, L, D]."""
    B, T, L, D = x.shape
    seq = x.permute(0, 2, 1, 3).reshape(B * L, T, D)
    tp = None if time_pos is None else time_pos.unsqueeze(1).expand(B, L, T).reshape(B * L, T)
    pm = pad.unsqueeze(1).expand(B, L, T).reshape(B * L, T)
    return temporal.levels[j](seq, time_indices=tp, pad_mask=pm).reshape(B, L, D)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="GFLOPs of Atomiser_Unet by number of levels kept")
    ap.add_argument("--config", required=True)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--dates", type=int, default=None, help="default: dataset.multi_temporal (or 1)")
    ap.add_argument("--atoms_per_pixel", type=int, default=13, help="per date (PASTIS: 10 S2 + 3 S1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--time", action="store_true", help="also measure latency (best of 5 runs)")
    ap.add_argument("--accuracy", default=None,
                    help="optional 'depth:score,...' to print next to the costs, e.g. 3:48.92,2:48.93,1:49.00,0:48.95")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    args = ap.parse_args()
    device = torch.device(args.device)

    from training.utils import read_yaml, Lookup_encoding
    from training.atomiser.Atomiser_unet import Atomiser_Unet
    config = read_yaml(args.config)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config)
    with contextlib.redirect_stdout(io.StringIO()):
        model = Atomiser_Unet(config=config, lookup_table=lookup).to(device).eval()
    for p in model.parameters():          # inference only (the FLOP counter's module tracker
        p.requires_grad_(False)           # cannot hook parameters used directly as inputs)
    if getattr(model.unet, "odec", None) is None:
        raise SystemExit("this script supports the overlap / pull decoders (unet_decoder: overlap)")
    temporal = getattr(model, "temporal_mode", "none") == "per_level"
    T = args.dates or int(config.get("dataset", {}).get("multi_temporal", 1))
    S = model.unet.S
    batch = synthetic_batch(args.H, args.W, args.atoms_per_pixel, T, temporal, device)
    acc = {}
    if args.accuracy:
        for item in args.accuracy.split(","):
            dpt, sc = item.split(":")
            acc[int(dpt)] = sc

    with contextlib.redirect_stdout(io.StringIO()):
        forward_to_depth(model, batch, S - 1, Counter(enabled=False))             # builds the geometry
    geom = model.pruning.last_geometry
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[gflops] {os.path.basename(args.config)} | {n_params:.1f}M params | "
          f"{'per-level temporal' if temporal else 'single-date / joint'} | image {args.H}x{args.W}, "
          f"{T} date(s), {args.atoms_per_pixel} atoms/pixel/date | cells per level {list(geom.sizes)}")

    rows = []
    for l in reversed(range(S)):
        cnt = Counter()
        forward_to_depth(model, batch, l, cnt)
        total = sum(cnt.parts.values())
        lat = None
        if args.time:
            ts = []
            for _ in range(5):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    forward_to_depth(model, batch, l, Counter(enabled=False))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            lat = min(ts) * 1e3
        rows.append((l, total, cnt.parts, lat))

    full = rows[0][1]
    print("\n| levels kept | GFLOPs | vs full | " + ("latency (ms) | " if args.time else "")
          + ("score | " if acc else "") + "\n|---|---|---|" + ("---|" if args.time else "") + ("---|" if acc else ""))
    for l, total, parts, lat in rows:
        name = f"0-{l}" if l > 0 else "0 only"
        print(f"| {name}{' (full)' if l == S - 1 else ''} | {total / 1e9:.1f} | {100 * total / full:.0f}% | "
              + (f"{lat:.1f} | " if lat is not None else "") + (f"{acc.get(l, '')} | " if acc else ""))

    print("\nBreakdown (GFLOPs):")
    comps = []
    for _, _, parts, _ in rows:
        for c in parts:
            if c not in comps:
                comps.append(c)
    print("| component | " + " | ".join(f"0-{l}" if l else "0 only" for l, *_ in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for c in comps:
        print(f"| {c} | " + " | ".join(f"{parts.get(c, 0) / 1e9:.2f}" for _, _, parts, _ in rows) + " |")


if __name__ == "__main__":
    main()
