"""
precompute_ezsp.py -- EZ-SP partition of every tile, once, offline
==================================================================

Location: scripts_atomizer/precompute_ezsp.py

Writes <tile>_ezsp.npz next to each tile: "part" [n_points, levels] int32, the
component of each point at each level (nested), in the order of the points in the
.laz file. The partition is computed WITHOUT the column constraint; training splits
components by column on the fly (PrecomputedEzspBuilder, unet_hier_nodes: ezsp_pre),
which stays valid under rotations / flips / crops.

Same features, graph and parameters as the online EZ-SP mode (taken from the config):
0.25 m voxels linked within `radius` voxels, local shape + height above local ground,
nested EZ-SP with unet_ezsp_regs / unet_ezsp_min_sizes.

    # one GPU job (resumable: existing files are skipped unless --overwrite)
    python scripts_atomizer/precompute_ezsp.py --config_model config_test-DALES-unet-ezsp-pre.yaml \
        --data_root ./data/DALES --splits train val test
    # split the work across jobs:  --shard 0 --num_shards 4  (then 1, 2, 3)
"""
import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

try:
    import laspy
except ImportError as e:
    raise ImportError("laspy is required") from e

from training.utils import read_yaml
from training.utils.datasets.point_hierarchy import PointHierarchy
from training.utils.datasets.ezsp_hierarchy import EzspHierarchyBuilder
from training.utils.datasets.dataset_dales_points import PointLayout as PL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config_model", required=True)
    ap.add_argument("--data_root", default="./data/DALES")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    a = read_yaml(str(_REPO_ROOT / "training" / "configs" / args.config_model)).get("Atomiser", {})
    P_m = float(a.get("unet_patch_size_m", 50.0)); half = P_m / 2
    builder = EzspHierarchyBuilder(regs=a.get("unet_ezsp_regs", [0.02, 0.1]),
                                   min_sizes=a.get("unet_ezsp_min_sizes", [10, 1]),
                                   sizes=a.get("unet_hier_sizes", [1.0, 2.0]),
                                   vox=float(a.get("unet_ezsp_vox_m", 0.25)),
                                   radius=int(a.get("unet_ezsp_radius", 2)), half=half)
    # local ground (height feature): same columns and rule as the model
    ph_mod = PointHierarchy(patch_size_m=P_m, points_per_leaf=int(a.get("unet_points_per_leaf", 250)),
                            density=float(a.get("unet_point_density", 46.1)), levels=int(a.get("unet_levels", 3)),
                            ground_quantile=float(a.get("unet_ground_quantile", 0.02)),
                            ground_jump_m=float(a.get("unet_ground_jump_m", 2.0)), verbose=False)

    files = []
    for sp in args.splits:
        d = Path(args.data_root) / sp
        files += sorted(d.glob("*.laz")) + sorted(d.glob("*.las"))
    files = files[args.shard::args.num_shards]
    print(f"[precompute_ezsp] {len(files)} tiles (shard {args.shard}/{args.num_shards}) on {dev}, "
          f"regs {builder.regs}, min sizes {builder.min_sizes}")
    t0, done, skipped, comps = time.time(), 0, 0, []
    for i, f in enumerate(files):
        out = f.with_name(f.stem + "_ezsp.npz")
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        las = laspy.read(str(f))
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (las.x, las.y, las.z))
        n = len(x)
        # same frame as the dataset: centred on the tile's bounding box
        x = x - 0.5 * (x.min() + x.max()); y = y - 0.5 * (y.min() + y.max())
        x = np.clip(x, -half, half); y = np.clip(y, -half, half)
        pts = np.zeros((1, n, PL.N_COLS), np.float32)
        pts[0, :, PL.X], pts[0, :, PL.Y], pts[0, :, PL.Z] = x, y, z - np.percentile(z, 2)
        pt = torch.from_numpy(pts).to(dev)
        with torch.no_grad():
            pg = ph_mod(pt, torch.zeros(1, n, dtype=torch.bool, device=dev), cells=False)
            P = pt[0, :, [PL.X, PL.Y, PL.Z]].double()
            rel = torch.stack([P[:, 0] + half, P[:, 1] + half, P[:, 2] - P[:, 2].min()], -1).clamp(min=0)
            labs = builder.partition_points(P, rel, torch.zeros(n, dtype=torch.long, device=dev),
                                            pg.point_h[0].double(), 1)
        part = torch.stack([pl for _, pl in labs], 1).cpu().numpy().astype(np.int32)
        np.savez(out, part=part)
        comps.append([int(part[:, l].max()) + 1 for l in range(part.shape[1])])
        done += 1
        if done % 50 == 0 or i == len(files) - 1:
            el = time.time() - t0
            print(f"[precompute_ezsp] {i + 1}/{len(files)} | {el / done:.2f} s/tile | "
                  f"components per tile (mean) {np.mean(comps, 0).round(0).tolist()}", flush=True)
    print(f"[precompute_ezsp] done: {done} written, {skipped} skipped, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
