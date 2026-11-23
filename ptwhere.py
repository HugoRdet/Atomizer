"""ptv3_where_from.py, put next to check_ptv3_ordering.py"""
import os, sys, json, h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts_baselines.check_ptv3_ordering import load_ckpt, update_cm, miou_from_cm, seeded, IGNORE_INDEX, NUM_CLASSES
from training.PTv3.model_ptv3_seg import PTv3Segmenter
from training.utils.datasets_baselines.utils_dataset_cashew_baselines import CashewBaselineDataset

ROOT = "./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant"
CKPT = "./checkpoints/cashew_baselines/bl_ptv3_pixels_ptv3-epoch=93-val_mIoU=0.8491.ckpt"
dev = "cuda"

m = PTv3Segmenter(in_channels=12, num_classes=7, gsd=10.0, patch_size=1024,
                  enc_channels=(32, 64, 128, 256, 384))
load_ckpt(m, CKPT)
m = m.to(dev).eval()

def raw_hashes(name):
    """64-bit hash of each pixel's raw 12-band spectrum, plus a mask of nodata pixels."""
    with h5py.File(os.path.join(ROOT, f"{name}.hdf5"), "r") as f:
        keys = list(f.keys())
        a = np.stack([np.asarray(f[next(k for k in keys if k.startswith(p))], np.float32)
                      for p in CashewBaselineDataset.BAND_PREFIXES], -1).reshape(-1, 12)
    bad = np.isnan(a).any(1) | (a == 0).all(1)
    u = a.view(np.uint32).astype(np.uint64)
    h = np.zeros(len(u), np.uint64)
    for i in range(12):
        h = (h ^ u[:, i]) * np.uint64(0x100000001B3)
    return h, bad

part = json.load(open(os.path.join(ROOT, "default_partition.json")))
train_set = np.unique(np.concatenate(
    [h[~b] for h, b in (raw_hashes(n) for n in part["train"])]))

ds = CashewBaselineDataset(root_path=ROOT, mode="test", crop_size=None, augment=False)
K = NUM_CLASSES
cms = {k: torch.zeros(K, K, dtype=torch.long) for k in ("all", "shuffled", "dup", "nodup")}
n_dup = n_tot = 0

with torch.inference_mode():
    for i in range(len(ds)):
        it = ds[i]
        x = it["image"]["s2"][None].to(dev)
        t = it["target"][None].to(dev)

        # (a) normal prediction
        p = seeded(m, x).argmax(1)

        # (b) scatter pixels randomly inside the tile (same shuffle for image
        #     and labels): keeps every spectrum, destroys all spatial context
        perm = torch.randperm(t.numel(), device=dev)
        xs = x.flatten(2)[:, :, perm].view_as(x)
        ts = t.flatten(1)[:, perm].view_as(t)
        ps = seeded(m, xs).argmax(1)

        # (c) which test pixels have an exact spectral duplicate in train
        h, bad = raw_hashes(it["metadata"]["sample_name"])
        dup = torch.from_numpy(np.isin(h, train_set) & ~bad).view(t.shape).to(dev)
        n_dup += int(dup.sum()); n_tot += dup.numel()

        update_cm(cms["all"], p, t)
        update_cm(cms["shuffled"], ps, ts)
        update_cm(cms["dup"], p, torch.where(dup, t, IGNORE_INDEX))
        update_cm(cms["nodup"], p, torch.where(~dup, t, IGNORE_INDEX))

print(f"test pixels with an exact duplicate in train: {n_dup / n_tot:.1%}")
for k, cm in cms.items():
    print(f"mIoU {k:9s}: {miou_from_cm(cm)[0]:.4f}   ({int(cm.sum())} px)")
