"""Downstream datasets = the repo's own Atomizer datasets (PANGAEA splits), configured for frozen evaluation:
full token grid (no token / query subsampling), no augmentation, no band dropout, deterministic order."""
from __future__ import annotations

import copy

import torch

SPECS = {
    #  name          task            classes  ignore  GSD
    "burnscars":   dict(task="seg", num_classes=2, ignore=255),
    "sen1floods11": dict(task="seg", num_classes=2, ignore=255),
    "pastis":      dict(task="seg_temporal", num_classes=20, ignore=255),   # 19 (void) and >= 20 are already IGNORE
    "eurosat":     dict(task="cls", num_classes=10, ignore=None),
}
DEFAULT_ROOTS = {"burnscars": "./data/hls_burn_scars", "sen1floods11": "./data/SENFLOOD",
                 "pastis": "./data/PASTIS-HD", "eurosat": "./data/geo-bench-1.0/classification_v1.0/m-eurosat"}

BIG = 10 ** 9


def _config_model(ds_cfg: dict | None, num_classes: int, multi_temporal: int):
    c = copy.deepcopy(ds_cfg) if ds_cfg else {}
    tr = c.setdefault("trainer", {})
    tr.update(max_tokens=BIG, max_tokens_reconstruction=BIG, mode="segmentation", num_classes=num_classes,
              band_dropout_augmentation={"enabled": False})
    c.setdefault("dataset", {})["multi_temporal"] = multi_temporal
    return c


def build_dataset(name, split, bands, lookup, root=None, ds_cfg=None, multi_temporal=6):
    """split: 'train' | 'validation' | 'test'. bands = read_yaml(bands.yaml) (same file as distillation)."""
    spec = SPECS[name]
    kw = dict(root_path=root or DEFAULT_ROOTS[name], mode=split, dataset_config=bands,
              config_model=_config_model(ds_cfg, spec["num_classes"], multi_temporal), look_up=lookup)
    if name == "burnscars":
        from training.utils.datasets.utils_dataset_BURNSCARS import BurnScarsDataset
        ds = BurnScarsDataset(**kw)
    elif name == "sen1floods11":
        from training.utils.datasets.utils_dataset_senflood_skip import Sen1Floods11SkipDataset
        ds = Sen1Floods11SkipDataset(**kw)
        ds.band_dropout_enabled = False
        ds._d4_augment = lambda image, label: (image, label)
    elif name == "pastis":
        from training.utils.datasets.pastis_temporal_dataset import PastisHDTemporalDataset
        ds = PastisHDTemporalDataset(**kw, use_s1=True)
        ds.augment = False
    elif name == "eurosat":
        from training.utils.datasets.utils_dataset_Eurosat import EurosatDataset
        ds = EurosatDataset(**kw)
        ds._d4_augment = lambda image: image
    else:
        raise KeyError(name)
    return ds


def collate_eval(samples):
    """Keeps only what the frozen evaluation needs: tokens, mask, shape, label (+ day of year for PASTIS)."""
    res = next(iter(samples[0]["groups"]))
    out = {
        "groups": {res: {"tokens": torch.stack([s["groups"][res]["tokens"] for s in samples]),
                         "mask": torch.stack([torch.as_tensor(s["groups"][res]["mask"]) for s in samples]),
                         "shape": samples[0]["groups"][res]["shape"]}},
        "label": torch.stack([torch.as_tensor(s["label"]) for s in samples]),
    }
    if "time_positions" in samples[0]:
        out["doy"] = torch.stack([torch.as_tensor(s["time_positions"]).float() for s in samples])
    return out


def label_subset(n, frac, seed):
    """Seed-fixed subset of train indices (same subset for every model at a given seed / fraction)."""
    if frac >= 1.0:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[: max(1, int(round(frac * n)))].sort().values
