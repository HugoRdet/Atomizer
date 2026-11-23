"""
utils_dataset_distill.py -- Atomizer-format datasets for the DUNE-style distillation.

Where it goes: training/utils/datasets/utils_dataset_distill.py

Two datasets, one per teacher (DUNE 'no data sharing'):
    OlmoSentinelAtoms : Sentinel tiles of the OlmoEarth h5 set, 14 bands (12 S2 + VV, VH), 128 x 128 @ 10 m,
                        one random date where S2 and S1 are both present. Teacher: OlmoEarth.
    FlairAtoms        : FLAIR-HUB aerial patches, 4 bands (R, G, B, NIR), `crop` x `crop` @ 0.2 m, read from the zips.
                        Teacher: DINOv3 (RGB only).

Sample format (same conventions as BurnScarsDataset, no queries/labels needed for distillation):
    {"groups": {resolution: {"tokens": [N, 8], "mask": [N] (all False), "shape": (C, H, W)}},
     "teacher": {...tensors the teacher needs...}, "dataset_name": "olmo" | "flair"}
Tokens come from TokenBuilder. Positions, spectral and resolution columns are identical for every sample of a layout,
so they are built ONCE (AtomTemplate) and only the value column is filled per sample.

Student values: the teacher's own normalized input (OlmoEarth-normalized S2/S1; FLAIR in [0, 1]) then a per-band z-score
with statistics computed once by check_distill_data.py --compute-stats (clamped to +-10, as in BurnScars).
Augmentation (train): random flips/rotations (and a random crop for FLAIR), applied to the pixels BEFORE both the atoms and
the teacher input, so student and teacher always see the same image. Positions never change (they come from pixel indices).

FLAIR views (distill.flair_views): a FLAIR item can be requested as dataset[(i, crop, up)]: a `crop` x `crop` patch whose
teacher input is upsampled `up` times (DINOv3 token = 16/up pixels of the crop). MixedViewBatchSampler draws one view per
FLAIR batch (the same on every rank) and yields keys for DistillConcat; FixedView pins one view (validation).
"""
import io
import os
import struct
import zipfile
import zlib
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .token_builder import TokenBuilder

IGNORE = 255
OLMO_S2 = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]  # h5 channel order
OLMO_S1 = ["VV", "VH"]


def parse_masks(col):
    """pandas Series of 12-char '0'/'1' strings (possibly empty) -> (N, 12) bool."""
    s = col.fillna("").astype(str).str.ljust(12, "0").str.slice(0, 12)
    return np.array(s.tolist(), dtype="U12").view(np.uint32).reshape(-1, 12) == ord("1")


def dihedral(a, k, flip):
    """Same random dihedral transform on the first two axes of an (H, W, ...) array."""
    a = np.rot90(a, k, axes=(0, 1))
    if flip:
        a = a[:, ::-1]
    return np.ascontiguousarray(a)


def spectral_indices(look_up, dataset_config, pairs):
    """pairs: [(bands.yaml section, band name), ...] in channel order -> LongTensor of look_up.table_wave indices
    (keys are (int(bandwidth), int(central_wavelength)), exactly as BurnScarsDataset._build_spectral_indices)."""
    out = []
    for section, name in pairs:
        e = dataset_config[section][name]
        key = (int(e["bandwidth"]), int(e["central_wavelength"]))
        if key not in look_up.table_wave:
            raise KeyError(f"[distill] {section}/{name} key={key} not in look_up.table_wave")
        out.append(look_up.table_wave[key])
    return torch.tensor(out, dtype=torch.long)


def read_zip_member_raw(path, offset, csize, ctype):
    """Read one zip member given its local-header offset, compressed size and method (from the index), with a plain
    open/seek/read: no central-directory parsing and no long-lived file handle (the full FLAIR set has 74 zips, up to 14 GB)."""
    with open(path, "rb") as f:
        f.seek(offset)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            raise IOError(f"bad zip local header at {path}:{offset}")
        n, m = struct.unpack("<HH", h[26:30])          # file-name length, extra-field length
        f.seek(offset + 30 + n + m)
        data = f.read(csize)
    if ctype == 0:                                      # stored
        return data
    if ctype == 8:                                      # deflate
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported zip compression {ctype}")


class AtomTemplate:
    """Static atom columns of one layout, built once with TokenBuilder (channel-major tokens, x = column, y = row)."""

    def __init__(self, builder, look_up, C, H, W, resolution, spectral_idx, time_idx=-1):
        label = torch.full((H, W), IGNORE, dtype=torch.long)
        self.tokens = builder.build_tokens(torch.zeros(C, H, W), label, resolution, spectral_idx,
                                           look_up.get_resolution_idx(resolution), time_idx)  # [C*H*W, 8]
        self.shape = (C, H, W)
        self.mask = torch.zeros(self.tokens.shape[0], dtype=torch.bool)

    def fill(self, image):
        """image [C, H, W] float -> tokens [N, 8] with the value column set."""
        t = self.tokens.clone()
        t[:, 0] = image.reshape(-1).float()
        return t


class _AtomDataset(Dataset):
    name = ""
    resolution = 0.0

    def _init_stats(self, stats_path, C, required):
        self.stats_path = stats_path
        if os.path.exists(stats_path):
            st = torch.load(stats_path, weights_only=True)
            self.mean, self.std = st["mean"].view(C, 1, 1), st["std"].view(C, 1, 1)
            print(f"[{self.name}] normalization stats loaded from {stats_path}")
        elif required:
            raise FileNotFoundError(f"{stats_path} not found: run check_distill_data.py --compute-stats once (before DDP)")
        else:
            self.mean, self.std = torch.zeros(C, 1, 1), torch.ones(C, 1, 1)

    def compute_and_save_stats(self, n=500, seed=0):
        """Per-channel mean/std of raw_image over n random samples; saved to stats_path."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), min(n, len(self)), replace=False)
        s = ss = None
        cnt = 0
        for i in idx:
            x = self.raw_image(int(i), aug=False).double()
            if s is None:
                s, ss = torch.zeros(x.shape[0], dtype=torch.float64), torch.zeros(x.shape[0], dtype=torch.float64)
            s += x.sum((1, 2))
            ss += (x ** 2).sum((1, 2))
            cnt += x.shape[1] * x.shape[2]
        mean = s / cnt
        std = (ss / cnt - mean ** 2).clamp_min(1e-12).sqrt()
        torch.save({"mean": mean.float(), "std": std.float()}, self.stats_path)
        print(f"[{self.name}] saved stats to {self.stats_path}: mean {mean.numpy().round(3)} std {std.numpy().round(3)}")
        self.mean, self.std = mean.float().view(-1, 1, 1), std.float().view(-1, 1, 1)

    def _zscore(self, x):
        return ((x - self.mean) / self.std).clamp(-10, 10)


# ---------------------------------------------------------------------------------------------------------
class OlmoSentinelAtoms(_AtomDataset):
    name, resolution, HW = "olmo", 10.0, 128

    def __init__(self, index_csv, tar_dir, split, look_up, dataset_config, aug, stats_path,
                 s2_section="bands_mmearth_s2", s1_section="bands_mmearth_s1", val_mod=200, stats_required=True):
        import pandas as pd
        assert split in ("train", "val")
        df = pd.read_csv(index_csv, usecols=["tar", "offset", "size", "sid", "s2_mask", "s1_mask", "error"], dtype=str)
        common = parse_masks(df["s2_mask"]) & parse_masks(df["s1_mask"])
        ok = common.any(1) & df["error"].isna().to_numpy()
        val = df["sid"].astype("int64").to_numpy() % val_mod == 0
        keep = ok & (val if split == "val" else ~val)
        self.tar, self.tar_dir = df["tar"].to_numpy()[keep], tar_dir
        self.offset = df["offset"].astype("int64").to_numpy()[keep]
        self.size = df["size"].astype("int64").to_numpy()[keep]
        self.common = common[keep]
        self.aug, self._fh, self._rng, self._norm = aug, {}, None, None
        pairs = [(s2_section, n) for n in OLMO_S2] + [(s1_section, n) for n in OLMO_S1]
        self.spectral_idx = spectral_indices(look_up, dataset_config, pairs)
        self.template = AtomTemplate(TokenBuilder(look_up), look_up, 14, self.HW, self.HW, self.resolution, self.spectral_idx)
        self._init_stats(stats_path, 14, stats_required)
        print(f"[olmo {split}] {int(keep.sum())} tiles | spectral idx {dict(zip(OLMO_S2 + OLMO_S1, self.spectral_idx.tolist()))}")

    def __len__(self):
        return len(self.tar)

    def _load(self, i, aug):
        import h5py
        import hdf5plugin  # noqa: F401
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            self._norm = Normalizer(std_multiplier=2.0)
        name = str(self.tar[i])
        if name not in self._fh:
            self._fh[name] = open(os.path.join(self.tar_dir, name), "rb")
        f = self._fh[name]
        f.seek(int(self.offset[i]))
        data = f.read(int(self.size[i]))
        cand = np.flatnonzero(self.common[i])
        t = int(self._rng.choice(cand)) if aug else int(cand[0])
        with h5py.File(io.BytesIO(data), "r") as h:
            arrs = {}
            for key, mod in (("sentinel2_l2a", Modality.SENTINEL2_L2A), ("sentinel1", Modality.SENTINEL1)):
                m = h[f"missing_timesteps_masks/{key}"][()]
                k = int(m[:t].sum())
                a = h[key][:, :, k:k + 1, :].astype(np.float32)                  # (H, W, 1, C) raw
                arrs[key] = self._norm.normalize(mod, a[None])[0].astype(np.float32)
            ts = h["timestamps"][()][t:t + 1].astype(np.int64)                   # (1, 3)
        if aug:
            k, flip = int(self._rng.integers(4)), bool(self._rng.integers(2))
            arrs = {key: dihedral(v, k, flip) for key, v in arrs.items()}
        return arrs["sentinel2_l2a"], arrs["sentinel1"], ts

    def raw_image(self, i, aug=False):
        s2n, s1n, _ = self._load(i, aug)
        return torch.from_numpy(np.concatenate([s2n[:, :, 0, :], s1n[:, :, 0, :]], -1)).permute(2, 0, 1).contiguous()  # [14,H,W]

    def __getitem__(self, i):
        s2n, s1n, ts = self._load(i, self.aug)
        x = torch.from_numpy(np.concatenate([s2n[:, :, 0, :], s1n[:, :, 0, :]], -1)).permute(2, 0, 1).contiguous()
        tokens = self.template.fill(self._zscore(x))
        return {"groups": {self.resolution: {"tokens": tokens, "mask": self.template.mask, "shape": self.template.shape}},
                "teacher": {"s2n": torch.from_numpy(s2n), "s1n": torch.from_numpy(s1n), "ts": torch.from_numpy(ts)},
                "dataset_name": self.name}


# ---------------------------------------------------------------------------------------------------------
class FlairAtoms(_AtomDataset):
    name, resolution = "flair", 0.2

    def __init__(self, index_csv, root, split, crop, look_up, dataset_config, aug, stats_path,
                 aerial_section="bands_flairhub_aerial", aerial_names=("RED", "GREEN", "BLUE", "NIR"),
                 share_s2=False, s2_section="bands_mmearth_s2", stats_required=True, crops=None):
        """crop: default crop side; crops: every crop side that may be requested (one token template each)."""
        import pandas as pd
        assert split in ("train", "val")
        cols = ["zip", "member", "split"] + [c for c in ("offset", "csize", "ctype") if c in pd.read_csv(index_csv, nrows=0).columns]
        df = pd.read_csv(index_csv, usecols=cols, dtype=str)
        df = df[df["split"] == ("test" if split == "val" else "train")]          # val = official test departments
        self.zip, self.member, self.root = df["zip"].to_numpy(), df["member"].to_numpy(), root
        self.raw = all(c in df.columns for c in ("offset", "csize", "ctype"))   # rebuilt index: read by byte offset
        if self.raw:
            self.offset, self.csize, self.ctype = (df[c].astype("int64").to_numpy() for c in ("offset", "csize", "ctype"))
        self.crop, self.aug, self._z, self._rng = crop, aug, OrderedDict(), None
        if share_s2:   # aerial R, G, B, NIR take the spectral index of Sentinel-2 B04, B03, B02, B08
            pairs = [(s2_section, n) for n in ("B04", "B03", "B02", "B08")]
        else:
            pairs = [(aerial_section, n) for n in aerial_names]
        self.spectral_idx = spectral_indices(look_up, dataset_config, pairs)
        builder = TokenBuilder(look_up)
        self.templates = {int(c): AtomTemplate(builder, look_up, 4, int(c), int(c), self.resolution, self.spectral_idx)
                          for c in sorted(set([int(crop)] + [int(c) for c in (crops or [])]))}
        self.template = self.templates[int(crop)]
        self._init_stats(stats_path, 4, stats_required)
        print(f"[flair {split}] {len(df)} patches, crops {sorted(self.templates)} (default {crop}) | spectral idx R,G,B,NIR = {self.spectral_idx.tolist()} "
              f"({'shared with S2' if share_s2 else aerial_section})")

    def __len__(self):
        return len(self.zip)

    def _load(self, i, aug, crop=None):
        import tifffile
        crop = int(crop or self.crop)
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        z = str(self.zip[i])
        if self.raw:
            data = read_zip_member_raw(os.path.join(self.root, z), int(self.offset[i]), int(self.csize[i]), int(self.ctype[i]))
        else:                                    # old index without offsets: ZipFile with a small LRU of open archives
            if z in self._z:
                self._z.move_to_end(z)
            else:
                if len(self._z) >= 16:
                    self._z.popitem(last=False)[1].close()
                self._z[z] = zipfile.ZipFile(os.path.join(self.root, z))
            data = self._z[z].read(str(self.member[i]))
        a = tifffile.imread(io.BytesIO(data))
        if a.ndim == 2:
            a = a[..., None]
        if a.shape[0] <= 8 and a.shape[-1] > 8:
            a = np.moveaxis(a, 0, -1)
        a = a[..., :4]
        assert a.dtype == np.uint8 and a.shape[-1] == 4, f"unexpected aerial patch {a.shape} {a.dtype}"
        H = a.shape[0]
        if crop > H or crop > a.shape[1]:
            raise ValueError(f"crop {crop} larger than the FLAIR patch {a.shape[:2]}")
        if crop < H:
            if aug:
                y0, x0 = (int(v) for v in self._rng.integers(0, H - crop + 1, size=2))
            else:
                y0 = x0 = (H - crop) // 2
            a = a[y0:y0 + crop, x0:x0 + crop]
        if aug:
            a = dihedral(a, int(self._rng.integers(4)), bool(self._rng.integers(2)))
        return np.ascontiguousarray(a)                                           # (crop, crop, 4) uint8

    def raw_image(self, i, aug=False):
        return torch.from_numpy(self._load(i, aug).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

    def __getitem__(self, key):
        """key: index, or (index, crop, up) for a FLAIR view (up: teacher input upsampling, applied by the teacher)."""
        i, crop, up = (key, self.crop, 1) if not isinstance(key, (tuple, list)) else (key[0], key[1], key[2])
        crop, up = int(crop), int(up)
        if crop not in self.templates:
            raise KeyError(f"crop {crop} not prepared (FlairAtoms crops={sorted(self.templates)})")
        tpl = self.templates[crop]
        a = self._load(int(i), self.aug, crop)
        x = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        tokens = tpl.fill(self._zscore(x))
        rgb = torch.from_numpy(np.ascontiguousarray(a[..., :3].transpose(2, 0, 1)))                # uint8 [3, H, W]
        return {"groups": {self.resolution: {"tokens": tokens, "mask": tpl.mask, "shape": tpl.shape}},
                "teacher": {"rgb": rgb}, "dataset_name": self.name, "view": (crop, up)}


def collate_distill(samples):
    """Stack samples of ONE dataset (the round-robin sampler guarantees homogeneous batches)."""
    res = next(iter(samples[0]["groups"]))
    return {
        "groups": {res: {"tokens": torch.stack([s["groups"][res]["tokens"] for s in samples]),
                         "mask": torch.stack([s["groups"][res]["mask"] for s in samples]),
                         "shape": samples[0]["groups"][res]["shape"]}},
        "teacher": {k: torch.stack([s["teacher"][k] for s in samples]) for k in samples[0]["teacher"]},
        "dataset_name": samples[0]["dataset_name"],
        **_view_of(samples),
    }


def _view_of(samples):
    if "view" not in samples[0]:
        return {}
    v = tuple(samples[0]["view"])
    if any(tuple(s["view"]) != v for s in samples):
        raise RuntimeError(f"mixed FLAIR views in one batch: {sorted(set(tuple(s['view']) for s in samples))}")
    return {"view": v}


# ---------------------------------------------------------------------------------------------------------
class DistillConcat(Dataset):
    """Several datasets addressed by keys (d, i) or (d, i, crop, up) (as yielded by MixedViewBatchSampler)."""

    def __init__(self, datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, key):
        d, rest = int(key[0]), tuple(key[1:])
        return self.datasets[d][rest[0]] if len(rest) == 1 else self.datasets[d][rest]


class FixedView(Dataset):
    """One FLAIR view of a FlairAtoms dataset (validation: one loader per view)."""

    def __init__(self, ds, crop, up):
        self.ds, self.crop, self.up = ds, int(crop), int(up)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[(i, self.crop, self.up)]


def _index_stream(n, g):
    while True:
        yield from torch.randperm(n, generator=g).tolist()


class MixedViewBatchSampler(Sampler):
    """Round-robin Sentinel / FLAIR batches (as RoundRobinDistributedBatchSampler) where every FLAIR batch takes a view
    drawn from `views` (dicts with crop, up, weight, batch). All ranks draw the same view and the same global index list
    (seed + epoch), then take their own slice, so every rank runs the same view at every step (DDP-safe; teacher
    standardizer gathers need equal shapes). Yields keys for DistillConcat([olmo, flair]): (0, i) and (1, i, crop, up).
    One epoch = as many rounds as the larger dataset needs at batch_per_dataset (training stops by optimizer steps)."""

    def __init__(self, n_olmo, n_flair, olmo_batch, views, num_replicas=1, rank=0, seed=42):
        self.n = (int(n_olmo), int(n_flair))
        self.ob, self.views = int(olmo_batch), list(views)
        self.W, self.rank, self.seed, self.epoch = int(num_replicas), int(rank), int(seed), 0
        self.weights = torch.tensor([float(v["weight"]) for v in self.views], dtype=torch.float64)
        if (self.weights <= 0).any():
            raise ValueError("flair_views weights must be > 0")
        mean_fb = float((self.weights * torch.tensor([float(v["batch"]) for v in self.views], dtype=torch.float64)).sum()
                        / self.weights.sum())
        self.rounds = max(1, max(self.n[0] // (self.ob * self.W), int(self.n[1] // (mean_fb * self.W))))

    def __len__(self):
        return 2 * self.rounds

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        base = self.seed + self.epoch
        g_o, g_f, g_v = (torch.Generator() for _ in range(3))
        g_o.manual_seed(base); g_f.manual_seed(base + 1_000_003); g_v.manual_seed(base + 2_000_029)
        so, sf = _index_stream(self.n[0], g_o), _index_stream(self.n[1], g_f)
        for _ in range(self.rounds):
            glob = [next(so) for _ in range(self.ob * self.W)]
            yield [(0, i) for i in glob[self.rank * self.ob:(self.rank + 1) * self.ob]]
            v = self.views[int(torch.multinomial(self.weights, 1, generator=g_v))]
            b = int(v["batch"])
            glob = [next(sf) for _ in range(b * self.W)]
            yield [(1, i, int(v["crop"]), int(v["up"])) for i in glob[self.rank * b:(self.rank + 1) * b]]
