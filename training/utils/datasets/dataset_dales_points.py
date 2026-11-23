"""
dataset_dales_points.py
=======================

Point-native DALES dataset for the Atomiser U-Net with the vertical stage.

What changes compared with the previous DalesDataset
----------------------------------------------------
* Positions are continuous metres (x, y centred on the tile, z relative to a
  label-free ground reference). No 0.2 m flooring, no TokenBuilder, no pixel
  frame: on DALES 84% of points share a 0.2 m pixel and ~35% (median tile)
  sit in a pixel spanning > 1 m in z, so flooring mixes stacked points.
* One atom per point, with a named column layout (PointLayout). Intensity and
  the echo pair are proper features instead of riding in column 6.
* Intensity is kept as a column for compatibility with sensors that record it.
  In the DALES files it is 0 for every point (standard field all zeros, no
  extra dimension), so the column is constant 0 there. When scene_stats is
  given, intensity is normalised per scene (dales_scene_stats.py); otherwise
  per tile. A tile with no intensity range gets 0, without warnings.
* Ground reference = a low z percentile of the tile, in EVERY split (the old
  code used the median of points labelled ground: label leakage at test time).
* Elevation value = compress(z_rel, value_scale_m) instead of clipping, so
  30-100 m structures stay distinguishable.
* Deterministic subsets use zlib.crc32 (Python's hash() changes per process);
  training augmentation is drawn freshly at every call.
* Free rotation in training (the geometry is computed on the fly, so no D4
  restriction and no precomputed assignment). Rotated points falling outside
  the fixed square are dropped, so every sample fits the same fixed lattice.
* Small tiles are filtered when the file list is built (from LAS headers),
  never silently replaced by another tile. Val/test keep every tile by default.
* No padding to 256k inside the dataset: collate_dales pads to the largest
  sample of the batch.

Height above LOCAL ground needs the columns (leaves), so it is computed in the
geometry module, not here.

Per-sample output (numpy -> torch)
----------------------------------
    points          [Nc, PointLayout.N_COLS] float32   context atoms
    point_labels    [Nc] int64                         labels of context points
    query_points    [Nq, PointLayout.N_COLS] float32   queries (same features)
    query_labels    [Nq] int64                         255 = ignore
    query_ctx_idx   [Nq] int64   row of the same point in `points`, -1 if absent
    query_src_idx   [Nq] int64   index of the point in the source tile file
    patch_id, scene_id (str), sensor_idx (int), center_xy [2], z_ground (float)

Queries are points, so the own-observation read of the query decoder uses the
query's own features (or query_ctx_idx); no site grid is involved.
"""
from __future__ import annotations

import json
import math
import warnings
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import laspy
except ImportError as e:  # pragma: no cover
    raise ImportError("laspy is required: pip install laspy[lazrs]") from e


# =============================================================================
# Layout, labels, small helpers
# =============================================================================
class PointLayout:
    """Column indices of one point atom."""
    X, Y, Z, VALUE, INTENSITY, ECHO_A, ECHO_B = range(7)
    N_COLS = 7
    NAMES = ("x_m", "y_m", "z_rel_m", "value", "intensity", "echo_a", "echo_b")
    POS = (X, Y)          # horizontal position (2D lattice)
    POS3 = (X, Y, Z)      # 3D position (vertical stage)


IGNORE_INDEX = 255
NUM_CLASSES = 8
CLASS_NAMES = ("ground", "vegetation", "cars", "trucks",
               "power_lines", "fences", "poles", "buildings")
# Raw DALES codes (verified on the train tiles: codes 0..8, 0 = unknown).
DALES_TO_LABEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}


def _build_lut(mapping, num_codes=256, ignore=IGNORE_INDEX):
    lut = np.full(num_codes, ignore, dtype=np.int64)
    for code, lab in mapping.items():
        lut[code] = lab
    return lut


REMAP_LUT = _build_lut(DALES_TO_LABEL)


def scene_of(stem: str) -> str:
    """Tile names are <scene>[_t<i>]_r<row>_c<col>; returns <scene>."""
    return stem.split("_r")[0].split("_t")[0]


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def compress(w: np.ndarray, s: float) -> np.ndarray:
    """w / (s + |w|): ~w/s near 0, saturates to +-1 (same as the model's)."""
    return w / (s + np.abs(w))


def echo_features(return_number: np.ndarray, number_of_returns: np.ndarray):
    """a = (r-1)/R (returns before), b = (R-r)/R (returns after), both in [0, 1)."""
    R = np.maximum(number_of_returns.astype(np.float64), 1.0)
    r = np.clip(return_number.astype(np.float64), 1.0, R)
    return (r - 1.0) / R, (R - r) / R


# =============================================================================
# Dataset
# =============================================================================
class DalesPointDataset(Dataset):
    """
    Args:
        tiled_dir:          directory of tiled patches (*.laz / *.las).
        split:              "train" | "val" | "test" (augmentation and random
                            subsets only in "train").
        scene_stats:        optional dales_scene_stats.json (intensity p1/p99
                            per scene). If None, intensity is normalised per
                            tile (all zeros on DALES either way).
        patch_size_m:       side of the fixed square lattice (tile size).
        max_context_points: cap on context atoms (256k covers every DALES tile).
        max_queries:        supervised queries per sample in training.
        eval_all_queries:   in val/test, every point of the tile is a query.
        min_points:         drop tiles below this count (train only by default).
        min_points_eval:    same for val/test (1 = keep everything).
        augment:            master switch (train only).
        rotation:           "free" | "d4" | "none".
        flip:               random mirror of x before rotation.
        scale_range:        e.g. (0.95, 1.05) for isotropic xy scaling; None = off.
        sigma_xy_m, sigma_z_m: Gaussian jitter in metres.
        ground_pct:         z percentile used as the tile ground reference.
        value_scale_m:      scale s of compress(z_rel, s) for the value column.
        sensor_idx:         sensor id passed to the model (DALES: one sensor).
        files:              optional explicit list of tile paths.
        load_partition:     also return the precomputed EZ-SP labels of each context
                            point ("point_part" [Nc, levels]), read from
                            <tile>_ezsp.npz next to each tile (precompute_ezsp.py).
        return_raw_xyz:     also return each query's integer LAS coordinates
                            (X, Y, Z as stored in the file) and the header
                            scale / offset, used to merge overlapping windows
                            per physical point (eval_dales_vote.py).
    """

    def __init__(
        self,
        tiled_dir: Optional[str] = None,
        split: str = "train",
        scene_stats: Optional[str] = None,
        patch_size_m: float = 50.0,
        max_context_points: int = 256_000,
        max_queries: int = 100_000,
        eval_all_queries: bool = True,
        min_points: int = 1000,
        min_points_eval: int = 1,
        augment: bool = True,
        rotation: str = "free",
        flip: bool = True,
        scale_range: Optional[Sequence[float]] = None,
        sigma_xy_m: float = 0.05,
        sigma_z_m: float = 0.02,
        ground_pct: float = 2.0,
        value_scale_m: float = 10.0,
        sensor_idx: int = 0,
        files: Optional[Sequence[str]] = None,
        return_raw_xyz: bool = False,
        load_partition: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), split
        assert rotation in ("free", "d4", "none"), rotation
        self.split = split
        self.train = split == "train"
        self.patch_size_m = float(patch_size_m)
        self.half = self.patch_size_m / 2.0
        self.max_context_points = max_context_points
        self.max_queries = max_queries
        self.eval_all_queries = eval_all_queries
        self.augment = bool(augment and self.train)
        self.rotation = rotation
        self.flip = flip
        self.scale_range = tuple(scale_range) if scale_range else None
        self.sigma_xy_m = float(sigma_xy_m)
        self.sigma_z_m = float(sigma_z_m)
        self.ground_pct = float(ground_pct)
        self.value_scale_m = float(value_scale_m)
        self.sensor_idx = int(sensor_idx)
        self.return_raw_xyz = bool(return_raw_xyz)
        self.load_partition = bool(load_partition)
        self.verbose = verbose

        # ---- file list, filtered from headers (cheap, no point decoding) ----
        if files is None:
            if tiled_dir is None:
                raise ValueError("pass tiled_dir or files")
            root = Path(tiled_dir)
            files = sorted(root.glob("*.laz")) + sorted(root.glob("*.las"))
        files = [Path(f) for f in files]
        thr = min_points if self.train else min_points_eval
        self.rows, dropped = [], 0
        for f in files:
            with laspy.open(str(f)) as fh:
                n = int(fh.header.point_count)
            if n < thr:
                dropped += 1
                continue
            self.rows.append({"path": str(f), "patch_id": f.stem,
                              "scene_id": scene_of(f.stem), "n_points": n})
        if not self.rows:
            raise RuntimeError(f"[DALES] no tiles left (split={split}, min_points={thr})")

        # ---- intensity statistics (optional) ----
        self.scene_stats = None
        if scene_stats is not None:
            with open(scene_stats) as fh:
                self.scene_stats = json.load(fh)

        if verbose:
            n = np.array([r["n_points"] for r in self.rows])
            print(f"[DALES] split={split}: {len(self.rows)} tiles "
                  f"({dropped} dropped < {thr} pts), points/tile median {int(np.median(n))}, "
                  f"max {n.max()}; augment={self.augment} rotation={self.rotation}")

    def __len__(self):
        return len(self.rows)

    # ------------------------------------------------------------------ helpers
    def _intensity_bounds(self, scene_id: str, inten: np.ndarray):
        if self.scene_stats is not None:
            s = self.scene_stats.get(scene_id) or self.scene_stats.get("_global")
            if s is not None:
                return float(s["p1"]), float(s["p99"])
        return float(np.percentile(inten, 1)), float(np.percentile(inten, 99))

    def _augment_xy(self, x, y, rng):
        if self.flip and rng.random() < 0.5:
            x = -x
        if self.rotation == "free":
            t = rng.uniform(0.0, 2.0 * math.pi)
        elif self.rotation == "d4":
            t = rng.integers(4) * (math.pi / 2.0)
        else:
            t = 0.0
        if t != 0.0:
            c, s = math.cos(t), math.sin(t)
            x, y = c * x - s * y, s * x + c * y
        if self.scale_range is not None:
            sc = rng.uniform(*self.scale_range)
            x, y = x * sc, y * sc
        if self.sigma_xy_m > 0:
            x = x + rng.normal(0.0, self.sigma_xy_m, x.shape)
            y = y + rng.normal(0.0, self.sigma_xy_m, y.shape)
        return x, y

    # ------------------------------------------------------------------ sample
    def load_numpy(self, index: int) -> dict:
        """Builds one sample as numpy arrays (torch conversion in __getitem__)."""
        row = self.rows[index]
        rng = (np.random.default_rng() if self.train
               else np.random.default_rng(stable_seed(row["patch_id"])))

        las = laspy.read(row["path"])
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        n = len(x)
        cls = np.clip(np.asarray(las.classification, dtype=np.int64), 0, 255)
        labels = REMAP_LUT[cls]
        inten = np.asarray(las.intensity, dtype=np.float64)
        ea, eb = echo_features(np.asarray(las.return_number),
                               np.asarray(las.number_of_returns))
        src_idx = np.arange(n, dtype=np.int64)

        # ---- horizontal frame: bbox centre (tiles are <= patch_size_m wide) ----
        cx = 0.5 * (x.min() + x.max())
        cy = 0.5 * (y.min() + y.max())
        x, y = x - cx, y - cy
        ext = max(np.ptp(x), np.ptp(y))
        if ext > self.patch_size_m + 1e-3:
            warnings.warn(f"[DALES] {row['patch_id']}: extent {ext:.2f} m > patch "
                          f"{self.patch_size_m} m, points outside the square are dropped")

        # ---- vertical frame: label-free ground reference ----
        z_ground = float(np.percentile(z, self.ground_pct))
        z_rel = z - z_ground

        # ---- augmentation (train only) ----
        if self.augment:
            x, y = self._augment_xy(x, y, rng)
            if self.sigma_z_m > 0:
                z_rel = z_rel + rng.normal(0.0, self.sigma_z_m, z_rel.shape)

        # ---- keep the fixed square (rotation corners, oversize tiles) ----
        inside = (np.abs(x) <= self.half) & (np.abs(y) <= self.half)
        if not inside.all():
            keep = np.flatnonzero(inside)
            x, y, z_rel = x[keep], y[keep], z_rel[keep]
            labels, inten, ea, eb, src_idx = (labels[keep], inten[keep], ea[keep],
                                              eb[keep], src_idx[keep])
        n = len(x)

        # ---- features ----
        lo, hi = self._intensity_bounds(row["scene_id"], inten)
        if hi > lo:
            inten_n = np.clip((inten - lo) / (hi - lo), 0.0, 1.0)
        else:                                   # no intensity range (DALES: all 0)
            inten_n = np.zeros_like(inten)
        feats = np.empty((n, PointLayout.N_COLS), dtype=np.float32)
        feats[:, PointLayout.X] = x
        feats[:, PointLayout.Y] = y
        feats[:, PointLayout.Z] = z_rel
        feats[:, PointLayout.VALUE] = compress(z_rel, self.value_scale_m)
        feats[:, PointLayout.INTENSITY] = inten_n
        feats[:, PointLayout.ECHO_A] = ea
        feats[:, PointLayout.ECHO_B] = eb

        # ---- context ----
        if self.max_context_points is not None and n > self.max_context_points:
            ctx = np.sort(rng.choice(n, self.max_context_points, replace=False))
        else:
            ctx = np.arange(n)
        ctx_of_point = np.full(n, -1, dtype=np.int64)
        ctx_of_point[ctx] = np.arange(len(ctx))

        # ---- queries ----
        if self.train:
            # supervised queries are labelled points of the context
            cand = ctx[labels[ctx] != IGNORE_INDEX]
            if self.max_queries is not None and len(cand) > self.max_queries:
                q = np.sort(rng.choice(cand, self.max_queries, replace=False))
            else:
                q = cand
        else:
            q = np.arange(n) if self.eval_all_queries else ctx

        out = {
            "points": feats[ctx],
            "point_labels": labels[ctx],
            "query_points": feats[q],
            "query_labels": labels[q],
            "query_ctx_idx": ctx_of_point[q],
            "query_src_idx": src_idx[q],
            "patch_id": row["patch_id"],
            "scene_id": row["scene_id"],
            "sensor_idx": self.sensor_idx,
            "center_xy": np.array([cx, cy], dtype=np.float64),
            "z_ground": z_ground,
        }
        if self.load_partition:
            pp = Path(row["path"]); ppath = pp.with_name(pp.stem + "_ezsp.npz")
            with np.load(ppath) as zz:
                part = zz["part"]
            if part.shape[0] != len(np.asarray(las.x)):
                raise RuntimeError(f"[DALES] {ppath.name}: {part.shape[0]} labels for "
                                   f"{len(np.asarray(las.x))} points; rerun precompute_ezsp.py")
            out["point_part"] = part[src_idx[ctx]].astype(np.int64)
        if self.return_raw_xyz:
            raw = np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1).astype(np.int64)
            out["query_xyz_int"] = raw[src_idx[q]]
            out["las_scale_offset"] = np.concatenate([np.asarray(las.header.scales, np.float64),
                                                      np.asarray(las.header.offsets, np.float64)])
        return out

    def __getitem__(self, index: int) -> dict:
        s = self.load_numpy(index)
        for k in ("points", "query_points"):
            s[k] = torch.from_numpy(s[k])
        for k in ("point_labels", "query_labels", "query_ctx_idx", "query_src_idx"):
            s[k] = torch.from_numpy(s[k]).long()
        s["center_xy"] = torch.from_numpy(s["center_xy"])
        if "point_part" in s:
            s["point_part"] = torch.from_numpy(s["point_part"])
        return s

    def class_counts(self, max_tiles: Optional[int] = None) -> np.ndarray:
        """Label histogram over (a subset of) tiles, e.g. for class weights."""
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        rows = self.rows if max_tiles is None else self.rows[:max_tiles]
        for r in rows:
            cls = np.clip(np.asarray(laspy.read(r["path"]).classification, np.int64), 0, 255)
            lab = REMAP_LUT[cls]
            counts += np.bincount(lab[lab != IGNORE_INDEX], minlength=NUM_CLASSES)
        return counts


# =============================================================================
# Collate: pad to the largest sample of the batch
# =============================================================================
def collate_dales(batch: list) -> dict:
    """
    Pads every per-point tensor to the largest sample in the batch.
    Masks follow the existing convention: True = padding / invalid.

        points        [B, Nc, C]   point_mask  [B, Nc]   point_labels [B, Nc]
        query_points  [B, Nq, C]   query_mask  [B, Nq]   query_labels [B, Nq]
        query_ctx_idx [B, Nq] (-1 = absent / padding)
        query_src_idx [B, Nq] (-1 = padding)
    """
    B = len(batch)
    C = batch[0]["points"].shape[1]
    nc = max(s["points"].shape[0] for s in batch)
    nq = max(s["query_points"].shape[0] for s in batch)

    points = torch.zeros(B, nc, C)
    point_mask = torch.ones(B, nc, dtype=torch.bool)
    point_labels = torch.full((B, nc), IGNORE_INDEX, dtype=torch.long)
    query_points = torch.zeros(B, nq, C)
    query_mask = torch.ones(B, nq, dtype=torch.bool)
    query_labels = torch.full((B, nq), IGNORE_INDEX, dtype=torch.long)
    query_ctx_idx = torch.full((B, nq), -1, dtype=torch.long)
    query_src_idx = torch.full((B, nq), -1, dtype=torch.long)

    for b, s in enumerate(batch):
        a, q = s["points"].shape[0], s["query_points"].shape[0]
        points[b, :a] = s["points"]
        point_mask[b, :a] = False
        point_labels[b, :a] = s["point_labels"]
        query_points[b, :q] = s["query_points"]
        query_mask[b, :q] = False
        query_labels[b, :q] = s["query_labels"]
        query_ctx_idx[b, :q] = s["query_ctx_idx"]
        query_src_idx[b, :q] = s["query_src_idx"]

    extra = {}
    if "point_part" in batch[0]:                       # precomputed partition labels, -1 = padding
        lv = batch[0]["point_part"].shape[1]
        pp = torch.full((B, nc, lv), -1, dtype=torch.long)
        for b, s in enumerate(batch):
            pp[b, :s["point_part"].shape[0]] = s["point_part"]
        extra["point_part"] = pp
    return {
        **extra,
        "points": points, "point_mask": point_mask, "point_labels": point_labels,
        "query_points": query_points, "query_mask": query_mask,
        "query_labels": query_labels, "query_ctx_idx": query_ctx_idx,
        "query_src_idx": query_src_idx,
        "n_points": torch.tensor([s["points"].shape[0] for s in batch]),
        "n_queries": torch.tensor([s["query_points"].shape[0] for s in batch]),
        "patch_id": [s["patch_id"] for s in batch],
        "scene_id": [s["scene_id"] for s in batch],
        "sensor_idx": torch.tensor([s["sensor_idx"] for s in batch]),
        "center_xy": torch.stack([s["center_xy"] for s in batch]),
        "z_ground": torch.tensor([s["z_ground"] for s in batch]),
    }
