"""
MuRA-T / SpaceNet-7 Cross-Sensor Segmentation Dataset for Atomizer-IO
======================================================================

Multi-temporal building segmentation on co-registered Planet, Sentinel-2,
and Landsat-8 imagery over 60 globally distributed AOIs.

Indexing: Per (AOI, sensor, month) — ~936 samples per epoch.
Each sample is an ANCHOR month that defines the label. Additionally,
N-1 other months from the same (AOI, sensor) are loaded as temporal
context. All N months contribute tokens to the encoder, giving it
multi-temporal evidence to make predictions.

Random cropping: 512×512 crops in Planet pixel space (~2.45 km).
Same crop position across all timestamps.

Query resolution: ALL sensors query at Planet resolution (4.78m).
This gives high-res supervision even from coarse sensors (Landsat 30m).
The encoder processes tokens at native resolution; only the decoder
queries at Planet resolution, interpolating between coarse latents.

Token format:
    [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
     col 0  1  2       3          4        5            6             7
"""

import csv
import json
import os
import random
import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False

from .token_builder import TokenBuilder


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _round_gsd(gsd: float) -> float:
    return round(gsd, 2)

def _next_mul8(n: int) -> int:
    return int(math.ceil(n / 8) * 8)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 2
IGNORE_INDEX = 255
DATASET_NAME = "MURAT"
PLANET_GSD = 4.78

CLASS_NAMES = {0: "Background", 1: "Building"}

CROP_SIZE_PLANET = 512

def _compute_canonical_sizes(crop_planet: int) -> dict:
    physical_extent = crop_planet * PLANET_GSD
    return {
        _round_gsd(4.78): crop_planet,
        _round_gsd(10.0): _next_mul8(round(physical_extent / 10.0)),
        _round_gsd(20.0): _next_mul8(round(physical_extent / 20.0)),
        _round_gsd(60.0): _next_mul8(round(physical_extent / 60.0)),
        _round_gsd(30.0): _next_mul8(round(physical_extent / 30.0)),
    }

CANONICAL_SIZES = _compute_canonical_sizes(CROP_SIZE_PLANET)

# ── Sensor band metadata ─────────────────────────────────────────────
PLANET_BANDS_INFO = {
    "R": {"central_wavelength": 610, "bandwidth": 100, "idx": 0},
    "G": {"central_wavelength": 545, "bandwidth": 70,  "idx": 1},
    "B": {"central_wavelength": 475, "bandwidth": 60,  "idx": 2},
}

S2_BANDS_INFO = {
    "B01": {"central_wavelength": 443,  "bandwidth": 20,  "idx": 0,  "gsd": 60.0},
    "B02": {"central_wavelength": 490,  "bandwidth": 65,  "idx": 1,  "gsd": 10.0},
    "B03": {"central_wavelength": 560,  "bandwidth": 35,  "idx": 2,  "gsd": 10.0},
    "B04": {"central_wavelength": 665,  "bandwidth": 30,  "idx": 3,  "gsd": 10.0},
    "B05": {"central_wavelength": 705,  "bandwidth": 15,  "idx": 4,  "gsd": 20.0},
    "B06": {"central_wavelength": 740,  "bandwidth": 15,  "idx": 5,  "gsd": 20.0},
    "B07": {"central_wavelength": 783,  "bandwidth": 20,  "idx": 6,  "gsd": 20.0},
    "B08": {"central_wavelength": 842,  "bandwidth": 115, "idx": 7,  "gsd": 10.0},
    "B8A": {"central_wavelength": 865,  "bandwidth": 20,  "idx": 8,  "gsd": 20.0},
    "B09": {"central_wavelength": 945,  "bandwidth": 20,  "idx": 9,  "gsd": 60.0},
    "B10": {"central_wavelength": 1375, "bandwidth": 30,  "idx": 10, "gsd": 60.0},
    "B11": {"central_wavelength": 1610, "bandwidth": 90,  "idx": 11, "gsd": 20.0},
    "B12": {"central_wavelength": 2190, "bandwidth": 180, "idx": 12, "gsd": 20.0},
}

LANDSAT_BANDS_INFO = {
    "B1": {"central_wavelength": 443,  "bandwidth": 16,  "idx": 0, "gsd": 30.0},
    "B2": {"central_wavelength": 482,  "bandwidth": 60,  "idx": 1, "gsd": 30.0},
    "B3": {"central_wavelength": 562,  "bandwidth": 57,  "idx": 2, "gsd": 30.0},
    "B4": {"central_wavelength": 655,  "bandwidth": 37,  "idx": 3, "gsd": 30.0},
    "B5": {"central_wavelength": 865,  "bandwidth": 28,  "idx": 4, "gsd": 30.0},
    "B6": {"central_wavelength": 1609, "bandwidth": 85,  "idx": 5, "gsd": 30.0},
    "B7": {"central_wavelength": 2201, "bandwidth": 187, "idx": 6, "gsd": 30.0},
}

S2_BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]
LANDSAT_BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

S2_RES_GROUPS = {
    10.0: ["B02", "B03", "B04", "B08"],
    20.0: ["B05", "B06", "B07", "B8A", "B11", "B12"],
    60.0: ["B01", "B09", "B10"],
}

SENSOR_QUERY_RES = {
    "planet":    4.78,
    "sentinel2": 10.0,
    "landsat8":  30.0,
}


# ═══════════════════════════════════════════════════════════════════════
# LABEL / CROP / PAD HELPERS
# ═══════════════════════════════════════════════════════════════════════

def rasterize_buildings(geojson_path, height, width, transform):
    if not HAS_FIONA:
        raise ImportError("fiona required")
    shapes = []
    try:
        with fiona.open(geojson_path, "r") as src:
            for feat in src:
                geom = feat["geometry"]
                if geom is not None:
                    shapes.append((geom, 1))
    except Exception as e:
        print(f"[MuRA-T] Warning: Could not read {geojson_path}: {e}")
        return np.zeros((height, width), dtype=np.uint8)
    if len(shapes) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    return rio_rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=0, dtype=np.uint8, all_touched=True,
    )


def downsample_label_nearest(label, target_h, target_w):
    H, W = label.shape
    if H == target_h and W == target_w:
        return label
    return F.interpolate(
        label.unsqueeze(0).unsqueeze(0).float(),
        size=(target_h, target_w), mode="nearest",
    ).squeeze(0).squeeze(0).long()


def month_to_doy(month_key):
    year, month = month_key.split("_")
    from datetime import datetime
    return datetime(int(year), int(month), 15).timetuple().tm_yday


def compute_crop_origin(image_h, image_w, crop_h, crop_w, random_crop=True):
    max_y = max(0, image_h - crop_h)
    max_x = max(0, image_w - crop_w)
    if random_crop and (max_y > 0 or max_x > 0):
        return random.randint(0, max_y), random.randint(0, max_x)
    return max_y // 2, max_x // 2


def planet_origin_to_sensor(
    planet_y0, planet_x0, planet_h, planet_w,
    sensor_h, sensor_w, crop_planet, sensor_gsd,
):
    ry = planet_y0 / max(planet_h - crop_planet, 1) if planet_h > crop_planet else 0.0
    rx = planet_x0 / max(planet_w - crop_planet, 1) if planet_w > crop_planet else 0.0
    physical_extent = crop_planet * PLANET_GSD
    crop_sensor = min(round(physical_extent / sensor_gsd), sensor_h, sensor_w)
    max_y = max(0, sensor_h - crop_sensor)
    max_x = max(0, sensor_w - crop_sensor)
    return round(ry * max_y), round(rx * max_x), crop_sensor


def pad_to_canonical(image, label, gsd):
    gsd_key = _round_gsd(gsd)
    S = CANONICAL_SIZES[gsd_key]
    C, H, W = image.shape
    pad_h, pad_w = S - H, S - W

    if pad_h < 0 or pad_w < 0:
        image = image[:, :S, :S]
        label = label[:S, :S]
        H, W = min(H, S), min(W, S)
        pad_h, pad_w = S - H, S - W

    valid_mask = torch.zeros(S, S, dtype=torch.bool)
    valid_mask[:H, :W] = True

    if pad_h == 0 and pad_w == 0:
        return image, label, valid_mask

    image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    label = F.pad(
        label.unsqueeze(0).float(), (0, pad_w, 0, pad_h), value=IGNORE_INDEX
    ).squeeze(0).long()

    return image, label, valid_mask


# ═══════════════════════════════════════════════════════════════════════
# MAIN DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════

class MuRATSegmentation(Dataset):
    """
    Multi-temporal cross-sensor building segmentation dataset.

    Indexed per (AOI, sensor, month) — ~936 samples per epoch.
    Each sample is an anchor month (defines label/queries).
    N-1 additional months from the same (AOI, sensor) are loaded
    as temporal context. All N months contribute tokens.

    All sensors query at Planet resolution (4.78m) for high-res
    supervision, regardless of input sensor resolution.
    """

    TASK_NAME = "murat_segmentation"

    def __init__(
        self,
        index_csv: str,
        stats_json: str,
        look_up,
        mode: str = "train",
        sensors: Optional[List[str]] = None,
        config_model: dict = None,
        max_queries: int = 65_536,
        label_cache_dir: Optional[str] = None,
        augment: bool = True,
        data_root: Optional[str] = None,
        crop_size: int = CROP_SIZE_PLANET,
        n_temporal: int = 6,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.sensors = sensors or ["planet", "sentinel2", "landsat8"]
        self.look_up = look_up
        self.config_model = config_model
        self.max_queries = max_queries
        self.augment = augment and (mode == "train")
        self.data_root = data_root or ""
        self.crop_size = crop_size
        self.n_temporal = n_temporal
        self.random_crop = (mode == "train")

        self.token_builder = TokenBuilder(look_up)

        if label_cache_dir is None:
            label_cache_dir = os.path.join(
                os.path.dirname(index_csv), ".label_cache"
            )
        self.label_cache_dir = label_cache_dir
        os.makedirs(self.label_cache_dir, exist_ok=True)

        self.samples = self._load_index(index_csv)
        print(f"[MuRA-T] {len(self.samples)} samples for mode='{mode}', "
              f"sensors={self.sensors}")

        self._siblings = self._build_sibling_lookup()
        self._planet_geo_cache = self._build_planet_geo_cache(index_csv)

        with open(stats_json, "r") as f:
            self.all_norm_stats = json.load(f)
        self._build_norm_tensors()

        self._setup_band_indices()
        self._setup_resolution_indices()

        sensor_counts = defaultdict(int)
        for s in self.samples:
            sensor_counts[s["sensor"]] += 1
        for sensor, count in sorted(sensor_counts.items()):
            print(f"[MuRA-T]   {sensor}: {count} samples")

        avg_siblings = sum(len(v) for v in self._siblings.values()) / max(len(self._siblings), 1)
        print(f"[MuRA-T] Avg months per (AOI,sensor): {avg_siblings:.1f}, "
              f"using {n_temporal} per sample")
        print(f"[MuRA-T] Crop: {crop_size}×{crop_size} Planet px "
              f"({crop_size * PLANET_GSD / 1000:.2f} km)")
        print(f"[MuRA-T] Query resolution: ALL sensors → {PLANET_GSD}m (Planet)")
        print(f"[MuRA-T] Canonical sizes:")
        for gsd in sorted(CANONICAL_SIZES.keys()):
            print(f"  {gsd:>8.2f}m → {CANONICAL_SIZES[gsd]}×{CANONICAL_SIZES[gsd]}")

    # ═════════════════════════════════════════════════════════════════
    # INDEX LOADING
    # ═════════════════════════════════════════════════════════════════

    def _load_index(self, csv_path: str) -> List[dict]:
        samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != self.mode:
                    continue
                if row["sensor"] not in self.sensors:
                    continue

                band_files = row["band_files"]
                label_path = row["label_path"]

                if self.data_root:
                    if "|" in band_files:
                        band_files = "|".join(
                            os.path.join(self.data_root, p)
                            for p in band_files.split("|")
                        )
                    else:
                        band_files = os.path.join(self.data_root, band_files)
                    label_path = os.path.join(self.data_root, label_path)

                samples.append({
                    "aoi": row["aoi"],
                    "sensor": row["sensor"],
                    "month": row["month"],
                    "band_files": band_files,
                    "label_path": label_path,
                    "num_bands": int(row["num_bands"]),
                })
        return samples

    def _build_sibling_lookup(self) -> dict:
        siblings = defaultdict(list)
        for s in self.samples:
            siblings[(s["aoi"], s["sensor"])].append({
                "month": s["month"],
                "band_files": s["band_files"],
                "label_path": s["label_path"],
            })
        for key in siblings:
            siblings[key].sort(key=lambda m: m["month"])
        return dict(siblings)

    def _select_context_months(self, aoi, sensor, anchor_month):
        siblings = self._siblings.get((aoi, sensor), [])
        n_need = self.n_temporal - 1

        anchor_info = None
        other_infos = []
        for s in siblings:
            if s["month"] == anchor_month:
                anchor_info = s
            else:
                other_infos.append(s)

        if anchor_info is None:
            return [{"month": anchor_month, "band_files": "", "label_path": ""}]

        if n_need == 0 or len(other_infos) == 0:
            return [anchor_info]

        if self.mode == "train":
            if len(other_infos) >= n_need:
                context = random.sample(other_infos, n_need)
            else:
                context = random.choices(other_infos, k=n_need)
        else:
            if len(other_infos) >= n_need:
                step = len(other_infos) / n_need
                indices = [int(i * step) for i in range(n_need)]
                context = [other_infos[i] for i in indices]
            else:
                context = list(other_infos)
                while len(context) < n_need:
                    context.append(other_infos[-1])

        all_months = context + [anchor_info]
        all_months.sort(key=lambda m: m["month"])
        return all_months

    # ═════════════════════════════════════════════════════════════════
    # INIT HELPERS
    # ═════════════════════════════════════════════════════════════════

    def _build_planet_geo_cache(self, csv_path):
        aoi_to_path = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sensor"] != "planet":
                    continue
                aoi = row["aoi"]
                if aoi in aoi_to_path:
                    continue
                path = row["band_files"]
                if self.data_root:
                    path = os.path.join(self.data_root, path)
                aoi_to_path[aoi] = path

        cache = {}
        for aoi, path in aoi_to_path.items():
            try:
                with rasterio.open(path) as src:
                    cache[aoi] = {
                        "transform": src.transform, "crs": src.crs,
                        "height": src.height, "width": src.width,
                    }
            except Exception as e:
                print(f"[MuRA-T] Warning: Planet geo for {aoi}: {e}")
        return cache

    def _build_norm_tensors(self):
        self.norm_stats = {}
        for sensor in ["planet", "sentinel2", "landsat8"]:
            if sensor not in self.all_norm_stats:
                continue
            ss = self.all_norm_stats[sensor]
            if sensor == "planet":
                bo = ["R", "G", "B"]
            elif sensor == "sentinel2":
                bo = S2_BAND_ORDER
            else:
                bo = LANDSAT_BAND_ORDER
            means = [ss[b]["mean"] if b in ss else 0.0 for b in bo]
            stds = [ss[b]["std"] if b in ss else 1.0 for b in bo]
            self.norm_stats[sensor] = {
                "mean": torch.tensor(means, dtype=torch.float32),
                "std": torch.tensor(stds, dtype=torch.float32).clamp(min=1e-6),
            }

    def _setup_band_indices(self):
        self.spectral_indices = {}

        pi = []
        for bn in ["R", "G", "B"]:
            info = PLANET_BANDS_INFO[bn]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                self.look_up.table_wave[key] = len(self.look_up.table_wave)
            pi.append(self.look_up.table_wave[key])
        self.spectral_indices["planet"] = torch.tensor(pi, dtype=torch.long)

        si = []
        for bn in S2_BAND_ORDER:
            info = S2_BANDS_INFO[bn]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                self.look_up.table_wave[key] = len(self.look_up.table_wave)
            si.append(self.look_up.table_wave[key])
        self.spectral_indices["sentinel2"] = torch.tensor(si, dtype=torch.long)

        self.s2_group_spectral_indices = {}
        for gsd, bns in S2_RES_GROUPS.items():
            gsd = _round_gsd(gsd)
            gi = []
            for bn in bns:
                info = S2_BANDS_INFO[bn]
                key = (info["bandwidth"], info["central_wavelength"])
                gi.append(self.look_up.table_wave[key])
            self.s2_group_spectral_indices[gsd] = torch.tensor(gi, dtype=torch.long)

        li = []
        for bn in LANDSAT_BAND_ORDER:
            info = LANDSAT_BANDS_INFO[bn]
            key = (info["bandwidth"], info["central_wavelength"])
            if key not in self.look_up.table_wave:
                self.look_up.table_wave[key] = len(self.look_up.table_wave)
            li.append(self.look_up.table_wave[key])
        self.spectral_indices["landsat8"] = torch.tensor(li, dtype=torch.long)

        print(f"[MuRA-T] Spectral indices: planet={len(pi)}, "
              f"sentinel2={len(si)}, landsat8={len(li)}")

    def _setup_resolution_indices(self):
        self.resolution_indices = {}
        all_gsds = {_round_gsd(SENSOR_QUERY_RES["planet"]), _round_gsd(30.0)}
        for gsd in S2_RES_GROUPS:
            all_gsds.add(_round_gsd(gsd))
        for gsd in sorted(all_gsds):
            self.resolution_indices[gsd] = self.look_up.get_resolution_idx(gsd)
            self.token_builder._ensure_resolution_registered(gsd)
        print(f"[MuRA-T] Resolution indices: {self.resolution_indices}")

    # ═════════════════════════════════════════════════════════════════
    # FILE LOADING
    # ═════════════════════════════════════════════════════════════════

    def _load_planet_image(self, band_files):
        with rasterio.open(band_files) as src:
            data = src.read([1, 2, 3]).astype(np.float32)
        image = torch.from_numpy(data)
        stats = self.norm_stats.get("planet")
        if stats is not None:
            image = (image - stats["mean"][:, None, None]) / stats["std"][:, None, None]
        return image

    def _load_landsat_image(self, band_files):
        paths = band_files.split("|")
        bands = []
        for path in paths:
            with rasterio.open(path) as src:
                bands.append(torch.from_numpy(src.read(1).astype(np.float32)))
        image = torch.stack(bands, dim=0)
        stats = self.norm_stats.get("landsat8")
        if stats is not None:
            image = (image - stats["mean"][:, None, None]) / stats["std"][:, None, None]
        return image

    def _load_s2_images(self, band_files):
        paths = band_files.split("|")
        assert len(paths) == 13
        band_paths = dict(zip(S2_BAND_ORDER, paths))
        res_images = {}
        for gsd_raw, bns in S2_RES_GROUPS.items():
            gsd = _round_gsd(gsd_raw)
            bands = []
            for bn in bns:
                with rasterio.open(band_paths[bn]) as src:
                    bands.append(torch.from_numpy(src.read(1).astype(np.float32)))
            group = torch.stack(bands, dim=0)
            stats = self.norm_stats.get("sentinel2")
            if stats is not None:
                idx = [S2_BAND_ORDER.index(bn) for bn in bns]
                group = (group - stats["mean"][idx, None, None]) / stats["std"][idx, None, None]
            res_images[gsd] = group
        return res_images

    # ═════════════════════════════════════════════════════════════════
    # LABEL
    # ═════════════════════════════════════════════════════════════════

    def _get_planet_label(self, label_path, aoi, month):
        planet_geo = self._planet_geo_cache.get(aoi)
        if planet_geo is None:
            return torch.zeros(1024, 1024, dtype=torch.long)
        H, W = planet_geo["height"], planet_geo["width"]
        cache_key = f"planet_{aoi}_{month}_{H}x{W}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_path = os.path.join(self.label_cache_dir, f"label_{cache_hash}.npy")
        if os.path.exists(cache_path):
            return torch.from_numpy(np.load(cache_path)).long()
        label = rasterize_buildings(label_path, H, W, planet_geo["transform"])
        try:
            np.save(cache_path, label)
        except Exception:
            pass
        return torch.from_numpy(label).long()

    # ═════════════════════════════════════════════════════════════════
    # AUGMENTATION
    # ═════════════════════════════════════════════════════════════════

    def _get_d4_params(self):
        return random.randint(0, 3), random.random() > 0.5

    def _apply_d4(self, image, label, valid_mask, k, flip):
        if k > 0:
            image = torch.rot90(image, k, dims=(-2, -1))
            label = torch.rot90(label, k, dims=(-2, -1))
            valid_mask = torch.rot90(valid_mask, k, dims=(-2, -1))
        if flip:
            image = torch.flip(image, dims=(-1,))
            label = torch.flip(label, dims=(-1,))
            valid_mask = torch.flip(valid_mask, dims=(-1,))
        return image, label, valid_mask

    @staticmethod
    def _build_token_mask(valid_mask, n_bands):
        return (~valid_mask).unsqueeze(0).expand(n_bands, -1, -1).reshape(-1)

    # ═════════════════════════════════════════════════════════════════
    # QUERY BUILDER (shared by all sensors)
    # ═════════════════════════════════════════════════════════════════

    def _build_planet_res_queries(self, anchor_label_crop, planet_y0, planet_x0,
                                   anchor_month, aug_k, aug_flip,
                                   first_spectral_idx):
        """
        Build queries at Planet resolution (4.78m) from anchor label.
        Used by ALL sensors — decoder interpolates between coarse latents.

        Args:
            anchor_label_crop: [H, W] label at Planet pixel resolution
            first_spectral_idx: spectral index for query token metadata
        """
        query_gsd = _round_gsd(PLANET_GSD)
        query_res_idx = self.resolution_indices[query_gsd]

        # Pad to canonical Planet size (512×512)
        dummy_img = torch.zeros(1, anchor_label_crop.shape[0], anchor_label_crop.shape[1])
        _, query_label, _ = pad_to_canonical(dummy_img, anchor_label_crop, query_gsd)

        if self.augment:
            dummy_vm = torch.ones_like(query_label, dtype=torch.bool)
            _, query_label, _ = self._apply_d4(
                torch.zeros(1, *query_label.shape), query_label, dummy_vm,
                aug_k, aug_flip,
            )

        anchor_doy = month_to_doy(anchor_month)
        anchor_time_idx = self.look_up.get_or_register_time_idx(anchor_doy)

        queries = self.token_builder.build_queries(
            label=query_label, resolution=query_gsd,
            first_spectral_idx=first_spectral_idx,
            resolution_idx=query_res_idx, time_idx=anchor_time_idx,
        )
        queries = self.token_builder.subsample_queries(
            queries, max_queries=self.max_queries,
            ignore_index=IGNORE_INDEX, prioritize_valid=True,
        )

        return queries, query_gsd

    # ═════════════════════════════════════════════════════════════════
    # DUMMY
    # ═════════════════════════════════════════════════════════════════

    def _make_dummy(self, sensor):
        gsd = _round_gsd(PLANET_GSD)  # all queries at Planet res
        q = torch.zeros(1, 8)
        q[:, 4] = IGNORE_INDEX
        return {
            "groups": {gsd: {
                "tokens": torch.zeros(1, 8),
                "mask": torch.ones(1, dtype=torch.bool),
                "shape": (1, 1),
            }},
            "tasks": {self.TASK_NAME: {
                "queries": q,
                "queries_mask": torch.ones(1, dtype=torch.bool),
            }},
            "target_resolution": gsd,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # __len__ / __getitem__
    # ═════════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sensor = sample["sensor"]
        aoi = sample["aoi"]
        anchor_month = sample["month"]

        try:
            all_months = self._select_context_months(aoi, sensor, anchor_month)

            if sensor == "planet":
                return self._load_planet_mt(aoi, anchor_month, all_months)
            elif sensor == "sentinel2":
                return self._load_s2_mt(aoi, anchor_month, all_months)
            elif sensor == "landsat8":
                return self._load_landsat_mt(aoi, anchor_month, all_months)
        except Exception as e:
            print(f"[MuRA-T] Error {sensor}/{aoi}/{anchor_month}: {e}")
            return self._make_dummy(sensor)

    # ═════════════════════════════════════════════════════════════════
    # PLANET MULTI-TEMPORAL
    # ═════════════════════════════════════════════════════════════════

    def _load_planet_mt(self, aoi, anchor_month, all_months):
        gsd = _round_gsd(SENSOR_QUERY_RES["planet"])
        res_idx = self.resolution_indices[gsd]

        first_img = self._load_planet_image(all_months[0]["band_files"])
        _, H, W = first_img.shape

        y0, x0 = compute_crop_origin(
            H, W, self.crop_size, self.crop_size,
            random_crop=self.random_crop,
        )

        aug_k, aug_flip = self._get_d4_params() if self.augment else (0, False)

        all_tokens = []
        all_masks = []
        ref_shape = None

        for mi in all_months:
            image = self._load_planet_image(mi["band_files"])
            _, Hi, Wi = image.shape

            cy = min(y0, max(0, Hi - self.crop_size))
            cx = min(x0, max(0, Wi - self.crop_size))
            image = image[:, cy:cy+self.crop_size, cx:cx+self.crop_size]

            label = self._get_planet_label(mi["label_path"], aoi, mi["month"])
            Hl, Wl = label.shape
            ly = min(y0, max(0, Hl - self.crop_size))
            lx = min(x0, max(0, Wl - self.crop_size))
            label = label[ly:ly+self.crop_size, lx:lx+self.crop_size]

            image, label, valid_mask = pad_to_canonical(image, label, gsd)

            if self.augment:
                image, label, valid_mask = self._apply_d4(
                    image, label, valid_mask, aug_k, aug_flip
                )

            image = image.contiguous()
            label = label.contiguous()
            valid_mask = valid_mask.contiguous()
            C, Ho, Wo = image.shape

            if ref_shape is None:
                ref_shape = (C, Ho, Wo)

            doy = month_to_doy(mi["month"])
            time_idx = self.look_up.get_or_register_time_idx(doy)

            tokens = self.token_builder.build_tokens(
                image=image, label=label, resolution=gsd,
                spectral_indices=self.spectral_indices["planet"],
                resolution_idx=res_idx, time_idx=time_idx,
            )
            token_mask = self._build_token_mask(valid_mask, C)

            all_tokens.append(tokens)
            all_masks.append(token_mask)

        groups = {gsd: {
            "tokens": torch.cat(all_tokens, dim=0),
            "mask": torch.cat(all_masks, dim=0),
            "shape": ref_shape,
        }}

        # Queries at Planet resolution (already native for Planet)
        anchor_label = self._get_planet_label(
            next(m for m in all_months if m["month"] == anchor_month)["label_path"],
            aoi, anchor_month,
        )
        Hl, Wl = anchor_label.shape
        ly = min(y0, max(0, Hl - self.crop_size))
        lx = min(x0, max(0, Wl - self.crop_size))
        anchor_label_crop = anchor_label[ly:ly+self.crop_size, lx:lx+self.crop_size]

        queries, query_gsd = self._build_planet_res_queries(
            anchor_label_crop, y0, x0, anchor_month, aug_k, aug_flip,
            first_spectral_idx=self.spectral_indices["planet"][0].item(),
        )

        return {
            "groups": groups,
            "tasks": {self.TASK_NAME: {
                "queries": queries,
                "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
            }},
            "target_resolution": query_gsd,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # LANDSAT MULTI-TEMPORAL
    # ═════════════════════════════════════════════════════════════════

    def _load_landsat_mt(self, aoi, anchor_month, all_months):
        gsd = _round_gsd(30.0)
        res_idx = self.resolution_indices[gsd]

        planet_geo = self._planet_geo_cache.get(aoi, {})
        H_planet = planet_geo.get("height", 1024)
        W_planet = planet_geo.get("width", 1024)

        planet_y0, planet_x0 = compute_crop_origin(
            H_planet, W_planet, self.crop_size, self.crop_size,
            random_crop=self.random_crop,
        )

        aug_k, aug_flip = self._get_d4_params() if self.augment else (0, False)

        all_tokens = []
        all_masks = []
        ref_shape = None

        for mi in all_months:
            image = self._load_landsat_image(mi["band_files"])
            _, Hi, Wi = image.shape

            sy, sx, sc = planet_origin_to_sensor(
                planet_y0, planet_x0, H_planet, W_planet,
                Hi, Wi, self.crop_size, gsd,
            )
            image = image[:, sy:sy+sc, sx:sx+sc]

            _, Hc, Wc = image.shape
            month_label = self._get_planet_label(mi["label_path"], aoi, mi["month"])
            label_crop = month_label[
                planet_y0:planet_y0+self.crop_size,
                planet_x0:planet_x0+self.crop_size,
            ]
            label = downsample_label_nearest(label_crop, Hc, Wc)

            image, label, valid_mask = pad_to_canonical(image, label, gsd)

            if self.augment:
                image, label, valid_mask = self._apply_d4(
                    image, label, valid_mask, aug_k, aug_flip
                )

            image = image.contiguous()
            label = label.contiguous()
            valid_mask = valid_mask.contiguous()
            C, Ho, Wo = image.shape

            if ref_shape is None:
                ref_shape = (C, Ho, Wo)

            doy = month_to_doy(mi["month"])
            time_idx = self.look_up.get_or_register_time_idx(doy)

            tokens = self.token_builder.build_tokens(
                image=image, label=label, resolution=gsd,
                spectral_indices=self.spectral_indices["landsat8"],
                resolution_idx=res_idx, time_idx=time_idx,
            )
            token_mask = self._build_token_mask(valid_mask, C)
            all_tokens.append(tokens)
            all_masks.append(token_mask)

        groups = {gsd: {
            "tokens": torch.cat(all_tokens, dim=0),
            "mask": torch.cat(all_masks, dim=0),
            "shape": ref_shape,
        }}

        # Queries at Planet resolution (high-res supervision from Landsat encoder)
        anchor_label_full = self._get_planet_label(
            next(m for m in all_months if m["month"] == anchor_month)["label_path"],
            aoi, anchor_month,
        )
        anchor_label_crop = anchor_label_full[
            planet_y0:planet_y0+self.crop_size,
            planet_x0:planet_x0+self.crop_size,
        ]

        queries, query_gsd = self._build_planet_res_queries(
            anchor_label_crop, planet_y0, planet_x0, anchor_month,
            aug_k, aug_flip,
            first_spectral_idx=self.spectral_indices["landsat8"][0].item(),
        )

        return {
            "groups": groups,
            "tasks": {self.TASK_NAME: {
                "queries": queries,
                "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
            }},
            "target_resolution": query_gsd,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # SENTINEL-2 MULTI-TEMPORAL
    # ═════════════════════════════════════════════════════════════════

    def _load_s2_mt(self, aoi, anchor_month, all_months):
        planet_geo = self._planet_geo_cache.get(aoi, {})
        H_planet = planet_geo.get("height", 1024)
        W_planet = planet_geo.get("width", 1024)

        planet_y0, planet_x0 = compute_crop_origin(
            H_planet, W_planet, self.crop_size, self.crop_size,
            random_crop=self.random_crop,
        )

        aug_k, aug_flip = self._get_d4_params() if self.augment else (0, False)

        group_tokens = defaultdict(list)
        group_masks = defaultdict(list)
        group_shapes = {}

        for mi in all_months:
            res_images = self._load_s2_images(mi["band_files"])
            doy = month_to_doy(mi["month"])
            time_idx = self.look_up.get_or_register_time_idx(doy)

            month_label = self._get_planet_label(mi["label_path"], aoi, mi["month"])
            month_label_crop = month_label[
                planet_y0:planet_y0+self.crop_size,
                planet_x0:planet_x0+self.crop_size,
            ]

            for gsd, group_image in res_images.items():
                _, Hg, Wg = group_image.shape

                sy, sx, sc = planet_origin_to_sensor(
                    planet_y0, planet_x0, H_planet, W_planet,
                    Hg, Wg, self.crop_size, gsd,
                )
                group_image = group_image[:, sy:sy+sc, sx:sx+sc]

                _, Hgc, Wgc = group_image.shape
                label_g = downsample_label_nearest(month_label_crop, Hgc, Wgc)

                group_image, label_g, valid_mask_g = pad_to_canonical(
                    group_image, label_g, gsd
                )

                if self.augment:
                    group_image, label_g, valid_mask_g = self._apply_d4(
                        group_image, label_g, valid_mask_g, aug_k, aug_flip
                    )

                group_image = group_image.contiguous()
                label_g = label_g.contiguous()
                valid_mask_g = valid_mask_g.contiguous()
                Cg, Hgo, Wgo = group_image.shape

                if gsd not in group_shapes:
                    group_shapes[gsd] = (Cg, Hgo, Wgo)

                res_idx = self.resolution_indices[gsd]
                tokens = self.token_builder.build_tokens(
                    image=group_image, label=label_g, resolution=gsd,
                    spectral_indices=self.s2_group_spectral_indices[gsd],
                    resolution_idx=res_idx, time_idx=time_idx,
                )
                token_mask = self._build_token_mask(valid_mask_g, Cg)

                group_tokens[gsd].append(tokens)
                group_masks[gsd].append(token_mask)

        groups = {}
        for gsd in group_tokens:
            groups[gsd] = {
                "tokens": torch.cat(group_tokens[gsd], dim=0),
                "mask": torch.cat(group_masks[gsd], dim=0),
                "shape": group_shapes[gsd],
            }

        # Queries at Planet resolution (high-res supervision from S2 encoder)
        anchor_label_full = self._get_planet_label(
            next(m for m in all_months if m["month"] == anchor_month)["label_path"],
            aoi, anchor_month,
        )
        anchor_label_crop = anchor_label_full[
            planet_y0:planet_y0+self.crop_size,
            planet_x0:planet_x0+self.crop_size,
        ]

        queries, query_gsd = self._build_planet_res_queries(
            anchor_label_crop, planet_y0, planet_x0, anchor_month,
            aug_k, aug_flip,
            first_spectral_idx=self.s2_group_spectral_indices[_round_gsd(10.0)][0].item(),
        )

        return {
            "groups": groups,
            "tasks": {self.TASK_NAME: {
                "queries": queries,
                "queries_mask": torch.zeros(queries.shape[0], dtype=torch.bool),
            }},
            "target_resolution": query_gsd,
            "dataset_name": DATASET_NAME,
        }

    # ═════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═════════════════════════════════════════════════════════════════

    def get_viz_sample(self, index):
        sample = self.samples[index]
        sensor = sample["sensor"]
        aoi = sample["aoi"]
        anchor_month = sample["month"]

        orig_augment = self.augment
        orig_random = self.random_crop
        self.augment = False
        self.random_crop = False

        try:
            all_months = self._select_context_months(aoi, sensor, anchor_month)
            if sensor == "planet":
                result = self._load_planet_mt(aoi, anchor_month, all_months)
            elif sensor == "sentinel2":
                result = self._load_s2_mt(aoi, anchor_month, all_months)
            elif sensor == "landsat8":
                result = self._load_landsat_mt(aoi, anchor_month, all_months)
        finally:
            self.augment = orig_augment
            self.random_crop = orig_random

        result["sensor"] = sensor
        result["aoi"] = aoi
        result["anchor_month"] = anchor_month
        result["all_months"] = [m["month"] for m in all_months]
        return result