"""
Visualise Atomiser_Unet latents: RGB image + PCA of every hierarchy level.

Saves ONE image in ./figures with one row per sample (25 by default):

    RGB | E_0 | E_1 | ... | E_{S-1} | D_0 | ... | D_{S-2}

E_l are the features saved by the encoder, D_l the decoder states
(segmentation models only; D_{S-1} is the bottleneck itself, so it is not
repeated). Samples are drawn at random with --seed, or given with --indices.

Each panel paints every latent's Voronoi cell (its pixels, from the cached
hierarchy geometry) with the first 3 principal components of that latent's
feature vector, mapped to R, G, B. PCA is computed per level and per sample.
Both the RGB image and the PCA maps are built from the SAME token positions,
so they are pixel-aligned by construction.

Usage:
    python scripts_atomizer/viz_unet_pca.py \
        --checkpoint ./checkpoints/eurosat/ATOMIZER_Unet_xp-epoch=42-val_top1=0.9500.ckpt \
        --config ./training/configs/config_test-EUROSAT.yaml \
        --dataset eurosat --split test

    python scripts_atomizer/viz_unet_pca.py \
        --checkpoint ./checkpoints/ATOMIZER_Unetxp-val_loss-epoch=50-val_loss=0.1234.ckpt \
        --config ./training/configs/config_test-BURNSCARS.yaml \
        --dataset burnscars --split test --num 25 --borders

    # PASTIS checkpoint applied to Cashew (zero-shot features)
    python scripts_atomizer/viz_unet_pca.py \
        --checkpoint ./checkpoints/pastis/<pastis-unet-legacy>.ckpt \
        --config ./training/configs/config_test-PASTIS-unet-legacy.yaml \
        --trained_on pastis --use_s1 --dataset cashew --split test \
        --drop_bands B01 B09 --doy 180 --k 48 --m 40
    #   (k counts ALL atoms per pixel, masked ones included: 12 x 4 px = 48;
    #    m = the 10 kept bands x 4 px = 40 valid atoms per leaf)

    # PASTIS joint-encoding checkpoint applied to PLANTEUR (same bands, 6 dates)
    python scripts_atomizer/viz_unet_pca.py \
        --checkpoint ./checkpoints/pastis/<pastis-unet-legacy>.ckpt \
        --config ./training/configs/config_test-PASTIS-unet-legacy.yaml \
        --trained_on pastis --use_s1 --dataset planteur --split test --pca_scope both

    # PASTIS (joint temporal encoding; RGB = last image of the series)
    python scripts_atomizer/viz_unet_pca.py \
        --checkpoint ./checkpoints/pastis/pastis_pastis_unet_legacy-epoch=42-val_mIoU=0.5000.ckpt \
        --config ./training/configs/config_test-PASTIS-unet-legacy.yaml \
        --dataset pastis --use_s1 --split test

Works with checkpoints from Model_ForestNet (weights under "model.") and from
Model_BurnScars_Skip (weights under "encoder."), and with plain encoder
state_dicts.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Datasets
# =============================================================================
DATASETS = {
    "burnscars": ("training.utils.datasets.utils_dataset_BURNSCARS", "BurnScarsDataset",
                  "./data/hls_burn_scars"),
    "eurosat": ("training.utils.datasets.utils_dataset_Eurosat", "EurosatDataset",
                "./data/geo-bench-1.0/classification_v1.0/m-eurosat"),
    "eurosat_sar": ("training.utils.datasets.utils_dataset_EUROSAT_SAR", "EuroSATSARDataset",
                    "./data"),
    "pastis": ("training.utils.datasets.utils_dataset_PASTIS", "PastisHDDataset",
               "./data/PASTIS-HD"),
    "cashew": ("training.utils.datasets.utils_dataset_cashew", "CashewSkipDataset",
               "./data/geo-bench-1.0/segmentation_v1.0/m-cashew-plant"),
    "planteur": ("training.utils.datasets.utils_dataset_PLANTEUR", "PlanteurDataset",
                 "./data/PLANTEUR"),
    "agripotential": ("training.utils.datasets.utils_dataset_AGRIPOTENTIAL", "AgriPotentialDataset",
                      "./data/AGRIPOTENTIAL"),
}
PASTIS_FAMILY = ("pastis", "planteur", "agripotential")

# PASTIS: same lookup-table preparation as train_pastis.py, BEFORE the model is
# built (otherwise resolution / spectral indices differ from training).
PASTIS_RESOLUTIONS = {1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048}


def prepare_lookup_pastis(lookup_table, use_s1):
    """Same registration as training. Uses the shared registry
    (training/utils/resolution_registry.py) when present, so analysis tools and
    training always agree; falls back to the original PASTIS list otherwise."""
    try:
        try:
            from training.utils.resolution_registry import register_all_resolutions
        except ImportError:                       # file kept next to the scripts
            from resolution_registry import register_all_resolutions
        register_all_resolutions(lookup_table)
    except ImportError:
        print("[viz] WARNING: resolution_registry.py not found (training/utils/ or "
              "scripts_atomizer/): using the original PASTIS registration (no 5 m index). "
              "This must match how the checkpoint was trained.")
        from training.utils.datasets.token_builder import TokenBuilder
        for res, ref_size in PASTIS_RESOLUTIONS.items():
            TokenBuilder.REFERENCE_SIZES[res] = ref_size
            lookup_table.get_or_register_modality(res, ref_size)
            lookup_table.get_resolution_idx(res)
    if use_s1:
        lookup_table.register_abstract_channel("VV_VH")
RGB_BAND_NAMES = ("B04", "B03", "B02")          # red, green, blue


# =============================================================================
# Checkpoint loading
# =============================================================================
def extract_encoder_state(sd: dict, model_keys) -> dict:
    """Find the prefix under which the encoder weights are stored
    ("model.", "encoder.", or none) and strip it."""
    model_keys = set(model_keys)
    best, best_hits = "", -1
    for prefix in ("", "model.", "encoder.", "module.model.", "module.encoder.", "module."):
        hits = sum(1 for k in sd if k.startswith(prefix) and k[len(prefix):] in model_keys)
        if hits > best_hits:
            best, best_hits = prefix, hits
    return {k[len(best):]: v for k, v in sd.items() if k.startswith(best)}, best


def load_model(config, lookup_table, ckpt_path, device, random_init=False, seed=0):
    """random_init=True: the same architecture, built from the config with a fixed
    seed and NO checkpoint (control for what training contributes)."""
    if random_init:
        torch.manual_seed(seed)
    if config.get("Atomiser", {}).get("unet_datasets"):           # multi-task chain model
        from training.atomiser.Atomiser_unet_multi import Atomiser_UnetMulti
        model = Atomiser_UnetMulti(config=config, lookup_table=lookup_table)
    else:
        from training.atomiser.Atomiser_unet import Atomiser_Unet
        model = Atomiser_Unet(config=config, lookup_table=lookup_table)
    if random_init:
        print(f"[viz] RANDOM INITIALISATION (seed {seed}): no checkpoint loaded")
        return model.to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    enc_sd, prefix = extract_encoder_state(sd, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    print(f"[viz] loaded {ckpt_path} (prefix '{prefix}'): "
          f"{len(enc_sd) - len(unexpected)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        if missing:
            print(f"[viz] missing keys, e.g. {missing[:5]}")
        if unexpected:
            print(f"[viz] unexpected keys, e.g. {unexpected[:5]}")
        hint = ""
        if any(k.startswith("temporal.") for k in unexpected):
            hint = (" The checkpoint has temporal-transformer weights: it was trained with "
                    "unet_temporal: per_level, but --config builds a joint-encoding model.")
        elif any(k.startswith("temporal.") for k in missing):
            hint = (" --config builds a per-level temporal model, but the checkpoint has no "
                    "temporal-transformer weights (joint-encoding checkpoint).")
        raise RuntimeError("Checkpoint does not match the model built from --config." + hint +
                           " Use the config the checkpoint was trained with.")
    return model.to(device).eval()


# =============================================================================
# Batch + features
# =============================================================================
def to_batch(sample, device):
    """Single dataset sample -> batch of one (no collate needed)."""
    batch = {"groups": {}}
    for res, g in sample["groups"].items():
        batch["groups"][res] = {"tokens": g["tokens"].unsqueeze(0).to(device),
                                "mask": torch.as_tensor(g["mask"]).unsqueeze(0).to(device),
                                "shape": g["shape"]}
    if "queries" in sample:
        q, qm = sample["queries"], sample["queries_mask"]
    else:                                   # PASTIS: queries under sample["tasks"][<task>]
        task = next(iter(sample["tasks"].values()))
        q, qm = task["queries"], task["queries_mask"]
    batch["queries"] = q.unsqueeze(0).to(device)
    batch["queries_mask"] = torch.as_tensor(qm).unsqueeze(0).to(device)
    if "time_positions" in sample:
        batch["time_positions"] = torch.as_tensor(sample["time_positions"]).unsqueeze(0).to(device)
    return batch


@torch.no_grad()
def level_features(model, batch, km=None):
    """Run the model's own forward pieces, returning the geometry, the encoder
    features E_l and (segmentation) the decoder states D_l, all [L_l, D_l].
    km = (k, m) overrides the model's evaluation partition (cross-dataset use)."""
    if hasattr(model, "heads"):                              # multi-task chain model
        with torch.no_grad():
            out = model(batch, return_levels=True)
        tokens, mask = model._gather_inputs(batch)
        if tokens.dim() == 4:                                 # RGB / label: last valid date
            valid = torch.nonzero(~mask[0].all(-1)).squeeze(1)
            tokens = tokens[0, int(valid[-1]) if valid.numel() else -1]
        else:
            tokens = tokens[0]
        return (tokens, out["geom"], [e[0].float().cpu() for e in out["enc_levels"]],
                [d[0].float().cpu() for d in out["dec_levels"]])
    tokens, mask = model._gather_inputs(batch)
    k, m = km if km is not None else model.km_eval
    if tokens.dim() == 4:                                   # per-date model: [1, T, N, 8]
        return _temporal_level_features(model, batch, tokens, mask, k, m)
    tok_sel, _, bias = model.pruning(tokens, mask, None, geo_k=m, atoms_per_latent=k)
    geom = model.pruning.last_geometry
    z0 = model.leaf(tok_sel, bias, geom)
    z0 = z0 + model.empty_emb * model.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
    skips = model.unet.encode(z0, geom)
    enc = [s[0].float().cpu() for s in skips]
    dec = None
    if model.task == "segmentation":
        _, levels = model.unet.decode(skips, geom, return_levels=True)
        dec = [d[0].float().cpu() for d in levels]
    return tokens[0], geom, enc, dec


@torch.no_grad()
def _temporal_level_features(model, batch, tokens, mask, k, m):
    """unet_temporal: per_level. Encoder panels = the AGGREGATED maps (one per
    level, after the temporal transformers); decoder panels = the decoder on
    them. The returned tokens are those of the LAST VALID date (RGB / label)."""
    B, T, N, C = tokens.shape
    tok_sel, _, bias = model.pruning(tokens.reshape(B * T, N, C), mask.reshape(B * T, N), None,
                                     geo_k=m, atoms_per_latent=k)
    geom = model.pruning.last_geometry
    z0 = model.leaf(tok_sel, bias, geom)
    z0 = z0 + model.empty_emb * model.pruning.last_empty.unsqueeze(-1).to(z0.dtype)
    skips = model.unet.encode(z0, geom)
    skips_t = [s.view(B, T, s.shape[1], s.shape[2]) for s in skips]
    pad = mask.all(-1)                                                       # [B, T]
    tp = batch.get("time_positions", batch.get("time_indices", None))
    tp = None if tp is None else tp.to(z0.device).float()
    agg = model.temporal(skips_t, tp, pad)
    enc = [a[0].float().cpu() for a in agg]
    _, levels = model.unet.decode(agg, geom, return_levels=True)
    dec = [d[0].float().cpu() for d in levels]
    valid = torch.nonzero(~pad[0]).squeeze(1)
    t_last = int(valid[-1]) if valid.numel() else T - 1
    return tokens[0, t_last], geom, enc, dec


def flat_to_per_date(sample, T, C2, C1, lookup_table):
    """PASTIS-family FLAT sample -> per-date sample for unet_temporal: per_level.

    Flat layout (PastisHDDataset / PlanteurDataset): [S2 frame 0..T-1 | S1 frame
    0..T-1], each frame channel-major over the H*W pixels. Date t = S2 frame t +
    S1 frame t (C2 + C1 atoms per pixel). time_positions = the S2 day of year of
    each date (S2 anchor, as PastisHDTemporalDataset). NOTE: training pairs each
    S2 date with the nearest-calendar S1 date; here S1 frame t is used."""
    g = next(iter(sample["groups"].values()))
    tok, msk = g["tokens"], torch.as_tensor(g["mask"]).bool()
    H, W = g["shape"][-2:]
    HW = H * W
    n2, n1 = T * C2 * HW, T * C1 * HW
    if tok.shape[0] != n2 + n1:
        raise ValueError(f"flat_to_per_date: {tok.shape[0]} tokens, expected "
                         f"{T}x({C2}+{C1})x{HW} = {n2 + n1} (check multi_temporal / use_s1)")
    s2t, s2m = tok[:n2].view(T, C2 * HW, -1), msk[:n2].view(T, C2 * HW)
    if C1:                                                              # S2 frame t + S1 frame t
        tokens = torch.cat([s2t, tok[n2:].view(T, C1 * HW, -1)], 1)     # [T, N_date, 8]
        mask = torch.cat([s2m, msk[n2:].view(T, C1 * HW)], 1)
    else:                                                               # S2 only (AgriPotential)
        tokens, mask = s2t, s2m
    inv = {v: k for k, v in lookup_table.table_time.items()}            # time_idx -> day
    doy = [float(inv.get(int(s2t[t, 0, 7]), 0.0)) for t in range(T)]
    out = dict(sample)
    out["groups"] = {next(iter(sample["groups"])): {"tokens": tokens, "mask": mask,
                                                    "shape": g["shape"]}}
    out["time_positions"] = torch.tensor(doy, dtype=torch.float32)
    return out


def adapt_batch(batch, drop_spectral_ids=(), time_idx=None):
    """Cross-dataset use of a checkpoint:
    * mask the atoms of bands the model never saw in training (their learned
      embeddings are untrained) — masked atoms keep their positions and are
      skipped by the leaf sampler;
    * give date-less atoms (time_idx < 0) a fixed time index, for a model trained
      only on dated atoms (its 'no date' embedding was never trained)."""
    for g in batch["groups"].values():
        t = g["tokens"]
        if drop_spectral_ids:
            drop = torch.zeros(t.shape[:2], dtype=torch.bool, device=t.device)
            for sid in drop_spectral_ids:
                drop |= t[..., 3].long() == int(sid)
            g["mask"] = g["mask"].bool() | drop
        if time_idx is not None:
            t[..., 7] = torch.where(t[..., 7] < 0, torch.full_like(t[..., 7], float(time_idx)), t[..., 7])
    return batch


def spectral_ids_by_name(dataset, names):
    """Spectral indices of the given band names (B01, B09, ...) in this dataset."""
    band_names = [str(n).upper().replace("B8A", "B08A") for n in getattr(dataset, "band_names", [])]
    out = []
    for n in names:
        n = n.upper().replace("B8A", "B08A")
        if n not in band_names:
            raise ValueError(f"--drop_bands: {n} not in this dataset's bands {band_names}")
        out.append(int(dataset.spectral_indices[band_names.index(n)]))
    return out


def normalize_features(feat: torch.Tensor, mode: str) -> torch.Tensor:
    """[L, D] features before PCA.
    none        : raw residual-stream features (what the model stores)
    layernorm   : per latent, zero mean / unit variance over its channels
                  (what the model's next LayerNorm sees, without the affine)
    standardize : per channel, zero mean / unit variance over the latents, so
                  every channel counts equally in the PCA
    """
    if mode == "none":
        return feat
    if mode == "layernorm":
        return torch.nn.functional.layer_norm(feat, feat.shape[-1:])
    if mode == "standardize":
        return (feat - feat.mean(0, keepdim=True)) / feat.std(0, keepdim=True).clamp_min(1e-6)
    raise ValueError(f"unknown normalisation {mode!r}")


def pca_rgb(feat: torch.Tensor):
    """[L, D] -> [L, 3] colours in [0, 1] from the first 3 PCs, and the explained
    variance ratio of those 3 PCs."""
    x = feat - feat.mean(0, keepdim=True)
    q = min(3, x.shape[0], x.shape[1])
    U, S, V = torch.linalg.svd(x, full_matrices=False)
    pcs = x @ V[:q].T                                            # [L, q]
    ratio = (S[:q] ** 2).sum() / (S ** 2).sum().clamp_min(1e-12)
    if q < 3:
        pcs = torch.cat([pcs, torch.zeros(pcs.shape[0], 3 - q)], 1)
    lo = torch.quantile(pcs, 0.01, dim=0)
    hi = torch.quantile(pcs, 0.99, dim=0)
    rgb = ((pcs - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
    return rgb.numpy(), float(ratio)


class GlobalPCA:
    """One PCA basis for a feature column (one level), fitted on latents pooled
    over many images, so colours mean the same thing in every image.
    Normalisation uses GLOBAL statistics ("standardize": per-channel mean / std
    of the pooled latents; "layernorm": per latent, as before)."""

    def __init__(self, normalize):
        self.normalize = normalize

    def _prep(self, x):
        if self.normalize == "standardize":
            return (x - self.mu_c) / self.sd_c
        if self.normalize == "layernorm":
            return torch.nn.functional.layer_norm(x, x.shape[-1:])
        return x

    def fit(self, X):
        X = X.float()
        if self.normalize == "standardize":
            self.mu_c = X.mean(0, keepdim=True)
            self.sd_c = X.std(0, keepdim=True).clamp_min(1e-6)
        X = self._prep(X)
        self.mean = X.mean(0, keepdim=True)
        Xc = X - self.mean
        _, S, V = torch.linalg.svd(Xc, full_matrices=False)
        self.V = V[:3]
        self.ratio = float((S[:3] ** 2).sum() / (S ** 2).sum().clamp_min(1e-12))
        pcs = Xc @ self.V.T
        self.lo = torch.quantile(pcs, 0.01, dim=0)
        self.hi = torch.quantile(pcs, 0.99, dim=0)
        return self

    def transform(self, feat):
        """[L, D] -> [L, 3] colours, and the share of THIS image's variance
        captured by the 3 global components."""
        x = self._prep(feat.float()) - self.mean
        pcs = x @ self.V.T
        xi = x - x.mean(0, keepdim=True)
        share = float(((pcs - pcs.mean(0)) ** 2).sum() / (xi ** 2).sum().clamp_min(1e-12))
        rgb = ((pcs - self.lo) / (self.hi - self.lo).clamp_min(1e-8)).clamp(0, 1)
        return rgb.numpy(), share


# =============================================================================
# Rendering on the site grid
# =============================================================================
def site_grid(geom):
    """Integer (row, col) of every site, from the geometry's own positions."""
    ij = torch.floor((geom.site_xy.cpu() - geom.origin.cpu()) / geom.site_spacing).long()
    return ij[:, 0].numpy(), ij[:, 1].numpy(), int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1


def paint_cells(geom, level, colors):
    """Paint each site with the colour of the latent that owns it at `level`."""
    r, c, H, W = site_grid(geom)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    canvas[r, c] = colors[geom.owner[level].cpu().numpy()]
    return canvas


def upsample(a, f):
    """Nearest-neighbour upsampling of an [H, W] or [H, W, 3] array by factor f."""
    return a.repeat(f, axis=0).repeat(f, axis=1)


def cell_borders(geom, level, f):
    """Thin Voronoi borders on a canvas upsampled by f: [H*f, W*f] boolean."""
    r, c, H, W = site_grid(geom)
    own = np.full((H, W), -1)
    own[r, c] = geom.owner[level].cpu().numpy()
    own = upsample(own, f)
    b = np.zeros_like(own, dtype=bool)
    b[:, :-1] |= own[:, :-1] != own[:, 1:]
    b[:-1, :] |= own[:-1, :] != own[1:, :]
    return b


def rgb_from_tokens(tokens, geom, spectral_ids):
    """RGB image built from the tokens themselves (so it is aligned with the
    PCA maps): for each of R, G, B, the value of the atom with that spectral
    index at each site. Time series (several atoms per site and band): the
    LAST one in token order, i.e. the last image of the series (frames are
    concatenated chronologically). Percentile stretch per channel."""
    r, c, H, W = site_grid(geom)
    img = np.zeros((H, W, 3), dtype=np.float32)
    t = tokens.cpu()
    a2s = geom.atom_to_site.cpu()
    n_sites = len(r)
    for ch, sid in enumerate(spectral_ids):
        atoms = torch.nonzero(t[:, 3].long() == sid).squeeze(1)          # token order
        if atoms.numel() == 0:
            continue
        last = torch.full((n_sites,), -1, dtype=torch.long)
        last.scatter_reduce_(0, a2s[atoms], atoms, reduce="amax")       # latest atom per site
        ok = last >= 0
        sites = torch.nonzero(ok).squeeze(1).numpy()
        img[r[sites], c[sites], ch] = t[last[ok], 0].numpy()
    for ch in range(3):
        v = img[..., ch]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        img[..., ch] = np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)
    return img


def label_from_tokens(tokens, geom, ignore_index=255):
    """Segmentation label map built from the tokens' label column (col 4), on the
    same site grid as the RGB and PCA panels (aligned by construction).
    Returns an int array [H, W] (ignore_index where unlabelled), or None when
    the tokens carry no spatial label (e.g. classification datasets)."""
    r, c, H, W = site_grid(geom)
    t = tokens.cpu()
    lab = np.full((H, W), ignore_index, dtype=np.int64)
    a2s = geom.atom_to_site.cpu().numpy()
    vals = t[:, 4].long().numpy()
    lab[r[a2s], c[a2s]] = vals                      # all atoms of a site share its label
    if (lab != ignore_index).sum() == 0:
        return None
    return lab


@torch.no_grad()
def prediction_map(model, batch, tokens, geom):
    """Per-pixel class predictions on the site grid: one query per pixel (the
    first atom of each site, so the query sits exactly on the pixel), through
    the model's full forward pass. Returns an int array [H, W]."""
    from training.atomiser.Atomiser_unet import QueryDecoder
    _, site_atoms, _ = QueryDecoder.site_tables(geom)
    q = tokens[site_atoms[:, 0]].unsqueeze(0)                      # [1, n_sites, 8]
    b = dict(batch)
    b["queries"] = q
    b["queries_mask"] = torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device)
    pred = model(b)["predictions"][0].argmax(-1).cpu().numpy()     # [n_sites]
    r, c, H, W = site_grid(geom)
    out = np.full((H, W), 255, dtype=np.int64)
    out[r, c] = pred
    return out


def label_to_rgb(lab, num_classes, ignore_index=255):
    """Discrete colours (tab20, then tab20b), grey for ignore_index."""
    import matplotlib.cm as cm
    pal = np.concatenate([cm.get_cmap("tab20")(np.arange(20))[:, :3],
                          cm.get_cmap("tab20b")(np.arange(20))[:, :3]], 0)
    img = np.full(lab.shape + (3,), 0.5, dtype=np.float32)
    ok = lab != ignore_index
    img[ok] = pal[np.clip(lab[ok], 0, len(pal) - 1) % len(pal)]
    return img


def rgb_spectral_ids(dataset):
    """Spectral indices of B04, B03, B02 in this dataset (fallback: first 3 bands)."""
    if not hasattr(dataset, "band_names") and hasattr(dataset, "S2_BANDS_INFO") \
            and not hasattr(dataset, "s2_spectral_indices"):
        lk = dataset.look_up
        info = dataset.S2_BANDS_INFO
        return [int(lk.table_wave[(info[b]["bandwidth"], info[b]["central_wavelength"])])
                for b in ("B04", "B03", "B02")]
    if not hasattr(dataset, "band_names") and hasattr(dataset, "s2_spectral_indices"):
        # PASTIS: S2 bands ordered B02, B03, B04, ... (S2_BANDS_INFO idx 0, 1, 2)
        s2 = dataset.s2_spectral_indices
        return [int(s2[2]), int(s2[1]), int(s2[0])]
    names = [str(n).upper().replace("B8A", "B08A") for n in getattr(dataset, "band_names", [])]
    ids = dataset.spectral_indices
    out = []
    for want in RGB_BAND_NAMES:
        if want in names:
            out.append(int(ids[names.index(want)]))
    if len(out) < 3:
        print(f"[viz] WARNING: could not find {RGB_BAND_NAMES} in band_names {names}; "
              f"using the first three bands")
        out = [int(i) for i in ids[:3]]
    return out


# =============================================================================
# Figure: one row per sample
# =============================================================================
def make_grid_figure(rows, title, borders=False, transpose=False, panel=2.2, normalize="none",
                     pcas=None):
    """rows: list of dicts {label, rgb, geom, enc, dec}. One row per sample:
    RGB | E_0 ... E_{S-1} | D_0 ... D_{S-2} (decoder columns if present)."""
    S = rows[0]["geom"].num_levels
    has_dec = rows[0]["dec"] is not None
    has_lab = rows[0].get("lab") is not None
    has_pred = rows[0].get("pred") is not None
    dec_levels = list(range(S - 2, -1, -1)) if has_dec else []    # U order: D_{S-2} ... D_0
    cols = (["RGB (B04, B03, B02)"] + (["Label"] if has_lab else []) + (["Prediction"] if has_pred else [])
            + [f"E{l}" for l in range(S)] + [f"D{l}" for l in dec_levels])
    # (D_{S-1} is the bottleneck E_{S-1} itself, not drawn twice)
    n = len(rows)
    fig, axes = plt.subplots(n, len(cols), figsize=(panel * len(cols), panel * n + 1.0),
                             squeeze=False)
    T = (lambda a: a.transpose(1, 0, 2)) if transpose else (lambda a: a)

    for r, row in enumerate(rows):
        geom = row["geom"]
        panels = [row["rgb"]]
        notes = [""]
        for key in (["lab"] if has_lab else []) + (["pred"] if has_pred else []):
            panels.append(row[key] if row.get(key) is not None else np.full_like(row["rgb"], 0.5))
            notes.append("")
        feats = list(row["enc"]) + [row["dec"][l] for l in dec_levels]
        kinds = [("E", l) for l in range(S)] + [("D", l) for l in dec_levels]
        use_global = pcas is not None and row.get("scope", "global") == "global"
        for (kind, l), f in zip(kinds, feats):
            if use_global:                          # one basis per level, fitted on the split
                colors, ratio = pcas[(kind, l)].transform(f)
            else:
                colors, ratio = pca_rgb(normalize_features(f, normalize))
            img = paint_cells(geom, l, colors)
            if borders:
                up = max(1, 256 // max(img.shape[:2]))
                img = upsample(img, up)
                img[cell_borders(geom, l, up)] = 0.0
            panels.append(img)
            notes.append(f"{100 * ratio:.0f}%")
        for c, (img, note) in enumerate(zip(panels, notes)):
            ax = axes[r, c]
            ax.imshow(T(img), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if note:
                ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        axes[r, 0].set_ylabel(row["label"], fontsize=8)

    g0 = rows[0]["geom"]
    for c, name in enumerate(cols):
        if name[0] not in ("E", "D") or not name[1:].isdigit():
            axes[0, c].set_title(name, fontsize=9)
        else:
            l = int(name[1:])
            dim = (rows[0]["enc"] if name[0] == "E" else rows[0]["dec"])[l].shape[1]
            axes[0, c].set_title(f"{name}: {g0.sizes[l]} latents, {dim}-d", fontsize=9)
    scopes = {row.get("scope", "global" if pcas is not None else "image") for row in rows}
    if scopes == {"global", "image"}:
        sub = (f"   (rows: GLOBAL PCA per level, fitted on the split, then per-IMAGE PCA; "
               f"features: {normalize}; corner: global rows = share of the image's variance "
               f"on the 3 global PCs, image rows = variance explained by PC1-3)")
    elif pcas is not None:
        sub = (f"   (GLOBAL PCA per level, fitted on the split; features: {normalize}; "
               f"corner: share of the image's variance on the 3 global PCs)")
    else:
        sub = f"   (features: {normalize}; corner: variance explained by PC1-3)"
    fig.suptitle(title + sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / (panel * n + 1.0)))
    return fig


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="RGB + per-level PCA of Atomiser_Unet latents")
    ap.add_argument("--checkpoint", default=None,
                    help="trained checkpoint (not needed with --random_init)")
    ap.add_argument("--random_init", action="store_true",
                    help="control: same architecture, untrained (seeded), no checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--num", type=int, default=25, help="number of samples (one row each)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit sample indices (overrides --num / --seed)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random sample choice")
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--borders", action="store_true", help="draw Voronoi cell borders")
    ap.add_argument("--transpose", action="store_true",
                    help="swap the two image axes if the maps appear transposed")
    ap.add_argument("--no_decoder", action="store_true", help="encoder levels only")
    ap.add_argument("--no_label", action="store_true",
                    help="do not show the segmentation label column")
    ap.add_argument("--pca_scope", default="image", choices=("image", "global", "both"),
                    help="image: one PCA per image and level; global: one PCA per level "
                         "fitted on latents pooled over the split (colours comparable "
                         "across images); both: two rows per sample, global then image")
    ap.add_argument("--no_pred", action="store_true",
                    help="do not show the prediction column (segmentation)")
    ap.add_argument("--fit_num", type=int, default=0,
                    help="global PCA: images used for fitting (0 = the whole split)")
    ap.add_argument("--fit_rows", type=int, default=30000,
                    help="global PCA: max latents pooled per level (random subset per image)")
    ap.add_argument("--normalize", default="none", choices=["none", "layernorm", "standardize"],
                    help="feature normalisation before PCA (see normalize_features)")
    ap.add_argument("--use_s1", action="store_true",
                    help="PASTIS: S2 + S1 (must match how the checkpoint was trained)")
    # cross-dataset: apply a checkpoint trained on one dataset to another one
    ap.add_argument("--trained_on", default=None, choices=sorted(DATASETS),
                    help="dataset the checkpoint was trained on, if different from --dataset "
                         "(e.g. --trained_on pastis --dataset cashew): prepares the lookup "
                         "table as for that dataset, so band / resolution indices match")
    ap.add_argument("--split_protocol", default="default",
                    help="PLANTEUR: 'default' or 'loio:<island>'")
    ap.add_argument("--crop", default="viticulture", help="AgriPotential: target crop")
    ap.add_argument("--task", default=None,
                    help="multi-task model: which task (head, slice of the chain) to use; "
                         "default: --dataset")
    ap.add_argument("--res_as", type=float, default=None,
                    help="present the atoms' resolution as this GSD to the embedding (e.g. 10 for a "
                         "PASTIS model on 5 m data, whose 5 m embedding was never trained); the "
                         "geometry keeps the true scale through --k")
    ap.add_argument("--date", type=int, default=None,
                    help="time-series datasets (PASTIS family): keep only date t (0-based), for a "
                         "SINGLE-DATE model (e.g. trained on Cashew); default: all dates")
    ap.add_argument("--n_dates", type=int, default=None,
                    help="number of dates in each PASTIS-family sample (sets "
                         "dataset.multi_temporal; default: the config's value)")
    ap.add_argument("--no_time", action="store_true",
                    help="present every atom without a date (time_idx = -1), for a model "
                         "trained on date-less data, whose day embeddings were never trained")
    ap.add_argument("--temporal_window", default="2019", choices=("2019", "agri", "all"),
                    help="PLANTEUR: date window (default 2019)")
    ap.add_argument("--k", type=int, default=None,
                    help="atoms per leaf for THIS dataset, counting masked atoms too "
                         "(pixels per leaf x atoms per pixel; default: the model's km_eval)")
    ap.add_argument("--m", type=int, default=None,
                    help="atoms sampled per leaf (default: = --k)")
    ap.add_argument("--drop_bands", nargs="*", default=[],
                    help="bands to mask (e.g. B01 B09: absent from the training data)")
    ap.add_argument("--doy", type=int, default=None,
                    help="day of year given to date-less atoms (model trained on dated atoms)")
    ap.add_argument("--configs_dataset", default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
    ap.add_argument("--bands_yaml", default="./data/bands_info/bands.yaml")
    args = ap.parse_args()

    from training.utils import read_yaml
    from training.utils.Lookup_positional import Lookup_encoding

    config = read_yaml(args.config)
    if args.n_dates is not None:                       # before the dataset reads it
        config.setdefault("dataset", {})["multi_temporal"] = int(args.n_dates)
    lookup = Lookup_encoding(read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config)
    multi = bool(config.get("Atomiser", {}).get("unet_datasets"))
    if multi or (args.trained_on or args.dataset) in PASTIS_FAMILY:
        # before the model, as in training (the multi-task script always registers VV_VH)
        prepare_lookup_pastis(lookup, args.use_s1 or multi)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint is None and not args.random_init:
        raise SystemExit("--checkpoint is required (or use --random_init)")
    model = load_model(config, lookup, args.checkpoint, device,
                       random_init=args.random_init, seed=args.seed)

    mod_name, cls_name, default_root = DATASETS[args.dataset]
    DatasetClass = getattr(importlib.import_module(mod_name), cls_name)
    root = args.data_root or default_root
    collate = None
    if args.dataset == "pastis":
        if config.get("Atomiser", {}).get("unet_temporal", "none") == "per_level":
            # per-date model: the per-date dataset + its collate (time_positions)
            tm = importlib.import_module("training.utils.datasets.pastis_temporal_dataset")
            DatasetClass, collate = tm.PastisHDTemporalDataset, tm.pastis_temporal_collate_fn
            if not args.use_s1:
                raise ValueError("unet_temporal: per_level uses PastisHDTemporalDataset: add --use_s1")
            print("[viz] per-level temporal model: PastisHDTemporalDataset, panels = aggregated maps")
        dataset = DatasetClass(root_path=root, mode=args.split, config_model=config,
                               look_up=lookup, use_s1=args.use_s1, use_spot=False)
    elif args.dataset == "agripotential":
        dataset = DatasetClass(root_path=root, mode=args.split, config_model=config, look_up=lookup,
                               crop=args.crop,
                               n_frames=int(config.get("dataset", {}).get("multi_temporal", 6)))
    elif args.dataset == "planteur":
        if config.get("Atomiser", {}).get("unet_temporal", "none") == "per_level":
            print("[viz] per-level temporal model on PLANTEUR: flat samples are reshaped "
                  "per date (S2 frame t + S1 frame t)")
        dataset = DatasetClass(root_path=root, mode=args.split, config_model=config,
                               look_up=lookup, use_s1=args.use_s1,
                               split_protocol=args.split_protocol,
                               temporal_window=args.temporal_window)
    else:
        dataset = DatasetClass(root_path=root, mode=args.split,
                               dataset_config=read_yaml(args.bands_yaml),
                               config_model=config, look_up=lookup)
    spec_ids = rgb_spectral_ids(dataset)

    # cross-dataset adaptation
    km = None
    if args.k is not None:
        if multi:
            raise SystemExit("--k: the multi-task model uses each task's k from its config")
        km = (args.k, args.m if args.m is not None else args.k)
        model.pruning.k_choices = None                     # allow any k at inference
    task = args.task or args.dataset
    if multi and task not in model.datasets:
        raise SystemExit(f"--task {task!r} is not a task of this model: {list(model.datasets)}")
    res_idx = lookup.get_resolution_idx(args.res_as) if args.res_as is not None else None
    drop_ids = spectral_ids_by_name(dataset, args.drop_bands) if args.drop_bands else []
    time_idx = lookup.get_or_register_time_idx(args.doy) if args.doy is not None else None
    seen = None
    if args.trained_on == "pastis" and args.trained_on != args.dataset:
        # which spectral indices did the PASTIS model actually see in training?
        P = getattr(importlib.import_module(DATASETS["pastis"][0]), DATASETS["pastis"][1])
        infos = dict(P.S2_BANDS_INFO, **(P.S1_BANDS_INFO if args.use_s1 else {}))
        seen = {lookup.table_wave[(i["bandwidth"], i["central_wavelength"])]
                for i in infos.values() if (i["bandwidth"], i["central_wavelength"]) in lookup.table_wave}
    elif args.trained_on == "cashew" and args.trained_on != args.dataset:
        # the Cashew model's training bands: the spectral indices of a Cashew sample
        Cmod, Ccls, Croot = DATASETS["cashew"]
        C = getattr(importlib.import_module(Cmod), Ccls)
        try:
            cds = C(root_path=Croot, mode="train", dataset_config=read_yaml(args.bands_yaml),
                    config_model=config, look_up=lookup)
            if hasattr(cds, "spectral_indices"):
                seen = {int(x) for x in torch.as_tensor(cds.spectral_indices).tolist()}
            else:
                cs = cds[0]
                ct = next(iter(cs["groups"].values()))["tokens"]
                seen = {int(x) for x in torch.as_tensor(ct)[..., 3].long().unique().tolist()}
        except Exception as e:                                        # data not at the default path
            print(f"[viz] WARNING: could not read a Cashew sample to check bands ({e}); "
                  f"bands are NOT checked")
    if seen is not None:
        if hasattr(dataset, "s2_spectral_indices"):          # PASTIS family (PLANTEUR)
            pairs = list(zip(sorted(dataset.S2_BANDS_INFO, key=lambda b: dataset.S2_BANDS_INFO[b]["idx"]),
                             dataset.s2_spectral_indices.tolist()))
            if getattr(dataset, "use_s1", False) and hasattr(dataset, "s1_spectral_indices"):
                pairs += list(zip(sorted(dataset.S1_BANDS_INFO, key=lambda b: dataset.S1_BANDS_INFO[b]["idx"]),
                                  dataset.s1_spectral_indices.tolist()))
        else:
            pairs = list(zip(getattr(dataset, "band_names", []), dataset.spectral_indices.tolist()))
        unseen = [n for n, sid in pairs if int(sid) not in seen and int(sid) not in drop_ids]
        if unseen:
            print(f"[viz] WARNING: bands {unseen} have spectral indices never seen in "
                  f"{args.trained_on} training (untrained embeddings). Add them to --drop_bands, "
                  f"or check that both datasets define them identically.")
        else:
            print(f"[viz] all non-masked {args.dataset} bands map to spectral indices seen in "
                  f"{args.trained_on} training ✓")
    if args.trained_on and args.trained_on != args.dataset:
        print(f"[viz] checkpoint trained on {args.trained_on}, applied to {args.dataset}: "
              f"k,m={km or (model.datasets[task]['km_eval'] if multi else model.km_eval)}"
              f"{f' (task {task})' if multi else ''}  masked bands={args.drop_bands or 'none'}  "
              f"doy for date-less atoms={args.doy}  date={args.date}  no_time={args.no_time}")

    if args.indices is not None:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(dataset), size=min(args.num, len(dataset)),
                                    replace=False).tolist())

    if multi:
        per_date = args.dataset in PASTIS_FAMILY and model.datasets[task]["temporal"]
    else:
        per_date = (args.dataset in ("planteur", "agripotential")
                    and config.get("Atomiser", {}).get("unet_temporal", "none") == "per_level")
    T_cfg = int(config.get("dataset", {}).get("multi_temporal", 10))

    if args.date is not None and not (args.dataset in PASTIS_FAMILY):
        raise SystemExit("--date applies to time-series datasets (pastis, planteur, agripotential)")
    if args.date is not None and per_date:
        raise SystemExit("--date is for single-date models; this model processes the dates itself")

    def finish(b):
        if args.no_time:                                      # --no_time: atoms without a date
            for g in b["groups"].values():
                g["tokens"][..., 7] = -1.0
            b["queries"][..., 7] = -1.0
            b.pop("time_positions", None)
        if res_idx is not None:                               # --res_as
            for g in b["groups"].values():
                g["tokens"][..., 6] = float(res_idx)
            b["queries"][..., 6] = float(res_idx)
        if multi:
            b["dataset"] = task
        return b

    def make_batch(sample):
        if per_date or args.date is not None:
            sample = flat_to_per_date(sample, T_cfg, dataset.NUM_S2_BANDS,
                                      dataset.NUM_S1_BANDS if getattr(dataset, "use_s1", False) else 0,
                                      lookup)
        if args.date is not None:                             # --date: one date, single-date layout
            res, g = next(iter(sample["groups"].items()))
            if not 0 <= args.date < g["tokens"].shape[0]:
                raise SystemExit(f"--date {args.date}: samples have {g['tokens'].shape[0]} dates")
            sample = dict(sample)
            sample["groups"] = {res: dict(g, tokens=g["tokens"][args.date], mask=g["mask"][args.date])}
            sample.pop("time_positions", None)
        if collate is None:
            return finish(adapt_batch(to_batch(sample, device), drop_ids, time_idx))
        b = collate([sample])
        def mv(v):
            if isinstance(v, torch.Tensor):
                return v.to(device)
            if isinstance(v, dict):
                return {k2: mv(v2) for k2, v2 in v.items()}
            return v
        b = mv(b)
        if "queries" not in b and "tasks" in b:
            task = next(iter(b["tasks"].values()))
            b["queries"], b["queries_mask"] = task["queries"], task["queries_mask"]
        return finish(adapt_batch(b, drop_ids, time_idx))

    # ---- global PCA: fit one basis per level on latents pooled over the split
    pcas = None
    if args.pca_scope in ("global", "both"):
        fit_idx = list(range(len(dataset)))
        if args.fit_num and args.fit_num < len(fit_idx):
            fit_idx = sorted(np.random.default_rng(args.seed + 1).choice(
                len(dataset), size=args.fit_num, replace=False).tolist())
        per_img = max(1, int(np.ceil(args.fit_rows / len(fit_idx))))
        pool, g = None, torch.Generator().manual_seed(args.seed)
        for n, idx in enumerate(fit_idx):
            _, geom_f, enc_f, dec_f = level_features(model, make_batch(dataset[idx]), km)
            S_f = geom_f.num_levels
            cols_f = {("E", l): f for l, f in enumerate(enc_f)}
            if dec_f is not None and not args.no_decoder:
                cols_f.update({("D", l): dec_f[l] for l in range(S_f - 1)})
            if pool is None:
                pool = {key: [] for key in cols_f}
            for key, f in cols_f.items():
                sel = torch.randperm(f.shape[0], generator=g)[:per_img]
                pool[key].append(f[sel])
            if (n + 1) % 50 == 0:
                print(f"[viz] global PCA: {n + 1}/{len(fit_idx)} images")
        pcas = {key: GlobalPCA(args.normalize).fit(torch.cat(p, 0)) for key, p in pool.items()}
        print("[viz] global PCA fitted on", len(fit_idx), "images:",
              ", ".join(f"{k}{l}: {torch.cat(pool[(k, l)], 0).shape[0]} latents, "
                        f"PC1-3 {100 * pc.ratio:.0f}%" for (k, l), pc in pcas.items()))

    rows = []
    for idx in indices:
        sample = dataset[idx]
        batch = make_batch(sample)
        tokens, geom, enc, dec = level_features(model, batch, km)
        label = f"#{idx}"
        lab = sample.get("label")
        if lab is not None and torch.as_tensor(lab).dim() == 0:
            with torch.no_grad():
                pred = model(batch)["predictions"].argmax(-1).item()
            label += f"\nlabel {int(lab)} / pred {pred}"
        lab_img = None
        if not args.no_label:
            lab = label_from_tokens(tokens, geom)
            if lab is not None:
                n_cls = int(getattr(dataset, "NUM_CLASSES", lab[lab != 255].max() + 1))
                lab_img = label_to_rgb(lab, n_cls)
        pred_img = None
        cross = bool(args.trained_on and args.trained_on != args.dataset)
        if not args.no_pred and getattr(model, "task", "segmentation") == "segmentation" and not cross:
            pm = prediction_map(model, batch, tokens, geom)
            n_cls = int(getattr(dataset, "NUM_CLASSES", max(int(pm.max()), 1) + 1))
            pred_img = label_to_rgb(pm, n_cls)
        base = {"rgb": rgb_from_tokens(tokens, geom, spec_ids), "lab": lab_img, "pred": pred_img,
                "geom": geom, "enc": enc, "dec": None if args.no_decoder else dec}
        if args.pca_scope == "both":
            rows.append(dict(base, label=label + "\nglobal PCA", scope="global"))
            rows.append(dict(base, label=label + "\nimage PCA", scope="image"))
        else:
            rows.append(dict(base, label=label, scope=args.pca_scope))
        print(f"[viz] sample {idx} done")

    os.makedirs(args.out, exist_ok=True)
    ck = ("random_init" if args.random_init
          else os.path.splitext(os.path.basename(args.checkpoint))[0])
    title = f"{args.dataset} {args.split} — {ck}"
    if args.trained_on and args.trained_on != args.dataset:
        title += f"  [trained on {args.trained_on}]"
    fig = make_grid_figure(rows, title, borders=args.borders, transpose=args.transpose,
                           normalize=args.normalize, pcas=pcas)
    norm_tag = ("" if args.normalize == "none" else f"_{args.normalize}") + \
               {"image": "", "global": "_global", "both": "_global+image"}[args.pca_scope]
    cross_tag = f"_from_{args.trained_on}" if args.trained_on and args.trained_on != args.dataset else ""
    path = os.path.join(args.out, f"pca_{args.dataset}{cross_tag}_{args.split}_{ck}{norm_tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[viz] saved {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
