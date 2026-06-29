"""
FRACTAL Atomizer — PCA Feature Visualization
=============================================

Extracts decoder features (before the final MLP) from a trained FRACTAL
checkpoint across selected scenes, fits a global PCA on the pooled features,
then saves per-scene colored point clouds as .ply files for visualization
in MeshLab, CloudCompare, or Open3D.

Both ablations (all / lidar_only) are processed in a SINGLE dataloader pass.

Usage
-----
    # Run scene finder first:
    python script_find_good_scene.py --out_csv good_scenes.csv

    # Then run PCA on selected scenes:
    python script_pca_fractal.py \
        --ckpt_path ./checkpoints/fractal/best.ckpt \
        --xp_name fractal_pca \
        --scene_indices 42 137 891 \
        --fp16 \
        --out_dir ./pca_outputs
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from training.utils import read_yaml, Lookup_encoding, create_flairhub_bands_info
from training.utils.datasets.token_builder import TokenBuilder
from training.trainer_FRACTAL import Model_Fractal
from training.utils.datasets.utils_dataset_fractal_viz import FractalDatasetViz
from training.utils.datasets.token_grouping import collate_grouped

# =============================================================================
# FRACTAL SETUP  (mirrors training script exactly)
# =============================================================================

ALL_FRACTAL_RESOLUTIONS = {0.2: 2048}

def register_all_resolutions(lookup_table):
    for res, ref_size in ALL_FRACTAL_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)

def create_fractal_bands_info():
    return {
        "bands_fractal_irgb_info": {
            "NIR": {"bandwidth": 100, "central_wavelength": 833, "idx": 0},
            "R":   {"bandwidth":  90, "central_wavelength": 660, "idx": 1},
            "G":   {"bandwidth":  80, "central_wavelength": 559, "idx": 2},
            "B":   {"bandwidth":  80, "central_wavelength": 492, "idx": 3},
        },
    }

# =============================================================================
# MODALITY DROP  (same as test script)
# =============================================================================

def drop_vhr_from_batch(batch: dict, vhr_spectral_indices: set) -> dict:
    if not vhr_spectral_indices:
        return batch
    groups_out = {}
    for res, group in batch["groups"].items():
        tokens = group["tokens"].clone()
        mask   = group["mask"].clone().float()
        batched = tokens.dim() == 3
        spec_idx = tokens[:, :, 3] if batched else tokens[:, 3]
        drop = torch.zeros_like(spec_idx, dtype=torch.bool)
        for sid in vhr_spectral_indices:
            drop |= (spec_idx == sid)
        if batched:
            tokens[:, :, 0][drop] = 0.0
        else:
            tokens[:, 0][drop] = 0.0
        mask[drop] = 1.0
        groups_out[res] = {**group, "tokens": tokens, "mask": mask}
    return {**batch, "groups": groups_out}


class VHRDropDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vhr_spectral_indices):
        self.dataset = dataset
        self.vhr_spectral_indices = vhr_spectral_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return drop_vhr_from_batch(self.dataset[idx], self.vhr_spectral_indices)

# =============================================================================
# FEATURE EXTRACTION — single dataloader pass, both ablations simultaneously
# =============================================================================

def _forward_and_collect(model, batch, ignore_index, fp16, debug=False):
    """
    Run one forward pass with return_features=True.
    Returns list of per-scene dicts for every item in the batch.
    """
    if debug:
        q = batch["queries"]
        print("\n[DEBUG] query token shape:", q.shape)
        print("[DEBUG] first 5 points, all cols:")
        print(q[0, :5, :].cpu().numpy())
        print("[DEBUG] col0 (z_norm):  ", q[0, :, 0].min().item(), "→", q[0, :, 0].max().item())
        print("[DEBUG] col1 (px x):   ", q[0, :, 1].min().item(), "→", q[0, :, 1].max().item())
        print("[DEBUG] col2 (px y):   ", q[0, :, 2].min().item(), "→", q[0, :, 2].max().item())
        print("[DEBUG] col5 (raw x m):", q[0, :, 5].min().item(), "→", q[0, :, 5].max().item())
        print("[DEBUG] col6 (raw y m):", q[0, :, 6].min().item(), "→", q[0, :, 6].max().item())
        print("[DEBUG] col7 (raw z m):", q[0, :, 7].min().item(), "→", q[0, :, 7].max().item())

    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else torch.inference_mode()

    with ctx:
        out      = model.model(batch, training=False, return_features=True)
        features = out["features"]                                   # [B, M, D]
        if fp16:
            features = features.float()
        logits   = model.model.reconstruction_head(features)        # [B, M, C]
        preds    = logits.argmax(dim=-1)                             # [B, M]

    queries = batch["queries"]                                       # [B, M, 8]
    scenes  = []
    for b in range(features.shape[0]):
        labels_b = queries[b, :, 4].long().cpu().numpy()
        valid    = labels_b != ignore_index

        # Raw metric xyz from cols 5/6/7 (populated by FractalDatasetViz)
        x = queries[b, :, 5].cpu().numpy()[valid]   # Lambert-93 easting (m)
        y = queries[b, :, 6].cpu().numpy()[valid]   # Lambert-93 northing (m)
        z = queries[b, :, 7].cpu().numpy()[valid]   # elevation asl (m)
        # Center x/y around scene origin — absolute Lambert-93 coords are
        # in the hundreds of thousands which causes float32 precision issues
        # in most 3D viewers. Z is kept relative to scene min so vertical
        # structure is preserved (buildings, bridges above ground).
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.min()
        xyz = np.stack([x, y, z], axis=-1)          # [N, 3]

        scenes.append({
            "xyz":      xyz,
            "features": features[b].cpu().numpy()[valid],
            "labels":   labels_b[valid],
            "preds":    preds[b].cpu().numpy()[valid],
        })
    return scenes


@torch.inference_mode()
def extract_features_dual(model, dataloader, device, vhr_spectral_indices,
                          ablations, n_scenes, ignore_index=255, fp16=False,
                          scene_indices=None):
    """
    Single dataloader pass — each batch is forwarded once per requested ablation.
    If scene_indices is set, only those dataset indices are processed.

    Returns:
        all_ablation_scenes: dict[ablation_name -> list[scene_dict]]
        collected_indices:   list[int]  — actual dataset indices collected
    """
    run_all        = "all"        in ablations
    run_lidar_only = "lidar_only" in ablations

    target_set  = set(scene_indices) if scene_indices else None
    max_scenes  = len(scene_indices) if scene_indices else n_scenes

    all_ablation_scenes = {a: [] for a in ablations}
    collected_indices   = []
    global_idx          = 0
    first_batch         = True
    model.eval()

    for batch in tqdm(dataloader, desc="Extracting (dual ablation)"):
        B = batch["queries"].shape[0]

        # Check done
        if all(len(all_ablation_scenes[a]) >= max_scenes for a in ablations):
            break

        # Scene index filtering — only works cleanly with batch_size=1
        if target_set is not None:
            if B == 1:
                if global_idx not in target_set:
                    global_idx += 1
                    continue
            # With B>1 we can't easily filter mid-batch; just process all
            # (user should use batch_size=1 with scene_indices)

        batch = _batch_to_device(batch, device)

        # ── ablation: all ──────────────────────────────────────────────
        if run_all and len(all_ablation_scenes["all"]) < max_scenes:
            for scene in _forward_and_collect(model, batch, ignore_index, fp16,
                                              debug=first_batch):
                if len(all_ablation_scenes["all"]) < max_scenes:
                    all_ablation_scenes["all"].append(scene)
                    if "all" == ablations[0]:
                        collected_indices.append(global_idx)

        # ── ablation: lidar_only ───────────────────────────────────────
        if run_lidar_only and len(all_ablation_scenes["lidar_only"]) < max_scenes:
            batch_dropped = drop_vhr_from_batch(batch, vhr_spectral_indices)
            for scene in _forward_and_collect(model, batch_dropped, ignore_index, fp16,
                                              debug=False):
                if len(all_ablation_scenes["lidar_only"]) < max_scenes:
                    all_ablation_scenes["lidar_only"].append(scene)

        first_batch = False
        global_idx += B

    for a, scenes in all_ablation_scenes.items():
        n_pts = sum(s["features"].shape[0] for s in scenes)
        print(f"[Extract] {a}: {len(scenes)} scenes, {n_pts:,} points")

    return all_ablation_scenes, collected_indices

# =============================================================================
# GLOBAL PCA
# =============================================================================

def fit_global_pca(scenes, n_components=3, batch_size=50_000):
    """
    Fit IncrementalPCA on all features pooled across scenes.
    IncrementalPCA avoids loading everything into RAM at once.
    """
    print(f"\n[PCA] Fitting global PCA (n_components={n_components}) "
          f"on {len(scenes)} scenes...")

    ipca = IncrementalPCA(n_components=n_components)

    # Partial fit — feed in chunks
    buffer = []
    buffer_size = 0

    for scene in tqdm(scenes, desc="PCA partial_fit"):
        feat = scene["features"]   # [N, D]
        buffer.append(feat)
        buffer_size += feat.shape[0]

        if buffer_size >= batch_size:
            chunk = np.concatenate(buffer, axis=0)
            ipca.partial_fit(chunk)
            buffer = []
            buffer_size = 0

    # Flush remainder (must have at least n_components samples)
    if buffer_size >= n_components:
        chunk = np.concatenate(buffer, axis=0)
        ipca.partial_fit(chunk)

    var_explained = ipca.explained_variance_ratio_.sum() * 100
    print(f"[PCA] Done. Variance explained by {n_components} components: "
          f"{var_explained:.1f}%")
    return ipca


def apply_pca_to_scenes(scenes, ipca, percentile_clip=(1, 99)):
    """
    Project all scene features with the fitted PCA, then map to RGB [0,255].

    Clipping is applied globally across all scenes so the color space is
    consistent: same feature value → same color in every scene.
    """
    # First pass: collect all projected values to compute global percentiles
    all_projected = []
    for scene in scenes:
        proj = ipca.transform(scene["features"])   # [N, 3]
        scene["_proj"] = proj
        all_projected.append(proj)

    all_proj = np.concatenate(all_projected, axis=0)   # [total_N, 3]

    # Per-component percentile clipping
    p_lo = np.percentile(all_proj, percentile_clip[0], axis=0)   # [3]
    p_hi = np.percentile(all_proj, percentile_clip[1], axis=0)   # [3]

    print(f"[PCA] Color clipping: lo={p_lo.round(3)}, hi={p_hi.round(3)}")

    for scene in scenes:
        proj = scene.pop("_proj")
        # Clip and normalize to [0, 1]
        proj_clipped = np.clip(proj, p_lo, p_hi)
        proj_norm    = (proj_clipped - p_lo) / (p_hi - p_lo + 1e-8)
        # Map to uint8
        scene["rgb"] = (proj_norm * 255).astype(np.uint8)

    return scenes

# =============================================================================
# PLY EXPORT
# =============================================================================

def save_ply(path, xyz, rgb):
    """
    Write a colored point cloud to an ASCII PLY file.
    xyz: [N, 3] float32,  rgb: [N, 3] uint8
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def save_scenes_ply(scenes, out_dir, prefix="scene", scene_indices=None):
    os.makedirs(out_dir, exist_ok=True)
    for i, scene in enumerate(scenes):
        tag  = scene_indices[i] if scene_indices else i
        path = os.path.join(out_dir, f"{prefix}_{tag:05d}.ply")
        save_ply(path, scene["xyz"], scene["rgb"])
        print(f"  → {path}  ({scene['xyz'].shape[0]:,} pts)")
    print(f"[PLY] {len(scenes)} files saved to {out_dir}/")


# =============================================================================
# SAVE LABELS PLY (ground truth colors for comparison)
# =============================================================================

FRACTAL_LABEL_COLORS = np.array([
    [128, 128, 128],   # 0 other        — grey
    [180, 120,  60],   # 1 ground       — brown
    [ 34, 139,  34],   # 2 vegetation   — forest green
    [220,  50,  50],   # 3 building     — red
    [ 30, 144, 255],   # 4 water        — dodger blue
    [255, 165,   0],   # 5 bridge       — orange
    [148,   0, 211],   # 6 perm struct  — purple
], dtype=np.uint8)


def save_label_ply(scenes, out_dir, prefix="scene_labels", scene_indices=None):
    """Save a second PLY with ground-truth class colors for side-by-side comparison."""
    os.makedirs(out_dir, exist_ok=True)
    for i, scene in enumerate(scenes):
        tag    = scene_indices[i] if scene_indices else i
        path   = os.path.join(out_dir, f"{prefix}_{tag:05d}.ply")
        labels = scene["labels"].astype(np.int32)
        labels = np.clip(labels, 0, len(FRACTAL_LABEL_COLORS) - 1)
        rgb    = FRACTAL_LABEL_COLORS[labels]
        save_ply(path, scene["xyz"], rgb)
    print(f"[PLY] {len(scenes)} label files saved to {out_dir}/")

# =============================================================================
# QUICK MATPLOTLIB PREVIEW
# =============================================================================

def preview_scene(scene, title="PCA feature colors", max_points=50_000):
    """
    Quick scatter-plot preview of a single scene (x-y plane, colored by PCA).
    Call this interactively — not used in the main script.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Preview] matplotlib not available.")
        return

    xyz = scene["xyz"]
    rgb = scene["rgb"].astype(float) / 255.0

    # Subsample for speed
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top-down (x-y)
    axes[0].scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=0.5, linewidths=0)
    axes[0].set_title(f"{title} — top-down (XY)")
    axes[0].set_aspect("equal")

    # Side view (x-z)
    axes[1].scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=0.5, linewidths=0)
    axes[1].set_title(f"{title} — side (XZ)")
    axes[1].set_aspect("equal")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# =============================================================================
# UTILS
# =============================================================================

def _batch_to_device(batch, device):
    """Recursively move all tensors in a nested dict/list to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_batch_to_device(v, device) for v in batch)
    return batch

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FRACTAL PCA feature visualization")
    parser.add_argument("--ckpt_path",           type=str, required=True)
    parser.add_argument("--xp_name",             type=str, required=True)
    parser.add_argument("--config_model",        type=str,
                        default="config_test-FRACTAL.yaml")
    parser.add_argument("--dataset_name",        type=str, default="u_regular")
    parser.add_argument("--root_path",           type=str, default="./data")
    parser.add_argument("--max_lidar_points",    type=int, default=None,
                        help="Max LiDAR points per scene. Omit for no cap "
                             "(recommended for viz — use all available points).")
    parser.add_argument("--max_encoder_lidar",   type=int, default=100_000,
                        help="Max LiDAR tokens fed to the ENCODER. "
                             "Decoder always sees all points. Default 100k.")
    parser.add_argument("--valid_patches_file",  type=str, default=None)
    parser.add_argument("--num_workers",         type=int, default=4)
    parser.add_argument("--batch_size",          type=int, default=4,
                        help="Scenes per forward pass. 4 is a good default.")
    parser.add_argument("--fp16",                action="store_true",
                        help="Use fp16 inference (~2x faster on Ampere GPUs).")
    parser.add_argument("--n_scenes",            type=int, default=50,
                        help="Number of test scenes for PCA fitting (ignored "
                             "if --scene_indices is set).")
    parser.add_argument("--scene_indices",       type=int, nargs="+", default=None,
                        help="Specific dataset indices to visualize (from "
                             "good_scenes.csv). Overrides --n_scenes for output; "
                             "PCA is still fit on --n_scenes scenes first.")
    parser.add_argument("--n_pca_components",    type=int, default=3)
    parser.add_argument("--percentile_clip",     type=float, nargs=2,
                        default=[1, 99],
                        help="Percentile clipping for PCA color mapping.")
    parser.add_argument("--out_dir",             type=str,
                        default="./pca_outputs")
    parser.add_argument("--ablations",           type=str, nargs="+",
                        default=["all", "lidar_only"],
                        choices=["all", "lidar_only"])
    parser.add_argument("--ignore_index",        type=int, default=255)
    parser.add_argument("--device",             type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[PCA] Using device: {device}")

    # ── Config + lookup ───────────────────────────────────────────────
    config_model    = read_yaml(f"./training/configs/{args.config_model}")
    configs_dataset = read_yaml(
        f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
    )

    fractal_bands = create_fractal_bands_info()
    flair_bands   = create_flairhub_bands_info()
    bands         = {**flair_bands, **fractal_bands}

    lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
    register_all_resolutions(lookup_table)
    lookup_table.register_abstract_channel("ELEVATION")
    lookup_table.register_abstract_channel("VV")
    lookup_table.register_abstract_channel("VH")
    lookup_table.register_abstract_channel("DSM")
    lookup_table.register_abstract_channel("DTM")

    # ── VHR spectral indices ──────────────────────────────────────────
    vhr_spectral_indices = set()
    for name, data in fractal_bands["bands_fractal_irgb_info"].items():
        key = (int(data["bandwidth"]), int(data["central_wavelength"]))
        if key in lookup_table.table_wave:
            vhr_spectral_indices.add(lookup_table.table_wave[key])
        else:
            raise KeyError(f"VHR band '{name}' key={key} not found in lookup table.")
    print(f"[PCA] VHR spectral indices: {sorted(vhr_spectral_indices)}")

    # ── Load model ────────────────────────────────────────────────────
    print(f"\n[PCA] Loading checkpoint: {args.ckpt_path}")
    model = Model_Fractal(
        config=config_model,
        wand=False,
        name=args.xp_name,
        transform=None,
        lookup_table=lookup_table,
        ignore_index=args.ignore_index,
        class_weights=None,
    )
    ckpt   = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state  = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[PCA] Missing: {len(result.missing_keys)}, "
          f"Unexpected: {len(result.unexpected_keys)}")
    model.eval()
    model.to(device)
    if args.fp16:
        model.half()
        print("[PCA] fp16 inference enabled")

    # ── Base dataset (single loader for both ablations) ───────────────
    base_ds = FractalDatasetViz(
        root_path=args.root_path,
        mode="test",
        dataset_config=bands,
        config_model=config_model,
        look_up=lookup_table,
        max_lidar_points=args.max_lidar_points,
        valid_patches_file=args.valid_patches_file,
        use_augmentation=False,
        max_encoder_lidar=args.max_encoder_lidar,
    )

    pca_path = os.path.join(args.out_dir, args.xp_name, "global_pca.pkl")
    pca_cached = os.path.exists(pca_path)

    loader = None
    if not pca_cached:
        loader = DataLoader(
            base_ds,
            batch_size=1 if args.scene_indices else args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_grouped,
            pin_memory=(device.type == "cuda"),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    # ── Step 1: fit PCA (or load from cache) ─────────────────────────
    import pickle
    pca_path = os.path.join(args.out_dir, args.xp_name, "global_pca.pkl")

    if os.path.exists(pca_path):
        print(f"\n[PCA] Step 1: loading cached PCA from {pca_path}")
        with open(pca_path, "rb") as f:
            pca_data = pickle.load(f)
        ipca = pca_data["ipca"]
        p_lo = pca_data["p_lo"]
        p_hi = pca_data["p_hi"]
        print(f"[PCA] Loaded. Clip: lo={p_lo.round(3)}, hi={p_hi.round(3)}")
    else:
        print(f"\n[PCA] Step 1: extracting {args.n_scenes} scenes for PCA fitting...")
        pca_scenes_dict, _ = extract_features_dual(
            model=model,
            dataloader=loader,
            device=device,
            vhr_spectral_indices=vhr_spectral_indices,
            ablations=["all"],
            n_scenes=args.n_scenes,
            ignore_index=args.ignore_index,
            fp16=args.fp16,
            scene_indices=None,
        )
        pca_scenes = pca_scenes_dict["all"]
        ipca = fit_global_pca(pca_scenes, n_components=args.n_pca_components,
                              batch_size=50_000)
        all_proj_fit = np.concatenate(
            [ipca.transform(s["features"]) for s in pca_scenes], axis=0)
        p_lo = np.percentile(all_proj_fit, args.percentile_clip[0], axis=0)
        p_hi = np.percentile(all_proj_fit, args.percentile_clip[1], axis=0)
        print(f"[PCA] Global clip: lo={p_lo.round(3)}, hi={p_hi.round(3)}")
        del pca_scenes

        os.makedirs(os.path.dirname(pca_path), exist_ok=True)
        with open(pca_path, "wb") as f:
            pickle.dump({"ipca": ipca, "p_lo": p_lo, "p_hi": p_hi}, f)
        print(f"[PCA] Saved to {pca_path}")

    # ── Step 2: extract target scenes ─────────────────────────────────
    loader2 = DataLoader(
        base_ds,
        batch_size=1 if args.scene_indices else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_grouped,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    all_ablation_scenes, collected_indices = extract_features_dual(
        model=model,
        dataloader=loader2,
        device=device,
        vhr_spectral_indices=vhr_spectral_indices,
        ablations=args.ablations,
        n_scenes=args.n_scenes,
        ignore_index=args.ignore_index,
        fp16=args.fp16,
        scene_indices=args.scene_indices,
    )

    # ── Step 3: color and save PLY ────────────────────────────────────
    scene_tags = collected_indices or args.scene_indices
    for ablation, scenes in all_ablation_scenes.items():
        print(f"\n[PLY] Coloring + saving: {ablation}")
        for scene in scenes:
            proj         = ipca.transform(scene["features"])
            proj_clipped = np.clip(proj, p_lo, p_hi)
            proj_norm    = (proj_clipped - p_lo) / (p_hi - p_lo + 1e-8)
            scene["rgb"] = (proj_norm * 255).astype(np.uint8)
            del scene["features"]

        out_dir = os.path.join(args.out_dir, args.xp_name, ablation)
        save_scenes_ply(scenes, out_dir, prefix="scene",
                        scene_indices=scene_tags)

        if ablation == args.ablations[0]:
            label_dir = os.path.join(args.out_dir, args.xp_name, "labels")
            save_label_ply(scenes, label_dir, prefix="scene_labels",
                           scene_indices=scene_tags)

    print(f"\n[PCA] Done → {os.path.join(args.out_dir, args.xp_name)}/")
    print(f"      Open .ply files in CloudCompare or MeshLab")


if __name__ == "__main__":
    main()
