"""
FRACTAL — Per-Class Voxel Reconstruction
==========================================

Reconstructs a mesh from a colored LiDAR point cloud using per-class
voxel occupancy + marching cubes. Unlike Poisson, this handles vertical
faces (building walls, bridge supports) naturally because it voxelizes
in 3D and surfaces the occupancy grid.

Pipeline per class:
    1. Extract points by class (from label PLY)
    2. Voxelize at class-appropriate resolution → binary occupancy grid
    3. Gaussian blur the grid (fills small gaps, smooths jagged voxels)
    4. Marching cubes on occupancy at iso=0.5
    5. Remove triangles too far from any real point (boundary cleanup)
    6. Transfer PCA colors from original points to mesh vertices (KNN)
    7. Save per-class .ply + merged .ply

Requirements:
    pip install scikit-image scipy numpy open3d

Usage
-----
    python script_voxel_reconstruct.py \
        --pca_ply   ./pca_outputs/fractal_pca/all/scene_00042.ply \
        --label_ply ./pca_outputs/fractal_pca/labels/scene_labels_00042.ply \
        --out_dir   ./meshes/scene_042

    # Tune voxel sizes if needed
    python script_voxel_reconstruct.py \
        --pca_ply   ./pca_outputs/fractal_pca/all/scene_00042.ply \
        --label_ply ./pca_outputs/fractal_pca/labels/scene_labels_00042.ply \
        --out_dir   ./meshes/scene_042 \
        --voxel_building 0.25 \
        --voxel_ground   0.5
"""

import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import open3d as o3d


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

FRACTAL_CLASSES = {
    0: "other",
    1: "ground",
    2: "vegetation",
    3: "building",
    4: "water",
    5: "bridge",
    6: "permanent_structure",
}

# Label colors — must match FRACTAL_LABEL_COLORS in script_pca_fractal.py
LABEL_COLORS = {
    0: np.array([128, 128, 128]),
    1: np.array([180, 120,  60]),
    2: np.array([ 34, 139,  34]),
    3: np.array([220,  50,  50]),
    4: np.array([ 30, 144, 255]),
    5: np.array([255, 165,   0]),
    6: np.array([148,   0, 211]),
}

# Per-class reconstruction config:
#   voxel_size:      metres per voxel. Smaller = more detail, more memory.
#   sigma:           Gaussian blur sigma in voxels. Higher = gap filling,
#                    smoother surface. 0 = no blur (sharp but holey).
#   max_dist:        Max distance (m) from mesh vertex to nearest real point.
#                    Triangles beyond this are artifact surfaces in empty space.
#   reconstruct:     False = skip mesh, save as point cloud instead.
CLASS_CONFIGS = {
    0: dict(reconstruct=False),
    1: dict(reconstruct=True,  voxel_size=0.5,  sigma=0.8, max_dist=1.5),   # ground
    2: dict(reconstruct=False),                                               # vegetation → points
    3: dict(reconstruct=True,  voxel_size=0.25, sigma=0.6, max_dist=0.8),   # building
    4: dict(reconstruct=True,  voxel_size=0.5,  sigma=0.8, max_dist=1.5),   # water
    5: dict(reconstruct=True,  voxel_size=0.25, sigma=0.6, max_dist=0.8),   # bridge
    6: dict(reconstruct=True,  voxel_size=0.25, sigma=0.6, max_dist=0.8),   # perm struct
}


# =============================================================================
# PLY I/O
# =============================================================================

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points,  dtype=np.float32)
    rgb = np.asarray(pcd.colors,  dtype=np.float32)   # [0, 1]
    return xyz, rgb


def save_mesh_ply(path, verts, faces, colors):
    """Save mesh as PLY with vertex colors."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(colors, 0, 1).astype(np.float64))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)
    print(f"  → {path}  ({len(verts):,} verts, {len(faces):,} tris)")
    return mesh


def save_pcd_ply(path, xyz, rgb):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)
    print(f"  → {path}  ({len(xyz):,} pts)")


# =============================================================================
# CLASS MASK FROM LABEL PLY
# =============================================================================

def extract_class_mask(label_rgb_float, class_id, tol=5):
    """
    Recover per-point class mask from label PLY vertex colors.
    label_rgb_float: [N, 3] float [0,1]  (Open3D convention)
    """
    uint8 = (label_rgb_float * 255).round().astype(np.int32)
    target = LABEL_COLORS[class_id].astype(np.int32)
    diff = np.abs(uint8 - target).max(axis=1)
    return diff <= tol


# =============================================================================
# VOXEL OCCUPANCY → MESH
# =============================================================================

def voxelize(xyz, voxel_size):
    """
    Convert point cloud to binary occupancy grid.

    Returns:
        grid:    bool array [Nx, Ny, Nz]
        origin:  np.array [3]  — world coords of voxel (0,0,0)
        vsize:   float          — voxel size
    """
    origin = xyz.min(axis=0) - voxel_size          # small margin
    indices = ((xyz - origin) / voxel_size).astype(np.int32)

    # Grid dimensions
    max_idx = indices.max(axis=0) + 2              # +2 margin for MC
    grid = np.zeros(max_idx, dtype=np.float32)

    # Mark occupied voxels
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return grid, origin, voxel_size


def occupancy_to_mesh(xyz, voxel_size, sigma, max_dist):
    """
    Full pipeline: points → voxels → blur → marching cubes → cleanup.

    Returns (verts, faces) in world coordinates, or (None, None) on failure.
    """
    # 1. Voxelize
    grid, origin, vsize = voxelize(xyz, voxel_size)
    print(f"    Grid: {grid.shape}  ({grid.sum():.0f} occupied voxels)")

    # 2. Gaussian blur to fill gaps and smooth surface
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    # 3. Marching cubes at iso=0.5
    # level=0.5 means "halfway between empty and full voxel"
    try:
        verts_vox, faces, normals, _ = marching_cubes(
            grid, level=0.5, spacing=(vsize, vsize, vsize)
        )
    except (ValueError, RuntimeError) as e:
        print(f"    Marching cubes failed: {e}")
        return None, None

    # 4. Convert voxel coords back to world coords
    verts_world = verts_vox + origin
    print(f"    Raw mesh: {len(verts_world):,} verts, {len(faces):,} tris")

    # 5. Remove triangles too far from any real point
    # These are artifact surfaces generated in empty space at class boundaries
    if max_dist > 0 and len(xyz) > 0:
        tree = cKDTree(xyz)
        # Check triangle centroids
        centroids = verts_world[faces].mean(axis=1)    # [F, 3]
        dists, _  = tree.query(centroids, k=1, workers=-1)
        keep_mask = dists <= max_dist
        faces     = faces[keep_mask]
        print(f"    After cleanup: {keep_mask.sum():,} tris "
              f"(removed {(~keep_mask).sum():,} artifact tris)")

        # Remove unreferenced vertices
        used_verts = np.unique(faces)
        remap = np.full(len(verts_world), -1, dtype=np.int32)
        remap[used_verts] = np.arange(len(used_verts))
        verts_world = verts_world[used_verts]
        faces = remap[faces]
        valid = (faces >= 0).all(axis=1)
        faces = faces[valid]

    if len(faces) == 0:
        print("    No faces remaining after cleanup.")
        return None, None

    return verts_world, faces


# =============================================================================
# COLOR TRANSFER
# =============================================================================

def transfer_colors(mesh_verts, src_xyz, src_rgb, k=3):
    """
    Inverse-distance weighted KNN color transfer.
    src_rgb: [N, 3] float [0, 1]
    Returns colors [V, 3] float [0, 1]
    """
    tree  = cKDTree(src_xyz)
    dists, idxs = tree.query(mesh_verts, k=min(k, len(src_xyz)), workers=-1)

    if k == 1 or dists.ndim == 1:
        return src_rgb[idxs]

    eps     = 1e-8
    weights = 1.0 / (dists + eps)
    weights /= weights.sum(axis=1, keepdims=True)
    colors  = np.einsum('vk,vkc->vc', weights, src_rgb[idxs])
    return np.clip(colors, 0, 1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-class voxel reconstruction for FRACTAL PLY")
    parser.add_argument("--pca_ply",         type=str, required=True)
    parser.add_argument("--label_ply",       type=str, required=True)
    parser.add_argument("--out_dir",         type=str, required=True)

    # Voxel size overrides per class (metres)
    parser.add_argument("--voxel_ground",    type=float, default=0.5)
    parser.add_argument("--voxel_building",  type=float, default=0.25)
    parser.add_argument("--voxel_water",     type=float, default=0.5)
    parser.add_argument("--voxel_bridge",    type=float, default=0.25)
    parser.add_argument("--voxel_perm",      type=float, default=0.25)

    # Sigma overrides (Gaussian blur in voxel units)
    parser.add_argument("--sigma_building",  type=float, default=0.6)
    parser.add_argument("--sigma_bridge",    type=float, default=0.6)

    # Color transfer neighbors
    parser.add_argument("--color_knn",       type=int, default=3)

    # Keep vegetation as point cloud
    parser.add_argument("--keep_vegetation", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Apply overrides
    CLASS_CONFIGS[1]["voxel_size"] = args.voxel_ground
    CLASS_CONFIGS[3]["voxel_size"] = args.voxel_building
    CLASS_CONFIGS[3]["sigma"]      = args.sigma_building
    CLASS_CONFIGS[4]["voxel_size"] = args.voxel_water
    CLASS_CONFIGS[5]["voxel_size"] = args.voxel_bridge
    CLASS_CONFIGS[5]["sigma"]      = args.sigma_bridge
    CLASS_CONFIGS[6]["voxel_size"] = args.voxel_perm

    # ── Load PLYs ─────────────────────────────────────────────────────
    print(f"\n[Voxel] Loading PCA PLY:   {args.pca_ply}")
    pca_xyz, pca_rgb = load_ply(args.pca_ply)
    print(f"        {len(pca_xyz):,} points")

    print(f"[Voxel] Loading label PLY: {args.label_ply}")
    lbl_xyz, lbl_rgb = load_ply(args.label_ply)
    print(f"        {len(lbl_xyz):,} points")

    assert len(pca_xyz) == len(lbl_xyz), (
        f"PCA PLY ({len(pca_xyz)}) and label PLY ({len(lbl_xyz)}) "
        f"must have the same number of points.")

    # ── Per-class reconstruction ──────────────────────────────────────
    all_verts  = []
    all_faces  = []
    all_colors = []
    vert_offset = 0

    for class_id, class_name in FRACTAL_CLASSES.items():
        cfg  = CLASS_CONFIGS[class_id]
        mask = extract_class_mask(lbl_rgb, class_id)
        n    = mask.sum()

        print(f"\n[{class_name}] {n:,} points", end="")

        if n < 50:
            print(" — too few, skipping")
            continue

        class_xyz = pca_xyz[mask]
        class_rgb = pca_rgb[mask]

        # Vegetation — save as point cloud, no mesh
        if not cfg["reconstruct"]:
            if class_id == 2 and args.keep_vegetation:
                print(" — saving as point cloud")
                path = os.path.join(args.out_dir,
                                    f"class_{class_id:02d}_{class_name}.ply")
                save_pcd_ply(path, class_xyz, class_rgb)
            else:
                print(" — skipping")
            continue

        print()

        # Voxel reconstruction
        verts, faces = occupancy_to_mesh(
            class_xyz,
            voxel_size=cfg["voxel_size"],
            sigma=cfg["sigma"],
            max_dist=cfg["max_dist"],
        )

        if verts is None:
            print(f"    Skipping {class_name} — reconstruction failed")
            continue

        # Color transfer
        print(f"    Transferring colors (k={args.color_knn})...")
        colors = transfer_colors(verts, class_xyz, class_rgb, k=args.color_knn)

        # Save per-class mesh
        path = os.path.join(args.out_dir,
                            f"class_{class_id:02d}_{class_name}.ply")
        save_mesh_ply(path, verts, faces, colors)

        # Accumulate for merged mesh
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        all_colors.append(colors)
        vert_offset += len(verts)

    # ── Merged mesh ───────────────────────────────────────────────────
    if all_verts:
        print(f"\n[Voxel] Merging {len(all_verts)} class meshes...")
        merged_verts  = np.concatenate(all_verts,  axis=0)
        merged_faces  = np.concatenate(all_faces,  axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        merged_path   = os.path.join(args.out_dir, "merged.ply")
        save_mesh_ply(merged_path, merged_verts, merged_faces, merged_colors)

    print(f"\n[Voxel] Done → {args.out_dir}/")
    print(f"  Tune --voxel_building / --sigma_building for sharper/smoother results.")
    print(f"  Lower voxel_size = more detail (but slower + more memory).")
    print(f"  Higher sigma = gap filling (but blurrier edges).")


if __name__ == "__main__":
    main()
