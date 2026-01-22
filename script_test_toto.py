import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def linear_probe_latent_positions(
    spatial_latents: np.ndarray,  # [625, latent_dim]
    coordinates: np.ndarray,       # [625, 2] in meters
):
    """
    Test if learned latent embeddings encode their spatial position.
    """
    latents = spatial_latents.copy()
    coords = coordinates.copy()
    
    num_latents, latent_dim = latents.shape
    print(f"Latents: {latents.shape}, Coordinates: {coords.shape}")
    print(f"Coord range X: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}] meters")
    print(f"Coord range Y: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}] meters")
    
    # =========================================================================
    # BASIC STATS - Check if latents are degenerate
    # =========================================================================
    print(f"\n{'='*60}")
    print("LATENT STATISTICS")
    print(f"{'='*60}")
    print(f"Latent mean: {latents.mean():.4f}")
    print(f"Latent std:  {latents.std():.4f}")
    print(f"Latent min:  {latents.min():.4f}")
    print(f"Latent max:  {latents.max():.4f}")
    
    # Check variance per latent
    latent_variances = latents.var(axis=1)
    print(f"Per-latent variance: mean={latent_variances.mean():.4f}, std={latent_variances.std():.4f}")
    print(f"  min variance: {latent_variances.min():.4f}, max: {latent_variances.max():.4f}")
    
    # Check if all latents are identical
    pairwise_diffs = np.abs(latents[0:1] - latents[1:]).mean()
    print(f"Mean abs diff between latent[0] and others: {pairwise_diffs:.6f}")
    
    # =========================================================================
    # COSINE SIMILARITY MATRIX
    # =========================================================================
    # Normalize latents
    latents_norm = latents / (np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity matrix
    cos_sim = latents_norm @ latents_norm.T  # [625, 625]
    
    print(f"\n{'='*60}")
    print("COSINE SIMILARITY STATS")
    print(f"{'='*60}")
    # Off-diagonal stats
    mask = ~np.eye(num_latents, dtype=bool)
    off_diag = cos_sim[mask]
    print(f"Off-diagonal cosine sim: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    print(f"  min={off_diag.min():.4f}, max={off_diag.max():.4f}")
    
    if off_diag.std() < 0.01:
        print("⚠️  WARNING: All latents are nearly identical (very low variance in similarity)!")
    if off_diag.mean() > 0.99:
        print("⚠️  WARNING: All latents are almost the same vector!")
    
    # =========================================================================
    # LINEAR PROBE
    # =========================================================================
    probe = Ridge(alpha=1.0)
    scores_x = cross_val_score(probe, latents, coords[:, 0], cv=5, scoring='r2')
    scores_y = cross_val_score(probe, latents, coords[:, 1], cv=5, scoring='r2')
    
    print(f"\n{'='*60}")
    print("LINEAR PROBE RESULTS (5-fold CV)")
    print(f"{'='*60}")
    print(f"Predict X from embedding: R² = {scores_x.mean():.3f} ± {scores_x.std():.3f}")
    print(f"Predict Y from embedding: R² = {scores_y.mean():.3f} ± {scores_y.std():.3f}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Cosine similarity matrix (raw order)
    ax = axes[0, 0]
    im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Cosine Similarity Matrix\n(index order)\noff-diag mean={off_diag.mean():.3f}')
    ax.set_xlabel('Latent index')
    ax.set_ylabel('Latent index')
    plt.colorbar(im, ax=ax)
    
    # 2. Cosine similarity matrix (sorted by X coordinate)
    ax = axes[0, 1]
    sort_by_x = np.argsort(coords[:, 0])
    cos_sim_sorted_x = cos_sim[sort_by_x][:, sort_by_x]
    im = ax.imshow(cos_sim_sorted_x, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Cosine Similarity\n(sorted by X position)')
    ax.set_xlabel('Latent (sorted by X)')
    ax.set_ylabel('Latent (sorted by X)')
    plt.colorbar(im, ax=ax)
    
    # 3. Cosine similarity matrix (sorted by Y coordinate)
    ax = axes[0, 2]
    sort_by_y = np.argsort(coords[:, 1])
    cos_sim_sorted_y = cos_sim[sort_by_y][:, sort_by_y]
    im = ax.imshow(cos_sim_sorted_y, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Cosine Similarity\n(sorted by Y position)')
    ax.set_xlabel('Latent (sorted by Y)')
    ax.set_ylabel('Latent (sorted by Y)')
    plt.colorbar(im, ax=ax)
    
    # 4. Similarity vs spatial distance scatter
    ax = axes[1, 0]
    spa_dists = squareform(pdist(coords, metric='euclidean'))
    triu_idx = np.triu_indices(num_latents, k=1)
    spa_flat = spa_dists[triu_idx]
    sim_flat = cos_sim[triu_idx]
    
    subsample = np.random.choice(len(spa_flat), size=min(5000, len(spa_flat)), replace=False)
    ax.scatter(spa_flat[subsample], sim_flat[subsample], alpha=0.1, s=5)
    ax.set_xlabel('Spatial Distance (meters)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Similarity vs Distance')
    ax.axhline(y=off_diag.mean(), color='r', linestyle='--', label=f'mean={off_diag.mean():.3f}')
    ax.legend()
    
    # 5. Latent norms (check if some are zero/tiny)
    ax = axes[1, 1]
    norms = np.linalg.norm(latents, axis=1)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=norms, cmap='viridis', s=20)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Latent L2 Norms\nmean={norms.mean():.3f}, std={norms.std():.3f}')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='L2 norm')
    
    # 6. First 2 PCA components colored by position
    ax = axes[1, 2]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # Color by distance from center
    dist_from_center = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    scatter = ax.scatter(latents_pca[:, 0], latents_pca[:, 1], 
                         c=dist_from_center, cmap='plasma', s=20, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA of Latents\n(colored by distance from center)')
    plt.colorbar(scatter, ax=ax, label='Dist from center (m)')
    
    plt.tight_layout()
    plt.savefig('latent_similarity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # Additional: Show a few latent vectors directly
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pick 4 latents: corners
    grid_size = int(np.sqrt(num_latents))  # Assuming square grid
    corner_indices = [0, grid_size-1, num_latents-grid_size, num_latents-1]
    corner_names = ['Top-left', 'Top-right', 'Bottom-left', 'Bottom-right']
    
    for idx, (corner_idx, name) in enumerate(zip(corner_indices, corner_names)):
        ax = axes2[idx // 2, idx % 2]
        ax.plot(latents[corner_idx], linewidth=0.5)
        ax.set_title(f'{name} (idx={corner_idx})\ncoord=({coords[corner_idx, 0]:.1f}, {coords[corner_idx, 1]:.1f})')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.set_ylim([latents.min() - 0.1, latents.max() + 0.1])
    
    plt.suptitle('./figures/Latent Vectors at Grid Corners', fontsize=14)
    plt.tight_layout()
    plt.savefig('./figures/latent_corners.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # Check: are corners different from each other?
    # =========================================================================
    print(f"\n{'='*60}")
    print("CORNER LATENT SIMILARITY")
    print(f"{'='*60}")
    for i, (idx_i, name_i) in enumerate(zip(corner_indices, corner_names)):
        for j, (idx_j, name_j) in enumerate(zip(corner_indices, corner_names)):
            if i < j:
                sim = cos_sim[idx_i, idx_j]
                print(f"{name_i} vs {name_j}: cos_sim = {sim:.4f}")
    
    return {
        'r2_x': scores_x.mean(),
        'r2_y': scores_y.mean(),
        'cos_sim_mean': off_diag.mean(),
        'cos_sim_std': off_diag.std(),
    }


# =============================================================================
# USAGE
# =============================================================================

ckpt = torch.load("./checkpoints/Atomiserxp_20260111_135250_qj8m-val_loss-epoch=66-val_loss=0.0325.ckpt", map_location='cpu')
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# Find keys
print("Keys containing 'latent':")
for key in state_dict.keys():
    if 'latent' in key.lower():
        print(f"  {key}: {state_dict[key].shape}")

grid = state_dict["encoder.geo_pruning.geometry.latent_grid"].numpy()
print(f"\nGrid shape: {grid.shape}")

# Try to find the right latent key
latent_key = None
for key in state_dict.keys():
    if 'spatial_latents' in key or ('latents' in key and 'global' not in key and 'content' not in key):
        latent_key = key
        break

if latent_key is None:
    raise ValueError("Could not find spatial_latents")

print(f"Using key: {latent_key}")
latents = state_dict[latent_key].numpy()
print(f"Latents shape: {latents.shape}")

# Take only spatial latents
latents = latents[:625, :]

#results = linear_probe_latent_positions(latents, grid)

grid = state_dict["encoder.geo_pruning.geometry.latent_grid"].numpy()  # [625, 2]
latents = state_dict["encoder.latents"].numpy()[:625, :]  # [625, 512]

print("="*60)
print("WHAT DID THE LATENTS ACTUALLY LEARN?")
print("="*60)

# =========================================================================
# 1. PCA Analysis - What are the main axes of variation?
# =========================================================================
pca = PCA(n_components=20)
latents_pca = pca.fit_transform(latents)

print(f"\nPCA Explained Variance (first 10 components):")
for i in range(10):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3%}")
print(f"  Total (20 PCs): {pca.explained_variance_ratio_.sum():.3%}")

# =========================================================================
# 2. Clustering - Do latents form natural groups?
# =========================================================================
print(f"\n{'='*60}")
print("CLUSTERING ANALYSIS")
print("="*60)

# Try different numbers of clusters
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latents)
    score = silhouette_score(latents, labels)
    silhouette_scores.append((k, score))
    print(f"  k={k}: silhouette={score:.3f}")

best_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"\nBest k = {best_k}")

# =========================================================================
# 3. Visualize clusters spatially
# =========================================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(latents)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. PCA colored by cluster
ax = axes[0, 0]
scatter = ax.scatter(latents_pca[:, 0], latents_pca[:, 1], c=cluster_labels, cmap='tab10', s=20, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title(f'PCA of Latents\n(colored by cluster, k={best_k})')
plt.colorbar(scatter, ax=ax, label='Cluster')

# 2. Clusters in spatial grid
ax = axes[0, 1]
scatter = ax.scatter(grid[:, 0], grid[:, 1], c=cluster_labels, cmap='tab10', s=20)
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title('Cluster Labels in Spatial Grid\n(Is there spatial pattern?)')
ax.set_aspect('equal')
plt.colorbar(scatter, ax=ax, label='Cluster')

# 3. PCA colored by X position
ax = axes[0, 2]
scatter = ax.scatter(latents_pca[:, 0], latents_pca[:, 1], c=grid[:, 0], cmap='viridis', s=20, alpha=0.7)
ax.set_xlabel(f'PC1')
ax.set_ylabel(f'PC2')
ax.set_title('PCA colored by X position\n(random = no X encoding)')
plt.colorbar(scatter, ax=ax, label='X (m)')

# 4. PCA colored by Y position
ax = axes[1, 0]
scatter = ax.scatter(latents_pca[:, 0], latents_pca[:, 1], c=grid[:, 1], cmap='plasma', s=20, alpha=0.7)
ax.set_xlabel(f'PC1')
ax.set_ylabel(f'PC2')
ax.set_title('PCA colored by Y position\n(random = no Y encoding)')
plt.colorbar(scatter, ax=ax, label='Y (m)')

# 5. PCA colored by distance from center
ax = axes[1, 1]
dist_from_center = np.sqrt(grid[:, 0]**2 + grid[:, 1]**2)
scatter = ax.scatter(latents_pca[:, 0], latents_pca[:, 1], c=dist_from_center, cmap='coolwarm', s=20, alpha=0.7)
ax.set_xlabel(f'PC1')
ax.set_ylabel(f'PC2')
ax.set_title('PCA colored by distance from center\n(radial pattern?)')
plt.colorbar(scatter, ax=ax, label='Dist (m)')

# 6. t-SNE for better visualization
ax = axes[1, 2]
print("\nComputing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latents_tsne = tsne.fit_transform(latents)
scatter = ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=cluster_labels, cmap='tab10', s=20, alpha=0.7)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of Latents\n(colored by cluster)')
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.tight_layout()
plt.savefig('latent_structure_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================================
# 4. Check if clusters have spatial meaning
# =========================================================================
print(f"\n{'='*60}")
print("CLUSTER SPATIAL ANALYSIS")
print("="*60)

for c in range(best_k):
    mask = cluster_labels == c
    cluster_coords = grid[mask]
    print(f"\nCluster {c} (n={mask.sum()}):")
    print(f"  X range: [{cluster_coords[:, 0].min():.1f}, {cluster_coords[:, 0].max():.1f}]")
    print(f"  Y range: [{cluster_coords[:, 1].min():.1f}, {cluster_coords[:, 1].max():.1f}]")
    print(f"  X mean: {cluster_coords[:, 0].mean():.1f}, Y mean: {cluster_coords[:, 1].mean():.1f}")

# =========================================================================
# 5. Check variance per dimension
# =========================================================================
print(f"\n{'='*60}")
print("LATENT DIMENSION ANALYSIS")
print("="*60)

dim_variances = latents.var(axis=0)
dim_means = latents.mean(axis=0)

print(f"Dimension variance: mean={dim_variances.mean():.6f}, std={dim_variances.std():.6f}")
print(f"  min={dim_variances.min():.6f}, max={dim_variances.max():.6f}")

# How many dimensions are "active" (high variance)?
threshold = dim_variances.mean() + dim_variances.std()
active_dims = (dim_variances > threshold).sum()
print(f"  'Active' dimensions (var > mean+std): {active_dims} / 512")

# Plot dimension variance
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(512), np.sort(dim_variances)[::-1], width=1.0)
ax.set_xlabel('Dimension (sorted by variance)')
ax.set_ylabel('Variance')
ax.set_title('Variance per Latent Dimension\n(Are only a few dimensions used?)')
ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.6f})')
ax.legend()
plt.tight_layout()
plt.savefig('dimension_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================================
# 6. Compare spatial vs global latents
# =========================================================================
print(f"\n{'='*60}")
print("SPATIAL vs GLOBAL LATENTS")
print("="*60)

all_latents = state_dict["encoder.latents"].numpy()  # [753, 512]
spatial_latents = all_latents[:625, :]
global_latents = all_latents[625:, :]  # [128, 512]

print(f"Spatial latents: {spatial_latents.shape}")
print(f"Global latents: {global_latents.shape}")

# Compare statistics
print(f"\nSpatial: mean={spatial_latents.mean():.4f}, std={spatial_latents.std():.4f}")
print(f"Global:  mean={global_latents.mean():.4f}, std={global_latents.std():.4f}")

# Similarity between spatial and global
spatial_norm = spatial_latents / (np.linalg.norm(spatial_latents, axis=1, keepdims=True) + 1e-8)
global_norm = global_latents / (np.linalg.norm(global_latents, axis=1, keepdims=True) + 1e-8)

cross_sim = spatial_norm @ global_norm.T  # [625, 128]
print(f"\nSpatial-Global similarity: mean={cross_sim.mean():.4f}, std={cross_sim.std():.4f}")