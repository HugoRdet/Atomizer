"""
FRACTAL Dataset — Visualization Variant
========================================

Thin subclass of FractalDataset that:

1. Stores raw metric coordinates (Lambert-93 x, y, z in metres) in the
   spare query token columns 5/6/7.

2. Decouples encoder and decoder point counts:
     - Encoder tokens: capped at max_encoder_lidar (default 100k) to
       avoid OOM during geographic pruning / cross-attention.
     - Decoder queries: ALL LiDAR points, no cap. The decoder runs in
       10k-point chunks (handled by forward()), so memory is fine.

   This means the latent grid is built from a representative 100k-point
   sample, but every point in the scene gets a decoded feature vector.

Query token column layout:
    col 0 : z_norm          (used by z_query_projection)
    col 1 : pixel x
    col 2 : pixel y
    col 3 : spectral_idx
    col 4 : label
    col 5 : raw x  (Lambert-93 metres)   ← viz only
    col 6 : raw y  (Lambert-93 metres)   ← viz only
    col 7 : raw z  (metres above sea level) ← viz only

Training pipeline is completely unaffected — this subclass is only
imported by the PCA / visualization scripts.
"""

import numpy as np
import torch

from .utils_dataset_fractal import FractalDataset, REMAP_LUT


class FractalDatasetViz(FractalDataset):
    """
    FractalDataset variant for visualization.

    Extra constructor args:
        max_encoder_lidar: int | None
            Cap on LiDAR points fed to the ENCODER (tokens/groups).
            Default 100_000. Set None for no cap (may OOM on large scenes).
            The DECODER always sees all points regardless of this value.
    """

    def __init__(self, *args, max_encoder_lidar=100_000, **kwargs):
        # Pass max_lidar_points=None to parent so it doesn't subsample
        # at load time — we handle subsampling ourselves below.
        kwargs["max_lidar_points"] = None
        super().__init__(*args, **kwargs)
        self.max_encoder_lidar = max_encoder_lidar
        print(f"[FractalDatasetViz] max_encoder_lidar={max_encoder_lidar} "
              f"(decoder sees ALL points)")

    def __getitem__(self, index):
        row = self.patch_rows[index]
        aug = self.augmenter.sample(index=index)

        # ── Load LIDAR ─────────────────────────────────────────────
        import laspy
        las = laspy.read(row["laz_path"])
        n_points_raw = las.x.shape[0]
        if n_points_raw < self.MIN_POINTS:
            return self.__getitem__((index + 1) % len(self))

        # Raw metric coordinates (for decoder query cols 5/6/7)
        raw_x = np.asarray(las.x, dtype=np.float32)
        raw_y = np.asarray(las.y, dtype=np.float32)
        raw_z = np.asarray(las.z, dtype=np.float32)

        # Patch-local pixel coords
        x_min = float(raw_x.min())
        y_max = float(raw_y.max())
        lidar_x = (raw_x - x_min) / self.VHR_RESOLUTION
        lidar_y = (y_max - raw_y)  / self.VHR_RESOLUTION
        lidar_x = np.clip(lidar_x, 0.0, self.PATCH_SIZE_PX - 1e-3)
        lidar_y = np.clip(lidar_y, 0.0, self.PATCH_SIZE_PX - 1e-3)

        return_number     = np.asarray(las.return_number,     dtype=np.int64)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int64)

        las_cls = np.asarray(las.classification, dtype=np.int64)
        las_cls = np.clip(las_cls, 0, REMAP_LUT.shape[0] - 1)
        labels  = REMAP_LUT[las_cls]

        # Z normalisation (uses all points for ground estimation — more robust)
        ground_mask = (labels == 1)
        if ground_mask.sum() >= self.GROUND_MEDIAN_MIN_PTS:
            local_ground = float(np.median(raw_z[ground_mask]))
        else:
            local_ground = float(np.percentile(raw_z, 5.0))
        z_rel  = raw_z - local_ground
        z_clip = np.clip(z_rel, self.Z_GROUND_REL_LO, self.Z_GROUND_REL_HI)
        z_norm = z_clip / self.Z_GROUND_REL_SCALE

        # D4 (identity at test time)
        xy_stacked = np.stack([lidar_x, lidar_y], axis=1).astype(np.float32)
        if not aug.is_identity:
            xy_stacked = self.augmenter.apply_to_xy(
                xy_stacked, aug, patch_size_px=self.PATCH_SIZE_PX)
        xy_stacked = np.clip(
            xy_stacked, 0.0, self.PATCH_SIZE_PX - 1e-3).astype(np.float32)
        lidar_x = xy_stacked[:, 0]
        lidar_y = xy_stacked[:, 1]

        # ── DECODER arrays: ALL points (no cap) ───────────────────
        # These go into queries — the forward() chunks them at 10k.
        dec_lidar_x = lidar_x
        dec_lidar_y = lidar_y
        dec_z_norm  = z_norm
        dec_labels  = labels
        dec_raw_x   = raw_x
        dec_raw_y   = raw_y
        dec_raw_z   = raw_z

        # ── ENCODER arrays: capped at max_encoder_lidar ────────────
        # Deterministic subsample so the latent grid is reproducible.
        rng = np.random.default_rng(seed=hash(row["patch_id"]) & 0xFFFFFFFF)
        if (self.max_encoder_lidar is not None
                and n_points_raw > self.max_encoder_lidar):
            sel = rng.choice(n_points_raw, size=self.max_encoder_lidar,
                             replace=False)
            enc_lidar_x = lidar_x[sel]
            enc_lidar_y = lidar_y[sel]
            enc_z_norm  = z_norm[sel]
            enc_labels  = labels[sel]
            enc_rn      = return_number[sel]
            enc_nor     = number_of_returns[sel]
        else:
            enc_lidar_x = lidar_x
            enc_lidar_y = lidar_y
            enc_z_norm  = z_norm
            enc_labels  = labels
            enc_rn      = return_number
            enc_nor     = number_of_returns

        n_enc = enc_lidar_x.shape[0]
        n_dec = dec_lidar_x.shape[0]

        # ── Encoder tensors ────────────────────────────────────────
        enc_positions = torch.from_numpy(
            np.stack([enc_lidar_x, enc_lidar_y], axis=1)).float()
        enc_values    = torch.from_numpy(enc_z_norm).float()
        enc_labels_t  = torch.from_numpy(enc_labels.astype(np.int64))

        # ── Decoder tensors (all points) ───────────────────────────
        dec_positions = torch.from_numpy(
            np.stack([dec_lidar_x, dec_lidar_y], axis=1)).float()
        dec_values    = torch.from_numpy(dec_z_norm).float()
        dec_labels_t  = torch.from_numpy(dec_labels.astype(np.int64))

        # ── Ortho ──────────────────────────────────────────────────
        import rasterio
        with rasterio.open(row["ortho_path"]) as src:
            ortho = src.read().astype(np.float32)
        ortho = torch.from_numpy(ortho)
        ortho = (ortho / 127.5) - 1.0
        ortho = torch.clamp(ortho, -10, 10)
        ortho = torch.nan_to_num(ortho, nan=0.0, posinf=10.0, neginf=-10.0)
        if not aug.is_identity:
            ortho = self.augmenter.apply(ortho, aug)

        dense_label = torch.full(
            (self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            self.IGNORE_INDEX, dtype=torch.long)

        # ── Tokenize VHR (unchanged) ───────────────────────────────
        vhr_tokens = self.token_builder.build_tokens(
            image=ortho,
            label=dense_label,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.vhr_spectral_indices,
            resolution_idx=self.resolution_idx,
            time_idx=self.TIME_IDX_NA,
        )

        # ── Tokenize LIDAR — ENCODER subset only ───────────────────
        lidar_tokens = self.token_builder.build_sparse_tokens(
            values=enc_values,
            positions=enc_positions,
            labels=enc_labels_t,
            resolution=self.VHR_RESOLUTION,
            spectral_indices=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
            return_number=enc_rn,
            number_of_returns=enc_nor,
        )

        # No padding on encoder tokens — variable length is fine
        vhr_mask     = torch.zeros(vhr_tokens.shape[0],   dtype=torch.bool)
        lidar_mask   = torch.zeros(lidar_tokens.shape[0], dtype=torch.bool)
        hires_tokens = torch.cat([vhr_tokens, lidar_tokens], dim=0)
        hires_mask   = torch.cat([vhr_mask, lidar_mask],     dim=0)

        groups = {
            self.VHR_RESOLUTION: {
                "tokens": hires_tokens,
                "mask":   hires_mask,
                "shape":  (self.NUM_VHR_BANDS,
                           self.PATCH_SIZE_PX, self.PATCH_SIZE_PX),
            }
        }

        # ── Queries: ALL decoder points ────────────────────────────
        queries = self.token_builder.build_sparse_queries(
            positions=dec_positions,
            labels=dec_labels_t,
            resolution=self.VHR_RESOLUTION,
            first_spectral_idx=self.lidar_spectral_idx,
            resolution_idx=self.resolution_idx,
            patch_size_px=self.PATCH_SIZE_PX,
            time_idx=self.TIME_IDX_NA,
        )
        queries[:, 0] = dec_values   # z_norm (used by z_query_projection)

        # Raw metric coords in spare cols 5/6/7
        queries[:, 5] = torch.from_numpy(dec_raw_x)
        queries[:, 6] = torch.from_numpy(dec_raw_y)
        queries[:, 7] = torch.from_numpy(dec_raw_z)

        # No padding on queries — forward() handles variable length via chunking
        queries_mask = torch.zeros(n_dec, dtype=torch.bool)

        return {
            "groups":            groups,
            "queries":           queries,       # [N_all, 8], cols 5/6/7 = raw xyz
            "queries_mask":      queries_mask,
            "label":             dec_labels_t,
            "n_real_lidar":      torch.tensor(n_dec, dtype=torch.long),
            "target_resolution": self.VHR_RESOLUTION,
            "image":             ortho,
            "patch_id":          row["patch_id"],
        }
