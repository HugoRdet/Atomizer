"""
Segmentation Visualization Callback
=====================================
Plots a row of panels per sample at every training epoch end (rank 0 only):
    [RGB, GT, Prediction, (Error, Confidence, Overconfident)]

The last three panels are optional and render only when the inputs are
available. Each panel is built in isolation so one failure does not
take down the whole figure.

Supports both:
    - flat-batch format (queries at top level): Sen1Floods11, BurnScars
    - task-wrapped format (queries under "tasks"): PASTIS
"""

import traceback

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from training.utils.datasets.token_grouping import collate_grouped


# =============================================================================
# HELPERS
# =============================================================================

def _try_collate_multitask(samples):
    """
    Choose the right collate based on the sample's format.

    If sample has 'tasks' key (PASTIS), use collate_multitask.
    Otherwise (Sen1Floods11, BurnScars), use collate_grouped.

    This avoids collate_multitask producing batches with an empty
    tasks={} container when given flat-format samples.
    """
    first = samples[0] if samples else {}
    has_tasks = isinstance(first, dict) and "tasks" in first and first["tasks"]

    if has_tasks:
        try:
            from training.utils.datasets.token_grouping import collate_multitask
            return collate_multitask(samples)
        except Exception as e:
            print(f"[SEG VIZ] collate_multitask failed, falling back: {e}")

    return collate_grouped(samples)


def _flatten_batch_queries(batch):
    """
    Ensure top-level 'queries' and 'queries_mask' regardless of input format.

    Handles three cases:
      1. Flat format: batch["queries"] already present → passthrough.
      2. Task format: batch["tasks"]["<name>"]["queries"] → lift to top level.
      3. Empty tasks: batch["tasks"] == {} → raise informative error.

    Note: checks flat format first, since some collates produce BOTH
    batch["queries"] AND batch["tasks"] = {} (empty task dict alongside
    flat queries). The flat branch wins.
    """
    # Case 1: flat format (preferred)
    if "queries" in batch and batch["queries"] is not None:
        return batch

    # Case 2: task format
    tasks = batch.get("tasks")
    if isinstance(tasks, dict) and len(tasks) > 0:
        task_data = next(iter(tasks.values()))
        batch = dict(batch)
        batch["queries"]      = task_data["queries"]
        batch["queries_mask"] = task_data["queries_mask"]
        return batch

    # Case 3: neither available — raise with diagnostic info
    raise KeyError(
        f"Batch has no 'queries' (flat) and no non-empty 'tasks' (wrapped). "
        f"Keys present: {list(batch.keys())}, "
        f"tasks={tasks!r}"
    )


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


def _coords_to_pixels(coords_m, geometry, H, W):
    """
    Convert latent coords (meters) to pixel coords (x, y).

    Tries geometry.meters_to_pixels first (if the API exists and works).
    Falls back to a linear remap: map coords bounding box → [0, W-1] × [0, H-1].

    Returns [L, 2] array of (px, py) in pixel space, or None if something's off.
    """
    if coords_m is None or len(coords_m) == 0:
        return None

    # Try the geometry helper
    if hasattr(geometry, "meters_to_pixels"):
        try:
            coords_t = torch.from_numpy(coords_m).float() \
                        if isinstance(coords_m, np.ndarray) \
                        else coords_m.float()
            px = geometry.meters_to_pixels(coords_t, image_size=H)
            if isinstance(px, torch.Tensor):
                px = px.cpu().numpy()
            px = np.asarray(px, dtype=np.float32)
            if px.ndim == 2 and px.shape[1] == 2:
                return px
        except Exception:
            pass  # fall through to linear remap

    # Fallback: linear remap
    coords = np.asarray(coords_m, dtype=np.float32)
    xs = coords[:, 0]; ys = coords[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    px = (xs - x_min) / x_span * (W - 1)
    py = (ys - y_min) / y_span * (H - 1)
    return np.stack([px, py], axis=1)


# =============================================================================
# CALLBACK
# =============================================================================

class SegmentationVizCallback(pl.Callback):
    """
    Robust viz callback with optional error / confidence / overconfident panels.

    Supports multiple datasets out of the box:
      - Sen1Floods11 (binary flood seg)
      - BurnScars (binary burn seg)
      - PASTIS (19-class crop seg, task-wrapped batch format)

    Pass `dataset_preset="pastis"` (or other preset names) to auto-configure
    class names and RGB indices, or pass them explicitly via constructor args.

    Notes on RGB indices (per sample["image"] channel order):
      Sen1Floods11 (S2+S1, B02..B12 then VV/VH):  B04=3, B03=2, B02=1 → [3, 2, 1]
      BurnScars (HLS, B02..B12):                  B04=3, B03=2, B02=1 → [3, 2, 1]
      PASTIS (S2 at [0..9], then S1 [10..11]):    B04=2, B03=1, B02=0 → [2, 1, 0]
    """

    RGB_INDICES = [3, 2, 1]

    # Dataset presets for class names and RGB band order
    PRESETS = {
        "sen1floods11": {
            "class_names": ["no_flood", "flood"],
            "rgb_indices": [3, 2, 1],
        },
        "burnscars": {
            "class_names": ["no_burn", "burn"],
            "rgb_indices": [3, 2, 1],
        },
        "pastis": {
            # 19 active classes + 1 ignore. Indices 0-18 are used; 19 is ignored.
            "class_names": [
                "Background", "Meadow", "Soft winter wheat", "Corn",
                "Winter barley", "Winter rapeseed", "Spring barley",
                "Sunflower", "Grapevine", "Beet", "Winter triticale",
                "Winter durum wheat", "Fruits/veg/flowers", "Potatoes",
                "Leguminous fodder", "Soybean", "Orchard", "Mixed cereal",
                "Sorghum",
            ],
            "rgb_indices": [2, 1, 0],   # PASTIS S2 starts at ch0
        },
    }

    def __init__(
        self,
        sample_indices=(0, 1, 2),
        log_every_n_epochs=1,
        use_wandb=True,
        ignore_index=255,
        class_names=None,
        rgb_indices=None,
        dataset_preset=None,
    ):
        super().__init__()
        self.sample_indices     = sample_indices
        self.log_every_n_epochs = log_every_n_epochs
        self.use_wandb          = use_wandb
        self.ignore_index       = ignore_index

        # Apply preset first, then allow explicit overrides
        preset = None
        if dataset_preset is not None:
            key = dataset_preset.lower()
            if key not in self.PRESETS:
                raise ValueError(
                    f"Unknown dataset_preset='{dataset_preset}'. "
                    f"Valid: {list(self.PRESETS.keys())}"
                )
            preset = self.PRESETS[key]

        preset_class_names = preset["class_names"] if preset else None
        preset_rgb_indices = preset["rgb_indices"] if preset else None

        self.class_names = (class_names
                            or preset_class_names
                            or ["no_flood", "flood"])
        self.rgb_indices = (rgb_indices
                            or preset_rgb_indices
                            or self.RGB_INDICES)

        print(f"[SEG VIZ] Initialized: {len(self.class_names)} classes, "
              f"RGB indices={self.rgb_indices}, "
              f"preset={dataset_preset or 'none'}")

    # =========================================================================
    # EPOCH HOOK
    # =========================================================================

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dataset = getattr(trainer.datamodule, "val_dataset", None) \
                  or getattr(trainer.datamodule, "train_dataset", None)
        if dataset is None or not hasattr(dataset, "get_viz_sample"):
            return

        # Check encoder capabilities
        encoder = pl_module.encoder
        has_error_predictor = getattr(encoder, "use_error_predictor", False)

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        figures = []
        for idx in self.sample_indices:
            if idx >= len(dataset):
                continue
            fig = self._render_sample(dataset, idx, pl_module, device,
                                       has_error_predictor, trainer.current_epoch)
            if fig is not None:
                figures.append((f"seg_sample_{idx}", fig))

        if self.use_wandb and figures:
            try:
                import wandb
                for name, fig in figures:
                    wandb.log({name: wandb.Image(fig)})
            except Exception as e:
                print(f"[SEG VIZ] wandb log failed: {e}")
            finally:
                plt.close("all")
        else:
            plt.close("all")

        if was_training:
            pl_module.train()

    # =========================================================================
    # SINGLE SAMPLE
    # =========================================================================

    def _render_sample(self, dataset, idx, pl_module, device,
                       has_error_predictor, epoch):
        """
        Render one sample. Returns matplotlib Figure or None on total failure.

        Individual panels fail independently so a broken error predictor
        doesn't kill the whole viz.
        """
        try:
            sample = dataset.get_viz_sample(idx)
        except Exception as e:
            print(f"[SEG VIZ] sample {idx}: get_viz_sample failed: {e}")
            return None

        try:
            label_2d = sample["label"]
            H, W     = label_2d.shape
            image    = sample["image"]  # [C, H, W]
        except Exception as e:
            print(f"[SEG VIZ] sample {idx}: malformed sample: {e}")
            return None

        # ── Run model ─────────────────────────────────────────────────
        try:
            batch = _try_collate_multitask([sample])
            batch = _flatten_batch_queries(batch)
            batch = _batch_to_device(batch, device)

            with torch.no_grad():
                result = pl_module.encoder(
                    batch,
                    training=False,
                    return_for_error=has_error_predictor,
                )
        except Exception as e:
            print(f"[SEG VIZ] sample {idx}: forward pass failed: {e}")
            traceback.print_exc()
            return None

        # Unpack predictions defensively
        if isinstance(result, dict):
            y_hat = result.get("predictions")
        else:
            y_hat = result
            result = {}  # normalize to dict for uniform access below

        if y_hat is None:
            print(f"[SEG VIZ] sample {idx}: no predictions in result")
            return None

        # ── Build panel data (each wrapped individually) ──────────────
        pred_2d = self._safe(
            lambda: self._extract_pred_2d(y_hat, H, W),
            idx, "pred", None)

        rgb = self._safe(
            lambda: self._extract_rgb(image),
            idx, "rgb", None)

        conf_2d = self._safe(
            lambda: self._extract_confidence(y_hat, H, W),
            idx, "conf", None)

        overconf_2d = None
        if pred_2d is not None and conf_2d is not None:
            overconf_2d = self._safe(
                lambda: self._extract_overconfident(
                    pred_2d, label_2d.cpu().numpy(), conf_2d),
                idx, "overconf", None)

        error_map, mini_pixels = None, None
        if has_error_predictor:
            out = self._safe(
                lambda: self._extract_error_map(result, encoder=pl_module.encoder,
                                                  H=H, W=W),
                idx, "error", (None, None))
            if out is not None:
                error_map, mini_pixels = out

        # ── Build figure (always works, even if panels are None) ──────
        return self._safe(
            lambda: self._make_figure(
                rgb,
                label_2d.cpu().numpy() if isinstance(label_2d, torch.Tensor) else label_2d,
                pred_2d,
                error_map, mini_pixels,
                conf_2d, overconf_2d,
                idx, epoch),
            idx, "figure", None)

    # =========================================================================
    # PANEL EXTRACTORS
    # =========================================================================

    @staticmethod
    def _extract_pred_2d(y_hat, H, W):
        """[1, M, C] → argmax → [H, W] (row-major over first H*W queries)."""
        preds = torch.argmax(y_hat, dim=-1).squeeze(0).cpu()
        if preds.numel() < H * W:
            raise ValueError(
                f"Not enough predictions: got {preds.numel()}, need {H * W}")
        return preds[:H * W].reshape(H, W).numpy()

    def _extract_rgb(self, image):
        """Select 3 channels and percentile-normalize for display."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
        C = image.shape[0]
        safe_rgb = [min(int(i), C - 1) for i in self.rgb_indices]
        rgb = image[safe_rgb]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.numpy()
        rgb = rgb.astype(np.float32, copy=True)
        for c in range(rgb.shape[0]):
            lo = np.percentile(rgb[c], 2)
            hi = np.percentile(rgb[c], 98)
            if hi - lo > 1e-6:
                rgb[c] = (rgb[c] - lo) / (hi - lo)
            else:
                rgb[c] = 0.0
        return np.clip(rgb, 0, 1)

    @staticmethod
    def _extract_confidence(y_hat, H, W):
        """Max softmax prob per pixel → [H, W]."""
        probs = torch.softmax(y_hat.squeeze(0), dim=-1).cpu()
        conf  = probs.max(dim=-1).values
        if conf.numel() < H * W:
            raise ValueError("Not enough confidence values")
        return conf[:H * W].reshape(H, W).numpy()

    def _extract_overconfident(self, pred_2d, label_2d, conf_2d):
        """Highlight pixels that are wrong AND confident."""
        valid    = label_2d != self.ignore_index
        wrong    = (pred_2d != label_2d) & valid
        overconf = np.zeros_like(conf_2d, dtype=np.float32)
        overconf[wrong] = conf_2d[wrong]
        return overconf

    def _extract_error_map(self, result, encoder, H, W):
        """
        Build (error_map [H,W], mini_pixels [K,2] or None).

        Falls back gracefully if scipy is missing or coord lookup fails.
        """
        if not HAS_SCIPY:
            return None, None

        predicted_errors = result.get("predicted_errors")
        latent_coords    = result.get("latent_coords")

        # Fallback: attempt to read latent_coords from a cached attribute
        if latent_coords is None:
            latent_coords = getattr(encoder, "_viz_last_latent_coords", None)

        if predicted_errors is None or latent_coords is None:
            return None, None

        # Normalize shapes: [B, L] and [B, L, 2] → [L] and [L, 2]
        if isinstance(predicted_errors, torch.Tensor):
            pe_np = predicted_errors.detach().cpu().numpy()
        else:
            pe_np = np.asarray(predicted_errors)
        if pe_np.ndim == 2:
            pe_np = pe_np[0]

        if isinstance(latent_coords, torch.Tensor):
            lc_np = latent_coords.detach().cpu().numpy()
        else:
            lc_np = np.asarray(latent_coords)
        if lc_np.ndim == 3:
            lc_np = lc_np[0]

        L_pred = pe_np.shape[0]

        # Split primary vs refinement latents
        primary_coords = lc_np[:L_pred]
        mini_coords    = lc_np[L_pred:] if lc_np.shape[0] > L_pred else None

        # Convert to pixels
        geometry = getattr(encoder, "input_processor", None)
        geometry = getattr(geometry, "geometry", None) if geometry else None
        if geometry is None:
            return None, None

        prim_px = _coords_to_pixels(primary_coords, geometry, H, W)
        if prim_px is None or len(prim_px) < 3:
            return None, None

        # griddata interpolation
        lx = prim_px[:, 0].clip(0, W - 1)
        ly = prim_px[:, 1].clip(0, H - 1)
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        try:
            emap = griddata(
                points     = np.stack([lx, ly], axis=1),
                values     = pe_np.astype(np.float32),
                xi         = (grid_x, grid_y),
                method     = "linear",
                fill_value = float(pe_np.mean()),
            ).astype(np.float32)
        except Exception as e:
            print(f"[SEG VIZ] griddata failed: {e}")
            return None, None

        mini_px = _coords_to_pixels(mini_coords, geometry, H, W) \
                  if mini_coords is not None and len(mini_coords) > 0 else None

        return emap, mini_px

    # =========================================================================
    # FIGURE
    # =========================================================================

    def _make_figure(self, rgb, label, pred,
                     error_map, mini_pixels,
                     conf_map, overconf_map,
                     sample_idx, epoch):

        n_classes = len(self.class_names)
        cmap_seg  = (plt.cm.get_cmap("tab20", n_classes)
                     if n_classes > 10
                     else plt.cm.get_cmap("tab10", n_classes))

        # Count panels — RGB/GT/Pred are always present; others optional
        panel_specs = []
        if rgb is not None:
            panel_specs.append(("rgb", rgb))
        panel_specs.append(("gt",   label))
        if pred is not None:
            panel_specs.append(("pred", pred))
        if error_map is not None:
            panel_specs.append(("error", (error_map, mini_pixels)))
        if conf_map is not None:
            panel_specs.append(("conf", conf_map))
        if overconf_map is not None:
            panel_specs.append(("overconf", overconf_map))

        n_panels = max(1, len(panel_specs))
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        for i, (kind, data) in enumerate(panel_specs):
            ax = axes[i]
            try:
                if kind == "rgb":
                    ax.imshow(np.transpose(data, (1, 2, 0)))
                    ax.set_title("RGB")

                elif kind == "gt":
                    masked = np.ma.masked_where(data == self.ignore_index, data)
                    ax.imshow(masked, cmap=cmap_seg,
                              vmin=0, vmax=n_classes - 1,
                              interpolation="nearest")
                    ax.set_title("GT Label")

                elif kind == "pred":
                    ax.imshow(data, cmap=cmap_seg,
                              vmin=0, vmax=n_classes - 1,
                              interpolation="nearest")
                    ax.set_title("Prediction")

                elif kind == "error":
                    emap, minis = data
                    im = ax.imshow(emap, cmap="hot_r", interpolation="bilinear")
                    ax.set_title("Predicted Error")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    if minis is not None and len(minis) > 0:
                        ax.scatter(
                            minis[:, 0], minis[:, 1],
                            c="white", s=8, linewidths=0.5,
                            edgecolors="black", zorder=5, alpha=0.8)

                elif kind == "conf":
                    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                                   interpolation="nearest")
                    ax.set_title("Confidence")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                elif kind == "overconf":
                    im = ax.imshow(data, cmap="Reds", vmin=0.0, vmax=1.0,
                                   interpolation="nearest")
                    ax.set_title("Overconfident Failures")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except Exception as e:
                print(f"[SEG VIZ] panel '{kind}' failed: {e}")
                ax.text(0.5, 0.5, f"{kind}\nfailed",
                        ha="center", va="center",
                        transform=ax.transAxes)
            ax.axis("off")

        # Title with accuracy
        try:
            valid = label != self.ignore_index
            if pred is not None and valid.sum() > 0:
                acc = (pred[valid] == label[valid]).mean() * 100
                fig.suptitle(
                    f"Sample {sample_idx} — Epoch {epoch} — Acc: {acc:.1f}%",
                    fontsize=14)
            else:
                fig.suptitle(f"Sample {sample_idx} — Epoch {epoch}", fontsize=14)
        except Exception:
            fig.suptitle(f"Sample {sample_idx} — Epoch {epoch}", fontsize=14)

        fig.tight_layout()
        return fig

    # =========================================================================
    # SMALL SAFETY WRAPPER
    # =========================================================================

    @staticmethod
    def _safe(fn, sample_idx, label, default):
        """Run fn(), return default on failure, log the error."""
        try:
            return fn()
        except Exception as e:
            print(f"[SEG VIZ] sample {sample_idx}: {label} failed: {e}")
            return default