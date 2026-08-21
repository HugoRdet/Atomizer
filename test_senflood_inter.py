# extract_spectral_attention_train.py
import os
import torch
from training.utils import read_yaml, Lookup_encoding
from training.trainer_senflood_inter import Model_SenFlood_Skip_Inter
from training.utils.datasets.utils_dataset_senflood_skip import Sen1Floods11SkipDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip
from training.utils.band_groups import build_band_group_lut_by_index

CKPT_PATH = "./checkpoints/senflood/atos.ckpt"
config_model = read_yaml("./training/configs/config_test-SENFLOOD.yaml")
bands_yaml = "./data/bands_info/bands.yaml"

lookup_table = Lookup_encoding(None, read_yaml(bands_yaml), config_model)
band_group_lut = build_band_group_lut_by_index(max_idx=14)

model = Model_SenFlood_Skip_Inter.load_from_checkpoint(
    CKPT_PATH, strict=False, config=config_model, wand=False,
    name="spectral_attn", transform=None, lookup_table=lookup_table,
    band_group_lut=band_group_lut,
)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.encoder.band_group_lut = band_group_lut.to(device)

data_module = UnifiedDataModule(
    path="./data/SENFLOOD", batch_size=1, num_workers=4,
    trans_modalities=None, trans_tokens=None, model=config_model["encoder"],
    dataset_config=read_yaml(bands_yaml), config_model=config_model,
    look_up=lookup_table, dataset_class=Sen1Floods11SkipDataset,
    collate_fn=collate_grouped_skip,
)

# ── SWITCH: train set instead of test set ────────────────────────────────
data_module.setup("fit")
train_loader = data_module.train_dataloader()
# NOTE: if UnifiedDataModule's train_dataloader shuffles by default, that's
# fine here (we're scanning, not training), but if you want deterministic /
# repeatable tile indices across reruns, check for a shuffle=False option or
# wrap the underlying dataset directly:
#   train_loader = torch.utils.data.DataLoader(
#       data_module.train_dataset, batch_size=1, shuffle=False,
#       num_workers=4, collate_fn=collate_grouped_skip)

def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _to_device(v, device)
        else:
            out[k] = v
    return out

OUT_SCAN_DIR = "./spectral_attn_scan_train"
os.makedirs(OUT_SCAN_DIR, exist_ok=True)
results = []

IGNORE_INDEX = 255
WATER_CLASS = 1
NONWATER_CLASS = 0
all_water_mass = []
all_nonwater_mass = []

with torch.no_grad():
    for idx, batch in enumerate(train_loader):
        batch = _to_device(batch, device)
        logits, mass = model.forward_with_attention(batch)  # mass: [1, M, 3]

        label = batch.get("label", None)
        if label is not None:
            label_flat = label.reshape(-1) if label.dim() > 1 else label.squeeze(0)
            mass_flat = mass.squeeze(0)
            if label_flat.numel() == mass_flat.shape[0]:
                valid = label_flat != IGNORE_INDEX
                water = valid & (label_flat == WATER_CLASS)
                nonwater = valid & (label_flat == NONWATER_CLASS)
                if water.sum() > 0:
                    all_water_mass.append(mass_flat[water].cpu())
                if nonwater.sum() > 0:
                    all_nonwater_mass.append(mass_flat[nonwater].cpu())

        dominant = mass.argmax(dim=-1)[0]
        strong = mass.max(dim=-1).values[0] > 0.3
        counts = [((dominant == g) & strong).sum().item() for g in range(3)]
        n_groups_present = sum(c >= 10_000 for c in counts)

        results.append({"idx": idx, "counts": counts, "n_groups_present": n_groups_present})

        if n_groups_present >= 2:
            torch.save({
                "mass": mass.cpu(), "logits": logits.cpu(),
                "label": batch.get("label", None),
                "image": batch.get("image", None),
                "counts": counts,
            }, f"{OUT_SCAN_DIR}/tile_{idx:04d}.pt")
            print(f"[{idx}] KEPT — counts SAR/SWIR/REST = {counts}")
        else:
            print(f"[{idx}] skip — counts SAR/SWIR/REST = {counts}")

        # Optional: cap the scan if the training set is large and this is slow
        # if idx >= 500:
        #     break

results.sort(key=lambda r: min(r["counts"]), reverse=True)
print("\nTop tiles by balanced group presence (TRAIN SET):")
for r in results[:15]:
    print(r)

if all_water_mass and all_nonwater_mass:
    water_cat = torch.cat(all_water_mass, dim=0)
    nonwater_cat = torch.cat(all_nonwater_mass, dim=0)
    print("\n" + "=" * 60)
    print("AGGREGATE WATER vs NON-WATER SPECTRAL ATTENTION (TRAIN SET)")
    print("=" * 60)
    water_mean = water_cat.mean(dim=0)
    nonwater_mean = nonwater_cat.mean(dim=0)
    for i, name in enumerate(["SAR", "SWIR", "REST"]):
        delta = water_mean[i].item() - nonwater_mean[i].item()
        print(f"{name:<8} water={water_mean[i].item():.4f}  "
              f"nonwater={nonwater_mean[i].item():.4f}  delta={delta:+.4f}")
# plot_spectral_attention.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

TILE_IDS = [33,126,143]
SCAN_DIR = "./spectral_attn_scan_train"
OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_NAMES = ["SAR", "SWIR", "REST"]
GROUP_COLORS = ["#1f77b4", "#d62728", "#7f7f7f"]  # blue=SAR, red=SWIR, grey=REST
GROUP_CMAP = ListedColormap(GROUP_COLORS)

# Ground-truth mask: 3 EXPLICIT categories, no NaN/transparent trick.
#   0 = non-water -> light grey, 1 = water -> dark blue, 2 = ignore -> yellow
MASK_COLORS_3 = ["#f0f0f0", "#08519c", "#e6d200"]
MASK_CMAP_3 = ListedColormap(MASK_COLORS_3)
IGNORE_INDEX = 255
WATER_CLASS = 1
NONWATER_CLASS = 0


def to_hw(t, H, W):
    t = t.squeeze()
    if t.dim() == 1:
        assert t.numel() == H * W, f"Cannot reshape {t.numel()} elements to {H}x{W}"
        return t.reshape(H, W)
    return t


def extract_rgb(image, H, W):
    if image is None:
        return None
    img = image.squeeze(0)
    if img.dim() == 4:
        img = img[0]
    C = img.shape[0]
    if C >= 3:
        rgb = img[[2, 1, 0], :, :] if C >= 4 else img[:3]
    else:
        rgb = img.repeat(3, 1, 1)
    rgb = rgb.permute(1, 2, 0).float().numpy()
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def plot_tile(tile_id):
    path = os.path.join(SCAN_DIR, f"tile_{tile_id:04d}.pt")
    if not os.path.exists(path):
        print(f"Missing {path}, skipping")
        return

    d = torch.load(path, map_location="cpu")
    mass = d["mass"].squeeze(0)          # [M, 3]
    label = d.get("label", None)
    image = d.get("image", None)

    M = mass.shape[0]
    H = W = int(round(M ** 0.5))
    assert H * W == M, f"Tile {tile_id}: M={M} is not a perfect square, need explicit H,W"

    # ── dominant band-group map — FULLY OPAQUE, no confidence fading ────────
    dominant = mass.argmax(dim=-1)                 # [M]
    dominant_hw = to_hw(dominant, H, W).numpy()

    # ── RGB panel ─────────────────────────────────────────────────────────
    rgb = extract_rgb(image, H, W)

    # ── mask panel: explicit 3-class array (non-water/water/ignore) ────────
    mask_hw = None
    mask_display = None
    if label is not None:
        mask_hw = to_hw(label, H, W).numpy()
        mask_display = np.full_like(mask_hw, fill_value=2, dtype=np.int64)
        mask_display[mask_hw == NONWATER_CLASS] = 0
        mask_display[mask_hw == WATER_CLASS] = 1

    # ── figure ────────────────────────────────────────────────────────────
    n_panels = 1 + (1 if rgb is not None else 0) + (1 if mask_display is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    if rgb is not None:
        axes[ax_idx].imshow(rgb)
        axes[ax_idx].set_title(f"Tile {tile_id} — RGB")
        axes[ax_idx].axis("off")
        ax_idx += 1

    # Attention map: plain categorical argmax, fully opaque, no confidence
    # encoding. Grey = REST wins argmax, blue = SAR wins, red = SWIR wins.
    axes[ax_idx].imshow(dominant_hw, cmap=GROUP_CMAP, vmin=0, vmax=2)
    axes[ax_idx].set_title(f"Tile {tile_id} — Dominant spectral group\n"
                            f"(argmax, no confidence weighting)")
    axes[ax_idx].axis("off")
    legend_elems = [Patch(facecolor=GROUP_COLORS[i], label=GROUP_NAMES[i])
                    for i in range(3)]
    axes[ax_idx].legend(handles=legend_elems, loc="lower right", fontsize=9,
                         framealpha=0.9)
    ax_idx += 1

    if mask_display is not None:
        axes[ax_idx].imshow(mask_display, cmap=MASK_CMAP_3, vmin=0, vmax=2)
        axes[ax_idx].set_title(f"Tile {tile_id} — Ground truth")
        axes[ax_idx].axis("off")
        legend_elems_mask = [
            Patch(facecolor=MASK_COLORS_3[0], edgecolor="grey", label="non-water"),
            Patch(facecolor=MASK_COLORS_3[1], label="water"),
            Patch(facecolor=MASK_COLORS_3[2], label="no data / ignore"),
        ]
        axes[ax_idx].legend(handles=legend_elems_mask, loc="lower right",
                             fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"tile_{tile_id:04d}_spectral_attn.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


for tid in TILE_IDS:
    plot_tile(tid)
