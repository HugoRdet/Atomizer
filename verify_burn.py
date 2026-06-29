"""
FRACTAL Atomizer — Modality Drop Inference Script
==================================================

Load a trained FRACTAL checkpoint and evaluate on the test split under
four test-time modality configurations:

  all         : VHR (NIR + R + G + B) + LIDAR  (no drop, headline number)
  rgb_lidar   : RGB + LIDAR                    (drop NIR)
  nir_lidar   : NIR + LIDAR                    (drop R, G, B)
  lidar_only  : LIDAR only                     (drop all 4 VHR bands)

Dropping is done at the batch level via masking:
  - Identify tokens whose spectral_idx (col 3) matches a dropped band
  - Set col 0 (reflectance) to 0
  - Set their attention mask to True (-> -inf in cross-attention -> zero weight)

LIDAR tokens are identified by the ELEVATION spectral_idx and are never
dropped. The latent grid layout is unchanged across ablations: same N
tokens, same Voronoi cells, same latent count. The only difference is
which tokens contribute meaningful signal at attention time.

Full-scene evaluation:
  By default this script uses eval_full_scene=True on the dataset, which
  makes queries cover ALL LIDAR points per scene (not just the
  subsampled context subset). This is the correct protocol for matching
  the RandLA-Net baseline which evaluates on all points.
  REQUIRES batch_size=1.

Usage
-----
    # All four ablations (default)
    python script_test_fractal_modality_drop.py \\
        --ckpt_path ./checkpoints/fractal/best.ckpt \\
        --xp_name fractal_drop_eval

    # Specific ablations only
    python script_test_fractal_modality_drop.py \\
        --ckpt_path ./checkpoints/fractal/best.ckpt \\
        --xp_name fractal_drop_eval \\
        --ablations all rgb_lidar nir_lidar
"""

import os
import re
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch.distributed as dist

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding, create_flairhub_bands_info
from training.utils.datasets.token_builder import TokenBuilder
from training.trainer_FRACTAL import Model_Fractal
from training.utils.datasets.utils_dataset_fractal import FractalDataset
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
# ABLATION DEFINITIONS
# =============================================================================
# Maps ablation name -> list of band NAMES to drop (from VHR).
# An empty list means "drop nothing" (full multi-modal).
# LIDAR (ELEVATION) is never dropped — these only affect VHR tokens.

ABLATION_DROP_BANDS = {
    "all":        [],                       # keep everything
    "rgb_lidar":  ["NIR"],                  # drop NIR, keep RGB + LIDAR
    "nir_lidar":  ["R", "G", "B"],          # drop RGB, keep NIR + LIDAR
    "lidar_only": ["NIR", "R", "G", "B"],   # drop all VHR, LIDAR only
}

ABLATION_DESCRIPTIONS = {
    "all":        "VHR (NIR + RGB) + LIDAR — full multi-modal",
    "rgb_lidar":  "drop NIR — keep RGB + LIDAR",
    "nir_lidar":  "drop R, G, B — keep NIR + LIDAR",
    "lidar_only": "drop all VHR tokens, keep LIDAR only",
}


def parse_ablation(name: str) -> str:
    if name not in ABLATION_DROP_BANDS:
        raise ValueError(
            f"Unknown ablation '{name}'. "
            f"Available: {list(ABLATION_DROP_BANDS)}"
        )
    return name


def resolve_dropped_spectral_indices(
    ablation_name: str,
    fractal_bands: dict,
    lookup_table,
) -> set:
    """
    Translate an ablation name into the set of spectral_idx values
    (col 3 of each token) that should be masked.

    Returns an empty set for "all" — no masking applied.
    """
    band_names_to_drop = ABLATION_DROP_BANDS[ablation_name]
    if not band_names_to_drop:
        return set()

    band_info = fractal_bands["bands_fractal_irgb_info"]
    dropped = set()
    for name in band_names_to_drop:
        if name not in band_info:
            raise KeyError(
                f"Band '{name}' (from ablation '{ablation_name}') not found "
                f"in fractal_bands config."
            )
        data = band_info[name]
        key = (int(data["bandwidth"]), int(data["central_wavelength"]))
        if key not in lookup_table.table_wave:
            raise KeyError(
                f"VHR band '{name}' key={key} not found in lookup table."
            )
        dropped.add(lookup_table.table_wave[key])
    return dropped


# =============================================================================
# MODALITY DROP — applied to a single batch dict
# =============================================================================

def drop_tokens_from_batch(batch: dict,
                            spectral_indices_to_drop: set) -> dict:
    """
    Zero the value and flag the attention mask for all tokens whose
    spectral_idx (col 3) is in spectral_indices_to_drop.

    Works on both unbatched [N, 8] and batched [B, N, 8] token tensors.
    Returns a new dict — original is not modified.

    If spectral_indices_to_drop is empty, returns the batch unchanged.
    """
    if not spectral_indices_to_drop:
        return batch

    groups_out = {}
    for res, group in batch["groups"].items():
        tokens = group["tokens"].clone()       # [B, N, 8] or [N, 8]
        mask   = group["mask"].clone().float()

        batched = tokens.dim() == 3
        spec_idx = tokens[:, :, 3] if batched else tokens[:, 3]

        drop = torch.zeros_like(spec_idx, dtype=torch.bool)
        for sid in spectral_indices_to_drop:
            drop |= (spec_idx == sid)

        if batched:
            tokens[:, :, 0][drop] = 0.0
        else:
            tokens[:, 0][drop] = 0.0

        mask[drop] = 1.0   # True = padding -> -inf in cross-attention

        groups_out[res] = {**group, "tokens": tokens, "mask": mask}

    return {**batch, "groups": groups_out}


# =============================================================================
# DATASET WRAPPER
# =============================================================================

class ModalityDropDataset(torch.utils.data.Dataset):
    """
    Wraps FractalDataset and applies modality dropping at item load time.

    spectral_indices_to_drop: set of spectral_idx values to mask. Empty
    set = no-op wrapper (used by the "all" ablation only if you want
    the wrapper for code uniformity; usually we just pass the base
    dataset directly in that case).
    """
    def __init__(self, dataset: FractalDataset,
                 spectral_indices_to_drop: set,
                 ablation_label: str = ""):
        self.dataset = dataset
        self.spectral_indices_to_drop = spectral_indices_to_drop
        self.ablation_label = ablation_label
        if spectral_indices_to_drop:
            print(f"[ModalityDropDataset] ablation={ablation_label!r}: "
                  f"masking spectral_idx={sorted(spectral_indices_to_drop)}")
        else:
            print(f"[ModalityDropDataset] ablation={ablation_label!r}: "
                  f"no masking (pass-through)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return drop_tokens_from_batch(item, self.spectral_indices_to_drop)


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(description="FRACTAL modality drop eval")
parser.add_argument("--ckpt_path",    type=str, required=True)
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--config_model", type=str,
                    default="config_test-FRACTAL.yaml")
parser.add_argument("--dataset_name", type=str, default="u_regular")
parser.add_argument("--root_path",    type=str, default="./data")
parser.add_argument("--max_lidar_points", type=int, default=100_000,
                    help="Context-side LIDAR subsampling cap. Queries cover "
                         "all points when --eval_full_scene is enabled.")
parser.add_argument("--valid_patches_file", type=str, default=None)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--batch_size",   type=int, default=1,
                    help="Per-GPU batch size for test. Forced to 1 when "
                         "--eval_full_scene is enabled (variable query "
                         "counts cannot be batched).")
parser.add_argument("--ignore_index", type=int, default=255)
parser.add_argument("--ablations",    type=str, nargs="+",
                    default=["all", "rgb_lidar", "nir_lidar", "lidar_only"],
                    help="Ablations to run. Choices: "
                         "all, rgb_lidar, nir_lidar, lidar_only")
parser.add_argument("--eval_full_scene", action="store_true", default=True,
                    help="Evaluate on full point clouds per scene "
                         "(default). Use --no_eval_full_scene to disable.")
parser.add_argument("--no_eval_full_scene", dest="eval_full_scene",
                    action="store_false",
                    help="Disable full-scene eval (subsample queries to "
                         "match training-time behavior).")
parser.add_argument("--wandb",        action="store_true")
args = parser.parse_args()


# ── Enforce batch_size=1 when full-scene eval is on ──────────────────
# Per-scene query count varies (50k - 500k+), so cannot be batched.
if args.eval_full_scene and args.batch_size != 1:
    print(f"\n[Eval] WARNING: eval_full_scene=True forces batch_size=1 "
          f"(was {args.batch_size}). Per-scene query counts vary and "
          f"cannot be collated.")
    args.batch_size = 1


# =============================================================================
# CONFIG + LOOKUP  (mirrors training script)
# =============================================================================

config_model    = read_yaml(f"./training/configs/{args.config_model}")
configs_dataset = read_yaml(
    f"./data/Tiny_BigEarthNet/configs_dataset_{args.dataset_name}.yaml"
)

fractal_bands = create_fractal_bands_info()
flair_bands   = create_flairhub_bands_info()
bands         = {**flair_bands, **fractal_bands}

lookup_table  = Lookup_encoding(configs_dataset, bands, config_model)
register_all_resolutions(lookup_table)
lookup_table.register_abstract_channel("ELEVATION")
lookup_table.register_abstract_channel("VV")
lookup_table.register_abstract_channel("VH")
lookup_table.register_abstract_channel("DSM")
lookup_table.register_abstract_channel("DTM")


# =============================================================================
# VALIDATE REQUESTED ABLATIONS UP FRONT
# =============================================================================

for name in args.ablations:
    parse_ablation(name)  # raises if unknown

print(f"\n[Eval] Will run {len(args.ablations)} ablation(s):")
for name in args.ablations:
    drop_idxs = resolve_dropped_spectral_indices(name, fractal_bands,
                                                  lookup_table)
    print(f"  - {name:<12}  {ABLATION_DESCRIPTIONS[name]}")
    print(f"      drop bands={ABLATION_DROP_BANDS[name]}, "
          f"spectral_idx={sorted(drop_idxs)}")


# =============================================================================
# LOAD MODEL ONCE
# =============================================================================

print(f"\n[Eval] Loading checkpoint: {args.ckpt_path}")
model = Model_Fractal(
    config=config_model,
    wand=False,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
    ignore_index=args.ignore_index,
    class_weights=None,
)

ckpt  = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
result = model.load_state_dict(state, strict=False)
print(f"[Eval] Missing keys: {len(result.missing_keys)}, "
      f"Unexpected keys: {len(result.unexpected_keys)}")
model.eval()


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    wandb.init(
        name=f"{args.xp_name}_modality_drop",
        project="Atomizer-FRACTAL",
        config={
            "ckpt": args.ckpt_path,
            "ablations": args.ablations,
            "eval_full_scene": args.eval_full_scene,
            "max_lidar_points": args.max_lidar_points,
        },
    )
    wandb_logger = WandbLogger(project="Atomizer-FRACTAL")


# =============================================================================
# RUN ABLATIONS
# =============================================================================

results = []

# Build the base dataset ONCE — same dataset, same scene set across ablations.
# Each ablation differs only in which spectral indices are masked at the
# batch level. This guarantees the test set is identical across rows of
# the final table (no sample-selection variance between ablations).
print(f"\n[Eval] Building base FractalDataset for test split "
      f"(eval_full_scene={args.eval_full_scene})")
base_ds = FractalDataset(
    root_path=args.root_path,
    mode="test",
    dataset_config=bands,
    config_model=config_model,
    look_up=lookup_table,
    max_lidar_points=args.max_lidar_points,
    valid_patches_file=args.valid_patches_file,
    use_augmentation=False,
    eval_full_scene=args.eval_full_scene,
)

for ablation_name in args.ablations:
    desc = ABLATION_DESCRIPTIONS[ablation_name]
    dropped_spectral_indices = resolve_dropped_spectral_indices(
        ablation_name, fractal_bands, lookup_table)

    print(f"\n{'─'*60}")
    print(f"  Ablation : {ablation_name}")
    print(f"  Desc     : {desc}")
    print(f"  Drop     : bands={ABLATION_DROP_BANDS[ablation_name]}, "
          f"spectral_idx={sorted(dropped_spectral_indices)}")
    print(f"{'─'*60}")

    # Wrap with the drop logic. When the drop set is empty ("all"), the
    # wrapper is a transparent pass-through, keeping per-ablation code
    # uniform.
    test_ds = ModalityDropDataset(
        base_ds,
        spectral_indices_to_drop=dropped_spectral_indices,
        ablation_label=ablation_name,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_grouped,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=-1,
        accelerator="gpu",
        precision="32-true",
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    test_results = trainer.test(model, test_loader, verbose=True)
    metrics      = test_results[0] if test_results else {}
    results.append((ablation_name, desc, metrics))

    if args.wandb and wandb_logger:
        import wandb as wb
        wb.log({f"{ablation_name}/{k}": v for k, v in metrics.items()})


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*70}")
print(f"  FRACTAL MODALITY DROP SUMMARY — {args.xp_name}")
print(f"  Checkpoint: {args.ckpt_path}")
print(f"  eval_full_scene: {args.eval_full_scene}")
print(f"{'='*70}")

all_keys = []
for _, _, m in results:
    for k in m:
        if k not in all_keys:
            all_keys.append(k)

print(f"\n  {'Ablation':<14} {'Description':<40}", end="")
for k in all_keys:
    print(f"  {k:<14}", end="")
print()
print(f"  {'─'*70}")

for name, desc, metrics in results:
    print(f"  {name:<14} {desc:<40}", end="")
    for k in all_keys:
        v = metrics.get(k, float("nan"))
        print(f"  {v:<14.4f}", end="")
    print()

print(f"\n{'='*70}\n")

out_path = f"./results_{args.xp_name}_fractal_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Checkpoint: {args.ckpt_path}\n")
    f.write(f"eval_full_scene: {args.eval_full_scene}\n\n")
    f.write(f"{'Ablation':<14} {'Description':<40}")
    for k in all_keys:
        f.write(f"  {k:<14}")
    f.write("\n" + "─"*70 + "\n")
    for name, desc, metrics in results:
        f.write(f"{name:<14} {desc:<40}")
        for k in all_keys:
            v = metrics.get(k, float("nan"))
            f.write(f"  {v:<14.4f}")
        f.write("\n")

print(f"[Eval] Results saved to {out_path}")

if args.wandb:
    import wandb as wb
    wb.finish()
