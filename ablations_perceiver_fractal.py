"""
FRACTAL PerceiverIO — Modality Drop Inference Script
=====================================================

Mirror of script_test_fractal_modality_drop.py for the PerceiverIO baseline.

Load a trained PerceiverIO FRACTAL checkpoint and evaluate on the test split
under four test-time modality configurations:

  all         : VHR (NIR + R + G + B) + LIDAR  (no drop, headline number)
  rgb_lidar   : RGB + LIDAR                    (drop NIR)
  nir_lidar   : NIR + LIDAR                    (drop R, G, B)
  lidar_only  : LIDAR only                     (drop all 4 VHR bands)

Dropping mechanism (differs from Atomizer version):
  In the Atomizer dataset, tokens carry a spectral_idx column that
  identifies which band they belong to. In the PerceiverIO dataset,
  VHR tokens are one-per-pixel with all 4 bands concatenated into a
  single 262-dim vector — there are no per-band tokens to individually
  mask.

  Instead we zero out the Fourier-encoded band columns directly:
    - NIR  occupies cols   0:33   of vhr_tokens
    - R    occupies cols  33:66
    - G    occupies cols  66:99
    - B    occupies cols  99:132
    - Y    occupies cols 132:197  (position, never dropped)
    - X    occupies cols 197:262  (position, never dropped)

  For each dropped band we zero its 33 Fourier columns and do NOT
  add the token to the attention mask (the pixel position is still
  valid — only the spectral values are nulled). This matches the
  Atomizer protocol of zeroing the reflectance while keeping the
  token present.

  For lidar_only, all VHR tokens are zeroed AND masked out entirely
  (True in vhr_mask) so the encoder receives no VHR signal at all.

Full-scene evaluation:
  eval_full_scene=True (default) makes queries cover ALL LIDAR points
  per scene. REQUIRES batch_size=1.

Usage
-----
    # All four ablations (default)
    python script_test_fractal_perceiver_modality_drop.py \\
        --ckpt_path ./checkpoints/fractal_perceiver/best.ckpt \\
        --xp_name perceiver_fractal_drop_eval

    # Specific ablations only
    python script_test_fractal_perceiver_modality_drop.py \\
        --ckpt_path ./checkpoints/fractal_perceiver/best.ckpt \\
        --xp_name perceiver_fractal_drop_eval \\
        --ablations all rgb_lidar lidar_only
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

seed_everything(42, workers=True)

from training.trainer_fractal_perceiver import Model_PerceiverFractal
from training.utils.datasets_baselines.utils_dataset_fractal_perceiver import (
    FractalPerceiverDataset,
    VAL_DIM,   # 33 — Fourier features per band value axis
)


# =============================================================================
# VHR TOKEN BAND COLUMN LAYOUT
# =============================================================================
# vhr_tokens: [N_vhr, 262]
# Layout: [Fourier(NIR), Fourier(R), Fourier(G), Fourier(B), Fourier(Y), Fourier(X)]
#          0:33           33:66        66:99        99:132      132:197    197:262

BAND_COLS = {
    "NIR": (0,   VAL_DIM),           # 0:33
    "R":   (VAL_DIM,   2 * VAL_DIM), # 33:66
    "G":   (2 * VAL_DIM, 3 * VAL_DIM), # 66:99
    "B":   (3 * VAL_DIM, 4 * VAL_DIM), # 99:132
}


# =============================================================================
# ABLATION DEFINITIONS
# =============================================================================

ABLATION_DROP_BANDS = {
    "all":        [],
    "rgb_lidar":  ["NIR"],
    "nir_lidar":  ["R", "G", "B"],
    "lidar_only": ["NIR", "R", "G", "B"],
}

ABLATION_DESCRIPTIONS = {
    "all":        "VHR (NIR + RGB) + LIDAR — full multi-modal",
    "rgb_lidar":  "drop NIR — keep RGB + LIDAR",
    "nir_lidar":  "drop R, G, B — keep NIR + LIDAR",
    "lidar_only": "drop all VHR tokens, keep LIDAR only",
}


# =============================================================================
# MODALITY DROP — applied to a single batch dict
# =============================================================================

def drop_vhr_bands_from_batch(batch: dict, bands_to_drop: list) -> dict:
    """
    Zero the Fourier columns of dropped bands in vhr_tokens.
    For lidar_only (all bands dropped), also set vhr_mask to True
    so the encoder receives no VHR signal.

    Args:
        batch:         dict from FractalPerceiverDataset DataLoader.
        bands_to_drop: list of band name strings, e.g. ["NIR"] or
                       ["NIR", "R", "G", "B"]. Empty list = no-op.

    Returns:
        New batch dict with modified vhr_tokens and vhr_mask.
        Original tensors are not modified (clone on write).
    """
    if not bands_to_drop:
        return batch

    vhr_tokens = batch["vhr_tokens"].clone()   # [B, N_vhr, 262]
    vhr_mask   = batch["vhr_mask"].clone()     # [B, N_vhr]

    # Zero the Fourier columns for each dropped band
    for band_name in bands_to_drop:
        if band_name not in BAND_COLS:
            raise ValueError(
                f"Unknown band '{band_name}'. "
                f"Valid options: {list(BAND_COLS.keys())}"
            )
        col_start, col_end = BAND_COLS[band_name]
        vhr_tokens[..., col_start:col_end] = 0.0

    # For lidar_only: mask out all VHR tokens entirely so they contribute
    # nothing to cross-attention (same behavior as Atomizer's mask=True)
    all_vhr_dropped = set(bands_to_drop) == {"NIR", "R", "G", "B"}
    if all_vhr_dropped:
        vhr_mask = torch.ones_like(vhr_mask, dtype=torch.bool)

    return {**batch, "vhr_tokens": vhr_tokens, "vhr_mask": vhr_mask}


# =============================================================================
# DATASET WRAPPER
# =============================================================================

class ModalityDropPerceiverDataset(Dataset):
    """
    Wraps FractalPerceiverDataset and applies modality dropping at item
    load time. bands_to_drop is a list of band name strings to zero out.
    Empty list = transparent pass-through.
    """

    def __init__(
        self,
        dataset: FractalPerceiverDataset,
        bands_to_drop: list,
        ablation_label: str = "",
    ):
        self.dataset       = dataset
        self.bands_to_drop = bands_to_drop
        self.ablation_label = ablation_label
        if bands_to_drop:
            cols = {b: BAND_COLS[b] for b in bands_to_drop}
            print(f"[ModalityDropPerceiverDataset] ablation={ablation_label!r}: "
                  f"zeroing bands={bands_to_drop}, cols={cols}")
            if set(bands_to_drop) == {"NIR", "R", "G", "B"}:
                print(f"[ModalityDropPerceiverDataset]   + masking all VHR tokens")
        else:
            print(f"[ModalityDropPerceiverDataset] ablation={ablation_label!r}: "
                  f"no masking (pass-through)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return drop_vhr_bands_from_batch(item, self.bands_to_drop)


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser(
    description="FRACTAL PerceiverIO modality drop eval"
)
parser.add_argument("--ckpt_path",    type=str, required=True)
parser.add_argument("--xp_name",      type=str, required=True)
parser.add_argument("--root_path",    type=str, default="./data")
parser.add_argument("--max_lidar_points", type=int, default=100_000)
parser.add_argument("--valid_patches_file", type=str, default=None)
parser.add_argument("--num_workers",  type=int, default=4)
parser.add_argument("--batch_size",   type=int, default=1)
parser.add_argument("--query_chunk_size", type=int, default=100_000,
                    help="Decode queries in chunks of this size to avoid "
                         "OOM during full-scene eval. Default 100_000.")
parser.add_argument("--ignore_index", type=int, default=255)

# Model architecture — must match the trained checkpoint
parser.add_argument("--num_latents",         type=int,   default=1024)
parser.add_argument("--latent_dim",          type=int,   default=768)
parser.add_argument("--depth",               type=int,   default=1)
parser.add_argument("--cross_heads",         type=int,   default=16)
parser.add_argument("--latent_heads",        type=int,   default=8)
parser.add_argument("--cross_dim_head",      type=int,   default=128)
parser.add_argument("--latent_dim_head",     type=int,   default=128)
parser.add_argument("--self_per_cross_attn", type=int,   default=6)
parser.add_argument("--echo_hidden_dim",     type=int,   default=64)

parser.add_argument("--ablations", type=str, nargs="+",
                    default=["all", "rgb_lidar", "nir_lidar", "lidar_only"],
                    help="Ablations to run.")
parser.add_argument("--eval_full_scene", action="store_true", default=True)
parser.add_argument("--no_eval_full_scene", dest="eval_full_scene",
                    action="store_false")
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

# Enforce batch_size=1 for full-scene eval
if args.eval_full_scene and args.batch_size != 1:
    print(f"\n[Eval] WARNING: eval_full_scene=True forces batch_size=1 "
          f"(was {args.batch_size}).")
    args.batch_size = 1

# Validate ablation names
for name in args.ablations:
    if name not in ABLATION_DROP_BANDS:
        raise ValueError(
            f"Unknown ablation '{name}'. "
            f"Available: {list(ABLATION_DROP_BANDS)}"
        )

print(f"\n{'='*70}")
print(f"  FRACTAL PerceiverIO — Modality Drop Eval: {args.xp_name}")
print(f"  Checkpoint: {args.ckpt_path}")
print(f"  eval_full_scene: {args.eval_full_scene}")
print(f"{'='*70}")
print(f"\n[Eval] Will run {len(args.ablations)} ablation(s):")
for name in args.ablations:
    print(f"  - {name:<12}  {ABLATION_DESCRIPTIONS[name]}")
    print(f"      drop bands={ABLATION_DROP_BANDS[name]}")


# =============================================================================
# LOAD MODEL ONCE
# =============================================================================

print(f"\n[Eval] Loading checkpoint: {args.ckpt_path}")
model = Model_PerceiverFractal(
    num_latents=args.num_latents,
    latent_dim=args.latent_dim,
    depth=args.depth,
    cross_heads=args.cross_heads,
    latent_heads=args.latent_heads,
    cross_dim_head=args.cross_dim_head,
    latent_dim_head=args.latent_dim_head,
    self_per_cross_attn=args.self_per_cross_attn,
    echo_hidden_dim=args.echo_hidden_dim,
    ignore_index=args.ignore_index,
    class_weights=None,
    query_chunk_size=args.query_chunk_size,
)

ckpt   = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
state  = ckpt.get("state_dict", ckpt)
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
            "ckpt":             args.ckpt_path,
            "ablations":        args.ablations,
            "eval_full_scene":  args.eval_full_scene,
            "max_lidar_points": args.max_lidar_points,
        },
    )
    wandb_logger = WandbLogger(project="Atomizer-FRACTAL")


# =============================================================================
# BUILD BASE DATASET ONCE
# =============================================================================

print(f"\n[Eval] Building base FractalPerceiverDataset for test split "
      f"(eval_full_scene={args.eval_full_scene})")
base_ds = FractalPerceiverDataset(
    root_path=args.root_path,
    mode="test",
    max_lidar_points=args.max_lidar_points,
    valid_patches_file=args.valid_patches_file,
    use_augmentation=False,
    eval_full_scene=args.eval_full_scene,
)


# =============================================================================
# RUN ABLATIONS
# =============================================================================

results = []

for ablation_name in args.ablations:
    bands_to_drop = ABLATION_DROP_BANDS[ablation_name]
    desc          = ABLATION_DESCRIPTIONS[ablation_name]

    print(f"\n{'─'*60}")
    print(f"  Ablation : {ablation_name}")
    print(f"  Desc     : {desc}")
    print(f"  Drop     : {bands_to_drop}")
    print(f"{'─'*60}")

    test_ds = ModalityDropPerceiverDataset(
        base_ds,
        bands_to_drop=bands_to_drop,
        ablation_label=ablation_name,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
print(f"  FRACTAL PERCEIVER MODALITY DROP SUMMARY — {args.xp_name}")
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

# Save to file
out_path = f"./results_{args.xp_name}_perceiver_modality_drop.txt"
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
