"""
Sen1Floods11 Atomiser (SKIP) — Modality Drop Inference Script
=============================================================

Load a trained Atomiser_Senflood_Skip checkpoint and evaluate on the test
split under different band configurations.

Unlike the baseline (which zeros channels in a [B,C,H,W] image), the Atomiser
works on tokens. Band-dropping here means MASKING the dropped bands' tokens —
exactly what the dataset's _apply_drop_mask does via bands.drop — applied
post-hoc in a wrapper so a single dataset serves all ablations.

Why masking the pool mask is sufficient for the SKIP model:
    A band-token is read in two attention paths, and BOTH read the same
    batch["groups"][res]["mask"]:
      1. encoder geographic cross-attention  (groups[res]["tokens"]/["mask"])
      2. decoder skip _pixel_skip            (gathers from the same pool/mask)
    So setting the pool mask once, before forward, covers both. The skip's
    force-keep guard already refuses to resurrect masked bands.

Token identification:
    Tokens carry spectral_idx in column 3. Band name -> spectral_idx is
    resolved exactly as the dataset does:
        key = (int(bandwidth), int(central_wavelength))
        spectral_idx = look_up.table_wave[key]
    Every token whose col-3 == a dropped spectral_idx is masked.

Usage
-----
    python script_test_senflood_skip_modality_drop.py \
        --ckpt ./checkpoints/atomiser_skip.ckpt \
        --xp_name skip_drop_eval \
        --ablations all s2_only s1_only rgb_only no_swir no_re
"""

import os
import argparse
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding

# >>> SKIP: skip trainer + dataset + collate
from training.trainer_SENFLOOD_skip import Model_SenFlood_Skip
from training.utils.datasets.utils_dataset_senflood_skip import Sen1Floods11SkipDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip


# =============================================================================
# BAND GROUPS  (names must match bands_senflood yaml keys)
# =============================================================================

ALL_S2    = ["B01","B02","B03","B04","B05","B06","B07","B08","B08A","B09","B10","B11","B12"]
ALL_S1    = ["VV","VH"]
ALL_BANDS = ALL_S2 + ALL_S1

BUILTIN_ABLATIONS = {
    "all":      [],                                                     # nothing dropped
    "s2_only":  ALL_S1,                                                 # drop S1
    "s1_only":  ALL_S2,                                                 # drop S2
    "rgb_only": [b for b in ALL_BANDS if b not in ["B02","B03","B04"]], # keep only RGB
    "no_swir":  ["B10","B11","B12"],
    "no_re":    ["B05","B06","B07","B08A"],
}


def parse_ablation(name: str):
    """Returns list of band names to drop (mask)."""
    if name in BUILTIN_ABLATIONS:
        return BUILTIN_ABLATIONS[name]
    for part in name.strip().split():
        if part.startswith("drop="):
            return [b.strip() for b in part[5:].split(",") if b.strip()]
    return []


# =============================================================================
# TOKEN-MASK WRAPPER  (Atomiser equivalent of ChannelDropWrapper)
# =============================================================================

class TokenDropWrapper(nn.Module):
    """
    Wraps the Atomiser encoder and masks dropped-band tokens before forward.

    For each token whose spectral_idx (col 3) is in `drop_spectral_indices`:
      - set its mask entry to True/1 (masked out of attention) in every
        resolution group
      - zero its reflectance (col 0)  [belt-and-suspenders; mirrors dataset]

    Because the encoder and the decoder skip both read the SAME group mask,
    this single masking covers both paths.

    The wrapper preserves the model's forward signature, including the skip
    keys (query_token_idx / query_token_valid) which it passes through
    untouched — those index the pool by ROW, not by band, so they remain
    valid; the masked bands are simply suppressed in attention.
    """

    def __init__(self, encoder: nn.Module, drop_spectral_indices: set):
        super().__init__()
        self.encoder = encoder
        self.drop_spectral_indices = set(int(s) for s in drop_spectral_indices)

    def _mask_batch(self, batch: dict) -> dict:
        if not self.drop_spectral_indices:
            return batch

        # shallow-copy the batch and the groups we mutate, so we never edit
        # the cached/loader tensors in place across ablations.
        new_batch = dict(batch)
        new_groups = {}
        for res, grp in batch["groups"].items():
            tokens = grp["tokens"]
            mask   = grp["mask"]

            tokens = tokens.clone()
            mask   = mask.clone()
            # ensure float/bool consistency with dataset (mask: 1.0 = padded)
            if mask.dtype == torch.bool:
                mask_is_bool = True
            else:
                mask_is_bool = False

            spec = tokens[..., 3]                      # [B, N] spectral_idx
            drop = torch.zeros_like(spec, dtype=torch.bool)
            for sid in self.drop_spectral_indices:
                drop |= (spec == sid)

            tokens[..., 0] = torch.where(
                drop, torch.zeros_like(tokens[..., 0]), tokens[..., 0])

            if mask_is_bool:
                mask = mask | drop
            else:
                mask = torch.maximum(mask, drop.to(mask.dtype))

            new_grp = dict(grp)
            new_grp["tokens"] = tokens
            new_grp["mask"]   = mask
            new_groups[res] = new_grp

        new_batch["groups"] = new_groups
        return new_batch

    def forward(self, batch, **kwargs):
        return self.encoder(self._mask_batch(batch), **kwargs)


# =============================================================================
# DROP RESOLUTION  (band name -> spectral_idx, exactly as the dataset does)
# =============================================================================

def resolve_drop_spectral_indices(drop_bands, bands_info, look_up):
    """
    Map band names to their spectral_idx values via the same lookup the
    dataset uses: key = (int(bandwidth), int(central_wavelength)).
    """
    dropped = set()
    for name in drop_bands:
        if name not in bands_info:
            raise KeyError(f"Band '{name}' not in bands_senflood metadata.")
        data = bands_info[name]
        key = (int(data["bandwidth"]), int(data["central_wavelength"]))
        if key not in look_up.table_wave:
            raise KeyError(f"Band '{name}' key={key} not in lookup table_wave.")
        dropped.add(int(look_up.table_wave[key]))
    return dropped


# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        type=str, required=True,
                    help="Path to the Atomiser_Senflood_Skip checkpoint")
parser.add_argument("--xp_name",     type=str, required=True)
parser.add_argument("--data_dir",    type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ablations",   type=str, nargs="+",
                    default=["all","s2_only","s1_only","rgb_only","no_swir","no_re"])
parser.add_argument("--wandb",       action="store_true")

# config paths (match the training launch script)
parser.add_argument("--config",      type=str,
                    default="./training/configs/config_test-SENFLOOD.yaml")
parser.add_argument("--configs_dataset", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--bands_yaml",  type=str,
                    default="./data/bands_info/bands.yaml")
args = parser.parse_args()


# =============================================================================
# CONFIG + LOOKUP
# =============================================================================

config_model    = read_yaml(args.config)
bands_yaml_dict = read_yaml(args.bands_yaml)
lookup_table    = Lookup_encoding(
    read_yaml(args.configs_dataset), bands_yaml_dict, config_model)

bands_info = bands_yaml_dict["bands_senflood"]

# Sanity: this eval is for the skip model. Warn if the checkpoint config
# wasn't trained with the skip on (the wrapper still works, but the point
# of this eval is the skip model).
if not config_model.get("Atomiser", {}).get("use_decoder_skip", False):
    print("[Eval][WARN] config has use_decoder_skip=False. This script targets "
          "the skip model; ensure the config matches the checkpoint.")


# =============================================================================
# WANDB
# =============================================================================

wandb_logger = None
if args.wandb and os.environ.get("LOCAL_RANK", "0") == "0":
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb.init(
        name=f"{args.xp_name}_skip_drop",
        project="SenFlood",
        config={"ckpt": args.ckpt, "ablations": args.ablations},
    )
    wandb_logger = WandbLogger(project="SenFlood")


# =============================================================================
# DATA MODULE  (single dataset, no band selection — drops applied in wrapper)
# =============================================================================

data_module = UnifiedDataModule(
    path=args.data_dir,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=args.num_workers,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=bands_yaml_dict,
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=Sen1Floods11SkipDataset,
    collate_fn=collate_grouped_skip,
)
data_module.setup("test")
print(f"[Eval] Test set: {len(data_module.test_dataset)} samples")


# =============================================================================
# LOAD MODEL
# =============================================================================

model = Model_SenFlood_Skip.load_from_checkpoint(
    args.ckpt,
    strict=False,
    config=config_model,
    wand=False,
    name=args.xp_name,
    transform=None,
    lookup_table=lookup_table,
)
model.eval()

# keep a handle to the real encoder so we can rewrap per ablation
base_encoder = model.encoder


# =============================================================================
# RUN ABLATIONS
# =============================================================================

all_results = {}

for ablation_name in args.ablations:
    drop_bands = parse_ablation(ablation_name)
    drop_sids  = resolve_drop_spectral_indices(drop_bands, bands_info, lookup_table)
    drop_str   = ",".join(drop_bands) if drop_bands else "none"

    print(f"\n  {'─'*50}")
    print(f"  Ablation : {ablation_name}   Drop : {drop_str}")
    print(f"  spectral_idx dropped : {sorted(drop_sids) if drop_sids else 'none'}")
    print(f"  {'─'*50}")

    # Wrap the encoder so the trainer's self.encoder(batch, ...) masks tokens.
    model.encoder = TokenDropWrapper(base_encoder, drop_sids)

    trainer = Trainer(
        devices=1,                      # single-device eval avoids DDP metric sync edge cases
        accelerator="gpu",
        precision="bf16-mixed",
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    results = trainer.test(model, datamodule=data_module, verbose=True)
    metrics = results[0] if results else {}
    all_results[ablation_name] = metrics

    if args.wandb and wandb_logger:
        import wandb
        wandb.log({f"{ablation_name}/{k}": v for k, v in metrics.items()})

# restore the real encoder
model.encoder = base_encoder


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  ATOMISER-SKIP MODALITY DROP SUMMARY — {args.xp_name}")
print(f"{'='*80}")

sample_metrics = next((m for m in all_results.values() if m), {})
metric_keys = list(sample_metrics.keys())

for mkey in metric_keys:
    print(f"\n  Metric: {mkey}")
    print(f"  {'Ablation':<14} {'Drop':<40} {'value':<12}")
    print(f"  {'─'*70}")
    for abl in args.ablations:
        drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
        v = all_results.get(abl, {}).get(mkey, float("nan"))
        print(f"  {abl:<14} {drop_str:<40} {v:<12.4f}")

# Flat test_mIoU table + file
out_path = f"./results_{args.xp_name}_skip_modality_drop.txt"
with open(out_path, "w") as f:
    f.write(f"Experiment: {args.xp_name}\n")
    f.write(f"Checkpoint: {args.ckpt}\n\n")
    f.write(f"{'Ablation':<14} {'Drop':<40} {'test_mIoU':<12}\n")
    f.write("─"*70 + "\n")
    for abl in args.ablations:
        drop_str = ",".join(parse_ablation(abl)) if parse_ablation(abl) else "none"
        v = all_results.get(abl, {}).get("test_mIoU", float("nan"))
        f.write(f"{abl:<14} {drop_str:<40} {v:<12.4f}\n")

print(f"\n[Eval] Results saved to {out_path}")

if args.wandb:
    import wandb
    wandb.finish()
