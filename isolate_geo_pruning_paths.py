"""
isolate_geo_pruning_paths.py
================================

Isolation test: does GeographicPruningDales's FALLBACK (on-the-fly,
patch_ids) path produce meaningfully different results than the
AUTHORITATIVE (precomputed, token_latent_assignment) path -- the one the
model has actually been trained/validated with exclusively?

Runs the SAME real tiled patch through the model:
    1. Precomputed path, TWICE (same input) -- baseline self-agreement
       noise, since _build_cell_from_assignment reshuffles per-call even
       in eval mode (tie-breaking within each latent's cell before
       slicing the first k -- this is NOT deterministic across calls).
    2. Fallback path, TWICE (same input) -- fallback's own self-agreement
       noise.
    3. Precomputed vs. fallback -- the comparison that actually matters.

If (3) is meaningfully worse than (1)/(2), that's a real path-specific
issue. If (3) looks similar to the (1)/(2) baseline noise level, the
earlier sliding-window score drop was very likely just the degenerate
windowing bug (tiled patches passed as "scenes"), not this path.

Usage:
    python isolate_geo_pruning_paths.py \
        --root_path ./data \
        --config_model config_test-DALES.yaml \
        --ckpt_path <ckpt> \
        --split test \
        --patch_index 0
"""

import argparse

import torch

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.token_builder import TokenBuilder
from training.utils.datasets.utils_dataset_dales import DalesDataset
from training.utils.datasets.token_grouping import collate_grouped
from training.atomiser.Atomiser_dales import Atomiser_Dales


def move_batch(b, device):
    out = {}
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                res: {gk: (gv.to(device) if isinstance(gv, torch.Tensor) else gv)
                      for gk, gv in g.items()}
                for res, g in v.items()
            }
        else:
            out[k] = v
    return out


def compare(name_a, logits_a, name_b, logits_b):
    diff = (logits_a - logits_b).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()

    pred_a = logits_a.argmax(dim=-1)
    pred_b = logits_b.argmax(dim=-1)
    disagree_frac = (pred_a != pred_b).float().mean().item()

    print(f"  {name_a} vs {name_b}:")
    print(f"    mean |logit diff| = {mean_abs:.6f}")
    print(f"    max  |logit diff| = {max_abs:.6f}")
    print(f"    argmax disagreement = {disagree_frac*100:.3f}% of queries")
    return mean_abs, max_abs, disagree_frac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--config_model", type=str,
                         default="config_test-DALES.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                         choices=["train", "val", "test"])
    parser.add_argument("--patch_index", type=int, default=0,
                         help="Index into the dataset's patch list to test")
    parser.add_argument("--max_lidar_points", type=int, default=256_000)
    args = parser.parse_args()

    config_model = read_yaml(f"./training/configs/{args.config_model}")
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"
    configs_dataset = read_yaml(configs_dataset_path)
    bands = {}

    lookup_table = Lookup_encoding(configs_dataset, bands, config_model)
    TokenBuilder.REFERENCE_SIZES[0.2] = 2048
    lookup_table.get_or_register_modality(0.2, 2048)
    lookup_table.get_resolution_idx(0.2)
    lookup_table.register_abstract_channel("ELEVATION")

    print(f"[isolate] Building {args.split} dataset...")
    ds = DalesDataset(
        root_path=args.root_path, mode=args.split,
        dataset_config=bands, config_model=config_model,
        look_up=lookup_table, max_lidar_points=args.max_lidar_points,
        use_augmentation=False,  # deterministic input, isolate the geo-pruning variable only
    )
    sample = ds[args.patch_index]
    print(f"[isolate] Testing patch: {sample['patch_id']}")

    batch_precomputed = collate_grouped([sample])
    assert "token_latent_assignment" in batch_precomputed, (
        "Sample has no token_latent_assignment -- did precompute run for "
        "this patch?"
    )

    # Build the FALLBACK-forcing batch: same everything, but with
    # token_latent_assignment removed so GeographicPruningDales.forward()
    # falls through to the on-the-fly (patch_ids) path.
    batch_fallback = dict(batch_precomputed)
    batch_fallback.pop("token_latent_assignment")
    assert "patch_id" in batch_fallback, (
        "Sample has no patch_id -- fallback path needs this to trigger."
    )

    print(f"[isolate] Loading checkpoint: {args.ckpt_path}")
    model = Atomiser_Dales(config=config_model, lookup_table=lookup_table)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_precomputed = move_batch(batch_precomputed, device)
    batch_fallback = move_batch(batch_fallback, device)

    with torch.no_grad():
        print("\n[isolate] Running precomputed path (call 1)...")
        logits_precomp_1 = model(batch_precomputed, training=False)

        print("[isolate] Running precomputed path (call 2)...")
        logits_precomp_2 = model(batch_precomputed, training=False)

        print("[isolate] Running fallback path (call 1)...")
        logits_fallback_1 = model(batch_fallback, training=False)

        print("[isolate] Running fallback path (call 2)...")
        logits_fallback_2 = model(batch_fallback, training=False)

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}\n")

    print("Baseline self-agreement (same path, called twice):")
    base_precomp = compare("precomputed#1", logits_precomp_1,
                            "precomputed#2", logits_precomp_2)
    base_fallback = compare("fallback#1", logits_fallback_1,
                             "fallback#2", logits_fallback_2)

    print("\nCross-path comparison (the one that actually matters):")
    cross = compare("precomputed#1", logits_precomp_1,
                     "fallback#1", logits_fallback_1)

    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    baseline_disagree = max(base_precomp[2], base_fallback[2])
    cross_disagree = cross[2]
    print(f"  Baseline (same-path) argmax disagreement: up to "
          f"{baseline_disagree*100:.3f}%")
    print(f"  Cross-path argmax disagreement:            "
          f"{cross_disagree*100:.3f}%")

    if cross_disagree > baseline_disagree * 3 and cross_disagree > 0.5:
        print(f"\n  -> Cross-path disagreement is MEANINGFULLY HIGHER than "
              f"baseline noise.\n"
              f"     This points to a REAL issue specific to the fallback "
              f"path, not just per-call reshuffling noise.")
    else:
        print(f"\n  -> Cross-path disagreement is IN LINE WITH baseline "
              f"noise.\n"
              f"     The fallback path itself looks fine -- the earlier "
              f"sliding-window score drop was very likely the degenerate "
              f"windowing bug (tiled patches passed as raw scenes), not "
              f"this geo-pruning path.")


if __name__ == "__main__":
    main()
