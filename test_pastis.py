"""
Standalone verification of the PASTIS skip gather index — RUN BEFORE TRAINING.
=============================================================================

Loads ONE real sample through PastisHDDataset and checks that, for every query,
gathering sat_tokens[query_token_idx[i]] returns atoms that all sit at that
query's (x, y) and span the expected bands x timesteps.

Token cols: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
                     1  2       3                                              7
"""

import argparse
import sys
import os
import torch

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

from training.utils import read_yaml, Lookup_encoding
from training.utils.datasets.utils_dataset_PASTIS import PastisHDDataset
from training.utils.datasets.token_builder import TokenBuilder


# Mirror train_pastis.py's resolution registration EXACTLY.
ALL_KNOWN_RESOLUTIONS = {
    1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048,
}


def _register_all_resolutions(lookup_table):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


def build_lookup_and_config(args, Lookup_encoding):
    """Mirror train_pastis.py setup so reference-grid offsets match the 0.38 run."""
    config_model_path    = "./training/configs/config_test-Atomiser_Atos_One.yaml"
    bands_yaml_path      = "./data/bands_info/bands.yaml"
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

    config_model = read_yaml(config_model_path)
    if "dataset" not in config_model:
        config_model["dataset"] = {}
    config_model["dataset"].setdefault("multi_temporal", 10)

    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset_path),
        read_yaml(bands_yaml_path),
        config_model,
    )
    _register_all_resolutions(lookup_table)
    if not args.no_s1:
        lookup_table.register_abstract_channel("VV_VH")
    return lookup_table, config_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data/PASTIS-HD")
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "validation", "test"])
    ap.add_argument("--no_s1", action="store_true")
    ap.add_argument("--no_spot", action="store_true")
    ap.add_argument("--n_check", type=int, default=200)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    look_up, config_model = build_lookup_and_config(args, Lookup_encoding)

    ds = PastisHDDataset(
        root_path=args.data_dir, mode=args.split,
        config_model=config_model, look_up=look_up,
        use_s1=not args.no_s1, use_spot=not args.no_spot,
    )
    print(f"[verify] dataset built: {len(ds)} samples, split={args.split}, "
          f"use_s1={not args.no_s1}, use_spot={not args.no_spot}")

    sample = ds[args.index]

    sat_res = ds.SAT_RESOLUTION
    sat_tokens = sample["groups"][sat_res]["tokens"]
    H, W = sample["groups"][sat_res]["shape"]
    queries = sample["tasks"][ds.TASK_NAME]["queries"]
    qti = sample["query_token_idx"]
    qtv = sample["query_token_valid"]

    N_sat = sat_tokens.shape[0]
    N_q, A = qti.shape
    print(f"[verify] sat_tokens={tuple(sat_tokens.shape)}, queries={tuple(queries.shape)}, "
          f"query_token_idx={tuple(qti.shape)} (A={A} atoms/pixel), H×W={H}×{W}")

    assert qti.min() >= 0 and qti.max() < N_sat, (
        f"INDEX OUT OF RANGE: min={qti.min()}, max={qti.max()}, N_sat={N_sat}.")
    assert bool(qtv.all()), "query_token_valid has False entries"
    print("[verify] PASS: indices in range, all valid")

    C2 = ds.NUM_S2_BANDS
    C1 = ds.NUM_S1_BANDS if not args.no_s1 else 0
    HW = H * W
    assert N_sat == A * HW, (
        f"POOL/INDEX MISMATCH: N_sat={N_sat} != A*HW={A*HW}.")
    print(f"[verify] PASS: N_sat == A*HW ({N_sat} == {A}*{HW})")

    # ── TIME_IDX DIAGNOSTIC: is the temporal indexing actually varying? ──
    # The band×time WARN below is benign IF time_idx genuinely varies across
    # frames (duplicate acquisition dates collapse a couple). It is a BUG if
    # time_idx is near-constant (temporal signal dead -> model can't use time).
    time_col = sat_tokens[:, 7]
    uniq_t = torch.unique(time_col)
    counts = {int(t): int((time_col == t).sum()) for t in uniq_t}
    print(f"[verify] TIME_IDX: {len(uniq_t)} distinct value(s) in pool: "
          f"{[int(t) for t in uniq_t]}")
    print(f"[verify] TIME_IDX counts per value: {counts}")
    expected_frames = config_model.get("dataset", {}).get("multi_temporal", 10)
    if len(uniq_t) >= max(2, expected_frames - 2):
        print(f"[verify]   -> looks HEALTHY (~{expected_frames} expected; a couple "
              f"collisions from duplicate dates is normal). Temporal signal present.")
    else:
        print(f"[verify]   *** WARNING: only {len(uniq_t)} distinct time_idx but "
              f"{expected_frames} frames expected. Temporal encoding may be DEAD "
              f"(broken _doy_to_time_indices?). Investigate before training — the "
              f"skip-vs-no-skip comparison would be on a temporally-crippled model. ***")

    xq = queries[:, 1]
    yq = queries[:, 2]
    n_check = min(args.n_check, N_q)
    check_ids = torch.linspace(0, N_q - 1, n_check).long().unique()

    bad = 0
    band_time_ok = 0
    for i in check_ids.tolist():
        rows = qti[i]
        atoms = sat_tokens[rows]
        ax, ay = atoms[:, 1], atoms[:, 2]
        if not (torch.allclose(ax, xq[i].expand_as(ax)) and
                torch.allclose(ay, yq[i].expand_as(ay))):
            bad += 1
            if bad <= 5:
                print(f"  [MISMATCH] query {i}: query(x,y)=({xq[i]:.1f},{yq[i]:.1f}) "
                      f"gathered x∈[{ax.min():.1f},{ax.max():.1f}], "
                      f"y∈[{ay.min():.1f},{ay.max():.1f}]")
            continue
        spec, tim = atoms[:, 3], atoms[:, 7]
        if len({*zip(spec.tolist(), tim.tolist())}) == A:
            band_time_ok += 1

    if bad == 0:
        print(f"[verify] PASS: all {len(check_ids)} checked queries gather atoms "
              f"at their own (x,y)")
    else:
        print(f"[verify] *** FAIL: {bad}/{len(check_ids)} queries gather the WRONG "
              f"pixel's atoms. DO NOT TRAIN. ***")
        sys.exit(1)

    if band_time_ok == len(check_ids):
        print(f"[verify] PASS: every checked query spans A={A} distinct (band,time) atoms")
    else:
        print(f"[verify] WARN: {len(check_ids)-band_time_ok}/{len(check_ids)} queries "
              f"did NOT span A distinct (band,time) pairs — expected given the "
              f"TIME_IDX diagnostic above (duplicate dates). Benign for the gather "
              f"index (rows are still distinct & correct).")

    sig = {tuple(qti[i].tolist()) for i in check_ids.tolist()}
    if len(sig) == len(check_ids):
        print("[verify] PASS: checked queries have distinct atom-row sets")
    else:
        print(f"[verify] *** FAIL: duplicate atom-row sets ({len(check_ids)-len(sig)} "
              f"collisions). ***")
        sys.exit(1)

    print("\n[verify] ALL CHECKS PASSED — gather index correct. Review the TIME_IDX "
          "diagnostic above; if healthy, safe to train.")


if __name__ == "__main__":
    main()
