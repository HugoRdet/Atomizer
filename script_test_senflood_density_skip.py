"""
Sen1Floods11 (SKIP) — Test ONE checkpoint under a given (tpl, cross_k)
=====================================================================

Inference-time density generalization for the SKIP model. Loads a single
trained Atomiser_Senflood_Skip checkpoint and evaluates on the test split at
an arbitrary latent density (tokens_per_latent) and cross-attention budget
(cross_k). No training.

IMPORTANT — this uses the SKIP stack:
    Model_SenFlood_Skip + Sen1Floods11SkipDataset + collate_grouped_skip
A previous non-skip version silently dropped the pixel_query / pixel_cross_attn
weights (strict=False), disabling the skip cascade and lowering the score.

Density override:
    TEST reads val_sampling (sample_config(training=False) -> val_sampling).
    We set BOTH train_sampling and val_sampling to [[tpl, ck]] and pass the
    patched config to load_from_checkpoint so it overrides the checkpoint's
    saved hyperparameters. To reproduce the trained test score, use the
    config's val_sampling value (here: tpl=2000, cross_k=1000).

Emits a single parseable line:
    RESULT tpl=<T> cross_k=<K> test_mIoU=<V> test_accuracy=<A>
"""

import os
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

from training.utils import read_yaml
from training.utils import Lookup_encoding

# >>> SKIP stack (must match the checkpoint's architecture)
from training.trainer_SENFLOOD_skip import Model_SenFlood_Skip
from training.utils.datasets.utils_dataset_senflood_skip import Sen1Floods11SkipDataset
from training.utils.datasets.dataloaders import UnifiedDataModule
from training.utils.datasets.collate_grouped_skip import collate_grouped_skip

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Test one SKIP ckpt at a given density")
parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
parser.add_argument("--xp_name", type=str, default="density_eval")
parser.add_argument("--tokens_per_latent", type=int, required=True)
parser.add_argument("--cross_k", type=int, required=True)
parser.add_argument("--config", type=str,
                    default="./training/configs/config_test-SENFLOOD.yaml")
parser.add_argument("--configs_dataset", type=str,
                    default="./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml")
parser.add_argument("--bands_yaml", type=str,
                    default="./data/bands_info/bands.yaml")
parser.add_argument("--data_dir", type=str, default="./data/SENFLOOD")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

tpl, ck = int(args.tokens_per_latent), int(args.cross_k)

# =============================================================================
# CONFIG + DENSITY OVERRIDE  (before model build)
# =============================================================================
config_model = read_yaml(args.config)
config_model.setdefault("latent_grid", {})
config_model["latent_grid"]["train_sampling"] = [[tpl, ck]]
config_model["latent_grid"]["val_sampling"]   = [[tpl, ck]]
print(f"[Test] Density override -> [[{tpl}, {ck}]] "
      f"(tokens_per_latent={tpl}, cross_k={ck})")

# sanity: this worker targets the skip model
if not config_model.get("Atomiser", {}).get("use_decoder_skip", False):
    print("[Test][WARN] config has use_decoder_skip=False but this is the SKIP "
          "worker. Ensure the config matches the skip checkpoint.")

lookup_table = Lookup_encoding(
    read_yaml(args.configs_dataset), read_yaml(args.bands_yaml), config_model)

# =============================================================================
# LOAD CHECKPOINT WITH PATCHED CONFIG  (SKIP model)
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

# =============================================================================
# DATA MODULE  (SKIP dataset + collate)
# =============================================================================
data_module = UnifiedDataModule(
    path=args.data_dir,
    batch_size=config_model["trainer"]["train_batch_size"],
    num_workers=args.num_workers,
    trans_modalities=None,
    trans_tokens=None,
    model=config_model["encoder"],
    dataset_config=read_yaml(args.bands_yaml),
    config_model=config_model,
    look_up=lookup_table,
    dataset_class=Sen1Floods11SkipDataset,
    collate_fn=collate_grouped_skip,
)

# =============================================================================
# TEST
# =============================================================================
trainer = Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=False,
    enable_progress_bar=True,
    enable_model_summary=False,
)

results = trainer.test(model, datamodule=data_module, verbose=True)
metrics = results[0] if results else {}

miou = metrics.get("test_mIoU", float("nan"))
acc  = metrics.get("test_accuracy", float("nan"))


# =============================================================================
# GFLOPS MEASUREMENT (torch.profiler, after scoring)
# =============================================================================
# Profile a few real test forward passes in eval/no_grad and average the
# profiler-reported FLOPs. Notes:
#   - with_flops=True counts recognized ops (matmul/conv); some elementwise/
#     custom ops are NOT counted, so the absolute number is a LOWER BOUND.
#     It is consistent across configs, so the cross-config COMPARISON is the
#     trustworthy part.
#   - one warmup pass is run and discarded (CUDA init / autotune).
#   - profiler "FLOPs" are multiply-adds counted as the op defines; we divide
#     by 1e9 to report GFLOPs PER FORWARD (per tile / per sample at bs=1).

def _to_device(b, dev):
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: _to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_device(v, dev) for v in b)
    return b

gflops = float("nan")

# Everything is written under ./profiler/<tag>/ where tag identifies this config.
PROFILE_DIR = "./profiler"
tag = f"tpl{tpl}_ck{ck}"
out_dir = os.path.join(PROFILE_DIR, tag)
os.makedirs(out_dir, exist_ok=True)
print(f"[Profile] Saving profiler artifacts to {out_dir}/")

try:
    from torch.profiler import profile, ProfilerActivity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # grab a few real test batches
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    n_profile = 30          # batches to average over
    n_warmup  = 1          # discarded
    batches = []
    for i, b in enumerate(test_loader):
        batches.append(_to_device(b, device))
        if len(batches) >= n_profile + n_warmup:
            break

    if not batches:
        print("[Profile] No test batches available; skipping profiling.")
    else:
        with torch.no_grad():
            # warmup (CUDA init / autotune / one-time cache population) — not profiled
            for b in batches[:n_warmup]:
                _ = model(b, training=False)
            if device == "cuda":
                torch.cuda.synchronize()

            # ---- profile the measured passes, keep the LAST prof for export ----
            flops_list = []
            prof_last = None
            for b in batches[n_warmup:]:
                with profile(activities=[ProfilerActivity.CPU,
                                         ProfilerActivity.CUDA],
                             with_flops=True,
                             record_shapes=True,
                             profile_memory=True) as prof:
                    _ = model(b, training=False)
                    if device == "cuda":
                        torch.cuda.synchronize()
                total = sum(evt.flops for evt in prof.key_averages()
                            if getattr(evt, "flops", None))
                flops_list.append(total)
                prof_last = prof

            # ---- aggregate GFLOPs ----
            if flops_list:
                mean_flops = sum(flops_list) / len(flops_list)
                gflops = mean_flops / 1e9
                print(f"[Profile] GFLOPs/forward (mean of {len(flops_list)} "
                      f"passes): {gflops:.3f}  "
                      f"[lower bound; profiler-counted ops only]")

            # =================================================================
            # SAVE EVERYTHING to ./profiler/<tag>/
            # =================================================================
            if prof_last is not None:
                ka = prof_last.key_averages()

                # 1) Chrome trace (open in chrome://tracing or perfetto.dev)
                try:
                    trace_path = os.path.join(out_dir, f"trace_{tag}.json")
                    prof_last.export_chrome_trace(trace_path)
                    print(f"[Profile] chrome trace -> {trace_path}")
                except Exception as ee:
                    print(f"[Profile] chrome trace export failed: {ee}")

                # 2) Full key-averages table, sorted by CUDA time AND by FLOPs
                try:
                    table_path = os.path.join(out_dir, f"table_{tag}.txt")
                    with open(table_path, "w") as f:
                        f.write(f"Config: tpl={tpl} cross_k={ck}\n")
                        f.write(f"GFLOPs/forward (mean): {gflops:.4f}\n")
                        f.write(f"num profiled passes: {len(flops_list)}\n")
                        f.write(f"per-pass FLOPs: {flops_list}\n\n")
                        f.write("=== sorted by self CUDA time ===\n")
                        try:
                            f.write(ka.table(sort_by="self_cuda_time_total",
                                             row_limit=50))
                        except Exception:
                            f.write(ka.table(sort_by="self_cpu_time_total",
                                             row_limit=50))
                        f.write("\n\n=== sorted by self CPU time ===\n")
                        f.write(ka.table(sort_by="self_cpu_time_total",
                                         row_limit=50))
                    print(f"[Profile] full table -> {table_path}")
                except Exception as ee:
                    print(f"[Profile] table export failed: {ee}")

                # 3) Per-op FLOPs CSV (the one you actually want for the breakdown)
                try:
                    import csv
                    csv_path = os.path.join(out_dir, f"ops_flops_{tag}.csv")
                    rows = []
                    for e in ka:
                        fl = getattr(e, "flops", None) or 0
                        rows.append((
                            e.key,
                            fl,
                            fl / 1e9,
                            getattr(e, "self_cuda_time_total", 0),
                            getattr(e, "self_cpu_time_total", 0),
                            getattr(e, "count", 0),
                        ))
                    # sort by FLOPs desc
                    rows.sort(key=lambda r: r[1], reverse=True)
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["op", "flops", "gflops",
                                    "self_cuda_time_us", "self_cpu_time_us", "count"])
                        for r in rows:
                            w.writerow(r)
                    print(f"[Profile] per-op FLOPs CSV -> {csv_path}")

                    # also echo top-10 to stdout
                    nonzero = [r for r in rows if r[1] > 0]
                    print(f"[Profile] Top FLOP ops (GFLOPs)  "
                          f"[{len(nonzero)} ops with nonzero flops]:")
                    for r in rows[:10]:
                        print(f"[Profile]   {r[0][:40]:<40} {r[2]:>10.2f}")
                    if not nonzero:
                        print("[Profile] WARNING: no ops reported nonzero FLOPs. "
                              "with_flops is unreliable on this torch build — "
                              "use the CUDA-time table or switch to fvcore.")
                except Exception as ee:
                    print(f"[Profile] CSV export failed: {ee}")

                # 4) REGION SUMMARY — attribute leaf-op FLOPs/time to the
                #    INNERMOST enclosing record_function label.
                #    FLOPs live on aten:: leaf ops (addmm/mm/bmm), NOT on the
                #    record_function parent rows, so key_averages() shows 0 for
                #    your labels. We fix that by walking the flat event list and
                #    assigning each leaf op to the deepest user label whose CPU
                #    time interval contains it.
                try:
                    import csv as _csv

                    # Your record_function names, taken from the model. Any event
                    # with one of these names is treated as a region boundary.
                    REGION_LABELS = {
                        "Compute grid config", "encode", "Latents init",
                        "geo pruning", "Cross Attention - Encoder",
                        "Encoder Process data", "Encoder Cross Attention:",
                        "Self Attention",
                        "Decoder pre processing", "Decoder Cross attention",
                        "Decoder skip", "Decoder Cross Attention",
                        "Decoder Logits",
                    }

                    # flat event list with timestamps (microseconds)
                    evlist = prof_last.events()

                    def _start(e):
                        # different torch versions expose different attrs
                        for a in ("time_range",):
                            tr = getattr(e, a, None)
                            if tr is not None:
                                return tr.start, tr.end
                        s = getattr(e, "cpu_interval", None)
                        if s is not None:
                            return s.start, s.end
                        return None

                    # collect label intervals (name, start, end)
                    label_iv = []
                    for e in evlist:
                        nm = getattr(e, "name", getattr(e, "key", ""))
                        if nm in REGION_LABELS:
                            iv = _start(e)
                            if iv:
                                label_iv.append((nm, iv[0], iv[1]))

                    # accumulate FLOPs + cuda time per label from leaf ops that
                    # fall inside the label's interval (deepest = smallest span)
                    region_flops = {n: 0 for n in REGION_LABELS}
                    region_cuda  = {n: 0.0 for n in REGION_LABELS}
                    region_count = {n: 0 for n in REGION_LABELS}

                    for e in evlist:
                        fl = getattr(e, "flops", None) or 0
                        cu = getattr(e, "cuda_time_total", 0) or 0
                        if fl == 0 and cu == 0:
                            continue
                        iv = _start(e)
                        if not iv:
                            continue
                        s, en = iv
                        # find deepest enclosing label (smallest interval containing e)
                        best = None
                        best_span = None
                        for (nm, ls, le) in label_iv:
                            if ls <= s and en <= le:
                                span = le - ls
                                if best_span is None or span < best_span:
                                    best, best_span = nm, span
                        if best is not None:
                            region_flops[best] += fl
                            region_cuda[best]  += cu
                            region_count[best] += 1

                    region_path = os.path.join(out_dir, f"regions_{tag}.csv")
                    with open(region_path, "w", newline="") as f:
                        w = _csv.writer(f)
                        w.writerow(["region", "gflops", "cuda_ms", "leaf_ops"])
                        # sort by gflops desc
                        for nm in sorted(REGION_LABELS,
                                         key=lambda n: region_flops[n],
                                         reverse=True):
                            w.writerow([nm, region_flops[nm]/1e9,
                                        region_cuda[nm]/1e3, region_count[nm]])
                    print(f"[Profile] REGION summary -> {region_path}")

                    print("[Profile] Per-region breakdown "
                          "(GFLOPs | CUDA ms | leaf ops):")
                    any_nonzero = False
                    for nm in sorted(REGION_LABELS,
                                     key=lambda n: region_flops[n], reverse=True):
                        gf = region_flops[nm]/1e9
                        cu = region_cuda[nm]/1e3
                        if region_flops[nm] > 0 or region_cuda[nm] > 0:
                            any_nonzero = True
                        print(f"[Profile]   {nm:<26} {gf:>9.1f} | "
                              f"{cu:>7.2f} ms | {region_count[nm]:>4d}")
                    if not any_nonzero:
                        print("[Profile]   (no FLOPs attributed — interval "
                              "matching found no enclosing labels; check that "
                              "record_function regions wrap the heavy ops)")
                except Exception as ee:
                    import traceback as _tb
                    print(f"[Profile] region summary failed: {ee}")
                    print(_tb.format_exc())
except Exception as e:
    import traceback
    print(f"[Profile] GFLOPs measurement failed: {e}")
    with open(os.path.join(out_dir, "ERROR.txt"), "w") as f:
        f.write(traceback.format_exc())
    gflops = float("nan")


# RESULT line now carries GFLOPs for the driver to parse.
print(f"RESULT tpl={tpl} cross_k={ck} "
      f"test_mIoU={miou:.6f} test_accuracy={acc:.6f} gflops={gflops:.6f}")
