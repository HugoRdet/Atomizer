"""
download_flair_dem.py -- download the FLAIR-HUB elevation zips (one per department), next to the aerial ones.

Run where there is internet (Jean Zay: a login node, or the prepost partition for a long download):
    python download_flair_dem.py --list                 # what the repo has: modality suffixes + sizes, nothing downloaded
    python download_flair_dem.py --dry-run              # the DEM zips that would be downloaded
    python download_flair_dem.py                        # download (resumable: files already present are skipped)

All departments are downloaded (train AND the official test departments: FlairAtoms uses the test departments as its
validation split, so the DEM is needed there too).
"""
import argparse
import os
import re
from collections import defaultdict

from huggingface_hub import HfApi, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/lustre/fsn1/projects/rech/ebp/uuk56xr/Atomizer/data/flair_hub")
ap.add_argument("--pattern", default=r"DEM", help="regex on the modality suffix selecting the elevation zips")
ap.add_argument("--list", action="store_true", help="list the modalities of the repo and exit")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

entries = [e for e in HfApi().list_repo_tree("IGNF/FLAIR-HUB", repo_type="dataset", path_in_repo="data", recursive=False)
           if getattr(e, "size", None) and e.path.endswith(".zip")]
name_re = re.compile(r"data/(D\d+)[-_]\d*_?(.+)\.zip$")


def modality(path):
    m = name_re.match(path)
    if m:
        return m.group(1), m.group(2)
    d = re.match(r"data/(D\d+)", path)
    return (d.group(1) if d else "?"), os.path.basename(path)[:-4]


by_mod = defaultdict(lambda: [0, 0])
for e in entries:
    _, mod = modality(e.path)
    by_mod[mod][0] += 1
    by_mod[mod][1] += e.size
if args.list:
    print(f"{len(entries)} zips in IGNF/FLAIR-HUB/data; example: {entries[0].path if entries else '-'}")
    for mod, (n, size) in sorted(by_mod.items()):
        print(f"  {mod:40s} {n:4d} zips {size / 1e9:8.1f} GB")
    raise SystemExit

sel = [e for e in entries if re.search(args.pattern, modality(e.path)[1])]
if not sel:
    raise SystemExit(f"no zip matches --pattern {args.pattern!r}; run with --list to see the modality names")
mods = sorted({modality(e.path)[1] for e in sel})
todo = [e for e in sel if not os.path.exists(os.path.join(args.root, e.path))]
print(f"modalities matching {args.pattern!r}: {mods}")
print(f"{len(sel)} zips ({sum(e.size for e in sel) / 1e9:.1f} GB), {len(todo)} to download "
      f"({sum(e.size for e in todo) / 1e9:.1f} GB) into {args.root}", flush=True)
if args.dry_run:
    for e in todo[:10]:
        print("  ", e.path)
    raise SystemExit
snapshot_download(repo_id="IGNF/FLAIR-HUB", repo_type="dataset", allow_patterns=[e.path for e in todo],
                  local_dir=args.root, max_workers=args.workers)
print("done", flush=True)
