import os, re
from huggingface_hub import HfApi, snapshot_download
TEST = {"D012","D015","D022","D026","D036","D061","D064","D068","D069","D071","D073","D075","D076","D083","D084","D085"}
root = "/lustre/fsn1/projects/rech/ebp/uuk56xr/Atomizer/data/flair_hub"
files = [e.path for e in HfApi().list_repo_tree("IGNF/FLAIR-HUB", repo_type="dataset", path_in_repo="data", recursive=False)
         if getattr(e, "size", None) and e.path.endswith("_AERIAL_RGBI.zip")]
dept = lambda p: re.match(r"data/(D\d+)", p).group(1)
todo = [p for p in files if dept(p) not in TEST and not os.path.exists(os.path.join(root, p))]
print(len(todo), "zips to download", flush=True)
snapshot_download(repo_id="IGNF/FLAIR-HUB", repo_type="dataset", allow_patterns=todo, local_dir=root, max_workers=8)
print("done", flush=True)
