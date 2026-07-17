#!/usr/bin/env bash
# Download LVBench videos (HF AIWinter/LVBench, 14-part ~62GB split zip) to
# ~/wenyan/dataset/LVBench/videos and symlink datas/LVBench/videos to it.
# Uses Python for download + zipfile extraction (no `unzip` dependency); parts are
# only removed after extraction succeeds.
set -uo pipefail
PY=/export/home/yulin/miniconda3/envs/ava/bin/python
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=1  # Rust multi-threaded downloader (fast + robust)
DEST="$HOME/wenyan/dataset/LVBench"
TMP="$DEST/tmp_downloads"
VIDEOS="$DEST/videos"
REPO=/export/home/yulin/wenyan/github/Project-Ava
mkdir -p "$TMP" "$VIDEOS"

$PY - "$TMP" "$VIDEOS" <<'EOF'
import sys, os, glob, zipfile
from huggingface_hub import hf_hub_download, list_repo_files
tmp, videos = sys.argv[1], sys.argv[2]
repo = "AIWinter/LVBench"

parts = sorted(f for f in list_repo_files(repo, repo_type="dataset")
               if f.startswith("all_videos_split.zip"))
print("parts:", parts, flush=True)
for f in parts:
    print("downloading", f, flush=True)
    hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=tmp)

combined = os.path.join(tmp, "all_videos_split.zip")
print("concatenating ->", combined, flush=True)
with open(combined, "wb") as out:
    for p in sorted(glob.glob(os.path.join(tmp, "all_videos_split.zip.*"))):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 24):
                out.write(chunk)

print("extracting mp4 ...", flush=True)
with zipfile.ZipFile(combined) as z:
    members = [m for m in z.namelist() if m.endswith(".mp4")]
    for i, m in enumerate(members, 1):
        target = os.path.join(videos, os.path.basename(m))
        if not os.path.exists(target):
            with z.open(m) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
        if i % 10 == 0:
            print(f"  extracted {i}/{len(members)}", flush=True)
print(f"extracted {len(members)} mp4", flush=True)
EOF

status=$?
if [ $status -eq 0 ] && [ "$(ls "$VIDEOS"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
    rm -rf "$TMP"
    ln -sfn "$VIDEOS" "$REPO/datas/LVBench/videos"
    echo "[lvbench-dl] done: $(ls "$VIDEOS"/*.mp4 | wc -l) mp4 in $VIDEOS"
else
    echo "[lvbench-dl] FAILED (status=$status); keeping $TMP for retry"
    exit 1
fi
