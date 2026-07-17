#!/usr/bin/env python3
"""Extract frames for the selected LVBench videos into AVA_cache/LVBench (parallel).
Reads results/lvbench_selection.json. Run before the graph stage so the shared frame
cache exists (the keyframe modes symlink it)."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from video_utils import VideoRepresentation  # noqa: E402

META = REPO / "datas/LVBench/LVBench.json"
VIDEOS_DIR = REPO / "datas/LVBench/videos"
CACHE_DIR = REPO / "AVA_cache/LVBench"
SELECTION = REPO / "results/json/lvbench_selection.json"


def preprocess(video_id, key):
    source = VIDEOS_DIR / f"{key}.mp4"
    if not source.is_file():
        return video_id, f"missing: {source}"
    try:
        VideoRepresentation(str(source), str(CACHE_DIR / str(video_id)))
        ok = (CACHE_DIR / str(video_id) / "config.json").is_file()
        return video_id, "ok" if ok else "missing config.json"
    except Exception as error:
        return video_id, f"{type(error).__name__}: {error}"


def main():
    videos = json.loads(META.read_text())
    video_ids = json.loads(SELECTION.read_text())["video_ids"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(vid, videos[vid - 1]["key"]) for vid in video_ids]

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as executor:
        futures = [executor.submit(preprocess, vid, key) for vid, key in jobs]
        results = [f.result() for f in as_completed(futures)]
        for vid, status in results:
            print(f"[{status}] video {vid}", flush=True)

    failures = [r for r in results if r[1] != "ok"]
    print(f"completed: {len(results) - len(failures)}/{len(results)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
