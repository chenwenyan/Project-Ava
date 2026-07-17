#!/usr/bin/env python3
"""Preprocess VideoMME-Long videos 601-620 into AVA's frame cache (parallel)."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from video_utils import VideoRepresentation  # noqa: E402

VIDEO_IDS = range(601, 621)
VIDEOS_DIR = REPO / "datas/VideoMME/videos"
CACHE_DIR = REPO / "AVA_cache/VideoMME"
METADATA = REPO / "datas/VideoMME/VideoMME.json"


def preprocess(video_id, name):
    source = VIDEOS_DIR / f"{name}.mp4"
    if not source.is_file():
        return video_id, f"missing: {source}"
    try:
        VideoRepresentation(str(source), str(CACHE_DIR / str(video_id)))
        ok = (CACHE_DIR / str(video_id) / "config.json").is_file()
        return video_id, "ok" if ok else "missing config.json"
    except Exception as error:
        return video_id, f"{type(error).__name__}: {error}"


def main():
    rows = json.loads(METADATA.read_text())
    names = {int(row["video_id"]): row["videoID"] for row in rows}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(preprocess, v, names[v]) for v in VIDEO_IDS]
        results = [f.result() for f in as_completed(futures)]
        for video_id, status in results:
            print(f"[{status}] video {video_id}", flush=True)

    failures = [r for r in results if r[1] != "ok"]
    print(f"completed: {len(results) - len(failures)}/{len(results)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
