#!/usr/bin/env python3
"""Replicate the AVA paper's LVBench ablation filter: randomly sample 20 videos and
their questions (~305). The paper doesn't publish its exact list, so we reproduce the
METHOD with a fixed seed for determinism. Optional per-video question cap for a faster
but still fair 3-way method comparison. Writes results/lvbench_selection.json + .csv.
"""

import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
META = REPO / "datas/LVBench/LVBench.json"
OUT_JSON = REPO / "results/json/lvbench_selection.json"
OUT_CSV = REPO / "results/csv/lvbench_selection.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-videos", type=int, default=20,
                        help="paper's random sample size (keeps the same sampling method)")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = keep all sampled; else keep the first N of the sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-q-per-video", type=int, default=0,
                        help="0 = all questions (paper: ~305 total); else cap per video")
    args = parser.parse_args()

    videos = json.loads(META.read_text())  # 1-indexed video_id = position+1
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(1, len(videos) + 1), args.num_videos))
    if args.limit > 0:
        chosen = chosen[:args.limit]

    selection, total_q = {}, 0
    rows = []
    for vid in chosen:
        info = videos[vid - 1]
        n = len(info["qa"])
        keep = list(range(n)) if args.max_q_per_video <= 0 else list(range(min(n, args.max_q_per_video)))
        selection[str(vid)] = keep
        total_q += len(keep)
        rows.append({"video_id": vid, "key": info["key"], "type": info["type"],
                     "duration_min": round(info["video_info"]["duration_minutes"], 1),
                     "n_questions": len(keep)})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"seed": args.seed, "num_videos": args.num_videos,
         "max_q_per_video": args.max_q_per_video, "total_questions": total_q,
         "video_ids": chosen, "questions": selection}, indent=2))
    with OUT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video_id", "key", "type", "duration_min", "n_questions"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"seed={args.seed} videos={chosen}")
    print(f"total questions={total_q} (cap/video={args.max_q_per_video or 'all'})")
    print(f"wrote: {OUT_JSON}, {OUT_CSV}")


if __name__ == "__main__":
    main()
