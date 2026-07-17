#!/usr/bin/env python3
"""CDF of keyframe-chunk durations for the VideoMME-Long test videos.

Primary: the raw keyframe chunks (VideoMME_kf events = one keyframe interval each).
For context it overlays AVA-default event durations and, if present, keyframe-merge
event durations. Saves a PNG + a per-video stats CSV of the keyframe chunks.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
    "lines.linewidth": 2.4, "axes.linewidth": 1.1, "savefig.dpi": 300,
})

REPO = Path(__file__).resolve().parents[2]
BASE_ID = 601
# (label, cache-suffix, color, linestyle, linewidth, emphasis)
SERIES_SPEC = [
    ("keyframe chunks (raw)", "_kf", "#1f4e79", "-", 2.6, True),
    ("AVA default events (merged)", "", "#c0392b", "--", 2.0, False),
    ("keyframe-merge events (merged)", "_kf_merge", "#27ae60", "-.", 2.0, False),
]
CACHE_BASE = {"videomme": "AVA_cache/VideoMME", "lvbench": "AVA_cache/LVBench"}
OUT = {"videomme": REPO / "results/pdf/videomme_keyframe_chunk_cdf.pdf",
       "lvbench": REPO / "results/pdf/lvbench_keyframe_chunk_cdf.pdf"}
SELECTION = REPO / "results/json/lvbench_selection.json"


def segment_durations(events_path):
    if not events_path.is_file():
        return []
    events = json.loads(events_path.read_text())
    return [e["duration"][1] - e["duration"][0] for e in events if "duration" in e]


def cdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def collect(root, video_ids):
    per_video, combined = {}, []
    for vid in video_ids:
        durs = segment_durations(root / str(vid) / "kg/events/events.json")
        if durs:
            per_video[vid] = durs
            combined.extend(durs)
    return per_video, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["videomme", "lvbench"], default="videomme")
    parser.add_argument("--num", type=int, default=5, help="videomme: first N long videos from 601")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset == "lvbench":
        video_ids = json.loads(SELECTION.read_text())["video_ids"]
    else:
        video_ids = list(range(BASE_ID, BASE_ID + args.num))
    args.out = args.out or OUT[args.dataset]
    series = [(label, REPO / (CACHE_BASE[args.dataset] + suf), c, ls, lw, emp)
              for label, suf, c, ls, lw, emp in SERIES_SPEC]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    kf_per_video, kf_all = {}, []
    for label, root, color, ls, lw, emphasis in series:
        per_video, combined = collect(root, video_ids)
        if not combined:
            continue
        if emphasis:
            kf_per_video, kf_all = per_video, combined
            for vid, durs in per_video.items():
                x, y = cdf(durs)
                ax.plot(x, y, lw=1, alpha=0.3, color=color)
        x, y = cdf(combined)
        ax.plot(x, y, lw=lw, ls=ls, color=color, label=f"{label} (n={len(combined)})")

    if not kf_all:
        raise SystemExit("no keyframe chunks found yet; run the keyframe pipeline first")

    ax.axvline(3, color="gray", ls=":", lw=1.4, alpha=0.7)
    ax.text(3.12, 0.03, "AVA base window = 3s", color="gray", fontsize=13)

    ax.set_xlabel("chunk / event duration (seconds)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(f"Keyframe-chunk duration CDF · videos {video_ids[0]}-{video_ids[-1]} "
                 f"(keyframe median {np.median(kf_all):.2f}s, n={len(kf_all)})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.legend(fontsize=13, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)  # PDF only

    stats_path = REPO / "results/csv" / (args.out.stem.replace("_cdf", "") + "_stats.csv")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "n_chunks", "min_s", "mean_s", "median_s", "p90_s", "max_s"])
        for vid, d in kf_per_video.items():
            a = np.asarray(d)
            writer.writerow([vid, len(d), round(a.min(), 2), round(a.mean(), 2),
                             round(np.median(a), 2), round(np.percentile(a, 90), 2), round(a.max(), 2)])
        a = np.asarray(kf_all)
        writer.writerow(["ALL", len(kf_all), round(a.min(), 2), round(a.mean(), 2),
                         round(np.median(a), 2), round(np.percentile(a, 90), 2), round(a.max(), 2)])

    print(f"keyframe chunks (all {len(kf_all)}): mean {a.mean():.2f}s, median {np.median(a):.2f}s, "
          f"p90 {np.percentile(a, 90):.2f}s, max {a.max():.2f}s")
    print(f"wrote: {args.out}\nwrote: {stats_path}")


if __name__ == "__main__":
    main()
