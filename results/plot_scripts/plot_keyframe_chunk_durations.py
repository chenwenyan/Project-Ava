#!/usr/bin/env python3
"""Duration distribution of keyframe-segmented video chunks (VideoMME).

Segments each source video at codec keyframe (I-frame) timestamps -- the exact
segmentation AVA's keyframe pipeline uses (run_keyframe_pipeline.keyframe_segments)
-- and plots the distribution of the resulting chunk durations.

Source paths come straight from datas/VideoMME/VideoMME.json (video_id ->
videos/<videoID>.mp4), so no AVA preprocessing/cache is required; durations are
read from the video container.

Default range: VideoMME IDs 601-900 (the full VideoMME video set). Processing is
parallelised across videos.

Outputs (results/):
  keyframe_chunk_duration_dist_<lo>_<hi>.pdf / .png  -- 4-panel distribution figure
  keyframe_chunk_durations_<lo>_<hi>.csv             -- one row per chunk
  keyframe_chunk_duration_stats_<lo>_<hi>.csv        -- per-video + overall summary
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]  # results/plot_scripts/ -> repo root
os.chdir(REPO)
OUT_DIR = REPO / "results"

# Per-dataset source resolution. `json` -> metadata list, `dir` -> mp4 folder,
# `key` -> field naming the mp4 stem, `stride` -> rows per video (VideoMME packs
# 3 QA rows per video), `base` -> first valid video id.
DATASETS = {
    "videomme": {"json": "datas/VideoMME/VideoMME.json", "dir": "datas/VideoMME/videos",
                 "key": "videoID", "stride": 3, "base": 601},
    "lvbench": {"json": "datas/LVBench/LVBench.json", "dir": "datas/LVBench/videos",
                "key": "key", "stride": 1, "base": 1},
}

# OSDI/publication style: thin outward ticks, no top/right spines,
# restrained grayscale-safe colours, editable embedded fonts (default font kept).
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "grid.linewidth": 0.5, "grid.linestyle": "--", "grid.alpha": 0.4,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def source_paths(dataset, lo, hi):
    """video_id -> source mp4 path for the given dataset."""
    spec = DATASETS[dataset]
    infos = json.loads((REPO / spec["json"]).read_text())
    videos_dir = REPO / spec["dir"]
    out = {}
    for vid in range(lo, hi + 1):
        idx = (vid - 1) * spec["stride"]
        if idx >= len(infos):
            break
        out[vid] = videos_dir / f'{infos[idx][spec["key"]]}.mp4'
    return out


def keyframe_segments(source_path, duration):
    """[(start, end), ...] cut at codec keyframe timestamps (mirrors the pipeline)."""
    bounds = sorted({k for k in _keyframe_times(source_path) if 0.0 <= k < duration})
    if not bounds or bounds[0] > 0.0:
        bounds.insert(0, 0.0)
    bounds.append(float(duration))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


def _keyframe_times(source_path):
    container = av.open(str(source_path))
    stream = container.streams.video[0]
    tb = stream.time_base
    times = [float(p.pts * tb) for p in container.demux(stream)
             if p.is_keyframe and p.pts is not None]
    container.close()
    return times


def process_one(item):
    """Worker: returns (video_id, video_duration, [chunk_durations]) or None."""
    vid, source = item
    source = Path(source)
    if not source.is_file():
        return ("missing", vid, str(source))
    try:
        container = av.open(str(source))
        stream = container.streams.video[0]
        tb = stream.time_base
        if stream.duration:
            duration = float(stream.duration * tb)
        elif container.duration:
            duration = container.duration / 1_000_000.0
        else:
            duration = 0.0
        keyframes = [float(p.pts * tb) for p in container.demux(stream)
                     if p.is_keyframe and p.pts is not None]
        container.close()
        if duration <= 0 and keyframes:
            duration = max(keyframes)
        bounds = sorted({k for k in keyframes if 0.0 <= k < duration})
        if not bounds or bounds[0] > 0.0:
            bounds.insert(0, 0.0)
        bounds.append(float(duration))
        durs = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)
                if bounds[i + 1] > bounds[i]]
        return ("ok", vid, duration, durs)
    except Exception as error:  # noqa: BLE001
        return ("error", vid, f"{type(error).__name__}: {error}")


def collect(dataset, lo, hi, workers):
    items = list(source_paths(dataset, lo, hi).items())
    per_video, vid_duration, failures = {}, {}, []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, (v, str(s))): v for v, s in items}
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res[0] == "ok":
                _, vid, duration, durs = res
                if durs:
                    per_video[vid] = durs
                    vid_duration[vid] = duration
            else:
                failures.append(res)
            if done % 25 == 0 or done == len(items):
                print(f"  processed {done}/{len(items)} videos "
                      f"({len(per_video)} ok, {len(failures)} failed)", flush=True)
    for kind, vid, info in sorted(failures):
        print(f"[{kind}] {vid}: {info}", flush=True)
    return per_video, vid_duration


def write_csvs(per_video, vid_duration, all_durs, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"keyframe_chunk_durations_{tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "chunk_index", "duration_s"])
        for vid in sorted(per_video):
            for i, d in enumerate(per_video[vid]):
                w.writerow([vid, i, round(d, 4)])

    with (OUT_DIR / f"keyframe_chunk_duration_stats_{tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "video_duration_s", "n_chunks", "min_s", "mean_s",
                    "median_s", "p90_s", "max_s", "std_s"])
        for vid in sorted(per_video):
            a = np.asarray(per_video[vid], dtype=float)
            w.writerow([vid, round(vid_duration[vid], 1), len(a), round(a.min(), 2),
                        round(a.mean(), 2), round(np.median(a), 2),
                        round(np.percentile(a, 90), 2), round(a.max(), 2), round(a.std(), 2)])
        a = np.asarray(all_durs, dtype=float)
        w.writerow(["ALL", round(sum(vid_duration.values()), 1), len(a), round(a.min(), 2),
                    round(a.mean(), 2), round(np.median(a), 2),
                    round(np.percentile(a, 90), 2), round(a.max(), 2), round(a.std(), 2)])


def plot(per_video, vid_duration, all_durs, tag):
    """Two-panel OSDI-style figure: (A) chunk-duration histogram, (B) ECDF."""
    a = np.asarray(all_durs, dtype=float)
    mean, median, p90 = a.mean(), np.median(a), np.percentile(a, 90)
    p99 = np.percentile(a, 99)
    n_vids = len(per_video)

    bar_c, line_c = "#4C72B0", "#333333"  # muted blue bars, dark reference lines
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # (A) histogram of ALL chunk durations
    bins = np.linspace(0, max(p99, 1.0), 41)
    ax_h.hist(a, bins=bins, color=bar_c, alpha=0.9, edgecolor="white", linewidth=0.3)
    ax_h.axvline(median, color=line_c, ls="-", lw=1.3, label=f"median {median:.2f}s")
    ax_h.axvline(mean, color=line_c, ls="--", lw=1.3, label=f"mean {mean:.2f}s")
    ax_h.axvline(p90, color=line_c, ls=":", lw=1.3, label=f"p90 {p90:.2f}s")
    ax_h.set_xlabel("Keyframe chunk duration (s)")
    ax_h.set_ylabel("Number of chunks")
    ax_h.set_xlim(0, bins[-1])
    ax_h.set_ylim(bottom=0)
    ax_h.grid(True, axis="y")
    ax_h.set_axisbelow(True)
    ax_h.legend(loc="upper right", borderpad=0.3, labelspacing=0.3)
    ax_h.text(0.02, 0.96, "(a)", transform=ax_h.transAxes, va="top", fontweight="bold")

    # (B) ECDF of all chunk durations
    xs = np.sort(a)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    xmax = np.percentile(a, 99.5)
    ax_c.plot(xs, ys, color=bar_c, lw=1.6)
    for val, ls, lab in [(median, "-", "median"), (p90, ":", "p90")]:
        ax_c.axvline(val, color=line_c, ls=ls, lw=1.1)
        ax_c.text(val + 0.012 * xmax, 0.06, f"{lab}\n{val:.2f}s", color=line_c, fontsize=8.5)
    ax_c.set_xlabel("Keyframe chunk duration (s)")
    ax_c.set_ylabel("CDF")
    ax_c.set_xlim(0, xmax)
    ax_c.set_ylim(0, 1)
    ax_c.grid(True)
    ax_c.set_axisbelow(True)
    ax_c.text(0.02, 0.96, "(b)", transform=ax_c.transAxes, va="top", fontweight="bold")

    fig.tight_layout(pad=0.4, w_pad=1.2)

    # Project plotting convention: PDF only (no PNG).
    pdf = OUT_DIR / f"keyframe_chunk_duration_dist_{tag}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {n_vids} videos, {len(a):,} chunks", flush=True)
    return pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="videomme")
    parser.add_argument("--lo", type=int, default=None, help="first video id (default: dataset base)")
    parser.add_argument("--hi", type=int, default=None, help="last video id (inclusive)")
    parser.add_argument("--num", type=int, default=None,
                        help="alternative to --hi: process the first N videos from --lo")
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    args = parser.parse_args()

    lo = args.lo if args.lo is not None else DATASETS[args.dataset]["base"]
    if args.hi is not None:
        hi = args.hi
    elif args.num is not None:
        hi = lo + args.num - 1
    else:
        hi = lo + (300 if args.dataset == "videomme" else 20) - 1
    tag = f"{args.dataset}_{lo}_{hi}"

    print(f"Collecting keyframe chunks for {args.dataset} {lo}-{hi} "
          f"({args.workers} workers)...", flush=True)
    per_video, vid_duration = collect(args.dataset, lo, hi, args.workers)
    if not per_video:
        sys.exit("no videos processed")
    all_durs = [d for durs in per_video.values() for d in durs]
    write_csvs(per_video, vid_duration, all_durs, tag)
    pdf = plot(per_video, vid_duration, all_durs, tag)

    a = np.asarray(all_durs, dtype=float)
    print(f"\nALL: {len(per_video)} videos, {len(a):,} chunks | "
          f"mean {a.mean():.2f}s, median {np.median(a):.2f}s, "
          f"p90 {np.percentile(a, 90):.2f}s, max {a.max():.2f}s")
    print(f"wrote: {pdf}")
    print(f"wrote: {OUT_DIR / f'keyframe_chunk_durations_{tag}.csv'}")
    print(f"wrote: {OUT_DIR / f'keyframe_chunk_duration_stats_{tag}.csv'}")


if __name__ == "__main__":
    main()
