#!/usr/bin/env python
"""Analyze model-input frame counts for Default (AVA) vs KF-Direct.

Both methods caption video chunks by feeding the VLM `video_chunk_num_frames`
(=6) evenly-spaced frames per chunk (get_frames_by_num, video_utils.py). The
methods differ only in how chunks are cut:
  * Default   -- fixed video_chunk_duration=3s windows (get_chunk_timestamp),
                 captioned BEFORE the BERTScore semantic merge, so the frame
                 input happens at base-window granularity (= descriptions.json).
  * KF-Direct -- one chunk per codec keyframe (I-frame) interval (events.json in
                 the *_kf cache; direct mode does no merge).

For every video this script replays the exact 6-frame sampling on each method's
real chunk boundaries and reports, per chunk and per video:
  n_input   -- image slots actually sent to the model (~6, the "输入 frame 数目")
  n_unique  -- distinct native frame indices (coverage; short chunks collapse)

Outputs per-chunk + per-video CSVs per (method, dataset) and prints headline
stats. Frame cache is read from the MAIN checkout (caches are untracked).
"""
import csv
import json
import math
import os

MAIN = "/export/home/yulin/wenyan/github/Project-Ava"
NUM_FRAMES = 6          # AVA.ava video_chunk_num_frames default
CHUNK_DURATION = 3      # AVA.ava video_chunk_duration default (Default method)

# method -> (cache root, chunking rule)
#   "events"  : one chunk per events.json entry (KF-Direct native cache)
#   "grid3s"  : fixed 3s base windows rebuilt from duration (Default method)
CONFIGS = {
    "lvbench": {
        "default":   (f"{MAIN}/AVA_cache/LVBench",    "grid3s"),
        "kf_direct": (f"{MAIN}/AVA_cache/LVBench_kf", "events"),
    },
    "videomme": {
        "default":   (f"{MAIN}/AVA_cache/VideoMME",    "grid3s"),
        "kf_direct": (f"{MAIN}/AVA_cache/VideoMME_kf", "events"),
    },
}


def time_points(start, end, n):
    """Mirror get_frames_by_num's time-point schedule exactly."""
    if start == end:
        end += 1
    if n == 1:
        return [(start + end) / 2]
    if n == 2:
        return [start, end]
    return [start + i * (end - start) / (n - 1) for i in range(n)]


def grid_windows(duration, step):
    """Mirror get_chunk_timestamp (overlap 0): 0,step,2step,... last clipped."""
    wins, t = [], 0
    while t < duration:
        e = t + step
        if e > duration:
            wins.append((t, duration))
            break
        wins.append((t, e))
        t = e
    return wins


def chunk_bounds(cache_root, vid, rule, duration):
    if rule == "events":
        events = json.load(open(os.path.join(cache_root, vid, "kg/events/events.json")))
        return [tuple(ev["duration"]) for ev in events]
    if rule == "grid3s":
        return grid_windows(duration, CHUNK_DURATION)
    raise ValueError(rule)


def analyze_video(cache_root, vid, rule):
    cfg = json.load(open(os.path.join(cache_root, vid, "config.json")))
    fps, duration = cfg["fps"], cfg["duration"]
    frames_dir = os.path.join(cache_root, vid, "frames")
    rows = []
    for ci, (s, e) in enumerate(chunk_bounds(cache_root, vid, rule, duration)):
        idxs = []
        for t in time_points(s, e, NUM_FRAMES):
            fi = int(t * fps)
            if os.path.exists(os.path.join(frames_dir, f"{fi}.jpg")):
                idxs.append(fi)
        rows.append({
            "video_id": vid, "chunk_index": ci, "duration_s": round(e - s, 4),
            "n_input": len(idxs), "n_unique": len(set(idxs)),
        })
    return rows


def main():
    import statistics as st
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "csv")
    for ds, methods in CONFIGS.items():
        # use the KF-Direct video set as the reference so Default is compared on
        # exactly the same videos (Default caches may hold extra ids).
        kf_root = methods["kf_direct"][0]
        ref_vids = {d for d in os.listdir(kf_root)
                    if d.isdigit() and os.path.isdir(os.path.join(kf_root, d))}
        for method, (cache_root, rule) in methods.items():
            vids = sorted((d for d in os.listdir(cache_root)
                           if d in ref_vids
                           and os.path.isdir(os.path.join(cache_root, d))),
                          key=int)
            seg_rows = []
            for vid in vids:
                seg_rows.extend(analyze_video(cache_root, vid, rule))

            tag = f"{method}_{ds}"
            seg_csv = os.path.join(out_dir, f"input_frames_{tag}.csv")
            with open(seg_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["video_id", "chunk_index",
                                                  "duration_s", "n_input", "n_unique"])
                w.writeheader()
                w.writerows(seg_rows)

            per_vid = {}
            for r in seg_rows:
                v = per_vid.setdefault(r["video_id"],
                                       {"n_chunks": 0, "input": 0, "unique": 0})
                v["n_chunks"] += 1
                v["input"] += r["n_input"]
                v["unique"] += r["n_unique"]
            vid_csv = os.path.join(out_dir, f"input_frames_by_video_{tag}.csv")
            with open(vid_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video_id", "n_chunks", "total_input_frames",
                            "total_unique_frames"])
                for v in sorted(per_vid, key=int):
                    d = per_vid[v]
                    w.writerow([v, d["n_chunks"], d["input"], d["unique"]])

            n_seg = len(seg_rows)
            tot_input = sum(r["n_input"] for r in seg_rows)
            vid_inputs = sorted(d["input"] for d in per_vid.values())
            print(f"\n=== {ds.upper()} / {method}  ({len(vids)} videos) ===")
            print(f"chunks                 : {n_seg}")
            print(f"input frames / chunk   : mean {tot_input/n_seg:.3f}")
            print(f"total input frames     : {tot_input}")
            print(f"input frames / video   : mean {st.mean(vid_inputs):.0f}  "
                  f"median {st.median(vid_inputs):.0f}  "
                  f"min {vid_inputs[0]}  max {vid_inputs[-1]}")


if __name__ == "__main__":
    main()
