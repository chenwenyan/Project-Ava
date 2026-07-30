#!/usr/bin/env python
"""Analyze CA-stage model-input frame counts (Default vs KF-Direct).

At the CA stage (generate_ca_self_consistency_result, AVA/operate.py) each
question feeds the VLM one video input per *candidate node*: up to 2 nodes are
picked from the ranked SA nodes (the top node, plus the next node whose top
answer differs), and for each the frames from its retrieved intervals are
sampled at 1 fps (get_frames_by_fps) then capped at max_frames=256.

Everything needed is cached per question in
  <cache>/<vid>/questions/<qid>/sorted_SA_score_result.json
(the ranked SA nodes, each with `frame_durations` and `final_score`), so this
replays the exact candidate selection + 1-fps sampling + 256 cap WITHOUT any
LLM. Frame counts are computed analytically (the native-fps frame cache is
complete, so every sampled index exists), matching get_frames_by_fps output.

Reports per-candidate-node CA input-frame counts (one video input per CA
inference) and prints how often the 256 cap binds. Reads caches from MAIN.
"""
import csv
import json
import math
import os

MAIN = "/export/home/yulin/wenyan/github/Project-Ava"
CA_FPS = 1
MAX_FRAMES = 256  # generate_ca_self_consistency_result default

CONFIGS = {
    "lvbench": {
        "default":   f"{MAIN}/AVA_cache/LVBench",
        "kf_direct": f"{MAIN}/AVA_cache/LVBench_kf",
    },
    "videomme": {
        "default":   f"{MAIN}/AVA_cache/VideoMME",
        "kf_direct": f"{MAIN}/AVA_cache/VideoMME_kf",
    },
}


def top_answer(node):
    return list(node["final_score"].keys())[0]


def select_candidates(sorted_nodes):
    """Mirror generate_ca_self_consistency_result: <=2 nodes, differing top answer."""
    cands = []
    for n in sorted_nodes:
        if len(cands) == 2:
            break
        if not cands:
            cands.append(n)
        elif top_answer(n) != top_answer(cands[0]):
            cands.append(n)
    return cands


def frame_count(frame_durations, fps):
    """Analytic get_frames_by_fps(fps=1) count summed over intervals (pre-cap)."""
    interval = int(fps / CA_FPS) if CA_FPS > 0 else 1
    interval = max(interval, 1)
    total = 0
    for s, e in frame_durations:
        sf = int(s * fps)
        if s == e:
            total += 1
        else:
            ef = int(e * fps)
            total += max(0, len(range(sf, ef, interval)))
    return total


def analyze(cache_root, ref_vids):
    rows = []
    vids = sorted((d for d in os.listdir(cache_root)
                   if d in ref_vids and os.path.isdir(os.path.join(cache_root, d))), key=int)
    for vid in vids:
        fps = json.load(open(os.path.join(cache_root, vid, "config.json")))["fps"]
        qdir = os.path.join(cache_root, vid, "questions")
        if not os.path.isdir(qdir):
            continue
        for qid in sorted((q for q in os.listdir(qdir) if q.isdigit()), key=int):
            f = os.path.join(qdir, qid, "sorted_SA_score_result.json")
            if not os.path.exists(f):
                continue
            nodes = json.load(open(f))
            for ci, node in enumerate(select_candidates(nodes)):
                raw = frame_count(node.get("frame_durations", []), fps)
                rows.append({
                    "video_id": vid, "question_id": qid, "candidate_index": ci,
                    "n_intervals": len(node.get("frame_durations", [])),
                    "raw_frames": raw,
                    "ca_input_frames": min(raw, MAX_FRAMES),
                    "capped": int(raw > MAX_FRAMES),
                })
    return rows


def main():
    import statistics as st
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "csv")
    for ds, methods in CONFIGS.items():
        kf_root = methods["kf_direct"]
        ref_vids = {d for d in os.listdir(kf_root)
                    if d.isdigit() and os.path.isdir(os.path.join(kf_root, d))}
        for method, cache_root in methods.items():
            rows = analyze(cache_root, ref_vids)
            tag = f"{method}_{ds}"
            csv_path = os.path.join(out_dir, f"ca_input_frames_{tag}.csv")
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["video_id", "question_id",
                    "candidate_index", "n_intervals", "raw_frames",
                    "ca_input_frames", "capped"])
                w.writeheader()
                w.writerows(rows)

            vals = sorted(r["ca_input_frames"] for r in rows)
            capped = sum(r["capped"] for r in rows)
            n_q = len({(r["video_id"], r["question_id"]) for r in rows})
            print(f"\n=== {ds.upper()} / {method} ===")
            print(f"questions {n_q}  candidate-node inferences {len(rows)} "
                  f"({len(rows)/n_q:.2f}/question)")
            print(f"CA input frames/inference: mean {st.mean(vals):.1f}  "
                  f"median {st.median(vals)}  min {vals[0]}  max {vals[-1]}")
            print(f"hit 256 cap: {capped}/{len(rows)} ({capped/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
