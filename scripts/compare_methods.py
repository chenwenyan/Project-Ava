#!/usr/bin/env python3
"""Compare AVA chunking methods on the same videos/models.

Methods (by output dir): default (fixed 3s + semantic merge), keyframe-direct
(keyframe=event, no merge), keyframe-merge (keyframe base windows + semantic merge).
Compares over the videos all present methods cover, writes comparison.csv, prints a
table (accuracy, runtime, graph/event size per stage).
"""

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    "videomme": {
        "methods": {
            "default": REPO / "outputs/videomme_long_601_620",
            "keyframe-direct": REPO / "outputs/videomme_keyframe_601_605",
            "keyframe-merge": REPO / "outputs/videomme_keyframe_merge_601_605",
        },
        "caches": {
            "default": REPO / "AVA_cache/VideoMME",
            "keyframe-direct": REPO / "AVA_cache/VideoMME_kf",
            "keyframe-merge": REPO / "AVA_cache/VideoMME_kf_merge",
        },
        "out": REPO / "results/csv/videomme_comparison.csv",
    },
    "lvbench": {
        "methods": {
            "default": REPO / "results/lvbench_default",
            "keyframe-direct": REPO / "results/lvbench_keyframe_direct",
            "keyframe-merge": REPO / "results/lvbench_keyframe_merge",
        },
        "caches": {
            "default": REPO / "AVA_cache/LVBench",
            "keyframe-direct": REPO / "AVA_cache/LVBench_kf",
            "keyframe-merge": REPO / "AVA_cache/LVBench_kf_merge",
        },
        "out": REPO / "results/csv/lvbench_comparison.csv",
    },
}
METHODS, CACHE_ROOTS = {}, {}  # set from --dataset in main()


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def cache_events(method, ids):
    """EKG index size: event count from the real cache (offline index, not timed)."""
    total = 0
    for vid in ids:
        path = CACHE_ROOTS[method] / str(vid) / "kg/events/events.json"
        if path.is_file():
            total += len(json.loads(path.read_text()))
    return total


def vids(records):
    return {int(str(k).split(":")[0]) for k in records}


def stage_metrics(records, video_ids):
    rows = [v for k, v in records.items() if int(str(k).split(":")[0]) in video_ids]
    timed = [r for r in rows if r.get("status") == "ok"]
    total = sum(float(r.get("seconds", 0)) for r in timed)
    scored = [r for r in rows if r.get("status") in {"ok", "skipped"} and r.get("gt")]
    correct = sum(r.get("answer") == r.get("gt") for r in scored)
    return {
        "total_seconds": round(total, 1),
        "mean_seconds": round(total / len(timed), 1) if timed else 0.0,
        "nodes": sum(int(r.get("nodes", 0)) for r in rows),
        "events": sum(int(r.get("events", 0)) for r in rows),
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "correct": correct,
        "scored": len(scored),
    }


def main():
    global METHODS, CACHE_ROOTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["videomme", "lvbench"], default="videomme")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    METHODS, CACHE_ROOTS = cfg["methods"], cfg["caches"]
    args.out = args.out or cfg["out"]

    data, present = {}, {}
    for name, d in METHODS.items():
        recs = {s: load(d / f"records_{s}.json") for s in ("graph", "sa", "ca")}
        if any(recs.values()):
            data[name] = recs
            present[name] = recs

    # Compare over videos covered by every present method's graph stage.
    covered = [vids(recs["graph"]) for recs in present.values() if recs["graph"]]
    shared = sorted(set.intersection(*covered)) if covered else []

    metrics = {name: {s: stage_metrics(recs[s], set(shared)) for s in ("graph", "sa", "ca")}
               for name, recs in present.items()}

    # EKG index size from the real caches; add an online-total stage (SA+CA per question).
    for name in present:
        metrics[name]["graph"]["events"] = cache_events(name, shared)
        metrics[name]["online"] = {
            "mean_seconds": round(metrics[name]["sa"]["mean_seconds"] + metrics[name]["ca"]["mean_seconds"], 1),
            "total_seconds": round(metrics[name]["sa"]["total_seconds"] + metrics[name]["ca"]["total_seconds"], 1),
        }

    names = list(present)
    # Focus: online per-stage latency (SA, CA) + online total; EKG index size (no offline time).
    report = {
        "index": ("events", "nodes"),
        "sa": ("mean_seconds", "accuracy", "correct", "scored"),
        "ca": ("mean_seconds", "accuracy", "correct", "scored"),
        "online": ("mean_seconds", "total_seconds"),
    }
    stage_src = {"index": "graph", "sa": "sa", "ca": "ca", "online": "online"}
    rows = []
    for stage, fields in report.items():
        for field in fields:
            values = {n: metrics[n][stage_src[stage]].get(field) for n in names}
            if all(v in (0, 0.0, None) for v in values.values()):
                continue
            rows.append({"stage": stage, "metric": field, **values})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["stage", "metric", *names])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Methods: {names}")
    print(f"Videos compared ({len(shared)}): {shared}\n")
    head = f"{'stage':6} {'metric':13}" + "".join(f"{n:>17}" for n in names)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['stage']:6} {r['metric']:13}" + "".join(f"{str(r[n]):>17}" for n in names)
        print(line)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
