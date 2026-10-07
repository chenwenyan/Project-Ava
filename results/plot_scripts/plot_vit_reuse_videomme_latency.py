"""Plot paired CA query latency with and without persistent ViT output reuse."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_records(path):
    data = json.loads(path.read_text())
    expected = {f"{video}:{question}" for video in range(601, 621)
                for question in range(3)}
    if set(data) != expected:
        raise ValueError(f"{path}: expected 60 questions, found {len(data)}; "
                         f"missing={sorted(expected - set(data))}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    off, on = load_records(args.off), load_records(args.on)
    keys = sorted(off, key=lambda key: tuple(map(int, key.split(":"))))
    rows = []
    for key in keys:
        video, question = map(int, key.split(":"))
        off_seconds = float(off[key]["seconds"])
        on_seconds = float(on[key]["seconds"])
        rows.append({
            "video_id": video, "question_id": question,
            "no_reuse_seconds": off_seconds, "reuse_seconds": on_seconds,
            "speedup": off_seconds / on_seconds if on_seconds > 0 else float("nan"),
            "no_reuse_status": off[key]["status"],
            "reuse_status": on[key]["status"],
            "no_reuse_answer": off[key]["answer"],
            "reuse_answer": on[key]["answer"],
        })
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = np.array([row["no_reuse_seconds"] for row in rows])
    reuse = np.array([row["reuse_seconds"] for row in rows])
    medians = [float(np.median(baseline)), float(np.median(reuse))]
    quartiles = [np.percentile(baseline, [25, 75]), np.percentile(reuse, [25, 75])]
    speedup = medians[0] / medians[1]
    metrics = json.loads(args.metrics.read_text())
    summary = {
        "cohort": "VideoMME-Long 601-620, 60 CA questions",
        "method": "kf3sig_a07_t07",
        "n_questions": len(rows),
        "median_no_reuse_seconds": medians[0],
        "median_reuse_seconds": medians[1],
        "median_latency_speedup": speedup,
        "total_no_reuse_seconds": float(baseline.sum()),
        "total_reuse_seconds": float(reuse.sum()),
        "total_latency_speedup": float(baseline.sum() / reuse.sum()),
        "no_reuse_non_ok": sum(row["no_reuse_status"] != "ok" for row in rows),
        "reuse_non_ok": sum(row["reuse_status"] != "ok" for row in rows),
        "different_answers": sum(row["no_reuse_answer"] != row["reuse_answer"]
                                 for row in rows),
        "cache_metrics": metrics,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=150)
    positions = [0, 0.75]
    bars = ax.bar(positions, medians, width=0.42,
                  color=["#9CBBC9", "#276B87"], edgecolor="none")
    for i, (bar, median, quartile) in enumerate(zip(bars, medians, quartiles)):
        ax.errorbar(positions[i], median,
                    yerr=[[median - quartile[0]], [quartile[1] - median]],
                    fmt="none", ecolor="#253746", capsize=3, linewidth=1)
        ax.text(bar.get_x() + bar.get_width() - 0.025, median - 1.0,
                f"{median:.1f}", ha="right", va="top", fontsize=8,
                color="#253746" if i == 0 else "white")
    ax.set_xticks(positions, ["No reuse", "ViT reuse"])
    ax.set_ylabel("Median CA latency (s/query)")
    ax.set_ylim(0, max(q[1] for q in quartiles) * 1.25)
    label = f"{speedup:.2f}× speedup"
    if speedup < 1:
        label += " (slower)"
    ax.text(0.5, 0.98, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, weight="bold")
    ax.set_title("VideoMME-Long 601–620 (60 queries)", pad=5, fontsize=9)
    fig.tight_layout(pad=0.5)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.pdf}", flush=True)


if __name__ == "__main__":
    main()
