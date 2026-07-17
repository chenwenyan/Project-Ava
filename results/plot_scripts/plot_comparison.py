#!/usr/bin/env python3
"""Bar charts comparing the chunking methods: SA/CA accuracy and mean per-item
runtime per stage, over the videos all methods share. Saves PDF + PNG."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
    "axes.linewidth": 1.1, "savefig.dpi": 300,
})

REPO = Path(__file__).resolve().parents[2]
DATASETS = {
    "videomme": {
        "methods": {
            "default": REPO / "outputs/videomme_long_601_620",
            "keyframe-direct": REPO / "outputs/videomme_keyframe_601_605",
            "keyframe-merge": REPO / "outputs/videomme_keyframe_merge_601_605",
        },
        "out": REPO / "results/pdf/videomme_method_comparison.pdf",
    },
    "lvbench": {
        "methods": {
            "default": REPO / "results/lvbench_default",
            "keyframe-direct": REPO / "results/lvbench_keyframe_direct",
            "keyframe-merge": REPO / "results/lvbench_keyframe_merge",
        },
        "out": REPO / "results/pdf/lvbench_method_comparison.pdf",
    },
}
METHODS = {}  # set from --dataset in main()
COLORS = {"default": "#c0392b", "keyframe-direct": "#1f4e79", "keyframe-merge": "#27ae60"}


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def vids(records):
    return {int(str(k).split(":")[0]) for k in records}


def accuracy(records, ids):
    rows = [v for k, v in records.items()
            if int(str(k).split(":")[0]) in ids and v.get("status") in {"ok", "skipped"} and v.get("gt")]
    return (sum(r["answer"] == r["gt"] for r in rows) / len(rows), len(rows)) if rows else (None, 0)


def mean_seconds(records, ids):
    timed = [v for k, v in records.items()
             if int(str(k).split(":")[0]) in ids and v.get("status") == "ok" and v.get("seconds")]
    return sum(float(r["seconds"]) for r in timed) / len(timed) if timed else 0.0


def main():
    global METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["videomme", "lvbench"], default="videomme")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    METHODS = DATASETS[args.dataset]["methods"]
    args.out = args.out or DATASETS[args.dataset]["out"]

    data = {name: {s: load(d / f"records_{s}.json") for s in ("graph", "sa", "ca")}
            for name, d in METHODS.items()}
    present = {n: r for n, r in data.items() if any(r.values())}
    covered = [vids(r["graph"]) for r in present.values() if r["graph"]]
    shared = set.intersection(*covered) if covered else set()
    names = list(present)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    x = np.arange(2)  # SA, CA
    width = 0.8 / max(len(names), 1)

    # Left: accuracy (SA, CA).
    ax = axes[0]
    for i, name in enumerate(names):
        accs = [accuracy(present[name][s], shared)[0] or 0 for s in ("sa", "ca")]
        bars = ax.bar(x + i * width, accs, width, label=name, color=COLORS.get(name, "#888"))
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.1%}", ha="center", fontsize=12)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(["SA accuracy", "CA accuracy"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Answer accuracy (videos {min(shared)}-{max(shared)}, n={len(shared)})")
    ax.legend(fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: online query latency per question (SA, CA, and their total).
    ax = axes[1]
    xs = np.arange(3)  # SA, CA, online total
    for i, name in enumerate(names):
        sa = mean_seconds(present[name]["sa"], shared)
        ca = mean_seconds(present[name]["ca"], shared)
        secs = [sa, ca, sa + ca]
        bars = ax.bar(xs + i * width, secs, width, label=name, color=COLORS.get(name, "#888"))
        for b, v in zip(bars, secs):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(xs + width * (len(names) - 1) / 2)
    ax.set_xticklabels(["SA / question", "CA / question", "online total / question"])
    ax.set_ylabel("mean seconds")
    ax.set_title("Online query latency per question")
    ax.legend(fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"AVA chunking methods — performance comparison ({args.dataset})", fontsize=20)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)  # PDF only
    print(f"methods={names}, videos={sorted(shared)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
