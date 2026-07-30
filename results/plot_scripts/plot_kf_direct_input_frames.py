#!/usr/bin/env python3
"""Model-input frame distribution: Default (AVA) vs KF-Direct (OSDI style, PDF only).

Both methods feed the captioning VLM a fixed 6 frames per video chunk; they
differ only in chunking (Default = fixed 3s base windows before semantic merge;
KF-Direct = one chunk per codec keyframe interval), so the per-video total input
frames = 6 x #chunks characterises each method's visual-input volume.

One panel per dataset; each overlays the two methods' ECDF of per-video total
input frames with median markers.

Data from scripts/analyze_kf_direct_frames.py:
  results/csv/input_frames_by_video_{default,kf_direct}_{lvbench,videomme}.csv
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/csv"
OUT = REPO / "results/pdf/kf_direct_input_frames.pdf"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})

# grayscale-safe: KF-Direct (our method) solid blue; Default comparison dashed orange.
METHODS = {
    "kf_direct": dict(label="KF-Direct", color="#3b6ea5", ls="-",  lw=1.8),
    "default":   dict(label="Default",   color="#d98b3a", ls="--", lw=1.8),
}
DATASETS = [("lvbench", "(a) LVBench"), ("videomme", "(b) VideoMME")]
# stagger median callouts so the two methods' labels never collide
LABEL_OFF = {"kf_direct": dict(xytext=(6, -15), ha="left"),
             "default":   dict(xytext=(-6, 9),  ha="right")}


def load(method, ds):
    path = CSV / f"input_frames_by_video_{method}_{ds}.csv"
    rows = list(csv.DictReader(open(path)))
    return np.array([int(r["total_input_frames"]) for r in rows], dtype=float)


def main():
    import os
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)

    for ax, (ds, title) in zip(axes, DATASETS):
        for method, st in METHODS.items():
            v = load(method, ds)
            xs = np.sort(v)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(np.concatenate([[0], xs]), np.concatenate([[0], ys]),
                    where="post", color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=st["label"])
            med = np.median(v)
            ax.plot([med], [0.5], marker="o", color=st["color"], markersize=5,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)
            ax.annotate(f"{med/1000:.1f}k", (med, 0.5), textcoords="offset points",
                        color=st["color"], fontsize=9, **LABEL_OFF[method])
        ax.set_title(title)
        ax.set_xlabel("Total input frames per video")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)

    axes[0].set_ylabel("Cumulative fraction of videos")

    fig.text(0.5, -0.02,
             "Both methods feed 6 frames per chunk; per-video total = 6 x #chunks "
             "(Default: 3s windows;  KF-Direct: keyframe intervals).",
             ha="center", fontsize=8.5, color="#444")

    fig.tight_layout()
    out = os.environ.get("KF_OUT", str(OUT))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for ds, _ in DATASETS:
        for method in METHODS:
            v = load(method, ds)
            print(f"{ds:9s} {method:9s}: n={len(v)} mean {v.mean():.0f} "
                  f"median {np.median(v):.0f} min {v.min():.0f} max {v.max():.0f}")


if __name__ == "__main__":
    main()
