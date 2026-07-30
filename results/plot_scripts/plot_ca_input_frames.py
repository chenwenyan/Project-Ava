#!/usr/bin/env python3
"""CA-stage model-input frame distribution: Default vs KF-Direct (OSDI, PDF only).

Per question, the CA stage (generate_ca_self_consistency_result) feeds the VLM
one video input per candidate node (<=2/question): frames from the node's
retrieved intervals sampled at 1 fps, capped at max_frames=256. This plots the
ECDF of that per-inference frame count for each method, one panel per dataset,
with the 256 cap marked.

Data from scripts/analyze_ca_input_frames.py:
  results/csv/ca_input_frames_{default,kf_direct}_{lvbench,videomme}.csv
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/csv"
OUT = REPO / "results/pdf/ca_input_frames.pdf"
CAP = 256

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})

METHODS = {
    "kf_direct": dict(label="KF-Direct", color="#3b6ea5", ls="-",  lw=1.8),
    "default":   dict(label="Default",   color="#d98b3a", ls="--", lw=1.8),
}
DATASETS = [("lvbench", "(a) LVBench"), ("videomme", "(b) VideoMME")]
LABEL_OFF = {"kf_direct": dict(xytext=(-6, 10), ha="right"),
             "default":   dict(xytext=(6, -14), ha="left")}


def load(method, ds):
    rows = list(csv.DictReader(open(CSV / f"ca_input_frames_{method}_{ds}.csv")))
    return np.array([int(r["ca_input_frames"]) for r in rows], dtype=float)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)

    for ax, (ds, title) in zip(axes, DATASETS):
        ax.axvline(CAP, color="#888", ls=":", lw=1.0, zorder=1)
        ax.text(CAP, 0.03, " 256 cap", color="#666", fontsize=8,
                ha="right", rotation=90, va="bottom")
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
            ax.annotate(f"{med:.0f}", (med, 0.5), textcoords="offset points",
                        color=st["color"], fontsize=9, **LABEL_OFF[method])
        ax.set_title(title)
        ax.set_xlabel("CA input frames per inference")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, CAP + 8)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)

    axes[0].set_ylabel("Cumulative fraction")

    fig.tight_layout()
    out = os.environ.get("KF_OUT", str(OUT))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for ds, _ in DATASETS:
        for method in METHODS:
            v = load(method, ds)
            print(f"{ds:9s} {method:9s}: n={len(v)} mean {v.mean():.1f} "
                  f"median {np.median(v):.0f} min {v.min():.0f} max {v.max():.0f}")


if __name__ == "__main__":
    main()
