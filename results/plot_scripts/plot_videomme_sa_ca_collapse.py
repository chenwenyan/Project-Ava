#!/usr/bin/env python3
"""Why Default's accuracy in videomme.pdf is low: the CA-stage collapse (OSDI, PDF).

videomme.pdf plots end-to-end (CA) accuracy. This shows that all methods reach
similar SA accuracy on VideoMME-long 601-620, but the CA stage flips many
correct SA answers to wrong -- worst for Default -- so Default's plotted CA
accuracy is lowest.

Panel (a): paired SA vs CA accuracy per method, with the SA->CA drop annotated.
Panel (b): SA->CA transition counts (destroyed vs rescued) per method.

Data: results/csv/videomme_sa_ca_transitions.csv (analyze_default_ca_collapse.py).
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "results/csv/videomme_sa_ca_transitions.csv"
OUT = REPO / "results/pdf/videomme_sa_ca_collapse.pdf"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})
SA_C = "#3b6ea5"
CA_C = "#d98b3a"


def main():
    rows = list(csv.DictReader(open(CSV)))
    names = [r["method"] for r in rows]
    sa = [float(r["sa_acc"]) for r in rows]
    ca = [float(r["ca_acc"]) for r in rows]
    destroyed = [int(r["destroyed_sa_ok_ca_wrong"]) for r in rows]
    rescued = [int(r["rescued_sa_wrong_ca_ok"]) for r in rows]
    x = np.arange(len(names))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # (a) SA vs CA accuracy
    ax1.bar(x - w/2, sa, w, color=SA_C, edgecolor="black", linewidth=0.8, label="SA")
    ax1.bar(x + w/2, ca, w, color=CA_C, hatch="///", edgecolor="black",
            linewidth=0.8, label="CA (end-to-end)")
    for xi, (s, c) in enumerate(zip(sa, ca)):
        ax1.text(xi - w/2, s + 1, f"{s:.0f}", ha="center", va="bottom", fontsize=8)
        ax1.text(xi + w/2, c + 1, f"{c:.0f}", ha="center", va="bottom", fontsize=8)
        ax1.annotate(f"-{s-c:.0f}", (xi, min(s, c)), (xi, (s + c) / 2),
                     ha="center", va="center", fontsize=8, color="#c0392b",
                     fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(sa) * 1.25)
    ax1.set_title("(a) SA vs CA accuracy")
    ax1.legend(frameon=False, loc="upper right")

    # (b) transition counts
    ax2.bar(x - w/2, destroyed, w, color="#c0392b", edgecolor="black",
            linewidth=0.8, label="SA✓→CA✗ destroyed")
    ax2.bar(x + w/2, rescued, w, color="#127c2a", edgecolor="black",
            linewidth=0.8, label="SA✗→CA✓ rescued")
    for xi, (d, r) in enumerate(zip(destroyed, rescued)):
        ax2.text(xi - w/2, d + 0.3, str(d), ha="center", va="bottom", fontsize=8)
        ax2.text(xi + w/2, r + 0.3, str(r), ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("# questions (of 60)")
    ax2.set_ylim(0, max(destroyed) * 1.3)
    ax2.set_title("(b) CA-stage answer flips")
    ax2.legend(frameon=False, loc="upper right")

    for ax in (ax1, ax2):
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    out = os.environ.get("KF_OUT", str(OUT))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
