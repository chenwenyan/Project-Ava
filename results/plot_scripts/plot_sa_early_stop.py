#!/usr/bin/env python3
"""OSDI-style reporting figure for SA progressive early stopping (60-question run).

Reads the analysis artifacts produced by run_sa_early_stop.py and renders a
paper-style multi-panel PDF (+ PNG) in a print-friendly grayscale palette:

  <results-dir>/sa_early_stop_summary.json      (sweep over stability_rounds S)
  <results-dir>/sa_early_stop_per_question.csv  (per-question detail at default S)

Output: <results-dir>/sa_early_stop_report.pdf / .png

Style notes (OSDI/systems-paper conventions): serif (Times) fonts, grayscale +
hatched bars so panels read in black-and-white, distinct marker/linestyle per
series, inward ticks, no top/right spines, Type-42 (TrueType) embedded fonts,
compact double-column width.

Run: PYTHONNOUSERSITE=1 <ava py> scripts/plot_sa_early_stop.py --default-s 4
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="findfont")

REPO = Path(__file__).resolve().parents[2]  # results/plot_scripts/ -> repo root

# OSDI / systems-paper grayscale palette (black-and-white safe).
INK = "#000000"
GRAY = "#7A7A7A"
LGRAY = "#BDBDBD"


def set_osdi_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman",
                       "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "lines.markersize": 5,
        "patch.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "figure.dpi": 120,
        "pdf.fonttype": 42,  # embed TrueType (camera-ready friendly)
        "ps.fonttype": 42,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load(results_dir):
    summary = json.loads((results_dir / "sa_early_stop_summary.json").read_text())
    with (results_dir / "sa_early_stop_per_question.csv").open() as f:
        per_q = list(csv.DictReader(f))
    return summary, per_q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=REPO / "results/sa_early_stop")
    ap.add_argument("--default-s", type=int, default=4)
    args = ap.parse_args()

    set_osdi_style()
    summary, per_q = load(args.results_dir)
    S = args.default_s
    sweep = summary["sweep"]
    ss = sorted(int(k) for k in sweep)
    n_q = summary["num_questions"]
    max_rounds = summary["max_rounds"]

    base_acc = [sweep[str(s)]["baseline_acc"] * 100 for s in ss]
    es_acc = [sweep[str(s)]["earlystop_acc"] * 100 for s in ss]
    saving = [sweep[str(s)]["mean_saving_ratio"] * 100 for s in ss]

    stop_rounds = [int(r[f"es{S}_stop_round"]) for r in per_q]
    savings_q = [float(r[f"es{S}_saving_ratio"]) * 100 for r in per_q]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    # (a) accuracy vs S, mean-saving on twin axis
    ax = axes[0, 0]
    ax.plot(ss, base_acc, ls="--", marker="o", mfc="white", color=INK,
            label="baseline (104 resp.)")
    ax.plot(ss, es_acc, ls="-", marker="s", color=INK, label="early-stop")
    ax.set_xlabel("stability rounds $S$")
    ax.set_ylabel("accuracy (%)")
    ax.set_xticks(ss)
    ax.set_ylim(45, 65)
    ax.set_title("(a) Accuracy vs. threshold")
    ax.legend(loc="lower left")
    despine(ax)
    ax2 = ax.twinx()
    ax2.plot(ss, saving, ls=":", marker="^", color=GRAY, label="mean saving")
    ax2.set_ylabel("mean response saving (%)", color=GRAY)
    ax2.tick_params(axis="y", labelcolor=GRAY)
    ax2.set_ylim(0, 40)
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper right")

    # (b) savings vs accuracy tradeoff
    ax = axes[0, 1]
    ax.plot(saving, es_acc, ls="-", marker="o", color=INK)
    for s, sv, a in zip(ss, saving, es_acc):
        ax.annotate(f"$S={s}$", (sv, a), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    ax.axhline(base_acc[0], color=GRAY, ls="--", lw=1.0, label="baseline acc.")
    ax.set_xlabel("mean response saving (%)")
    ax.set_ylabel("early-stop accuracy (%)")
    ax.set_title("(b) Saving / accuracy trade-off")
    ax.legend(loc="lower right")
    despine(ax)

    # (c) early-stop round distribution at default S
    ax = axes[1, 0]
    ax.hist(stop_rounds, bins=range(1, max_rounds + 2), align="left", rwidth=0.82,
            facecolor="white", edgecolor=INK, hatch="////")
    mean_sr = sum(stop_rounds) / len(stop_rounds)
    ax.axvline(mean_sr, color=INK, ls="--", lw=1.1, label=f"mean {mean_sr:.2f}")
    ax.set_xlabel(f"early-stop round ($S={S}$)")
    ax.set_ylabel("# questions")
    ax.set_xticks(range(1, max_rounds + 1))
    ax.set_title("(c) Where early-stop fires")
    ax.legend(loc="upper left")
    despine(ax)

    # (d) per-question savings distribution at default S
    ax = axes[1, 1]
    ax.hist(savings_q, bins=range(0, 101, 10), facecolor=LGRAY, edgecolor=INK,
            hatch="....")
    mean_sv = sum(savings_q) / len(savings_q)
    ax.axvline(mean_sv, color=INK, ls="--", lw=1.1, label=f"mean {mean_sv:.1f}%")
    ax.set_xlabel(f"response saving vs. 104 (%), $S={S}$")
    ax.set_ylabel("# questions")
    ax.set_title("(d) Per-question savings")
    ax.legend(loc="upper right")
    despine(ax)

    fig.suptitle(
        f"SA self-consistency progressive early stopping "
        f"(VideoMME-long, {n_q} Q, 104-response budget)",
        fontsize=9.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=1.4, w_pad=1.6)
    pdf = args.results_dir / "sa_early_stop_report.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
