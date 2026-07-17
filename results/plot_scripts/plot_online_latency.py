#!/usr/bin/env python3
"""OSDI-style latency breakdown of the AVA online (query-answering) stage.

Reads the per-question timing CSV from run_online_latency.py and renders a
paper-style PDF (PDF only, per project plotting convention):

  (a) per-question stacked bars  -- online latency composed of the 5 components
  (b) mean breakdown             -- mean seconds + % share per component

Components (bottom -> top): tree_search, SA-LLM, SA-score, CA-VLM, CA-score.

Run: PYTHONNOUSERSITE=1 <ava py> scripts/plot_online_latency.py --csv results/online_latency_videomme_601_603.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]  # results/plot_scripts/ -> repo root

# component key -> (label incl. the model that runs the stage, fill).
# Palette: shades of the ¥10 RMB note (teal). Model wiring is AVA default:
# tree search + SA = qwenlm (Qwen2.5-32B), CA = qwenvl (Qwen2.5-VL-7B),
# SA/CA scoring = BERTScore (DeBERTa-xl-MNLI).
COMPONENTS = [
    ("tree_search_s", "tree search  ·  Qwen2.5-32B",        "#A7DCE0"),
    ("sa_llm_s",      "SA self-consistency  ·  Qwen2.5-32B", "#2E8B9E"),
    ("sa_score_s",    "SA scoring  ·  DeBERTa-xl",           "#7FC6CE"),
    ("ca_vlm_s",      "CA self-consistency  ·  Qwen2.5-VL-7B", "#165A6B"),
    ("ca_score_s",    "CA scoring  ·  DeBERTa-xl",           "#0C3A46"),
]
INK = "#0C3A46"      # dark teal used for reference lines / annotations
EDGE = "#0C3A46"     # dark-teal edges so hatch patterns read over the fill
# distinct hatch per stage (grayscale-safe on top of the teal ramp)
HATCHES = ["////", "....", "xxxx", "\\\\\\\\", "++"]


def set_osdi_style():
    # OSDI layout conventions, but keep matplotlib's default font.
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3, "patch.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "figure.dpi": 120, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load(csv_path):
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                vals = {k: float(r[k]) for k, *_ in COMPONENTS}
            except (ValueError, KeyError):
                continue
            if sum(vals.values()) <= 0:
                continue
            vals["label"] = f'{r["video_id"]}:{r["question_id"]}'
            rows.append(vals)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--png", type=Path, default=None,
                    help="optional debug PNG copy (results stays PDF-only)")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"no fully-timed questions in {args.csv}")
    out = args.out or args.csv.with_name(
        args.csv.stem.replace("online_latency", "online_latency_breakdown") + ".pdf")

    set_osdi_style()
    keys = [k for k, *_ in COMPONENTS]
    labels = [lab for _, lab, _ in COMPONENTS]
    fills = [c for _, _, c in COMPONENTS]

    data = {k: np.array([r[k] for r in rows]) for k in keys}
    xlabels = [r["label"] for r in rows]
    totals = np.sum([data[k] for k in keys], axis=0)
    means = {k: data[k].mean() for k in keys}
    grand_mean = sum(means.values())
    n = len(rows)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                     gridspec_kw={"width_ratios": [1.7, 1]})

    # (a) per-question stacked bars
    x = np.arange(n)
    bottom = np.zeros(n)
    for k, fill, hatch in zip(keys, fills, HATCHES):
        ax_a.bar(x, data[k], bottom=bottom, width=0.8, facecolor=fill,
                 edgecolor=EDGE, linewidth=0.5, hatch=hatch)
        bottom += data[k]
    ax_a.set_ylabel("online latency (s)")
    ax_a.set_xlabel("question (video:qid)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, rotation=90)
    ax_a.set_xlim(-0.6, n - 0.4)
    ax_a.set_ylim(0, totals.max() * 1.08)
    ax_a.axhline(grand_mean, color=INK, ls=":", lw=1.0)
    ax_a.text(n - 0.5, grand_mean, f" mean {grand_mean:.0f}s", va="bottom",
              ha="right", fontsize=7.5)
    ax_a.set_title(f"(a) Per-question breakdown ($n={n}$)")
    despine(ax_a)

    # (b) mean breakdown as a single stacked bar with % labels
    bottom = 0.0
    for k, lab, fill, hatch in zip(keys, labels, fills, HATCHES):
        h = means[k]
        ax_b.bar(0, h, bottom=bottom, width=0.6, facecolor=fill, edgecolor=EDGE,
                 linewidth=0.5, hatch=hatch, label=f"{lab}")
        pct = 100 * h / grand_mean
        if pct >= 3:
            ax_b.text(0.34, bottom + h / 2, f"{h:.1f}s ({pct:.0f}%)",
                      va="center", ha="left", fontsize=7.2)
        bottom += h
    ax_b.set_ylabel("mean latency (s)")
    ax_b.set_xlim(-0.5, 1.6)
    ax_b.set_ylim(0, grand_mean * 1.05)
    ax_b.set_xticks([])
    ax_b.set_title("(b) Mean breakdown")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1,
                handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    despine(ax_b)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(out)   # PDF only (project convention)
    if args.png:
        fig.savefig(args.png, dpi=200)
    plt.close(fig)

    print(f"components (mean seconds / share):")
    for k, lab, *_ in COMPONENTS:
        print(f"  {lab:32s} {means[k]:7.2f}s  {100*means[k]/grand_mean:5.1f}%")
    print(f"  {'TOTAL':32s} {grand_mean:7.2f}s")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
