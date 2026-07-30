#!/usr/bin/env python3
"""End-to-end (CA-stage) accuracy of SA early-stop vs full budget (OSDI, PDF only).

Companion to sa_early_stop_savings.pdf (whose panel (a) reports SA accuracy).
Here every config's answer went through the full pipeline: SA candidates from
the same per-question trajectory, then the CA stage (qwenvl frame recheck) was
re-run per distinct signature by scripts/run_sa_earlystop_ca.py.
baseline = CA on the round-8 (full 104-response) candidates, flat across S;
early-stop(S) = CA on the stop-round candidates.

NOTE: the CA stage is degenerate on VideoMME-long (256-frame inputs overflow
the VLM context -> mostly empty responses -> defaults to "A"; see
results/sa_early_stop/CA_degeneration_diagnosis.md), so absolute numbers sit
near the GT-A base rate and differences between configs are within noise.

Data: results/csv/sa_early_stop_ca_accuracy.csv
Out:  results/pdf/sa_early_stop_ca_accuracy.pdf
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

BASE_STYLE = dict(color="#d98b3a", hatch="///")  # full budget: hatched orange
ES_STYLE = dict(color="#3b6ea5", hatch="")       # early-stop: solid blue
GAIN = "#127c2a"
LOSS = "#b3202c"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=REPO / "results/csv/sa_early_stop_ca_accuracy.csv")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results/pdf/sa_early_stop_ca_accuracy.pdf")
    args = parser.parse_args()

    rows = {r["config"]: r for r in csv.DictReader(args.csv.open())}
    ss = sorted(int(k[1:]) for k in rows if k.startswith("S"))
    base = float(rows["baseline"]["ca_accuracy_pct"])
    es = [float(rows[f"S{s}"]["ca_accuracy_pct"]) for s in ss]
    n = int(rows["baseline"]["n"])

    x = list(range(len(ss)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.75))

    ax.bar([i - w / 2 for i in x], [base] * len(ss), w, edgecolor="black",
           linewidth=0.8, label="full budget", **BASE_STYLE)
    ax.bar([i + w / 2 for i in x], es, w, edgecolor="black", linewidth=0.8,
           label="early-stop", **ES_STYLE)
    for i in x:
        ax.text(i - w / 2, base + 0.8, f"{base:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, es[i] + 0.8, f"{es[i]:.0f}", ha="center", va="bottom", fontsize=8)
        d = es[i] - base
        ax.text(i + w / 2, es[i] + 4.2, f"{d:+.0f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=GAIN if d >= 0 else LOSS)
    ax.set_ylabel("CA accuracy (%)")
    ax.set_ylim(0, max([base] + es) * 1.5)
    ax.set_title(f"End-to-end (CA) accuracy, {n} questions")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S={s}" for s in ss])
    ax.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.4,
              columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")  # PDF only
    print(f"baseline CA acc={base:.1f}%  early-stop CA acc={es} (S={ss})")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
