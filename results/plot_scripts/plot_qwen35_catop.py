#!/usr/bin/env python3
"""OSDI-style CA candidate-budget ablation (top-2 default vs top-3/top-4) for the
new-model runs. Grouped bars per chunking method; panel (a) end-to-end CA
accuracy, panel (b) CA latency per query. PDF only.

SA stage is identical across budgets (results reused), so only CA differs.
Data: outputs/<ds>_q35_<method>[_601_620][_top{3,4}]/records_ca.json
Out:  results/pdf/<ds>_qwen35_catop.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
METHODS = {"Default": "default", "KF-Direct (3s)": "kf3s", "KF-Merge": "kfmerge"}
TOPS = [2, 3, 4]
# Grayscale-safe sequence: lightness ramp + distinct hatches.
TOP_STYLE = {
    2: dict(color="#d98b3a", hatch="///"),
    3: dict(color="#3b6ea5", hatch=""),
    4: dict(color="#7fa8d0", hatch="\\\\\\"),
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})


def record_dir(ds, method, top):
    suffix = "" if top == 2 else f"_top{top}"
    if ds == "videomme":
        return REPO / f"outputs/videomme_q35_{method}_601_620{suffix}"
    return REPO / f"outputs/lvbench_q35_{method}{suffix}"


def stats(ds, method, top):
    recs = json.loads((record_dir(ds, method, top) / "records_ca.json").read_text())
    ok = [r for r in recs.values() if r["status"] == "ok"]
    acc = 100 * sum(r["answer"] == r["gt"] for r in ok) / len(ok) if ok else 0.0
    lat = sum(float(r["seconds"]) for r in ok) / len(ok) if ok else 0.0
    return acc, lat, len(ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["videomme", "lvbench"], required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or REPO / f"results/pdf/{args.dataset}_qwen35_catop.pdf"

    acc = {t: [] for t in TOPS}
    lat = {t: [] for t in TOPS}
    n = None
    for label, m in METHODS.items():
        for t in TOPS:
            a, l, k = stats(args.dataset, m, t)
            acc[t].append(a)
            lat[t].append(l)
            n = k

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.75))
    x = list(range(len(METHODS)))
    w = 0.26

    for j, t in enumerate(TOPS):
        off = (j - 1) * w
        bars = ax1.bar([i + off for i in x], acc[t], w, edgecolor="black",
                       linewidth=0.8, label=f"top-{t}", **TOP_STYLE[t])
        for bar, v in zip(bars, acc[t]):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=7.5)
        ax2.bar([i + off for i in x], lat[t], w, edgecolor="black",
                linewidth=0.8, label=f"top-{t}", **TOP_STYLE[t])
        for i, v in enumerate(lat[t]):
            ax2.text(i + off, v + max(max(lat[tt]) for tt in TOPS) * 0.02, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=7.5)

    ax1.set_ylabel("CA accuracy (%)")
    ax1.set_ylim(0, max(max(acc[t]) for t in TOPS) * 1.42)
    ax1.set_title("(a) End-to-end accuracy")
    ax2.set_ylabel("CA latency / query (s)")
    ax2.set_ylim(0, max(max(lat[t]) for t in TOPS) * 1.42)
    ax2.set_title("(b) CA latency")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(list(METHODS))
        ax.legend(frameon=False, ncol=3, loc="upper center", handlelength=1.2,
                  columnspacing=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")  # PDF only
    print(f"{args.dataset} (n={n}/method)")
    for t in TOPS:
        print(f"top-{t}: acc={[round(a,1) for a in acc[t]]} lat={[round(l) for l in lat[t]]}s")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
