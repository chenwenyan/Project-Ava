#!/usr/bin/env python3
"""OSDI-style comparison for the NEW-model runs (SA=Qwen3.5-27B-FP8,
CA=Qwen3-VL-8B-Instruct-FP8): CA accuracy + online latency for the three
chunking methods on VideoMME-Long 601-620. PDF only.

Same layout as plot_osdi_comparison.py: (a) end-to-end CA accuracy bars,
(b) stacked SA+CA online latency, KF-Direct-vs-Default callouts.
Data: outputs/videomme_q35_{default,kf3s,kfmerge}_601_620/records_{sa,ca}.json
Out:  results/pdf/videomme_qwen35.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATASETS = {
    "videomme": {
        "dirs": {
            "Default": REPO / "outputs/videomme_q35_default_601_620",
            "KF-Direct (3s)": REPO / "outputs/videomme_q35_kf3s_601_620",
            "KF-Merge": REPO / "outputs/videomme_q35_kfmerge_601_620",
        },
        "out": REPO / "results/pdf/videomme_qwen35.pdf",
    },
    "lvbench": {
        "dirs": {
            "Default": REPO / "outputs/lvbench_q35_default",
            "KF-Direct (3s)": REPO / "outputs/lvbench_q35_kf3s",
            "KF-Merge": REPO / "outputs/lvbench_q35_kfmerge",
        },
        "out": REPO / "results/pdf/lvbench_qwen35.pdf",
    },
}
SA_STYLE = dict(color="#3b6ea5", hatch="")     # SA: solid blue
CA_STYLE = dict(color="#d98b3a", hatch="///")  # CA: hatched orange
GAIN = "#127c2a"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def vids(recs):
    return {int(str(k).split(":")[0]) for k in recs}


def accuracy(recs, ids):
    rows = [v for k, v in recs.items()
            if int(str(k).split(":")[0]) in ids and v["status"] == "ok" and v.get("gt")]
    return 100 * sum(r["answer"] == r["gt"] for r in rows) / len(rows) if rows else 0.0


def mean_seconds(recs, ids):
    t = [float(v["seconds"]) for k, v in recs.items()
         if int(str(k).split(":")[0]) in ids and v["status"] == "ok" and v.get("seconds")]
    return sum(t) / len(t) if t else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="videomme")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    args.out = args.out or cfg["out"]

    data = {n: {s: load(d / f"records_{s}.json") for s in ("sa", "ca")}
            for n, d in cfg["dirs"].items()}
    shared = set.intersection(*[vids(r["sa"]) for r in data.values() if r["sa"]])
    names = list(cfg["dirs"])
    sa_acc = [accuracy(data[n]["sa"], shared) for n in names]
    ca_acc = [accuracy(data[n]["ca"], shared) for n in names]
    sa_lat = [mean_seconds(data[n]["sa"], shared) for n in names]
    ca_lat = [mean_seconds(data[n]["ca"], shared) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.75))
    x = list(range(len(names)))
    di, ki = 0, 1  # Default, KF-Direct

    bars = ax1.bar(x, ca_acc, 0.55, edgecolor="black", linewidth=0.8, **CA_STYLE)
    for bar, v in zip(bars, ca_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                 va="bottom", fontsize=8.5)
    ax1.set_ylabel("CA accuracy (%)")
    ax1.set_ylim(0, (max(ca_acc) or 1) * 1.42)
    ax1.set_title("(a) End-to-end accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    if ca_acc[di]:
        rel = (ca_acc[ki] - ca_acc[di]) / ca_acc[di] * 100
        ax1.text(ki, ca_acc[ki] + ax1.get_ylim()[1] * 0.11, f"{rel:+.0f}%\nvs Default",
                 ha="center", va="bottom", fontsize=9, color=GAIN, fontweight="bold")

    ax2.bar(x, sa_lat, 0.55, label="SA", edgecolor="black", linewidth=0.8, **SA_STYLE)
    ax2.bar(x, ca_lat, 0.55, bottom=sa_lat, label="CA", edgecolor="black", linewidth=0.8, **CA_STYLE)
    for i in x:
        total = sa_lat[i] + ca_lat[i]
        ax2.text(i, total + max(sa_lat) * 0.03, f"{total:.0f}s", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylabel("Online latency / query (s)")
    ax2.set_ylim(0, max(sa_lat[i] + ca_lat[i] for i in x) * 1.32)
    ax2.set_title("(b) Online query latency")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.4, columnspacing=1.0)
    total = [sa_lat[i] + ca_lat[i] for i in x]
    if total[di]:
        rel = (total[ki] - total[di]) / total[di] * 100
        ax2.text(ki, total[ki] + ax2.get_ylim()[1] * 0.05, f"{rel:+.0f}%\nvs Default",
                 ha="center", va="bottom", fontsize=9, color=GAIN, fontweight="bold")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")  # PDF only
    print(f"methods={names} videos={len(shared)}")
    print(f"SA acc={[round(a,1) for a in sa_acc]} CA acc={[round(a,1) for a in ca_acc]}")
    print(f"SA lat={[round(l) for l in sa_lat]}s CA lat={[round(l) for l in ca_lat]}s "
          f"total={[round(t) for t in total]}s")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
