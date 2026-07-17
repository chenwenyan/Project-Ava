#!/usr/bin/env python3
"""OSDI/systems-paper-style figure: answer accuracy + online query latency for the
three chunking methods. Two compact panels, serif fonts, grayscale-safe fills with
hatches, minimal chrome, vector PDF + PNG. --dataset {videomme, lvbench}."""

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
            "Default": REPO / "outputs/videomme_long_601_620",
            "KF-Direct": REPO / "outputs/videomme_keyframe_601_605",
            "KF-Merge": REPO / "outputs/videomme_keyframe_merge_601_605",
        },
        "out": REPO / "results/pdf/osdi_videomme.pdf", "title": "VideoMME-Long",
    },
    "lvbench": {
        "dirs": {
            "Default": REPO / "results/lvbench_default",
            "KF-Direct": REPO / "results/lvbench_keyframe_direct",
            "KF-Merge": REPO / "results/lvbench_keyframe_merge",
        },
        "out": REPO / "results/pdf/osdi_lvbench.pdf", "title": "LVBench",
    },
}
# Grayscale-safe: distinct lightness + hatch so it survives B&W printing.
SA_STYLE = dict(color="#3b6ea5", hatch="")     # SA: solid blue
CA_STYLE = dict(color="#d98b3a", hatch="///")  # CA: hatched orange

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
            if int(str(k).split(":")[0]) in ids and v["status"] in {"ok", "skipped"} and v.get("gt")]
    return 100 * sum(r["answer"] == r["gt"] for r in rows) / len(rows) if rows else 0.0


def mean_seconds(recs, ids):
    t = [float(v["seconds"]) for k, v in recs.items()
         if int(str(k).split(":")[0]) in ids and v["status"] == "ok" and v.get("seconds")]
    return sum(t) / len(t) if t else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["videomme", "lvbench"], default="videomme")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    out = args.out or cfg["out"]

    data = {name: {s: load(d / f"records_{s}.json") for s in ("sa", "ca")}
            for name, d in cfg["dirs"].items()}
    shared = set.intersection(*[vids(r["sa"]) for r in data.values() if r["sa"]])
    names = list(cfg["dirs"])
    sa_acc = [accuracy(data[n]["sa"], shared) for n in names]
    ca_acc = [accuracy(data[n]["ca"], shared) for n in names]
    sa_lat = [mean_seconds(data[n]["sa"], shared) for n in names]
    ca_lat = [mean_seconds(data[n]["ca"], shared) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75))
    x = list(range(len(names)))

    di, ki = names.index("Default"), names.index("KF-Direct")
    GAIN = "#127c2a"  # highlight colour for KF-Direct-vs-Default relative change

    # (a) End-to-end CA accuracy: one bar per method (SA is intermediate, not shown).
    bars = ax1.bar(x, ca_acc, 0.55, edgecolor="black", linewidth=0.8, **CA_STYLE)
    for bar, v in zip(bars, ca_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                 va="bottom", fontsize=8.5)
    ax1.set_ylabel("CA accuracy (%)")
    ax1.set_ylim(0, (max(ca_acc) or 1) * 1.42)
    ax1.set_title("(a) End-to-end accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    if ca_acc[di]:  # relative accuracy gain of KF-Direct vs Default
        rel = (ca_acc[ki] - ca_acc[di]) / ca_acc[di] * 100
        ax1.text(ki, ca_acc[ki] + ax1.get_ylim()[1] * 0.11, f"{rel:+.0f}%\nvs Default",
                 ha="center", va="bottom", fontsize=9, color=GAIN, fontweight="bold")

    # (b) Online latency: stacked SA + CA per method (= total time to the CA answer).
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
    if total[di]:  # relative latency reduction of KF-Direct vs Default
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")  # PDF only
    print(f"methods={names} videos={sorted(shared)}")
    print(f"SA acc={[round(a,1) for a in sa_acc]} CA acc={[round(a,1) for a in ca_acc]}")
    print(f"online latency={[round(sa_lat[i]+ca_lat[i]) for i in x]}s")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
