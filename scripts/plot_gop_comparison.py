#!/usr/bin/env python3
"""KF-Direct GOP-granularity comparison: accuracy + latency of the GOP-re-encoded
VideoMME 601-620 runs (gop3s / gop5s, mode B) vs the native ~7s-GOP KF-Direct baseline.

Reads each run's summary.json (written by aggregate_timing.py) plus its events/graph
caches for segment & node counts, writes results/csv/gop_comparison.csv and an
OSDI-style results/pdf/gop_accuracy_latency_comparison.pdf (PDF only). Variants whose
summary.json is missing (e.g. gop5s still running) are skipped automatically, so this
can be run for an intermediate gop3s-only comparison and re-run later with gop5s."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name -> (output_dir with summary.json, kf cache root for seg/node counts, native-GOP seconds)
# ordered coarse->fine by GOP cap so the x-axis reads as a granularity sweep.
RUNS = [
    ("7s\n(native)", "outputs/videomme_keyframe_601_605",       "AVA_cache/VideoMME_kf",       7.0),
    ("5s",           "outputs/videomme_keyframe_gop5s_601_620", "AVA_cache/VideoMME_kf_gop5s", 5.0),
    ("3s",           "outputs/videomme_keyframe_gop3s_601_620", "AVA_cache/VideoMME_kf_gop3s", 3.0),
]
VIDEO_IDS = list(range(601, 621))

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})
# OSDI grayscale-safe: distinct lightness + hatch so it survives B&W printing.
BAR = {"SA": dict(color="#2f6f8f", hatch=""), "CA": dict(color="#a9cfe0", hatch="///")}
LINE = "#c04a3b"  # online-latency accent


def load_summary(out_dir):
    p = REPO / out_dir / "summary.json"
    if not p.exists():
        return None
    return {s["stage"]: s for s in json.loads(p.read_text())}


def seg_nodes(cache_root):
    """Total keyframe segments (events.json) and graph nodes across the 20 videos."""
    segs = nodes = 0
    for vid in VIDEO_IDS:
        ev = REPO / cache_root / str(vid) / "kg/events/events.json"
        if ev.is_file():
            segs += len(json.loads(ev.read_text()))
        gm = REPO / cache_root / str(vid) / "kg/graph_event_knowledge_graph.graphml"
        if gm.is_file():
            nodes += gm.read_text(errors="ignore").count("<node ")
    return segs, nodes


def main():
    rows = []
    for name, out_dir, cache_root, gop_s in RUNS:
        summ = load_summary(out_dir)
        if summ is None:
            print(f"[skip] {name}: no summary.json yet ({out_dir})")
            continue
        sa, ca = summ.get("sa", {}), summ.get("ca", {})
        segs, nodes = seg_nodes(cache_root)
        rows.append({
            "name": name.replace("\n", " "), "label": name, "gop_s": gop_s,
            "sa_acc": sa.get("accuracy", 0.0), "ca_acc": ca.get("accuracy", 0.0),
            "sa_mean_s": sa.get("mean_seconds", 0.0), "ca_mean_s": ca.get("mean_seconds", 0.0),
            "online_mean_s": sa.get("mean_seconds", 0.0) + ca.get("mean_seconds", 0.0),
            "graph_total_s": summ.get("graph", {}).get("total_seconds", 0.0),
            "segments": segs, "nodes": nodes,
        })
    if not rows:
        sys.exit("no runs with summary.json yet -- nothing to plot")

    # ---- CSV ----
    csv = REPO / "results/csv/gop_comparison.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["name", "gop_s", "segments", "nodes", "sa_acc", "ca_acc",
            "sa_mean_s", "ca_mean_s", "online_mean_s", "graph_total_s"]
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    for r in rows:
        print(f"{r['name']:10s} seg={r['segments']:6d} nodes={r['nodes']:6d} "
              f"SA_acc={r['sa_acc']:.3f} CA_acc={r['ca_acc']:.3f} "
              f"online_mean={r['online_mean_s']:.1f}s")

    labels = [r["label"] for r in rows]
    x = list(range(len(rows)))

    # ---- plot: (a) accuracy grouped bars, (b) latency (stacked SA+CA bars + online line) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # (a) SA & CA accuracy
    w = 0.38
    for k, dx in (("SA", -w / 2), ("CA", w / 2)):
        key = "sa_acc" if k == "SA" else "ca_acc"
        vals = [r[key] for r in rows]
        bars = ax1.bar([i + dx for i in x], vals, w, edgecolor="black",
                       linewidth=0.7, label=f"{k} accuracy", **BAR[k])
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("Accuracy (VideoMME 601-620)")
    ax1.set_title("(a) Accuracy vs GOP cap")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("max GOP cap")
    ax1.set_ylim(0, max(max(r["sa_acc"], r["ca_acc"]) for r in rows) * 1.28)
    ax1.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.2)

    # (b) latency: stacked SA+CA mean seconds, with online-mean callout
    sa_v = [r["sa_mean_s"] for r in rows]
    ca_v = [r["ca_mean_s"] for r in rows]
    ax2.bar(x, sa_v, 0.5, edgecolor="black", linewidth=0.7, label="SA mean", **BAR["SA"])
    ax2.bar(x, ca_v, 0.5, bottom=sa_v, edgecolor="black", linewidth=0.7,
            label="CA mean", **BAR["CA"])
    for i, r in enumerate(rows):
        ax2.text(i, r["online_mean_s"] + max(r2["online_mean_s"] for r2 in rows) * 0.02,
                 f"{r['online_mean_s']:.0f}s", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=LINE)
    ax2.set_ylabel("Online latency / question (s)")
    ax2.set_title("(b) Latency vs GOP cap")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("max GOP cap")
    ax2.set_ylim(0, max(r["online_mean_s"] for r in rows) * 1.22)
    ax2.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.2)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    out = REPO / "results/pdf/gop_accuracy_latency_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote: {csv}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
