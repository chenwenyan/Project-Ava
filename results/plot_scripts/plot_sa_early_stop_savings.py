#!/usr/bin/env python3
"""OSDI/systems-paper-style figure for SA self-consistency progressive early stopping.

Matches the house style of plot_osdi_comparison.py: two compact panels, sans-serif
fonts, colored bars with black edges + hatches (grayscale-safe), value labels,
green relative-change callouts, dotted y-grid, minimal chrome, vector PDF only.

Panels (x-axis = stability threshold S):
  (a) accuracy: full-budget baseline vs early-stop.
  (b) response budget per query: full 104 vs mean responses used by early-stop.
  (c) latency per query (s): full (8-round wall time) vs early-stop.
      Early-stop latency is estimated as full_seconds * stop_round / max_rounds
      (each round = one 13-response batch, so wall time ~ rounds executed).

Reads results/sa_early_stop/sa_early_stop_summary.json + trajectories/traj_*.json.
Writes results/pdf/sa_early_stop_savings.pdf.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
FULL_BUDGET = 104  # 13 SA nodes x 8 rounds

# Grayscale-safe: distinct lightness + hatch so it survives B&W printing
# (same palette as plot_osdi_comparison.py).
BASE_STYLE = dict(color="#d98b3a", hatch="///")  # full budget: hatched orange
ES_STYLE = dict(color="#3b6ea5", hatch="")       # early-stop: solid blue
GAIN = "#127c2a"                                  # good change (acc up / cost down)
LOSS = "#b3202c"                                  # bad change (accuracy regression)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})


def stop_round(rounds, S, max_rounds):
    """Round at which early-stop would fire (stable_count reaches S)."""
    for r in rounds:
        if r["stable_count"] >= S:
            return r["round"]
    return max_rounds


def latency_by_s(traj_dir, ss):
    """Mean per-query wall time: full (8 rounds) and early-stop (~rounds executed)."""
    trajs = [json.loads(Path(p).read_text()) for p in glob.glob(str(traj_dir / "traj_*.json"))]
    trajs = [t for t in trajs if t.get("status") == "ok" and t["meta"].get("rounds")]
    full = [t["seconds"] for t in trajs]
    full_lat = sum(full) / len(full)
    es_lat = {}
    for S in ss:
        es = [t["seconds"] * stop_round(t["meta"]["rounds"], S, t["meta"]["max_rounds"])
              / t["meta"]["max_rounds"] for t in trajs]
        es_lat[S] = sum(es) / len(es)
    return full_lat, es_lat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=REPO / "results/sa_early_stop/sa_early_stop_summary.json")
    parser.add_argument("--traj-dir", type=Path,
                        default=REPO / "results/sa_early_stop/trajectories")
    parser.add_argument("--out", type=Path, default=REPO / "results/pdf/sa_early_stop_savings.pdf")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    sweep = summary["sweep"]
    ss = sorted(int(k) for k in sweep)
    n_q = summary["num_questions"]

    base_acc = [sweep[str(s)]["baseline_acc"] * 100 for s in ss]
    es_acc = [sweep[str(s)]["earlystop_acc"] * 100 for s in ss]
    saving = [sweep[str(s)]["mean_saving_ratio"] for s in ss]
    es_resp = [FULL_BUDGET * (1 - sv) for sv in saving]
    full_resp = [FULL_BUDGET] * len(ss)

    full_lat, es_lat = latency_by_s(args.traj_dir, ss)
    es_latency = [es_lat[s] for s in ss]
    full_latency = [full_lat] * len(ss)

    x = list(range(len(ss)))
    w = 0.38
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 2.75))

    # (a) accuracy: baseline vs early-stop
    ax1.bar([i - w / 2 for i in x], base_acc, w, edgecolor="black", linewidth=0.8,
            label="full (104)", **BASE_STYLE)
    ax1.bar([i + w / 2 for i in x], es_acc, w, edgecolor="black", linewidth=0.8,
            label="early-stop", **ES_STYLE)
    for i in x:
        ax1.text(i - w / 2, base_acc[i] + 1, f"{base_acc[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax1.text(i + w / 2, es_acc[i] + 1, f"{es_acc[i]:.0f}", ha="center", va="bottom", fontsize=8)
        if base_acc[i]:
            rel = (es_acc[i] - base_acc[i]) / base_acc[i] * 100
            ax1.text(i + w / 2, es_acc[i] + 5.5, f"{rel:+.0f}%", ha="center",
                     va="bottom", fontsize=8.5, fontweight="bold",
                     color=GAIN if rel >= 0 else LOSS)
    ax1.set_ylabel("accuracy (%)")
    ax1.set_ylim(0, max(max(base_acc), max(es_acc)) * 1.45)
    ax1.set_title("(a) Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S={s}" for s in ss])
    ax1.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.4, columnspacing=1.0)

    # (b) response budget per query: full 104 vs early-stop mean responses
    ax2.bar([i - w / 2 for i in x], full_resp, w, edgecolor="black", linewidth=0.8,
            label="full (104)", **BASE_STYLE)
    ax2.bar([i + w / 2 for i in x], es_resp, w, edgecolor="black", linewidth=0.8,
            label="early-stop", **ES_STYLE)
    for i in x:
        ax2.text(i + w / 2, es_resp[i] + 1.5, f"{es_resp[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax2.text(i + w / 2, es_resp[i] + FULL_BUDGET * 0.10,
                 f"-{saving[i] * 100:.0f}%", ha="center", va="bottom", fontsize=8.5,
                 color=GAIN, fontweight="bold")
    ax2.text(x[0] - w / 2, FULL_BUDGET + 1.5, "104", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("responses / query")
    ax2.set_ylim(0, FULL_BUDGET * 1.35)
    ax2.set_title("(b) Response budget")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S={s}" for s in ss])
    ax2.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.4, columnspacing=1.0)

    # (c) latency per query: full vs early-stop wall time
    ax3.bar([i - w / 2 for i in x], full_latency, w, edgecolor="black", linewidth=0.8,
            label="full (104)", **BASE_STYLE)
    ax3.bar([i + w / 2 for i in x], es_latency, w, edgecolor="black", linewidth=0.8,
            label="early-stop", **ES_STYLE)
    for i in x:
        ax3.text(i + w / 2, es_latency[i] + full_lat * 0.015, f"{es_latency[i]:.0f}",
                 ha="center", va="bottom", fontsize=8)
        rel = (es_latency[i] - full_lat) / full_lat * 100
        ax3.text(i + w / 2, es_latency[i] + full_lat * 0.11, f"{rel:+.0f}%",
                 ha="center", va="bottom", fontsize=8.5, color=GAIN, fontweight="bold")
    ax3.text(x[0] - w / 2, full_lat + full_lat * 0.015, f"{full_lat:.0f}",
             ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("latency / query (s)")
    ax3.set_ylim(0, full_lat * 1.35)
    ax3.set_title("(c) Online SA latency")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"S={s}" for s in ss])
    ax3.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.4, columnspacing=1.0)

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")  # PDF only
    print(f"S={ss}")
    print(f"baseline acc={[round(a,1) for a in base_acc]} early-stop acc={[round(a,1) for a in es_acc]}")
    print(f"early-stop responses/q={[round(r,1) for r in es_resp]} saving={[round(s*100) for s in saving]}%")
    print(f"full latency={full_lat:.0f}s early-stop latency={[round(l) for l in es_latency]}s")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
