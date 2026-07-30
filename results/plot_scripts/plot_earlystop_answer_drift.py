#!/usr/bin/env python3
"""Why SA early-stop beats the full-budget default on VideoMME (OSDI, PDF only).

For the questions where early-stop(S=2) and the full-8-round default disagree,
show the per-round top1 answer and whether it is correct. Both methods read the
SAME trajectory; default reads round 8, early-stop reads its stop round. The
picture shows the correct answer holding a stable middle plateau and then being
flipped out by a near-tie wrong answer in the last rounds -- which the default
snapshot at round 8 catches and early-stop avoids.

Panel (a): answer-correctness grid (question x round) with stop round & the
           round-8 default readout marked.
Panel (b): top1 vs 2nd-best score margin per round for the same questions,
           evidencing that the flips are near-ties (margin ~ 0.01).

Data: results/sa_early_stop/trajectories/traj_<vid>_<qid>.json (VideoMME 601-620).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "results/sa_early_stop/trajectories"
OUT = REPO / "results/pdf/earlystop_answer_drift.pdf"
S = 2
ROUNDS = 8

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})
C_OK = "#3b6ea5"    # correct
C_BAD = "#d98b3a"   # wrong


def stop_round(rounds, s):
    for r in rounds:
        if r["stable_count"] >= s:
            return r["round"]
    return rounds[-1]["round"]


def main():
    trajs = {}
    for fn in os.listdir(TRAJ):
        if fn.startswith("traj_") and fn.endswith(".json"):
            t = json.load(open(TRAJ / fn))
            trajs[(t["video_id"], t["question_id"])] = t

    # keep questions where es(S=2) and default disagree
    disagree = []
    for key, t in trajs.items():
        rounds = t["meta"]["rounds"]
        gt = t["gt"]
        base = rounds[-1]["top1_answer"]
        es = rounds[stop_round(rounds, S) - 1]["top1_answer"]
        if base != es:
            disagree.append((key, t, gt, base, es))
    # rescues first (es correct), then breaks
    disagree.sort(key=lambda x: (x[4] != x[2], x[0]))

    fig, (axg, axm) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    labels = []
    for row, (key, t, gt, base, es) in enumerate(disagree):
        rounds = t["meta"]["rounds"]
        sr = stop_round(rounds, S)
        labels.append(f"{key[0]}:{key[1]}  gt={gt}")
        for r in rounds:
            ok = r["top1_answer"] == gt
            axg.add_patch(Rectangle((r["round"] - 0.5, row - 0.5), 1, 1,
                                    facecolor=C_OK if ok else "white",
                                    edgecolor=C_BAD if not ok else C_OK,
                                    linewidth=0.8))
            axg.text(r["round"], row, r["top1_answer"], ha="center", va="center",
                     fontsize=8, color="white" if ok else C_BAD, fontweight="bold")
        # stop-round marker (early-stop readout) and default readout (round 8)
        axg.scatter([sr], [row], marker="v", s=55, color="black", zorder=5,
                    clip_on=False)
        axg.add_patch(Rectangle((ROUNDS - 0.5, row - 0.5), 1, 1, fill=False,
                                edgecolor="black", linewidth=2.0, zorder=4))

        margins = [r["top1_score"] - r["best_different_score"] for r in rounds]
        col = C_OK if es == gt else C_BAD
        axm.plot(range(1, ROUNDS + 1), margins, marker="o", ms=3, lw=1.3,
                 color=col, alpha=0.85)
        axm.scatter([sr], [margins[sr - 1]], marker="v", s=45, color="black", zorder=5)

    axg.set_xlim(0.5, ROUNDS + 0.5)
    axg.set_ylim(-0.5, len(disagree) - 0.5)
    axg.set_yticks(range(len(disagree)))
    axg.set_yticklabels(labels)
    axg.set_xticks(range(1, ROUNDS + 1))
    axg.set_xlabel("SA self-consistency round")
    axg.set_title("(a) Per-round top1 answer")
    axg.invert_yaxis()
    axg.tick_params(length=0)
    for sp in axg.spines.values():
        sp.set_visible(False)

    axm.axhline(0, color="#888", ls=":", lw=0.8)
    axm.set_xlim(0.8, ROUNDS + 0.2)
    axm.set_xticks(range(1, ROUNDS + 1))
    axm.set_xlabel("SA self-consistency round")
    axm.set_ylabel("top1 − 2nd-best score")
    axm.set_title("(b) Winning margin (near-tie)")
    axm.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    axm.set_axisbelow(True)
    axm.spines["top"].set_visible(False)
    axm.spines["right"].set_visible(False)
    axm.tick_params(length=3, width=0.8)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", ls="", mfc=C_OK, mec=C_OK, ms=9, label="top1 correct"),
        Line2D([0], [0], marker="s", ls="", mfc="white", mec=C_BAD, ms=9, label="top1 wrong"),
        Line2D([0], [0], marker="v", ls="", mfc="black", mec="black", ms=8, label=f"early-stop readout (S={S})"),
        Line2D([0], [0], marker="s", ls="", mfc="white", mec="black", mew=2, ms=9, label="default readout (round 8)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)

    fig.tight_layout()
    out = os.environ.get("KF_OUT", str(OUT))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    rescues = sum(1 for _, _, gt, base, es in disagree if es == gt)
    print(f"disagreements {len(disagree)}: rescues {rescues}, breaks {len(disagree)-rescues}")


if __name__ == "__main__":
    main()
