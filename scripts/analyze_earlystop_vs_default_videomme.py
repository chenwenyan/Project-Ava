#!/usr/bin/env python
"""Explain WHY SA early-stop beats the full-budget default on VideoMME.

Both answers come from the SAME per-question trajectory (results/sa_early_stop/
trajectories/traj_<vid>_<qid>.json, VideoMME-long 601-620, generated with
early_stop disabled so all 8 rounds exist). Only the *round whose top1 answer we
read* differs:
  * default / baseline = top1 answer after the full 8 rounds (round[-1])
  * early-stop(S)       = top1 answer at the round where the stability signature
                          has held for S consecutive rounds (derive_earlystop)

So any accuracy gap is purely because the top1 answer is NON-MONOTONIC across
rounds: accumulating more self-consistency responses re-ranks the BERTScore
aggregate and sometimes flips the top1 from correct (early, stable) to wrong
(late). This script quantifies the flips and dumps the round-by-round answer
trace + score margins for the questions early-stop rescues.
"""
import csv
import json
import os
from collections import Counter

MAIN = "/export/home/yulin/wenyan/github/Project-Ava"
TRAJ = f"{MAIN}/results/sa_early_stop/trajectories"
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "csv", "earlystop_vs_default_videomme.csv")
S_LIST = [2, 4]  # 2 = sweet spot, 4 = shipped default stability


def derive_stop_round(rounds, S):
    for r in rounds:
        if r["stable_count"] >= S:
            return r["round"], r["top1_answer"]
    return rounds[-1]["round"], rounds[-1]["top1_answer"]


def main():
    trajs = []
    for fn in sorted(os.listdir(TRAJ)):
        if fn.startswith("traj_") and fn.endswith(".json"):
            trajs.append(json.load(open(os.path.join(TRAJ, fn))))
    print(f"loaded {len(trajs)} trajectories (VideoMME 601-620)\n")

    # per-question CSV: baseline vs early-stop(S=2,4) answers + full answer trace
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "question_id", "gt", "baseline_answer",
                    "baseline_correct", "es2_answer", "es2_correct", "es2_stop_round",
                    "es4_answer", "es4_correct", "es4_stop_round", "answer_trace"])
        for t in sorted(trajs, key=lambda x: (x["video_id"], x["question_id"])):
            gt = t["gt"]; rounds = t["meta"]["rounds"]
            base = rounds[-1]["top1_answer"]
            r2, a2 = derive_stop_round(rounds, 2)
            r4, a4 = derive_stop_round(rounds, 4)
            w.writerow([t["video_id"], t["question_id"], gt, base, int(base == gt),
                        a2, int(a2 == gt), r2, a4, int(a4 == gt), r4,
                        "".join(r["top1_answer"] for r in rounds)])
    print(f"wrote {OUT_CSV}\n")

    for S in S_LIST:
        base_ok = es_ok = 0
        gains, losses = [], []          # gain: base wrong, es right
        for t in trajs:
            gt = t["gt"]
            rounds = t["meta"]["rounds"]
            base_ans = rounds[-1]["top1_answer"]
            stop_round, es_ans = derive_stop_round(rounds, S)
            b, e = (base_ans == gt), (es_ans == gt)
            base_ok += b
            es_ok += e
            rec = dict(vid=t["video_id"], qid=t["question_id"], gt=gt,
                       base=base_ans, es=es_ans, stop=stop_round,
                       answers=[r["top1_answer"] for r in rounds])
            if e and not b:
                gains.append(rec)
            elif b and not e:
                losses.append(rec)
        n = len(trajs)
        print(f"=== S={S} ===")
        print(f"default(full 8r) acc = {base_ok}/{n} = {base_ok/n*100:.1f}%")
        print(f"early-stop(S={S}) acc = {es_ok}/{n} = {es_ok/n*100:.1f}%   "
              f"(Δ = {(es_ok-base_ok)/n*100:+.1f} pt, net {es_ok-base_ok} Q)")
        print(f"early-stop RESCUES (default✗ → es✓): {len(gains)}")
        print(f"early-stop BREAKS  (default✓ → es✗): {len(losses)}")

        # round-by-round answer flip pattern on the whole set
        flip_any = sum(1 for t in trajs
                       if len({r["top1_answer"] for r in t["meta"]["rounds"]}) > 1)
        settle_late = sum(1 for t in trajs
                          if t["meta"]["rounds"][-1]["top1_answer"]
                          != t["meta"]["rounds"][-2]["top1_answer"])
        print(f"questions whose top1 changes at least once over 8 rounds: {flip_any}/{n}")
        print(f"questions whose answer still changes at the LAST round (7→8): {settle_late}")

        if S == 2:
            print("\n--- rescued questions (early-stop right, default wrong) ---")
            for g in gains:
                trace = "".join(g["answers"])
                print(f"  {g['vid']}:{g['qid']} gt={g['gt']} stop@r{g['stop']} "
                      f"es={g['es']} base={g['base']}  rounds[{trace}]")
            print("\n--- broken questions (early-stop wrong, default right) ---")
            for l in losses:
                trace = "".join(l["answers"])
                print(f"  {l['vid']}:{l['qid']} gt={l['gt']} stop@r{l['stop']} "
                      f"es={l['es']} base={l['base']}  rounds[{trace}]")
        print()


if __name__ == "__main__":
    main()
