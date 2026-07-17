#!/usr/bin/env python3
"""Measure the effect of progressive SA self-consistency early stopping.

Method (cost-efficient, sampling-controlled):
  * For each VideoMME-long question we run the progressive SA loop with
    early_stop DISABLED, so all `--sa-max-rounds` rounds are generated and the
    full per-round signature/score trajectory is recorded (one response/node
    per round -> the standard 104-response budget at 13 nodes x 8 rounds).
  * The early-stopping decision is then DERIVED from that single trajectory:
    the round where the stability signature has been stable for
    `--sa-stability-rounds` consecutive rounds. Because generation is identical
    round-by-round and the stop rule is deterministic, the derived answer is
    exactly what an online early-stop run would have produced -- with no extra
    sampling and no per-config variance. Baseline = top1 at the final round.

Outputs (under --output-dir, default results/sa_early_stop):
  * traj_<vid>_<qid>.json  per-question full trajectory (cache; resumable)
  * ../sa_early_stop_per_question.csv
  * ../sa_early_stop_summary.json / .md   (accuracy + savings, sweep over S)

Run:
  PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
    <ava py> scripts/run_sa_early_stop.py --num 20
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

BASE_ID = 601
FULL_CONSISTENCY = 8  # AVA default self_consistency_num -> 13 nodes x 8 = 104


def gt_letter(rows, video_id, question_id):
    qas = rows[(video_id - 1) * 3 : video_id * 3]
    return qas[question_id]["answer"].strip().upper()[:1]


def derive_earlystop(rounds, stability_rounds, max_rounds, num_nodes):
    """Given the full trajectory, find the round early-stop would have fired."""
    stop_round = None
    for r in rounds:
        if r["stable_count"] >= stability_rounds:
            stop_round = r["round"]
            early_answer = r["top1_answer"]
            break
    early_stopped = stop_round is not None
    if not early_stopped:
        stop_round = rounds[-1]["round"]
        early_answer = rounds[-1]["top1_answer"]
    responses_at_stop = stop_round * num_nodes
    full_budget = max_rounds * num_nodes
    return {
        "early_stopped": early_stopped,
        "stop_round": stop_round,
        "early_answer": early_answer,
        "responses_at_stop": responses_at_stop,
        "full_budget": full_budget,
        "saving_ratio": (full_budget - responses_at_stop) / full_budget if full_budget else 0.0,
    }


def generate_trajectories(args, rows, traj_dir):
    """Run the full-budget progressive loop per question (resumable)."""
    from llms.init_model import init_model
    from AVA.operate import generate_sa_self_consistency_result_progressive
    from bert_score import BERTScorer

    video_ids = list(range(BASE_ID, BASE_ID + args.num))
    pending = []
    for vid in video_ids:
        for qid in range(3):
            ti = REPO / f"AVA_cache/VideoMME/{vid}/questions/{qid}/tree_information.json"
            cache = traj_dir / f"traj_{vid}_{qid}.json"
            if cache.exists():
                print(f"[skip] {vid}:{qid} cached", flush=True)
                continue
            if not ti.is_file():
                print(f"[miss] {vid}:{qid} no tree_information", flush=True)
                continue
            pending.append((vid, qid, ti, cache))

    if not pending:
        print("all trajectories cached; nothing to generate", flush=True)
        return

    print(f"loading qwenlm (32B) + BERTScorer for {len(pending)} questions ...", flush=True)
    llm = init_model("qwenlm", args.gpus)
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")

    for vid, qid, ti, cache in pending:
        tree = json.loads(ti.read_text())
        started = time.time()
        try:
            _, meta = generate_sa_self_consistency_result_progressive(
                tree, llm,
                self_consistency_num=FULL_CONSISTENCY,
                max_rounds=args.sa_max_rounds,
                stability_rounds=args.sa_stability_rounds,
                early_stop=False,  # full trajectory; stop is derived in analysis
                scorer=scorer,
            )
            status = "ok"
        except Exception as error:  # noqa: BLE001
            meta = {"error": f"{type(error).__name__}: {error}"}
            status = "error"
        record = {
            "video_id": vid, "question_id": qid,
            "gt": gt_letter(rows, vid, qid),
            "seconds": round(time.time() - started, 2),
            "status": status,
            "meta": meta,
        }
        cache.write_text(json.dumps(record, indent=2))
        print(f"[{status}] {vid}:{qid} rounds={meta.get('executed_rounds')} "
              f"nodes={meta.get('num_nodes')} {record['seconds']}s", flush=True)


def analyze(args, traj_dir, out_dir):
    trajs = sorted(traj_dir.glob("traj_*.json"))
    rows = [json.loads(p.read_text()) for p in trajs]
    rows = [r for r in rows if r.get("status") == "ok" and r.get("meta", {}).get("rounds")]
    if not rows:
        print("no completed trajectories to analyze yet", flush=True)
        return

    # sweep every usable stability threshold S = 1 .. max_rounds-1 (plus the
    # requested default), so the full accuracy/savings trend is visible.
    sweep = sorted(set(range(1, args.sa_max_rounds)) | {args.sa_stability_rounds})

    per_q_path = out_dir / "sa_early_stop_per_question.csv"
    fields = ["video_id", "question_id", "gt", "num_nodes", "seconds",
              "baseline_answer", "baseline_correct",
              f"es{args.sa_stability_rounds}_answer", f"es{args.sa_stability_rounds}_correct",
              f"es{args.sa_stability_rounds}_stopped", f"es{args.sa_stability_rounds}_stop_round",
              f"es{args.sa_stability_rounds}_saving_ratio"]
    per_q_writer_rows = []

    summary = {"num_questions": len(rows), "full_consistency": FULL_CONSISTENCY,
               "max_rounds": args.sa_max_rounds, "sweep": {}}

    for S in sweep:
        n = len(rows)
        base_correct = es_correct = agree = stopped = 0
        total_saving = 0.0
        stop_rounds = []
        for r in rows:
            meta = r["meta"]
            num_nodes = meta["num_nodes"]
            gt = r["gt"]
            baseline_answer = meta["rounds"][-1]["top1_answer"]
            d = derive_earlystop(meta["rounds"], S, args.sa_max_rounds, num_nodes)
            base_ok = baseline_answer == gt
            es_ok = d["early_answer"] == gt
            base_correct += base_ok
            es_correct += es_ok
            agree += d["early_answer"] == baseline_answer
            stopped += d["early_stopped"]
            total_saving += d["saving_ratio"]
            stop_rounds.append(d["stop_round"])
            if S == args.sa_stability_rounds:
                per_q_writer_rows.append({
                    "video_id": r["video_id"], "question_id": r["question_id"],
                    "gt": gt, "num_nodes": num_nodes, "seconds": r["seconds"],
                    "baseline_answer": baseline_answer, "baseline_correct": int(base_ok),
                    f"es{S}_answer": d["early_answer"], f"es{S}_correct": int(es_ok),
                    f"es{S}_stopped": int(d["early_stopped"]),
                    f"es{S}_stop_round": d["stop_round"],
                    f"es{S}_saving_ratio": round(d["saving_ratio"], 4),
                })
        summary["sweep"][S] = {
            "baseline_acc": round(base_correct / n, 4),
            "earlystop_acc": round(es_correct / n, 4),
            "acc_delta": round((es_correct - base_correct) / n, 4),
            "agreement_with_baseline": round(agree / n, 4),
            "pct_early_stopped": round(stopped / n, 4),
            "mean_stop_round": round(sum(stop_rounds) / n, 3),
            "mean_saving_ratio": round(total_saving / n, 4),
        }

    with per_q_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(per_q_writer_rows, key=lambda x: (x["video_id"], x["question_id"])))

    (out_dir / "sa_early_stop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args, out_dir)
    try:
        make_plot(rows, summary, args, out_dir)
    except Exception as error:  # noqa: BLE001
        print(f"[plot] skipped: {type(error).__name__}: {error}", flush=True)
    print(f"analysis written to {out_dir}", flush=True)
    print(json.dumps(summary["sweep"], indent=2))


def write_markdown(summary, args, out_dir):
    S = args.sa_stability_rounds
    lines = [
        "# SA self-consistency progressive early stopping",
        "",
        f"- Questions analyzed: **{summary['num_questions']}** (VideoMME-long, 601+)",
        f"- Full budget: {summary['max_rounds']} rounds x nodes "
        f"(= {FULL_CONSISTENCY}x13 = 104 responses at 13 SA nodes)",
        "- Baseline = top1 after the full budget; early-stop derived from the same trajectory.",
        "",
        "## Sweep over stability_rounds (S)",
        "",
        "| S | baseline acc | early-stop acc | Δacc | agree w/ baseline | % early-stopped | mean stop round | mean saving |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s, m in summary["sweep"].items():
        star = " (default)" if s == S else ""
        lines.append(
            f"| {s}{star} | {m['baseline_acc']:.1%} | {m['earlystop_acc']:.1%} | "
            f"{m['acc_delta']:+.1%} | {m['agreement_with_baseline']:.1%} | "
            f"{m['pct_early_stopped']:.1%} | {m['mean_stop_round']:.2f} | "
            f"{m['mean_saving_ratio']:.1%} |"
        )
    m = summary["sweep"][S]
    lines += [
        "",
        f"## Headline (S={S}, --sa-stability-rounds {S})",
        "",
        f"- **Mean response savings vs fixed 104: {m['mean_saving_ratio']:.1%}** "
        f"(mean stop at round {m['mean_stop_round']:.2f}/{summary['max_rounds']}).",
        f"- **Accuracy impact: {m['acc_delta']:+.1%}** "
        f"(baseline {m['baseline_acc']:.1%} -> early-stop {m['earlystop_acc']:.1%}).",
        f"- Early-stop fired for {m['pct_early_stopped']:.1%} of questions; "
        f"the derived answer matched the full-budget answer {m['agreement_with_baseline']:.1%} of the time.",
    ]
    (out_dir / "sa_early_stop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(rows, summary, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = args.sa_stability_rounds
    stop_rounds, savings = [], []
    for r in rows:
        d = derive_earlystop(r["meta"]["rounds"], S, args.sa_max_rounds, r["meta"]["num_nodes"])
        stop_rounds.append(d["stop_round"])
        savings.append(d["saving_ratio"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].hist(stop_rounds, bins=range(1, args.sa_max_rounds + 2),
                 align="left", rwidth=0.8, color="#4C72B0")
    axes[0].set_xlabel("early-stop round")
    axes[0].set_ylabel("# questions")
    axes[0].set_title(f"Stop round (S={S})")
    axes[0].set_xticks(range(1, args.sa_max_rounds + 1))

    axes[1].hist(savings, bins=[i / 10 for i in range(0, 11)], color="#55A868")
    axes[1].set_xlabel("response saving ratio vs 104")
    axes[1].set_ylabel("# questions")
    axes[1].set_title(f"Savings (mean {summary['sweep'][S]['mean_saving_ratio']:.0%})")

    ss = sorted(summary["sweep"].keys())
    base = [summary["sweep"][s]["baseline_acc"] for s in ss]
    es = [summary["sweep"][s]["earlystop_acc"] for s in ss]
    sav = [summary["sweep"][s]["mean_saving_ratio"] for s in ss]
    ax = axes[2]
    ax.plot(ss, base, "o--", color="#333333", label="baseline acc")
    ax.plot(ss, es, "o-", color="#C44E52", label="early-stop acc")
    ax.plot(ss, sav, "s-", color="#55A868", label="mean saving")
    ax.set_xlabel("stability_rounds S")
    ax.set_xticks(ss)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs savings tradeoff")
    ax.legend(fontsize=8)

    fig.suptitle("AVA SA self-consistency: progressive early stopping", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "sa_early_stop_savings.pdf")  # PDF only (project convention)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num", type=int, default=20, help="first N long videos from 601")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--sa-max-rounds", type=int, default=8)
    parser.add_argument("--sa-stability-rounds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/sa_early_stop")
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip generation; rebuild analysis from cached trajectories")
    args = parser.parse_args()

    out_dir = args.output_dir
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((REPO / "datas/VideoMME/VideoMME.json").read_text())

    if not args.analyze_only:
        generate_trajectories(args, rows, traj_dir)
    analyze(args, traj_dir, out_dir)


if __name__ == "__main__":
    main()
