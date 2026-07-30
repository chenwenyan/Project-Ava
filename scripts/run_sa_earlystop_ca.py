#!/usr/bin/env python3
"""Run the CA stage for the SA early-stop configs so the savings figure can
report FINAL (CA) accuracy instead of intermediate SA accuracy.

Both the full-budget baseline and each early-stop S read the SAME per-question
SA trajectory (results/sa_early_stop/trajectories/traj_<vid>_<qid>.json). The CA
stage picks <=2 candidate SA nodes = the (top1, best_different) of that round's
signature and re-answers from their frames (qwenvl, the pipeline's CA model).
Because the early-stop trajectory was resampled progressively, its SA ranking
differs from the original pipeline run, so cached CA answers cannot be reused --
we must run CA on the trajectory-derived candidates for BOTH baseline (round 8)
and each early-stop stop round.

Cost is minimized by running CA once per DISTINCT signature per question
(baseline round-8 signature is often == a stop-round signature). Resumable:
per-question cache results/sa_early_stop/ca_reruns/ca_<vid>_<qid>.json maps a
signature key -> CA answer letter.

Run (ava env; shares GPU 0 with other jobs, qwenvl needs ~31 GB):
  PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
    <ava py> scripts/run_sa_earlystop_ca.py
Then re-run with --analyze-only to (re)write the accuracy CSV without a GPU.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Code is import-compatible in any checkout, but the caches/trajectories/outputs
# are untracked and live only in the MAIN checkout -- read + write there.
MAIN = Path("/export/home/yulin/wenyan/github/Project-Ava")
sys.path.insert(0, str(MAIN))
os.chdir(MAIN)

TRAJ_DIR = MAIN / "results/sa_early_stop/trajectories"
RERUN_DIR = MAIN / "results/sa_early_stop/ca_reruns"
CACHE = MAIN / "AVA_cache/VideoMME"
SA_RECORDS = MAIN / "outputs/videomme_long_601_620/records_sa.json"
OUT_CSV = MAIN / "results/csv/sa_early_stop_ca_accuracy.csv"
S_LIST = [1, 2, 3, 4, 5, 6, 7]


def stop_round_record(rounds, S):
    for r in rounds:
        if r["stable_count"] >= S:
            return r
    return rounds[-1]


def sig_key(r):
    return json.dumps([r["top1_path"], r["top1_answer"],
                       r.get("best_different_path"), r.get("best_different_answer")])


def letter(x):
    m = re.search(r"[A-D]", str(x).upper())
    return m.group(0) if m else ""


# ---- faithful copy of AVA.operate.calculate_ca_score but with a shared scorer ----
def ca_score(node_results, scorer):
    for node_result in node_results:
        responses = node_result["responses"]
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        thoughts_counts = {"A": [], "B": [], "C": [], "D": []}
        node_score = {c: [0.0, 0.0] for c in "ABCD"}

        def extract_json(s):
            m = re.search(r"{.*}", s, re.DOTALL)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        for response in responses:
            j = extract_json(response)
            if isinstance(j, dict) and "Answer" in j and "Analysis" in j:
                a = j["Answer"]
                if a in answer_counts:
                    answer_counts[a] += 1
                    thoughts_counts[a].append(j["Analysis"])
        scn = len(responses)
        for c in answer_counts:
            node_score[c][0] = answer_counts[c] / scn if scn else 0.0
        for c in answer_counts:
            if len(thoughts_counts[c]) > 1:
                f1_scores = []
                for i in range(len(thoughts_counts[c])):
                    t1, t2 = [], []
                    for j2 in range(len(thoughts_counts[c])):
                        if i != j2:
                            t1.append(thoughts_counts[c][i])
                            t2.append(thoughts_counts[c][j2])
                    f1, _, _ = scorer.score(t1, t2)
                    f1_scores.append(f1.mean().item())
                node_score[c][1] = sum(f1_scores) / len(f1_scores)
        node_result["scores"] = node_score
    return node_results


def build_nodes(rec, tree_by_path):
    """Reconstruct the <=2 CA candidate nodes from a round's signature."""
    from_specs = [(rec["top1_path"], rec["top1_answer"], rec["top1_score"])]
    if rec.get("best_different_path"):
        from_specs.append((rec["best_different_path"], rec["best_different_answer"],
                           rec["best_different_score"]))
    nodes = []
    for path, ans, score in from_specs:
        tn = tree_by_path.get(tuple(path))
        if tn is None:
            return None  # path not in tree (shouldn't happen)
        nodes.append({"action": "SA", "depth": tn["depth"], "path": path,
                      "frame_durations": tn["frame_durations"],
                      "final_score": {ans: score}})
    return nodes


def run_generation(args):
    from llms.init_model import init_model
    from AVA.operate import generate_ca_self_consistency_result, choose_best_ca_answer
    from bert_score import BERTScorer
    from video_utils import VideoRepresentation

    sa_recs = json.loads(SA_RECORDS.read_text())
    trajs = [json.loads((TRAJ_DIR / f).read_text())
             for f in sorted(os.listdir(TRAJ_DIR)) if f.startswith("traj_")]

    # figure out pending (question, signature) work
    pending = 0
    for t in trajs:
        rounds = t["meta"]["rounds"]
        sigs = {sig_key(rounds[-1])} | {sig_key(stop_round_record(rounds, S)) for S in S_LIST}
        cache = RERUN_DIR / f"ca_{t['video_id']}_{t['question_id']}.json"
        done = json.loads(cache.read_text()) if cache.exists() else {}
        pending += sum(1 for s in sigs if s not in done)
    print(f"pending CA inferences: {pending}", flush=True)
    if pending == 0:
        print("all CA reruns cached; nothing to generate", flush=True)
        return

    print("loading qwenvl (Qwen2.5-VL-7B) + BERTScorer ...", flush=True)
    llm = init_model("qwenvl", args.gpus)
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    videos = {}

    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    for t in trajs:
        vid, qid = t["video_id"], t["question_id"]
        rounds = t["meta"]["rounds"]
        query = sa_recs.get(f"{vid}:{qid}", {}).get("question", "")
        tree = json.loads((CACHE / str(vid) / "questions" / str(qid)
                           / "tree_information.json").read_text())
        tree_by_path = {tuple(n["path"]): n for n in tree}

        if vid not in videos:
            cfg = json.loads((CACHE / str(vid) / "config.json").read_text())
            videos[vid] = VideoRepresentation(cfg["source_path"], str(CACHE / str(vid)))
        video = videos[vid]

        cache = RERUN_DIR / f"ca_{vid}_{qid}.json"
        done = json.loads(cache.read_text()) if cache.exists() else {}

        wanted = {sig_key(rounds[-1]): rounds[-1]}
        for S in S_LIST:
            r = stop_round_record(rounds, S)
            wanted[sig_key(r)] = r
        for sk, rec in wanted.items():
            if sk in done:
                continue
            nodes = build_nodes(rec, tree_by_path)
            if nodes is None:
                done[sk] = ""
            else:
                ca_res = generate_ca_self_consistency_result(
                    query=query, sorted_sa_nodes=nodes, llm=llm, video=video,
                    self_consistency_num=4, max_frames=256, max_retries=3)
                scored = ca_score(ca_res, scorer)
                best = choose_best_ca_answer(scored)
                done[sk] = letter(list(best[0]["final_score"].keys())[0])
            cache.write_text(json.dumps(done, indent=2))
            print(f"[ca] {vid}:{qid} sig#{list(wanted).index(sk)} -> {done[sk]}", flush=True)


def analyze():
    import csv
    trajs = [json.loads((TRAJ_DIR / f).read_text())
             for f in sorted(os.listdir(TRAJ_DIR)) if f.startswith("traj_")]
    configs = ["baseline"] + [f"S{S}" for S in S_LIST]
    correct = {c: 0 for c in configs}
    n = 0
    rows = []
    for t in trajs:
        vid, qid, gt = t["video_id"], t["question_id"], t["gt"]
        rounds = t["meta"]["rounds"]
        cache = RERUN_DIR / f"ca_{vid}_{qid}.json"
        if not cache.exists():
            continue
        done = json.loads(cache.read_text())
        ans = {"baseline": done.get(sig_key(rounds[-1]), "")}
        for S in S_LIST:
            ans[f"S{S}"] = done.get(sig_key(stop_round_record(rounds, S)), "")
        if any(ans[c] == "" for c in configs):
            continue  # incomplete
        n += 1
        for c in configs:
            correct[c] += int(ans[c] == gt)
        rows.append({"video_id": vid, "question_id": qid, "gt": gt, **ans})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "ca_accuracy_pct", "correct", "n"])
        for c in configs:
            acc = 100 * correct[c] / n if n else 0.0
            w.writerow([c, round(acc, 1), correct[c], n])
            print(f"{c:9s} CA acc = {correct[c]}/{n} = {acc:.1f}%")
    perq = OUT_CSV.with_name("sa_early_stop_ca_per_question.csv")
    with open(perq, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "question_id", "gt"] + configs)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}  ({n} complete questions)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    if not args.analyze_only:
        run_generation(args)
    analyze()


if __name__ == "__main__":
    main()
