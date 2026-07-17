#!/usr/bin/env python3
"""Measure the AVA *online* (query-answering) stage latency breakdown.

Default AVA settings. For each (video, question) the online path is:

  1. tree_search        retrieval + tree planning over the knowledge graph (qwenlm)
  2. sa_llm             SA self-consistency generation, 8 responses/node    (qwenlm)
  3. sa_score           SA BERTScore scoring + best-answer selection        (deberta)
  4. ca_vlm             CA self-consistency over sampled frames, 4/node     (qwenvl)
  5. ca_score           CA BERTScore scoring + best-answer selection        (deberta)

Requires a pre-built knowledge graph for each video (offline stage done).
The two large models are never resident together: Phase 1 loads qwenlm (+ a
shared BERTScorer) and times steps 1-3 for every question; Phase 2 loads qwenvl
and times steps 4-5. This mirrors the repo's model-per-stage pipeline pattern.

The component functions are called directly (same calls query_SA.py / query_CA.py
make) so no on-disk question cache is read or written -- every number is measured
fresh. A single shared BERTScorer is injected into both scoring steps so the
deberta *load* cost is not counted per question (steady-state scoring latency).

Writes results/online_latency_<tag>.csv (one row per question).
"""

import argparse
import copy
import csv
import gc
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import AVA.operate as operate  # noqa: E402
from AVA.operate import (  # noqa: E402
    tree_search,
    generate_sa_self_consistency_result,
    calculate_sa_score,
    choose_best_sa_answer,
    generate_ca_self_consistency_result,
    calculate_ca_score,
    choose_best_ca_answer,
)

SA_MODEL, CA_MODEL = "qwenlm", "qwenvl"
SA_SELF_CONSISTENCY = 8   # AVA default (generate_sa_self_consistency_result)
CA_SELF_CONSISTENCY = 4   # AVA default (generate_CA_answer)
CA_MAX_FRAMES = 256
GRAPH = "kg/graph_event_knowledge_graph.graphml"
FIELDS = ["video_id", "question_id", "tree_search_s", "sa_llm_s", "sa_score_s",
          "ca_vlm_s", "ca_score_s", "online_total_s", "n_sa_nodes", "n_ca_nodes", "status"]


def has_graph(cache_root, vid):
    return (cache_root / str(vid) / GRAPH).is_file()


def clock(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="videomme")
    parser.add_argument("--video-ids", type=int, nargs="+", default=None,
                        help="explicit video ids (default: first --num from --lo)")
    parser.add_argument("--lo", type=int, default=601)
    parser.add_argument("--num", type=int, default=3, help="number of videos from --lo")
    parser.add_argument("--questions", type=int, default=None,
                        help="cap questions per video (default: all)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cache_root = REPO / f"AVA_cache/{'VideoMME' if args.dataset == 'videomme' else 'LVBench'}"
    video_ids = args.video_ids or list(range(args.lo, args.lo + args.num))
    video_ids = [v for v in video_ids if has_graph(cache_root, v)]
    if not video_ids:
        sys.exit("no videos with a built graph found")
    tag = f"{args.dataset}_{video_ids[0]}_{video_ids[-1]}"
    out_csv = args.out or REPO / f"results/online_latency_{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    from dataset.init_dataset import init_dataset
    from llms.init_model import init_model
    from AVA.ava import AVA
    from bert_score import BERTScorer

    dataset = init_dataset(args.dataset)

    # (vid, qid, question) work list.
    work = []
    for vid in video_ids:
        qas = dataset.get_video_info(vid)["qa"]
        n = len(qas) if args.questions is None else min(args.questions, len(qas))
        for qid in range(n):
            work.append((vid, qid, qas[qid]["question"]))
    print(f"[online-latency] {len(work)} questions over {len(video_ids)} videos "
          f"({video_ids[0]}-{video_ids[-1]})", flush=True)

    # Shared BERTScorer (deberta) reused by both SA and CA scoring so the model
    # LOAD is one-off, not billed per question. Patch operate.BERTScorer so the
    # internal construction inside calculate_ca_score reuses it too.
    print("[online-latency] loading BERTScorer (deberta-xlarge-mnli)...", flush=True)
    shared_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    operate.BERTScorer = lambda *a, **k: shared_scorer

    # One AVA per video (loads the built graph + embedding model once); reused
    # across that video's questions and across both phases.
    records = {}  # (vid,qid) -> row dict
    avas = {}

    def get_ava(vid, model):
        if vid not in avas:
            avas[vid] = AVA(video=dataset.get_video(vid), llm_model=model)
        return avas[vid]

    # ---- Phase 1: qwenlm — tree_search + SA generation + SA scoring ----------
    print(f"\n=== Phase 1 (SA, model={SA_MODEL}): tree_search + SA-LLM + SA-score ===", flush=True)
    sa_llm = init_model(SA_MODEL, args.gpus)
    sa_ranked = {}  # (vid,qid) -> ranked SA nodes (carry frame_durations for CA)
    for vid, qid, question in work:
        key = (vid, qid)
        row = {"video_id": vid, "question_id": qid, "status": "ok",
               "tree_search_s": "", "sa_llm_s": "", "sa_score_s": "",
               "ca_vlm_s": "", "ca_score_s": "", "n_sa_nodes": 0, "n_ca_nodes": 0}
        try:
            ava = get_ava(vid, sa_llm)
            tree_info, t_tree = clock(
                tree_search, question, sa_llm, ava.video,
                ava.events_vdb, ava.entities_vdb, ava.features_vdb)
            if not tree_info:
                row["status"] = "no_sa_nodes"
                records[key] = row
                print(f"[SA] {vid}:{qid} no SA nodes", flush=True)
                continue
            sa_tree, t_sa = clock(
                generate_sa_self_consistency_result, tree_info, sa_llm,
                self_consistency_num=SA_SELF_CONSISTENCY)

            cleaned = [{k: v for k, v in item.items()
                        if k not in {"structed_information", "input_prompt"}}
                       for item in sa_tree]

            def score_and_rank(nodes):
                scored = calculate_sa_score(copy.deepcopy(nodes), scorer=shared_scorer)
                cleaned_score = [{k: v for k, v in it.items() if k != "responses"}
                                 for it in scored]
                return choose_best_sa_answer(cleaned_score)

            ranked, t_score = clock(score_and_rank, cleaned)

            sa_ranked[key] = ranked
            row.update({"tree_search_s": round(t_tree, 3), "sa_llm_s": round(t_sa, 3),
                        "sa_score_s": round(t_score, 3), "n_sa_nodes": len(tree_info)})
            print(f"[SA] {vid}:{qid} tree={t_tree:.2f}s sa_llm={t_sa:.2f}s "
                  f"sa_score={t_score:.2f}s nodes={len(tree_info)}", flush=True)
        except Exception as error:  # noqa: BLE001
            row["status"] = f"error:{type(error).__name__}:{error}"
            print(f"[SA] {vid}:{qid} ERROR {error}", flush=True)
        records[key] = row
        _flush(out_csv, records)

    del sa_llm
    gc.collect()
    _empty_cuda()

    # ---- Phase 2: qwenvl — CA generation + CA scoring ------------------------
    print(f"\n=== Phase 2 (CA, model={CA_MODEL}): CA-VLM + CA-score ===", flush=True)
    ca_llm = init_model(CA_MODEL, args.gpus)
    for vid, qid, question in work:
        key = (vid, qid)
        row = records.get(key)
        if row is None or key not in sa_ranked:
            continue
        try:
            ava = get_ava(vid, ca_llm)
            ca_nodes, t_ca = clock(
                generate_ca_self_consistency_result,
                query=question, sorted_sa_nodes=sa_ranked[key], llm=ca_llm,
                video=ava.video, self_consistency_num=CA_SELF_CONSISTENCY,
                max_frames=CA_MAX_FRAMES, max_retries=3)
            _, t_cascore = clock(
                lambda n: choose_best_ca_answer(calculate_ca_score(n)), ca_nodes)
            row.update({"ca_vlm_s": round(t_ca, 3), "ca_score_s": round(t_cascore, 3),
                        "n_ca_nodes": len(ca_nodes)})
            times = [row[k] for k in ("tree_search_s", "sa_llm_s", "sa_score_s",
                                      "ca_vlm_s", "ca_score_s") if isinstance(row[k], (int, float))]
            row["online_total_s"] = round(sum(times), 3)
            print(f"[CA] {vid}:{qid} ca_vlm={t_ca:.2f}s ca_score={t_cascore:.2f}s "
                  f"total={row['online_total_s']:.2f}s", flush=True)
        except Exception as error:  # noqa: BLE001
            row["status"] = f"error_ca:{type(error).__name__}:{error}"
            print(f"[CA] {vid}:{qid} ERROR {error}", flush=True)
        records[key] = row
        _flush(out_csv, records)

    del ca_llm
    gc.collect()
    _empty_cuda()

    ok = [r for r in records.values() if r["status"] == "ok" and r.get("online_total_s")]
    print(f"\n[online-latency] {len(ok)}/{len(records)} questions fully timed")
    print(f"wrote: {out_csv}")


def _flush(out_csv, records):
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(records, key=lambda k: (k[0], k[1])):
            w.writerow(records[key])


def _empty_cuda():
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


if __name__ == "__main__":
    main()
