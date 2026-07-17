#!/usr/bin/env python3
"""Run AVA on VideoMME-Long videos 601-620 with resumable, timed results.

Stages (model per paper §6): graph=qwenvl (EKG), sa=qwenlm (agentic search),
ca=qwenvl (frame recheck). Each stage skips work already on disk, so reruns resume.
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # AVA and dataset adapters use repo-relative paths

BASE_ID = 601  # VideoMME long split spans 601-900
MODELS = {"graph": "qwenvl", "sa": "qwenlm", "ca": "qwenvl"}
HEADERS = {
    "graph": ["video_id", "seconds", "nodes", "status"],
    "sa": ["video_id", "question_id", "seconds", "answer", "gt", "status",
           "executed_rounds", "total_responses", "full_budget_responses",
           "early_stopped", "saving_ratio"],
    "ca": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
}


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def persist(output_dir, stage, records):
    (output_dir / f"records_{stage}.json").write_text(json.dumps(records, indent=2))
    with (output_dir / f"timing_{stage}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS[stage], extrasaction="ignore")
        writer.writeheader()
        for key in sorted(records, key=lambda k: [int(p) for p in k.split(":")]):
            writer.writerow(records[key])


def graph_nodes(video_id):
    path = REPO / f"AVA_cache/VideoMME/{video_id}/kg/graph_event_knowledge_graph.graphml"
    return path.read_text(errors="ignore").count("<node ") if path.is_file() else 0


def answer_letter(value):
    match = re.search(r"\b([A-D])\b", str(value).upper())
    return match.group(1) if match else ""


def preflight(video_ids):
    rows = load_json(REPO / "datas/VideoMME/VideoMME.json", [])
    failures = []
    for video_id in video_ids:
        qas = rows[(video_id - 1) * 3 : video_id * 3]
        name = qas[0]["videoID"] if len(qas) == 3 else None
        ok = (
            name is not None
            and len({row["videoID"] for row in qas}) == 1
            and (REPO / f"datas/VideoMME/videos/{name}.mp4").is_file()
            and (REPO / f"AVA_cache/VideoMME/{video_id}/config.json").is_file()
        )
        print(f"preflight {video_id}: {'ok' if ok else 'FAILED'}")
        if not ok:
            failures.append(video_id)
    if failures:
        raise SystemExit(f"preflight failed for: {failures}")


def run_graph(dataset, model, video_ids, output_dir):
    from AVA.ava import AVA

    records = load_json(output_dir / "records_graph.json", {})
    for video_id in video_ids:
        if graph_nodes(video_id):
            status = "skipped"
            seconds = 0.0
        else:
            started = time.time()
            status = "ok"
            try:
                AVA(video=dataset.get_video(video_id), llm_model=model).construct()
            except Exception as error:
                status = f"error:{type(error).__name__}:{error}"
            seconds = round(time.time() - started, 2)
            if status == "ok" and not graph_nodes(video_id):
                status = "empty_graph"
        nodes = graph_nodes(video_id)
        records[str(video_id)] = {
            "video_id": video_id, "seconds": seconds, "nodes": nodes, "status": status,
        }
        persist(output_dir, "graph", records)
        print(f"[graph] {video_id}: {status} ({nodes} nodes)", flush=True)


def sa_early_stop_kwargs(args):
    return dict(
        sa_progressive=True,
        sa_early_stop=not args.disable_sa_early_stop,
        sa_max_rounds=args.sa_max_rounds,
        sa_stability_rounds=args.sa_stability_rounds,
    )


def run_questions(dataset, model, video_ids, output_dir, stage, args):
    from AVA.ava import AVA

    ava_kwargs = sa_early_stop_kwargs(args) if stage == "sa" else {}
    records = load_json(output_dir / f"records_{stage}.json", {})
    for video_id in video_ids:
        if not graph_nodes(video_id):
            print(f"[{stage}] {video_id}: skipped; graph missing", flush=True)
            continue
        video = dataset.get_video(video_id)
        for question_id, qa in enumerate(dataset.get_video_info(video_id)["qa"]):
            key = f"{video_id}:{question_id}"
            if records.get(key, {}).get("status") == "ok":
                print(f"[{stage}] {key}: skipped", flush=True)
                continue
            started = time.time()
            answer, status = "", "ok"
            try:
                ava = AVA(video=video, llm_model=model, **ava_kwargs)
                if stage == "sa":
                    ava.query_tree_search(qa["question"], question_id)
                    raw = ava.generate_SA_answer(qa["question"], question_id)
                else:
                    raw = ava.generate_CA_answer(qa["question"], question_id)
                answer = answer_letter(raw)
                if not answer:
                    status = f"invalid_answer:{raw}"
            except Exception as error:
                status = f"error:{type(error).__name__}:{error}"
            record = {
                "video_id": video_id, "question_id": question_id,
                "seconds": round(time.time() - started, 2), "answer": answer,
                "gt": answer_letter(qa.get("answer", "")), "status": status,
                "question": qa["question"],
            }
            if stage == "sa":
                record.update(read_sa_early_stop_meta(video, question_id))
            records[key] = record
            persist(output_dir, stage, records)
            print(f"[{stage}] {key}: {status} answer={answer}", flush=True)


def read_sa_early_stop_meta(video, question_id):
    """Pull the compact early-stop summary written by AVA.generate_SA_answer."""
    meta_path = Path(video.work_dir) / "questions" / str(question_id) / "SA_early_stop_meta.json"
    if not meta_path.is_file():
        return {}
    meta = load_json(meta_path, {})
    return {
        "executed_rounds": meta.get("executed_rounds"),
        "total_responses": meta.get("total_responses"),
        "full_budget_responses": meta.get("full_budget_responses"),
        "early_stopped": meta.get("early_stopped"),
        "saving_ratio": meta.get("saving_ratio"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["graph", "sa", "ca", "all"], default="all")
    parser.add_argument("--num", type=int, default=20, help="first N long videos from 601 (max 300)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/videomme_long_601_620")
    parser.add_argument("--sa-max-rounds", type=int, default=8,
                        help="max progressive SA self-consistency rounds (1 response/node per round)")
    parser.add_argument("--sa-stability-rounds", type=int, default=4,
                        help="consecutive stable-signature rounds required to stop early")
    parser.add_argument("--disable-sa-early-stop", action="store_true",
                        help="run the full SA budget (still round-by-round) with no early stopping")
    args = parser.parse_args()

    video_ids = list(range(BASE_ID, BASE_ID + args.num))
    preflight(video_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from dataset.init_dataset import init_dataset
    from llms.init_model import init_model

    dataset = init_dataset("videomme")
    stages = ["graph", "sa", "ca"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== {stage}: videos {video_ids[0]}-{video_ids[-1]}, model={MODELS[stage]} ===", flush=True)
        model = init_model(MODELS[stage], args.gpus)
        if stage == "graph":
            run_graph(dataset, model, video_ids, args.output_dir)
        else:
            run_questions(dataset, model, video_ids, args.output_dir, stage, args)
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    main()
