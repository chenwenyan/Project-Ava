#!/usr/bin/env python3
"""Run the three chunking methods on the LVBench ablation subset (results/lvbench_selection.json).

  --mode default : AVA fixed 3s windows + BERTScore semantic merge (cache AVA_cache/LVBench)
  --mode direct  : codec keyframe interval = one event, no merge   (cache AVA_cache/LVBench_kf)
  --mode merge   : keyframe base windows + semantic merge          (cache AVA_cache/LVBench_kf_merge)

Same models/data as the VideoMME experiment; only segmentation differs across modes. Results
(resumable) go to results/lvbench_<mode>/. Run --mode default FIRST: it extracts the shared frames
into AVA_cache/LVBench, which the keyframe modes symlink.
"""

import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

from video_utils import VideoRepresentation  # noqa: E402
import AVA.operate as operate  # noqa: E402
import AVA.events as events_mod  # noqa: E402
from run_keyframe_pipeline import (  # noqa: E402
    keyframe_extract_events, keyframe_get_chunk_timestamp,
)

MODELS = {"graph": "qwenvl", "sa": "qwenlm", "ca": "qwenvl"}
MODE_CACHE = {"default": "LVBench", "direct": "LVBench_kf", "merge": "LVBench_kf_merge"}
MODE_OUT = {"default": "results/lvbench_default", "direct": "results/lvbench_keyframe_direct",
            "merge": "results/lvbench_keyframe_merge"}
DEFAULT_CACHE = REPO / "AVA_cache/LVBench"
SELECTION = REPO / "results/json/lvbench_selection.json"
HEADERS = {
    "graph": ["video_id", "seconds", "nodes", "events", "status"],
    "sa": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
    "ca": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
}
ORIG_EXTRACT = operate.extract_events
ORIG_CHUNK = events_mod.get_chunk_timestamp
CACHE_ROOT = DEFAULT_CACHE  # set in main() per mode


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
    path = CACHE_ROOT / str(video_id) / "kg/graph_event_knowledge_graph.graphml"
    return path.read_text(errors="ignore").count("<node ") if path.is_file() else 0


def event_count(video_id):
    path = CACHE_ROOT / str(video_id) / "kg/events/events.json"
    return len(json.loads(path.read_text())) if path.is_file() else 0


def answer_letter(value):
    match = re.search(r"\b([A-D])\b", str(value).upper())
    return match.group(1) if match else ""


def get_video(dataset, mode, video_id):
    """default mode -> real LVBench cache (extracts frames); keyframe modes -> symlink frames."""
    if mode == "default":
        return dataset.get_video(video_id)
    src = DEFAULT_CACHE / str(video_id)
    dst = CACHE_ROOT / str(video_id)
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "frames").exists():
        os.symlink(src / "frames", dst / "frames")
    if not (dst / "config.json").exists():
        shutil.copy(src / "config.json", dst / "config.json")
    source_path = json.loads((src / "config.json").read_text())["source_path"]
    return VideoRepresentation(source_path, str(dst))


def run_graph(dataset, model, mode, video_ids, output_dir):
    from AVA.ava import AVA
    records = load_json(output_dir / "records_graph.json", {})
    for video_id in video_ids:
        if graph_nodes(video_id):
            status, seconds = "skipped", 0.0
        else:
            started = time.time()
            status = "ok"
            try:
                AVA(video=get_video(dataset, mode, video_id), llm_model=model).construct()
            except Exception as error:
                status = f"error:{type(error).__name__}:{error}"
            seconds = round(time.time() - started, 2)
            if status == "ok" and not graph_nodes(video_id):
                status = "empty_graph"
        records[str(video_id)] = {
            "video_id": video_id, "seconds": seconds,
            "nodes": graph_nodes(video_id), "events": event_count(video_id), "status": status,
        }
        persist(output_dir, "graph", records)
        print(f"[graph] {video_id}: {status} ({graph_nodes(video_id)} nodes)", flush=True)


def run_questions(dataset, model, mode, video_ids, question_map, output_dir, stage):
    from AVA.ava import AVA
    records = load_json(output_dir / f"records_{stage}.json", {})
    for video_id in video_ids:
        if not graph_nodes(video_id):
            print(f"[{stage}] {video_id}: skipped; graph missing", flush=True)
            continue
        video = get_video(dataset, mode, video_id)
        all_qa = dataset.get_video_info(video_id)["qa"]
        for question_id in question_map[str(video_id)]:
            qa = all_qa[question_id]
            key = f"{video_id}:{question_id}"
            if records.get(key, {}).get("status") == "ok":
                print(f"[{stage}] {key}: skipped", flush=True)
                continue
            started = time.time()
            answer, status = "", "ok"
            try:
                ava = AVA(video=video, llm_model=model)
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
            records[key] = {
                "video_id": video_id, "question_id": question_id,
                "seconds": round(time.time() - started, 2), "answer": answer,
                "gt": answer_letter(qa.get("answer", "")), "status": status,
                "question": qa["question"],
            }
            persist(output_dir, stage, records)
            print(f"[{stage}] {key}: {status} answer={answer}", flush=True)


def main():
    global CACHE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["default", "direct", "merge"], required=True)
    parser.add_argument("--stage", choices=["graph", "sa", "ca", "all"], default="all")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    CACHE_ROOT = REPO / "AVA_cache" / MODE_CACHE[args.mode]
    output_dir = REPO / MODE_OUT[args.mode]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "direct":
        operate.extract_events = keyframe_extract_events
    elif args.mode == "merge":
        events_mod.get_chunk_timestamp = keyframe_get_chunk_timestamp
    # default: originals unchanged

    selection = json.loads(SELECTION.read_text())
    video_ids = selection["video_ids"]
    question_map = selection["questions"]

    from dataset.init_dataset import init_dataset
    from llms.init_model import init_model
    dataset = init_dataset("lvbench")

    stages = ["graph", "sa", "ca"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== {stage} (lvbench-{args.mode}): {len(video_ids)} videos, model={MODELS[stage]} ===", flush=True)
        model = init_model(MODELS[stage], args.gpus)
        if stage == "graph":
            run_graph(dataset, model, args.mode, video_ids, output_dir)
        else:
            run_questions(dataset, model, args.mode, video_ids, question_map, output_dir, stage)
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    main()
