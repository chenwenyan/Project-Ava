#!/usr/bin/env python3
"""KF-Direct GOP sweep on the LVBench ablation subset (results/json/lvbench_selection.json).

This is the LVBench analogue of scripts/run_keyframe_pipeline.py's --src-subdir/--cache-tag
path: run the codec-keyframe = event (no-merge) method, but demux a *re-encoded* video whose
GOP cap was tightened (videos_gop3s / videos_gop5s from scripts/reencode_lvbench_gop.py) so the
I-frame segmentation is denser. Everything else is byte-for-byte the AVA default: the frames fed
to the VLM are the ORIGINAL cached frames (symlinked from AVA_cache/LVBench); only the I-frame
*timestamps* that drive keyframe_extract_events come from the re-encoded file.

Per gop tag N: cache AVA_cache/LVBench_kf_gopNs, output results/lvbench_keyframe_gopNs, source
redirected videos/ -> videos_gopNs/. Resumable (records_*.json skip status=="ok"). Run graph
first (builds the EKG), then sa, then ca; ca records keep the question text for offline token
reconstruction. Same models as every other run: graph/ca=Qwen2.5-VL-7B, sa=Qwen2.5-32B-AWQ.
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

REPO = Path("/export/home/yulin/wenyan/github/Project-Ava")  # operate on the main checkout
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

from video_utils import VideoRepresentation  # noqa: E402
import AVA.operate as operate  # noqa: E402
from run_keyframe_pipeline import keyframe_extract_events  # noqa: E402

MODELS = {"graph": "qwenvl", "sa": "qwenlm", "ca": "qwenvl"}
DEFAULT_CACHE = REPO / "AVA_cache/LVBench"          # AVA-default cache (frames + native config)
SELECTION = REPO / "results/json/lvbench_selection.json"
HEADERS = {
    "graph": ["video_id", "seconds", "nodes", "events", "status"],
    "sa": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
    "ca": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
}
CACHE_ROOT = None   # set in main() per gop tag
SRC_SUBDIR = None   # e.g. "videos_gop3s"


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


def get_video(video_id):
    """KF-Direct video: reuse ORIGINAL frames (symlink) + own kg, but redirect the cached
    config's source_path to the re-encoded (denser-GOP) file so keyframe segmentation
    demuxes that file's I-frame timestamps."""
    src = DEFAULT_CACHE / str(video_id)
    dst = CACHE_ROOT / str(video_id)
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "frames").exists():
        os.symlink(src / "frames", dst / "frames")
    cfg = json.loads((src / "config.json").read_text())
    cfg["source_path"] = cfg["source_path"].replace("/videos/", f"/{SRC_SUBDIR}/")
    if not (dst / "config.json").exists():
        (dst / "config.json").write_text(json.dumps(cfg))
    return VideoRepresentation(cfg["source_path"], str(dst))


def run_graph(model, video_ids, output_dir):
    from AVA.ava import AVA
    records = load_json(output_dir / "records_graph.json", {})
    for video_id in video_ids:
        if graph_nodes(video_id):
            status, seconds = "skipped", 0.0
        else:
            started = time.time()
            status = "ok"
            try:
                AVA(video=get_video(video_id), llm_model=model).construct()
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


def run_questions(dataset, model, video_ids, question_map, output_dir, stage):
    from AVA.ava import AVA
    records = load_json(output_dir / f"records_{stage}.json", {})
    for video_id in video_ids:
        if not graph_nodes(video_id):
            print(f"[{stage}] {video_id}: skipped; graph missing", flush=True)
            continue
        video = get_video(video_id)
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
    global CACHE_ROOT, SRC_SUBDIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gop", type=int, required=True, choices=[3, 5])
    parser.add_argument("--stage", choices=["graph", "sa", "ca", "all"], default="all")
    parser.add_argument("--num", type=int, default=20, help="first N selected LVBench videos")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    tag = f"gop{args.gop}s"
    SRC_SUBDIR = f"videos_{tag}"
    CACHE_ROOT = REPO / f"AVA_cache/LVBench_kf_{tag}"
    output_dir = REPO / f"results/lvbench_keyframe_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    operate.extract_events = keyframe_extract_events  # KF-Direct: keyframe interval = one event

    selection = json.loads(SELECTION.read_text())
    video_ids = selection["video_ids"][:args.num]
    question_map = selection["questions"]

    from dataset.init_dataset import init_dataset
    from llms.init_model import init_model
    dataset = init_dataset("lvbench")

    stages = ["graph", "sa", "ca"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== {stage} (lvbench KF-Direct {tag}): {len(video_ids)} videos, "
              f"model={MODELS[stage]} ===", flush=True)
        model = init_model(MODELS[stage], args.gpus)
        if stage == "graph":
            run_graph(model, video_ids, output_dir)
        else:
            run_questions(dataset, model, video_ids, question_map, output_dir, stage)
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    main()
