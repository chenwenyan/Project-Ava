#!/usr/bin/env python3
"""AVA on VideoMME-Long with codec-keyframe video chunking. Two modes:

  --mode direct : keyframe interval = one event (no BERTScore semantic merge).
  --mode merge  : keyframe intervals replace the fixed 3s base windows, then AVA's
                  BERTScore semantic merge runs unchanged.

Only the segmentation changes; captioning, entity extraction, graph, SA, and CA are
AVA-default. Each mode uses its own cache (VideoMME_kf / VideoMME_kf_merge) and output
dir so all methods stay isolated and directly comparable. Frames are symlinked from
the default cache (not re-extracted).
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

import av

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from video_utils import VideoRepresentation  # noqa: E402
import AVA.operate as operate  # noqa: E402
import AVA.events as events_mod  # noqa: E402
from AVA.events import batch_generate_descriptions, format_events  # noqa: E402

BASE_ID = 601
MODELS = {"graph": "qwenvl", "sa": "qwenlm", "ca": "qwenvl"}
DEFAULT_CACHE = REPO / "AVA_cache/VideoMME"       # AVA-default cache (frames live here)
MODE_CACHE = {"direct": REPO / "AVA_cache/VideoMME_kf",
              "merge": REPO / "AVA_cache/VideoMME_kf_merge"}
MODE_OUTPUT = {"direct": REPO / "outputs/videomme_keyframe_601_605",
               "merge": REPO / "outputs/videomme_keyframe_merge_601_605"}
HEADERS = {
    "graph": ["video_id", "seconds", "nodes", "events", "status"],
    "sa": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
    "ca": ["video_id", "question_id", "seconds", "answer", "gt", "status"],
}
CACHE_ROOT = MODE_CACHE["direct"]  # set in main() per mode


# --- keyframe segmentation -------------------------------------------------

def keyframe_segments(source_path, duration):
    """Return [(start, end), ...] cut at codec keyframe (I-frame) timestamps."""
    container = av.open(source_path)
    stream = container.streams.video[0]
    tb = stream.time_base
    keyframes = []
    for packet in container.demux(stream):
        if packet.is_keyframe and packet.pts is not None:
            keyframes.append(float(packet.pts * tb))
    container.close()

    bounds = sorted({k for k in keyframes if 0.0 <= k < duration})
    if not bounds or bounds[0] > 0.0:
        bounds.insert(0, 0.0)
    bounds.append(float(duration))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def keyframe_extract_events(llm, video, global_config):
    """direct mode: drop-in for AVA.events.extract_events, keyframe interval -> one event."""
    file_path = os.path.join(global_config["working_dir"], "events")
    os.makedirs(file_path, exist_ok=True)
    events_json = os.path.join(file_path, "events.json")
    if os.path.exists(events_json):
        with open(events_json) as file:
            events = json.load(file)
        if events and events[-1]["duration"][-1] == video.config["duration"]:
            return format_events(events)

    segments = keyframe_segments(video.config["source_path"], video.config["duration"])
    print(f"[keyframe] {len(segments)} segments", flush=True)

    descriptions = batch_generate_descriptions(
        llm=llm, video=video, chunk_durations=segments,
        file_path=file_path, batch_size=64, global_config=global_config,
    )
    events = [
        {"duration": [start, end], "description": desc}
        for (start, end), desc in zip(segments, descriptions) if desc
    ]
    with open(events_json, "w") as file:
        json.dump(events, file, indent=4)
    return format_events(events)


def keyframe_get_chunk_timestamp(video, chunk_duration, chunk_overlap):
    """merge mode: replace AVA's fixed base windows with keyframe intervals; the rest
    of extract_events (caption + BERTScore semantic merge) runs unchanged."""
    segments = keyframe_segments(video.config["source_path"], video.config["duration"])
    print(f"[keyframe] {len(segments)} base windows (semantic merge kept)", flush=True)
    return segments


def get_keyframe_video(video_id):
    """Mode's VideoRepresentation: reuses default frames (symlink), own kg."""
    src = DEFAULT_CACHE / str(video_id)
    dst = CACHE_ROOT / str(video_id)
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "frames").exists():
        os.symlink(src / "frames", dst / "frames")
    if not (dst / "config.json").exists():
        shutil.copy(src / "config.json", dst / "config.json")
    source_path = json.loads((src / "config.json").read_text())["source_path"]
    return VideoRepresentation(source_path, str(dst))


# --- resumable runner ------------------------------------------------------

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
                AVA(video=get_keyframe_video(video_id), llm_model=model).construct()
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


def run_questions(dataset, model, video_ids, output_dir, stage):
    from AVA.ava import AVA

    records = load_json(output_dir / f"records_{stage}.json", {})
    for video_id in video_ids:
        if not graph_nodes(video_id):
            print(f"[{stage}] {video_id}: skipped; graph missing", flush=True)
            continue
        video = get_keyframe_video(video_id)
        for question_id, qa in enumerate(dataset.get_video_info(video_id)["qa"]):
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
    parser.add_argument("--mode", choices=["direct", "merge"], default="direct")
    parser.add_argument("--stage", choices=["graph", "sa", "ca", "all"], default="all")
    parser.add_argument("--num", type=int, default=5, help="first N long videos from 601")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    CACHE_ROOT = MODE_CACHE[args.mode]
    output_dir = args.output_dir or MODE_OUTPUT[args.mode]

    # Install keyframe segmentation for the chosen mode.
    if args.mode == "direct":
        operate.extract_events = keyframe_extract_events
    else:  # merge: only the base-window generator changes
        events_mod.get_chunk_timestamp = keyframe_get_chunk_timestamp

    video_ids = list(range(BASE_ID, BASE_ID + args.num))
    output_dir.mkdir(parents=True, exist_ok=True)

    from dataset.init_dataset import init_dataset
    from llms.init_model import init_model

    dataset = init_dataset("videomme")
    stages = ["graph", "sa", "ca"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== {stage} (keyframe-{args.mode}): videos {video_ids[0]}-{video_ids[-1]}, "
              f"model={MODELS[stage]} ===", flush=True)
        model = init_model(MODELS[stage], args.gpus)
        if stage == "graph":
            run_graph(model, video_ids, output_dir)
        else:
            run_questions(dataset, model, video_ids, output_dir, stage)
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    main()
