#!/usr/bin/env python3
"""VideoMME/LVBench with swapped models: SA=Qwen3.5-27B, CA=Qwen3-VL-8B-Instruct.

Three chunking methods, GRAPHS REUSED from the existing (old-model) caches --
only the SA (agentic tree search + self-consistency) and CA (frame recheck)
stages are re-run with the new models, so the comparison isolates the QA models.

  dataset   method    graph reused from              new isolated cache
  videomme  default   AVA_cache/VideoMME             AVA_cache/VideoMME_q35_default
  videomme  kf3s      AVA_cache/VideoMME_kf_gop3s    AVA_cache/VideoMME_q35_kf3s
  videomme  kfmerge   AVA_cache/VideoMME_kf_merge    AVA_cache/VideoMME_q35_kfmerge
  lvbench   default   AVA_cache/LVBench              AVA_cache/LVBench_q35_default
  lvbench   kf3s      AVA_cache/LVBench_kf_gop3s     AVA_cache/LVBench_q35_kf3s
  lvbench   kfmerge   AVA_cache/LVBench_kf_merge     AVA_cache/LVBench_q35_kfmerge

videomme: videos 601..600+num, 3 questions each (num default 20).
lvbench: first `num` videos of results/json/lvbench_selection.json with that file's
question ids (num default 10 = the subset where all three methods have graphs;
gop3s graphs were only built for the first 10 selected videos).

Isolation: per video the new cache symlinks frames/ + kg/ from the source cache
and starts an EMPTY questions/ dir, so tree search + SA + CA all re-run with the
new SA model and never touch the old-model results. SA runs the ORIGINAL
non-progressive full budget (13 nodes x 8 responses), same as the old baselines.
CA keeps max_frames=256 (fits Qwen3-VL's 256K context, unlike Qwen2.5-VL's 32K).

Models are served by vLLM (OpenAI API) from the `qwen35` env -- start the server
first (see scripts/run_qwen35_all.sh); this driver runs in the `ava` env and only
needs GPU for JinaCLIP/BERTScore.

Usage:
  python scripts/run_qwen35_experiment.py --dataset lvbench --method default --stage sa
Resumable per question (records_<stage>.json, status=="ok" skipped).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# caches/outputs are untracked and live only in the MAIN checkout
MAIN = Path("/export/home/yulin/wenyan/github/Project-Ava")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(MAIN)

from llm_api_client import OpenAIChatLM, OpenAIChatVL, wait_for_server  # noqa: E402

SRC_CACHE = {
    "videomme": {
        "default": MAIN / "AVA_cache/VideoMME",
        "kf3s": MAIN / "AVA_cache/VideoMME_kf_gop3s",
        "kfmerge": MAIN / "AVA_cache/VideoMME_kf_merge",
    },
    "lvbench": {
        "default": MAIN / "AVA_cache/LVBench",
        "kf3s": MAIN / "AVA_cache/LVBench_kf_gop3s",
        "kfmerge": MAIN / "AVA_cache/LVBench_kf_merge",
    },
}
NEW_CACHE_PREFIX = {"videomme": "VideoMME_q35", "lvbench": "LVBench_q35"}
DEFAULT_NUM = {"videomme": 20, "lvbench": 10}
SELECTION = MAIN / "results/json/lvbench_selection.json"
STAGE_MODEL = {
    "sa": "Qwen/Qwen3.5-27B-FP8",
    "ca": "Qwen/Qwen3-VL-8B-Instruct-FP8",
}
HEADERS = ["video_id", "question_id", "seconds", "answer", "gt", "status"]


def question_plan(dataset_name, num):
    """[(video_id, [question_ids])] for the run scope."""
    if dataset_name == "videomme":
        return [(vid, [0, 1, 2]) for vid in range(601, 601 + num)]
    sel = json.loads(SELECTION.read_text())
    return [(vid, sel["questions"][str(vid)]) for vid in sel["video_ids"][:num]]


def output_dir(dataset_name, method, num, ca_top=2):
    suffix = "" if ca_top == 2 else f"_top{ca_top}"
    if dataset_name == "videomme":
        return MAIN / f"outputs/videomme_q35_{method}_601_{600 + num}{suffix}"
    return MAIN / f"outputs/lvbench_q35_{method}{suffix}"


def prepare_cache(dataset_name, method, video_id):
    """Symlink graph-stage artifacts; fresh questions/ for new-model results."""
    src = SRC_CACHE[dataset_name][method] / str(video_id)
    dst = MAIN / f"AVA_cache/{NEW_CACHE_PREFIX[dataset_name]}_{method}" / str(video_id)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("frames", "kg"):
        link = dst / name
        if not link.exists():
            os.symlink(src / name, link)
    if not (dst / "config.json").exists():
        (dst / "config.json").write_text((src / "config.json").read_text())
    (dst / "questions").mkdir(exist_ok=True)
    return dst


def graph_nodes(work_dir):
    p = work_dir / "kg/graph_event_knowledge_graph.graphml"
    return p.read_text(errors="ignore").count("<node ") if p.is_file() else 0


def answer_letter(value):
    m = re.search(r"\b([A-D])\b", str(value).upper())
    return m.group(1) if m else ""


def persist(out_dir, stage, records):
    (out_dir / f"records_{stage}.json").write_text(json.dumps(records, indent=2))
    with (out_dir / f"timing_{stage}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(records, key=lambda k: [int(p) for p in k.split(":")]):
            w.writerow(records[k])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["videomme", "lvbench"], default="videomme")
    p.add_argument("--method", choices=["default", "kf3s", "kfmerge"], required=True)
    p.add_argument("--stage", choices=["sa", "ca"], required=True)
    p.add_argument("--num", type=int, default=None,
                   help="videos: videomme 601..600+num (default 20); "
                        "lvbench first num of the selection (default 10)")
    p.add_argument("--base-url", default="http://127.0.0.1:8035/v1")
    p.add_argument("--model", default=None,
                   help="served model id (default: stage's model)")
    p.add_argument("--ca-top", type=int, default=2,
                   help="CA candidate-node budget (default 2 = paper top-2; "
                        "!=2 uses suffixed caches/outputs). ca stage only.")
    args = p.parse_args()
    num = args.num or DEFAULT_NUM[args.dataset]
    if args.ca_top != 2 and args.stage != "ca":
        raise SystemExit("--ca-top only applies to --stage ca (SA is unaffected)")

    model_id = args.model or STAGE_MODEL[args.stage]
    wait_for_server(args.base_url, model_id, timeout=60)
    llm = (OpenAIChatLM if args.stage == "sa" else OpenAIChatVL)(
        model_id, args.base_url)

    from AVA.ava import AVA
    from video_utils import VideoRepresentation
    from dataset.init_dataset import init_dataset

    dataset = init_dataset(args.dataset)
    out = output_dir(args.dataset, args.method, num, args.ca_top)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((out / f"records_{args.stage}.json").read_text()) \
        if (out / f"records_{args.stage}.json").exists() else {}

    for video_id, question_ids in question_plan(args.dataset, num):
        work_dir = prepare_cache(args.dataset, args.method, video_id)
        if not graph_nodes(work_dir):
            print(f"[{args.stage}] {video_id}: skipped; graph missing", flush=True)
            continue
        cfg = json.loads((work_dir / "config.json").read_text())
        video = VideoRepresentation(cfg["source_path"], str(work_dir))
        all_qa = dataset.get_video_info(video_id)["qa"]
        for question_id in question_ids:
            qa = all_qa[question_id]
            key = f"{video_id}:{question_id}"
            if records.get(key, {}).get("status") == "ok":
                print(f"[{args.stage}] {key}: skipped", flush=True)
                continue
            started = time.time()
            answer, status = "", "ok"
            try:
                ava = AVA(video=video, llm_model=llm, ca_num_candidates=args.ca_top)
                if args.stage == "sa":
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
            persist(out, args.stage, records)
            print(f"[{args.stage}] {key}: {status} answer={answer} "
                  f"({records[key]['seconds']:.0f}s)", flush=True)

    ok = [r for r in records.values() if r["status"] == "ok"]
    acc = sum(r["answer"] == r["gt"] for r in ok) / len(ok) if ok else 0.0
    print(f"== {args.dataset}/{args.method}/{args.stage}: {len(ok)} ok, acc={acc:.3f} ==",
          flush=True)


if __name__ == "__main__":
    main()
