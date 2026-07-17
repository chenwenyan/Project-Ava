#!/usr/bin/env python3
"""Export VideoMME-Long run timing + results to CSV (and a summary.json).

Writes into --output-dir:
  results.csv   one row per question: SA/CA answers, correctness, seconds.
  summary.csv   one row per stage: counts, total/mean seconds, accuracy.
  summary.json  same summary as JSON.
"""

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGES = ("graph", "sa", "ca")


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(stage, records):
    rows = list(records.values())
    ok = [r for r in rows if r.get("status") in {"ok", "skipped"}]
    timed = [r for r in rows if r.get("status") == "ok"]
    total = round(sum(float(r.get("seconds", 0)) for r in timed), 2)
    out = {
        "stage": stage, "records": len(rows), "successful": len(ok),
        "failed": len(rows) - len(ok), "total_seconds": total,
        "mean_seconds": round(total / len(timed), 2) if timed else 0.0,
    }
    if stage == "graph":
        out["total_nodes"] = sum(int(r.get("nodes", 0)) for r in rows)
    else:
        scored = [r for r in ok if r.get("gt")]
        correct = sum(r.get("answer") == r.get("gt") for r in scored)
        out["correct"] = correct
        out["total_questions"] = len(scored)
        out["accuracy"] = round(correct / len(scored), 4) if scored else None
    return out


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/videomme_long_601_620")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {stage: load(args.output_dir / f"records_{stage}.json") for stage in STAGES}
    graph, sa, ca = records["graph"], records["sa"], records["ca"]

    # Per-question merged results (SA + CA), sorted by video_id:question_id.
    keys = sorted(set(sa) | set(ca), key=lambda k: [int(p) for p in k.split(":")])
    results = []
    for key in keys:
        vid = key.split(":")[0]
        s, c = sa.get(key, {}), ca.get(key, {})
        gt = s.get("gt") or c.get("gt", "")
        results.append({
            "video_id": vid, "question_id": key.split(":")[1],
            "gt": gt,
            "sa_answer": s.get("answer", ""),
            "sa_correct": int(s.get("answer") == gt) if gt and s.get("answer") else "",
            "sa_seconds": s.get("seconds", ""), "sa_status": s.get("status", ""),
            "ca_answer": c.get("answer", ""),
            "ca_correct": int(c.get("answer") == gt) if gt and c.get("answer") else "",
            "ca_seconds": c.get("seconds", ""), "ca_status": c.get("status", ""),
            "graph_seconds": graph.get(vid, {}).get("seconds", ""),
            "question": (s.get("question") or c.get("question", "")).replace("\n", " | "),
        })
    write_csv(args.output_dir / "results.csv", [
        "video_id", "question_id", "gt",
        "sa_answer", "sa_correct", "sa_seconds", "sa_status",
        "ca_answer", "ca_correct", "ca_seconds", "ca_status",
        "graph_seconds", "question",
    ], results)

    # Per-stage summary.
    summary = [summarize(stage, records[stage]) for stage in STAGES if records[stage]]
    fields = ["stage", "records", "successful", "failed", "total_seconds",
              "mean_seconds", "correct", "total_questions", "accuracy", "total_nodes"]
    write_csv(args.output_dir / "summary.csv", fields, summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {args.output_dir}/results.csv, summary.csv, summary.json")


if __name__ == "__main__":
    main()
