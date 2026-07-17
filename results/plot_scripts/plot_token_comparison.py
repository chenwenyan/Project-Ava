#!/usr/bin/env python3
"""Reconstruct, per method, the INPUT token counts fed to the LLM (SA answering,
Qwen2.5-32B) and the VLM (Qwen2.5-VL: graph frame-captioning + CA frame-recheck),
entirely offline from the on-disk caches -- no model generation is re-run.

Three multimodal/text call types are accounted (all exact):
  * Graph-VLM : one dense-description call per base chunk (=len(descriptions.json)),
                each fed a fixed `video_chunk_num_frames` frames + generate_description
                prompt. Per-call size is constant; only the #calls differs by method.
  * SA-LLM    : per tree node, input_prompt (cached in tree_information.json) submitted
                `len(responses)` times (self-consistency). Text-only.
  * CA-VLM    : per candidate node, checkframe_and_answer_COT prompt + frames sampled at
                1 fps over the node's frame_durations (capped 256), submitted
                `len(responses)` times.

Vision tokens/frame are computed exactly from the Qwen2.5-VL image processor on a real
540x360 cached frame. Text tokens from the matching HF tokenizers. Writes a per-method
CSV (results/csv/lvbench_token_comparison.csv) and an OSDI-style PDF
(results/pdf/lvbench_token_comparison.pdf, PDF only)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

NUM_VIDEOS = 5   # first N selected LVBench videos
VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LM_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
VIDEO_CHUNK_NUM_FRAMES = 6          # AVA ava.py default
SELECTION = REPO / "results/json/lvbench_selection.json"
METHODS = {  # name -> (frame/graph cache, records dir with per-q question text)
    "Default":   ("AVA_cache/LVBench",          "results/lvbench_default"),
    "KF-Direct": ("AVA_cache/LVBench_kf",       "results/lvbench_keyframe_direct"),
    "KF-Merge":  ("AVA_cache/LVBench_kf_merge", "results/lvbench_keyframe_merge"),
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "savefig.dpi": 300,
})
# "10-yuan RMB" palette: the ¥10 note's dominant teal / blue-green family.
# Distinct lightness + hatch so it survives B&W printing.
STYLE = {
    "SA (LLM)":       dict(color="#0f5b5c", hatch=""),     # deep teal
    "Graph (VLM)":    dict(color="#3ba49a", hatch=".."),   # medium teal
    "CA (VLM)":       dict(color="#9fd3c9", hatch="///"),  # light aqua
}
METHOD_COLORS = ["#0f5b5c", "#3ba49a", "#9fd3c9"]  # Default, KF-Direct, KF-Merge
GAIN = "#0c4a3e"  # deep teal-green for the relative-change annotation


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def frames_by_fps_count(frame_durations, video_fps, cap=256):
    """Replicate VideoRepresentation.get_frames_by_fps(fps=1) frame count over a list
    of (start,end) durations, then the CA max_frames=256 cap."""
    step = int(video_fps)  # int(video_fps / 1)
    total = 0
    for start, end in frame_durations:
        if start == end:
            total += 1
        else:
            total += len(range(int(start * video_fps), int(end * video_fps), step))
    return min(total, cap)


def main():
    proc = AutoProcessor.from_pretrained(VL_MODEL)
    tok_vl = proc.tokenizer
    tok_lm = AutoTokenizer.from_pretrained(LM_MODEL)

    # exact vision tokens for one real 540x360 cached frame
    sample = next((REPO / "AVA_cache/LVBench/4/frames").glob("*.jpg"))
    grid = proc.image_processor(images=[Image.open(sample).convert("RGB")],
                                return_tensors="pt")["image_grid_thw"][0].tolist()
    merge = proc.image_processor.merge_size
    TPF = (grid[0] * grid[1] * grid[2]) // (merge * merge)      # vision tokens / frame

    from AVA.prompt import PROMPTS
    GRAPH_CALL = len(tok_vl(PROMPTS["generate_description"])["input_ids"]) + \
        VIDEO_CHUNK_NUM_FRAMES * TPF                             # constant per caption call

    sel = json.loads(SELECTION.read_text())
    video_ids, qmap = sel["video_ids"][:NUM_VIDEOS], sel["questions"]

    # rows[name] = {call_type: {"tokens": total_input_tokens, "calls": n_calls}}
    rows = {}
    for name, (cache, recdir) in METHODS.items():
        cache = REPO / cache
        agg = {k: {"tokens": 0, "calls": 0} for k in STYLE}
        rec_ca = load(REPO / recdir / "records_ca.json") or {}
        for vid in video_ids:
            vdir = cache / str(vid)
            fps = json.loads((vdir / "config.json").read_text())["fps"]

            # --- Graph-VLM: one caption call per base chunk (constant size) ---
            desc = load(vdir / "kg/events/descriptions.json") or []
            agg["Graph (VLM)"]["tokens"] += len(desc) * GRAPH_CALL
            agg["Graph (VLM)"]["calls"] += len(desc)

            for qid in qmap[str(vid)]:
                qdir = vdir / "questions" / str(qid)
                tree = load(qdir / "tree_information.json")
                sa = load(qdir / "SA_self_consistency_result.json")
                ca = load(qdir / "CA_self_consistency_result.json")
                if tree is None:
                    continue
                dur_by_path = {tuple(n["path"]): n["frame_durations"] for n in tree}
                prompt_by_path = {tuple(n["path"]): n["input_prompt"] for n in tree}

                # --- SA-LLM: input_prompt x self-consistency, per node ---
                for node in (sa or []):
                    prompt = prompt_by_path.get(tuple(node["path"]))
                    if prompt is None:
                        continue
                    n_tok = len(tok_lm(prompt)["input_ids"])
                    reps = len(node["responses"])
                    agg["SA (LLM)"]["tokens"] += n_tok * reps
                    agg["SA (LLM)"]["calls"] += reps

                # --- CA-VLM: checkframe prompt + sampled frames, per candidate ---
                query = rec_ca.get(f"{vid}:{qid}", {}).get("question", "")
                text_tok = len(tok_vl(PROMPTS["checkframe_and_answer_COT"]
                                      .format(user_query=query))["input_ids"])
                for cand in (ca or []):
                    durs = dur_by_path.get(tuple(cand["path"]))
                    if durs is None:
                        continue
                    nf = frames_by_fps_count(durs, fps)
                    reps = len(cand["responses"])
                    agg["CA (VLM)"]["tokens"] += (text_tok + nf * TPF) * reps
                    agg["CA (VLM)"]["calls"] += reps
        rows[name] = agg

    # ---- report + CSV ----
    print(f"vision tokens/frame(540x360)={TPF}  graph per-call={GRAPH_CALL}")
    names = list(METHODS)
    types = list(STYLE)
    csv = REPO / "results/csv/lvbench_token_comparison.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    with csv.open("w") as f:
        f.write("method,call_type,total_input_tokens,num_calls,mean_tokens_per_call\n")
        for name in names:
            for t in types:
                a = rows[name][t]
                mean = a["tokens"] / a["calls"] if a["calls"] else 0
                f.write(f"{name},{t},{a['tokens']},{a['calls']},{mean:.1f}\n")
                print(f"{name:10s} {t:13s} total={a['tokens']/1e6:8.2f}M "
                      f"calls={a['calls']:6d} mean/call={mean:8.1f}")

    # ---- plot: (a) mean tokens/call by type (log), (b) total tokens stacked ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.85))
    di, ki = names.index("Default"), names.index("KF-Direct")

    # (a) grouped bars, log y: per-call mean input tokens spans ~30x across types
    w = 0.26
    for j, name in enumerate(names):
        vals = [rows[name][t]["tokens"] / rows[name][t]["calls"]
                if rows[name][t]["calls"] else 0 for t in types]
        xs = [i + (j - 1) * w for i in range(len(types))]
        bars = ax1.bar(xs, vals, w, edgecolor="black", linewidth=0.7,
                       color=METHOD_COLORS[j], label=name)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v, f"{v/1000:.0f}k", ha="center",
                     va="bottom", fontsize=8.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Input tokens / call (log)")
    ax1.set_title("(a) Per-call input tokens")
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels([t.replace(" ", "\n") for t in types])
    ax1.set_ylim(700, 1.1e5)
    ax1.legend(frameon=False, ncol=1, loc="upper left", handlelength=1.2, fontsize=8.5)

    # (b) stacked total input tokens (millions) per method, split by call type
    xs = list(range(len(names)))
    bottoms = [0.0] * len(names)
    for t in types:
        vals = [rows[n][t]["tokens"] / 1e6 for n in names]
        ax2.bar(xs, vals, 0.55, bottom=bottoms, edgecolor="black", linewidth=0.7,
                label=t, **STYLE[t])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, tot in enumerate(bottoms):
        ax2.text(i, tot, f"{tot:.1f}M", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylabel("Total input tokens (millions)")
    ax2.set_title("(b) Total input tokens")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, max(bottoms) * 1.24)
    ax2.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.2, fontsize=8.5)
    if bottoms[di]:  # relative total-token reduction of KF-Direct vs Default
        rel = (bottoms[ki] - bottoms[di]) / bottoms[di] * 100
        ax2.text(ki, bottoms[ki] + max(bottoms) * 0.06, f"{rel:+.0f}%\nvs Default",
                 ha="center", va="bottom", fontsize=9, color=GAIN, fontweight="bold")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    fig.tight_layout()
    out = REPO / "results/pdf/lvbench_token_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote: {csv}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
