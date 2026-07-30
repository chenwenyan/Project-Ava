#!/usr/bin/env python3
"""Mode-B GOP re-encode of VideoMME videos 601-620.

Mode B = keep libx264's scene-cut adaptive I-frames (default sc_threshold=40),
change ONLY the maximum GOP upper bound `-g` (= target_seconds * fps, rounded).
Encoder is CPU libx264 (the H200 datacenter GPU has no NVENC engine), CRF 18,
preset medium, audio stream-copied.

For each (target_gop_seconds, video_id) it writes
  <SRC_ROOT>/videos_gop{N}s/<basename>.mp4
mirroring the original videos/<basename>.mp4 layout, so a downstream run can just
point its source dir at videos_gop{N}s/. Existing valid outputs are skipped
(resumable). A small pool runs several encodes concurrently."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FFMPEG = "/export/home/yulin/miniconda3/envs/vllm/bin/ffmpeg"
FFPROBE = "/export/home/yulin/miniconda3/envs/vllm/bin/ffprobe"

VIDEO_IDS = list(range(601, 621))
GOP_SECONDS = [3, 5]          # mode-B targets to sweep (below current 7s cap)
CRF = 18
PRESET = "medium"
N_PARALLEL = 4                # each libx264 medium uses ~13 of 64 cores
LOG = REPO / "reencode_gop.log"


def log(msg):
    line = f"{msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a") as f:
        f.write(line)


def cache_config(vid):
    return json.loads((REPO / f"AVA_cache/VideoMME/{vid}/config.json").read_text())


def nframes(path):
    """Decoded frame count via ffprobe (validates a finished output)."""
    out = subprocess.run(
        [FFPROBE, "-hide_banner", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return -1


def encode_one(vid, gop_s):
    cfg = cache_config(vid)
    src = Path(cfg["source_path"])
    fps = cfg["fps"]
    g = max(1, round(gop_s * fps))
    out_dir = src.parent.parent / f"videos_gop{gop_s}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / src.name

    if out.exists() and nframes(out) == cfg["frame_count"]:
        log(f"[skip] gop{gop_s}s id{vid} {src.name} (already {cfg['frame_count']} frames)")
        return (vid, gop_s, "skip", g)

    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
           "-g", str(g), "-keyint_min", "1", "-sc_threshold", "40",
           "-c:a", "copy", str(out)]
    log(f"[run ] gop{gop_s}s id{vid} {src.name} fps={fps} -g={g} "
        f"({cfg['duration']/60:.1f}min)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[FAIL] gop{gop_s}s id{vid} rc={r.returncode}: {r.stderr[-400:]}")
        return (vid, gop_s, "fail", g)
    nf = nframes(out)
    ok = abs(nf - cfg["frame_count"]) <= 2      # allow +-2 frame remux slack
    log(f"[{'done' if ok else 'WARN'}] gop{gop_s}s id{vid} frames {nf}/{cfg['frame_count']}")
    return (vid, gop_s, "done" if ok else "warn", g)


def main():
    LOG.write_text("")
    jobs = [(vid, g) for g in GOP_SECONDS for vid in VIDEO_IDS]
    log(f"=== re-encode {len(jobs)} jobs "
        f"({len(VIDEO_IDS)} videos x {len(GOP_SECONDS)} GOP) parallel={N_PARALLEL} ===")
    results = []
    with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        futs = {ex.submit(encode_one, vid, g): (vid, g) for vid, g in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    done = sum(1 for _, _, s, _ in results if s in ("done", "skip"))
    log(f"=== finished: {done}/{len(jobs)} ok; "
        f"warn={sum(s=='warn' for *_, s, _ in [(r[0],r[1],r[2],r[3]) for r in results])} "
        f"fail={sum(1 for _,_,s,_ in results if s=='fail')} ===")
    for vid, g, s, gval in sorted(results):
        if s not in ("done", "skip"):
            log(f"  !! id{vid} gop{g}s -> {s}")


if __name__ == "__main__":
    main()
