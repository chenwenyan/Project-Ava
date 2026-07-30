#!/bin/bash
# KF-Direct on GOP-re-encoded VideoMME 601-620: run graph->SA->CA for each GOP
# variant into an isolated cache/output, then score accuracy + wall-clock latency.
# Sequential (single GPU, models load per stage). Resumable: the pipeline skips
# videos/questions already done, so re-running continues where it stopped.
set -euo pipefail
cd /export/home/yulin/wenyan/github/Project-Ava

PY=/export/home/yulin/miniconda3/envs/ava/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
NUM=20                       # videos 601..620
TAGS=(gop3s gop5s)
LOG=gop_experiment.log
: > "$LOG"

echo "=== GOP experiment: KF-Direct on ${TAGS[*]}, videos 601-$((601+NUM-1)) ===" | tee -a "$LOG"
for tag in "${TAGS[@]}"; do
  out="outputs/videomme_keyframe_${tag}_601_620"
  echo "" | tee -a "$LOG"
  echo "########## $tag -> $out ##########  $(date '+%F %T')" | tee -a "$LOG"

  # verify all 20 re-encoded sources exist before starting this variant
  src="/export/home/yulin/wenyan/dataset/VideoMME/videos_${tag}"
  n=$(ls -1 "$src"/*.mp4 2>/dev/null | wc -l)
  if [ "$n" -lt "$NUM" ]; then
    echo "!! $tag: only $n/$NUM re-encoded videos in $src -- skipping" | tee -a "$LOG"
    continue
  fi

  $PY scripts/run_keyframe_pipeline.py \
      --mode direct --stage all --num "$NUM" --gpus 1 \
      --cache-tag "$tag" --src-subdir "videos_${tag}" \
      --output-dir "$out" 2>&1 | tee -a "$LOG"

  echo "---- scoring $tag ----" | tee -a "$LOG"
  $PY scripts/aggregate_timing.py --output-dir "$out" 2>&1 | tee -a "$LOG"
  echo "########## $tag done $(date '+%F %T') ##########" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "=== ALL GOP VARIANTS DONE $(date '+%F %T') ===" | tee -a "$LOG"
