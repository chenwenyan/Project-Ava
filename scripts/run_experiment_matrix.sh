#!/usr/bin/env bash
# Run all three chunking methods on the first N VideoMME-Long videos (601..600+N),
# then aggregate + 3-way compare + keyframe-chunk CDF. All steps resume from disk,
# so re-running is safe. Single GPU: methods run sequentially. Usage: run_experiment_matrix.sh [N]
set -uo pipefail
cd /export/home/yulin/wenyan/github/Project-Ava
PY=/export/home/yulin/miniconda3/envs/ava/bin/python
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
N="${1:-20}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "METHOD 1/3 default (fixed 3s + semantic merge), num=$N"
$PY scripts/run_pipeline_stage.py --stage all --num "$N" --gpus 1 || log "default FAILED (continuing)"

log "METHOD 2/3 keyframe-direct (keyframe=event, no merge), num=$N"
$PY scripts/run_keyframe_pipeline.py --mode direct --stage all --num "$N" --gpus 1 || log "keyframe-direct FAILED (continuing)"

log "METHOD 3/3 keyframe-merge (keyframe base windows + semantic merge), num=$N"
$PY scripts/run_keyframe_pipeline.py --mode merge --stage all --num "$N" --gpus 1 || log "keyframe-merge FAILED (continuing)"

log "ANALYSIS: per-method aggregate + 3-way compare + keyframe-chunk CDF"
$PY scripts/aggregate_timing.py --output-dir outputs/videomme_long_601_620
$PY scripts/aggregate_timing.py --output-dir outputs/videomme_keyframe_601_605
$PY scripts/aggregate_timing.py --output-dir outputs/videomme_keyframe_merge_601_605
$PY scripts/compare_methods.py
$PY results/plot_scripts/plot_segment_cdf.py --num "$N"
$PY results/plot_scripts/plot_comparison.py
log "ALL DONE"
