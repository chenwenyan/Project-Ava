#!/usr/bin/env bash
# Full LVBench experiment: preprocess frames -> 3 methods (default/direct/merge) ->
# analysis (aggregate + 3-way compare + CDF + comparison chart). Resumable.
# Requires datas/LVBench/videos populated and results/lvbench_selection.json.
set -uo pipefail
cd /export/home/yulin/wenyan/github/Project-Ava
PY=/export/home/yulin/miniconda3/envs/ava/bin/python
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "PREPROCESS frames (selected LVBench videos)"
$PY scripts/preprocess_lvbench.py || { log "preprocess FAILED"; exit 1; }

log "METHOD 1/3 default (fixed 3s + semantic merge)"
$PY scripts/run_lvbench.py --mode default --stage all --gpus 1 || log "default FAILED (continuing)"
log "METHOD 2/3 keyframe-direct (keyframe=event, no merge)"
$PY scripts/run_lvbench.py --mode direct --stage all --gpus 1 || log "direct FAILED (continuing)"
log "METHOD 3/3 keyframe-merge (keyframe base windows + semantic merge)"
$PY scripts/run_lvbench.py --mode merge --stage all --gpus 1 || log "merge FAILED (continuing)"

log "ANALYSIS"
$PY scripts/aggregate_timing.py --output-dir results/lvbench_default
$PY scripts/aggregate_timing.py --output-dir results/lvbench_keyframe_direct
$PY scripts/aggregate_timing.py --output-dir results/lvbench_keyframe_merge
$PY scripts/compare_methods.py --dataset lvbench
$PY results/plot_scripts/plot_segment_cdf.py --dataset lvbench
$PY results/plot_scripts/plot_comparison.py --dataset lvbench
$PY results/plot_scripts/plot_osdi_comparison.py --dataset lvbench
log "ALL DONE"
