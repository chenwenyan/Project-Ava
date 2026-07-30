#!/bin/bash
# Full new-model experiment: SA=Qwen3.5-27B-FP8, CA=Qwen3-VL-8B-Instruct-FP8,
# VideoMME 601-620, methods: default, kf3s (KF-Direct gop3s), kfmerge.
#
# Serves each model once with vLLM (qwen35 env) and runs all three methods
# against it: 2 server launches total. Driver runs in the ava env.
#
# Server flags learned the hard way (2026-07-27 smoke tests):
#  - no CUDA toolkit on host -> avoid every flashinfer/deepgemm JIT path:
#    VLLM_USE_DEEP_GEMM=0, VLLM_USE_FLASHINFER_SAMPLER=0, --gdn-prefill-backend triton
#  - qwen35 env bin must be on PATH (ninja etc. for any residual JIT)
#  - Qwen3.5's GDN linear attention preallocates per-seq state: default
#    max_num_seqs OOMs on a shared GPU -> cap it
#  - GPU is shared with another user's training (~90 GiB) -> low
#    --gpu-memory-utilization; raise GPU_UTIL when the GPU is exclusive
#  - VL keeps 131072 ctx so 256-frame CA prompts (~65k tok) never overflow
# Log: results/qwen35_experiment.log   Usage: nohup bash scripts/run_qwen35_all.sh &
set -u
MAIN=/export/home/yulin/wenyan/github/Project-Ava
QENV=~/miniconda3/envs/qwen35/bin
AVAPY=~/miniconda3/envs/ava/bin/python
PORT=8035
DATASET=${DATASET:-videomme}   # videomme | lvbench
NUM=${NUM:-}                   # empty -> driver default (videomme 20, lvbench 10)
SA_MODEL=${SA_MODEL:-Qwen/Qwen3.5-27B-FP8}
CA_MODEL=${CA_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}
GPU_UTIL=${GPU_UTIL:-0.28}
MIN_FREE_MIB=${MIN_FREE_MIB:-46000}
cd "$MAIN"

export PATH="$QENV:$PATH"
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

wait_gpu_room() {
  # Latency policy (user 2026-07-29): measure only on an EXCLUSIVE GPU -- wait
  # until no other compute process is running, not just until memory frees up.
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$apps" -eq 0 ] && [ "$free" -ge "$MIN_FREE_MIB" ] && break
    echo "$(date +%H:%M:%S) GPU not exclusive (${apps} procs, ${free} MiB free); waiting..."
    sleep 120
  done
}

start_server() { # $1 model, extra args after
  model=$1; shift
  echo "=== starting vllm serve $model ==="
  nohup "$QENV/vllm" serve "$model" --port $PORT --gpu-memory-utilization "$GPU_UTIL" \
    "$@" > "results/vllm_$(basename "$model").log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "server ready (pid $SERVER_PID)"; return 0
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER DIED, see log"; return 1; }
    sleep 10
  done
  echo "server start timeout"; return 1
}

stop_server() {
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
  sleep 20
}

server_alive() {
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1
}

run_stage() { # $1 stage, $2 model id; SERVER_FLAGS holds the serve args.
  # Self-healing: the vLLM EngineCore occasionally dies silently; if the server
  # is dead after a driver pass, restart it and rerun (driver resumes, retrying
  # error-status questions). Up to 3 passes per method.
  for m in default kf3s kfmerge; do
    for attempt in 1 2 3; do
      if ! server_alive; then
        echo "=== server dead; restarting $2 (attempt $attempt) ==="
        stop_server
        wait_gpu_room
        start_server "$2" $SERVER_FLAGS || exit 1
      fi
      echo "=== $DATASET / $1 / $m pass$attempt $(date +%H:%M:%S) ==="
      PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$AVAPY" scripts/run_qwen35_experiment.py --dataset "$DATASET" --method "$m" \
        --stage "$1" --model "$2" ${NUM:+--num "$NUM"} \
        || echo "WARN: $1/$m pass$attempt exited nonzero"
      server_alive && break
    done
  done
}

SERVER_FLAGS="--max-model-len 65536 --max-num-seqs 32 --gdn-prefill-backend triton"
wait_gpu_room
start_server "$SA_MODEL" $SERVER_FLAGS || exit 1
run_stage sa "$SA_MODEL"
stop_server

SERVER_FLAGS="--max-model-len 131072 --max-num-seqs 4 --limit-mm-per-prompt {\"image\":260}"
wait_gpu_room
start_server "$CA_MODEL" $SERVER_FLAGS || exit 1
run_stage ca "$CA_MODEL"
stop_server

if [ "$DATASET" = videomme ]; then
  for m in default kf3s kfmerge; do
    "$AVAPY" scripts/aggregate_timing.py --output-dir "outputs/videomme_q35_${m}_601_$((600 + ${NUM:-20}))" \
      || echo "WARN: aggregate $m failed"
  done
fi
echo "ALL DONE $(date)"
