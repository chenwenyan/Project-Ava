#!/bin/bash
# CA candidate-budget ablation: rerun ONLY the CA stage with top-3 and top-4
# candidate nodes (vs the already-run top-2 default), reusing the existing
# q35 SA results on both datasets. Models match each dataset's top-2 run:
#   VideoMME -> Qwen/Qwen3-VL-8B-Instruct-FP8   (its top-2 CA was FP8)
#   LVBench  -> Qwen/Qwen3-VL-8B-Instruct       (its top-2 CA was BF16)
# Outputs: outputs/<ds>_q35_<method>[_601_620]_top{3,4}/records_ca.json
# Log: results/qwen35_catop.log   Usage: nohup bash scripts/run_qwen35_catop.sh &
set -u
MAIN=/export/home/yulin/wenyan/github/Project-Ava
QENV=~/miniconda3/envs/qwen35/bin
AVAPY=~/miniconda3/envs/ava/bin/python
PORT=8035
GPU_UTIL=${GPU_UTIL:-0.5}
MIN_FREE_MIB=${MIN_FREE_MIB:-46000}
cd "$MAIN"

export PATH="$QENV:$PATH"
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
SERVER_FLAGS="--max-model-len 131072 --max-num-seqs 4 --limit-mm-per-prompt {\"image\":260}"
SERVER_PID=""

wait_gpu_room() {
  # Latency policy (user 2026-07-29): only run on an EXCLUSIVE GPU.
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$apps" -eq 0 ] && [ "$free" -ge "$MIN_FREE_MIB" ] && break
    echo "$(date +%H:%M:%S) GPU not exclusive (${apps} procs, ${free} MiB free); waiting..."
    sleep 120
  done
}

server_alive() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

start_server() {
  echo "=== starting vllm serve $1 ==="
  nohup "$QENV/vllm" serve "$1" --port $PORT --gpu-memory-utilization "$GPU_UTIL" \
    $SERVER_FLAGS > "results/vllm_catop_$(basename "$1").log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 240); do
    server_alive && { echo "server ready (pid $SERVER_PID)"; return 0; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER DIED, see log"; return 1; }
    sleep 10
  done
  echo "server start timeout"; return 1
}

stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; sleep 15; }

run_ca() { # $1 dataset, $2 model
  for top in 3 4; do
    for m in default kf3s kfmerge; do
      for attempt in 1 2 3; do
        if ! server_alive; then
          echo "=== server dead; restarting $2 (attempt $attempt) ==="
          stop_server; wait_gpu_room; start_server "$2" || exit 1
        fi
        echo "=== $1 / ca top$top / $m pass$attempt $(date +%H:%M:%S) ==="
        PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "$AVAPY" scripts/run_qwen35_experiment.py --dataset "$1" --method "$m" \
          --stage ca --model "$2" --ca-top "$top" ${NUM:+--num "$NUM"} \
          || echo "WARN: $1/ca-top$top/$m pass$attempt exited nonzero"
        server_alive && break
      done
    done
  done
}

# ONLY=videomme|lvbench limits the run; NUM overrides the driver's video count.
ONLY=${ONLY:-}
NUM=${NUM:-}

if [ "$ONLY" != lvbench ]; then
  wait_gpu_room
  start_server Qwen/Qwen3-VL-8B-Instruct-FP8 || exit 1
  run_ca videomme Qwen/Qwen3-VL-8B-Instruct-FP8
  stop_server
fi

if [ "$ONLY" != videomme ]; then
  wait_gpu_room
  start_server Qwen/Qwen3-VL-8B-Instruct || exit 1
  run_ca lvbench Qwen/Qwen3-VL-8B-Instruct
  stop_server
fi
echo "CATOP ALL DONE $(date)"
