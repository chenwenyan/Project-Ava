#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/data/wenyan/conda/envs/qwen35/bin/python
MODEL=Qwen/Qwen3-VL-8B-Instruct
IMAGE="$REPO/AVA_cache/VideoMME/601/frames/0.jpg"
CACHE="$REPO/results/vit_token_cache_smoke"
METRICS="$CACHE/metrics.json"
OUTPUT="$REPO/results/vit_token_cache_smoke.json"
LOG="$REPO/results/vit_token_cache_smoke_server.log"
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')

mkdir -p "$CACHE"
export AVA_VIT_TOKEN_CACHE_DIR="$CACHE"
export AVA_VIT_TOKEN_CACHE_NAMESPACE="Qwen3-VL-8B-Instruct-smoke-20261007"
export AVA_VIT_TOKEN_CACHE_METRICS="$METRICS"
export HF_HOME=${HF_HOME:-/data/hf-cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data/hf-cache/hub}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.serve_vit_token_cache serve api_server "$MODEL" \
  --backend pytorch --server-name 127.0.0.1 --server-port "$PORT" \
  --model-name "$MODEL" --tp 1 --session-len 8192 --max-batch-size 1 \
  --cache-max-entry-count 0.1 --vision-max-batch-size 1 >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

READY=0
for _ in $(seq 1 360); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 60 "$LOG"
    exit 1
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  tail -n 60 "$LOG"
  exit 1
fi
"$PY" -m scripts.smoke_vit_token_cache --port "$PORT" --image "$IMAGE" \
  --metrics "$METRICS" --output "$OUTPUT"
