#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AVA_PY=/data/wenyan/conda/envs/ava/bin/python
VL_PY=/data/wenyan/conda/envs/qwen35/bin/python
LMDEPLOY=/data/wenyan/conda/envs/qwen35/bin/lmdeploy
MODEL=Qwen/Qwen3-VL-8B-Instruct
METHOD=kf3sig_a07_t07
OUT="$REPO/results/vit_reuse_adaptive_videomme_601_620"
server_pid=""

mkdir -p "$OUT"
cd "$REPO"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/data/hf-cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data/hf-cache/hub}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/data/wenyan/hf-modules-cache}
export TORCH_HOME=${TORCH_HOME:-/data/wenyan/torch-cache}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

for vid in $(seq 601 620); do
  test -f "AVA_cache/VideoMME_${METHOD}/${vid}/kg/graph_event_knowledge_graph.graphml"
  for q in 0 1 2; do
    test -f "AVA_cache/VideoMME_q35_${METHOD}/${vid}/questions/${q}/SA_self_consistency_result.json"
  done
done

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup EXIT INT TERM

for phase in off on; do
  if [[ "$phase" == on ]]; then
    PORT=23833
    export AVA_VIT_TOKEN_CACHE_DIR="$OUT/token_cache"
    export AVA_VIT_TOKEN_CACHE_NAMESPACE="Qwen3-VL-8B-Instruct-videomme-adaptive-601-620"
    export AVA_VIT_TOKEN_CACHE_METRICS="$OUT/cache_metrics.json"
    export AVA_VIT_STORAGE=tiered
    export AVA_VIT_GPU_CACHE_GIB=20
    export AVA_VIT_HOST_RESERVE_GIB=32
    CUDA_VISIBLE_DEVICES=0 "$VL_PY" -m scripts.serve_vit_token_cache serve api_server "$MODEL" \
      --backend pytorch --server-name 127.0.0.1 --server-port "$PORT" \
      --model-name "$MODEL" --tp 1 --session-len 131072 --max-batch-size 4 \
      --cache-max-entry-count 0.3 --vision-max-batch-size 4 \
      >"$OUT/server_${phase}.log" 2>&1 &
  else
    PORT=23832
    CUDA_VISIBLE_DEVICES=0 "$LMDEPLOY" serve api_server "$MODEL" \
      --backend pytorch --server-name 127.0.0.1 --server-port "$PORT" \
      --model-name "$MODEL" --tp 1 --session-len 131072 --max-batch-size 4 \
      --cache-max-entry-count 0.3 --vision-max-batch-size 4 \
      >"$OUT/server_${phase}.log" 2>&1 &
  fi
  server_pid=$!
  ready=0
  for _ in $(seq 1 720); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 80 "$OUT/server_${phase}.log"
      exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -n 80 "$OUT/server_${phase}.log"
    exit 1
  fi
  "$VL_PY" -m scripts.warmup_vit_reuse_server --ports "$PORT" --model "$MODEL"
  CUDA_VISIBLE_DEVICES=0 "$AVA_PY" scripts/run_qwen35_experiment.py \
    --dataset videomme --method "$METHOD" --stage ca --start-id 601 --num 20 \
    --run-tag "vitreuse_adaptive_${phase}" \
    --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
    >"$OUT/eval_${phase}.log" 2>&1
  cleanup
done

"$AVA_PY" results/plot_scripts/plot_vit_reuse_videomme_latency.py \
  --off "results/qwen3_videomme_${METHOD}_vitreuse_adaptive_off_601_620/records_ca.json" \
  --on "results/qwen3_videomme_${METHOD}_vitreuse_adaptive_on_601_620/records_ca.json" \
  --metrics "$OUT/cache_metrics.json" \
  --csv results/csv/videomme_vit_reuse_adaptive_latency_601_620.csv \
  --summary "$OUT/summary.json" \
  --pdf results/pdf/videomme_vit_reuse_adaptive_latency_speedup.pdf
