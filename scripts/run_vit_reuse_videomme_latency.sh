#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AVA_PY=/data/wenyan/conda/envs/ava/bin/python
VL_PY=/data/wenyan/conda/envs/qwen35/bin/python
LMDEPLOY=/data/wenyan/conda/envs/qwen35/bin/lmdeploy
MODEL=Qwen/Qwen3-VL-8B-Instruct
METHOD=kf3sig_a07_t07
PORT_OFF=23830
PORT_ON=23831
STORAGE=${VIT_REUSE_STORAGE:-disk}
case "$STORAGE" in
  disk)
    OUT="$REPO/results/vit_reuse_videomme_601_620"
    TAG_OFF=vitreuse_off
    TAG_ON=vitreuse_on
    STEM=videomme_vit_reuse
    ;;
  tiered)
    OUT="$REPO/results/vit_reuse_tiered_videomme_601_620"
    TAG_OFF=vitreuse_tiered_off
    TAG_ON=vitreuse_tiered_on
    STEM=videomme_vit_reuse_tiered
    ;;
  *)
    echo "Unsupported VIT_REUSE_STORAGE: $STORAGE" >&2
    exit 2
    ;;
esac

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

server_off=""
server_on=""
cleanup() {
  for pid in "$server_off" "$server_on"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 "$LMDEPLOY" serve api_server "$MODEL" \
  --backend pytorch --server-name 127.0.0.1 --server-port "$PORT_OFF" \
  --model-name "$MODEL" --tp 1 --session-len 131072 --max-batch-size 4 \
  --cache-max-entry-count 0.3 --vision-max-batch-size 4 \
  >"$OUT/server_off.log" 2>&1 &
server_off=$!

export AVA_VIT_TOKEN_CACHE_DIR="$OUT/token_cache"
export AVA_VIT_TOKEN_CACHE_NAMESPACE="Qwen3-VL-8B-Instruct-videomme-601-620"
export AVA_VIT_TOKEN_CACHE_METRICS="$OUT/cache_metrics.json"
export AVA_VIT_STORAGE="$STORAGE"
export AVA_VIT_GPU_CACHE_GIB=20
CUDA_VISIBLE_DEVICES=1 "$VL_PY" -m scripts.serve_vit_token_cache serve api_server "$MODEL" \
  --backend pytorch --server-name 127.0.0.1 --server-port "$PORT_ON" \
  --model-name "$MODEL" --tp 1 --session-len 131072 --max-batch-size 4 \
  --cache-max-entry-count 0.3 --vision-max-batch-size 4 \
  >"$OUT/server_on.log" 2>&1 &
server_on=$!

for spec in "$PORT_OFF:$server_off:off" "$PORT_ON:$server_on:on"; do
  IFS=: read -r port pid label <<< "$spec"
  ready=0
  for _ in $(seq 1 720); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 80 "$OUT/server_${label}.log"
      exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -n 80 "$OUT/server_${label}.log"
    exit 1
  fi
done

"$VL_PY" -m scripts.warmup_vit_reuse_server --ports "$PORT_OFF" "$PORT_ON" --model "$MODEL"

CUDA_VISIBLE_DEVICES=0 "$AVA_PY" scripts/run_qwen35_experiment.py \
  --dataset videomme --method "$METHOD" --stage ca --start-id 601 --num 20 \
  --run-tag "$TAG_OFF" --base-url "http://127.0.0.1:${PORT_OFF}/v1" \
  --model "$MODEL" >"$OUT/eval_off.log" 2>&1 &
eval_off=$!
CUDA_VISIBLE_DEVICES=1 "$AVA_PY" scripts/run_qwen35_experiment.py \
  --dataset videomme --method "$METHOD" --stage ca --start-id 601 --num 20 \
  --run-tag "$TAG_ON" --base-url "http://127.0.0.1:${PORT_ON}/v1" \
  --model "$MODEL" >"$OUT/eval_on.log" 2>&1 &
eval_on=$!

status=0
wait "$eval_off" || status=1
wait "$eval_on" || status=1
if [[ "$status" != 0 ]]; then
  tail -n 50 "$OUT/eval_off.log"
  tail -n 50 "$OUT/eval_on.log"
  exit 1
fi

"$AVA_PY" results/plot_scripts/plot_vit_reuse_videomme_latency.py \
  --off "results/qwen3_videomme_${METHOD}_${TAG_OFF}_601_620/records_ca.json" \
  --on "results/qwen3_videomme_${METHOD}_${TAG_ON}_601_620/records_ca.json" \
  --metrics "$OUT/cache_metrics.json" \
  --csv "results/csv/${STEM}_latency_601_620.csv" \
  --summary "$OUT/summary.json" \
  --pdf "results/pdf/${STEM}_latency_speedup.pdf"
