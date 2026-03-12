#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
API_SERVER_COUNT="${API_SERVER_COUNT:-2}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"
API_KEY="${API_KEY:-EMPTY}"

exec vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --enable-expert-parallel \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --api-server-count "${API_SERVER_COUNT}" \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt video=0 \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --enforce-eager
