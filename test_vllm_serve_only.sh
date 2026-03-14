#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-$SCRIPT_DIR}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# Keep original parallel config for your cluster smoke test.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-600}"
HOLD_SECONDS="${HOLD_SECONDS:-10}"

VLLM_BIN="${VLLM_BIN:-$PROJECT_ROOT/.venv/bin/vllm}"

JOB_ID="${SLURM_JOB_ID:-manual}"
NODE_RANK="${SLURM_PROCID:-${SLURM_NODEID:-0}}"
LOG_ROOT="${LOG_ROOT:-$AGENT_DIR/../agent_datas/processed/vllm_multi_node_state/logs/vllm_smoke_job_${JOB_ID}}"
NODE_LOG_DIR="${LOG_ROOT}/node_${NODE_RANK}"
VLLM_LOG="${NODE_LOG_DIR}/vllm_server.log"

mkdir -p "$NODE_LOG_DIR"

if [[ ! -x "$VLLM_BIN" ]]; then
  if command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
  else
    echo "[ERROR] vllm executable not found. Set VLLM_BIN or install in PATH."
    exit 2
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl is required for readiness checks"
  exit 2
fi

echo "[INFO] job_id=${JOB_ID} node_rank=${NODE_RANK} host=$(hostname)"
echo "[INFO] model=${MODEL}"
echo "[INFO] vllm_bin=${VLLM_BIN}"
echo "[INFO] base_url=${VLLM_BASE_URL}"
echo "[INFO] TP=${TENSOR_PARALLEL_SIZE} DP=${DATA_PARALLEL_SIZE}"
echo "[INFO] log=${VLLM_LOG}"

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[INFO] stopping vLLM pid=${VLLM_PID}"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
"$VLLM_BIN" serve "$MODEL" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[INFO] started vLLM pid=${VLLM_PID}, waiting readiness..."

for ((sec = 1; sec <= VLLM_STARTUP_TIMEOUT_S; sec++)); do
  if curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[INFO] readiness passed at ${VLLM_BASE_URL}/models (after ${sec}s)"
    if [[ "$HOLD_SECONDS" -gt 0 ]]; then
      echo "[INFO] hold ${HOLD_SECONDS}s then exit"
      sleep "$HOLD_SECONDS"
    fi
    exit 0
  fi
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[ERROR] vLLM exited before readiness"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi
  sleep 1
done

echo "[ERROR] readiness timeout (${VLLM_STARTUP_TIMEOUT_S}s)"
tail -n 200 "$VLLM_LOG" || true
exit 1
