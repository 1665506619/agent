#!/usr/bin/env bash
#SBATCH --job-name=vllm-agent-1node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# ------------------------
# User-configurable values
# ------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/InsructSAM_data_promote/sam3/agent}"
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

AGENT_MAX_GENERATIONS="${AGENT_MAX_GENERATIONS:-4}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"

VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-600}"

# Any extra args passed to this script are forwarded to vllm_agent.py
EXTRA_AGENT_ARGS=("$@")


if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl is required for readiness check but was not found."
  exit 1
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: $PROJECT_ROOT"
  exit 1
fi

if [[ ! -d "$AGENT_DIR" ]]; then
  echo "[ERROR] AGENT_DIR does not exist: $AGENT_DIR"
  exit 1
fi

VLLM_LOG="${PROJECT_ROOT}/vllm_server_${SLURM_JOB_ID:-manual}.log"
AGENT_LOG="${PROJECT_ROOT}/vllm_agent_${SLURM_JOB_ID:-manual}.log"

VLLM_PID=""

cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[INFO] Stopping vLLM server (pid=${VLLM_PID})"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] PROJECT_ROOT: ${PROJECT_ROOT}"
echo "[INFO] AGENT_DIR: ${AGENT_DIR}"
echo "[INFO] MODEL: ${MODEL}"
echo "[INFO] vLLM base URL: ${VLLM_BASE_URL}"

echo "[INFO] Starting vLLM server..."
cd "$PROJECT_ROOT"
export VLLM_ENGINE_READY_TIMEOUT_S=1200
.venv/bin/vllm serve "$MODEL" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM pid: ${VLLM_PID}, log: ${VLLM_LOG}"

echo "[INFO] Waiting for vLLM to become ready..."
ready=0
for ((sec = 1; sec <= VLLM_STARTUP_TIMEOUT_S; sec++)); do
  if curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    ready=1
    break
  fi

  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[ERROR] vLLM process exited before readiness."
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi

  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "[ERROR] vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT_S}s"
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi

echo "[INFO] vLLM is ready. Running vllm_agent.py..."
cd "$AGENT_DIR"

python vllm_agent.py \
  --model "$MODEL" \
  --max-generations "$AGENT_MAX_GENERATIONS" \
  --base-url "$VLLM_BASE_URL" \
  --num-workers "$AGENT_NUM_WORKERS" \
  "${EXTRA_AGENT_ARGS[@]}" \
  >"$AGENT_LOG" 2>&1 || AGENT_EXIT_CODE=$?

AGENT_EXIT_CODE="${AGENT_EXIT_CODE:-0}"
if [[ "$AGENT_EXIT_CODE" -ne 0 ]]; then
  echo "[ERROR] vllm_agent.py failed, exit code: ${AGENT_EXIT_CODE}"
  tail -n 200 "$AGENT_LOG" || true
  exit "$AGENT_EXIT_CODE"
fi

echo "[INFO] vllm_agent.py finished successfully."
echo "[INFO] Agent log: $AGENT_LOG"
echo "[INFO] vLLM log:  $VLLM_LOG"
