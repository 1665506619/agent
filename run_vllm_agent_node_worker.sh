#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

if [[ -z "$MANIFEST_PATH" ]]; then
  echo "[ERROR] --manifest is required"
  exit 2
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[ERROR] manifest not found: $MANIFEST_PATH"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 路径配置：单节点 worker 只负责本 rank 的任务和日志
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/InsructSAM_data_promote/sam3/agent}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
ORIGIN_DIR="${ORIGIN_DIR:-$AGENT_DATA_DIR/origin}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"
STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"
NODE_RESULT_DIR="${NODE_RESULT_DIR:-$STATE_ROOT/node_results}"

MODEL="${MODEL:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/weights/Qwen3-VL-30B-A3B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:-/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-64000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

AGENT_MAX_GENERATIONS="${AGENT_MAX_GENERATIONS:-20}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
AGENT_DATASET_CACHE_SIZE="${AGENT_DATASET_CACHE_SIZE:-4}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$AGENT_DATA_DIR/../weights/sam3.pt}"
LLM_REQUEST_INTERVAL_S="${LLM_REQUEST_INTERVAL_S:-0}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-0}"
LLM_RETRY_BACKOFF_S="${LLM_RETRY_BACKOFF_S:-8}"

VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-2400}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

VLLM_BIN="${VLLM_BIN:-$PROJECT_ROOT/.venv/bin/vllm}"
AGENT_PYTHON="${AGENT_PYTHON:-python}"

NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_ROOT="${LOG_ROOT:-$STATE_ROOT/logs/job_${JOB_ID}}"
NODE_LOG_DIR="${LOG_ROOT}/node_${NODE_RANK}"

DELTA_PATH="${DELTA_PATH:-$NODE_RESULT_DIR/rank_${NODE_RANK}.jsonl}"
VLLM_LOG="${NODE_LOG_DIR}/vllm_server.log"
WORKER_LOG="${NODE_LOG_DIR}/annotation_worker.log"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$NODE_LOG_DIR" "$NODE_RESULT_DIR"

echo "[DEBUG][node ${NODE_RANK}] host=$(hostname) pid=$$"
echo "[DEBUG][node ${NODE_RANK}] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[DEBUG][node ${NODE_RANK}] vllm_bin=${VLLM_BIN}"
echo "[DEBUG][node ${NODE_RANK}] model=${MODEL}"
echo "[DEBUG][node ${NODE_RANK}] base_url=${VLLM_BASE_URL} tp=${TENSOR_PARALLEL_SIZE} dp=${DATA_PARALLEL_SIZE}"
echo "[DEBUG][node ${NODE_RANK}] engine_ready_timeout=${VLLM_ENGINE_READY_TIMEOUT_S} startup_timeout=${VLLM_STARTUP_TIMEOUT_S}"
echo "[DEBUG][node ${NODE_RANK}] hf_hub_offline=${HF_HUB_OFFLINE} transformers_offline=${TRANSFORMERS_OFFLINE}"
echo "[DEBUG][node ${NODE_RANK}] node_result_path=${DELTA_PATH}"
echo "[DEBUG][node ${NODE_RANK}] sam3_checkpoint_path=${SAM3_CHECKPOINT_PATH}"
if command -v pgrep >/dev/null 2>&1; then
  echo "[DEBUG][node ${NODE_RANK}] pre-existing vllm serve processes:"
  pgrep -af "vllm.*serve" || true
fi
if command -v ss >/dev/null 2>&1; then
  echo "[DEBUG][node ${NODE_RANK}] listeners on port ${VLLM_PORT}:"
  ss -ltnp "( sport = :${VLLM_PORT} )" || true
fi

if [[ -n "$SAM3_CHECKPOINT_PATH" && ! -f "$SAM3_CHECKPOINT_PATH" ]]; then
  echo "[ERROR][node ${NODE_RANK}] SAM3 checkpoint not found: $SAM3_CHECKPOINT_PATH"
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "[ERROR][node ${NODE_RANK}] local model directory not found: $MODEL"
  exit 2
fi
#拉起VLLM
echo "[INFO][node ${NODE_RANK}] starting vLLM on ${VLLM_BASE_URL}"
cd "$PROJECT_ROOT"
VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S" \
HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
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

echo "[INFO][node ${NODE_RANK}] waiting for vLLM readiness"
ready=0
for ((sec = 1; sec <= VLLM_STARTUP_TIMEOUT_S; sec++)); do
  # 就绪探测：OpenAI 兼容接口可用后再跑 annotation worker
  if curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[ERROR][node ${NODE_RANK}] vLLM exited before readiness"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "[ERROR][node ${NODE_RANK}] vLLM readiness timeout (${VLLM_STARTUP_TIMEOUT_S}s)"
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi

echo "[INFO][node ${NODE_RANK}] running annotation worker"
# annotation worker 负责消费本 rank 的 annotation 任务并累计写入本 rank 的结果文件
worker_cmd=(
  "$AGENT_PYTHON" "$SCRIPT_DIR/run_vllm_agent_annotation_worker.py"
  --manifest "$MANIFEST_PATH"
  --rank "$NODE_RANK"
  --origin-dir "$ORIGIN_DIR"
  --processed-dir "$PROCESSED_DIR"
  --image-root "$IMAGE_ROOT"
  --model "$MODEL"
  --base-url "$VLLM_BASE_URL"
  --max-generations "$AGENT_MAX_GENERATIONS"
  --num-workers "$AGENT_NUM_WORKERS"
  --dataset-cache-size "$AGENT_DATASET_CACHE_SIZE"
  --delta-path "$DELTA_PATH"
  --llm-request-interval-s "$LLM_REQUEST_INTERVAL_S"
  --llm-max-retries "$LLM_MAX_RETRIES"
  --llm-retry-backoff-s "$LLM_RETRY_BACKOFF_S"
  --job-id "$JOB_ID"
)

if [[ -n "$SAM3_CHECKPOINT_PATH" ]]; then
  worker_cmd+=(--sam3-checkpoint-path "$SAM3_CHECKPOINT_PATH")
fi

(
  cd "$AGENT_DIR"
  "${worker_cmd[@]}"
) >"$WORKER_LOG" 2>&1

echo "[INFO][node ${NODE_RANK}] worker completed. result=${DELTA_PATH}"
