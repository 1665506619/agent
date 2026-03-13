#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-$SCRIPT_DIR}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
ORIGIN_DIR="${ORIGIN_DIR:-$AGENT_DATA_DIR/origin}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"

STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"
DELTA_DIR="${DELTA_DIR:-$STATE_ROOT/deltas}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$STATE_ROOT/merge_checkpoint.json}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:-/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data}"

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-0}}}"
ORIGIN_RECURSIVE="${ORIGIN_RECURSIVE:-0}"
MERGE_INTERVAL_S="${MERGE_INTERVAL_S:-60}"

JOB_ID="${SLURM_JOB_ID:-manual}"
MANIFEST_PATH="${MANIFEST_PATH:-$STATE_ROOT/manifests/manifest_node_worker_min_${JOB_ID}_${NODE_RANK}.json}"
SYNC_DIR="${SYNC_DIR:-$STATE_ROOT/sync/job_${JOB_ID}}"

MANIFEST_PYTHON="${MANIFEST_PYTHON:-python}"
SHARD_BUILDER_SCRIPT="${SHARD_BUILDER_SCRIPT:-$SCRIPT_DIR/build_vllm_agent_file_shards.py}"
NODE_WORKER_SCRIPT="${NODE_WORKER_SCRIPT:-$SCRIPT_DIR/run_vllm_agent_node_worker.sh}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$SCRIPT_DIR/apply_vllm_agent_deltas.py}"

if [[ ! -d "$ORIGIN_DIR" ]]; then
  echo "[ERROR] ORIGIN_DIR does not exist: $ORIGIN_DIR"
  exit 2
fi
if [[ ! -d "$AGENT_DIR" ]]; then
  echo "[ERROR] AGENT_DIR does not exist: $AGENT_DIR"
  exit 2
fi
if [[ ! -f "$SHARD_BUILDER_SCRIPT" ]]; then
  echo "[ERROR] shard builder script not found: $SHARD_BUILDER_SCRIPT"
  exit 2
fi
if [[ ! -f "$NODE_WORKER_SCRIPT" ]]; then
  echo "[ERROR] node worker script not found: $NODE_WORKER_SCRIPT"
  exit 2
fi
if [[ ! -f "$MERGE_SCRIPT" ]]; then
  echo "[ERROR] merge script not found: $MERGE_SCRIPT"
  exit 2
fi

mkdir -p "$PROCESSED_DIR" "$DELTA_DIR" "$(dirname "$MANIFEST_PATH")" "$(dirname "$CHECKPOINT_PATH")" "$SYNC_DIR"

echo "[INFO] job_id=${JOB_ID}"
echo "[INFO] num_nodes=${NUM_NODES}"
echo "[INFO] node_rank=${NODE_RANK}"
echo "[INFO] origin_dir=${ORIGIN_DIR}"
echo "[INFO] processed_dir=${PROCESSED_DIR}"
echo "[INFO] state_root=${STATE_ROOT}"
echo "[INFO] manifest_path=${MANIFEST_PATH}"
echo "[INFO] sync_dir=${SYNC_DIR}"

merge_once() {
  local active_job_id="${1:-}"
  local merge_cmd=(
    "$MANIFEST_PYTHON" "$MERGE_SCRIPT"
    --origin-dir "$ORIGIN_DIR"
    --processed-dir "$PROCESSED_DIR"
    --delta-dir "$DELTA_DIR"
    --checkpoint-path "$CHECKPOINT_PATH"
    --truncate-consumed
  )
  if [[ -n "$active_job_id" ]]; then
    merge_cmd+=(--active-job-id "$active_job_id")
  fi
  "${merge_cmd[@]}"
}

build_manifest() {
  echo "[INFO][rank ${NODE_RANK}] building annotation-level manifest..."
  local manifest_cmd=(
    "$MANIFEST_PYTHON" "$SHARD_BUILDER_SCRIPT"
    --origin-dir "$ORIGIN_DIR"
    --processed-dir "$PROCESSED_DIR"
    --num-shards "$NUM_NODES"
    --output "$MANIFEST_PATH"
  )
  if [[ "$ORIGIN_RECURSIVE" == "1" ]]; then
    manifest_cmd+=(--recursive)
  fi
  "${manifest_cmd[@]}"
}

start_periodic_merge() {
  (
    while true; do
      if ! merge_once "$JOB_ID"; then
        echo "[WARN][rank ${NODE_RANK}] periodic merge failed; will retry in ${MERGE_INTERVAL_S}s"
      fi
      sleep "$MERGE_INTERVAL_S"
    done
  ) &
}

export PROJECT_ROOT
export AGENT_DIR
export AGENT_DATA_DIR
export ORIGIN_DIR
export PROCESSED_DIR
export STATE_ROOT
export DELTA_DIR
export MODEL
export IMAGE_ROOT

export VLLM_BIN="${VLLM_BIN:-}"
export AGENT_PYTHON="${AGENT_PYTHON:-}"
export VLLM_HOST="${VLLM_HOST:-}"
export VLLM_PORT="${VLLM_PORT:-}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
export AGENT_MAX_GENERATIONS="${AGENT_MAX_GENERATIONS:-}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-}"
export AGENT_DATASET_CACHE_SIZE="${AGENT_DATASET_CACHE_SIZE:-}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}"
export LLM_REQUEST_INTERVAL_S="${LLM_REQUEST_INTERVAL_S:-}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-}"
export LLM_RETRY_BACKOFF_S="${LLM_RETRY_BACKOFF_S:-}"
export VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-}"
export LOG_ROOT="${LOG_ROOT:-}"

merge_once "$JOB_ID"
build_manifest

MERGER_PID=""
if [[ "$MERGE_INTERVAL_S" -gt 0 ]]; then
  echo "[INFO][rank ${NODE_RANK}] starting periodic merge loop (interval=${MERGE_INTERVAL_S}s)"
  start_periodic_merge
  MERGER_PID="$!"
fi

cleanup() {
  if [[ -n "$MERGER_PID" ]]; then
    kill "$MERGER_PID" >/dev/null 2>&1 || true
    wait "$MERGER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO][rank ${NODE_RANK}] launching node worker directly"
SLURM_NODEID="${NODE_RANK}" SLURM_PROCID="${NODE_RANK}" SLURM_JOB_ID="${JOB_ID}" \
  bash "$NODE_WORKER_SCRIPT" --manifest "$MANIFEST_PATH"
