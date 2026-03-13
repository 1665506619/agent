#!/usr/bin/env bash
#SBATCH --job-name=vllm-agent-8nodes
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 目录配置：origin 输入、processed 输出、state 中间状态目录
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/InsructSAM_data_promote/sam3/agent}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
ORIGIN_DIR="${ORIGIN_DIR:-$AGENT_DATA_DIR/origin}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"

STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"
DELTA_DIR="${DELTA_DIR:-$STATE_ROOT/deltas}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$STATE_ROOT/merge_checkpoint.json}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:-/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data}"

NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-8}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-64}}"
ORIGIN_RECURSIVE="${ORIGIN_RECURSIVE:-0}"
MERGE_INTERVAL_S="${MERGE_INTERVAL_S:-60}"

JOB_ID="${SLURM_JOB_ID:-manual}"
MANIFEST_PATH="${MANIFEST_PATH:-$STATE_ROOT/manifests/manifest_${JOB_ID}.json}"

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

mkdir -p "$PROCESSED_DIR" "$DELTA_DIR" "$(dirname "$MANIFEST_PATH")" "$(dirname "$CHECKPOINT_PATH")"

echo "[INFO] job_id=${JOB_ID}"
echo "[INFO] num_nodes=${NUM_NODES}"
echo "[INFO] origin_dir=${ORIGIN_DIR}"
echo "[INFO] processed_dir=${PROCESSED_DIR}"
echo "[INFO] state_root=${STATE_ROOT}"
echo "[INFO] manifest_path=${MANIFEST_PATH}"

merge_once() {
  local active_job_id="${1:-}"
  # 将 delta 的增量结果合并回 processed，并可按策略清理已消费 delta 文件
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

start_periodic_merge() {
  # 后台周期 merge，保证长作业运行中 processed 也会持续推进
  (
    while true; do
      if ! merge_once "$JOB_ID"; then
        echo "[WARN] periodic merge failed; will retry in ${MERGE_INTERVAL_S}s"
      fi
      sleep "$MERGE_INTERVAL_S"
    done
  ) &
}

# Make sure previous WAL progress is materialized before creating a new task manifest.
merge_once "$JOB_ID"

echo "[INFO] building annotation-level manifest..."
# 根据 processed 当前状态生成“未完成 annotation 任务清单”
manifest_cmd=(
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

# Export runtime settings for node workers.
export PROJECT_ROOT
export AGENT_DIR
export AGENT_DATA_DIR
export ORIGIN_DIR
export PROCESSED_DIR
export STATE_ROOT
export DELTA_DIR
export MODEL
export IMAGE_ROOT

# Optional knobs consumed by node worker.
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

MERGER_PID=""
if [[ "$MERGE_INTERVAL_S" -gt 0 ]]; then
  echo "[INFO] starting periodic merge loop (interval=${MERGE_INTERVAL_S}s)"
  start_periodic_merge
  MERGER_PID="$!"
fi

# 拉起每节点一个 worker，按 rank 消费 manifest 里的任务
set +e
srun \
  --nodes "$NUM_NODES" \
  --ntasks "$NUM_NODES" \
  --ntasks-per-node 1 \
  --cpus-per-task "$CPUS_PER_TASK" \
  --kill-on-bad-exit=1 \
  --label \
  bash "$NODE_WORKER_SCRIPT" --manifest "$MANIFEST_PATH"
SRUN_RC=$?
set -e

if [[ -n "$MERGER_PID" ]]; then
  kill "$MERGER_PID" >/dev/null 2>&1 || true
  wait "$MERGER_PID" 2>/dev/null || true
fi

# 收尾 merge：确保本次作业 WAL 全量落盘，并清理已消费 delta
merge_once

echo "[INFO] final merge completed"
if [[ "$SRUN_RC" -ne 0 ]]; then
  echo "[ERROR] srun workers failed rc=${SRUN_RC}"
  exit "$SRUN_RC"
fi

echo "[INFO] multi-node run finished successfully"
