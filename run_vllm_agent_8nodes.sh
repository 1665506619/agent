#!/usr/bin/env bash

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

NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-0}}}"
ORIGIN_RECURSIVE="${ORIGIN_RECURSIVE:-0}"
MERGE_INTERVAL_S="${MERGE_INTERVAL_S:-120}"
MANIFEST_WAIT_TIMEOUT_S="${MANIFEST_WAIT_TIMEOUT_S:-1800}"
WORKER_WAIT_TIMEOUT_S="${WORKER_WAIT_TIMEOUT_S:-86400}"

JOB_ID="${SLURM_JOB_ID:-manual}"
MANIFEST_PATH="${MANIFEST_PATH:-$STATE_ROOT/manifests/manifest_${JOB_ID}.json}"
SYNC_DIR="${SYNC_DIR:-$STATE_ROOT/sync/job_${JOB_ID}}"
MANIFEST_READY_PATH="$SYNC_DIR/manifest.ready"
MANIFEST_FAILED_PATH="$SYNC_DIR/manifest.failed"

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
if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]] || [[ "$NUM_NODES" -lt 1 ]]; then
  echo "[ERROR] NUM_NODES must be a positive integer, got: ${NUM_NODES}"
  exit 2
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [[ "$NODE_RANK" -lt 0 ]]; then
  echo "[ERROR] NODE_RANK must be a non-negative integer, got: ${NODE_RANK}"
  exit 2
fi
if [[ "$NODE_RANK" -ge "$NUM_NODES" ]]; then
  echo "[ERROR] NODE_RANK=${NODE_RANK} out of range for NUM_NODES=${NUM_NODES}"
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

wait_for_manifest() {
  local waited=0
  while [[ "$waited" -lt "$MANIFEST_WAIT_TIMEOUT_S" ]]; do
    if [[ -f "$MANIFEST_FAILED_PATH" ]]; then
      echo "[ERROR][rank ${NODE_RANK}] manifest preparation failed: $(cat "$MANIFEST_FAILED_PATH" 2>/dev/null || true)"
      return 1
    fi
    if [[ -f "$MANIFEST_READY_PATH" && -f "$MANIFEST_PATH" ]]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[ERROR][rank ${NODE_RANK}] timeout waiting for manifest (${MANIFEST_WAIT_TIMEOUT_S}s): $MANIFEST_PATH"
  return 1
}

worker_done_file() {
  local rank="$1"
  echo "$SYNC_DIR/worker_rank_${rank}.done"
}

worker_failed_file() {
  local rank="$1"
  echo "$SYNC_DIR/worker_rank_${rank}.failed"
}

mark_worker_status() {
  local rank="$1"
  local rc="$2"
  rm -f "$(worker_done_file "$rank")" "$(worker_failed_file "$rank")"
  if [[ "$rc" -eq 0 ]]; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$(worker_done_file "$rank")"
  else
    printf "rc=%s time=%s\n" "$rc" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$(worker_failed_file "$rank")"
  fi
}

wait_for_all_worker_statuses() {
  local waited=0
  while [[ "$waited" -lt "$WORKER_WAIT_TIMEOUT_S" ]]; do
    local done_count
    local fail_count
    done_count=$(find "$SYNC_DIR" -maxdepth 1 -type f -name 'worker_rank_*.done' | wc -l)
    fail_count=$(find "$SYNC_DIR" -maxdepth 1 -type f -name 'worker_rank_*.failed' | wc -l)
    local total=$((done_count + fail_count))

    if [[ "$total" -ge "$NUM_NODES" ]]; then
      echo "[INFO][rank 0] worker status collected: done=${done_count}, failed=${fail_count}, expected=${NUM_NODES}"
      if [[ "$fail_count" -gt 0 ]]; then
        return 1
      fi
      return 0
    fi

    sleep 2
    waited=$((waited + 2))
  done

  echo "[ERROR][rank 0] timeout waiting worker status files (${WORKER_WAIT_TIMEOUT_S}s)"
  return 1
}

MERGER_PID=""
if [[ "$NODE_RANK" -eq 0 ]]; then
  rm -f "$MANIFEST_READY_PATH" "$MANIFEST_FAILED_PATH"
  rm -f "$SYNC_DIR"/worker_rank_*.done "$SYNC_DIR"/worker_rank_*.failed

  if ! merge_once "$JOB_ID"; then
    echo "pre_merge_failed" > "$MANIFEST_FAILED_PATH"
    echo "[ERROR][rank 0] pre-merge failed"
    exit 1
  fi
  if ! build_manifest; then
    echo "build_manifest_failed" > "$MANIFEST_FAILED_PATH"
    echo "[ERROR][rank 0] build manifest failed"
    exit 1
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MANIFEST_READY_PATH"
  if [[ "$MERGE_INTERVAL_S" -gt 0 ]]; then
    echo "[INFO][rank 0] starting periodic merge loop (interval=${MERGE_INTERVAL_S}s)"
    start_periodic_merge
    MERGER_PID="$!"
  fi
else
  if ! wait_for_manifest; then
    mark_worker_status "$NODE_RANK" 1
    exit 1
  fi
fi

set +e
SLURM_NODEID="$NODE_RANK" SLURM_PROCID="$NODE_RANK" SLURM_JOB_ID="$JOB_ID" \
  bash "$NODE_WORKER_SCRIPT" --manifest "$MANIFEST_PATH"
WORKER_RC=$?
set -e

mark_worker_status "$NODE_RANK" "$WORKER_RC"

if [[ "$NODE_RANK" -ne 0 ]]; then
  if [[ "$WORKER_RC" -ne 0 ]]; then
    echo "[ERROR][rank ${NODE_RANK}] worker failed rc=${WORKER_RC}"
    exit "$WORKER_RC"
  fi
  echo "[INFO][rank ${NODE_RANK}] worker completed"
  exit 0
fi

OVERALL_RC=0
if ! wait_for_all_worker_statuses; then
  OVERALL_RC=1
fi

if [[ -n "$MERGER_PID" ]]; then
  kill "$MERGER_PID" >/dev/null 2>&1 || true
  wait "$MERGER_PID" 2>/dev/null || true
fi

if ! merge_once; then
  echo "[ERROR][rank 0] final merge failed"
  OVERALL_RC=1
else
  echo "[INFO][rank 0] final merge completed"
fi

if [[ "$OVERALL_RC" -ne 0 ]]; then
  echo "[ERROR][rank 0] multi-node run failed"
  exit "$OVERALL_RC"
fi

echo "[INFO][rank 0] multi-node run finished successfully"
