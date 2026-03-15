#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_vllm_agent_8nodes.sh}"
NUM_RANKS="${NUM_RANKS:-8}"
NUM_NODES="${NUM_NODES:-8}"
SUBMIT_INTERVAL_HOURS="${SUBMIT_INTERVAL_HOURS:-5}"
LOGROOT="${LOGROOT:-./slurm_logs/0223}"
JOB_NAME="${JOB_NAME:-finetune}"
IMAGE="${IMAGE:-/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/smajumdar/region/data/data_promote/qwen_copy.sqsh}"
DURATION="${DURATION:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
TASKS_PER_NODE="${TASKS_PER_NODE:-1}"
DEPENDENT_CLONES="${DEPENDENT_CLONES:-0}"
EMAIL_MODE="${EMAIL_MODE:-never}"
ONCE="${ONCE:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Submit 8 single-node vLLM agent jobs every 5 hours by default.

Options:
  --once                  Submit one batch only, then exit.
  --interval-hours H      Hours between batches. Default: ${SUBMIT_INTERVAL_HOURS}
  --num-ranks N           Number of rank jobs per batch. Default: ${NUM_RANKS}
  --num-nodes N           Value forwarded to run_vllm_agent_8nodes.sh --num-nodes. Default: ${NUM_NODES}
  --logroot PATH          submit_job --logroot value. Default: ${LOGROOT}
  --job-name NAME         submit_job -n value. Default: ${JOB_NAME}
  --image PATH            submit_job --image value.
  --duration HOURS        submit_job --duration value. Default: ${DURATION}
  --run-script PATH       Entry script to launch for each rank. Default: ${RUN_SCRIPT}
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE=1
      shift
      ;;
    --interval-hours)
      SUBMIT_INTERVAL_HOURS="$2"
      shift 2
      ;;
    --num-ranks)
      NUM_RANKS="$2"
      shift 2
      ;;
    --num-nodes)
      NUM_NODES="$2"
      shift 2
      ;;
    --logroot)
      LOGROOT="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --run-script)
      RUN_SCRIPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if ! command -v submit_job >/dev/null 2>&1; then
  echo "[ERROR] submit_job not found in PATH"
  exit 2
fi
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[ERROR] run script not found: $RUN_SCRIPT"
  exit 2
fi
if ! [[ "$NUM_RANKS" =~ ^[0-9]+$ ]] || [[ "$NUM_RANKS" -lt 1 ]]; then
  echo "[ERROR] --num-ranks must be a positive integer, got: ${NUM_RANKS}"
  exit 2
fi
if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]] || [[ "$NUM_NODES" -lt 1 ]]; then
  echo "[ERROR] --num-nodes must be a positive integer, got: ${NUM_NODES}"
  exit 2
fi
if ! [[ "$SUBMIT_INTERVAL_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] --interval-hours must be numeric, got: ${SUBMIT_INTERVAL_HOURS}"
  exit 2
fi

RUN_SCRIPT="$(realpath "$RUN_SCRIPT")"
mkdir -p "$LOGROOT"

submit_one_rank() {
  local rank="$1"
  local launch_cmd
  launch_cmd="bash $RUN_SCRIPT --rank $rank --num-nodes $NUM_NODES"

  echo "[INFO] submitting rank=${rank}"
  echo "[INFO] command=${launch_cmd}"

  submit_job \
    --gpu "$GPU_COUNT" \
    --tasks_per_node "$TASKS_PER_NODE" \
    --nodes 1 \
    -n "$JOB_NAME" \
    --image="$IMAGE" \
    --logroot "$LOGROOT" \
    --email_mode "$EMAIL_MODE" \
    --duration "$DURATION" \
    --dependent_clones "$DEPENDENT_CLONES" \
    -c "$launch_cmd"
}

submit_batch() {
  local batch_id="$1"
  echo "[INFO] ===== batch ${batch_id} start: $(date '+%F %T') ====="
  for ((rank = 0; rank < NUM_RANKS; rank++)); do
    submit_one_rank "$rank"
  done
  echo "[INFO] ===== batch ${batch_id} submitted ====="
}

batch_id=1
while true; do
  submit_batch "$batch_id"
  if [[ "$ONCE" == "1" ]]; then
    break
  fi

  echo "[INFO] sleeping ${SUBMIT_INTERVAL_HOURS}h before next batch"
  sleep "${SUBMIT_INTERVAL_HOURS}h"
  batch_id=$((batch_id + 1))
done
