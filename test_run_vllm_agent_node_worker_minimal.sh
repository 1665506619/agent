#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-$SCRIPT_DIR}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"
STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"

JOB_ID="${SLURM_JOB_ID:-manual}"
NODE_RANK="${SLURM_PROCID:-${SLURM_NODEID:-0}}"
MANIFEST_DIR="${MANIFEST_DIR:-$STATE_ROOT/manifests}"
MANIFEST_PATH="${MANIFEST_PATH:-$MANIFEST_DIR/manifest_node_worker_min_${JOB_ID}_${NODE_RANK}.json}"
NODE_WORKER_SCRIPT="${NODE_WORKER_SCRIPT:-$SCRIPT_DIR/run_vllm_agent_node_worker.sh}"

mkdir -p "$MANIFEST_DIR"

if [[ ! -f "$NODE_WORKER_SCRIPT" ]]; then
  echo "[ERROR] node worker script not found: $NODE_WORKER_SCRIPT"
  exit 2
fi

cat >"$MANIFEST_PATH" <<EOF
{
  "created_at_utc": "manual",
  "num_shards": 1,
  "shards": [
    {
      "rank": ${NODE_RANK},
      "num_tasks": 1,
      "tasks": [
        {
          "dataset_path": "/tmp/dummy.json",
          "sample_idx": 0,
          "annotation_idx": 0
        }
      ]
    }
  ]
}
EOF

echo "[INFO] generated manifest: ${MANIFEST_PATH}"
echo "[INFO] node_rank=${NODE_RANK}"
echo "[INFO] launching node worker directly"

SLURM_NODEID="${NODE_RANK}" SLURM_PROCID="${NODE_RANK}" SLURM_JOB_ID="${JOB_ID}" \
  bash "$NODE_WORKER_SCRIPT" --manifest "$MANIFEST_PATH"
