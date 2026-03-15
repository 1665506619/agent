#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT_DIR="${AGENT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/InsructSAM_data_promote/sam3/agent}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
ORIGIN_DIR="${ORIGIN_DIR:-$AGENT_DATA_DIR/origin}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"
STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"
NODE_RESULT_DIR="${NODE_RESULT_DIR:-$STATE_ROOT/node_results}"

MERGE_PYTHON="${MERGE_PYTHON:-python}"
FINAL_MERGE_SCRIPT="${FINAL_MERGE_SCRIPT:-$SCRIPT_DIR/merge_vllm_agent_node_results_once.py}"

if [[ ! -d "$ORIGIN_DIR" ]]; then
  echo "[ERROR] ORIGIN_DIR does not exist: $ORIGIN_DIR"
  exit 2
fi
if [[ ! -d "$NODE_RESULT_DIR" ]]; then
  echo "[ERROR] NODE_RESULT_DIR does not exist: $NODE_RESULT_DIR"
  exit 2
fi
if [[ ! -f "$FINAL_MERGE_SCRIPT" ]]; then
  echo "[ERROR] final merge script not found: $FINAL_MERGE_SCRIPT"
  exit 2
fi

echo "[INFO] final merge"
echo "[INFO] origin_dir=${ORIGIN_DIR}"
echo "[INFO] processed_dir=${PROCESSED_DIR}"
echo "[INFO] node_result_dir=${NODE_RESULT_DIR}"

"$MERGE_PYTHON" "$FINAL_MERGE_SCRIPT" \
  --origin-dir "$ORIGIN_DIR" \
  --processed-dir "$PROCESSED_DIR" \
  --results-dir "$NODE_RESULT_DIR"
