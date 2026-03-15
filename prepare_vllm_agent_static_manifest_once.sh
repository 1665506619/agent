#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT_DIR="${AGENT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/InsructSAM_data_promote/sam3/agent}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
ORIGIN_DIR="${ORIGIN_DIR:-$AGENT_DATA_DIR/origin}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"
STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"

NUM_SHARDS="${NUM_SHARDS:-8}"
ORIGIN_RECURSIVE="${ORIGIN_RECURSIVE:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

MANIFEST_PATH="${MANIFEST_PATH:-$STATE_ROOT/manifests/manifest_static_${NUM_SHARDS}nodes.json}"
MANIFEST_PYTHON="${MANIFEST_PYTHON:-python}"
SHARD_BUILDER_SCRIPT="${SHARD_BUILDER_SCRIPT:-$SCRIPT_DIR/build_vllm_agent_file_shards.py}"

if [[ ! -d "$ORIGIN_DIR" ]]; then
  echo "[ERROR] ORIGIN_DIR does not exist: $ORIGIN_DIR"
  exit 2
fi
if [[ ! -f "$SHARD_BUILDER_SCRIPT" ]]; then
  echo "[ERROR] shard builder script not found: $SHARD_BUILDER_SCRIPT"
  exit 2
fi
if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "[ERROR] NUM_SHARDS must be a positive integer, got: ${NUM_SHARDS}"
  exit 2
fi

mkdir -p "$PROCESSED_DIR" "$(dirname "$MANIFEST_PATH")"

if [[ -f "$MANIFEST_PATH" && "$FORCE_REBUILD" != "1" ]]; then
  echo "[INFO] manifest already exists: $MANIFEST_PATH"
  echo "[INFO] set FORCE_REBUILD=1 to regenerate it"
  exit 0
fi

manifest_cmd=(
  "$MANIFEST_PYTHON" "$SHARD_BUILDER_SCRIPT"
  --origin-dir "$ORIGIN_DIR"
  --processed-dir "$PROCESSED_DIR"
  --num-shards "$NUM_SHARDS"
  --output "$MANIFEST_PATH"
)
if [[ "$ORIGIN_RECURSIVE" == "1" ]]; then
  manifest_cmd+=(--recursive)
fi

echo "[INFO] generating static manifest"
echo "[INFO] origin_dir=${ORIGIN_DIR}"
echo "[INFO] processed_dir=${PROCESSED_DIR}"
echo "[INFO] num_shards=${NUM_SHARDS}"
echo "[INFO] manifest_path=${MANIFEST_PATH}"
"${manifest_cmd[@]}"
