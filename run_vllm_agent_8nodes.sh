#!/usr/bin/env bash

#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

CLI_NODE_RANK=""
CLI_NUM_NODES=""
CLI_MANIFEST_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank)
      CLI_NODE_RANK="$2"
      shift 2
      ;;
    --num-nodes)
      CLI_NUM_NODES="$2"
      shift 2
      ;;
    --manifest)
      CLI_MANIFEST_PATH="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 思路是先离线生成一次静态 manifest，
# 然后每次提交任务只负责启动 vLLM、读取静态 manifest 并完成本 rank 标注。

# 目录配置：origin 输入、processed 输出、state 中间状态目录
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/vllm_for_data}"
AGENT_DIR="${AGENT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/InsructSAM_data_promote/sam3/agent}"
AGENT_DATA_DIR="${AGENT_DATA_DIR:-$AGENT_DIR/../agent_datas}"
ORIGIN_DIR="${ORIGIN_DIR:-$AGENT_DATA_DIR/origin}"
PROCESSED_DIR="${PROCESSED_DIR:-$AGENT_DATA_DIR/processed}"

STATE_ROOT="${STATE_ROOT:-$PROCESSED_DIR/vllm_multi_node_state}"
NODE_RESULT_DIR="${NODE_RESULT_DIR:-$STATE_ROOT/node_results}"

MODEL="${MODEL:-/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/weights/Qwen3-VL-30B-A3B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:-/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data}"

NUM_NODES="${CLI_NUM_NODES:-${NUM_NODES:-8}}"
NODE_RANK="${CLI_NODE_RANK:-${NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-0}}}}"

JOB_ID="${SLURM_JOB_ID:-manual}"
MANIFEST_PATH="${CLI_MANIFEST_PATH:-${MANIFEST_PATH:-$STATE_ROOT/manifests/manifest_static_${NUM_NODES}nodes.json}}"
NODE_WORKER_SCRIPT="${NODE_WORKER_SCRIPT:-$SCRIPT_DIR/run_vllm_agent_node_worker.sh}"

#检查目录
if [[ ! -d "$ORIGIN_DIR" ]]; then
  echo "[ERROR] ORIGIN_DIR does not exist: $ORIGIN_DIR"
  exit 2
fi
if [[ ! -d "$AGENT_DIR" ]]; then
  echo "[ERROR] AGENT_DIR does not exist: $AGENT_DIR"
  exit 2
fi
if [[ ! -f "$NODE_WORKER_SCRIPT" ]]; then
  echo "[ERROR] node worker script not found: $NODE_WORKER_SCRIPT"
  exit 2
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[ERROR] static manifest not found: $MANIFEST_PATH"
  echo "[ERROR] run prepare_vllm_agent_static_manifest_once.sh first"
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

mkdir -p "$PROCESSED_DIR" "$NODE_RESULT_DIR"

echo "[INFO] job_id=${JOB_ID}"
echo "[INFO] num_nodes=${NUM_NODES}"
echo "[INFO] node_rank=${NODE_RANK}"
echo "[INFO] origin_dir=${ORIGIN_DIR}"
echo "[INFO] processed_dir=${PROCESSED_DIR}"
echo "[INFO] state_root=${STATE_ROOT}"
echo "[INFO] node_result_dir=${NODE_RESULT_DIR}"
echo "[INFO] manifest_path=${MANIFEST_PATH}"

# 这个过程包括拉起VLLM服务+执行标注并累计写入本rank结果文件
set +e
NODE_RESULT_DIR="$NODE_RESULT_DIR" \
SLURM_NODEID="$NODE_RANK" SLURM_PROCID="$NODE_RANK" SLURM_JOB_ID="$JOB_ID" \
  bash "$NODE_WORKER_SCRIPT" --manifest "$MANIFEST_PATH"
WORKER_RC=$?
set -e

if [[ "$WORKER_RC" -ne 0 ]]; then
  echo "[ERROR][rank ${NODE_RANK}] shard annotation failed rc=${WORKER_RC}"
  exit "$WORKER_RC"
fi

echo "[INFO][rank ${NODE_RANK}] shard annotation completed"
echo "[INFO][rank ${NODE_RANK}] cumulative node result file: ${NODE_RESULT_DIR}/rank_${NODE_RANK}.jsonl"
