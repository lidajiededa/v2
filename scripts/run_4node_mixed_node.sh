#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_4node_mixed_node.sh MODEL_PATH MASTER_IP NODE_RANK NODE_IP [PORT]

Examples:
  # node0, API server node
  bash scripts/run_4node_mixed_node.sh /data/openpangu-505B 10.0.0.1 0 10.0.0.1 8000

  # node1..node3, headless DP workers
  bash scripts/run_4node_mixed_node.sh /data/openpangu-505B 10.0.0.1 1 10.0.0.2 8000
  bash scripts/run_4node_mixed_node.sh /data/openpangu-505B 10.0.0.1 2 10.0.0.3 8000
  bash scripts/run_4node_mixed_node.sh /data/openpangu-505B 10.0.0.1 3 10.0.0.4 8000

Run one vllm serve process on every node. NODE_RANK must be 0..3.
USAGE
  exit 2
fi

MODEL_PATH="$1"
MASTER_IP="$2"
NODE_RANK="$3"
NODE_IP="$4"
PORT="${5:-8000}"

DP_SIZE="${DP_SIZE:-4}"
DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-1}"
TP_SIZE="${TP_SIZE:-8}"
API_SERVER_COUNT="${API_SERVER_COUNT:-4}"
RPC_PORT="${RPC_PORT:-13389}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
ASCEND_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICES}"
export HCCL_IF_IP="${NODE_IP}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,pangu_v2_moe}"
export VLLM_PANGU_V2_MODEL_IMPL="${VLLM_PANGU_V2_MODEL_IMPL:-auto}"
export VLLM_PANGU_V2_KV_DEBUG="${VLLM_PANGU_V2_KV_DEBUG:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-0}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}"
export CUSTOM_MODEL_CONFIG_PATH="${CUSTOM_MODEL_CONFIG_PATH:-low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json}"

COMMON_ARGS=(
  "${MODEL_PATH}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --data-parallel-size "${DP_SIZE}"
  --data-parallel-size-local "${DP_LOCAL_SIZE}"
  --data-parallel-start-rank "${NODE_RANK}"
  --data-parallel-address "${MASTER_IP}"
  --data-parallel-rpc-port "${RPC_PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --distributed-executor-backend mp
  --enable-expert-parallel
  --no-disable-hybrid-kv-cache-manager
  --no-enable-prefix-caching
  --block-size 128
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enforce-eager
)

if [[ "${NODE_RANK}" == "0" ]]; then
  exec vllm serve "${COMMON_ARGS[@]}" --api-server-count "${API_SERVER_COUNT}"
fi

exec vllm serve "${COMMON_ARGS[@]}" --headless
