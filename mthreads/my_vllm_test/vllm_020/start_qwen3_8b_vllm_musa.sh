#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/SQT-v1.0.5-test/models/qwen3-8b}"
PORT="${PORT:-19000}"
GPU_IDS="${GPU_IDS:-0}"
ENABLE_LMCACHE="${ENABLE_LMCACHE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
LMCACHE_PATH="${LMCACHE_PATH:-/data/_backup_to_local/LMCache}"
LOG_DIR="${LOG_DIR:-/data/my_vllm_test/vllm_020}"

if [[ "${ENABLE_LMCACHE}" == "1" ]]; then
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-lmcache}"
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen3_8b_lmcache_${PORT}.log}"
else
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-pure}"
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen3_8b_pure_${PORT}.log}"
fi

source /root/.virtualenvs/sglang-0.5.6/bin/activate

unset PYTHONPATH
export PATH="/root/.virtualenvs/sglang-0.5.6/bin:/root/.cargo/bin:/root/.local/bin:/root/.local/bin:/driver/usr/bin:/usr/local/mtshmem/bin:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/local/lib/:/usr/local/mtshmem/lib/:/usr/local/musa/lib:/usr/local/openmpi/lib:/usr/local/musa/mudnn/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUSA_VISIBLE_DEVICES="${GPU_IDS}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"

cmd=(
  vllm serve "${MODEL_PATH}"
  --trust-remote-code
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --block-size "${BLOCK_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,30,32,50,64,100,128,256]}'
)

if [[ "${ENABLE_LMCACHE}" == "1" ]]; then
  export PYTHONPATH="${LMCACHE_PATH}:${PYTHONPATH:-}"
  export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
  export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-20}"
  export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
  export LMCACHE_USE_EXPERIMENTAL="${LMCACHE_USE_EXPERIMENTAL:-True}"
  cmd+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
fi

mkdir -p "${LOG_DIR}"
echo "ENABLE_LMCACHE=${ENABLE_LMCACHE}"
echo "GPU_IDS=${GPU_IDS}"
echo "PORT=${PORT}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "LOG_FILE=${LOG_FILE}"
echo "CMD=${cmd[*]}"

nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
echo "$!"
