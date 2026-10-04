#!/usr/bin/env bash
# 用法:  bash run.sh [模型路径] [TP] [max_model_len] [port]
# 默认: Qwen2.5-7B, TP=8, max_len=32768, port=8001（避开 workshop 占的 8000）
#
# 示例（node165 推荐 - 共占 GPU）:
#   bash run.sh                                                  # 全部默认
#   bash run.sh /data/SETS-2.0-test/models/Qwen2.5-7B 8 32768 8001
#   DEBUG=1 bash run.sh                                          # 开 MCCL debug
#
# 显存预算（node165 上有 workshop 占了 56GB/卡，每卡剩 ~24GB）:
#   gpu-memory-utilization=0.15 → vllm 用 12GB/卡（权重 2GB + KV 10GB），勉强够
#   gpu-memory-utilization=0.12 → vllm 用 9.6GB/卡（权重 2GB + KV 7.6GB），稳

set -euo pipefail

MODEL_PATH=${1:-/data/SETS-2.0-test/models/Qwen2.5-14B}   # 14B (40 heads) 兼容 TP=8；7B (28 heads) 不兼容
TP=${2:-8}
MAX_MODEL_LEN=${3:-32768}
PORT=${4:-8001}    # 避开 workshop 的 8000

# 检测模型路径存在
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] 模型路径不存在: $MODEL_PATH"
  echo "可用模型: $(ls /data/SETS-2.0-test/models/ 2>/dev/null | tr '\n' ' ')"
  exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# ============== 测试模式预设（一键切换"死锁模式"和"workshop 模式"）==============
# MODE=our      → 复现 bug 的配置（默认，触发死锁条件）
# MODE=workshop → workshop 验证可跑的配置
# MODE=minimal  → 最小化所有特性（除了 TP），用于二分定位
MODE=${MODE:-our}
case "$MODE" in
  our)
    # 我们之前的死锁配置：开 prefix-caching，不设 custom_allreduce
    : ${ENABLE_PREFIX_CACHING:=1}
    : ${USE_CUSTOM_ALLREDUCE:=0}
    : ${GPU_MEM_UTIL:=0.12}
    : ${MAX_NUM_SEQS:=64}
    : ${ENFORCE_EAGER:=0}
    ;;
  workshop)
    # workshop 验证可跑：关 prefix-caching，开 USE_CUSTOM_ALLREDUCE=1
    : ${ENABLE_PREFIX_CACHING:=0}
    : ${USE_CUSTOM_ALLREDUCE:=1}
    : ${GPU_MEM_UTIL:=0.12}
    : ${MAX_NUM_SEQS:=64}
    : ${ENFORCE_EAGER:=0}
    ;;
  minimal)
    # 最小化：关一切特性，只留 TP
    : ${ENABLE_PREFIX_CACHING:=0}
    : ${USE_CUSTOM_ALLREDUCE:=0}
    : ${GPU_MEM_UTIL:=0.12}
    : ${MAX_NUM_SEQS:=64}
    : ${ENFORCE_EAGER:=1}    # 关 CUDA graph
    ;;
  *)
    echo "[ERROR] 未知 MODE: $MODE  (可选: our / workshop / minimal)"; exit 1
    ;;
esac
echo "[MODE=$MODE]  prefix_caching=$ENABLE_PREFIX_CACHING  custom_allreduce=$USE_CUSTOM_ALLREDUCE  gpu_util=$GPU_MEM_UTIL  max_seqs=$MAX_NUM_SEQS  eager=$ENFORCE_EAGER"

# ============== vllm_musa 必需环境变量 ==============
export VLLM_USE_V1=${VLLM_USE_V1:-0}     # V0 默认;`VLLM_USE_V1=1 bash run.sh` 切 V1 引擎
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # 允许超出模型默认 max_position
export HF_ENDPOINT=https://hf-mirror.com # HF 镜像（本地路径用不到，但有些 tokenizer 会探测）
# export MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 指定卡，不设默认 all

# 应用 USE_CUSTOM_ALLREDUCE（如果 MODE 开启）
if [ "$USE_CUSTOM_ALLREDUCE" = "1" ]; then
  export USE_CUSTOM_ALLREDUCE=1
  echo "[ENV] USE_CUSTOM_ALLREDUCE=1  (vllm_musa 走 custom path，旁路 MCCL allreduce)"
fi

# ============== Debug 输出（排查 MCCL 死锁用，会让日志暴涨）==============
# 用法：跑前 export DEBUG=1，或在命令行 DEBUG=1 bash run.sh ...
#
# 注意：TORCH_DISTRIBUTED_DEBUG=DETAIL 会启用严格的 tensor 检查，
# 但 vllm/vllm_musa 在 inference_mode 外部对 inference tensor 做 inplace 操作，
# DETAIL 模式会把这个抓出来直接报错（profile_run 阶段就挂），所以不要开。
# DEBUG=1 只开 MCCL 日志（够用），不开 torch distributed DETAIL。
if [ "${DEBUG:-0}" = "1" ]; then
  # MCCL 通信日志（这是定位死锁的核心，会打印每个 collective 的参数）
  export MCCL_DEBUG=INFO
  export MCCL_DEBUG_SUBSYS=COLL,INIT,NET
  # vllm engine 日志（更详细的调度信息）
  export VLLM_LOGGING_LEVEL=DEBUG
  echo "[DEBUG] MCCL_DEBUG=INFO  VLLM_LOGGING_LEVEL=DEBUG"
fi
# 想看 torch 分布式底层调用细节时单独开（会让 vllm_musa profile_run 直接 crash，不能常开）
if [ "${DEBUG_TORCH:-0}" = "1" ]; then
  export TORCH_DISTRIBUTED_DEBUG=INFO   # 用 INFO 而非 DETAIL 避免 inference tensor 检查
  export TORCH_CPP_LOG_LEVEL=INFO
  echo "[DEBUG_TORCH] TORCH_DISTRIBUTED_DEBUG=INFO"
fi
# MUSA 同步阻塞（让 GPU 错误变同步、stack 准确，但慢 30%+）
if [ "${MUSA_BLOCKING:-0}" = "1" ]; then
  export MUSA_LAUNCH_BLOCKING=1
  echo "[MUSA_BLOCKING] MUSA_LAUNCH_BLOCKING=1"
fi
# 死锁监测（任意 collective 超过 N 秒未完成自动报错并打印 stack）
# 默认开 5 分钟，能复现死锁也不会真等到永远
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TIMEOUT_MS=${TORCH_NCCL_TIMEOUT_MS:-60000}    # 1 分钟

# ============== 拼装 vllm serve 参数（按 MODE 动态加）==============
VLLM_ARGS=(
  "${MODEL_PATH}"
  --host 0.0.0.0
  --port "${PORT}"
  --served-model-name "${MODEL_NAME}"
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --pipeline-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --block-size "${BLOCK_SIZE:-64}"   # V0 用 64,V1 必须用 32(VLLM_USE_V1=1 时一定要 BLOCK_SIZE=32)
)

# 条件参数：prefix-caching
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
  VLLM_ARGS+=( --enable-prefix-caching )
fi

# Bug 2 修复(2026-05-27):Qwen2.5 的 generation_config.json 漏了 <|im_end|>(151645)作为 eos,
# 只配了 <|endoftext|>(151643),导致模型生成完 <|im_end|> 不停、跑出 turn 边界吐假对话。
# 这里覆盖 eos_token_id 把两个都加上。FIX_EOS=0 可关掉(对照测试用)。
if [ "${FIX_EOS:-1}" = "1" ]; then
  VLLM_ARGS+=( --override-generation-config '{"eos_token_id": [151645, 151643]}' )
fi

# 条件参数：CUDA Graph / eager
if [ "$ENFORCE_EAGER" = "1" ]; then
  VLLM_ARGS+=( --enforce-eager )
else
  VLLM_ARGS+=( --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,24,32,48,64], "simple_cuda_graph": true}' )
fi

# ============== 启动 vllm serve ==============
set -x
vllm serve "${VLLM_ARGS[@]}"
set +x

# ============== 三种 MODE 配置对比 ==============
#
# our (默认): 我们之前死锁的配置 → 用于复现 bug
#   ENABLE_PREFIX_CACHING=1  USE_CUSTOM_ALLREDUCE=0  ENFORCE_EAGER=0
#   CUDA Graph 开、prefix cache 开，强制走 MCCL allreduce
#
# workshop: 复刻 workshop 跑通的配置 → 验证新镜像 + workshop 配置能跑通
#   ENABLE_PREFIX_CACHING=0  USE_CUSTOM_ALLREDUCE=1  ENFORCE_EAGER=0
#   关 prefix cache，开 vllm_musa 的 custom_allreduce（旁路 MCCL allreduce）
#
# minimal: 最小化所有特性 → 用于二分定位
#   ENABLE_PREFIX_CACHING=0  USE_CUSTOM_ALLREDUCE=0  ENFORCE_EAGER=1
#   关 CUDA Graph、关 prefix cache、不开 custom allreduce
#
# 用法:
#   bash run.sh                                # 默认 MODE=our
#   MODE=workshop bash run.sh                  # 切到 workshop 配置
#   MODE=minimal bash run.sh                   # 最小化配置
#   MODE=our ENABLE_PREFIX_CACHING=0 bash run.sh  # 临时单独覆盖某项

# ============== 共占 GPU 注意事项 ==============
# 1. workshop 容器在 port 8000，我们走 8001
# 2. workshop 已用 ~56GB/卡 (gpu-mem-util=0.7)，我们最多用 0.15-0.20
# 3. 三种 MODE 的差异请看上面的预设
#
# 没开的优化（按需自行追加）:
# --enable-chunked-prefill   长 prompt prefill/decode 重叠（V0 兼容性差，慎用）
# --kv-cache-dtype fp8       KV 量化到 FP8（vllm_musa 上需先验证）
# --quantization fp8         权重 FP8（仅适配 FP8 模型如 Qwen3-235B-A22B-FP8）
# --speculative-config '{...}'  投机解码，需要小草稿模型
#
# ============== 调试模式 ==============
# 1) 普通跑：     bash run.sh /data/SETS/models/qwen3-8b 8 32768
# 2) 加 MCCL 日志:DEBUG=1 bash run.sh ...
# 3) 加 torch 日志:DEBUG_TORCH=1 bash run.sh ... （和 DEBUG=1 可叠加）
# 4) 加 MUSA 同步:MUSA_BLOCKING=1 bash run.sh ... （GPU 错误同步化，慢但精准）
# 5) 关 cudagraph: 手动注释 --compilation-config 那一行
# 6) 关 prefix cache: 手动注释 --enable-prefix-caching 那一行
#
# Debug 模式下的关键日志位置：
#   server.log 里搜：
#     'MCCL INFO'           - 每个 collective 调用参数（count、dtype、buffer addr）
#     'TORCH_NCCL_TIMEOUT'  - 死锁触发的超时报错（自动开了 3 分钟）
#     'broadcast' / 'allreduce'  - vllm 内部的 collective 调用点
#
# 排查死锁的推荐组合：
#   DEBUG=1 nohup bash run.sh ... > server_debug.log 2>&1 &
#   死锁后看 server_debug.log 末尾的 MCCL 日志
