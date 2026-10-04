#!/usr/bin/env bash
# GPU 监视器 + 自动触发 run_all.sh
#
# 用法:
#   bash wait_and_run.sh                                 # 默认设置，等到 GPU 空就跑
#   FREE_THRESHOLD_GB=20 bash wait_and_run.sh            # 要求每卡余量 ≥ 20GB
#   STABLE_SECONDS=30 bash wait_and_run.sh               # 余量需要持续稳定 30s
#   MAX_WAIT=3600 bash wait_and_run.sh                   # 最多等 1 小时
#   MODE=workshop bash wait_and_run.sh                   # 透传 MODE 给 run_all.sh
#
# 在 gy_work 容器外（host）跑也行，会自动 docker exec 进容器
# 推荐用法：在 host 端跑这个脚本，它会监控 mthreads-gmi（host 视角），触发 docker exec gy_work bash run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============== 参数 ==============
# 每卡至少多少 GB 空闲才触发
FREE_THRESHOLD_GB=${FREE_THRESHOLD_GB:-15}
# 余量需要持续稳定多少秒（防抖：可能是请求间隙瞬时余量）
STABLE_SECONDS=${STABLE_SECONDS:-15}
# 采样间隔（秒）
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-5}
# 最大等待时间（秒），超过就放弃
MAX_WAIT=${MAX_WAIT:-3600}
# 是否在容器里跑（默认 1）
USE_DOCKER=${USE_DOCKER:-1}
CONTAINER_NAME=${CONTAINER_NAME:-gy_work}

# 崩溃检测 + 重试
MAX_RETRIES=${MAX_RETRIES:-5}              # run_all.sh 失败后最多重试几次
RETRY_COOLDOWN=${RETRY_COOLDOWN:-30}        # 重试前先冷却几秒（让 GPU 清理）
# OOM / 启动失败的关键字模式（脚本会在 run_all.sh 失败时扫 log 看是不是这类错）
OOM_PATTERNS=(
  "out of memory"
  "OOM"
  "CUDA error"
  "MUSA error"
  "MUSA out of memory"
  "Insufficient.*memory"
  "Failed to allocate"
  "No available memory for the cache blocks"
)

# 透传给 run_all.sh 的环境变量（白名单）
PASS_ENV_VARS=(MODE DEBUG DEBUG_TORCH MUSA_BLOCKING SCENARIOS CONCURRENCY REQUESTS REQUEST_TIMEOUT
               BENCH_TIMEOUT OVERALL_TIMEOUT VLLM_READY_TIMEOUT
               GPU_MEM_UTIL MAX_NUM_SEQS ENABLE_PREFIX_CACHING
               USE_CUSTOM_ALLREDUCE ENFORCE_EAGER MODEL_PATH TP MAX_MODEL_LEN PORT
               TORCH_NCCL_TIMEOUT_MS
               VLLM_MUSA_IMPL VLLM_MUSA_FLASH_ATTN VLLM_MUSA_ORIGINAL_FLASH_ATTN
               VLLM_MUSA_RESTORE_AFTER)

# ============== 工具函数 ==============
log() {
  echo "[$(date '+%F %T')] $*"
}

# 拿每张卡的 free MiB（host 端 mthreads-gmi）
# 输出：每行一个数字，单位 MiB
get_free_per_card() {
  mthreads-gmi 2>/dev/null \
    | grep -oE '[0-9]+MiB\([0-9]+MiB\)' \
    | awk -F'[(MiB)]+' '{used=$1; total=$2; print total - used}'
}

# 检查所有卡是否都 >= 阈值
all_cards_above_threshold() {
  local threshold_mib=$1
  local min_free=99999999
  while read -r free; do
    [ -n "$free" ] || continue
    if [ "$free" -lt "$min_free" ]; then
      min_free=$free
    fi
  done < <(get_free_per_card)
  echo "$min_free"
  [ "$min_free" -ge "$threshold_mib" ]
}

# 构造透传环境变量
build_env_args() {
  local args=""
  for v in "${PASS_ENV_VARS[@]}"; do
    if [ -n "${!v:-}" ]; then
      args="$args $v='${!v}'"
    fi
  done
  echo "$args"
}

# 触发 run_all.sh
trigger_run() {
  local env_args=$(build_env_args)
  log "▶ 触发 run_all.sh"
  log "  环境: $env_args"

  if [ "$USE_DOCKER" = "1" ]; then
    if ! docker ps --format '{{.Names}}' | grep -qw "$CONTAINER_NAME"; then
      log "[ERROR] 容器 $CONTAINER_NAME 不在运行中"
      return 1
    fi
    # 用 docker exec 触发
    docker exec "$CONTAINER_NAME" bash -lc "$env_args bash /data/my_vllm_test/run_all.sh"
  else
    # 本地直接跑
    bash -lc "$env_args bash $SCRIPT_DIR/run_all.sh"
  fi
}

# 检查最近一次 run 是不是 OOM / CUDA 错误（race 导致 GPU 不够）
# 读 run_all.status 找最新 RUN_DIR，扫 server.log
is_oom_failure() {
  local status_file="$SCRIPT_DIR/run_all.status"
  if [ ! -f "$status_file" ]; then
    return 1  # 没有状态文件，没法判断
  fi
  local run_dir
  run_dir=$(grep '^run_dir=' "$status_file" 2>/dev/null | cut -d= -f2-)
  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    return 1
  fi
  local server_log="$run_dir/server.log"
  if [ ! -f "$server_log" ]; then
    return 1
  fi
  # 扫 OOM 模式
  for pat in "${OOM_PATTERNS[@]}"; do
    if grep -iE "$pat" "$server_log" >/dev/null 2>&1; then
      log "  [崩溃分析] 发现 OOM/CUDA 错误模式: \"$pat\""
      grep -iE "$pat" "$server_log" 2>/dev/null | head -3 | sed 's/^/    /'
      return 0
    fi
  done
  return 1
}

# ============== 启动信息 ==============
THRESHOLD_MIB=$((FREE_THRESHOLD_GB * 1024))
echo "═══════════════════════════════════════════════════════"
echo "  wait_and_run.sh  $(date '+%F %T')"
echo "═══════════════════════════════════════════════════════"
echo "  GPU 余量阈值 (每卡): ${FREE_THRESHOLD_GB} GB = ${THRESHOLD_MIB} MiB"
echo "  需要持续稳定:        ${STABLE_SECONDS} 秒"
echo "  采样间隔:           ${SAMPLE_INTERVAL} 秒"
echo "  最大等待:           ${MAX_WAIT} 秒"
echo "  USE_DOCKER:         $USE_DOCKER (容器: $CONTAINER_NAME)"
echo "  MODE:               ${MODE:-our (默认)}"
echo "─────────────────────────────────────────────"

# ============== 监控循环 ==============
START_TS=$(date +%s)
STABLE_START=0   # 0 = 当前不稳定
LAST_LOG=0
TICK=0

while true; do
  ELAPSED=$(($(date +%s) - START_TS))

  # 总超时
  if [ $ELAPSED -gt $MAX_WAIT ]; then
    log "❌ MAX_WAIT (${MAX_WAIT}s) 超时，放弃"
    exit 1
  fi

  # 采样
  MIN_FREE_MIB=$(get_free_per_card | sort -n | head -1)
  if [ -z "$MIN_FREE_MIB" ]; then
    log "⚠️  无法读 mthreads-gmi，重试..."
    sleep "$SAMPLE_INTERVAL"
    continue
  fi
  MIN_FREE_GB=$(awk "BEGIN { printf \"%.1f\", $MIN_FREE_MIB / 1024 }")

  # 判断
  if [ "$MIN_FREE_MIB" -ge "$THRESHOLD_MIB" ]; then
    # 余量够
    if [ "$STABLE_START" -eq 0 ]; then
      STABLE_START=$(date +%s)
      log "✨ 余量达标 (最小 ${MIN_FREE_GB} GB)，开始计时稳定性"
    fi
    STABLE_FOR=$(($(date +%s) - STABLE_START))
    if [ "$STABLE_FOR" -ge "$STABLE_SECONDS" ]; then
      log "✅ 持续 ${STABLE_FOR}s 余量都够，触发测试"
      break
    fi
    # 进度日志（每 5 秒）
    if [ $((TICK % 1)) -eq 0 ]; then
      log "  稳定中 ${STABLE_FOR}/${STABLE_SECONDS}s  (最小余量 ${MIN_FREE_GB} GB)"
    fi
  else
    # 余量不够，重置稳定计时
    if [ "$STABLE_START" -ne 0 ]; then
      log "💤 余量降到 ${MIN_FREE_GB} GB (<${FREE_THRESHOLD_GB})，稳定中断"
      STABLE_START=0
    fi
    # 进度日志（每 30 秒打一次）
    NOW=$(date +%s)
    if [ $((NOW - LAST_LOG)) -ge 30 ]; then
      log "  等待中... 已等 ${ELAPSED}s，最小余量 ${MIN_FREE_GB} GB / ${FREE_THRESHOLD_GB} GB"
      LAST_LOG=$NOW
    fi
  fi

  TICK=$((TICK + 1))
  sleep "$SAMPLE_INTERVAL"
done

# ============== 触发（带崩溃重试）==============
echo "─────────────────────────────────────────────"
ATTEMPT=0
LAST_RC=0
while [ $ATTEMPT -le $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT + 1))
  log "▶ 第 $ATTEMPT/$((MAX_RETRIES + 1)) 次尝试"

  set +e
  trigger_run
  LAST_RC=$?
  set -e

  echo "─────────────────────────────────────────────"
  log "run_all.sh 退出，exit code = $LAST_RC"

  # 退出码 0 = 成功
  if [ $LAST_RC -eq 0 ]; then
    log "✅ 成功完成"
    exit 0
  fi

  # 检测是不是 race 导致的 OOM/启动失败
  if is_oom_failure; then
    log "⚠️  检测到 OOM/CUDA 错误（可能是和 workshop 测试抢资源），$RETRY_COOLDOWN 秒后重新等待 GPU"
  else
    # 不是 OOM 的失败可能是 bug 死锁、超时等
    log "❌ 非 OOM 失败 (rc=$LAST_RC)，可能是 bug 复现成功 / 超时 / 其他错误"
    log "   不重试，看 run_all.status 的 run_dir 详查"
    exit $LAST_RC
  fi

  # 还在重试预算内
  if [ $ATTEMPT -le $MAX_RETRIES ]; then
    log "💤 冷却 $RETRY_COOLDOWN 秒后重新进入 GPU 等待循环..."
    sleep "$RETRY_COOLDOWN"

    # 重新进入监控循环
    log "▶ 重新等待 GPU 余量"
    STABLE_START=0
    LAST_LOG=0
    REWAIT_START=$(date +%s)
    while true; do
      ELAPSED=$(($(date +%s) - START_TS))
      if [ $ELAPSED -gt $MAX_WAIT ]; then
        log "❌ MAX_WAIT (${MAX_WAIT}s) 超时，放弃重试"
        exit 1
      fi
      MIN_FREE_MIB=$(get_free_per_card | sort -n | head -1)
      [ -z "$MIN_FREE_MIB" ] && { sleep "$SAMPLE_INTERVAL"; continue; }
      MIN_FREE_GB=$(awk "BEGIN { printf \"%.1f\", $MIN_FREE_MIB / 1024 }")

      if [ "$MIN_FREE_MIB" -ge "$THRESHOLD_MIB" ]; then
        if [ "$STABLE_START" -eq 0 ]; then
          STABLE_START=$(date +%s)
        fi
        if [ $(($(date +%s) - STABLE_START)) -ge $STABLE_SECONDS ]; then
          log "✅ 余量再次达标，重试"
          break
        fi
      else
        STABLE_START=0
        NOW=$(date +%s)
        if [ $((NOW - LAST_LOG)) -ge 30 ]; then
          log "  重试等待中... 已等总 ${ELAPSED}s，最小余量 ${MIN_FREE_GB} GB / ${FREE_THRESHOLD_GB} GB"
          LAST_LOG=$NOW
        fi
      fi
      sleep "$SAMPLE_INTERVAL"
    done
    echo "─────────────────────────────────────────────"
  fi
done

log "❌ 重试 $MAX_RETRIES 次都失败，放弃"
exit $LAST_RC
