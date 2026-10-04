#!/usr/bin/env bash
# 一键测试：启 vllm → 等就绪 → 跑 benchmark → 清理
#
# 用法:
#   bash run_all.sh                                       # 默认 MODE=our（复现 bug）
#   MODE=workshop bash run_all.sh                         # 切到 workshop 验证可跑的配置
#   MODE=minimal bash run_all.sh                          # 最小化配置（用于二分定位）
#   bash run_all.sh /data/SETS-2.0-test/models/Qwen2.5-7B 8 32768
#   DEBUG=1 bash run_all.sh                               # 开 MCCL debug
#   SCENARIOS=long_context bash run_all.sh                # 只跑 long_context
#   CONCURRENCY=1 REQUESTS=4 bash run_all.sh              # 复现 bug 模式
#
# 三种 MODE:
#   MODE=our       prefix_caching=1, custom_allreduce=0, eager=0    ← 复现死锁
#   MODE=workshop  prefix_caching=0, custom_allreduce=1, eager=0    ← 复刻 workshop 可跑配置
#   MODE=minimal   prefix_caching=0, custom_allreduce=0, eager=1    ← 关一切特性
#
# 在容器内跑（推荐 docker exec 模式）:
#   docker exec -it gy_work bash /data/my_vllm_test/run_all.sh
#
# 在调度器里跑（作为 docker-compose 的 command）:
#   command: bash -lc "cd /data/my_vllm_test && bash run_all.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============== 参数（可命令行覆盖也可 env 覆盖）==============
MODEL_PATH=${1:-${MODEL_PATH:-/data/SETS-2.0-test/models/Qwen2.5-14B}}
TP=${2:-${TP:-8}}
MAX_MODEL_LEN=${3:-${MAX_MODEL_LEN:-32768}}
PORT=${PORT:-8001}

# benchmark 参数（env 覆盖）
SCENARIOS=${SCENARIOS:-long_context}
CONCURRENCY=${CONCURRENCY:-1}
REQUESTS=${REQUESTS:-4}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-90}

# vllm_musa 后端实现切换:
#   current  - 保持容器内当前 flash_attn.py 不动(可能带 Path A / CODEX patch)
#   original - 启动 vllm 前临时恢复原始 vllm_musa flash_attn.py,结束后默认恢复启动前文件
VLLM_MUSA_IMPL=${VLLM_MUSA_IMPL:-current}
VLLM_MUSA_FLASH_ATTN=${VLLM_MUSA_FLASH_ATTN:-/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py}
VLLM_MUSA_ORIGINAL_FLASH_ATTN=${VLLM_MUSA_ORIGINAL_FLASH_ATTN:-/data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py}
VLLM_MUSA_RESTORE_AFTER=${VLLM_MUSA_RESTORE_AFTER:-1}

# ============== 三层超时设置 ==============
# 1. REQUEST_TIMEOUT  - benchmark 单条请求超时（默认 90s）
# 2. BENCH_TIMEOUT    - benchmark.py 整体超时（默认 600s = 10min）
# 3. OVERALL_TIMEOUT  - 整个 run_all.sh 兜底超时（默认 1500s = 25min）
# 4. VLLM_READY_TIMEOUT - 等 vllm 启动就绪的最长时间（默认 300s）
BENCH_TIMEOUT=${BENCH_TIMEOUT:-900}
OVERALL_TIMEOUT=${OVERALL_TIMEOUT:-1500}
VLLM_READY_TIMEOUT=${VLLM_READY_TIMEOUT:-300}

# 配置模式（透传给 run.sh）
MODE=${MODE:-our}
export MODE

# 输出目录（按时间戳，加 MODE 标签）
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/runs/${TS}_${MODE}}"
mkdir -p "$RUN_DIR"
SERVER_LOG="$RUN_DIR/server.log"
BENCH_LOG="$RUN_DIR/bench.log"
BENCH_JSON="$RUN_DIR/bench.json"
META_FILE="$RUN_DIR/meta.txt"
FLASH_ATTN_BEFORE="$RUN_DIR/flash_attn.before_run.py"

# 状态：记录到这个文件，外部可以查
STATUS_FILE="$SCRIPT_DIR/run_all.status"
write_status() {
  cat > "$STATUS_FILE" <<EOF
state=$1
exit_code=$2
detail=$3
run_dir=$RUN_DIR
updated_at=$(date '+%F %T')
EOF
}

cleanup() {
  local rc=$?
  echo
  echo "=== Cleanup ==="
  # 杀 watchdog
  if [ -n "${WATCHDOG_PID:-}" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill -9 "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "杀 vllm (PID=$VLLM_PID)"
    kill -9 "$VLLM_PID" 2>/dev/null || true
  fi
  # 兜底杀掉 vllm 主进程 + 所有 worker
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
  sleep 3
  # ==== CODEX MOD START: restore temporary original vllm_musa 2026-05-21 ====
  if [ "${VLLM_MUSA_IMPL:-current}" = "original" ] && \
     [ "${VLLM_MUSA_RESTORE_AFTER:-1}" = "1" ] && \
     [ -f "${FLASH_ATTN_BEFORE:-}" ]; then
    echo "恢复启动前 flash_attn.py: $VLLM_MUSA_FLASH_ATTN"
    cp "$FLASH_ATTN_BEFORE" "$VLLM_MUSA_FLASH_ATTN" 2>/dev/null || true
  fi
  # ==== CODEX MOD END: restore temporary original vllm_musa 2026-05-21 ====
  if [ $rc -ne 0 ]; then
    write_status "failed" "$rc" "脚本异常退出 (rc=$rc)"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

# ============== 整体兜底 watchdog ==============
# OVERALL_TIMEOUT 秒后如果还没退出，强制杀掉本脚本
SELF_PID=$$
(
  sleep "$OVERALL_TIMEOUT"
  echo "" >&2
  echo "[WATCHDOG] 整体超时 ${OVERALL_TIMEOUT}s，强杀脚本 PID=$SELF_PID" >&2
  kill -TERM "$SELF_PID" 2>/dev/null || true
  sleep 5
  kill -KILL "$SELF_PID" 2>/dev/null || true
) &
WATCHDOG_PID=$!
disown $WATCHDOG_PID 2>/dev/null || true

# ============== 记录运行元数据 ==============
{
  echo "timestamp: $TS"
  echo "mode: $MODE"
  echo "model_path: $MODEL_PATH"
  echo "tp: $TP"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "port: $PORT"
  echo "scenarios: $SCENARIOS"
  echo "concurrency: $CONCURRENCY"
  echo "requests: $REQUESTS"
  echo "request_timeout: $REQUEST_TIMEOUT"
  echo "vllm_musa_impl: $VLLM_MUSA_IMPL"
  echo "vllm_musa_flash_attn: $VLLM_MUSA_FLASH_ATTN"
  echo "vllm_musa_original_flash_attn: $VLLM_MUSA_ORIGINAL_FLASH_ATTN"
  echo "vllm_musa_restore_after: $VLLM_MUSA_RESTORE_AFTER"
  echo "DEBUG: ${DEBUG:-0}"
  echo "ENABLE_PREFIX_CACHING: ${ENABLE_PREFIX_CACHING:-(由 MODE 决定)}"
  echo "USE_CUSTOM_ALLREDUCE: ${USE_CUSTOM_ALLREDUCE:-(由 MODE 决定)}"
  echo "GPU_MEM_UTIL: ${GPU_MEM_UTIL:-(由 MODE 决定)}"
  echo "MAX_NUM_SEQS: ${MAX_NUM_SEQS:-(由 MODE 决定)}"
  echo "ENFORCE_EAGER: ${ENFORCE_EAGER:-(由 MODE 决定)}"
  echo "container: $(hostname)"
  echo "started_at: $(date '+%F %T')"
} > "$META_FILE"

echo "═══════════════════════════════════════════════════════"
echo "  run_all.sh  $TS"
echo "═══════════════════════════════════════════════════════"
cat "$META_FILE"
echo "─────────────────────────────────────────────"
echo "  输出目录: $RUN_DIR"
echo "─────────────────────────────────────────────"
write_status "starting" "-" "准备启动 vllm"

# ============== vllm_musa 实现切换 ==============
# ==== CODEX MOD START: temporary vllm_musa implementation switch 2026-05-21 ====
if [ ! -f "$VLLM_MUSA_FLASH_ATTN" ]; then
  echo "[ERROR] 找不到当前 flash_attn.py: $VLLM_MUSA_FLASH_ATTN"
  exit 1
fi
cp "$VLLM_MUSA_FLASH_ATTN" "$FLASH_ATTN_BEFORE"

case "$VLLM_MUSA_IMPL" in
  current)
    echo "[VLLM_MUSA_IMPL=current] 保持当前 flash_attn.py 不动"
    ;;
  original)
    if [ ! -f "$VLLM_MUSA_ORIGINAL_FLASH_ATTN" ]; then
      echo "[ERROR] 找不到原版 flash_attn.py: $VLLM_MUSA_ORIGINAL_FLASH_ATTN"
      exit 1
    fi
    echo "[VLLM_MUSA_IMPL=original] 临时恢复原版 flash_attn.py"
    cp "$VLLM_MUSA_ORIGINAL_FLASH_ATTN" "$VLLM_MUSA_FLASH_ATTN"
    ;;
  *)
    echo "[ERROR] 未知 VLLM_MUSA_IMPL=$VLLM_MUSA_IMPL (可选: current/original)"
    exit 1
    ;;
esac

{
  echo "vllm_musa_flash_attn_sha256: $(sha256sum "$VLLM_MUSA_FLASH_ATTN" | awk '{print $1}')"
  echo "vllm_musa_flash_attn_has_codex: $(grep -q 'CODEX MOD\|CODEX PATH\|PATH-A' "$VLLM_MUSA_FLASH_ATTN" && echo yes || echo no)"
} >> "$META_FILE"
# ==== CODEX MOD END: temporary vllm_musa implementation switch 2026-05-21 ====

# ============== 启 vllm（后台）==============
echo
echo "▶ Step 1/4: 启动 vllm serve（后台）"
nohup bash run.sh "$MODEL_PATH" "$TP" "$MAX_MODEL_LEN" "$PORT" > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!
echo "  vllm PID: $VLLM_PID"
echo "  vllm 日志: $SERVER_LOG"

# ============== 等就绪 ==============
echo
echo "▶ Step 2/4: 等 vllm 就绪（最多 ${VLLM_READY_TIMEOUT}s）"
write_status "waiting_ready" "-" "等 vllm 加载完成"
READY_PATTERN="Application startup complete"
START_TS=$(date +%s)
while true; do
  if grep -q "$READY_PATTERN" "$SERVER_LOG" 2>/dev/null; then
    ELAPSED=$(($(date +%s) - START_TS))
    echo "  ✅ vllm 就绪（用时 ${ELAPSED}s）"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "  ❌ vllm 进程已退出，看 $SERVER_LOG 末尾："
    tail -30 "$SERVER_LOG"
    write_status "failed" "1" "vllm 启动崩溃"
    exit 1
  fi
  ELAPSED=$(($(date +%s) - START_TS))
  if [ $ELAPSED -gt $VLLM_READY_TIMEOUT ]; then
    echo "  ❌ 等就绪超时 ${VLLM_READY_TIMEOUT}s"
    tail -30 "$SERVER_LOG"
    write_status "failed" "2" "vllm 启动超时"
    exit 2
  fi
  # 每 30s 打一次进度
  if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
    echo "    [${ELAPSED}s] 等待中... 最新 server.log："
    tail -2 "$SERVER_LOG" | sed 's/^/      /'
  fi
  sleep 3
done

# ============== 跑 benchmark ==============
echo
echo "▶ Step 3/4: 跑 benchmark.py（整体超时 ${BENCH_TIMEOUT}s）"
write_status "benchmarking" "-" "压测中"
BENCH_START=$(date +%s)
set +e
# timeout 命令：超过 BENCH_TIMEOUT 先发 TERM，5 秒后还没退就 KILL
timeout --signal=TERM --kill-after=5s "${BENCH_TIMEOUT}s" \
  python benchmark.py \
    --port "$PORT" \
    --scenarios "$SCENARIOS" \
    --concurrency "$CONCURRENCY" \
    --requests "$REQUESTS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --vllm-musa-impl "$VLLM_MUSA_IMPL" \
    --vllm-musa-flash-attn "$VLLM_MUSA_FLASH_ATTN" \
    --output "$BENCH_JSON" \
    2>&1 | tee "$BENCH_LOG"
BENCH_RC=${PIPESTATUS[0]}
set -e
BENCH_ELAPSED=$(($(date +%s) - BENCH_START))
echo
echo "  benchmark 用时: ${BENCH_ELAPSED}s, exit code: $BENCH_RC"
# exit code 124 = timeout 命令自己的超时退出码
if [ $BENCH_RC -eq 124 ]; then
  echo "  ⚠️  benchmark 触发整体超时 (${BENCH_TIMEOUT}s)，被 timeout 命令强杀"
fi

# ============== 总结 ==============
echo
echo "▶ Step 4/4: 总结"
{
  echo "ended_at: $(date '+%F %T')"
  echo "bench_elapsed: ${BENCH_ELAPSED}s"
  echo "bench_exit_code: $BENCH_RC"
} >> "$META_FILE"

# 抓 benchmark 的 summary 行（如果有）
if [ -f "$BENCH_LOG" ]; then
  echo
  echo "─── benchmark Summary ───"
  grep -A 100 'Summary' "$BENCH_LOG" 2>/dev/null | head -20 || echo "（无 Summary 段，可能是中途挂了）"
fi

# 抓 MCCL 死锁信号（如果开了 DEBUG）
if [ "${DEBUG:-0}" = "1" ] && [ -f "$SERVER_LOG" ]; then
  echo
  echo "─── MCCL collective 调用统计（DEBUG=1 时）───"
  grep -c "MCCL INFO" "$SERVER_LOG" 2>/dev/null | xargs echo "MCCL INFO 行数:"
  echo "最后一条 MCCL 日志:"
  grep "MCCL INFO" "$SERVER_LOG" 2>/dev/null | tail -1 | head -c 200
  echo
fi

if [ $BENCH_RC -eq 0 ]; then
  write_status "success" "0" "测试完成"
  echo "✅ 全部完成: $RUN_DIR"
else
  write_status "failed" "$BENCH_RC" "benchmark 失败"
  echo "❌ benchmark 失败 ($BENCH_RC)，看: $BENCH_LOG"
fi

exit $BENCH_RC
