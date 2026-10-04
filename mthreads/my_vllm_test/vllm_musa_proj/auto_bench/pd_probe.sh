#!/bin/bash
# PD 并发爬坡 + num_workers A/B:对比 mooncake 发送线程池默认(10) vs 32,
# 看"搬运+开销"随并发是否回落 → 判定瓶颈是连接器发送池还是 toy 代理。
# 用法:bash pd_probe.sh           # 跑 A/B(默认 + 32),两次加载,共约 35-45min
#       bash pd_probe.sh 32       # 只跑指定 num_workers 一次
set -e
BENCH=/mnt/seed17/001688/models/Qwen/bench
PY=/root/.virtualenvs/sglang-0.5.6/bin/python
MODEL=/mnt/seed17/001688/models/Qwen3-32B

export AUTOBENCH_PD=1
export AUTOBENCH_TP=4
export AUTOBENCH_MODELS="[\"$MODEL\"]"
export AUTOBENCH_EXTRA_ARGS="{\"$MODEL\":[\"--max-num-seqs\",\"256\"]}"

run_one() {   # $1 = num_workers ("" = 默认10)
  local nw="$1"
  if [ -n "$nw" ]; then export AUTOBENCH_PD_NUM_WORKERS="$nw"; else unset AUTOBENCH_PD_NUM_WORKERS; fi
  echo "============================================================"
  echo "[probe] 起 PD 栈  num_workers=${nw:-默认(10)}  (加载约 10-15min) ..."
  $PY $BENCH/run_list.py
  echo "[probe] ===== 并发爬坡 (num_workers=${nw:-默认}) ====="
  $PY $BENCH/pd_probe.py
  echo "[probe] 停 PD 栈 ..."
  $PY $BENCH/run_list.py stop || true
  sleep 5
}

if [ -n "$1" ]; then
  trap '$PY $BENCH/run_list.py stop || true' EXIT INT
  run_one "$1"
else
  trap '$PY $BENCH/run_list.py stop || true' EXIT INT
  run_one ""      # 默认 num_workers=10
  run_one "32"    # 调到 32 对照
fi
echo "[probe] 全部完成。"
