#!/bin/bash
# PD 过载观测:起 PD 栈 → 后台采样(prefill/decode 队列+KV+卡util)→ 真 bench_serving 压 3.5k/1k 过载 → 汇总。
# 目的:定"prefill 空等(被 decode-KV 反压)"还是"prefill 真算不过来"。纯观测,不改源码。
# 用法:bash pd_monitor_run.sh [rate] [num_prompts] [spec]
#   默认 rate=1.0 num_prompts=500 spec=on(带 EAGLE3 投机);spec=off 则不带,可做对照
set -e
BENCH=/mnt/seed17/001688/models/Qwen/bench
PY=/root/.virtualenvs/sglang-0.5.6/bin/python
MODEL=/mnt/seed17/001688/models/Qwen3-32B
DRAFT=/mnt/seed17/001688/models/Qwen3-32B_eagle3
DATASET=$BENCH/ShareGPT_V3_unfiltered_cleaned_split.json
RATE=${1:-1.0}
NP=${2:-500}
SPEC=${3:-on}
CSV=$BENCH/pd_monitor_$(date +%Y%m%d_%H%M%S)_spec-$SPEC.csv   # 带时间戳+spec标记,不覆盖

export AUTOBENCH_PD=1
export AUTOBENCH_TP=4
export AUTOBENCH_MODELS="[\"$MODEL\"]"
# extra_args 用 python 拼,避免手写嵌套 JSON 转义出错。spec=on 时加 EAGLE3(与集中式同款)。
if [ "$SPEC" = "on" ]; then
  export AUTOBENCH_EXTRA_ARGS="$($PY -c "import json; s=json.dumps({'method':'eagle3','model':'$DRAFT','num_speculative_tokens':3}); print(json.dumps({'$MODEL':['--max-num-seqs','256','--speculative-config',s]}))")"
else
  export AUTOBENCH_EXTRA_ARGS="{\"$MODEL\":[\"--max-num-seqs\",\"256\"]}"
fi
echo "[mon] SPEC=$SPEC  EXTRA_ARGS=$AUTOBENCH_EXTRA_ARGS"

MON_PID=""
cleanup() {
  [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true
  echo "[mon] 停 PD 栈 ..."; $PY $BENCH/run_list.py stop || true
}
trap cleanup EXIT INT

echo "[mon] 起 PD 栈(加载约 10-15min)..."
$PY $BENCH/run_list.py

echo "[mon] 确认 metric 字段名(看 running/waiting + KV cache 用量的真实字段名):"
curl -s 127.0.0.1:8200/metrics | grep -iE '^vllm:(num_requests|.*cache_usage)' | grep -v '^#' | head -20

echo "[mon] 采样器后台启动,先空采 10s 基线(此时队列应为 0)..."
$PY $BENCH/pd_monitor.py "$CSV" &
MON_PID=$!
sleep 10

echo "[mon] ===== 压 3.5k/1k @ rate=$RATE  num_prompts=$NP (过载) ====="
$PY $BENCH/bench_serving.py --backend vllm --host 127.0.0.1 --port 8000 \
  --model "$MODEL" --served-model-name qwen3-32b --tokenizer "$MODEL" \
  --dataset-name random --dataset-path "$DATASET" \
  --num-prompts "$NP" --random-input-len 3584 --random-output-len 1024 \
  --random-range-ratio 1.0 --request-rate "$RATE" --burstiness 102 \
  --warmup-requests 0 --apply-chat-template --disable-tqdm || true

sleep 5
kill "$MON_PID" 2>/dev/null || true; MON_PID=""
echo "[mon] ===== 采样已保存(逐点,人工分析)====="
echo "[mon] CSV: $CSV  ($(wc -l < "$CSV") 行)"
echo "[mon] 末尾几行预览:"
tail -8 "$CSV"
