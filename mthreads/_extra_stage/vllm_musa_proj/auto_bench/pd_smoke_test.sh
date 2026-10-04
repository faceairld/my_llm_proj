#!/bin/bash
# PD 分离冒烟测试(2026-06-16):验证 mooncake KV 搬运在 MTT S5000 上是否可用。
# 改编自 vllm_musa 官方 example/disaggregated_serving/disaggregated_serving.sh,
# 调整为我们的布局:prefill tp4(卡0-3,producer)+ decode tp4(卡4-7,consumer)+ proxy。
# 通过 = 能正常返回补全;失败多半是 mooncake 协议/RDMA 网卡问题(见末尾排查)。
#
# 用法:bash pd_smoke_test.sh [model_path] [protocol]
#   protocol 默认 rdma;若 rdma 不通,试 tcp:  bash pd_smoke_test.sh '' tcp
set -x

BENCH=/mnt/seed17/001688/models/Qwen/bench
MODEL=${1:-/mnt/seed17/001688/models/Qwen3-32B}
PROTO=${2:-rdma}
PY=/root/.virtualenvs/sglang-0.5.6/bin/python
SERVED=qwen3-32b
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
LOGDIR=$BENCH/pd_smoke_logs
mkdir -p "$LOGDIR"

cleanup() {
  echo "[smoke] cleanup ..."
  for p in "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null
  done
}
trap cleanup EXIT INT

wait_for_server() {  # $1=port
  timeout 1800 bash -c "until curl -s localhost:$1/v1/models >/dev/null; do sleep 3; done" \
    && echo "[smoke] port $1 ready" || { echo "[smoke] port $1 TIMEOUT"; return 1; }
}

KV_PRE='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"'"$PROTO"'"}}'
KV_DEC='{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"'"$PROTO"'"}}'

# --- prefill 实例:卡 0-3,tp4,KV 生产者 ---
MUSA_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size 4 --trust-remote-code \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager \
  --port $PREFILL_PORT \
  --kv-transfer-config "$KV_PRE" > "$LOGDIR/prefill.log" 2>&1 &
PREFILL_PID=$!

# --- decode 实例:卡 4-7,tp4,KV 消费者 ---
MUSA_VISIBLE_DEVICES=4,5,6,7 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  vllm serve "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size 4 --trust-remote-code \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager \
  --port $DECODE_PORT \
  --kv-transfer-config "$KV_DEC" > "$LOGDIR/decode.log" 2>&1 &
DECODE_PID=$!

wait_for_server $PREFILL_PORT || exit 1
wait_for_server $DECODE_PORT  || exit 1

# --- proxy:对外 8000,路由 prefill→decode ---
$PY "$BENCH/toy_proxy_server.py" \
  --prefiller-host 127.0.0.1 --prefiller-port $PREFILL_PORT \
  --decoder-host 127.0.0.1 --decoder-port $DECODE_PORT \
  --port $PROXY_PORT > "$LOGDIR/proxy.log" 2>&1 &
PROXY_PID=$!
sleep 5

echo "[smoke] ===== 发测试请求 ====="
OUT=$(curl -X POST -s http://localhost:$PROXY_PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$SERVED"'","prompt":"San Francisco is a","max_tokens":20,"temperature":0}')
echo "[smoke] 返回: $OUT"

if echo "$OUT" | grep -q '"text"'; then
  echo "[smoke] ✅ PD 分离通了(mooncake $PROTO 可用)"
else
  echo "[smoke] ❌ 失败。排查:"
  echo "         - prefill 日志: $LOGDIR/prefill.log"
  echo "         - decode  日志: $LOGDIR/decode.log"
  echo "         - proxy   日志: $LOGDIR/proxy.log"
  echo "         - 若 rdma 报网卡错,重试 tcp:  bash pd_smoke_test.sh '$MODEL' tcp"
  echo "         - 若 rdma 选错网卡,指定:  MOONCAKE_RDMA_DEVICES=mlx5_bond_2 bash pd_smoke_test.sh"
fi
