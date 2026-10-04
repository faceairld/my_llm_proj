#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <MODEL_PATH> [RESULT_DIR] [PORT]"
  exit 1
fi

MODEL_PATH="$1"
RESULT_DIR="${2:-/home/bench_results}"
MODEL_NAME="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
BENCH_PY="/home/bench_serving.py"
PORT="${3:-8000}"
WARMUP_REQUESTS=200
NUM_PROMPTS=200

export TP=1

# Benchmark cases: input_len output_len input/output_label request_rate
BENCH_CASES=(
  "2048 1024 2k/1k 3"
  "2048 1024 2k/1k 3.2"
  "2048 1024 2k/1k 3.4"
  "3072 1024 3k/1k 2"
  "3072 1024 3k/1k 2.2"
  "3072 1024 3k/1k 2.4"
  "3072 1024 3k/1k 2.6"
  "3584 1024 3.5k/1k 2"
  "3584 1024 3.5k/1k 2.2"
  "4096 1024 4k/1k 1"
  "4096 1024 4k/1k 1.2"
  "4096 1024 4k/1k 1.4"
  "4096 1024 4k/1k 1.6"
  "4096 1024 4k/1k 1.8"
  "4096 1024 4k/1k 2"
  "4096 1536 4k/1.5k 1"
  "4096 1536 4k/1.5k 1.2"
  "4096 1536 4k/1.5k 1.4"
)

mkdir -p "$RESULT_DIR"
CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
CSV_FILE="$RESULT_DIR/benchmark_report_${CURRENT_TIME}.csv"

echo "[Step 1] Checking existing vLLM server..."
if ! curl -sSf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[ERROR] vLLM server is not reachable at http://127.0.0.1:${PORT}."
  echo "Please start it first with: /home/run.sh \"$MODEL_PATH\""
  exit 1
fi
echo "[OK] Server is ready."

echo "model_name,tp,input_len,output_len,io_label,request_rate,num_prompts,req_tp,in_tok_tp,out_tok_tp,mean_ttft,median_ttft,p99_ttft,mean_tpot,median_tpot,p99_tpot,mean_itl,p99_itl,mean_e2e,real_concurrency,duration,total_input_tokens,total_output_tokens,status" > "$CSV_FILE"

echo "[Step 2] Running benchmark cases..."
for case_item in "${BENCH_CASES[@]}"; do
  read -r input_len output_len io_label request_rate <<< "$case_item"
  json_out="$RESULT_DIR/temp_${MODEL_NAME}_${input_len}_${output_len}_r${request_rate}.json"

  echo " -> Case: in=$input_len, out=$output_len, rate=$request_rate"

  python3 "$BENCH_PY" \
    --backend vllm \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --dataset-path /home/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio 1.0 \
    --request-rate "$request_rate" \
    --burstiness 102 \
    --warmup-requests "$WARMUP_REQUESTS" \
    --apply-chat-template \
    --disable-tqdm \
    --output-file "$json_out" \
    > /dev/null 2>&1 || true

  if [ -f "$json_out" ]; then
    stats="$(python3 - <<'PY' "$json_out"
import json
import sys

path = sys.argv[1]

try:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    data = json.loads(content) if content else {}
except Exception:
    data = {}

fields = [
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "concurrency",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
]
print(",".join(str(data.get(k, 0)) for k in fields))
PY
)"
    echo "${MODEL_NAME},${TP},${input_len},${output_len},${io_label},${request_rate},${NUM_PROMPTS},${stats},OK" >> "$CSV_FILE"
  else
    echo "${MODEL_NAME},${TP},${input_len},${output_len},${io_label},${request_rate},${NUM_PROMPTS},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FAIL" >> "$CSV_FILE"
  fi

  rm -f "$json_out"
done

echo "[Done] Benchmark completed."
echo "CSV report: $CSV_FILE"
