#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <MODEL_PATH>"
  exit 1
fi

export MODEL_PATH="$1"
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
TP=1

set -x
VLLM_USE_V1=0 vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --served-model-name "${MODEL_NAME}" \
  --block-size 64 \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size 1 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,30,32,50,64,100,128,256]}'
set +x
