#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry that now targets the generic any-preprocess runner.

HF_MODEL_ID=${HF_MODEL_ID:-"Qwen/Qwen2.5-VL-3B-Instruct"}
HF_PROCESSOR_ID=${HF_PROCESSOR_ID:-"$HF_MODEL_ID"}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "[bash] Training with HF_MODEL_ID=$HF_MODEL_ID HF_PROCESSOR_ID=$HF_PROCESSOR_ID"

time python scripts/train_any_preprocess.py \
  --model-id "$HF_MODEL_ID" \
  --processor-id "$HF_PROCESSOR_ID" \
  --jax-platform tpu \
  --log-tpu-memory \
  --fixed-max-length 4096 \
  --fixed-batch-size 8 \
  --sample-image-url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
  --sample-prompt "Describe this image." \
  --train-split "train[:3000]" \
  --train-epochs 5 \
  --eval-split "test[:300]" \
  --eval-epochs 5