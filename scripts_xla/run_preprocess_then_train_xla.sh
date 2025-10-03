#!/usr/bin/env bash
set -euo pipefail

# Example wrapper to chain preprocess -> train in separate processes.
# Edit these variables directly as needed.

MODEL_ID="unsloth/gemma-3-4b-it"
DATASET="unsloth/LaTeX_OCR"
TRAIN_SPLIT="train"
EVAL_SPLIT="test[:3000]"
BOUNDARIES="512,1024,2048,4096"
BATCH_SIZE=128
OUT_DIR="/dev/shm/gemma_shards"

echo "[1/2] Preprocess -> ${OUT_DIR}"
python scripts/preprocess_gemma_dynamic_xla.py \
  --model-id "$MODEL_ID" \
  --dataset "$DATASET" \
  --train-split "$TRAIN_SPLIT" \
  --eval-split "$EVAL_SPLIT" \
  --bucket-boundaries "$BOUNDARIES" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUT_DIR"

echo "[2/2] Train from shards"
python scripts_xla/train_gemma_from_shards_xla.py \
  --model-id "$MODEL_ID" \
  --shards-dir "$OUT_DIR" \
  --train-epochs 5 
