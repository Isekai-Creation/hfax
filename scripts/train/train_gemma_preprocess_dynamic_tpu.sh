export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
time python scripts/train_gemma_preprocess.py \
  --jax-platform tpu \
  --log-tpu-memory \
  --log-batch-tuning \
  --bucket-boundaries "4096" \
  --max-dynamic-batch 8 \
  --train-split "train[:3000]" \
  --train-epochs 5 \
  --eval-split "test[:3000]" \
  --eval-epochs 5
