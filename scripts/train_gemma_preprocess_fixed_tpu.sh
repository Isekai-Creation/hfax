export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
time python scripts/train_gemma_preprocess.py \
  --jax-platform tpu \
  --log-tpu-memory \
  --fixed-max-length 4096 \
  --fixed-batch-size 8 \
  --train-split "train[:3000]" \
  --train-epochs 5 \
  --eval-split "test[:300]" \
  --eval-epochs 5
