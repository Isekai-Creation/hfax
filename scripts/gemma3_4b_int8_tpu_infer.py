#!/usr/bin/env python3
"""
TPU inference with quantization (INT8) for Gemma3 4B using hfax.

Quickstart (Colab TPU runtime):
  !pip install -q hfax  # or gemma if published under that name
  !python - <<'PY'
from pathlib import Path
import os, sys
os.chdir('/content')
PY

Then run this script:
  python -m scripts.gemma3_4b_int8_tpu_infer \
    --prompt "The capital of France is" \
    --max-new-tokens 30 \
    --quant-method INT8

Notes
- Uses hfax.gm.nn.IntWrapper for int inference layers.
- Quantizes pre-trained checkpoint weights with hfax.peft.quantize.
- Runs on TPU if JAX sees TPU devices; otherwise falls back to default backend.
"""

from __future__ import annotations

import time
import argparse
import os
from typing import Optional

# Allocate full memory (JAX default may be fractional on GPU; harmless on TPU)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.00")

import jax
import jax.numpy as jnp

try:
    import hfax as gemma
except Exception:  # pragma: no cover
    import gemma  # type: ignore
gm = gemma.gm
peft = gemma.peft


def _select_device() -> str:
    backend = jax.default_backend()
    devs = jax.devices()
    info = f"JAX backend={backend}; devices={[d.platform for d in devs]}"
    print(info)
    if any(d.platform == "tpu" for d in devs):
        return "tpu"
    return backend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemma3 4B INT8 inference on TPU")
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=256, help="init seq len")
    p.add_argument("--quant-method", type=str, default="INT8", choices=["INT8", "INT4"])
    p.add_argument(
        "--text-only", action="store_true", help="disable vision encoder to save memory"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    platform = _select_device()

    # 1) Build quantized inference wrapper around Gemma3 4B
    # dtype maps to integer dtype used in Int layers
    q_dtype = jnp.int8 if args.quant_method != "INT4" else jnp.int4
    model = gm.nn.IntWrapper(
        model=gm.nn.Gemma3_4B(text_only=True if args.text_only else False),
        dtype=q_dtype,
    )

    # 2) Initialize parameter structure (shapes/dtypes)
    token_ids = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
    variables = model.init(jax.random.key(0), token_ids)
    params = variables["params"]
    del variables  # shapes captured; we will load weights next

    # 3) Load pre-trained params and quantize
    #    For 4B instruction-tuned; adjust to *_PT if needed.
    ckpt = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    print(f"Loading checkpoint: {ckpt}")
    original = gm.ckpts.load_params(ckpt)
    print("Quantizing params (", args.quant_method, ") ...")
    params = peft.quantize(
        original,
        method=args.quant_method,
        checkpoint_kernel_key="w",
    )
    del original

    # 4) Tokenizer and prompt
    tokenizer = gm.text.Gemma3Tokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    prompt = jnp.asarray([tokenizer.special_tokens.BOS] + prompt_ids, dtype=jnp.int32)

    # 5) One-step logits (return_last_only)
    print("Running single-step forward (last-token logits)...")
    out = model.apply(
        {"params": params},
        tokens=prompt,
        return_last_only=True,  # Only predict the last token
    )
    # Optional visualization of token distribution
    try:
        import treescope as _ts  # noqa: F401

        tokenizer.plot_logits(out.logits)
    except Exception:
        pass
    print(f"Logits shape (last token): {out.logits.shape}")

    start_time = time.time()

    # 6) Sampling end-to-end
    sampler = gm.text.Sampler(model=model, params=params, tokenizer=tokenizer)
    text = sampler.sample(args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n=== Sample ===\n" + text)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    start_time = time.time()

    # 6) Sampling end-to-end
    sampler = gm.text.Sampler(model=model, params=params, tokenizer=tokenizer)
    text = sampler.sample(args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n=== Sample ===\n" + text)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
