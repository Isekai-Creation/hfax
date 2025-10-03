#!/usr/bin/env python3
"""Measure TPU memory usage and batch caps for Gemma3 4B at given token sizes.

Runs a native hfax BucketBatchTuner to find the max safe batch size per token
bucket using live TPU memory readings, then executes a verified train step and
reports memory usage (per-replica) for each configuration.

Example:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python scripts/test_batch_tuning.py \
    --tokens 512,4096 \
    --base-batch 8 \
    --max-batch 128 \
    --jax-platform tpu \
    --verbose
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from etils import enp

import hfax
from kauldron import kd
import optax

from hfax.utils.batch_tuner import BucketBatchTuner
from hfax.utils.tpu_mem import per_replica_free_total, log_tpu_memory, get_xla_cur_mem
from transformers import AutoProcessor
import scripts.train_gemma_preprocess as sg


def build_trainstep() -> kd.train.train_step.TrainStep:
    model = hfax.nn.Gemma3_4B(tokens="batch.input")
    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits="preds.logits", labels="batch.target", mask="batch.loss_mask"
    )
    rng_streams = kd.train.RngStreams(
        (
            kd.train.RngStream(
                name="dropout", init=True, train=True, eval=False, per_step=True
            ),
        ),
        seed=42,
    )
    return kd.train.train_step.TrainStep(
        model=model,
        optimizer=optax.adafactor(learning_rate=1e-3),
        rng_streams=rng_streams,
        sharding=kd.sharding.ShardingStrategy(params=kd.sharding.FSDPSharding()),
        init_transform=hfax.ckpts.LoadCheckpoint(
            path=hfax.ckpts.CheckpointPath.GEMMA3_4B_IT
        ),
        aux=kd.train.auxiliaries.Auxiliaries(
            losses={"loss": loss}, metrics=(), summaries=()
        ),
    )


def tile_sample(
    sample: Dict[str, np.ndarray], batch_size: int
) -> Dict[str, np.ndarray]:
    return {k: np.stack([v] * batch_size, axis=0) for k, v in sample.items()}


def measure_memory_from_bucket(
    trainstep: kd.train.train_step.TrainStep,
    state_entry,
    batch_size: int,
    sample: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], Tuple[object, object]]:
    # Ensure a clean slate to reflect memory for this batch only.
    try:
        jax.clear_caches()
    except Exception:
        pass
    # Build element spec and init
    elem_spec = {
        k: enp.ArraySpec(shape=(batch_size,) + v.shape, dtype=v.dtype)
        for k, v in sample.items()
    }

    mem_pre = per_replica_free_total()
    host_state, shardings = state_entry or (None, None)
    if host_state is None:
        state_dev = trainstep.init(elem_spec=elem_spec)
        host_state = jax.device_get(state_dev)
        shardings = jax.tree.map(lambda x: getattr(x, "sharding", None), state_dev)

    def _put(val, sh):
        if sh is not None:
            return jax.device_put(val, sh)
        return jax.device_put(val)

    state = jax.tree.map(_put, host_state, shardings)
    mem_post_init = per_replica_free_total()

    arrs = tile_sample(sample, batch_size)
    batch = {k: jnp.asarray(arrs[k]) for k in sample}
    # Clear caches before the training step to avoid prior compilation artifacts
    try:
        jax.clear_caches()
    except Exception:
        pass
    state, _ = trainstep.step(state, batch)
    jax.block_until_ready(state.step)
    mem_post_step = per_replica_free_total()
    # Fallback: if totals unknown, at least report used_bytes via tpu-info usage table
    if mem_post_step is None:
        try:
            used_total, used_max = get_xla_cur_mem()
            mem_post_step = (None, None)  # sentinel for formatting
        except Exception:
            used_total = used_max = None

    def to_gib(x: int | None) -> float:
        return float(x) / (1024**3) if x is not None else 0.0

    free0, total0 = mem_pre if mem_pre is not None else (None, None)
    free1, total1 = mem_post_init if mem_post_init is not None else (None, None)
    if mem_post_step is None or mem_post_step == (None, None):
        free2 = total2 = None
    else:
        free2, total2 = mem_post_step

    used0 = (total0 - free0) if (free0 is not None and total0 is not None) else None
    used1 = (total1 - free1) if (free1 is not None and total1 is not None) else None
    used2 = (
        (total2 - free2)
        if (free2 is not None and total2 is not None)
        else (used_max if "used_max" in locals() else None)
    )

    res = {
        "total_gib": to_gib(total2),
        "used_gib_after_init": to_gib(used1),
        "used_gib_after_step": to_gib(used2),
        "delta_gib_from_start": to_gib(
            (used2 - used0) if (used2 is not None and used0 is not None) else 0
        ),
    }
    # Clear caches after the batch to avoid polluting the next measurement
    try:
        jax.clear_caches()
    except Exception:
        pass
    # Update host snapshot to latest state for reuse
    host_state = jax.device_get(state)
    shardings = jax.tree.map(lambda x: getattr(x, "sharding", None), state)
    return res, (host_state, shardings)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokens",
        type=str,
        default="512,4096",
        help="Comma-separated token lengths to test (buckets)",
    )
    ap.add_argument("--base-batch", type=int, default=8)
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument(
        "--jax-platform",
        choices=["cpu", "tpu", "gpu"],
        default=os.environ.get("JAX_PLATFORM_NAME", "cpu"),
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--batch-expand-depth", type=int, default=2)
    ap.add_argument("--batch-expand-max", type=int, default=8)
    ap.add_argument(
        "--split",
        type=str,
        default="train[:3000]",
        help="Dataset split to sample for preprocessing",
    )
    args = ap.parse_args()

    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    jax.config.update("jax_platform_name", args.jax_platform)

    token_sizes = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]

    print("=== GEMMA3_4B Batch Tuning + Memory Report ===")
    print("TPU per-replica memory before start:")
    log_tpu_memory("startup")

    trainstep = build_trainstep()

    # Use real preprocessing from Gemma script on LaTeX_OCR; restrict buckets
    sg.BUCKET_BOUNDARIES = tuple(sorted(set(token_sizes)))
    processor = AutoProcessor.from_pretrained(sg.MODEL_ID, trust_remote_code=True)

    def clamp_split(split: str, limit: int) -> str:
        pattern = r"^([^:]+)(?:\[:(\d+)\])?$"
        m = re.match(pattern, split.strip())
        if not m:
            return split
        name, current = m.group(1), m.group(2)
        limit = max(1, limit)
        if current is not None:
            limit = min(limit, int(current))
        return f"{name}[:{limit}]"

    limited_split = clamp_split(args.split, args.max_batch)
    if args.verbose and limited_split != args.split:
        print(
            f"Limiting split from '{args.split}' to '{limited_split}' (max {args.max_batch} samples)"
        )

    cache_dir = Path(sg.DATA_CACHE_DIR) / "batch_tuning_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "__".join(
        [
            sg.MODEL_ID.replace("/", "__"),
            limited_split.replace(":", "-"),
            "tokens=" + "-".join(str(t) for t in token_sizes),
            f"max={args.max_batch}",
        ]
    )
    cache_path = cache_dir / f"{cache_key}.pkl"

    if cache_path.exists():
        if args.verbose:
            print(f"Loading cached bucket samples from {cache_path}")
        with cache_path.open("rb") as fh:
            bucket_samples, stats = pickle.load(fh)
    else:
        real_samples, stats = sg._prepare_bucketed_samples(
            split=limited_split,
            processor=processor,
            seed=42,
            max_length=max(sg.BUCKET_BOUNDARIES),
            max_per_bucket=args.max_batch,
        )
        bucket_samples = {L: (real_samples.get(L, []) or []) for L in token_sizes}
        with cache_path.open("wb") as fh:
            pickle.dump((bucket_samples, stats), fh)
        if args.verbose:
            print(f"Cached bucket samples to {cache_path}")

    tuner = BucketBatchTuner(
        base_batch_size=args.base_batch,
        max_dynamic_batch=args.max_batch,
        safety_ratio=0.05,
        safety_min_bytes=(1 << 30),
        verbose=args.verbose,
        use_candidate_expansion=(
            args.batch_expand_depth > 0 and args.batch_expand_max > 0
        ),
        candidate_depth=args.batch_expand_depth,
        candidate_expansions=args.batch_expand_max,
    )
    tune_res = tuner.tune_for_buckets(
        bucket_samples=bucket_samples, trainstep=trainstep
    )
    if tune_res is None:
        print("No tuning result (tpu-info unavailable?). Exiting.")
        return

    print("\nPer-bucket best batch sizes:")
    for L in sorted(tune_res.best_per_bucket.keys()):
        per_sample = tune_res.per_sample_bytes.get(L)
        per_sample_gib = per_sample / (1024**3) if per_sample else 0.0
        print(
            f"  length={L}: best_batch={tune_res.best_per_bucket[L]} predicted={tune_res.predicted_caps.get(L)} per_sample≈{per_sample_gib:.2f} GiB"
        )

    host_states: Dict[int, Tuple[object, object]] = {}

    print("\nMeasuring memory at tuned batch sizes (per-replica):")
    for L in sorted(tune_res.best_per_bucket.keys()):
        B = tune_res.best_per_bucket[L]
        if not bucket_samples.get(L):
            print(f"  length={L} no sample available; skipping")
            continue
        stats, state_entry = measure_memory_from_bucket(
            trainstep,
            host_states.get(L),
            B,
            bucket_samples[L][0],
        )
        host_states[L] = state_entry
        print(
            f"  length={L} batch={B} -> used_after_step={stats['used_gib_after_step']:.2f} GiB / total={stats['total_gib']:.2f} GiB (Δ={stats['delta_gib_from_start']:.2f} GiB)"
        )


if __name__ == "__main__":
    main()
