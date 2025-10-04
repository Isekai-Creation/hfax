#!/usr/bin/env python3
"""Unified Gemma3 preprocessing/training: dynamic or fixed mode.

Use --fixed-max-length and --fixed-batch-size for the fixed (non-dynamic) path,
or omit them to enable dynamic bucketing and batch-size tuning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import requests
from datasets import load_dataset
from etils import enp
try:
    from tqdm.contrib.concurrent import process_map  # type: ignore
except Exception:
    process_map = None  # type: ignore
from jax import errors as jax_errors
from PIL import Image
from grain import python as grain
from transformers import AutoProcessor
from hfax.utils.batch_tuner import BucketBatchTuner
from hfax.utils.tpu_mem import log_tpu_memory, XLAMemoryTimer, per_replica_free_total
from hfax.utils.logging import logger

import hfax
from kauldron import kd

try:  # Optional progress bars
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None

# -----------------------------------------------------------------------------
# Constants & defaults
# -----------------------------------------------------------------------------

MODEL_ID = "unsloth/gemma-3-4b-it"
BUCKET_BOUNDARIES = (512, 1024, 2048, 4096)
BASE_BATCH_SIZE = 8  # tuned for 4K context
MAX_DYNAMIC_BATCH_DEFAULT = 128

DEFAULT_TRAIN_NUM_EPOCHS = 5
DEFAULT_EVAL_NUM_EPOCHS = 1
RNG_SEED = 42
DATASET_PATH = "unsloth/LaTeX_OCR"
DATA_CACHE_DIR = "/dev/shm/dataset_cache"
INSTRUCTION = "Convert the equation images to LaTeX equations."
CPU_COUNT = os.cpu_count() or 1
TRAIN_NUM_WORKERS = max(1, min(32, CPU_COUNT))
EVAL_NUM_WORKERS = max(1, min(32, CPU_COUNT))
CHECKPOINT_ROOT = Path("/dev/shm/kauldron_runs")

# Memory provider selection: only torch_xla in this environment
MEM_PROVIDER = "tpu_info"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-tpu-memory",
        action="store_true",
        help="Print TPU memory usage before training (requires torch_xla or tpu-info).",
    )
    parser.add_argument(
        "--log-batch-tuning",
        action="store_true",
        help="Verbose logs for dynamic batch-size tuning and memory usage.",
    )
    parser.add_argument(
        "--no-live-memory-query",
        action="store_true",
        help="Disable live TPU memory querying during batch tuning/logging (stability mode).",
    )
    parser.add_argument(
        "--skip-oom-validator",
        action="store_true",
        help="Skip JAX-based OOM validator during batch tuning (stability workaround).",
    )
    parser.add_argument(
        "--jax-platform",
        choices=["cpu", "tpu", "gpu"],
        default=os.environ.get("JAX_PLATFORM_NAME", "cpu"),
        help="Backend to target with JAX (default: %(default)s).",
    )
    parser.add_argument(
        "--mem-interval",
        type=float,
        default=0.0,
        help="If > 0, monitor TPU memory in background at this interval (seconds).",
    )
    # No external python needed; we use torch_xla directly in py3.12.
    parser.add_argument(
        "--bucket-boundaries",
        type=str,
        default=None,
        help="Comma-separated token boundaries for buckets (e.g. '512' or '512,1024').",
    )
    parser.add_argument(
        "--fixed-max-length",
        type=int,
        default=None,
        help="If set, use a single fixed max sequence length for padding/bucketing.",
    )
    parser.add_argument(
        "--force-batch-size",
        type=int,
        default=None,
        help="If set, bypass dynamic search and force this batch size for train/eval.",
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=None,
        help="Alias for --force-batch-size (fixed global batch size).",
    )
    parser.add_argument(
        "--max-dynamic-batch",
        type=int,
        default=MAX_DYNAMIC_BATCH_DEFAULT,
        help="Upper bound for dynamically scaled batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-expand-depth",
        type=int,
        default=2,
        help="Midpoint expansion depth for candidate batches (0 disables).",
    )
    parser.add_argument(
        "--batch-expand-max",
        type=int,
        default=8,
        help="Max number of midpoint expansions to add.",
    )
    parser.add_argument(
        "--train-split",
        default="train[:3000]",
        help="Dataset split for training (default: %(default)s).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=DEFAULT_TRAIN_NUM_EPOCHS,
        help="Number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-split",
        default="test[:300]",
        help="Dataset split for evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=DEFAULT_EVAL_NUM_EPOCHS,
        help="Number of evaluation epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help="Skip running an evaluation before training.",
    )
    parser.add_argument(
        "--sample-image-url",
        type=str,
        default=None,
        help="Optional image URL for chat sampling before/after training.",
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="What can you say about this image: <start_of_image>",
        help="Prompt used for chat sampling when an image URL is provided.",
    )
    parser.add_argument(
        "--skip-chat-samples",
        action="store_true",
        help="Skip pre/post chat sampling even if a sample image is provided.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=str,
        default=None,
        help="Path to append JSONL metric records (defaults to run directory).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, log summary metrics to this Weights & Biases project.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity (team).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


class PrecomputedSampleDataSource(grain.RandomAccessDataSource):
    """Random access data source for pre-batched samples."""

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, record_key: int) -> Dict[str, np.ndarray]:
        sample = self._samples[record_key]
        return {k: np.array(v, copy=True) for k, v in sample.items()}


def _collect_image_token_ids(processor: AutoProcessor, pad_token_id: int) -> set[int]:
    image_token_ids: set[int] = set()

    for key in ("boi_token", "eoi_token"):
        token_value = processor.tokenizer.special_tokens_map.get(key)
        if token_value is None:
            continue
        tokens = [token_value] if isinstance(token_value, str) else list(token_value)
        for tok in tokens:
            converted = processor.tokenizer.convert_tokens_to_ids(tok)
            if converted is not None:
                image_token_ids.add(int(converted))

    placeholder_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is not None:
        image_token_ids.add(int(placeholder_id))

    image_token_ids.discard(pad_token_id)
    # Historic Gemma image placeholder.
    image_token_ids.add(262144)
    return image_token_ids


def _process_chunk_worker(args: Tuple[str, str, str, str, Tuple[int, ...], int, str, int, int]) -> List[Tuple[int | None, Dict[str, np.ndarray] | None]]:
    """Picklable top-level worker for process_map.

    Args:
      args: (dataset_path, split, cache_dir, model_id, boundaries, max_length, instruction, start, end)
    Returns:
      List of (bucket_size, sample_dict) pairs (may include (None, None)).
    """
    dataset_path, split, cache_dir, model_id, boundaries, max_length, instruction, start, end = args
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    ds_local = load_dataset(dataset_path, split=split, cache_dir=cache_dir).with_format("python")
    pad_token_id = proc.tokenizer.pad_token_id
    image_token_ids = _collect_image_token_ids(proc, pad_token_id)
    out: List[Tuple[int | None, Dict[str, np.ndarray] | None]] = []
    for idx in range(start, end):
        ex = ds_local[idx]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": ex["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ex["text"]}],
            },
        ]
        rendered_prompt = proc.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        encoded = proc(
            text=[rendered_prompt],
            images=[ex["image"].convert("RGB")],
            return_tensors="np",
            padding=False,
            max_length=max_length,
            truncation=True,
        )
        input_ids = encoded["input_ids"][0].astype(np.int32)
        bucket_size = next((b for b in boundaries if len(input_ids) <= b), None)
        if bucket_size is None:
            out.append((None, None))
            continue
        padded_input = np.full((bucket_size,), pad_token_id, dtype=np.int32)
        padded_input[: len(input_ids)] = input_ids
        labels = padded_input.copy()
        if image_token_ids:
            labels[np.isin(labels, list(image_token_ids))] = -100
        labels[labels == pad_token_id] = -100
        targets = labels[:, None]
        loss_mask = (targets != -100).astype(np.int32)
        out.append((bucket_size, {"input": padded_input, "target": targets, "loss_mask": loss_mask}))
    return out


def _prepare_bucketed_samples(
    *,
    split: str,
    processor: AutoProcessor,
    seed: int,
    max_length: int = max(BUCKET_BOUNDARIES),
    max_per_bucket: int | None = None,
) -> Tuple[Dict[int, List[Dict[str, np.ndarray]]], Dict[str, Any]]:
    dataset = load_dataset(
        DATASET_PATH,
        split=split,
        cache_dir=DATA_CACHE_DIR,
    )
    dataset = dataset.with_format("python")

    pad_token_id = processor.tokenizer.pad_token_id
    image_token_ids = _collect_image_token_ids(processor, pad_token_id)

    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]] = {
        boundary: [] for boundary in BUCKET_BOUNDARIES
    }
    max_per_bucket = None if max_per_bucket is None else max(1, int(max_per_bucket))
    dropped_long = 0
    total_examples = len(dataset)
    max_workers = CPU_COUNT

    print(
        f"  [{split}] preprocessing {total_examples} examples "
        f"with {max_workers} workers..."
    )

    def _process_single(
        example: Dict[str, Any],
    ) -> Tuple[int | None, Dict[str, np.ndarray] | None]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": example["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["text"]}],
            },
        ]

        rendered_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        encoded = processor(
            text=[rendered_prompt],
            images=[example["image"].convert("RGB")],
            return_tensors="np",
            padding=False,
            max_length=max_length,
            truncation=True,
        )
        input_ids = encoded["input_ids"][0].astype(np.int32)
        bucket_size = next((b for b in BUCKET_BOUNDARIES if len(input_ids) <= b), None)
        if bucket_size is None:
            return None, None

        padded_input = np.full((bucket_size,), pad_token_id, dtype=np.int32)
        padded_input[: len(input_ids)] = input_ids

        labels = padded_input.copy()
        if image_token_ids:
            labels[np.isin(labels, list(image_token_ids))] = -100
        labels[labels == pad_token_id] = -100

        targets = labels[:, None]
        loss_mask = (targets != -100).astype(np.int32)

        return bucket_size, {
            "input": padded_input,
            "target": targets,
            "loss_mask": loss_mask,
        }

    def _bucket_full() -> bool:
        if max_per_bucket is None:
            return False
        return all(len(bucket_samples[b]) >= max_per_bucket for b in bucket_samples)

    # Prefer process-based parallelism for heavier CPU work if available, unless capped.
    if max_per_bucket is not None:
        rng = random.Random(seed)
        indices = list(range(total_examples))
        rng.shuffle(indices)
        for idx in indices:
            bucket_size, sample = _process_single(dataset[idx])
            if bucket_size is None or sample is None:
                dropped_long += 1
                continue
            if len(bucket_samples[bucket_size]) >= max_per_bucket:
                continue
            bucket_samples[bucket_size].append(sample)
            if _bucket_full():
                break
    elif process_map is not None and max_workers > 1:
        workers = max(1, min(max_workers, 32))
        chunk_size = max(64, total_examples // workers)
        chunks: List[Tuple[int, int]] = []
        s = 0
        while s < total_examples:
            e = min(total_examples, s + chunk_size)
            chunks.append((s, e))
            s = e
        task_args = [
            (DATASET_PATH, split, DATA_CACHE_DIR, MODEL_ID, BUCKET_BOUNDARIES, max_length, INSTRUCTION, st, ed)
            for (st, ed) in chunks
        ]
        for res in process_map(_process_chunk_worker, task_args, max_workers=workers, desc=f"[{split}] chunks", unit="chunk"):
            for bucket_size, sample in res:
                if bucket_size is None or sample is None:
                    dropped_long += 1
                    continue
                bucket_samples[bucket_size].append(sample)
    else:
        # Threaded fallback with smooth progress
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single, ex) for ex in dataset]
            if tqdm is not None:
                for fut in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total_examples,
                    desc=f"[{split}]",
                    unit="sample",
                    leave=True,
                    dynamic_ncols=True,
                    miniters=1,
                    mininterval=0.1,
                ):
                    bucket_size, sample = fut.result()
                    if bucket_size is None or sample is None:
                        dropped_long += 1
                        continue
                    bucket_samples[bucket_size].append(sample)
            else:
                for fut in concurrent.futures.as_completed(futures):
                    bucket_size, sample = fut.result()
                    if bucket_size is None or sample is None:
                        dropped_long += 1
                        continue
                    bucket_samples[bucket_size].append(sample)

    stats = {
        "total_examples": total_examples,
        "usable_examples": sum(len(v) for v in bucket_samples.values()),
        "dropped_long": dropped_long,
        "bucket_counts": {k: len(v) for k, v in bucket_samples.items()},
    }
    return bucket_samples, stats


# -----------------------------------------------------------------------------
# Batching helpers
# -----------------------------------------------------------------------------


def _compute_dynamic_batch_size(
    bucket_counts: Dict[int, int],
    *,
    max_dynamic_batch: int,
    validator: Optional[Callable[[int], bool]] = None,
) -> int:
    total_examples = sum(bucket_counts.values())
    if total_examples == 0:
        return BASE_BATCH_SIZE

    validation_cache: Dict[int, bool] = {}

    def _can_form_batches(batch_size: int) -> bool:
        if batch_size <= 0:
            return False
        total_batches = sum(count // batch_size for count in bucket_counts.values())
        if total_batches <= 0:
            return False
        if validator is None:
            return True
        if batch_size in validation_cache:
            return validation_cache[batch_size]
        result = validator(batch_size)
        validation_cache[batch_size] = result
        return result

    if not _can_form_batches(BASE_BATCH_SIZE):
        fallback = max(bucket_counts.values(), default=1)
        return max(1, min(max_dynamic_batch, fallback))

    candidates = list(range(BASE_BATCH_SIZE, max_dynamic_batch + BASE_BATCH_SIZE, BASE_BATCH_SIZE))
    candidates = [c for c in candidates if c <= max_dynamic_batch]
    if not candidates:
        candidates = [max_dynamic_batch]
    best = BASE_BATCH_SIZE
    left, right = 0, len(candidates) - 1
    while left <= right:
        mid = (left + right) // 2
        candidate = candidates[mid]
        if _can_form_batches(candidate):
            best = candidate
            left = mid + 1
        else:
            right = mid - 1

    return max(1, min(best, max_dynamic_batch))


def _create_batched_samples(
    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]],
    *,
    batch_size: int,
    seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[int, int], Dict[int, int]]:
    rng = random.Random(seed)
    batched: List[Dict[str, np.ndarray]] = []
    utilized: Dict[int, int] = {}
    dropped: Dict[int, int] = {}

    for boundary, samples in bucket_samples.items():
        rng.shuffle(samples)
        full_batches = len(samples) // batch_size
        used = full_batches * batch_size
        utilized[boundary] = used
        dropped[boundary] = len(samples) - used

        for idx in range(full_batches):
            chunk = samples[idx * batch_size : (idx + 1) * batch_size]
            batched.append(
                {
                    key: np.stack([sample[key] for sample in chunk], axis=0)
                    for key in chunk[0]
                }
            )

    rng.shuffle(batched)
    return batched, utilized, dropped


def _make_worst_case_batch(
    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]], batch_size: int
) -> Optional[Dict[str, np.ndarray]]:
    template: Optional[Dict[str, np.ndarray]] = None
    for boundary in sorted(bucket_samples.keys(), reverse=True):
        samples = bucket_samples.get(boundary, [])
        if samples:
            template = samples[0]
            break
    if template is None:
        for samples in bucket_samples.values():
            if samples:
                template = samples[0]
                break
    if template is None:
        return None

    def _tile(array: np.ndarray) -> np.ndarray:
        return np.stack([array] * batch_size, axis=0)

    return {key: _tile(value) for key, value in template.items()}


def _batch_to_elem_spec(batch: Dict[str, np.ndarray]) -> Dict[str, enp.ArraySpec]:
    return {
        key: enp.ArraySpec(shape=value.shape, dtype=value.dtype) for key, value in batch.items()
    }


def _make_batch_validator(
    *,
    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]],
    trainstep: "kd.train.train_step.TrainStep",
) -> Callable[[int], bool]:
    cache: Dict[int, bool] = {}

    def _validator(batch_size: int) -> bool:
        if batch_size in cache:
            return cache[batch_size]
        batch = _make_worst_case_batch(bucket_samples, batch_size)
        if batch is None:
            cache[batch_size] = False
            return False

        elem_spec = _batch_to_elem_spec(batch)
        batch_jnp = {key: jnp.asarray(value) for key, value in batch.items()}

        try:
            state = trainstep.init(elem_spec=elem_spec)
            state, _ = trainstep.step(state, batch_jnp)
            jax.block_until_ready(state.step)
            cache[batch_size] = True
            return True
        except Exception as exc:
            message = str(exc)
            oom_markers = (
                "RESOURCE_EXHAUSTED",
                "Out of memory",
                "OOM",
                "insufficient memory",
                "HbmAllocator",
                "Resource exhausted",
            )
            if any(m in message for m in oom_markers) or "memory" in message.lower():
                cache[batch_size] = False
                return False
            raise
        finally:
            jax.clear_caches()

    return _validator


def _maybe_download_image(url: Optional[str]) -> Optional[np.ndarray]:
    if not url:
        return None
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        print(f"Sample image fetch failed ({exc}); continuing without image.")
        return None


def _tree_to_floats(mapping: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if mapping is None:
        return {}
    floats: Dict[str, float] = {}
    for key, value in mapping.items():
        try:
            floats[key] = float(np.array(value))
        except (TypeError, ValueError):
            continue
    return floats


def _append_metrics_record(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


@dataclass
class WandbHandle:
    run: Any


def _maybe_init_wandb(args: argparse.Namespace, run_dir: Path) -> Optional[WandbHandle]:
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "wandb is not installed but --wandb-project was provided."
        ) from exc

    config = {
        "max_dynamic_batch": args.max_dynamic_batch,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "log_tpu_memory": args.log_tpu_memory,
        
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=str(run_dir),
        config=config,
    )
    return WandbHandle(run=run)


def _finish_wandb(handle: Optional[WandbHandle]) -> None:
    if handle is None:
        return
    handle.run.finish()


def _log_metrics(
    *,
    label: str,
    aux_output: Any,
    jsonl_path: Optional[Path],
    wandb_handle: Optional[WandbHandle],
    step: int,
) -> None:
    loss_values = _tree_to_floats(getattr(aux_output, "loss_values", None))
    metric_values = _tree_to_floats(getattr(aux_output, "metric_values", None))

    if loss_values:
        print(f"{label} losses:")
        for key, value in sorted(loss_values.items()):
            print(f"  {key}: {value:.6f}")
    if metric_values:
        print(f"{label} metrics:")
        for key, value in sorted(metric_values.items()):
            print(f"  {key}: {value:.6f}")

    record = {
        "label": label,
        "step": step,
        "losses": loss_values,
        "metrics": metric_values,
    }
    if jsonl_path is not None:
        _append_metrics_record(jsonl_path, record)

    if wandb_handle is not None and (loss_values or metric_values):
        wandb_data = {f"{label}/loss/{k}": v for k, v in loss_values.items()}
        wandb_data.update({f"{label}/metric/{k}": v for k, v in metric_values.items()})
        wandb_data["step"] = step
        wandb_handle.run.log(wandb_data)


def _run_evaluator(
    *,
    trainer: kd.train.Trainer,
    state: Any,
    label: str,
    jsonl_path: Optional[Path],
    wandb_handle: Optional[WandbHandle],
) -> None:
    evaluator = trainer.evals.get("eval")
    if evaluator is None:
        print(f"{label}: evaluator not configured, skipping.")
        return
    print(f"\nRunning {label} evaluation...")
    aux_state = evaluator.evaluate(state=state, step=int(state.step))
    if aux_state is None:
        print(f"{label}: evaluator returned no metrics.")
        return
    aux_output = aux_state.compute(flatten=False)
    _log_metrics(
        label=label,
        aux_output=aux_output,
        jsonl_path=jsonl_path,
        wandb_handle=wandb_handle,
        step=int(state.step),
    )


def _run_chat_sample(
    *,
    label: str,
    model: Any,
    params: Any,
    tokenizer: Any,
    prompt: str,
    image: Optional[np.ndarray],
    jsonl_path: Optional[Path],
    wandb_handle: Optional[WandbHandle],
    run_dir: Path,
) -> None:
    sampler = hfax.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
    )
    kwargs = {}
    if image is not None:
        kwargs["images"] = [image]
    print(f"\nRunning {label} chat sample...")
    response = sampler.chat(prompt, **kwargs)
    print(f"Prompt: {prompt}")
    if image is not None:
        print("Image input provided.")
    print(f"Response ({label}): {response}")

    object.__setattr__(sampler, "last_state", None)
    sampler.turns.clear()
    try:
        jax.clear_caches()
        print("Cleared JAX compilation cache.")
    except AttributeError:
        pass

    record = {
        "label": f"{label}_chat",
        "prompt": prompt,
        "response": response,
    }
    if jsonl_path is not None:
        _append_metrics_record(jsonl_path, record)
    if wandb_handle is not None:
        wandb_handle.run.log({
            f"chat/{label}/prompt": prompt,
            f"chat/{label}/response": response,
        })

    chat_path = run_dir / f"{label}_chat.txt"
    with chat_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Prompt:\n{prompt}\n\nResponse:\n{response}\n")


# -----------------------------------------------------------------------------
# TPU memory diagnostics
# -----------------------------------------------------------------------------


def _maybe_log_tpu_memory(*, log_memory: bool, label: str = "event") -> None:
    if not log_memory:
        return
    try:
        log_tpu_memory(label)
    except Exception as exc:
        logger.warning("[mem] log failed: %s", exc)


def _log_tpu_memory_event(*, enabled: bool, label: str) -> None:
    if not enabled:
        return
    _maybe_log_tpu_memory(log_memory=True, label=label)


# -----------------------------------------------------------------------------
# TPU memory–driven batch upper bound (via tpu-info CLI or fallbacks)
# -----------------------------------------------------------------------------


def _tpu_memory_free_and_total() -> Optional[Tuple[int, int]]:
    """Return (free_bytes, total_bytes) per replica if available.

    Uses tpu_info when available and falls back to torch_xla. Returns None if no
    provider can be used in the current environment.
    """
    # Only attempt on TPU backends.
    try:
        backend = jax.default_backend()
        if backend != "tpu":
            return None
    except Exception:
        return None

    try:
        return per_replica_free_total()
    except Exception:
        return None


def _tpuinfo_cli_query(*, spmd: bool) -> Optional[Tuple[int, int]]:
    # Kept for API compatibility; delegate to per_replica_free_total when spmd.
    if spmd:
        return per_replica_free_total()
    return None


def _tpu_info_query_subprocess() -> Optional[Tuple[int, int]]:
    # Legacy alias; use per_replica_free_total.
    return per_replica_free_total()


def _estimate_upper_batch_from_tpu_memory(
    *,
    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]],
    validator: Callable[[int], bool],
    max_dynamic_batch: int,
    verbose: bool = False,
) -> Optional[int]:
    """Estimate a safe upper bound for batch size using live TPU memory.

    Strategy:
    - Validate once at BASE_BATCH_SIZE to ensure program compiles and runs.
    - Read free memory (per replica) via tpu_info/torch_xla.
    - Approximate per-sample memory as reserved_at_base / BASE_BATCH_SIZE.
      This is conservative (includes fixed overhead) but avoids OOM.
    - Predict upper bound and verify with validator; if it fails, binary search
      downward until success.
    Returns None if no memory provider is available.
    """
    mem = _tpu_memory_free_and_total()
    if mem is None:
        if verbose:
            print("[batch-tuning] No TPU memory provider available; using OOM-guided search only.")
        return None
    if verbose:
        print(f"[batch-tuning] Pre-step memory: free={mem[0]} bytes, total={mem[1]} bytes")

    # First, validate at BASE_BATCH_SIZE and measure reserved after the step.
    # We obtain reserved by re-querying total-free.
    if verbose:
        print(f"[batch-tuning] Validating BASE_BATCH_SIZE={BASE_BATCH_SIZE}...")
    if not validator(BASE_BATCH_SIZE):  # pragma: no cover - guarded by validator
        if verbose:
            print("[batch-tuning] Failed at BASE_BATCH_SIZE; aborting memory-based cap.")
        return None
    post_mem = _tpu_memory_free_and_total()
    if post_mem is None:
        if verbose:
            print("[batch-tuning] Post-step memory read failed; aborting memory-based cap.")
        return None
    free_bytes, total_bytes = post_mem
    reserved_bytes = max(0, total_bytes - free_bytes)
    if verbose:
        print(
            f"[batch-tuning] Post-step memory: free={free_bytes}, total={total_bytes}; "
            f"reserved≈{reserved_bytes}"
        )

    # Conservative per-sample estimate.
    per_sample = max(1, reserved_bytes // BASE_BATCH_SIZE)
    if verbose:
        print(f"[batch-tuning] Estimated per-sample bytes: {per_sample}")

    # Leave a safety margin: 5% of total or 1 GiB, whichever is larger.
    safety = max(int(total_bytes * 0.05), 1 << 30)
    headroom = max(0, free_bytes - safety)
    if headroom <= 0:
        if verbose:
            print(
                f"[batch-tuning] No headroom after safety margin (safety={safety}); "
                f"returning BASE_BATCH_SIZE={BASE_BATCH_SIZE}"
            )
        return BASE_BATCH_SIZE

    predicted = int(headroom // per_sample)
    # Round to multiple of BASE_BATCH_SIZE.
    if predicted <= 0:
        if verbose:
            print("[batch-tuning] Predicted cap <= 0; returning BASE_BATCH_SIZE.")
        return BASE_BATCH_SIZE
    predicted = (predicted // BASE_BATCH_SIZE) * BASE_BATCH_SIZE
    predicted = max(BASE_BATCH_SIZE, min(predicted, max_dynamic_batch))
    if verbose:
        print(
            f"[batch-tuning] Safety={safety}, headroom={headroom}; predicted cap={predicted}; "
            f"flag max={max_dynamic_batch}"
        )

    # Verify and refine downward if needed.
    lo, hi = BASE_BATCH_SIZE, predicted
    best = BASE_BATCH_SIZE
    while lo <= hi:
        mid = ((lo + hi) // (2 * BASE_BATCH_SIZE)) * BASE_BATCH_SIZE
        mid = max(BASE_BATCH_SIZE, mid)
        if verbose:
            print(f"[batch-tuning] Try mid={mid} ...", end="")
        if validator(mid):
            best = mid
            lo = mid + BASE_BATCH_SIZE
            if verbose:
                print("OK")
        else:
            hi = mid - BASE_BATCH_SIZE
            if verbose:
                print("OOM/NO")
    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()
    args = parse_args()

    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    jax.config.update("jax_platform_name", args.jax_platform)

    # Using tpu-info CLI; no external python interpreter needed.

    print("=== Gemma Kauldron Fine-tuning Setup ===")

    print("\n1. Initializing tokenizer...")
    tokenizer = hfax.text.Gemma3Tokenizer()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✓ Tokenizer initialized.")

    # Normalize fixed-batch alias
    if getattr(args, 'fixed_batch_size', None) is not None and args.force_batch_size is None:
        args.force_batch_size = int(args.fixed_batch_size)

    # Optionally override bucket boundaries or switch to fixed-max mode
    global BUCKET_BOUNDARIES
    if args.fixed_max_length is not None:
        BUCKET_BOUNDARIES = (int(args.fixed_max_length),)
        print(f"[config] Fixed max length enabled: {BUCKET_BOUNDARIES[0]} tokens.")
    elif args.bucket_boundaries:
        try:
            BUCKET_BOUNDARIES = tuple(int(x.strip()) for x in args.bucket_boundaries.split(",") if x.strip())
            if not BUCKET_BOUNDARIES:
                raise ValueError
        except Exception:
            raise ValueError("--bucket-boundaries must be a comma-separated list of ints, e.g. '512' or '512,1024'")

    print("\n2. Preprocessing and bucketing dataset...")
    if args.log_tpu_memory:
        try:
            log_tpu_memory("preprocess/start")
        except Exception as exc:
            logger.warning("[mem] preprocess start log failed: %s", exc)
    train_bucket_samples, train_stats = _prepare_bucketed_samples(
        split=args.train_split,
        processor=processor,
        seed=RNG_SEED,
        max_length=max(BUCKET_BOUNDARIES),
    )
    eval_bucket_samples, eval_stats = _prepare_bucketed_samples(
        split=args.eval_split,
        processor=processor,
        seed=RNG_SEED + 1,
        max_length=max(BUCKET_BOUNDARIES),
    )
    if args.log_tpu_memory:
        try:
            log_tpu_memory("preprocess/end")
        except Exception as exc:
            logger.warning("[mem] preprocess end log failed: %s", exc)

    print(
        "Train bucket counts:",
        {k: v for k, v in train_stats["bucket_counts"].items() if v},
    )
    print(
        "Eval bucket counts:",
        {k: v for k, v in eval_stats["bucket_counts"].items() if v},
    )
    print(
        f"Dropped {train_stats['dropped_long']} long examples (train)"
        f" and {eval_stats['dropped_long']} (eval)."
    )
    print(
        f"  Training data workers: {TRAIN_NUM_WORKERS},"
        f" eval data workers: {EVAL_NUM_WORKERS}"
    )

    print("\n3. Defining Gemma 3 4B model...")
    model = hfax.nn.Gemma3_4B(tokens="batch.input")
    print("✓ Model defined.")

    print("\n4. Defining loss function (SoftmaxCrossEntropy)...")
    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits="preds.logits",
        labels="batch.target",
        mask="batch.loss_mask",
    )
    print("✓ Loss function defined.")

    validation_optimizer = optax.adafactor(learning_rate=1e-3)
    # Provide rng_streams to avoid unresolved ROOT_CFG_REF during init.
    rng_streams = kd.train.RngStreams(
        [kd.train.RngStream(name="dropout", init=True, train=True, eval=False, per_step=True)],
        seed=RNG_SEED,
    )
    validation_trainstep = kd.train.train_step.TrainStep(
        model=model,
        optimizer=validation_optimizer,
        rng_streams=rng_streams,
        sharding=kd.train.train_step.sharding_lib.ShardingStrategy(),
        init_transform=kd.train.train_step.partial_loader.NoopTransform(),
        aux=kd.train.auxiliaries.Auxiliaries(losses={"loss": loss}),
    )

    if args.skip_oom_validator:
        print("[batch-tuning] Skipping JAX-based OOM validator per --skip-oom-validator.")
        train_validator = None
        eval_validator = None
    else:
        train_validator = _make_batch_validator(
            bucket_samples=train_bucket_samples,
            trainstep=validation_trainstep,
        )
        eval_validator = _make_batch_validator(
            bucket_samples=eval_bucket_samples,
            trainstep=validation_trainstep,
        )

    upper_hint: Optional[int] = None
    if args.force_batch_size is not None:
        train_batch_size = eval_batch_size = max(1, int(args.force_batch_size))
        print(f"[batch-tuning] Forcing batch size to {train_batch_size} via --force-batch-size.")
    else:
        # Prefer native tuner using TPU memory; falls back to validator-only.
        tuned_global: Optional[int] = None
        if not args.no_live_memory_query:
            tuner = BucketBatchTuner(
                base_batch_size=BASE_BATCH_SIZE,
                max_dynamic_batch=args.max_dynamic_batch,
                safety_ratio=0.05,
                safety_min_bytes=(1 << 30),
                verbose=(args.log_batch_tuning or args.log_tpu_memory),
                use_candidate_expansion=(args.batch_expand_depth > 0 and args.batch_expand_max > 0),
                candidate_depth=max(0, int(args.batch_expand_depth)),
                candidate_expansions=max(0, int(args.batch_expand_max)),
            )
            tune_res = tuner.tune_for_buckets(
                bucket_samples=train_bucket_samples,
                trainstep=validation_trainstep,
            )
            if tune_res is not None and tune_res.best_per_bucket:
                print("[batch-tuning] per-bucket best sizes:", tune_res.best_per_bucket)
                tuned_global = min(tune_res.best_per_bucket.values())
                print(f"[batch-tuning] derived global batch size: {tuned_global}")
        if tuned_global is None:
            # Fallback to validator-driven global search
            train_batch_size = _compute_dynamic_batch_size(
                train_stats["bucket_counts"],
                max_dynamic_batch=args.max_dynamic_batch,
                validator=train_validator,
            )
            eval_batch_size = _compute_dynamic_batch_size(
                eval_stats["bucket_counts"],
                max_dynamic_batch=args.max_dynamic_batch,
                validator=eval_validator,
            )
        else:
            train_batch_size = eval_batch_size = tuned_global

    train_batched_samples, train_utilized, train_dropped = _create_batched_samples(
        train_bucket_samples,
        batch_size=train_batch_size,
        seed=RNG_SEED,
    )
    if not train_batched_samples:
        raise ValueError(
            "Dynamic batch size resulted in zero training batches. "
            "Lower --max-dynamic-batch or increase data."
        )

    eval_batched_samples, eval_utilized, eval_dropped = _create_batched_samples(
        eval_bucket_samples,
        batch_size=eval_batch_size,
        seed=RNG_SEED + 1,
    )
    if not eval_batched_samples:
        raise ValueError(
            "Dynamic batch size resulted in zero eval batches. "
            "Lower --max-dynamic-batch or increase data."
        )

    train_stats.update(
        {
            "bucket_utilized": train_utilized,
            "bucket_dropped": train_dropped,
            "usable_examples": sum(train_utilized.values()),
            "batch_size": train_batch_size,
            "steps_per_epoch": len(train_batched_samples),
        }
    )
    eval_stats.update(
        {
            "bucket_utilized": eval_utilized,
            "bucket_dropped": eval_dropped,
            "usable_examples": sum(eval_utilized.values()),
            "batch_size": eval_batch_size,
            "steps_per_epoch": len(eval_batched_samples),
        }
    )

    print("Train utilized per bucket:", {k: v for k, v in train_utilized.items() if v})
    if any(train_dropped.values()):
        print(
            "Train dropped due to batching:",
            {k: v for k, v in train_dropped.items() if v},
        )
    print(f"Train usable examples (batched): {train_stats['usable_examples']}")

    print("Eval utilized per bucket:", {k: v for k, v in eval_utilized.items() if v})
    if any(eval_dropped.values()):
        print(
            "Eval dropped due to batching:",
            {k: v for k, v in eval_dropped.items() if v},
        )
    print(f"Eval usable examples (batched): {eval_stats['usable_examples']}")

    target_tokens = train_stats["batch_size"] * max(BUCKET_BOUNDARIES)
    train_tokens_total = sum(
        boundary * count for boundary, count in train_utilized.items()
    )
    eval_tokens_total = sum(
        boundary * count for boundary, count in eval_utilized.items()
    )
    print(
        f"Computed train batch size: {train_batch_size} (~{target_tokens} tokens target)"
    )
    print(f"Computed eval batch size: {eval_batch_size}")
    print(
        f"Approx train tokens/step: {train_tokens_total / max(1, len(train_batched_samples)):.0f}"
    )
    print(
        f"Approx eval tokens/step: {eval_tokens_total / max(1, len(eval_batched_samples)):.0f}"
    )

    train_ds = kd.data.py.DataSource(
        data_source=PrecomputedSampleDataSource(train_batched_samples),
        shuffle=False,
        batch_size=None,
        num_epochs=args.train_epochs,
        num_workers=TRAIN_NUM_WORKERS,
    )
    available_batches = len(train_ds)
    expected_per_epoch = len(train_batched_samples)
    if available_batches != expected_per_epoch * args.train_epochs:
        raise ValueError(
            "Mismatch between dataset batches and expected batches: "
            f"len(ds)={available_batches} vs expected={expected_per_epoch * args.train_epochs}"
        )
    print(
        "len(ds):",
        available_batches,
        "batches across",
        args.train_epochs,
        "epochs",
    )
    print(
        f"Planned training steps: {available_batches}"
        f" ({expected_per_epoch} batches per epoch)"
    )
    print("✓ Data pipeline created.")

    eval_ds = kd.data.py.DataSource(
        data_source=PrecomputedSampleDataSource(eval_batched_samples),
        shuffle=False,
        batch_size=None,
        num_epochs=args.eval_epochs,
        num_workers=EVAL_NUM_WORKERS,
    )

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = CHECKPOINT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Checkpoint run directory: {run_dir}")

    metrics_jsonl_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else run_dir / "metrics.jsonl"
    )
    wandb_handle = _maybe_init_wandb(args, run_dir)
    sample_image = None
    if not args.skip_chat_samples:
        sample_image = _maybe_download_image(args.sample_image_url)

    print("\n5. Configuring the Kauldron trainer...")
    trainer = kd.train.Trainer(
        seed=RNG_SEED,
        workdir=str(run_dir),
        checkpointer=kd.ckpts.Checkpointer(save_interval_steps=600),
        train_ds=train_ds,
        evals={
            "eval": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(600),
                ds=eval_ds,
            )
        },
        model=model,
        init_transform=hfax.ckpts.LoadCheckpoint(
            path=hfax.ckpts.CheckpointPath.GEMMA3_4B_IT,
        ),
        num_train_steps=len(train_ds),
        train_losses={"loss": loss},
        optimizer=optax.adafactor(learning_rate=1e-3),
    )
    print("✓ Trainer configured.")

    # Pre-training detailed memory log
    _log_tpu_memory_event(
        enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
        label="pre_training",
        
    )

    pre_state = None
    if not args.skip_pre_eval or sample_image is not None:
        pre_state = trainer.init_state()
        if not args.skip_pre_eval:
            _log_tpu_memory_event(
                enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
                label="pre_training_eval/pre",
                
            )
            _run_evaluator(
                trainer=trainer,
                state=pre_state,
                label="pre_training_eval",
                jsonl_path=metrics_jsonl_path,
                wandb_handle=wandb_handle,
            )
            _log_tpu_memory_event(
                enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
                label="pre_training_eval/post",
                
            )
        if sample_image is not None:
            _log_tpu_memory_event(
                enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
                label="chat_sample/pre_training/pre",
                
            )
            _run_chat_sample(
                label="pre_training",
                model=model,
                params=pre_state.params,
                tokenizer=tokenizer,
                prompt=args.sample_prompt,
                image=sample_image,
                jsonl_path=metrics_jsonl_path,
                wandb_handle=wandb_handle,
                run_dir=run_dir,
            )
            _log_tpu_memory_event(
                enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
                label="chat_sample/pre_training/post",
            )

    print(f"\n6. Starting training for {len(train_ds)} steps...")
    state = None
    aux = None
    mem_timer = None
    if getattr(args, "mem_interval", 0.0) and args.mem_interval > 0:
        try:
            mem_timer = XLAMemoryTimer(interval=float(args.mem_interval))
            mem_timer.start()
            logger.info("[mem] background timer started (interval=%.2fs)", args.mem_interval)
        except Exception as exc:
            logger.warning("[mem] timer start failed: %s", exc)
    try:
        state, aux = trainer.train()
        print("\n✓ Training finished.")
    except StopIteration:
        print("\nTraining iterator exhausted (StopIteration); treating as finished.")
    finally:
        if mem_timer is not None:
            try:
                stats = mem_timer.stop(format_mb=True)
                logger.info(
                    "[mem] peak usage: total_used=%.2f GiB, peak_per_device=%.2f GiB, total=%.2f GiB, per_device_total=%.2f GiB",
                    stats.get("max_mem", 0.0),
                    stats.get("max_mem_per_device", 0.0),
                    stats.get("mem_total", 0.0),
                    stats.get("mem_per_device", 0.0),
                )
            except Exception as exc:
                logger.warning("[mem] timer stop failed: %s", exc)

    # Post-training detailed memory log
    if state is not None:
        _log_tpu_memory_event(
            enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
            label="post_training",
            
        )

    if aux is not None:
        try:
            aux_output = aux.compute(flatten=False)
        except AttributeError:
            print("Training auxiliaries have no compute method; skipping logging.")
        else:
            _log_metrics(
                label="training",
                aux_output=aux_output,
                jsonl_path=metrics_jsonl_path,
                wandb_handle=wandb_handle,
                step=int(getattr(state, 'step', 0) or 0),
            )

    print("\n7. Running evaluations and chat sampling...")
    if state is not None:
        _log_tpu_memory_event(
            enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
            label="post_training_eval/pre",
            
        )
        _run_evaluator(
            trainer=trainer,
            state=state,
            label="post_training_eval",
            jsonl_path=metrics_jsonl_path,
            wandb_handle=wandb_handle,
        )
        if args.log_tpu_memory:
            try:
                log_tpu_memory("post_training_eval")
            except Exception as exc:
                logger.warning("[mem] post_training_eval log failed: %s", exc)
        _log_tpu_memory_event(
            enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
            label="post_training_eval/post",
            
        )

    if sample_image is not None:
        _run_chat_sample(
            label="post_training",
            model=model,
            params=state.params,
            tokenizer=tokenizer,
            prompt=args.sample_prompt,
            image=sample_image,
            jsonl_path=metrics_jsonl_path,
            wandb_handle=wandb_handle,
            run_dir=run_dir,
        )
        if args.log_tpu_memory:
            try:
                log_tpu_memory("chat_sample/post_training")
            except Exception as exc:
                logger.warning("[mem] chat_sample log failed: %s", exc)

    elapsed_seconds = time.time() - start_time
    print(
        "Total run time: "
        f"{int(elapsed_seconds // 3600):02d}:{int((elapsed_seconds % 3600) // 60):02d}:{int(elapsed_seconds % 60):02d}"
    )
    runtime_record = {"label": "runtime", "elapsed_seconds": elapsed_seconds}
    if metrics_jsonl_path is not None:
        _append_metrics_record(metrics_jsonl_path, runtime_record)
    if wandb_handle is not None:
        wandb_handle.run.log({"runtime/elapsed_seconds": elapsed_seconds})

    _finish_wandb(wandb_handle)
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
