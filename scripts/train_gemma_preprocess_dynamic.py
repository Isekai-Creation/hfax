#!/usr/bin/env python3
"""Fine-tune Gemma3 4B with dynamic batching driven by bucket statistics."""

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
from jax import errors as jax_errors
from PIL import Image
from grain import python as grain
from transformers import AutoProcessor

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
MEM_PROVIDER = "torch_xla"

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
        "--xla-spmd",
        action="store_true",
        help="When logging TPU memory, adjust readings for XLA SPMD/FSDP mode.",
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
    # No external python needed; we use torch_xla directly in py3.12.
    parser.add_argument(
        "--bucket-boundaries",
        type=str,
        default=None,
        help="Comma-separated token boundaries for buckets (e.g. '512' or '512,1024').",
    )
    parser.add_argument(
        "--force-batch-size",
        type=int,
        default=None,
        help="If set, bypass dynamic search and force this batch size for train/eval.",
    )
    parser.add_argument(
        "--max-dynamic-batch",
        type=int,
        default=MAX_DYNAMIC_BATCH_DEFAULT,
        help="Upper bound for dynamically scaled batch size (default: %(default)s).",
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


def _prepare_bucketed_samples(
    *,
    split: str,
    processor: AutoProcessor,
    seed: int,
    max_length: int = max(BUCKET_BOUNDARIES),
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator: Iterable[Tuple[int | None, Dict[str, np.ndarray] | None]] = (
            executor.map(_process_single, dataset)
        )
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=total_examples,
                desc=f"[{split}]",
                unit="sample",
                leave=True,
            )

        for bucket_size, sample in iterator:
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
        "xla_spmd": args.xla_spmd,
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


def _maybe_log_tpu_memory(*, log_memory: bool, xla_spmd: bool) -> None:
    if not log_memory:
        return

    mem_total = mem_used = mem_free = 0

    # Try external Python 3.10 helper first
    ext = _tpuinfo_cli_query(spmd=xla_spmd)
    if ext is not None:
        mem_free, mem_total = ext
        mem_used = max(0, mem_total - mem_free)
    else:
        if xla_spmd:
            sub = _tpu_info_query_subprocess()
            if sub is not None:
                mem_free, mem_total = sub
                mem_used = max(0, mem_total - mem_free)
            else:
                # Fallback to torch_xla single-device reading.
                try:
                    import torch_xla.core.xla_model as xm  # type: ignore
                    device = xm.xla_device()
                    mem_info = xm.get_memory_info(device)
                    mem_used = mem_info.get("bytes_used", 0)
                    mem_total = mem_info.get("bytes_limit", 0)
                    mem_free = max(0, mem_total - mem_used)
                    print("tpu-info unavailable; fell back to torch_xla memory for logging.")
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to query TPU memory: {exc}")
                    return
        else:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                device = xm.xla_device()
                mem_info = xm.get_memory_info(device)
                mem_used = mem_info.get("bytes_used", 0)
                mem_total = mem_info.get("bytes_limit", 0)
                mem_free = max(0, mem_total - mem_used)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"Failed to query torch_xla memory info: {exc}")
                return

    if mem_total:
        label = (
            f"TPU memory (per-replica): used={mem_used / 1e9:.2f} GB, "
            f"free={mem_free / 1e9:.2f} GB, total={mem_total / 1e9:.2f} GB"
            if xla_spmd
            else f"TPU memory usage: {mem_used / 1e9:.2f} / {mem_total / 1e9:.2f} GB (used/total)"
        )
        print(label)


def _log_tpu_memory_event(*, enabled: bool, label: str, xla_spmd: bool) -> None:
    if not enabled:
        return
    print(f"[mem] {label}:")
    _maybe_log_tpu_memory(log_memory=True, xla_spmd=xla_spmd)


# -----------------------------------------------------------------------------
# TPU memory–driven batch upper bound (via tpu-info CLI or fallbacks)
# -----------------------------------------------------------------------------


def _tpu_memory_free_and_total(xla_spmd: bool) -> Optional[Tuple[int, int]]:
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

    # Prefer external Python 3.10 helper script first.
    res = _tpuinfo_cli_query(spmd=xla_spmd)
    if res is not None:
        return res

    # Then try in-process tpu_info (may crash on py3.12; kept as best-effort)
    res = _tpu_info_query_subprocess()
    if res is not None:
        return res

    # Fallback: torch_xla (single device view; free = limit - used)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        device = xm.xla_device()
        mem_info = xm.get_memory_info(device)
        total = int(mem_info.get("bytes_limit", 0))
        used = int(mem_info.get("bytes_used", 0))
        free = max(0, total - used)
        if total > 0:
            return free, total
    except Exception:
        pass

    return None


def _tpuinfo_cli_query(*, spmd: bool) -> Optional[Tuple[int, int]]:
    """Query memory using the installed `tpu-info` CLI (Python 3.12 safe).

    Tries JSON mode first (newer tpu-info), then falls back to parsing text.
    Returns (free_bytes, total_bytes) per replica when spmd=True, else device totals.
    """
    import subprocess, json as _json, re

    # Try JSON output (newer tpu-info versions support single-metric mode).
    try:
        proc = subprocess.run(
            ["tpu-info", "--metric", "hbm_usage", "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
            data = _json.loads(proc.stdout)
            # Expect structure with a list of chips each having total/used (bytes).
            chips = data.get("chips") or data.get("devices") or []
            totals = []
            useds = []
            for c in chips:
                t = int(c.get("total_bytes") or c.get("total") or 0)
                u = int(c.get("used_bytes") or c.get("used") or 0)
                if t > 0:
                    totals.append(t)
                    useds.append(u)
            if totals:
                if spmd:
                    per_total = min(totals)
                    per_used = max(useds)
                    return max(0, per_total - per_used), per_total
                else:
                    total = sum(totals)
                    used = sum(useds)
                    return max(0, total - used), total
    except Exception:
        pass

    # Fallback: parse human output.
    try:
        proc = subprocess.run(
            ["tpu-info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode != 0:
            return None
        txt = proc.stdout
        # Find lines like "0.00 GiB / 31.75 GiB"; collect per-chip totals/used.
        gib_line = re.findall(r"([0-9]+\.[0-9]+)\s*GiB\s*/\s*([0-9]+\.[0-9]+)\s*GiB", txt)
        totals = []
        useds = []
        for used_gib, total_gib in gib_line:
            t = int(float(total_gib) * (1 << 30))
            u = int(float(used_gib) * (1 << 30))
            if t > 0:
                totals.append(t)
                useds.append(u)
        if totals:
            if spmd:
                per_total = min(totals)
                per_used = max(useds)
                return max(0, per_total - per_used), per_total
            else:
                total = sum(totals)
                used = sum(useds)
                return max(0, total - used), total
    except Exception:
        return None
    return None


def _tpu_info_query_subprocess() -> Optional[Tuple[int, int]]:
    """Query per-replica (free, total) via tpu_info in a separate process.

    Returns None if the helper fails or tpu_info is not present/works.
    """
    import subprocess, textwrap, json as _json

    code = textwrap.dedent(
        r'''
import json
try:
  from tpu_info import device as tpu_device
  from tpu_info import metrics as tpu_metrics
  chip_type, count = tpu_device.get_local_chips()
  if not chip_type or not count:
    raise SystemExit(2)
  device_usage = tpu_metrics.get_chip_usage(chip_type)
  per_chip = [(int(d.total_memory), int(d.memory_usage)) for d in device_usage]
  if not per_chip:
    raise SystemExit(3)
  free = min(t - u for (t, u) in per_chip)
  total = min(t for (t, _u) in per_chip)
  print(json.dumps({'free': free, 'total': total}))
  raise SystemExit(0)
except Exception:
  raise SystemExit(1)
''')
    try:
        proc = subprocess.run(
            ["python", "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        obj = _json.loads(proc.stdout.strip())
        free = int(obj.get("free", 0))
        total = int(obj.get("total", 0))
        if total > 0:
            return free, total
    except Exception:
        return None
    return None


def _estimate_upper_batch_from_tpu_memory(
    *,
    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]],
    validator: Callable[[int], bool],
    max_dynamic_batch: int,
    xla_spmd: bool,
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
    mem = _tpu_memory_free_and_total(xla_spmd)
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
    post_mem = _tpu_memory_free_and_total(xla_spmd)
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

    # Optionally override bucket boundaries from CLI.
    global BUCKET_BOUNDARIES
    if args.bucket_boundaries:
        try:
            BUCKET_BOUNDARIES = tuple(int(x.strip()) for x in args.bucket_boundaries.split(",") if x.strip())
            if not BUCKET_BOUNDARIES:
                raise ValueError
        except Exception:
            raise ValueError("--bucket-boundaries must be a comma-separated list of ints, e.g. '512' or '512,1024'")

    print("\n2. Preprocessing and bucketing dataset...")
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

    if args.force_batch_size is not None:
        train_batch_size = eval_batch_size = max(1, int(args.force_batch_size))
        print(f"[batch-tuning] Forcing batch size to {train_batch_size} via --force-batch-size.")
    else:
        # If allowed, estimate an upper bound from live TPU memory and cap search.
        if args.no_live_memory_query:
            print("[batch-tuning] Live memory queries disabled (--no-live-memory-query).")
            upper_hint = None
        else:
            upper_hint = _estimate_upper_batch_from_tpu_memory(
                bucket_samples=train_bucket_samples,
                validator=(train_validator if train_validator is not None else (lambda _: True)),
                max_dynamic_batch=args.max_dynamic_batch,
                xla_spmd=args.xla_spmd,
                verbose=(args.log_batch_tuning or args.log_tpu_memory),
            )
    # Use memory-derived upper bound whenever available (validator optional)
    if upper_hint is not None:
        print(f"Using TPU memory-derived upper batch bound: {upper_hint}")
        train_max = min(args.max_dynamic_batch, upper_hint)
    else:
        if args.skip_oom_validator:
            # Be conservative when validator is off.
            train_max = min(args.max_dynamic_batch, max(BASE_BATCH_SIZE, BASE_BATCH_SIZE * 2))
            print(f"[batch-tuning] Validator disabled; capping train max to {train_max}.")
        else:
            train_max = args.max_dynamic_batch

    train_batch_size = _compute_dynamic_batch_size(
        train_stats["bucket_counts"],
        max_dynamic_batch=train_max,
        validator=train_validator,
    )

    # For eval, reuse the same cap; eval usually uses identical shapes.
    if args.skip_oom_validator:
        eval_upper = min(args.max_dynamic_batch, train_max)
    else:
        eval_upper = upper_hint if upper_hint is not None else args.max_dynamic_batch
    eval_batch_size = _compute_dynamic_batch_size(
        eval_stats["bucket_counts"],
        max_dynamic_batch=eval_upper,
        validator=eval_validator,
    )

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
        xla_spmd=args.xla_spmd,
    )

    pre_state = None
    if not args.skip_pre_eval or sample_image is not None:
        pre_state = trainer.init_state()
        if not args.skip_pre_eval:
            _log_tpu_memory_event(
                enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
                label="pre_training_eval/pre",
                xla_spmd=args.xla_spmd,
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
                xla_spmd=args.xla_spmd,
            )
        if sample_image is not None:
            _log_tpu_memory_event(
                enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
                label="chat_sample/pre_training/pre",
                xla_spmd=args.xla_spmd,
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
                xla_spmd=args.xla_spmd,
            )

    print(f"\n6. Starting training for {len(train_ds)} steps...")
    state = None
    aux = None
    try:
        state, aux = trainer.train()
        print("\n✓ Training finished.")
    except StopIteration:
        print("\nTraining iterator exhausted (StopIteration); treating as finished.")

    # Post-training detailed memory log
    if state is not None:
        _log_tpu_memory_event(
            enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
            label="post_training",
            xla_spmd=args.xla_spmd,
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
            xla_spmd=args.xla_spmd,
        )
        _run_evaluator(
            trainer=trainer,
            state=state,
            label="post_training_eval",
            jsonl_path=metrics_jsonl_path,
            wandb_handle=wandb_handle,
        )
        _log_tpu_memory_event(
            enabled=(not args.no_live_memory_query and (args.log_tpu_memory or args.log_batch_tuning)),
            label="post_training_eval/post",
            xla_spmd=args.xla_spmd,
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
