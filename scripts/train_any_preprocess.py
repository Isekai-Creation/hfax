#!/usr/bin/env python3
"""Unified training entrypoint (HF Qwen2.5-VL and Gemma3).

This script can run in 2 modes:
  - HF/Qwen mode (default): trains/inits a Hugging Face model such as
    "Qwen/Qwen2.5-VL-3B-Instruct" on CPU or GPU (PyTorch).
  - Gemma3/Kauldron mode: retains the original Gemma3 Kauldron path with
    dynamic preprocessing and batching.

Select the mode with ``--model-id``. If it contains "Qwen", the HF path is
used; otherwise the Gemma3 path is used. The Qwen path is intentionally kept
minimal and dependency-light.
"""

from __future__ import annotations
import os

# Set HF_HOME to /dev/shm/
os.environ.setdefault("HF_HOME", "/dev/shm/")

# IMPORT LOGGER BEFORE ANYTHING ELSE
import time
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler()],
)

# Log the log_postfix value
logger = logging.getLogger("rich")

import hfax

import argparse
import concurrent.futures
import io
import json
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
from datasets import load_dataset, DownloadConfig
from etils import enp

try:
    from tqdm.contrib.concurrent import process_map  # type: ignore
except Exception:
    process_map = None  # type: ignore
from jax import errors as jax_errors
from PIL import Image
from grain import python as grain
from transformers import (
    AutoTokenizer,
    AutoProcessor,
)
from huggingface_hub import snapshot_download
import torch

from kauldron import kd

try:  # Optional progress bars
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None

# Structured logging via hfax.utils.logging

# -----------------------------------------------------------------------------
# Retry helpers for network-bound HF ops
# -----------------------------------------------------------------------------


def _retry_call(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    name: str = "op",
):
    """Run `fn()` with simple exponential backoff.

    Returns fn()'s result or raises the last exception on final failure.
    """
    delay = base_delay
    last_exc = None
    for i in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network variability
            last_exc = exc
            if i >= attempts:
                break
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                name,
                i,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(max_delay, delay * 2.0)
    assert last_exc is not None
    raise last_exc


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


HF_LOCAL_ROOT = Path(os.environ.get("HF_LOCAL_ROOT", "/dev/shm/hf_local"))
HF_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)


def _ensure_local_repo(
    repo_id: str,
    *,
    repo_type: str = "model",
    allow_patterns: Optional[List[str]] = None,
) -> Path:
    target = HF_LOCAL_ROOT / f"{repo_type}s" / _sanitize_repo_id(repo_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or not any(target.iterdir()):
        snap_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(target),
            local_dir_use_symlinks=True,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
        _ = snap_path
    return target


def _load_processor_with_retry(model_id: str):
    # Fetch only small processor/tokenizer files to avoid large weight downloads.
    local_dir = _ensure_local_repo(
        model_id,
        repo_type="model",
        allow_patterns=[
            "tokenizer*",
            "*processor_config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.*",
            "generation_config.json",
            "config.json",
            "*.model",  # sentencepiece
            "vocab.*",
            "merges.txt",
        ],
    )
    return AutoProcessor.from_pretrained(
        str(local_dir), trust_remote_code=True, local_files_only=True
    )


_PROC_CACHE: Dict[str, AutoProcessor] = {}


def _get_cached_processor(model_id: str) -> AutoProcessor:
    proc = _PROC_CACHE.get(model_id)
    if proc is None:
        proc = _load_processor_with_retry(model_id)
        _PROC_CACHE[model_id] = proc
    return proc


def _load_dataset_with_retry(dataset_path: str, *, split: str, cache_dir: str):
    def _fn():
        dc = DownloadConfig(max_retries=5)
        return load_dataset(
            dataset_path, split=split, cache_dir=cache_dir, download_config=dc
        )

    return _retry_call(
        _fn,
        attempts=3,
        base_delay=0.5,
        max_delay=4.0,
        name=f"load_dataset({dataset_path}:{split})",
    )


# -----------------------------------------------------------------------------
# Constants & defaults
# -----------------------------------------------------------------------------

# Defaults
MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
# Optional separate processor id; falls back to MODEL_ID if unset
PROCESSOR_ID_ENV = os.environ.get("HF_PROCESSOR_ID")
# Internal default for the Gemma/Kauldron path only.
GEMMA_MODEL_ID = "unsloth/gemma-3-4b-it"
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
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

MEM_PROVIDER = "tpu_info"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Mode selection / model
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=(
            "Hugging Face model id to load. If it contains 'Qwen', the script "
            "uses the HF/PyTorch path. Otherwise the Gemma3/Kauldron path is used."
        ),
    )
    parser.add_argument(
        "--processor-id",
        type=str,
        default=PROCESSOR_ID_ENV,
        help=("Hugging Face processor/tokenizer id to load (defaults to --model-id)."),
    )
    parser.add_argument(
        "--hf-max-length",
        type=int,
        default=4096,
        help="Max tokens for HF preprocessing (Qwen path).",
    )
    parser.add_argument(
        "--hf-batch-size",
        type=int,
        default=1,
        help="Global batch size for HF training (Qwen path).",
    )
    parser.add_argument(
        "--hf-epochs",
        type=int,
        default=1,
        help="Epochs for HF training (Qwen path).",
    )
    parser.add_argument(
        "--hf-lr",
        type=float,
        default=1e-5,
        help="Learning rate for HF training (Qwen path).",
    )
    parser.add_argument(
        "--log-tpu-memory",
        action="store_true",
        help="Print TPU memory usage before training (via tpu-info).",
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
    # No external python needed; TPU memory checks use tpu-info when available.
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
    args = parser.parse_args()
    if not getattr(args, "processor_id", None):
        args.processor_id = args.model_id
    return args


# -----------------------------------------------------------------------------
# HF/Qwen path (PyTorch) — minimal and robust
# -----------------------------------------------------------------------------


def _collect_image_token_ids_hf(processor: AutoProcessor) -> set[int]:
    pad_token_id = int(processor.tokenizer.pad_token_id)
    ids: set[int] = set()
    for key in ("boi_token", "eoi_token"):
        token_value = processor.tokenizer.special_tokens_map.get(key)
        if token_value is None:
            continue
        tokens = [token_value] if isinstance(token_value, str) else list(token_value)
        for tok in tokens:
            converted = processor.tokenizer.convert_tokens_to_ids(tok)
            if converted is not None:
                ids.add(int(converted))
    placeholder_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is not None:
        ids.add(int(placeholder_id))
    ids.discard(pad_token_id)
    # Historic placeholder sometimes used; harmless if unused in tokenizer.
    ids.add(262144)
    return ids


def _build_hf_collate_fn(processor: AutoProcessor):
    image_token_ids = _collect_image_token_ids_hf(processor)
    pad_id = int(processor.tokenizer.pad_token_id)

    # Optional Qwen vision helper
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception:  # pragma: no cover - optional dep
        process_vision_info = None  # type: ignore

    def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[object] = []
        vision_infos_list = []
        for ex in batch:
            messages = ex["messages"]  # type: ignore[assignment]
            # Render chat template to a plain string
            text = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            img = ex["image"]  # type: ignore[index]
            if hasattr(img, "convert"):
                img = img.convert("RGB")  # type: ignore[assignment]
            texts.append(text)
            images.append(img)
            if process_vision_info is not None:
                vision_infos_list.append(process_vision_info(messages))
        # Prepare tensors; pass vision_infos if available
        if vision_infos_list:
            inputs = processor(
                text=texts,
                images=images,
                vision_infos=vision_infos_list,  # type: ignore[arg-type]
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        input_ids: torch.Tensor = inputs["input_ids"]  # (B, L)
        labels = input_ids.clone()
        if image_token_ids:
            mask = torch.isin(
                labels, torch.tensor(sorted(image_token_ids), dtype=labels.dtype)
            )
            labels[mask] = -100
        labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate


def _make_hf_datasets(
    processor: AutoProcessor,
    *,
    split: str,
    instruction: str,
    max_examples: Optional[int] = None,
):
    """Return a PyTorch-friendly dataset with 'messages' and 'image' per row."""
    ds = load_dataset(DATASET_PATH, split=split, cache_dir=DATA_CACHE_DIR)
    if max_examples is not None:
        ds = ds.select(range(min(len(ds), max_examples)))

    def _map_fn(ex):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": ex["image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ex["text"]}]},
        ]
        return {"messages": messages, "image": ex["image"]}

    ds = ds.map(
        _map_fn,
        remove_columns=[c for c in ds.column_names if c not in ("image", "text")],
    )
    # Switch to python objects to keep PIL Images
    ds = ds.with_format("python")
    return ds


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


def _process_chunk_worker(
    args: Tuple[str, str, str, str, Tuple[int, ...], int, str, int, int],
) -> List[Tuple[int | None, Dict[str, np.ndarray] | None]]:
    """Picklable top-level worker for process_map.

    Args:
      args: (dataset_path, split, cache_dir, model_id, boundaries, max_length, instruction, start, end)
    Returns:
      List of (bucket_size, sample_dict) pairs (may include (None, None)).
    """
    (
        dataset_path,
        split,
        cache_dir,
        model_id,
        boundaries,
        max_length,
        instruction,
        start,
        end,
    ) = args
    # Load processor and dataset inside the worker. Any network failures are
    # converted into empty outputs so exceptions do not cross process
    # boundaries (some exceptions like httpx.RequestError are not picklable
    # across processes on certain versions and can crash the pool).
    try:
        proc = _get_cached_processor(model_id)
        ds_local = _load_dataset_with_retry(
            dataset_path, split=split, cache_dir=cache_dir
        ).with_format("python")
    except Exception as exc:
        logger.warning("Worker failed to init processor/dataset after retries: %s", exc)
        # Return an empty result for this chunk; the caller will just skip it.
        return []
    pad_token_id = proc.tokenizer.pad_token_id
    image_token_ids = _collect_image_token_ids(proc, pad_token_id)
    out: List[Tuple[int | None, Dict[str, np.ndarray] | None]] = []
    for idx in range(start, end):
        try:
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
            out.append(
                (
                    bucket_size,
                    {"input": padded_input, "target": targets, "loss_mask": loss_mask},
                )
            )
        except Exception:
            # Skip examples that trigger transient network/decoding issues
            # (e.g., httpx.RequestError when fetching remote assets).
            out.append((None, None))
    return out


def _prepare_bucketed_samples(
    *,
    split: str,
    processor: AutoProcessor,
    seed: int,
    max_length: int = max(BUCKET_BOUNDARIES),
    proc_id: str,
) -> Tuple[Dict[int, List[Dict[str, np.ndarray]]], Dict[str, Any]]:
    try:
        dataset = _load_dataset_with_retry(
            DATASET_PATH,
            split=split,
            cache_dir=DATA_CACHE_DIR,
        )
    except Exception as exc:
        logger.warning(
            "Top-level dataset load failed after retries for %s: %s", split, exc
        )
        # Empty dataset: downstream logic will handle and report usable=0
        from datasets import Dataset

        dataset = Dataset.from_dict({"image": [], "text": []})
    dataset = dataset.with_format("python")

    pad_token_id = processor.tokenizer.pad_token_id
    image_token_ids = _collect_image_token_ids(processor, pad_token_id)

    bucket_samples: Dict[int, List[Dict[str, np.ndarray]]] = {
        boundary: [] for boundary in BUCKET_BOUNDARIES
    }
    dropped_long = 0
    total_examples = len(dataset)
    max_workers = CPU_COUNT

    logger.info(
        f"  [{split}] preprocessing {total_examples} examples "
        f"with {max_workers} workers..."
    )

    def _process_single(
        example: Dict[str, Any],
    ) -> Tuple[int | None, Dict[str, np.ndarray] | None]:
        try:
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
            bucket_size = next(
                (b for b in BUCKET_BOUNDARIES if len(input_ids) <= b), None
            )
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
        except Exception:
            # Skip examples that trigger transient network/decoding issues.
            return None, None

    # Prefer process-based parallelism for heavier CPU work if available.
    if process_map is not None and max_workers > 1:
        workers = max(1, min(max_workers, 32))
        chunk_size = max(64, total_examples // workers)
        chunks: List[Tuple[int, int]] = []
        s = 0
        while s < total_examples:
            e = min(total_examples, s + chunk_size)
            chunks.append((s, e))
            s = e
        task_args = [
            (
                DATASET_PATH,
                split,
                DATA_CACHE_DIR,
                proc_id,
                BUCKET_BOUNDARIES,
                max_length,
                INSTRUCTION,
                st,
                ed,
            )
            for (st, ed) in chunks
        ]
        try:
            for res in process_map(
                _process_chunk_worker,
                task_args,
                max_workers=workers,
                desc=f"[{split}] chunks",
                unit="chunk",
            ):
                for bucket_size, sample in res:
                    if bucket_size is None or sample is None:
                        dropped_long += 1
                        continue
                    bucket_samples[bucket_size].append(sample)
        except Exception as exc:
            logger.warning(
                "[%s] process-based preprocessing failed (%s). Falling back to threads.",
                split,
                exc,
            )
            # Clear any partial results; we'll rebuild below using threads.
            bucket_samples = {boundary: [] for boundary in BUCKET_BOUNDARIES}
            dropped_long = 0
            # Fall through to threaded path
            pass
    if not any(bucket_samples.values()):
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

    candidates = list(
        range(BASE_BATCH_SIZE, max_dynamic_batch + BASE_BATCH_SIZE, BASE_BATCH_SIZE)
    )
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
        key: enp.ArraySpec(shape=value.shape, dtype=value.dtype)
        for key, value in batch.items()
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
        logger.warning("Sample image fetch failed (%s); continuing without image.", exc)
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
        logger.info("%s losses:", label)
        for key, value in sorted(loss_values.items()):
            logger.info("  %s: %.6f", key, value)
    if metric_values:
        logger.info("%s metrics:", label)
        for key, value in sorted(metric_values.items()):
            logger.info("  %s: %.6f", key, value)

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
        logger.info("%s: evaluator not configured, skipping.", label)
        return
    logger.info("Running %s evaluation...", label)
    aux_state = evaluator.evaluate(state=state, step=int(state.step))
    if aux_state is None:
        logger.info("%s: evaluator returned no metrics.", label)
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
    logger.info("Running %s chat sample...", label)
    response = sampler.chat(prompt, **kwargs)
    logger.info("Prompt: %s", prompt)
    if image is not None:
        logger.info("Image input provided.")
    logger.info("Response (%s): %s", label, response)

    object.__setattr__(sampler, "last_state", None)
    sampler.turns.clear()
    try:
        jax.clear_caches()
        logger.info("Cleared JAX compilation cache.")
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
        wandb_handle.run.log(
            {
                f"chat/{label}/prompt": prompt,
                f"chat/{label}/response": response,
            }
        )

    chat_path = run_dir / f"{label}_chat.txt"
    with chat_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Prompt:\n{prompt}\n\nResponse:\n{response}\n")


# -----------------------------------------------------------------------------
# TPU memory diagnostics
# -----------------------------------------------------------------------------


def _maybe_log_tpu_memory(*, log_memory: bool) -> None:
    if not log_memory:
        return

    mem_total = mem_used = mem_free = 0

    # Try external Python 3.10 helper first
    ext = _tpuinfo_cli_query(spmd=True)
    if ext is not None:
        mem_free, mem_total = ext
        mem_used = max(0, mem_total - mem_free)
    else:
        sub = _tpu_info_query_subprocess()
        if sub is not None:
            mem_free, mem_total = sub
            mem_used = max(0, mem_total - mem_free)
        else:
            logger.info("Failed to query TPU memory via tpu_info.")
            return

    if mem_total:
        logger.info(
            "TPU memory (per-replica): used=%.2f GB free=%.2f GB total=%.2f GB",
            mem_used / 1e9,
            mem_free / 1e9,
            mem_total / 1e9,
        )


def _log_tpu_memory_event(*, enabled: bool, label: str) -> None:
    if not enabled:
        return
    logger.info("[mem] %s:", label)
    _maybe_log_tpu_memory(log_memory=True)


# -----------------------------------------------------------------------------
# TPU memory–driven batch upper bound (via tpu-info CLI or fallbacks)
# -----------------------------------------------------------------------------


def _tpu_memory_free_and_total() -> Optional[Tuple[int, int]]:
    """Return (free_bytes, total_bytes) per replica if available.

    Uses tpu_info when available. Returns None if no
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
    res = _tpuinfo_cli_query(spmd=True)
    if res is not None:
        return res

    # Then try in-process tpu_info (subprocess Python 3.12 safe)
    res = _tpu_info_query_subprocess()
    if res is not None:
        return res

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
        gib_line = re.findall(
            r"([0-9]+\.[0-9]+)\s*GiB\s*/\s*([0-9]+\.[0-9]+)\s*GiB", txt
        )
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
        r"""
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
  logger.info(json.dumps({'free': free, 'total': total}))
  raise SystemExit(0)
except Exception:
  raise SystemExit(1)
"""
    )
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
    verbose: bool = False,
) -> Optional[int]:
    """Estimate a safe upper bound for batch size using live TPU memory.

    Strategy:
    - Validate once at BASE_BATCH_SIZE to ensure program compiles and runs.
    - Read free memory (per replica) via tpu_info.
    - Approximate per-sample memory as reserved_at_base / BASE_BATCH_SIZE.
      This is conservative (includes fixed overhead) but avoids OOM.
    - Predict upper bound and verify with validator; if it fails, binary search
      downward until success.
    Returns None if no memory provider is available.
    """
    mem = _tpu_memory_free_and_total()
    if mem is None:
        if verbose:
            logger.info(
                "[batch-tuning] No TPU memory provider available; using OOM-guided search only."
            )
        return None
    if verbose:
        logger.info(
            f"[batch-tuning] Pre-step memory: free={mem[0]} bytes, total={mem[1]} bytes"
        )

    # First, validate at BASE_BATCH_SIZE and measure reserved after the step.
    # We obtain reserved by re-querying total-free.
    if verbose:
        logger.info(f"[batch-tuning] Validating BASE_BATCH_SIZE={BASE_BATCH_SIZE}...")
    if not validator(BASE_BATCH_SIZE):  # pragma: no cover - guarded by validator
        if verbose:
            logger.info(
                "[batch-tuning] Failed at BASE_BATCH_SIZE; aborting memory-based cap."
            )
        return None
    post_mem = _tpu_memory_free_and_total()
    if post_mem is None:
        if verbose:
            logger.info(
                "[batch-tuning] Post-step memory read failed; aborting memory-based cap."
            )
        return None
    free_bytes, total_bytes = post_mem
    reserved_bytes = max(0, total_bytes - free_bytes)
    if verbose:
        logger.info(
            f"[batch-tuning] Post-step memory: free={free_bytes}, total={total_bytes}; "
            f"reserved≈{reserved_bytes}"
        )

    # Conservative per-sample estimate.
    per_sample = max(1, reserved_bytes // BASE_BATCH_SIZE)
    if verbose:
        logger.info(f"[batch-tuning] Estimated per-sample bytes: {per_sample}")

    # Leave a safety margin: 5% of total or 1 GiB, whichever is larger.
    safety = max(int(total_bytes * 0.05), 1 << 30)
    headroom = max(0, free_bytes - safety)
    if headroom <= 0:
        if verbose:
            logger.info(
                f"[batch-tuning] No headroom after safety margin (safety={safety}); "
                f"returning BASE_BATCH_SIZE={BASE_BATCH_SIZE}"
            )
        return BASE_BATCH_SIZE

    predicted = int(headroom // per_sample)
    # Round to multiple of BASE_BATCH_SIZE.
    if predicted <= 0:
        if verbose:
            logger.info("[batch-tuning] Predicted cap <= 0; returning BASE_BATCH_SIZE.")
        return BASE_BATCH_SIZE
    predicted = (predicted // BASE_BATCH_SIZE) * BASE_BATCH_SIZE
    predicted = max(BASE_BATCH_SIZE, min(predicted, max_dynamic_batch))
    if verbose:
        logger.info(
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
            logger.info(f"[batch-tuning] Try mid={mid} ...", end="")
        if validator(mid):
            best = mid
            lo = mid + BASE_BATCH_SIZE
            if verbose:
                logger.info("OK")
        else:
            hi = mid - BASE_BATCH_SIZE
            if verbose:
                logger.info("OOM/NO")
    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()
    args = parse_args()

    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    jax.config.update("jax_platform_name", args.jax_platform)

    print("=== Kauldron Fine-tuning Setup ===")

    # Using tpu-info CLI; no external python interpreter needed.

    logger.info("=== Kauldron Fine-tuning Setup ===")

    logger.info("1. Initializing tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    proc_id = GEMMA_MODEL_ID if args.processor_id is None else args.processor_id
    logger.info("[Gemma] Ensuring local processor snapshot id=%s", proc_id)
    proc_local = _ensure_local_repo(proc_id, repo_type="model")
    processor = AutoProcessor.from_pretrained(
        str(proc_local), trust_remote_code=True, local_files_only=True
    )
    logger.info("✓ Tokenizer and processor initialized.")

    # Normalize fixed-batch alias
    if (
        getattr(args, "fixed_batch_size", None) is not None
        and args.force_batch_size is None
    ):
        args.force_batch_size = int(args.fixed_batch_size)

    # Optionally override bucket boundaries or switch to fixed-max mode
    global BUCKET_BOUNDARIES
    if args.fixed_max_length is not None:
        BUCKET_BOUNDARIES = (int(args.fixed_max_length),)
        logger.info(
            "[config] Fixed max length enabled: %d tokens.", BUCKET_BOUNDARIES[0]
        )
    elif args.bucket_boundaries:
        try:
            BUCKET_BOUNDARIES = tuple(
                int(x.strip()) for x in args.bucket_boundaries.split(",") if x.strip()
            )
            if not BUCKET_BOUNDARIES:
                raise ValueError
        except Exception:
            raise ValueError(
                "--bucket-boundaries must be a comma-separated list of ints, e.g. '512' or '512,1024'"
            )

    logger.info("2. Preprocessing and bucketing dataset...")
    train_bucket_samples, train_stats = _prepare_bucketed_samples(
        split=args.train_split,
        processor=processor,
        seed=RNG_SEED,
        max_length=max(BUCKET_BOUNDARIES),
        proc_id=proc_id,
    )
    eval_bucket_samples, eval_stats = _prepare_bucketed_samples(
        split=args.eval_split,
        processor=processor,
        seed=RNG_SEED + 1,
        max_length=max(BUCKET_BOUNDARIES),
        proc_id=proc_id,
    )

    logger.info(
        "Train bucket counts: %s",
        {k: v for k, v in train_stats["bucket_counts"].items() if v},
    )
    logger.info(
        "Eval bucket counts: %s",
        {k: v for k, v in eval_stats["bucket_counts"].items() if v},
    )
    logger.info(
        "Dropped %d long examples (train) and %d (eval).",
        train_stats["dropped_long"],
        eval_stats["dropped_long"],
    )
    logger.info(
        "Training data workers: %d eval data workers: %d",
        TRAIN_NUM_WORKERS,
        EVAL_NUM_WORKERS,
    )

    logger.info(f"3. Defining {args.model_id} model (baseline HF placeholder)...")
    # default: Load the model on the available device(s)
    logger.info(
        "[Gemma] Loading placeholder HF model id=%s device_map=auto torch_dtype=auto",
        args.model_id,
    )
    logger.info("✓ Model defined.")

    logger.info("4. Defining loss function (SoftmaxCrossEntropy)...")
    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits="preds.logits",
        labels="batch.target",
        mask="batch.loss_mask",
    )
    logger.info("✓ Loss function defined.")

    validation_optimizer = optax.adafactor(learning_rate=1e-3)
    # Provide rng_streams to avoid unresolved ROOT_CFG_REF during init.
    rng_streams = kd.train.RngStreams(
        [
            kd.train.RngStream(
                name="dropout", init=True, train=True, eval=False, per_step=True
            )
        ],
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
        logger.info(
            "[batch-tuning] Skipping JAX-based OOM validator per --skip-oom-validator."
        )
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
        logger.info(
            "[batch-tuning] Forcing batch size to %d via --force-batch-size.",
            train_batch_size,
        )
    else:
        # If allowed, estimate an upper bound from live TPU memory and cap search.
        if args.no_live_memory_query:
            logger.info(
                "[batch-tuning] Live memory queries disabled (--no-live-memory-query)."
            )
            upper_hint = None
        else:
            upper_hint = _estimate_upper_batch_from_tpu_memory(
                bucket_samples=train_bucket_samples,
                validator=(
                    train_validator if train_validator is not None else (lambda _: True)
                ),
                max_dynamic_batch=args.max_dynamic_batch,
                verbose=(args.log_batch_tuning or args.log_tpu_memory),
            )
        # Use memory-derived upper bound whenever available (validator optional)
        if upper_hint is not None:
            logger.info("Using TPU memory-derived upper batch bound: %d", upper_hint)
            train_max = min(args.max_dynamic_batch, upper_hint)
        else:
            if args.skip_oom_validator:
                # Be conservative when validator is off.
                train_max = min(
                    args.max_dynamic_batch, max(BASE_BATCH_SIZE, BASE_BATCH_SIZE * 2)
                )
                logger.info(
                    "[batch-tuning] Validator disabled; capping train max to %d.",
                    train_max,
                )
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
            eval_upper = (
                upper_hint if upper_hint is not None else args.max_dynamic_batch
            )
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

    logger.info(
        "Train utilized per bucket:", {k: v for k, v in train_utilized.items() if v}
    )
    if any(train_dropped.values()):
        logger.info(
            "Train dropped due to batching:",
            {k: v for k, v in train_dropped.items() if v},
        )
    logger.info(f"Train usable examples (batched): {train_stats['usable_examples']}")

    logger.info(
        "Eval utilized per bucket:", {k: v for k, v in eval_utilized.items() if v}
    )
    if any(eval_dropped.values()):
        logger.info(
            "Eval dropped due to batching:",
            {k: v for k, v in eval_dropped.items() if v},
        )
    logger.info(f"Eval usable examples (batched): {eval_stats['usable_examples']}")

    target_tokens = train_stats["batch_size"] * max(BUCKET_BOUNDARIES)
    train_tokens_total = sum(
        boundary * count for boundary, count in train_utilized.items()
    )
    eval_tokens_total = sum(
        boundary * count for boundary, count in eval_utilized.items()
    )
    logger.info(
        f"Computed train batch size: {train_batch_size} (~{target_tokens} tokens target)"
    )
    logger.info(f"Computed eval batch size: {eval_batch_size}")
    logger.info(
        f"Approx train tokens/step: {train_tokens_total / max(1, len(train_batched_samples)):.0f}"
    )
    logger.info(
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
    logger.info(
        "len(ds):",
        available_batches,
        "batches across",
        args.train_epochs,
        "epochs",
    )
    logger.info(
        f"Planned training steps: {available_batches}"
        f" ({expected_per_epoch} batches per epoch)"
    )
    logger.info("✓ Data pipeline created.")

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
    logger.info(f"Checkpoint run directory: {run_dir}")

    metrics_jsonl_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else run_dir / "metrics.jsonl"
    )
    wandb_handle = _maybe_init_wandb(args, run_dir)
    sample_image = None
    if not args.skip_chat_samples:
        sample_image = _maybe_download_image(args.sample_image_url)

    logger.info("\n5. Configuring the Kauldron trainer...")
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
    logger.info("✓ Trainer configured.")

    # Pre-training detailed memory log
    _log_tpu_memory_event(
        enabled=(
            not args.no_live_memory_query
            and (args.log_tpu_memory or args.log_batch_tuning)
        ),
        label="pre_training",
    )

    pre_state = None
    if not args.skip_pre_eval or sample_image is not None:
        pre_state = trainer.init_state()
        if not args.skip_pre_eval:
            _log_tpu_memory_event(
                enabled=(
                    not args.no_live_memory_query
                    and (args.log_tpu_memory or args.log_batch_tuning)
                ),
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
                enabled=(
                    not args.no_live_memory_query
                    and (args.log_tpu_memory or args.log_batch_tuning)
                ),
                label="pre_training_eval/post",
            )
        if sample_image is not None:
            _log_tpu_memory_event(
                enabled=(
                    not args.no_live_memory_query
                    and (args.log_tpu_memory or args.log_batch_tuning)
                ),
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
                enabled=(
                    not args.no_live_memory_query
                    and (args.log_tpu_memory or args.log_batch_tuning)
                ),
                label="chat_sample/pre_training/post",
            )

    logger.info(f"\n6. Starting training for {len(train_ds)} steps...")
    state = None
    aux = None
    try:
        state, aux = trainer.train()
        logger.info("\n✓ Training finished.")
    except StopIteration:
        logger.info(
            "\nTraining iterator exhausted (StopIteration); treating as finished."
        )

    # Post-training detailed memory log
    if state is not None:
        _log_tpu_memory_event(
            enabled=(
                not args.no_live_memory_query
                and (args.log_tpu_memory or args.log_batch_tuning)
            ),
            label="post_training",
        )

    if aux is not None:
        try:
            aux_output = aux.compute(flatten=False)
        except AttributeError:
            logger.info(
                "Training auxiliaries have no compute method; skipping logging."
            )
        else:
            _log_metrics(
                label="training",
                aux_output=aux_output,
                jsonl_path=metrics_jsonl_path,
                wandb_handle=wandb_handle,
                step=int(getattr(state, "step", 0) or 0),
            )

    logger.info("\n7. Running evaluations and chat sampling...")
    if state is not None:
        _log_tpu_memory_event(
            enabled=(
                not args.no_live_memory_query
                and (args.log_tpu_memory or args.log_batch_tuning)
            ),
            label="post_training_eval/pre",
        )
        _run_evaluator(
            trainer=trainer,
            state=state,
            label="post_training_eval",
            jsonl_path=metrics_jsonl_path,
            wandb_handle=wandb_handle,
        )
        _log_tpu_memory_event(
            enabled=(
                not args.no_live_memory_query
                and (args.log_tpu_memory or args.log_batch_tuning)
            ),
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

    elapsed_seconds = time.time() - start_time
    logger.info(
        "Total run time: %02d:%02d:%02d",
        int(elapsed_seconds // 3600),
        int((elapsed_seconds % 3600) // 60),
        int(elapsed_seconds % 60),
    )
    runtime_record = {"label": "runtime", "elapsed_seconds": elapsed_seconds}
    if metrics_jsonl_path is not None:
        _append_metrics_record(metrics_jsonl_path, runtime_record)
    if wandb_handle is not None:
        wandb_handle.run.log({"runtime/elapsed_seconds": elapsed_seconds})

    _finish_wandb(wandb_handle)
    logger.info("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
