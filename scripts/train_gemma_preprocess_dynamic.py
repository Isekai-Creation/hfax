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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import jax
import numpy as np
import optax
import requests
from datasets import load_dataset
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

TRAIN_NUM_EPOCHS = 5
EVAL_NUM_EPOCHS = 1
RNG_SEED = 42
DATASET_PATH = "unsloth/LaTeX_OCR"
DATA_CACHE_DIR = "/dev/shm/dataset_cache"
INSTRUCTION = "Convert the equation images to LaTeX equations."
CPU_COUNT = os.cpu_count() or 1
TRAIN_NUM_WORKERS = max(1, min(32, CPU_COUNT))
EVAL_NUM_WORKERS = max(1, min(32, CPU_COUNT))
CHECKPOINT_ROOT = Path("/dev/shm/kauldron_runs")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-dynamic-batch",
        type=int,
        default=MAX_DYNAMIC_BATCH_DEFAULT,
        help="Upper bound for dynamically scaled batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--log-tpu-memory",
        action="store_true",
        help="Print TPU memory usage before training (requires torch_xla or tpu-info).",
    )
    parser.add_argument(
        "--xla-spmd",
        action="store_true",
        help="When logging TPU memory, adjust readings for XLA SPMD/FSDP mode.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=["cpu", "tpu", "gpu"],
        default=os.environ.get("JAX_PLATFORM_NAME", "cpu"),
        help="Backend to target with JAX (default: %(default)s).",
    )
    parser.add_argument(
        "--train-split",
        default="train[:3000]",
        help="Dataset split for training (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-split",
        default="test[:300]",
        help="Dataset split for evaluation (default: %(default)s).",
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
) -> int:
    total_examples = sum(bucket_counts.values())
    if total_examples == 0:
        return BASE_BATCH_SIZE

    weighted_sum = sum(bucket * count for bucket, count in bucket_counts.items())
    average_bucket = max(1, weighted_sum // total_examples)

    target_tokens = BASE_BATCH_SIZE * max(BUCKET_BOUNDARIES)
    suggested = max(BASE_BATCH_SIZE, target_tokens // average_bucket)
    suggested = min(max_dynamic_batch, suggested)

    max_available = max(bucket_counts.values()) if bucket_counts else suggested
    if max_available:
        suggested = min(suggested, max_available)

    return max(1, suggested)


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


def _maybe_download_image(url: Optional[str]) -> Optional[np.ndarray]:
    if not url:
        return None
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(image, dtype=np.uint8)


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

    if xla_spmd:
        try:
            from tpu_info import device as tpu_device  # type: ignore
            from tpu_info import metrics as tpu_metrics  # type: ignore
        except ImportError:
            print("tpu-info not available; skipping TPU SPMD memory logging.")
            return
        try:
            chip_type, count = tpu_device.get_local_chips()
            if not chip_type or not count:
                print("No TPU devices detected for SPMD memory logging.")
                return
            device_usage = tpu_metrics.get_chip_usage(chip_type)
            mem_reserved = sum(chip.memory_usage for chip in device_usage)
            mem_total = sum(chip.total_memory for chip in device_usage)
            mem_reserved /= 3
            mem_total /= 3
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"Failed to query tpu-info metrics: {exc}")
            return
    else:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except ImportError:  # pragma: no cover
            print("torch_xla not installed; skipping TPU memory logging.")
            return
        try:
            device = xm.xla_device()
            mem_info = xm.get_memory_info(device)
            mem_reserved = mem_info.get("bytes_used", 0)
            mem_total = mem_info.get("bytes_limit", 0)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"Failed to query torch_xla memory info: {exc}")
            return

    if mem_total:
        print(
            f"TPU memory usage: {mem_reserved / 1e9:.2f} / {mem_total / 1e9:.2f} GB"
            " (reserved / total)"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()
    args = parse_args()

    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    jax.config.update("jax_platform_name", args.jax_platform)

    print("=== Gemma Kauldron Fine-tuning Setup ===")

    print("\n1. Initializing tokenizer...")
    tokenizer = hfax.text.Gemma3Tokenizer()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✓ Tokenizer initialized.")

    print("\n2. Preprocessing and bucketing dataset...")
    train_bucket_samples, train_stats = _prepare_bucketed_samples(
        split=args.train_split,
        processor=processor,
        seed=RNG_SEED,
    )
    eval_bucket_samples, eval_stats = _prepare_bucketed_samples(
        split=args.eval_split,
        processor=processor,
        seed=RNG_SEED + 1,
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

    train_batch_size = _compute_dynamic_batch_size(
        train_stats["bucket_counts"],
        max_dynamic_batch=args.max_dynamic_batch,
    )
    eval_batch_size = _compute_dynamic_batch_size(
        eval_stats["bucket_counts"],
        max_dynamic_batch=args.max_dynamic_batch,
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
        num_epochs=TRAIN_NUM_EPOCHS,
        num_workers=TRAIN_NUM_WORKERS,
    )
    available_batches = len(train_ds)
    expected_per_epoch = len(train_batched_samples)
    if available_batches != expected_per_epoch * TRAIN_NUM_EPOCHS:
        raise ValueError(
            "Mismatch between dataset batches and expected batches: "
            f"len(ds)={available_batches} vs expected={expected_per_epoch * TRAIN_NUM_EPOCHS}"
        )
    print(
        "len(ds):",
        available_batches,
        "batches across",
        TRAIN_NUM_EPOCHS,
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
        num_epochs=EVAL_NUM_EPOCHS,
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

    print("\n3. Defining Gemma 3 4B model...")
    model = hfax.nn.Gemma3_4B(
        tokens="batch.input",
    )
    print("✓ Model defined.")

    print("\n4. Defining loss function (SoftmaxCrossEntropy)...")
    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits="preds.logits",
        labels="batch.target",
        mask="batch.loss_mask",
    )
    print("✓ Loss function defined.")

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

    _maybe_log_tpu_memory(log_memory=args.log_tpu_memory, xla_spmd=args.xla_spmd)

    pre_state = None
    if not args.skip_pre_eval or sample_image is not None:
        pre_state = trainer.init_state()
        if not args.skip_pre_eval:
            _run_evaluator(
                trainer=trainer,
                state=pre_state,
                label="pre_training_eval",
                jsonl_path=metrics_jsonl_path,
                wandb_handle=wandb_handle,
            )
        if sample_image is not None:
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

    print(f"\n6. Starting training for {len(train_ds)} steps...")
    state, aux = trainer.train()
    print("\n✓ Training finished.")

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
                step=int(state.step),
            )

    print("\n7. Running evaluations and chat sampling...")
    _run_evaluator(
        trainer=trainer,
        state=state,
        label="post_training_eval",
        jsonl_path=metrics_jsonl_path,
        wandb_handle=wandb_handle,
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
