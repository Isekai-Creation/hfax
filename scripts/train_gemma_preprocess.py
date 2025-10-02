#!/usr/bin/env python3
"""Preprocess Gemma3 data with max-length padding and optional dynamic batching."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

MODEL_ID = "unsloth/gemma-3-4b-it"
RNG_SEED = 42
DATASET_PATH = "unsloth/LaTeX_OCR"
DATA_CACHE_DIR = "/dev/shm/dataset_cache"
INSTRUCTION = "Convert the equation images to LaTeX equations."
CPU_COUNT = os.cpu_count() or 1
TRAIN_NUM_WORKERS = max(1, min(32, CPU_COUNT))
EVAL_NUM_WORKERS = max(1, min(32, CPU_COUNT))
CHECKPOINT_ROOT = Path("/dev/shm/kauldron_runs")

# -----------------------------------------------------------------------------
# TPU memory helpers (subprocess tpu_info + torch_xla fallback)
# -----------------------------------------------------------------------------


def _tpu_info_query_subprocess() -> Optional[Tuple[int, int]]:
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


def _maybe_log_tpu_memory(*, log_memory: bool, xla_spmd: bool) -> None:
    if not log_memory:
        return
    try:
        if jax.default_backend() != "tpu":
            print("TPU memory logging requested but backend is not TPU.")
            return
    except Exception:
        return

    free = total = used = 0
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        device = xm.xla_device()
        mem_info = xm.get_memory_info(device)
        total = int(mem_info.get("bytes_limit", 0))
        used = int(mem_info.get("bytes_used", 0))
        free = max(0, total - used)
    except Exception:
        res = _tpu_info_query_subprocess()
        if res is not None:
            free, total = res
            used = max(0, total - free)
        else:
            print("Failed to query TPU memory via torch_xla or tpu-info.")
            return

    if total:
        if xla_spmd:
            print(
                f"TPU memory (per-replica): used={used/1e9:.2f} GB, free={free/1e9:.2f} GB, total={total/1e9:.2f} GB"
            )
        else:
            print(
                f"TPU memory usage: {used/1e9:.2f} / {total/1e9:.2f} GB (used/total)"
            )


def _log_tpu_memory_event(*, enabled: bool, label: str, xla_spmd: bool) -> None:
    if not enabled:
        return
    print(f"[mem] {label}:")
    _maybe_log_tpu_memory(log_memory=True, xla_spmd=xla_spmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-tpu-memory",
        action="store_true",
        help="Print TPU memory usage at key phases (uses tpu_info subprocess or torch_xla).",
    )
    parser.add_argument(
        "--xla-spmd",
        action="store_true",
        help="Adjust TPU memory reporting for SPMD/replicated sharding.",
    )
    parser.add_argument(
        "--global-batch",
        type=int,
        default=8,
        help="Global batch size per training step (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Pad/truncate sequences to this many tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=1,
        help="Number of eval epochs (default: %(default)s).",
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
        "--jax-platform",
        choices=["cpu", "tpu", "gpu"],
        default="cpu",
        help="Run JAX on this platform (default: %(default)s).",
    )
    parser.add_argument(
        "--drop-incomplete-batches",
        action="store_true",
        help="Drop final partial batches instead of padding them.",
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


class PrecomputedSampleDataSource(grain.RandomAccessDataSource):
    """Random access data source backed by pre-batched samples."""

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
        for token in tokens:
            converted = processor.tokenizer.convert_tokens_to_ids(token)
            if converted is not None:
                image_token_ids.add(int(converted))
    placeholder_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is not None:
        image_token_ids.add(int(placeholder_id))
    image_token_ids.discard(pad_token_id)
    image_token_ids.add(262144)
    return image_token_ids


def _preprocess_split(
    *,
    split: str,
    processor: AutoProcessor,
    seed: int,
    max_length: int,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    dataset = load_dataset(
        DATASET_PATH,
        split=split,
        cache_dir=DATA_CACHE_DIR,
    )
    dataset = dataset.with_format("python")

    pad_token_id = processor.tokenizer.pad_token_id
    image_token_ids = _collect_image_token_ids(processor, pad_token_id)

    total_examples = len(dataset)
    print(f"  [{split}] preprocessing {total_examples} examples...")

    def _process_single(example: Dict[str, Any]) -> Dict[str, np.ndarray]:
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
        batch = processor(
            text=[rendered_prompt],
            images=[example["image"].convert("RGB")],
            return_tensors="np",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        input_ids = batch["input_ids"][0].astype(np.int32)

        labels = input_ids.copy()
        if image_token_ids:
            labels[np.isin(labels, list(image_token_ids))] = -100
        labels[labels == pad_token_id] = -100

        targets = labels[:, None]
        loss_mask = (targets != -100).astype(np.int32)

        return {
            "input": input_ids,
            "target": targets,
            "loss_mask": loss_mask,
        }

    samples: List[Dict[str, np.ndarray]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count() or 1
    ) as executor:
        iterator: Iterable[Dict[str, np.ndarray]] = executor.map(
            _process_single, dataset
        )
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=total_examples,
                desc=f"[{split}]",
                unit="sample",
                leave=True,
            )
        for sample in iterator:
            samples.append(sample)

    stats = {
        "total_examples": total_examples,
        "usable_examples": len(samples),
    }
    return samples, stats


def _pad_batch(
    batch: Dict[str, np.ndarray], target_batch_size: int, pad_token_id: int
) -> Tuple[Dict[str, np.ndarray], int]:
    current = next(iter(batch.values()))
    current_size = current.shape[0]
    if current_size == target_batch_size:
        return batch, 0
    if current_size > target_batch_size:
        raise ValueError(
            f"Batch size {current_size} exceeds target {target_batch_size}."
        )
    pad_amount = target_batch_size - current_size

    def _pad_array(array: np.ndarray, value: int) -> np.ndarray:
        pad_width = [(0, pad_amount)] + [(0, 0)] * (array.ndim - 1)
        return np.pad(
            array, pad_width=pad_width, mode="constant", constant_values=value
        )

    padded_batch: Dict[str, np.ndarray] = {}
    for key, array in batch.items():
        if key == "input":
            pad_value = pad_token_id
        elif key == "target":
            pad_value = -100
        elif key == "loss_mask":
            pad_value = 0
        else:
            pad_value = 0
        padded_batch[key] = _pad_array(array, pad_value)
    return padded_batch, pad_amount


def _make_batches(
    samples: List[Dict[str, np.ndarray]],
    batch_size: int,
    *,
    pad_token_id: int,
    drop_incomplete: bool,
) -> Tuple[List[Dict[str, np.ndarray]], int, int, int]:
    total_examples = len(samples)
    full_batches, remainder = divmod(total_examples, batch_size)

    limit = full_batches * batch_size if drop_incomplete else total_examples
    trimmed_samples = samples[:limit]

    batched: List[Dict[str, np.ndarray]] = []
    for start in range(0, full_batches * batch_size, batch_size):
        chunk = trimmed_samples[start : start + batch_size]
        batched.append(
            {key: np.stack([s[key] for s in chunk], axis=0) for key in chunk[0]}
        )

    dropped = remainder if drop_incomplete else 0
    padded_examples = 0
    steps_per_epoch = full_batches

    if not drop_incomplete and remainder:
        chunk = samples[-remainder:]
        batch = {key: np.stack([s[key] for s in chunk], axis=0) for key in chunk[0]}
        batch, pad_amount = _pad_batch(batch, batch_size, pad_token_id)
        padded_examples += pad_amount
        batched.append(batch)
        steps_per_epoch += 1

    if drop_incomplete:
        steps_per_epoch = full_batches

    return batched, steps_per_epoch, dropped, padded_examples


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
        "global_batch": args.global_batch,
        "max_length": args.max_length,
        "train_epochs": args.train_epochs,
        "eval_epochs": args.eval_epochs,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "drop_incomplete_batches": args.drop_incomplete_batches,
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

    jax.clear_caches()
    print("Cleared JAX compilation cache.")

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


def main() -> None:
    start_time = time.time()
    args = parse_args()

    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    os.environ["JAX_PLATFORMS"] = args.jax_platform
    jax.config.update("jax_platform_name", args.jax_platform)

    print("=== Gemma Kauldron Fine-tuning Setup ===")

    print("\n1. Initializing tokenizer...")
    tokenizer = hfax.text.Gemma3Tokenizer()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✓ Tokenizer initialized.")

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad_token_id for padding.")
    pad_token_id = int(pad_token_id)

    print("\n2. Preprocessing dataset with max-length padding...")
    train_samples, train_stats = _preprocess_split(
        split=args.train_split,
        processor=processor,
        seed=RNG_SEED,
        max_length=args.max_length,
    )
    eval_samples, eval_stats = _preprocess_split(
        split=args.eval_split,
        processor=processor,
        seed=RNG_SEED + 1,
        max_length=args.max_length,
    )

    print(
        f"Train usable examples: {train_stats['usable_examples']} (total {train_stats['total_examples']})"
    )
    print(
        f"Eval usable examples: {eval_stats['usable_examples']} (total {eval_stats['total_examples']})"
    )
    print("Global batch size:", args.global_batch)

    train_batches, train_steps_per_epoch, train_dropped, train_padded = _make_batches(
        train_samples,
        args.global_batch,
        pad_token_id=pad_token_id,
        drop_incomplete=args.drop_incomplete_batches,
    )
    if train_steps_per_epoch == 0:
        raise ValueError("Not enough training samples to form a single batch.")
    if train_dropped:
        print(f"Dropping {train_dropped} training examples to keep full batches.")
    if train_padded:
        print(f"Padded {train_padded} training examples to complete the final batch.")

    eval_batches, eval_steps_per_epoch, eval_dropped, eval_padded = _make_batches(
        eval_samples,
        args.global_batch,
        pad_token_id=pad_token_id,
        drop_incomplete=args.drop_incomplete_batches,
    )
    if eval_steps_per_epoch == 0:
        raise ValueError("Not enough eval samples to form a single batch.")
    if eval_dropped:
        print(f"Dropping {eval_dropped} eval examples to keep full batches.")
    if eval_padded:
        print(f"Padded {eval_padded} eval examples to complete the final batch.")

    train_ds = kd.data.py.DataSource(
        data_source=PrecomputedSampleDataSource(train_batches),
        shuffle=False,
        batch_size=None,
        num_epochs=args.train_epochs,
        num_workers=TRAIN_NUM_WORKERS,
        batch_drop_remainder=False,
    )
    expected_train_steps = train_steps_per_epoch * args.train_epochs
    available_train_steps = len(train_ds)
    if available_train_steps != expected_train_steps:
        raise ValueError(
            f"Expected {expected_train_steps} training steps, got {available_train_steps}."
        )
    if available_train_steps <= 0:
        raise ValueError("Training dataset produced zero batches.")
    # Kauldron's Trainer interprets num_train_steps as the final step index,
    # so it will execute `num_train_steps + 1` batches. Subtract one to avoid
    # requesting more batches than we precomputed.
    planned_train_steps = available_train_steps
    final_train_step_index = planned_train_steps - 1
    print(
        "len(train_ds):",
        available_train_steps,
        "steps across",
        args.train_epochs,
        "epochs",
    )

    eval_ds = kd.data.py.DataSource(
        data_source=PrecomputedSampleDataSource(eval_batches),
        shuffle=False,
        batch_size=None,
        num_epochs=args.eval_epochs,
        num_workers=EVAL_NUM_WORKERS,
        batch_drop_remainder=False,
    )
    available_eval_steps = len(eval_ds)
    print(
        "len(eval_ds):",
        available_eval_steps,
        "steps across",
        args.eval_epochs,
        "epochs",
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
        num_train_steps=final_train_step_index,
        train_losses={"loss": loss},
        optimizer=optax.adafactor(learning_rate=1e-3),
    )
    print("✓ Trainer configured.")
    _log_tpu_memory_event(
        enabled=args.log_tpu_memory,
        label="pre_training",
        xla_spmd=args.xla_spmd,
    )
    pre_state = None
    if not args.skip_pre_eval or sample_image is not None:
        pre_state = trainer.init_state()
        if not args.skip_pre_eval:
            _log_tpu_memory_event(
                enabled=args.log_tpu_memory,
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
                enabled=args.log_tpu_memory,
                label="pre_training_eval/post",
                xla_spmd=args.xla_spmd,
            )
        if sample_image is not None:
            _log_tpu_memory_event(
                enabled=args.log_tpu_memory,
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
                enabled=args.log_tpu_memory,
                label="chat_sample/pre_training/post",
                xla_spmd=args.xla_spmd,
            )

    print(f"\n6. Starting training for {planned_train_steps} steps...")
    state, aux = trainer.train()
    print("\n✓ Training finished.")
    _log_tpu_memory_event(
        enabled=args.log_tpu_memory,
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
                step=int(state.step),
            )

    print("\n7. Running evaluations and chat sampling...")
    _log_tpu_memory_event(
        enabled=args.log_tpu_memory,
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
        enabled=args.log_tpu_memory,
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
