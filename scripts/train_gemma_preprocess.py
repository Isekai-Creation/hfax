#!/usr/bin/env python3
"""Preprocess Gemma3 data with max-length padding and optional dynamic batching."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import jax
import numpy as np
import optax
from datasets import load_dataset
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        return np.pad(array, pad_width=pad_width, mode="constant", constant_values=value)

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


def main() -> None:
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
        print(
            f"Padded {train_padded} training examples to complete the final batch."
        )

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
        print(
            f"Padded {eval_padded} eval examples to complete the final batch."
        )

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

    print(f"\n6. Starting training for {planned_train_steps} steps...")
    state, aux = trainer.train()
    print("\n✓ Training finished.")

    print("\n7. Running a sample evaluation...")
    sampler = hfax.text.ChatSampler(
        model=model,
        params=state.params,
        tokenizer=tokenizer,
    )
    prompt = "Hello! My next holidays are in Paris."
    response = sampler.chat(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
