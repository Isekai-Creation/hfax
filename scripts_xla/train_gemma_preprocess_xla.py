#!/usr/bin/env python3
"""Fine-tune Gemma3 4B on TPU via torch_xla with fixed max-length padding.

The script mirrors scripts/train_gemma_preprocess.py but uses PyTorch + XLA
instead of Kauldron/JAX. It keeps the same dataset, preprocessing, and basic
logging, and uses tpu_info for TPU memory visibility.
"""

from __future__ import annotations

# Silence Hugging Face tokenizers fork warning (must be set before import)
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


try:
    import numpy as np
    import torch_xla.core.xla_model as xm
    import torch_xla as xla
    from torch_xla import runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        _prepare_spmd_partition_spec,
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    xr.initialize_cache("/dev/shm")

    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices // 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)

    print("_________________________XLA is Available!")
    XLA_AVAILABLE = True
except:
    print("_________________________XLA is not installed.")
    XLA_AVAILABLE = False

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

import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--train-epochs", type=int, default=5)
    p.add_argument("--eval-epochs", type=int, default=1)
    p.add_argument("--train-split", default="train[:3000]")
    p.add_argument("--eval-split", default="test[:3000]")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--log-tpu-memory", action="store_true")
    p.add_argument("--skip-pre-eval", action="store_true")
    p.add_argument("--metrics-jsonl", type=str, default=None)
    # Memory metrics use tpu_info exclusively; no XLA fallback.
    return p.parse_args()


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {k: torch.from_numpy(v) for k, v in s.items()}


def _collect_image_token_ids(processor: AutoProcessor, pad_token_id: int) -> set[int]:
    image_token_ids: set[int] = set()
    for key in ("boi_token", "eoi_token"):
        token_value = processor.tokenizer.special_tokens_map.get(key)
        if token_value is None:
            continue
        tokens = [token_value] if isinstance(token_value, str) else list(token_value)
        for tok in tokens:
            conv = processor.tokenizer.convert_tokens_to_ids(tok)
            if conv is not None:
                image_token_ids.add(int(conv))
    placeholder_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if placeholder_id is not None:
        image_token_ids.add(int(placeholder_id))
    image_token_ids.discard(pad_token_id)
    image_token_ids.add(262144)
    return image_token_ids


def preprocess_split(
    *, split: str, processor: AutoProcessor, max_length: int
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    ds = load_dataset(DATASET_PATH, split=split, cache_dir=DATA_CACHE_DIR)
    ds = ds.with_format("python")

    pad_token_id = processor.tokenizer.pad_token_id
    image_token_ids = _collect_image_token_ids(processor, pad_token_id)
    total = len(ds)

    def _proc(ex: Dict[str, Any]) -> Dict[str, np.ndarray]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": ex["image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ex["text"]}]},
        ]
        rendered = processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        batch = processor(
            text=[rendered],
            images=[ex["image"].convert("RGB")],
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
        return {"input": input_ids, "target": targets, "loss_mask": loss_mask}

    samples: List[Dict[str, np.ndarray]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT) as ex:
        for s in ex.map(_proc, ds):
            samples.append(s)

    stats = {"total_examples": total, "usable_examples": len(samples)}
    return samples, stats


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def _tpuinfo_used_total() -> tuple[int, int]:
    """Return (used_bytes, total_bytes) from tpu_info only."""
    try:
        from tpu_info import device as tpu_device  # type: ignore
        from tpu_info import metrics  # type: ignore
    except Exception as e:
        raise ImportError(
            "Please install tpu-info for TPU memory metrics, https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/tree/main/tpu_info"
        ) from e
    chip_type, count = tpu_device.get_local_chips()
    if not chip_type or not count:
        raise RuntimeError("No TPU devices found.")
    used = 0
    total = 0
    for chip in metrics.get_chip_usage(chip_type):
        used += int(chip.memory_usage)
        total += int(chip.total_memory)
    used //= 3
    total //= 3
    return used, total


def _maybe_log_tpu_memory(label: str) -> None:
    try:
        used_b, total_b = _tpuinfo_used_total()
        used = used_b / 1e9
        total = total_b / 1e9
        print(f"[mem] {label}: used={used:.2f} / total={total:.2f} GB")
    except Exception as e:
        print(f"[mem] {label}: failed to query ({e})")


def train_loop(
    *, model, optimizer, scheduler, train_loader, device, epochs: int
) -> None:
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(pl.MpDeviceLoader(train_loader, device)):
            input_ids = batch["input"].to(device)
            labels = batch["target"].squeeze(-1).to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if step % 10 == 0:
                xm.mark_step()
                print(f"train epoch {epoch} step {step} loss {loss.item():.4f}")


def eval_loop(*, model, eval_loader, device) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in pl.MpDeviceLoader(eval_loader, device):
            input_ids = batch["input"].to(device)
            labels = batch["target"].squeeze(-1).to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += float(outputs.loss.item())
            steps += 1
    return total_loss / max(1, steps)


def main() -> None:
    start = time.time()
    args = parse_args()
    torch.manual_seed(RNG_SEED)
    device = xla.device()

    print("=== Gemma torch_xla Fine-tuning (fixed length) ===")
    print("1. Initializing processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.to(device)
    model = FSDPv2(model)
    print("✓ Model loaded.")

    print("\n2. Preprocessing...")
    train_samples, train_stats = preprocess_split(
        split=args.train_split, processor=processor, max_length=args.max_length
    )
    eval_samples, eval_stats = preprocess_split(
        split=args.eval_split, processor=processor, max_length=args.max_length
    )
    print(
        "Train usable:",
        train_stats["usable_examples"],
        "Eval usable:",
        eval_stats["usable_examples"],
    )

    train_ds = PrecomputedDataset(train_samples)
    eval_ds = PrecomputedDataset(eval_samples)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.global_batch,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
        collate_fn=collate_fn,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.global_batch,
        shuffle=False,
        num_workers=EVAL_NUM_WORKERS,
        collate_fn=collate_fn,
    )

    total_steps = len(train_loader) * args.train_epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps,
    )

    if args.log_tpu_memory:
        _maybe_log_tpu_memory("pre_training")

    if not args.skip_pre_eval:
        pre_loss = eval_loop(model=model, eval_loader=eval_loader, device=device)
        print(f"pre eval loss: {pre_loss:.4f}")
        if args.log_tpu_memory:
            _maybe_log_tpu_memory("pre_eval/post")

    print(f"\n3. Training for {total_steps} steps...")
    train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        epochs=args.train_epochs,
    )
    print("✓ Training finished.")

    if args.log_tpu_memory:
        _maybe_log_tpu_memory("post_training")

    post_loss = eval_loop(model=model, eval_loader=eval_loader, device=device)
    print(f"post eval loss: {post_loss:.4f}")

    elapsed = time.time() - start
    print(
        f"Total run time: {int(elapsed//3600):02d}:{int(elapsed%3600//60):02d}:{int(elapsed%60):02d}"
    )


if __name__ == "__main__":
    main()
