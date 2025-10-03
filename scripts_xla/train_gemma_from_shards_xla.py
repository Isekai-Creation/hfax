#!/usr/bin/env python3
from __future__ import annotations

"""
Train Gemma on TPU from pre-batched shards produced by preprocess_gemma_dynamic_xla.py.

No Hugging Face tokenizers/processor imports are used here to avoid the
fork-related warning under SPMD/FSDP. Memory metrics come from tpu_info only.

Example
  python scripts/train_gemma_from_shards_xla.py \
    --model-id unsloth/gemma-3-4b-it \
    --shards-dir /dev/shm/gemma_shards \
    --train-epochs 5 --lr 1e-4 --weight-decay 0.0 \
    --log-tpu-memory
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

# Enable SPMD and XLA persistent cache; set the global mesh for FSDP.
try:
    import numpy as np
    from torch_xla import runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        _prepare_spmd_partition_spec,
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    xr.initialize_cache("/dev/shm")
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices // 1, 1)
    device_ids = np.array(range(num_devices))
    # Axis names must include 'fsdp' for weight/activation sharding
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)

    print("_________________________XLA is Available!")
    XLA_AVAILABLE = True
except Exception:
    print("_________________________XLA is not installed.")
    XLA_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', default='unsloth/gemma-3-4b-it')
    p.add_argument('--shards-dir', required=True)
    p.add_argument('--train-epochs', type=int, default=5)
    p.add_argument('--eval-epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--log-tpu-memory', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def _tpuinfo_used_total() -> tuple[int, int]:
    try:
        from tpu_info import device as tpu_device  # type: ignore
        from tpu_info import metrics  # type: ignore
    except Exception as e:
        raise ImportError(
            'Please install tpu-info for TPU memory metrics, '
            'https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/tree/main/tpu_info'
        ) from e
    chip_type, count = tpu_device.get_local_chips()
    if not chip_type or not count:
        raise RuntimeError('No TPU devices found.')
    used = 0
    total = 0
    for chip in metrics.get_chip_usage(chip_type):
        used += int(chip.memory_usage)
        total += int(chip.total_memory)
    used //= 3
    total //= 3
    return used, total


class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, files: List[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return torch.load(self.files[i])


def _list_shards(d: Path, split: str) -> List[Path]:
    sd = d / split
    if not sd.exists():
        raise FileNotFoundError(f'Missing split directory: {sd}')
    files = sorted(p for p in sd.glob('*.pt'))
    if not files:
        raise FileNotFoundError(f'No .pt shards found in {sd}')
    return files


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = xla.device()

    shards_dir = Path(args.shards_dir)
    train_files = _list_shards(shards_dir, 'train')
    eval_files = _list_shards(shards_dir, 'eval')
    print(f'Found {len(train_files)} train shards, {len(eval_files)} eval shards in {shards_dir}')

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    # Wrap in SPMD FSDP v2 if available
    if 'FSDPv2' in globals():
        model = FSDPv2(model)
    print('✓ Model loaded.')

    ds_train = ShardDataset(train_files)
    ds_eval = ShardDataset(eval_files)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=None, shuffle=False, num_workers=min(32, os.cpu_count() or 1))
    eval_loader = torch.utils.data.DataLoader(ds_eval, batch_size=None, shuffle=False, num_workers=min(32, os.cpu_count() or 1))

    steps = len(train_loader) * args.train_epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(optim, max(1, steps // 20), steps)

    if args.log_tpu_memory:
        try:
            used, total = _tpuinfo_used_total()
            print(f'[mem] pre_training: used={used/1e9:.2f} / total={total/1e9:.2f} GB')
        except Exception as e:
            print(f'[mem] pre_training: failed ({e})')

    print(f'Training for {steps} steps...')
    model.train()
    for epoch in range(args.train_epochs):
        for step, batch in enumerate(pl.MpDeviceLoader(train_loader, device)):
            inp = batch['input'].to(device)
            lab = batch['target'].squeeze(-1).to(device)
            out = model(input_ids=inp, labels=lab)
            loss = out.loss
            loss.backward(); xm.optimizer_step(optim); optim.zero_grad(set_to_none=True); sched.step()
            if step % 10 == 0:
                xm.mark_step(); print(f'train epoch {epoch} step {step} loss {loss.item():.4f}')
    print('✓ Training finished.')

    if args.log_tpu_memory:
        try:
            used, total = _tpuinfo_used_total()
            print(f'[mem] post_training: used={used/1e9:.2f} / total={total/1e9:.2f} GB')
        except Exception as e:
            print(f'[mem] post_training: failed ({e})')

    print('Evaluating...')
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in pl.MpDeviceLoader(eval_loader, device):
            inp = batch['input'].to(device); lab = batch['target'].squeeze(-1).to(device)
            out = model(input_ids=inp, labels=lab)
            tot += float(out.loss.item()); n += 1
    print(f'post eval loss: {tot/max(1,n):.4f}')


if __name__ == '__main__':
    main()
