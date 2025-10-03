#!/usr/bin/env python3
from __future__ import annotations

# Silence Hugging Face tokenizers fork warning (must be set before import)
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""
Dynamic-length preprocessing + TPU training for Gemma (torch_xla).

Phases
- Preprocess: bucket by sequence length, pad to bucket, build targets/masks.
- Persist (optional): save batches to disk as .pt for reproducible runs.
- Train: stream pre-batched tensors through torch_xla MpDeviceLoader.

Memory tracking
- Non-SPMD: uses torch_xla.get_memory_info.
- SPMD/FSDP: uses tpu_info (install from the TPU diagnostics repo) since
  MemoryInfo is not available for SPMD virtual devices.

Example
  python scripts/train_gemma_preprocess_dynamic_xla.py \
    --model-id unsloth/gemma-3-4b-it \
    --train-split 'train[:3000]' --eval-split 'test[:3000]' \
    --bucket-boundaries 512,1024,2048,4096 \
    --max-dynamic-batch 128 \
    --persist-dir /dev/shm/gemma_preproc --persist-format pt \
    --log-tpu-memory --xla-spmd
"""

import argparse
import concurrent.futures
import dataclasses
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from datasets import load_dataset
try:
    from tqdm.auto import tqdm
    from tqdm.contrib.concurrent import thread_map
except Exception:
    tqdm = None
    thread_map = None  # type: ignore
from transformers import AutoModelForCausalLM, AutoProcessor, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic preprocessing + TPU training (torch_xla)")
    p.add_argument('--model-id', default='unsloth/gemma-3-4b-it')
    p.add_argument('--train-split', default='train[:3000]')
    p.add_argument('--eval-split', default='test[:3000]')
    p.add_argument('--bucket-boundaries', default='512,1024,2048,4096')
    p.add_argument('--max-dynamic-batch', type=int, default=128)
    p.add_argument('--train-epochs', type=int, default=5)
    p.add_argument('--eval-epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--instruction', default='Convert the equation images to LaTeX equations.')
    p.add_argument('--dataset', default='unsloth/LaTeX_OCR')
    p.add_argument('--data-cache', default='/dev/shm/dataset_cache')

    # Preprocess/persist options
    p.add_argument('--persist-dir', default='', help='If set, saves pre-batched tensors to this directory')
    p.add_argument('--persist-format', default='pt', choices=['pt', 'npz'])
    p.add_argument('--preprocess-only', action='store_true')

    # TPU/XLA options
    p.add_argument('--log-tpu-memory', action='store_true')
    p.add_argument('--cpu', action='store_true', help='Debug on CPU')
    return p.parse_args()


def _collect_image_token_ids(processor: AutoProcessor, pad_token_id: int) -> set[int]:
    s: set[int] = set()
    for key in ("boi_token", "eoi_token"):
        v = processor.tokenizer.special_tokens_map.get(key)
        if v is None:
            continue
        vv = [v] if isinstance(v, str) else list(v)
        for t in vv:
            cid = processor.tokenizer.convert_tokens_to_ids(t)
            if cid is not None:
                s.add(int(cid))
    pid = processor.tokenizer.convert_tokens_to_ids("<image>")
    if pid is not None:
        s.add(int(pid))
    s.discard(pad_token_id)
    s.add(262144)
    return s


def _prepare_bucketed_samples(
    *, split: str, processor: AutoProcessor, dataset_path: str, cache_dir: str, max_length: int, boundaries: Tuple[int, ...], instruction: str
) -> Tuple[Dict[int, List[Dict[str, np.ndarray]]], Dict[str, Any]]:
    ds = load_dataset(dataset_path, split=split, cache_dir=cache_dir).with_format('python')
    total = len(ds)
    workers = max(1, min(os.cpu_count() or 1, 32))
    print(f"  [{split}] preprocessing {total} examples with {workers} threads...")

    pad = processor.tokenizer.pad_token_id
    image_ids = _collect_image_token_ids(processor, pad)
    buckets: Dict[int, List[Dict[str, np.ndarray]]] = {b: [] for b in boundaries}
    dropped = 0

    def _proc_one(ex: Dict[str, Any]) -> Tuple[int | None, Dict[str, np.ndarray] | None]:
        conv = [
            {'role': 'user', 'content': [{'type': 'text', 'text': instruction}, {'type': 'image', 'image': ex['image']}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': ex['text']}]},
        ]
        rendered = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        bat = processor(text=[rendered], images=[ex['image'].convert('RGB')], return_tensors='np', padding=False, max_length=max_length, truncation=True)
        ids = bat['input_ids'][0].astype(np.int32)
        bnd = next((bb for bb in boundaries if len(ids) <= bb), None)
        if bnd is None:
            return None, None
        padded = np.full((bnd,), pad, dtype=np.int32)
        padded[: len(ids)] = ids
        labels = padded.copy()
        if image_ids:
            labels[np.isin(labels, list(image_ids))] = -100
        labels[labels == pad] = -100
        targets = labels[:, None]
        loss_mask = (targets != -100).astype(np.int32)
        return bnd, {'input': padded, 'target': targets, 'loss_mask': loss_mask}

    if thread_map is not None and tqdm is not None:
        results = thread_map(_proc_one, ds, max_workers=workers, desc=f"[{split}]", unit="sample", dynamic_ncols=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_proc_one, ds))
    for b, sample in results:
        if b is None or sample is None:
            dropped += 1
            continue
        buckets[b].append(sample)
    return buckets, {'bucket_counts': {k: len(v) for k, v in buckets.items()}, 'dropped_long': dropped}


def _tpuinfo_used_total() -> Tuple[int, int]:
    """Always return (used_bytes, total_bytes) from tpu_info for TPU."""
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
    # Empirical SPMD adjustment on v3-8
    used //= 3
    total //= 3
    return used, total


def _make_worst_case_batch(buckets: Dict[int, List[Dict[str, np.ndarray]]], bs: int) -> Optional[Dict[str, torch.Tensor]]:
    tmpl = None
    for b in sorted(buckets.keys(), reverse=True):
        if buckets[b]:
            tmpl = buckets[b][0]
            break
    if tmpl is None:
        return None

    def tile(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.stack([a] * bs, 0))

    return {k: tile(v) for k, v in tmpl.items()}


def _validator_factory(*, model, device, buckets) -> Callable[[int], bool]:
    cache: Dict[int, bool] = {}

    def valid(bs: int) -> bool:
        if bs in cache:
            return cache[bs]
        batch = _make_worst_case_batch(buckets, bs)
        if batch is None:
            cache[bs] = False
            return False
        try:
            model.eval()
            inp = batch['input'].to(device)
            lab = batch['target'].squeeze(-1).to(device)
            with torch.no_grad():
                out = model(input_ids=inp, labels=lab)
                _ = float(out.loss.item())
                xm.mark_step()
            cache[bs] = True
            return True
        except RuntimeError as e:
            if any(m in str(e) for m in ('Out of memory', 'RESOURCE_EXHAUSTED', 'insufficient memory')):
                cache[bs] = False
                return False
            raise

    return valid


def _compute_bs(counts: Dict[int, int], *, base: int, max_dynamic_batch: int, validator: Optional[Callable[[int], bool]]) -> int:
    if sum(counts.values()) == 0:
        return base

    def can(bs: int) -> bool:
        if bs <= 0:
            return False
        if sum(c // bs for c in counts.values()) <= 0:
            return False
        return validator(bs) if validator is not None else True

    best = base
    L = base
    R = max_dynamic_batch
    while L <= R:
        mid_raw = (L + R) // 2
        mid = (mid_raw // base) * base
        mid = max(base, mid)
        if can(mid):
            best = mid
            L = mid + base
        else:
            R = mid - base
    return best


def _create_batched_samples(
    buckets: Dict[int, List[Dict[str, np.ndarray]]], *, batch_size: int, seed: int
) -> Tuple[List[Dict[str, np.ndarray]], Dict[int, int], Dict[int, int]]:
    rng = random.Random(seed)
    batched: List[Dict[str, np.ndarray]] = []
    used = {}
    dropped = {}
    for b, arr in buckets.items():
        rng.shuffle(arr)
        full = len(arr) // batch_size
        use = full * batch_size
        used[b] = use
        dropped[b] = len(arr) - use
        for i in range(full):
            ch = arr[i * batch_size : (i + 1) * batch_size]
            batched.append({k: np.stack([s[k] for s in ch], 0) for k in ch[0]})
    rng.shuffle(batched)
    return batched, used, dropped


class PrecompMem(torch.utils.data.Dataset):
    def __init__(self, arr: List[Dict[str, np.ndarray]]):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return {k: torch.from_numpy(v) for k, v in self.arr[i].items()}


class PrecompDisk(torch.utils.data.Dataset):
    def __init__(self, files: List[Path], fmt: str):
        self.files = files
        self.fmt = fmt

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        if self.fmt == 'pt':
            d = torch.load(f)
        else:
            npz = np.load(f)
            d = {k: torch.from_numpy(npz[k]) for k in npz.files}
        return d


def _persist_batches(batched: List[Dict[str, np.ndarray]], out_dir: Path, split: str, fmt: str) -> List[Path]:
    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for i, batch in enumerate(batched):
        fp = out_split / f'batch_{i:06d}.{fmt}'
        if fmt == 'pt':
            torch.save(batch, fp)
        else:
            np.savez_compressed(fp, **batch)
        files.append(fp)
    return files


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cpu') if args.cpu else xla.device()

    boundaries = tuple(int(x.strip()) for x in args.bucket_boundaries.split(','))
    print('=== Gemma torch_xla (dynamic preprocess + train) ===')
    print('1) Loading processor & model...')
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    print('✓ Model ready.')

    print('\n2) Preprocess & bucket...')
    buckets_train, stats_train = _prepare_bucketed_samples(
        split=args.train_split,
        processor=processor,
        dataset_path=args.dataset,
        cache_dir=args.data_cache,
        max_length=max(boundaries),
        boundaries=boundaries,
        instruction=args.instruction,
    )
    buckets_eval, stats_eval = _prepare_bucketed_samples(
        split=args.eval_split,
        processor=processor,
        dataset_path=args.dataset,
        cache_dir=args.data_cache,
        max_length=max(boundaries),
        boundaries=boundaries,
        instruction=args.instruction,
    )
    print('Train counts:', {k: v for k, v in stats_train['bucket_counts'].items() if v})
    print('Eval counts :', {k: v for k, v in stats_eval['bucket_counts'].items() if v})

    print('\n3) Batch-size search...')
    try:
        used, total = _tpuinfo_used_total()
        print(f'[mem] pre-step used={used} total={total}')
    except Exception as e:
        print(f'[mem] skip: {e}')
    validator = _validator_factory(model=model, device=device, buckets=buckets_train)
    base = 8
    bs_train = _compute_bs(stats_train['bucket_counts'], base=base, max_dynamic_batch=args.max_dynamic_batch, validator=validator)
    bs_eval = _compute_bs(stats_eval['bucket_counts'], base=base, max_dynamic_batch=args.max_dynamic_batch, validator=validator)
    print('Computed batch sizes: train', bs_train, 'eval', bs_eval)

    print('\n4) Create batches...')
    batched_train, used_t, drop_t = _create_batched_samples(buckets_train, batch_size=bs_train, seed=args.seed)
    batched_eval, used_e, drop_e = _create_batched_samples(buckets_eval, batch_size=bs_eval, seed=args.seed + 1)
    print('Train used per bucket:', {k: v for k, v in used_t.items() if v})
    print('Eval  used per bucket:', {k: v for k, v in used_e.items() if v})

    files_train: List[Path] = []
    files_eval: List[Path] = []
    if args.persist_dir:
        out_dir = Path(args.persist_dir)
        print('Persisting pre-batched tensors to', out_dir)
        files_train = _persist_batches(batched_train, out_dir, 'train', args.persist_format)
        files_eval = _persist_batches(batched_eval, out_dir, 'eval', args.persist_format)

    if args.preprocess_only:
        print('Preprocess-only flag set; exiting.')
        return

    print('\n5) Build loaders...')
    if args.persist_dir:
        ds_train = PrecompDisk(files_train, args.persist_format)
        ds_eval = PrecompDisk(files_eval, args.persist_format)
    else:
        ds_train = PrecompMem(batched_train)
        ds_eval = PrecompMem(batched_eval)

    def collate(items):
        keys = items[0].keys()
        return {k: torch.stack([it[k] for it in items], 0) for k in keys}

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=None, shuffle=False, num_workers=min(32, os.cpu_count() or 1), collate_fn=collate)
    eval_loader = torch.utils.data.DataLoader(ds_eval, batch_size=None, shuffle=False, num_workers=min(32, os.cpu_count() or 1), collate_fn=collate)

    steps = len(train_loader) * args.train_epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(optim, max(1, steps // 20), steps)

    if args.log_tpu_memory:
        try:
            used_b, total_b = _tpuinfo_used_total()
            print(f'[mem] pre_training: used={used_b/1e9:.2f} / total={total_b/1e9:.2f} GB')
        except Exception as e:
            print(f'[mem] pre_training: failed ({e})')

    print(f"\n6) Train for {steps} steps...")
    model.train()
    for epoch in range(args.train_epochs):
        for step, b in enumerate(pl.MpDeviceLoader(train_loader, device)):
            inp = b['input'].to(device)
            lab = b['target'].squeeze(-1).to(device)
            out = model(input_ids=inp, labels=lab)
            loss = out.loss
            loss.backward()
            xm.optimizer_step(optim)
            optim.zero_grad(set_to_none=True)
            sched.step()
            if step % 10 == 0:
                xm.mark_step()
                print(f'train epoch {epoch} step {step} loss {loss.item():.4f}')
    print('✓ Training finished.')

    if args.log_tpu_memory:
        try:
            used_b, total_b = _tpuinfo_used_total()
            print(f'[mem] post_training: used={used_b/1e9:.2f} / total={total_b/1e9:.2f} GB')
        except Exception as e:
            print(f'[mem] post_training: failed ({e})')

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in pl.MpDeviceLoader(eval_loader, device):
            inp = b['input'].to(device)
            lab = b['target'].squeeze(-1).to(device)
            out = model(input_ids=inp, labels=lab)
            tot += float(out.loss.item())
            n += 1
    print(f'post eval loss: {tot / max(1, n):.4f}')


if __name__ == '__main__':
    main()
