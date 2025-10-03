#!/usr/bin/env python3
from __future__ import annotations

"""
Preprocess Gemma data into length-bucketed, padded, pre-batched shards.

This script runs in a separate process (no torch_xla, no model init) to avoid
any interaction between Hugging Face tokenizers and DataLoader forking used in
training. It saves ready-to-train batches to disk as .pt files.

Example
  python scripts/preprocess_gemma_dynamic_xla.py \
    --model-id unsloth/gemma-3-4b-it \
    --dataset unsloth/LaTeX_OCR \
    --train-split 'train[:3000]' --eval-split 'test[:3000]' \
    --bucket-boundaries 512,1024,2048,4096 \
    --batch-size 64 \
    --output-dir /dev/shm/gemma_shards
"""

import argparse
import concurrent.futures
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
try:
    from tqdm.auto import tqdm
    from tqdm.contrib.concurrent import thread_map
except Exception:
    tqdm = None
    thread_map = None  # type: ignore
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from transformers import AutoProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', default='unsloth/gemma-3-4b-it')
    p.add_argument('--dataset', default='unsloth/LaTeX_OCR')
    p.add_argument('--train-split', default='train[:3000]')
    p.add_argument('--eval-split', default='test[:3000]')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bucket-boundaries', default='512,1024,2048,4096')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--instruction', default='Convert the equation images to LaTeX equations.')
    p.add_argument('--cache-dir', default='/dev/shm/dataset_cache')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1)))
    return p.parse_args()


def _collect_image_token_ids(processor: AutoProcessor, pad_token_id: int) -> set[int]:
    s: set[int] = set()
    for key in ('boi_token', 'eoi_token'):
        v = processor.tokenizer.special_tokens_map.get(key)
        if v is None:
            continue
        vv = [v] if isinstance(v, str) else list(v)
        for t in vv:
            cid = processor.tokenizer.convert_tokens_to_ids(t)
            if cid is not None:
                s.add(int(cid))
    pid = processor.tokenizer.convert_tokens_to_ids('<image>')
    if pid is not None:
        s.add(int(pid))
    s.discard(pad_token_id)
    s.add(262144)
    return s


def _prepare_buckets(*, split: str, dataset: str, cache_dir: str, processor: AutoProcessor, boundaries: Tuple[int, ...], max_length: int, instruction: str, workers: int, model_id: str) -> Tuple[Dict[int, List[Dict[str, np.ndarray]]], Dict[str, Any]]:
    ds = load_dataset(dataset, split=split, cache_dir=cache_dir).with_format('python')
    pad = processor.tokenizer.pad_token_id
    image_ids = _collect_image_token_ids(processor, pad)
    buckets: Dict[int, List[Dict[str, np.ndarray]]] = {b: [] for b in boundaries}
    drop = 0

    def _proc(ex: Dict[str, Any]) -> Tuple[int | None, Dict[str, np.ndarray] | None]:
        conv = [
            {'role': 'user', 'content': [{'type': 'text', 'text': instruction}, {'type': 'image', 'image': ex['image']}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': ex['text']}]},
        ]
        rendered = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        bat = processor(text=[rendered], images=[ex['image'].convert('RGB')], return_tensors='np', padding=False, max_length=max_length, truncation=True)
        ids = bat['input_ids'][0].astype(np.int32)
        b = next((bb for bb in boundaries if len(ids) <= bb), None)
        if b is None:
            return None, None
        padded = np.full((b,), pad, dtype=np.int32)
        padded[: len(ids)] = ids
        labels = padded.copy()
        if image_ids:
            labels[np.isin(labels, list(image_ids))] = -100
        labels[labels == pad] = -100
        targets = labels[:, None]
        loss_mask = (targets != -100).astype(np.int32)
        return b, {'input': padded, 'target': targets, 'loss_mask': loss_mask}

    total = len(ds)
    print(f"  [{split}] preprocessing {total} examples with {workers} threads...")
    if thread_map is not None and tqdm is not None:
        results = thread_map(_proc, ds, max_workers=workers, desc=f"[{split}]", unit="sample", dynamic_ncols=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_proc, ds))
    for b, sample in results:
        if b is None or sample is None:
            drop += 1
            continue
        buckets[b].append(sample)
    return buckets, {'bucket_counts': {k: len(v) for k, v in buckets.items()}, 'dropped_long': drop}


def _persist_batches(buckets: Dict[int, List[Dict[str, np.ndarray]]], *, out_dir: Path, split: str, batch_size: int) -> Dict[str, Any]:
    out = out_dir / split
    out.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    written = 0
    for b in sorted(buckets.keys()):
        arr = buckets[b]
        # full batches only
        full = len(arr) // batch_size
        for i in range(full):
            chunk = arr[i * batch_size : (i + 1) * batch_size]
            batch = {k: np.stack([s[k] for s in chunk], 0) for k in chunk[0]}
            fp = out / f'b{b}_idx{i:06d}.pt'
            torch.save(batch, fp)
            files.append(str(fp))
            written += 1
    manifest = {
        'split': split,
        'count_batches': written,
        'files': files,
    }
    with open(out / 'manifest.json', 'w') as f:
        json.dump(manifest, f)
    return manifest


def main() -> None:
    args = parse_args()
    boundaries = tuple(int(x.strip()) for x in args.bucket_boundaries.split(','))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('1) Loading processor...')
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print('2) Bucketing train...')
    b_train, st_train = _prepare_buckets(
        split=args.train_split,
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        processor=processor,
        boundaries=boundaries,
        max_length=args.max_length,
        instruction=args.instruction,
        workers=args.workers,
        model_id=args.model_id,
    )
    print('   train bucket counts:', {k: v for k, v in st_train['bucket_counts'].items() if v})

    print('3) Bucketing eval...')
    b_eval, st_eval = _prepare_buckets(
        split=args.eval_split,
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        processor=processor,
        boundaries=boundaries,
        max_length=args.max_length,
        instruction=args.instruction,
        workers=args.workers,
        model_id=args.model_id,
    )
    print('   eval  bucket counts:', {k: v for k, v in st_eval['bucket_counts'].items() if v})

    print('4) Writing pre-batched shards...')
    man_train = _persist_batches(b_train, out_dir=out_dir, split='train', batch_size=args.batch_size)
    man_eval = _persist_batches(b_eval, out_dir=out_dir, split='eval', batch_size=args.batch_size)
    with open(out_dir / 'MANIFEST.json', 'w') as f:
        json.dump({'train': man_train, 'eval': man_eval}, f)
    print('   wrote', man_train['count_batches'], 'train batches,', man_eval['count_batches'], 'eval batches to', out_dir)


if __name__ == '__main__':
    main()
