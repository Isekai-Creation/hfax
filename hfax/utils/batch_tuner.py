from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from etils import enp

from .logging import logger
from .tpu_mem import per_replica_free_total


ArrayBatch = Dict[str, np.ndarray]


@dataclass
class BucketTuneResult:
    best_per_bucket: Dict[int, int]
    per_sample_bytes: Dict[int, int]
    predicted_caps: Dict[int, int]


def _make_worst_case_batch(samples: List[ArrayBatch], batch_size: int) -> Optional[ArrayBatch]:
    if not samples:
        return None
    template = samples[0]
    def _tile(x: np.ndarray) -> np.ndarray:
        return np.stack([x] * batch_size, axis=0)
    return {k: _tile(v) for k, v in template.items()}


def _validate_step(trainstep, batch_np: ArrayBatch) -> bool:
    # Clear any compiled/computation cache so each validation reflects only this try.
    try:
        jax.clear_caches()
    except Exception:
        pass
    batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
    try:
        elem_spec = {k: enp.ArraySpec(shape=v.shape, dtype=v.dtype) for k, v in batch_np.items()}
        state = trainstep.init(elem_spec=elem_spec)
        state, _ = trainstep.step(state, batch)
        jax.block_until_ready(state.step)
        return True
    except Exception as exc:
        msg = str(exc)
        oom = (
            "RESOURCE_EXHAUSTED",
            "Out of memory",
            "OOM",
            "insufficient memory",
            "HbmAllocator",
            "Resource exhausted",
        )
        if any(m in msg for m in oom) or "memory" in msg.lower():
            return False
        # Re-raise non-oom errors
        raise
    finally:
        try:
            jax.clear_caches()
        except Exception:
            pass


class BucketBatchTuner:
    def __init__(
        self,
        *,
        base_batch_size: int = 8,
        max_dynamic_batch: int = 128,
        safety_ratio: float = 0.05,
        safety_min_bytes: int = 1 << 30,
        verbose: bool = False,
        # Candidate-expansion controls
        use_candidate_expansion: bool = True,
        candidate_depth: int = 2,
        candidate_expansions: int = 8,
        num_devices: Optional[int] = None,
    ) -> None:
        self.base = int(max(1, base_batch_size))
        self.max_dynamic = int(max(1, max_dynamic_batch))
        self.safety_ratio = float(max(0.0, safety_ratio))
        self.safety_min = int(max(0, safety_min_bytes))
        self.verbose = bool(verbose)
        self.use_candidate_expansion = bool(use_candidate_expansion)
        self.candidate_depth = int(max(0, candidate_depth))
        self.candidate_expansions = int(max(0, candidate_expansions))
        if num_devices is None:
            try:
                num_devices = max(1, jax.local_device_count())
            except Exception:
                num_devices = 1
        self.num_devices = int(max(1, num_devices))
        self._bucket_states: Dict[int, Tuple[object, object]] = {}

    def _ensure_bucket_state(
        self,
        *,
        bucket: int,
        trainstep,
        sample_template: ArrayBatch,
    ) -> None:
        if bucket in self._bucket_states:
            return
        elem_spec = {
            k: enp.ArraySpec(shape=(self.base,) + v.shape, dtype=v.dtype)
            for k, v in sample_template.items()
        }
        state = trainstep.init(elem_spec=elem_spec)
        host_state = jax.device_get(state)
        shardings = jax.tree.map(
            lambda x: getattr(x, "sharding", None), state
        )
        self._bucket_states[bucket] = (host_state, shardings)

    def _fresh_state(self, bucket: int):
        entry = self._bucket_states.get(bucket)
        if entry is None:
            raise RuntimeError("State not initialized for bucket")
        host_state, shardings = entry

        def put(val, sharding):
            if sharding is not None:
                return jax.device_put(val, sharding)
            return jax.device_put(val)

        return jax.tree.map(put, host_state, shardings)

    def _validate_step(
        self,
        *,
        bucket: int,
        trainstep,
        batch_np: ArrayBatch,
    ) -> bool:
        try:
            jax.clear_caches()
        except Exception:
            pass
        batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
        try:
            state = self._fresh_state(bucket)
            state, _ = trainstep.step(state, batch)
            jax.block_until_ready(state.step)
            return True
        except Exception as exc:
            msg = str(exc)
            oom = (
                "RESOURCE_EXHAUSTED",
                "Out of memory",
                "OOM",
                "insufficient memory",
                "HbmAllocator",
                "Resource exhausted",
            )
            if any(m in msg for m in oom) or "memory" in msg.lower():
                return False
            raise
        finally:
            try:
                jax.clear_caches()
            except Exception:
                pass

    def _estimate_cap(
        self,
        *,
        bucket: int,
        trainstep,
        samples: List[ArrayBatch],
    ) -> Optional[Tuple[int, int]]:
        """Return (per_sample_bytes, predicted_cap) or None if not on TPU."""
        if not samples:
            return None
        self._ensure_bucket_state(
            bucket=bucket, trainstep=trainstep, sample_template=samples[0]
        )
        mem = per_replica_free_total()
        if mem is None:
            if self.verbose:
                logger.info("[tuner] No TPU memory provider available; skipping.")
            return None

        # Validate once at base batch size
        base_batch = _make_worst_case_batch(samples, self.base)
        if base_batch is None:
            return None
        if self.verbose:
            logger.info("[tuner] validating base batch_size=%d", self.base)
        if not self._validate_step(bucket=bucket, trainstep=trainstep, batch_np=base_batch):
            if self.verbose:
                logger.info("[tuner] base batch failed; abort.")
            return None

        # Read memory after base step
        post = per_replica_free_total()
        if post is None:
            return None
        free_bytes, total_bytes = post
        reserved = max(0, total_bytes - free_bytes)
        per_sample = max(1, reserved // self.base)
        safety = max(int(total_bytes * self.safety_ratio), self.safety_min)
        headroom = max(0, free_bytes - safety)
        predicted = max(self.base, min(self.max_dynamic, (headroom // per_sample)))
        # Round down to multiple of base
        predicted = (predicted // self.base) * self.base
        predicted = max(self.base, min(predicted, self.max_dynamic))
        if self.verbose:
            to_gib = lambda x: x / (1024**3)
            logger.info(
                "[tuner] total=%.2f GiB free=%.2f GiB reserved=%.2f GiB per_sample=%.2f GiB predicted=%d",
                to_gib(total_bytes),
                to_gib(free_bytes),
                to_gib(reserved),
                to_gib(per_sample),
                predicted,
            )
        return per_sample, predicted

    def _binary_search(
        self,
        *,
        bucket: int,
        trainstep,
        samples: List[ArrayBatch],
        cap: int,
    ) -> int:
        lo, hi = self.base, max(self.base, cap)
        best = self.base
        while lo <= hi:
            mid = ((lo + hi) // (2 * self.base)) * self.base
            mid = max(self.base, mid)
            batch = _make_worst_case_batch(samples, mid)
            ok = self._validate_step(bucket=bucket, trainstep=trainstep, batch_np=batch) if batch is not None else False
            if self.verbose:
                logger.info("[tuner] try %d ... %s", mid, "OK" if ok else "OOM")
            if ok:
                best = mid
                lo = mid + self.base
            else:
                hi = mid - self.base
        return max(self.base, min(best, self.max_dynamic))

    # ---------- Candidate generation ----------
    @staticmethod
    def _ceil_to_multiple(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    @staticmethod
    def _floor_to_multiple(x: int, m: int) -> int:
        return (x // m) * m

    def _doubling_skeleton(self, lo: int, hi: int) -> List[int]:
        vals = [lo]
        v = lo
        while v * 2 <= hi:
            v *= 2
            vals.append(v)
        if vals[-1] != hi:
            vals.append(hi)
        return vals

    def _nearest_multiple(self, x: int, m: int) -> int:
        # Prefer the nearest multiple; tie -> lower
        down = self._floor_to_multiple(x, m)
        up = self._ceil_to_multiple(x, m)
        if (x - down) <= (up - x):
            return down
        return up

    def _expand_candidates(self, lo: int, hi: int) -> List[int]:
        # Build initial skeleton and then insert midpoints breadth-first up to limits
        lo = max(lo, self.num_devices)
        lo = self._ceil_to_multiple(lo, self.num_devices)
        hi = self._floor_to_multiple(hi, self.num_devices)
        if lo > hi:
            return []
        base_list = self._doubling_skeleton(lo, hi)
        candidates = list(dict.fromkeys(base_list))  # preserve order, unique

        # Build interval queue from current candidates
        from collections import deque
        def intervals(seq: List[int]) -> List[Tuple[int,int]]:
            return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]

        q = deque(intervals(base_list))
        depth = 0
        expansions_done = 0
        # We track intervals per level to respect max depth
        while q and depth < self.candidate_depth and expansions_done < self.candidate_expansions:
            level_count = len(q)
            for _ in range(level_count):
                a, b = q.popleft()
                if b - a < self.num_devices:
                    continue
                mid_raw = (a + b) // 2
                mid = self._nearest_multiple(mid_raw, self.num_devices)
                # Constrain strictly inside (a, b)
                if mid <= a:
                    mid = a + self.num_devices
                if mid >= b:
                    mid = b - self.num_devices
                if mid <= a or mid >= b:
                    continue
                if mid not in candidates:
                    # Insert while preserving sort order
                    candidates.append(mid)
                    expansions_done += 1
                    # Push sub-intervals for next levels
                    q.append((a, mid))
                    q.append((mid, b))
                    if expansions_done >= self.candidate_expansions:
                        break
            depth += 1
        # Return unique sorted ascending
        return sorted(set(candidates))

    def _search_by_candidates(
        self,
        *,
        bucket: int,
        trainstep,
        samples: List[ArrayBatch],
        lo: int,
        hi: int,
    ) -> int:
        base_list = self._expand_candidates(lo, hi) if self.use_candidate_expansion else []
        if not base_list:
            base_list = [lo, hi]
        # Ensure bounds included and sorted unique
        base_list.extend([lo, hi])
        cands = sorted(set(x for x in base_list if lo <= x <= hi))
        if not cands:
            cands = [lo]

        best = lo
        left, right = 0, len(cands) - 1
        while left <= right:
            mid = (left + right) // 2
            bsz = cands[mid]
            batch = _make_worst_case_batch(samples, bsz)
            ok = self._validate_step(bucket=bucket, trainstep=trainstep, batch_np=batch) if batch is not None else False
            if self.verbose:
                logger.info("[tuner] candidate %d ... %s", bsz, "OK" if ok else "OOM")
            if ok:
                best = bsz
                left = mid + 1
            else:
                right = mid - 1
        return max(self.base, min(best, self.max_dynamic))

    def tune_for_buckets(
        self,
        *,
        bucket_samples: Dict[int, List[ArrayBatch]],
        trainstep,
    ) -> Optional[BucketTuneResult]:
        if not bucket_samples:
            return None
        per_sample_bytes: Dict[int, int] = {}
        predicted_caps: Dict[int, int] = {}
        best_per_bucket: Dict[int, int] = {}

        # Sort buckets desc so larger contexts set lower caps conservatively
        for boundary in sorted(bucket_samples.keys(), reverse=True):
            samples = bucket_samples.get(boundary, [])
            if not samples:
                continue
            est = self._estimate_cap(bucket=boundary, trainstep=trainstep, samples=samples)
            if est is None:
                # No memory provider: fall back to candidate search up to max_dynamic
                predicted = self.max_dynamic
                predicted_caps[boundary] = predicted
                per_sample_bytes[boundary] = 0
                best = self._search_by_candidates(
                    bucket=boundary,
                    trainstep=trainstep,
                    samples=samples,
                    lo=self.base,
                    hi=predicted,
                )
            else:
                per_sample, predicted = est
                per_sample_bytes[boundary] = per_sample
                predicted_caps[boundary] = predicted
                best = self._search_by_candidates(
                    bucket=boundary,
                    trainstep=trainstep,
                    samples=samples,
                    lo=self.base,
                    hi=predicted,
                )
            best_per_bucket[boundary] = best
            if self.verbose:
                logger.info("[tuner] bucket %d -> best %d", boundary, best)

        if not best_per_bucket:
            return None
        return BucketTuneResult(
            best_per_bucket=best_per_bucket,
            per_sample_bytes=per_sample_bytes,
            predicted_caps=predicted_caps,
        )
