#!/usr/bin/env python3
"""
TPU memory query helper (Python 3.10 compatible).

Features
- Prefer tpu_info for per-chip stats; fall back to torch_xla when needed.
- Computes per-replica memory for SPMD: total=min(total per chip), used=max(used per chip),
  free=min(total-used per chip).
- Outputs human-readable or JSON. Optional watch mode.

Usage examples
- Human one-shot (SPMD):
  `python3.10 scripts/tpu_memory_info.py --spmd`
- JSON output:
  `python3.10 scripts/tpu_memory_info.py --spmd --format json`
- Watch 5 samples at 2s:
  `python3.10 scripts/tpu_memory_info.py --spmd --watch 5 --interval 2`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChipUsage:
    total_memory: int
    memory_usage: int


@dataclass
class MemSnapshot:
    provider: str
    timestamp: str
    spmd: bool
    chip_count: int
    chips: List[ChipUsage]
    per_replica_total: Optional[int]
    per_replica_used: Optional[int]
    per_replica_free: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "timestamp": self.timestamp,
            "spmd": self.spmd,
            "chip_count": self.chip_count,
            "chips": [
                {"total": c.total_memory, "used": c.memory_usage, "free": c.total_memory - c.memory_usage}
                for c in self.chips
            ],
            "per_replica": {
                "total_bytes": self.per_replica_total,
                "used_bytes": self.per_replica_used,
                "free_bytes": self.per_replica_free,
            },
        }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _calc_per_replica_spmd(chips: List[ChipUsage]) -> Tuple[int, int, int]:
    if not chips:
        return 0, 0, 0
    totals = [c.total_memory for c in chips]
    useds = [c.memory_usage for c in chips]
    frees = [t - u for t, u in zip(totals, useds)]
    per_replica_total = min(totals)
    per_replica_used = max(useds)
    per_replica_free = min(frees)
    return per_replica_total, per_replica_used, per_replica_free


def _query_with_tpu_info(spmd: bool) -> Optional[MemSnapshot]:
    try:
        from tpu_info import device as tpu_device  # type: ignore
        from tpu_info import metrics as tpu_metrics  # type: ignore
    except Exception:
        return None
    try:
        chip_type, count = tpu_device.get_local_chips()
        if not chip_type or not count:
            return None
        device_usage = tpu_metrics.get_chip_usage(chip_type)
        chips = [
            ChipUsage(total_memory=int(d.total_memory), memory_usage=int(d.memory_usage))
            for d in device_usage
        ]
        per_total = per_used = per_free = None
        if spmd:
            per_total, per_used, per_free = _calc_per_replica_spmd(chips)
        return MemSnapshot(
            provider="tpu_info",
            timestamp=_iso_now(),
            spmd=spmd,
            chip_count=len(chips),
            chips=chips,
            per_replica_total=per_total,
            per_replica_used=per_used,
            per_replica_free=per_free,
        )
    except Exception:
        return None


def query_memory(spmd: bool) -> MemSnapshot:
    """Query TPU memory.

    Args:
      spmd: Interpret results under SPMD semantics.
    Returns:
      MemSnapshot (raises RuntimeError if no provider works).
    """
    snap = _query_with_tpu_info(spmd)
    if snap is None:
        raise RuntimeError("tpu_info is required to query TPU memory")
    return snap


def _print_human(snap: MemSnapshot) -> None:
    def gb(x: Optional[int]) -> str:
        return f"{(x or 0)/1e9:.2f} GB"

    print(f"Provider: {snap.provider} | Time: {snap.timestamp} | SPMD={snap.spmd}")
    if snap.spmd:
        print(
            f"Per-replica: used={gb(snap.per_replica_used)}, "
            f"free={gb(snap.per_replica_free)}, total={gb(snap.per_replica_total)}"
        )
    print(f"Local chips: {snap.chip_count}")
    for idx, c in enumerate(snap.chips):
        free = c.total_memory - c.memory_usage
        print(
            f"  chip[{idx}]: used={gb(c.memory_usage)}, free={gb(free)}, total={gb(c.total_memory)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # tpu_info is the only provider supported
    ap.add_argument("--spmd", action="store_true", help="Interpret results as SPMD per-replica")
    ap.add_argument("--format", choices=["human", "json"], default="human")
    ap.add_argument("--watch", type=int, default=1, help="Number of samples to take (default: 1)")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between samples in watch mode")
    args = ap.parse_args()

    for i in range(max(1, args.watch)):
        try:
            snap = query_memory(spmd=args.spmd)
        except Exception as exc:
            print(f"TPU memory query failed: {exc}", file=sys.stderr)
            sys.exit(2)

        if args.format == "json":
            print(json.dumps(snap.to_dict(), separators=(",", ":")))
        else:
            _print_human(snap)
        if i + 1 < args.watch:
            time.sleep(max(0.0, args.interval))


if __name__ == "__main__":
    main()
