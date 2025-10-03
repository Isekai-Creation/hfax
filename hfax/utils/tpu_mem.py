from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from typing import Optional, Tuple, List, Dict

from .logging import logger


def _tpuinfo_cli_query_json() -> Optional[List[Dict[str, int]]]:
    """Return list of chip dicts with total_bytes/used_bytes via tpu-info JSON."""
    try:
        proc = subprocess.run(
            ["tpu-info", "--metric", "hbm_usage", "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
            data = json.loads(proc.stdout)
            chips = data.get("chips") or data.get("devices") or []
            out = []
            for c in chips:
                t = int(c.get("total_bytes") or c.get("total") or 0)
                u = int(c.get("used_bytes") or c.get("used") or 0)
                if t > 0:
                    out.append({"total": t, "used": u})
            return out or None
    except Exception:
        return None
    return None


def _tpuinfo_cli_query_text() -> Optional[List[Dict[str, int]]]:
    """Parse human output of tpu-info into list of {total, used} bytes per chip."""
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
        # Matches like "0.00 GiB / 31.75 GiB"
        gib_line = re.findall(r"([0-9]+\.[0-9]+)\s*GiB\s*/\s*([0-9]+\.[0-9]+)\s*GiB", txt)
        chips: List[Dict[str, int]] = []
        for used_gib, total_gib in gib_line:
            t = int(float(total_gib) * (1 << 30))
            u = int(float(used_gib) * (1 << 30))
            if t > 0:
                chips.append({"total": t, "used": u})
        return chips or None
    except Exception:
        return None


def _query_chips() -> Optional[List[Dict[str, int]]]:
    return _tpuinfo_cli_query_json() or _tpuinfo_cli_query_text()


def get_xla_mem_info() -> Tuple[int, int]:
    """Return (mem_total_all, mem_total_per_device) in bytes.

    Uses tpu-info via subprocess. Raises RuntimeError if unavailable.
    """
    chips = _query_chips()
    if not chips:
        raise RuntimeError("tpu-info not available or returned no devices")
    totals = [int(c["total"]) for c in chips]
    return sum(totals), min(totals)


def get_xla_cur_mem() -> Tuple[int, int]:
    """Return (used_total_all, max_used_per_device) in bytes via tpu-info."""
    chips = _query_chips()
    if not chips:
        raise RuntimeError("tpu-info not available or returned no devices")
    useds = [int(c["used"]) for c in chips]
    return sum(useds), max(useds)


def log_tpu_memory(label: str) -> None:
    """Log per-replica memory: used/free/total (GiB) using tpu-info."""
    chips = _query_chips()
    if not chips:
        logger.info("[mem] %s: tpu-info unavailable", label)
        return
    totals = [int(c["total"]) for c in chips]
    useds = [int(c["used"]) for c in chips]
    per_total = min(totals) if totals else 0
    per_used = max(useds) if useds else 0
    per_free = max(0, per_total - per_used)
    logger.info(
        "[mem] %s: used=%.2f GiB free=%.2f GiB total=%.2f GiB (per-replica)",
        label,
        per_used / (1 << 30),
        per_free / (1 << 30),
        per_total / (1 << 30),
    )


def per_replica_free_total() -> Optional[Tuple[int, int]]:
    """Return (free_bytes, total_bytes) for a single replica.

    Computed as min(total_per_chip) and free = per_total - max(used_per_chip).
    """
    chips = _query_chips()
    if not chips:
        return None
    totals = [int(c["total"]) for c in chips]
    useds = [int(c["used"]) for c in chips]
    if not totals:
        return None
    per_total = min(totals)
    per_used = max(useds) if useds else 0
    return max(0, per_total - per_used), per_total


class XLAMemoryTimer:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.start_time = None
        self.memory_usage = {
            "mem_total": 0,
            "mem_per_device": 0,
            "max_mem": 0,
            "max_mem_per_device": 0,
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _monitor_memory(self):
        while not self._stop_event.is_set():
            try:
                current_total_used, max_used_per_device = get_xla_cur_mem()
                self.memory_usage["max_mem"] = max(self.memory_usage["max_mem"], current_total_used)
                self.memory_usage["max_mem_per_device"] = max(
                    self.memory_usage["max_mem_per_device"], max_used_per_device
                )
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Timer is already running.")
        self.start_time = time.time()
        try:
            mem_total, mem_per_device = get_xla_mem_info()
            self.memory_usage["mem_total"] = max(self.memory_usage["mem_total"], mem_total)
            self.memory_usage["mem_per_device"] = max(
                self.memory_usage["mem_per_device"], mem_per_device
            )
        except Exception:
            pass
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()

    def stop(self, format_mb: bool = True) -> Dict[str, float]:
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        self.start_time = None
        res = dict(self.memory_usage)
        # reset
        self.memory_usage = {
            "mem_total": 0,
            "mem_per_device": 0,
            "max_mem": 0,
            "max_mem_per_device": 0,
        }
        if format_mb:
            # Convert bytes to GB (consistent with user ask using 1024 base)
            for k in list(res.keys()):
                res[k] = res[k] / (1024 ** 3)
        return res
