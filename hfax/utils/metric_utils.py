from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable

from hfax.metrics import RunMetrics
from hfax.utils.logging import logger


def safe_stats(values: Iterable[float]) -> tuple[float, float]:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.median(values)


def parse_token_counts(raw_values: Iterable[str] | None) -> list[int]:
    if not raw_values:
        return []
    token_counts: list[int] = []
    for entry in raw_values:
        for part in re.split(r"[\s,]+", str(entry)):
            if part:
                token_counts.append(int(part))
    # Remove duplicates while preserving order
    seen = set()
    unique_counts = []
    for count in token_counts:
        if count not in seen:
            unique_counts.append(count)
            seen.add(count)
    return unique_counts


def log_run_metrics(
    context: str, metrics: RunMetrics, *, extra: dict[str, Any] | None = None
):
    fields = {
        "total_s": metrics.total_time,
        "first_token_s": metrics.first_token_time,
        "post_first_s": metrics.post_first_token_time,
        "tokens": metrics.total_tokens,
        "tps_overall": metrics.tokens_per_second_overall,
        "tps_decode": metrics.tokens_per_second_post_first,
        "tps_prefirst": metrics.tokens_per_second_pre_first,
        "sec_per_token": metrics.seconds_per_token_overall,
        "sec_per_token_decode": metrics.seconds_per_token_post_first,
    }
    if extra:
        fields.update(extra)
    field_str = " | ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in fields.items()
    )
    logger.info("%s -> %s", context, field_str)


def log_metrics_summary(label: str, metrics_list: list[RunMetrics]):
    if not metrics_list:
        logger.info("%s -> no runs executed.", label)
        return

    total_avg, total_med = safe_stats(m.total_time for m in metrics_list)
    first_avg, first_med = safe_stats(m.first_token_time for m in metrics_list)
    post_avg, post_med = safe_stats(m.post_first_token_time for m in metrics_list)
    tokens_avg, tokens_med = safe_stats(float(m.total_tokens) for m in metrics_list)
    overall_tps_avg, overall_tps_med = safe_stats(
        m.tokens_per_second_overall for m in metrics_list
    )
    decode_tps_avg, decode_tps_med = safe_stats(
        m.tokens_per_second_post_first for m in metrics_list
    )
    prefirst_tps_avg, prefirst_tps_med = safe_stats(
        m.tokens_per_second_pre_first for m in metrics_list
    )
    sec_token_avg, sec_token_med = safe_stats(
        m.seconds_per_token_overall for m in metrics_list
    )
    sec_token_decode_avg, sec_token_decode_med = safe_stats(
        m.seconds_per_token_post_first for m in metrics_list
    )

    logger.info(
        "%s summary -> runs=%d | total_s(avg/med)=%.4f/%.4f | first_s(avg/med)=%.4f/%.4f | post_first_s(avg/med)=%.4f/%.4f | tokens(avg/med)=%.2f/%.2f | tps_overall(avg/med)=%.2f/%.2f | tps_decode(avg/med)=%.2f/%.2f | tps_prefirst(avg/med)=%.2f/%.2f | sec_per_token(avg/med)=%.4f/%.4f | sec_per_token_decode(avg/med)=%.4f/%.4f",
        label,
        len(metrics_list),
        total_avg,
        total_med,
        first_avg,
        first_med,
        post_avg,
        post_med,
        tokens_avg,
        tokens_med,
        overall_tps_avg,
        overall_tps_med,
        decode_tps_avg,
        decode_tps_med,
        prefirst_tps_avg,
        prefirst_tps_med,
        sec_token_avg,
        sec_token_med,
        sec_token_decode_avg,
        sec_token_decode_med,
    )


def metrics_brief(metrics_list: list[RunMetrics]) -> str:
    if not metrics_list:
        return "no data"
    total_avg, _ = safe_stats(m.total_time for m in metrics_list)
    first_avg, _ = safe_stats(m.first_token_time for m in metrics_list)
    overall_tps_avg, _ = safe_stats(m.tokens_per_second_overall for m in metrics_list)
    decode_tps_avg, _ = safe_stats(
        m.tokens_per_second_post_first for m in metrics_list
    )
    prefirst_tps_avg, _ = safe_stats(
        m.tokens_per_second_pre_first for m in metrics_list
    )
    return (
        f"avg_total={total_avg:.4f}s, avg_first_token={first_avg:.4f}s, "
        f"avg_tps={overall_tps_avg:.2f}, avg_decode_tps={decode_tps_avg:.2f}, "
        f"avg_prefirst_tps={prefirst_tps_avg:.2f}"
    )


def summarize_metrics(metrics_list: list[RunMetrics]) -> dict[str, float] | None:
    if not metrics_list:
        return None
    avg_total, _ = safe_stats(m.total_time for m in metrics_list)
    avg_first, _ = safe_stats(m.first_token_time for m in metrics_list)
    avg_post, _ = safe_stats(m.post_first_token_time for m in metrics_list)
    avg_tokens, _ = safe_stats(float(m.total_tokens) for m in metrics_list)
    avg_tps, _ = safe_stats(m.tokens_per_second_overall for m in metrics_list)
    avg_decode_tps, _ = safe_stats(
        m.tokens_per_second_post_first for m in metrics_list
    )
    avg_prefirst_tps, _ = safe_stats(
        m.tokens_per_second_pre_first for m in metrics_list
    )
    avg_sec_per_token, _ = safe_stats(
        m.seconds_per_token_overall for m in metrics_list
    )
    avg_sec_per_token_decode, _ = safe_stats(
        m.seconds_per_token_post_first for m in metrics_list
    )
    return {
        "avg_total": avg_total,
        "avg_first": avg_first,
        "avg_post": avg_post,
        "avg_tokens": avg_tokens,
        "avg_tps": avg_tps,
        "avg_decode_tps": avg_decode_tps,
        "avg_prefirst_tps": avg_prefirst_tps,
        "avg_sec_per_token": avg_sec_per_token,
        "avg_sec_per_token_decode": avg_sec_per_token_decode,
    }
