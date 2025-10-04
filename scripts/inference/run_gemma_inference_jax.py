#!/usr/bin/env python3
"""
Runs inference on a Gemma model using the hfax library with optional quantization and benchmarking.
"""
from __future__ import annotations

import argparse
import io
import math
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import hfax
import jax
import jax.numpy as jnp
import numpy as np
import requests
from PIL import Image

from hfax.gm.text import _template
from hfax.utils import metrics_store
from hfax.utils.logging import logger

# Import logging and profiling utilities

# Convenience aliases
gm = hfax.gm
peft = hfax.peft


@dataclass(frozen=True)
class RunMetrics:
    """Per-run timing and throughput metrics."""

    total_time: float
    first_token_time: float
    total_tokens: int

    @property
    def post_first_token_time(self) -> float:
        return max(self.total_time - self.first_token_time, 0.0)

    @property
    def decode_tokens(self) -> int:
        return max(self.total_tokens - 1, 0)

    @property
    def tokens_per_second_overall(self) -> float:
        return (
            (self.total_tokens / self.total_time)
            if self.total_time > 0
            else float("nan")
        )

    @property
    def tokens_per_second_post_first(self) -> float:
        return (
            self.decode_tokens / self.post_first_token_time
            if self.post_first_token_time > 0 and self.decode_tokens > 0
            else float("nan")
        )

    @property
    def tokens_per_second_pre_first(self) -> float:
        return (
            (1.0 / self.first_token_time) if self.first_token_time > 0 else float("inf")
        )

    @property
    def seconds_per_token_overall(self) -> float:
        return (
            (self.total_time / self.total_tokens)
            if self.total_tokens > 0
            else float("nan")
        )

    @property
    def seconds_per_token_post_first(self) -> float:
        return (
            self.post_first_token_time / self.decode_tokens
            if self.decode_tokens > 0
            else float("nan")
        )


def _safe_stats(values: Iterable[float]) -> tuple[float, float]:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.median(values)


def _parse_token_counts(raw_values: Iterable[str] | None) -> list[int]:
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


def _log_run_metrics(
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


def _log_metrics_summary(label: str, metrics_list: list[RunMetrics]):
    if not metrics_list:
        logger.info("%s -> no runs executed.", label)
        return

    total_avg, total_med = _safe_stats(m.total_time for m in metrics_list)
    first_avg, first_med = _safe_stats(m.first_token_time for m in metrics_list)
    post_avg, post_med = _safe_stats(m.post_first_token_time for m in metrics_list)
    tokens_avg, tokens_med = _safe_stats(float(m.total_tokens) for m in metrics_list)
    overall_tps_avg, overall_tps_med = _safe_stats(
        m.tokens_per_second_overall for m in metrics_list
    )
    decode_tps_avg, decode_tps_med = _safe_stats(
        m.tokens_per_second_post_first for m in metrics_list
    )
    prefirst_tps_avg, prefirst_tps_med = _safe_stats(
        m.tokens_per_second_pre_first for m in metrics_list
    )
    sec_token_avg, sec_token_med = _safe_stats(
        m.seconds_per_token_overall for m in metrics_list
    )
    sec_token_decode_avg, sec_token_decode_med = _safe_stats(
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


def _metrics_brief(metrics_list: list[RunMetrics]) -> str:
    if not metrics_list:
        return "no data"
    total_avg, _ = _safe_stats(m.total_time for m in metrics_list)
    first_avg, _ = _safe_stats(m.first_token_time for m in metrics_list)
    overall_tps_avg, _ = _safe_stats(m.tokens_per_second_overall for m in metrics_list)
    decode_tps_avg, _ = _safe_stats(
        m.tokens_per_second_post_first for m in metrics_list
    )
    prefirst_tps_avg, _ = _safe_stats(
        m.tokens_per_second_pre_first for m in metrics_list
    )
    return (
        f"avg_total={total_avg:.4f}s, avg_first_token={first_avg:.4f}s, "
        f"avg_tps={overall_tps_avg:.2f}, avg_decode_tps={decode_tps_avg:.2f}, "
        f"avg_prefirst_tps={prefirst_tps_avg:.2f}"
    )


def _summarize_metrics(metrics_list: list[RunMetrics]) -> dict[str, float] | None:
    if not metrics_list:
        return None
    avg_total, _ = _safe_stats(m.total_time for m in metrics_list)
    avg_first, _ = _safe_stats(m.first_token_time for m in metrics_list)
    avg_post, _ = _safe_stats(m.post_first_token_time for m in metrics_list)
    avg_tokens, _ = _safe_stats(float(m.total_tokens) for m in metrics_list)
    avg_tps, _ = _safe_stats(m.tokens_per_second_overall for m in metrics_list)
    avg_decode_tps, _ = _safe_stats(
        m.tokens_per_second_post_first for m in metrics_list
    )
    avg_prefirst_tps, _ = _safe_stats(
        m.tokens_per_second_pre_first for m in metrics_list
    )
    avg_sec_per_token, _ = _safe_stats(
        m.seconds_per_token_overall for m in metrics_list
    )
    avg_sec_per_token_decode, _ = _safe_stats(
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




def setup_model(args: argparse.Namespace) -> tuple:
    """Sets up the model, parameters, and tokenizer."""
    logger.info("Loading model, tokenizer, and parameters...")

    # 1. Build model (normal or quantized)
    if args.quant_method == "NONE":
        logger.info("Using normal (non-quantized) model.")
        model = gm.nn.Gemma3_4B()
    else:
        logger.info(f"Using {args.quant_method} quantized model.")
        q_dtype = jnp.int8 if args.quant_method == "INT8" else jnp.int4
        model = gm.nn.IntWrapper(model=gm.nn.Gemma3_4B(text_only=True), dtype=q_dtype)

    # 2. Load parameters
    ckpt_path = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    logger.info(f"Loading parameters from: {ckpt_path}")
    params = gm.ckpts.load_params(path=ckpt_path)

    # 3. Quantize if needed
    if args.quant_method != "NONE":
        logger.info(f"Quantizing parameters with method: {args.quant_method}...")
        params = peft.quantize(
            params,
            method=args.quant_method,
            checkpoint_kernel_key="w",
        )
        logger.info("Quantization complete.")

    # 4. Tokenizer
    tokenizer = gm.text.Gemma3Tokenizer()

    return model, params, tokenizer


def run_inference_with_metrics(
    sampler: gm.text.ChatSampler,
    prompt: str,
    max_new_tokens: int,
    image: np.ndarray | None = None,
) -> tuple[str, RunMetrics]:
    """Runs inference while capturing detailed timing metrics."""

    # Reset state for single-turn benchmarking to avoid cache carryover.
    object.__setattr__(sampler, "last_state", None)
    object.__setattr__(sampler, "turns", [])

    formatted_prompt = _template.PROMPT.format(prompt)

    start_time = time.perf_counter()
    stream_iter = sampler.sampler.sample(
        formatted_prompt,
        images=image,
        max_new_tokens=max_new_tokens,
        stream=True,
        return_state=True,
        last_state=None,
    )

    last_output = None
    first_token_time = None

    for output in stream_iter:
        # Ensure device work is accounted for in timing.
        jax.block_until_ready(output.state.predicted_tokens)
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now - start_time
        last_output = output

    end_time = time.perf_counter()

    if last_output is None:
        # Fallback: no tokens generated during streaming. Run non-streaming call
        # to obtain text and approximate metrics.
        logger.warning("Streaming yielded no tokens; falling back to blocking call.")
        text = sampler.chat(prompt, max_new_tokens=max_new_tokens, images=image)
        total_time = end_time - start_time
        total_tokens = len(sampler.tokenizer.encode(text))
        metrics = RunMetrics(
            total_time=total_time,
            first_token_time=total_time,
            total_tokens=total_tokens,
        )
        return text, metrics

    text = last_output.text.removesuffix("<end_of_turn>")
    total_time = end_time - start_time
    total_tokens = int(last_output.state.step)
    first_token_time = first_token_time or total_time

    metrics = RunMetrics(
        total_time=total_time,
        first_token_time=first_token_time,
        total_tokens=total_tokens,
    )

    # Update sampler state to mirror ChatSampler.chat behaviour.
    object.__setattr__(sampler, "last_state", last_output.state)

    return text, metrics


def main(args: argparse.Namespace):
    """Main inference and benchmarking function."""
    logger.info("Run type set to: %s", args.run_type)
    logger.info("TPU type set to: %s", args.tpu_type)
    logger.info("Beginning run with options:")
    for key, value in sorted(vars(args).items()):
        logger.info("  %s: %s", key, value)

    model, params, tokenizer = setup_model(args)

    logger.info("Instantiating ChatSampler...")
    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
    )

    logger.info("Preparing inputs...")
    image = None
    needs_image = (
        args.benchmark_runs > 0 and args.benchmark_with_image
    ) or not args.final_text_only
    if needs_image:
        try:
            response = requests.get(args.image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image = np.array(image)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image: {e}")
            image = None
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            image = None

    warmup_records: list[dict[str, Any]] = []
    benchmark_records: list[dict[str, Any]] = []

    token_counts = _parse_token_counts(args.benchmark_token_counts)
    if not token_counts:
        token_counts = [args.max_new_tokens]

    run_image = args.benchmark_with_image
    run_text = args.benchmark_text_only
    if not run_image and not run_text:
        if image is not None:
            run_image = True
        else:
            run_text = True

    config_plan = []
    if run_image:
        if image is not None:
            config_plan.append({"label": "With Image", "with_image": True})
        else:
            logger.warning(
                "Image requested for benchmarking but unavailable; skipping image benchmarks."
            )
    if run_text:
        config_plan.append({"label": "Text-Only", "with_image": False})

    for tokens in token_counts:
        logger.info("=== Token bucket: %s ===", tokens)
        for config in config_plan:
            use_image = config["with_image"]
            if use_image and image is None:
                continue

            label = config["label"]
            logger.info(
                "Preparing %s runs (tokens=%s, image_used=%s)",
                label,
                tokens,
                use_image,
            )

            if args.warmup_runs > 0:
                warmup_metrics_list: list[RunMetrics] = []
                for run_idx in range(args.warmup_runs):
                    _, metrics = run_inference_with_metrics(
                        sampler,
                        args.prompt,
                        tokens,
                        image=image if use_image else None,
                    )
                    warmup_metrics_list.append(metrics)
                    _log_run_metrics(
                        f"Warmup {label} run {run_idx + 1}/{args.warmup_runs}",
                        metrics,
                        extra={
                            "max_new_tokens": tokens,
                            "image_used": use_image,
                        },
                    )
                _log_metrics_summary(
                    f"Warmup {label} (tokens={tokens})",
                    warmup_metrics_list,
                )
                warmup_records.append(
                    {
                        "label": label,
                        "with_image": use_image,
                        "token_count": tokens,
                        "metrics": warmup_metrics_list,
                    }
                )

            if args.benchmark_runs > 0:
                benchmark_metrics_list: list[RunMetrics] = []
                for run_idx in range(args.benchmark_runs):
                    _, metrics = run_inference_with_metrics(
                        sampler,
                        args.prompt,
                        tokens,
                        image=image if use_image else None,
                    )
                    benchmark_metrics_list.append(metrics)
                    _log_run_metrics(
                        f"Benchmark {label} run {run_idx + 1}/{args.benchmark_runs}",
                        metrics,
                        extra={
                            "max_new_tokens": tokens,
                            "image_used": use_image,
                        },
                    )
                _log_metrics_summary(
                    f"Benchmark {label} (tokens={tokens})",
                    benchmark_metrics_list,
                )
                benchmark_records.append(
                    {
                        "label": label,
                        "with_image": use_image,
                        "token_count": tokens,
                        "metrics": benchmark_metrics_list,
                    }
                )

            jax.clear_caches()

    # --- Final Output ---
    logger.info("Generating final output...")
    final_image = None if args.final_text_only else image
    final_text, final_metrics = run_inference_with_metrics(
        sampler,
        args.prompt,
        args.max_new_tokens,
        image=final_image,
    )
    logger.info("--- Generated Text ---")
    logger.info(final_text)
    logger.info("--------------------")

    logger.info("--- Run Summary ---")
    logger.info("TPU type: %s", args.tpu_type)
    logger.info("Quantization: %s", args.quant_method)
    logger.info("Batch size: %s", args.batch_size)
    if warmup_records:
        logger.info("Warmup summaries:")
        for record in warmup_records:
            logger.info(
                "  %s | tokens=%s | image_used=%s -> %s",
                record["label"],
                record["token_count"],
                record["with_image"],
                _metrics_brief(record["metrics"]),
            )
    else:
        logger.info("Warmup summaries: no warmup runs executed.")

    if benchmark_records:
        logger.info("Benchmark summaries:")
        for result in benchmark_records:
            logger.info(
                "  %s | requested_tokens=%s | image_used=%s -> %s",
                result["label"],
                result["token_count"],
                result["with_image"],
                _metrics_brief(result["metrics"]),
            )
    else:
        logger.info("Benchmark summaries: no benchmark runs executed.")

    logger.info(
        "Final inference metrics -> %s | tokens=%s | image_used=%s",
        _metrics_brief([final_metrics]),
        final_metrics.total_tokens,
        not args.final_text_only,
    )
    logger.info(
        "Final throughput detail -> total_tps=%.2f | decode_tps=%.2f | seconds_per_token=%.4f | seconds_per_token_decode=%.4f | time_to_first_token=%.4fs",
        final_metrics.tokens_per_second_overall,
        final_metrics.tokens_per_second_post_first,
        final_metrics.seconds_per_token_overall,
        final_metrics.seconds_per_token_post_first,
        final_metrics.first_token_time,
    )
    logger.info("-------------------")

    report_entries: list[dict[str, Any]] = []
    for record in warmup_records:
        summary = _summarize_metrics(record["metrics"])
        if summary is None:
            continue
        report_entries.append(
            {
                "phase": "warmup",
                "label": record["label"],
                "with_image": record["with_image"],
                "token_count": record["token_count"],
                "runs": len(record["metrics"]),
                "metrics_summary": summary,
                "notes": f"image_used={record['with_image']}",
            }
        )

    for record in benchmark_records:
        summary = _summarize_metrics(record["metrics"])
        if summary is None:
            continue
        report_entries.append(
            {
                "phase": "benchmark",
                "label": record["label"],
                "with_image": record["with_image"],
                "token_count": record["token_count"],
                "runs": len(record["metrics"]),
                "metrics_summary": summary,
                "notes": f"image_used={record['with_image']}",
            }
        )

    final_summary = _summarize_metrics([final_metrics])
    if final_summary is not None:
        report_entries.append(
            {
                "phase": "final",
                "label": "Final Output",
                "with_image": not args.final_text_only,
                "token_count": args.max_new_tokens,
                "runs": 1,
                "metrics_summary": final_summary,
                "notes": f"image_used={not args.final_text_only}",
            }
        )

    run_metadata = {
        "script": Path(__file__).name,
        "run_type": args.run_type,
        "tpu_type": args.tpu_type,
        "quant_method": args.quant_method,
        "batch_size": args.batch_size,
        "benchmark_runs": args.benchmark_runs,
        "warmup_runs": args.warmup_runs,
    }

    metrics_store.save_records(report_entries, run_meta=run_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Inference with hfax")
    parser.add_argument(
        "--quant-method",
        type=str,
        default="NONE",
        choices=["NONE", "INT8", "INT4"],
        help="Quantization method to use.",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="URL of the image to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What can you say about this image: <start_of_image>",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs before benchmarking.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of benchmark runs.",
    )
    parser.add_argument(
        "--benchmark-token-counts",
        type=str,
        nargs="+",
        help=(
            "List of token count buckets to benchmark (space or comma separated)."
            " Defaults to --max-new-tokens when omitted."
        ),
    )
    parser.add_argument(
        "--tpu-type",
        type=str,
        default="none",
        choices=["none", "v4", "v5p", "v6e"],
        help="TPU hardware generation used for this run.",
    )
    parser.add_argument(
        "--benchmark-text-only",
        action="store_true",
        help="Benchmark text-only generations without providing an image.",
    )
    parser.add_argument(
        "--benchmark-with-image",
        action="store_true",
        help="Benchmark generations that include the provided image.",
    )
    parser.add_argument(
        "--final-text-only",
        action="store_true",
        help="Generate the final output without providing an image.",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["inference", "training"],
        default="inference",
        help="Record results under this workload type.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Effective batch size for this run (stored in benchmark reports).",
    )
    cli_args = parser.parse_args()
    main(cli_args)
