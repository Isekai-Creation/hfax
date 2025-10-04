from dataclasses import dataclass
import math
import statistics
from typing import Iterable

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
