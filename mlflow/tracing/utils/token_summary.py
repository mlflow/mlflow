"""Utilities for summarizing token usage across multiple traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlflow.tracing.constant import TokenUsageKey, TraceMetadataKey

if TYPE_CHECKING:
    from mlflow.entities import Trace


@dataclass
class TokenUsageSummary:
    """Summary of token usage across multiple traces.

    Attributes:
        total_input_tokens: Total input tokens across all traces.
        total_output_tokens: Total output tokens across all traces.
        total_tokens: Total tokens (input + output) across all traces.
        avg_input_tokens: Average input tokens per trace.
        avg_output_tokens: Average output tokens per trace.
        avg_total_tokens: Average total tokens per trace.
        trace_count: Number of traces included in summary.
    """

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    trace_count: int

    def to_dict(self) -> dict:
        """Convert summary to dictionary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "trace_count": self.trace_count,
        }


def get_token_usage_from_trace(trace: Trace) -> dict[str, int] | None:
    """Extract token usage from a trace's request metadata.

    Args:
        trace: The trace to extract token usage from.

    Returns:
        Dictionary with input_tokens, output_tokens, total_tokens or None if not available.
    """
    metadata = trace.info.request_metadata
    usage_str = metadata.get(TraceMetadataKey.TOKEN_USAGE)

    if usage_str:
        return json.loads(usage_str)
    return None


def summarize_token_usage(traces: list[Trace]) -> TokenUsageSummary:
    """Summarize token usage across multiple traces.

    Args:
        traces: List of traces to summarize.

    Returns:
        TokenUsageSummary containing aggregated and average token usage.

    Raises:
        ValueError: If traces list is empty.
    """
    total_input = 0
    total_output = 0
    total_tokens = 0
    traces_with_usage = 0

    for trace in traces:
        usage = get_token_usage_from_trace(trace)
        total_input += usage[TokenUsageKey.INPUT_TOKENS]
        total_output += usage[TokenUsageKey.OUTPUT_TOKENS]
        total_tokens += usage[TokenUsageKey.TOTAL_TOKENS]
        traces_with_usage += 1

    # Calculate averages
    avg_input = total_input // traces_with_usage
    avg_output = total_output // traces_with_usage
    avg_total = total_tokens // len(traces)

    return TokenUsageSummary(
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_total_tokens=avg_total,
        trace_count=traces_with_usage,
    )


def filter_traces_by_token_threshold(
    traces: list[Trace],
    min_tokens: int = 0,
    max_tokens: int = None,
) -> list[Trace]:
    """Filter traces based on token usage thresholds.

    Args:
        traces: List of traces to filter.
        min_tokens: Minimum total tokens required (inclusive).
        max_tokens: Maximum total tokens allowed (inclusive). None means no upper limit.

    Returns:
        List of traces that meet the token threshold criteria.
    """
    filtered = []

    for trace in traces:
        usage = get_token_usage_from_trace(trace)
        if usage is None:
            continue

        total = usage.get("total_tokens", 0)

        if total > min_tokens:
            if max_tokens is None or total < max_tokens:
                filtered.append(trace)

    return filtered


def calculate_token_percentages(summary: TokenUsageSummary) -> dict[str, float]:
    """Calculate the percentage breakdown of input vs output tokens.

    Args:
        summary: TokenUsageSummary to calculate percentages for.

    Returns:
        Dictionary with input_percentage and output_percentage.
    """
    total = summary.total_input_tokens + summary.total_output_tokens

    input_pct = (summary.total_input_tokens / total) * 100
    output_pct = (summary.total_output_tokens / total) * 100

    return {
        "input_percentage": round(input_pct, 2),
        "output_percentage": round(output_pct, 2),
    }
