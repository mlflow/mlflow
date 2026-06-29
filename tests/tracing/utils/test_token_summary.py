"""Tests for token usage summarization utilities."""

import json
from unittest.mock import MagicMock

import pytest

from mlflow.tracing.constant import TokenUsageKey, TraceMetadataKey
from mlflow.tracing.utils.token_summary import (
    TokenUsageSummary,
    calculate_token_percentages,
    filter_traces_by_token_threshold,
    get_token_usage_from_trace,
    summarize_token_usage,
)


def _create_mock_trace(input_tokens: int, output_tokens: int, total_tokens: int):
    """Helper to create a mock trace with token usage."""
    trace = MagicMock()
    usage = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: total_tokens,
    }
    trace.info.request_metadata = {
        TraceMetadataKey.TOKEN_USAGE: json.dumps(usage),
    }
    return trace


class TestGetTokenUsageFromTrace:
    def test_extracts_token_usage(self):
        trace = _create_mock_trace(100, 50, 150)
        usage = get_token_usage_from_trace(trace)

        assert usage[TokenUsageKey.INPUT_TOKENS] == 100
        assert usage[TokenUsageKey.OUTPUT_TOKENS] == 50
        assert usage[TokenUsageKey.TOTAL_TOKENS] == 150

    def test_returns_none_when_no_usage(self):
        trace = MagicMock()
        trace.info.request_metadata = {}

        usage = get_token_usage_from_trace(trace)
        assert usage is None


class TestSummarizeTokenUsage:
    def test_summarizes_single_trace(self):
        traces = [_create_mock_trace(100, 50, 150)]
        summary = summarize_token_usage(traces)

        assert summary.total_input_tokens == 100
        assert summary.total_output_tokens == 50
        assert summary.total_tokens == 150
        assert summary.trace_count == 1

    def test_summarizes_multiple_traces(self):
        traces = [
            _create_mock_trace(100, 50, 150),
            _create_mock_trace(200, 100, 300),
            _create_mock_trace(150, 75, 225),
        ]
        summary = summarize_token_usage(traces)

        assert summary.total_input_tokens == 450
        assert summary.total_output_tokens == 225
        assert summary.total_tokens == 675
        assert summary.trace_count == 3


class TestFilterTracesByTokenThreshold:
    def test_filters_by_min_tokens(self):
        traces = [
            _create_mock_trace(100, 50, 150),
            _create_mock_trace(200, 100, 300),
            _create_mock_trace(50, 25, 75),
        ]
        filtered = filter_traces_by_token_threshold(traces, min_tokens=100)

        assert len(filtered) == 2

    def test_filters_by_max_tokens(self):
        traces = [
            _create_mock_trace(100, 50, 150),
            _create_mock_trace(200, 100, 300),
            _create_mock_trace(50, 25, 75),
        ]
        filtered = filter_traces_by_token_threshold(traces, max_tokens=200)

        assert len(filtered) == 2


class TestCalculateTokenPercentages:
    def test_calculates_percentages(self):
        summary = TokenUsageSummary(
            total_input_tokens=750,
            total_output_tokens=250,
            total_tokens=1000,
            avg_input_tokens=250.0,
            avg_output_tokens=83.33,
            avg_total_tokens=333.33,
            trace_count=3,
        )
        percentages = calculate_token_percentages(summary)

        assert percentages["input_percentage"] == 75.0
        assert percentages["output_percentage"] == 25.0


class TestTokenUsageSummary:
    def test_to_dict(self):
        summary = TokenUsageSummary(
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            avg_input_tokens=100.0,
            avg_output_tokens=50.0,
            avg_total_tokens=150.0,
            trace_count=1,
        )
        result = summary.to_dict()

        assert result["total_input_tokens"] == 100
        assert result["total_output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["trace_count"] == 1
