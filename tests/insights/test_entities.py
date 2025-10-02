from datetime import datetime
from pathlib import Path

import pytest

from mlflow.insights.models.entities import (
    Census,
    ErrorSpan,
    OperationalMetrics,
    QualityMetric,
    QualityMetrics,
    SlowTool,
    TimeBucket,
)


@pytest.fixture
def operational_metrics():
    """Create sample operational metrics."""

    return OperationalMetrics(
        total_traces=1000,
        ok_count=950,
        error_count=50,
        error_rate=5.0,
        first_trace_timestamp="2025-09-17T17:42:06.078000",
        last_trace_timestamp="2025-09-18T15:54:01.653000",
        max_latency_ms=20000.0,
        p50_latency_ms=7500.0,
        p90_latency_ms=10000.0,
        p95_latency_ms=11000.0,
        p99_latency_ms=12500.0,
        time_buckets=[
            TimeBucket(
                time_bucket="2025-09-17T17:00:00",
                total_traces=100,
                ok_count=95,
                error_count=5,
                error_rate=5.0,
                p95_latency_ms=10500.0,
            ),
            TimeBucket(
                time_bucket="2025-09-17T18:00:00",
                total_traces=200,
                ok_count=190,
                error_count=10,
                error_rate=5.0,
                p95_latency_ms=11000.0,
            ),
        ],
        top_error_spans=[
            ErrorSpan(
                error_span_name="database_query",
                count=25,
                pct_of_errors=50.0,
                sample_trace_ids=["tr-001", "tr-002"],
            ),
            ErrorSpan(
                error_span_name="api_call",
                count=15,
                pct_of_errors=30.0,
                sample_trace_ids=["tr-003", "tr-004"],
            ),
        ],
        top_slow_tools=[
            SlowTool(
                tool_span_name="process_data",
                count=500,
                median_latency_ms=8000.0,
                p95_latency_ms=11000.0,
                sample_trace_ids=["tr-005", "tr-006"],
            ),
            SlowTool(
                tool_span_name="generate_report",
                count=300,
                median_latency_ms=3000.0,
                p95_latency_ms=4000.0,
                sample_trace_ids=["tr-007", "tr-008"],
            ),
        ],
    )


@pytest.fixture
def quality_metrics():
    """Create sample quality metrics."""

    return QualityMetrics(
        minimal_responses=QualityMetric(
            value=2.5,
            description="Percentage of responses shorter than 50 characters",
            sample_trace_ids=["tr-009", "tr-010"],
        ),
        response_quality_issues=QualityMetric(
            value=8.3,
            description="Percentage of responses with quality problems",
            sample_trace_ids=["tr-011", "tr-012"],
        ),
        rushed_processing=QualityMetric(
            value=12.1,
            description="Percentage of complex requests processed too quickly",
            sample_trace_ids=["tr-013", "tr-014"],
        ),
        verbosity=QualityMetric(
            value=6.7,
            description="Percentage of overly verbose responses",
            sample_trace_ids=["tr-015", "tr-016"],
        ),
    )


@pytest.fixture
def census(operational_metrics, quality_metrics):
    """Create a census instance."""

    return Census.create_with_timestamp(
        table_name="test_table",
        operational_metrics=operational_metrics,
        quality_metrics=quality_metrics,
        additional_metadata={"environment": "test", "version": "1.0"},
    )


def test_census_creation(census: Census):
    assert census.metadata.table_name == "test_table"
    assert census.metadata.additional_metadata == {
        "environment": "test",
        "version": "1.0",
    }
    assert census.operational_metrics.total_traces == 1000
    assert census.operational_metrics.error_rate == 5.0
    assert census.quality_metrics.minimal_responses.value == 2.5


def test_time_bucket_model():
    bucket = TimeBucket(
        time_bucket="2025-09-17T17:00:00",
        total_traces=100,
        ok_count=95,
        error_count=5,
        error_rate=5.0,
        p95_latency_ms=10500.0,
    )

    assert bucket.time_bucket == datetime.fromisoformat("2025-09-17T17:00:00")
    assert bucket.total_traces == 100
    assert bucket.error_rate == 5.0


def test_error_span_model():
    span = ErrorSpan(
        error_span_name="database_query",
        count=25,
        pct_of_errors=50.0,
        sample_trace_ids=["tr-001", "tr-002"],
    )

    assert span.error_span_name == "database_query"
    assert span.count == 25
    assert span.pct_of_errors == 50.0
    assert len(span.sample_trace_ids) == 2


def test_slow_tool_model():
    tool = SlowTool(
        tool_span_name="process_data",
        count=500,
        median_latency_ms=8000.0,
        p95_latency_ms=11000.0,
        sample_trace_ids=["tr-005", "tr-006"],
    )

    assert tool.tool_span_name == "process_data"
    assert tool.count == 500
    assert tool.median_latency_ms == 8000.0
    assert tool.p95_latency_ms == 11000.0


def test_quality_metric_model():
    metric = QualityMetric(
        value=2.5,
        description="Test metric",
        sample_trace_ids=["tr-001", "tr-002"],
    )

    assert metric.value == 2.5
    assert metric.description == "Test metric"
    assert len(metric.sample_trace_ids) == 2


def test_census_yaml_serialization(census: Census, tmp_path: Path):
    yaml_file = tmp_path / "census.yaml"

    yaml_content = census.to_yaml()
    yaml_file.write_text(yaml_content)

    loaded_census = Census.from_yaml(yaml_content)

    assert loaded_census.metadata.table_name == census.metadata.table_name
    assert loaded_census.operational_metrics.total_traces == census.operational_metrics.total_traces
    assert loaded_census.operational_metrics.error_rate == census.operational_metrics.error_rate
    assert (
        loaded_census.quality_metrics.minimal_responses.value
        == census.quality_metrics.minimal_responses.value
    )

    assert len(loaded_census.operational_metrics.time_buckets) == 2
    assert loaded_census.operational_metrics.time_buckets[0].time_bucket == datetime.fromisoformat(
        "2025-09-17T17:00:00"
    )

    assert len(loaded_census.operational_metrics.top_error_spans) == 2
    assert loaded_census.operational_metrics.top_error_spans[0].error_span_name == "database_query"


def test_census_empty_lists():
    metrics = OperationalMetrics(
        total_traces=100,
        ok_count=100,
        error_count=0,
        error_rate=0.0,
        first_trace_timestamp="2025-09-17T17:00:00",
        last_trace_timestamp="2025-09-17T18:00:00",
        max_latency_ms=1000.0,
        p50_latency_ms=500.0,
        p90_latency_ms=800.0,
        p95_latency_ms=900.0,
        p99_latency_ms=950.0,
        time_buckets=[],
        top_error_spans=[],
        top_slow_tools=[],
    )

    quality = QualityMetrics(
        minimal_responses=QualityMetric(
            value=0.0,
            description="No issues",
            sample_trace_ids=[],
        ),
        response_quality_issues=QualityMetric(
            value=0.0,
            description="No issues",
            sample_trace_ids=[],
        ),
        rushed_processing=QualityMetric(
            value=0.0,
            description="No issues",
            sample_trace_ids=[],
        ),
        verbosity=QualityMetric(
            value=0.0,
            description="No issues",
            sample_trace_ids=[],
        ),
    )

    census = Census.create_with_timestamp(
        table_name="test_table",
        operational_metrics=metrics,
        quality_metrics=quality,
    )

    # Census created successfully with empty lists
    assert len(census.operational_metrics.time_buckets) == 0
    assert len(census.operational_metrics.top_error_spans) == 0
    assert len(census.operational_metrics.top_slow_tools) == 0
