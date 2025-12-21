import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import pytest
from opentelemetry import trace as trace_api

from mlflow.entities import (
    Assessment,
    AssessmentSource,
    AssessmentSourceType,
    Expectation,
    Feedback,
    trace_location,
)
from mlflow.entities.assessment import AssessmentError
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import POSTGRES
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import (
    AssessmentMetricDimensionKey,
    AssessmentMetricKey,
    SpanMetricDimensionKey,
    SpanMetricKey,
    TraceMetadataKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
    TraceTagKey,
)
from mlflow.utils.time import get_current_time_millis

from tests.store.tracking.test_sqlalchemy_store import create_test_span

pytestmark = pytest.mark.notrackingurimock


def _get_expected_percentile_value(
    store: SqlAlchemyStore,
    percentile_value: float,
    min_val: float,
    max_val: float,
    values: list[float],
) -> float:
    """
    Calculate expected percentile value based on database type.

    PostgreSQL uses linear interpolation (PERCENTILE_CONT).
    MySQL, SQLite, and MSSQL use min + (percentile_value / 100) * (max - min) approximation.
    """
    db_type = store._get_dialect()
    # Convert percentile_value from 0-100 to 0-1 ratio
    percentile_ratio = percentile_value / 100.0

    if db_type == POSTGRES:
        # Linear interpolation for PERCENTILE_CONT
        # For a sorted list, percentile index = percentile_ratio * (n - 1)
        sorted_values = sorted(values)
        n = len(sorted_values)
        index = percentile_ratio * (n - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, n - 1)
        fraction = index - lower_idx
        return sorted_values[lower_idx] + fraction * (
            sorted_values[upper_idx] - sorted_values[lower_idx]
        )
    else:
        # Approximation for MySQL, SQLite, and MSSQL
        return min_val + percentile_ratio * (max_val - min_val)


def test_query_trace_metrics_count_no_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_count_no_dimensions")

    for i in range(5):
        trace_info = TraceInfo(
            trace_id=f"trace{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100 + i * 10,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 5},
    }


def test_query_trace_metrics_count_by_status(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_count_by_status")

    traces_data = [
        ("trace1", TraceStatus.OK),
        ("trace2", TraceStatus.OK),
        ("trace3", TraceStatus.OK),
        ("trace4", TraceStatus.ERROR),
        ("trace5", TraceStatus.ERROR),
    ]

    for trace_id, status in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=status,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[TraceMetricDimensionKey.TRACE_STATUS],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {TraceMetricDimensionKey.TRACE_STATUS: "ERROR"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {TraceMetricDimensionKey.TRACE_STATUS: "OK"},
        "values": {"COUNT": 3},
    }


def test_query_trace_metrics_count_by_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_count_by_name")

    traces_data = [
        ("trace1", "workflow_a"),
        ("trace2", "workflow_a"),
        ("trace3", "workflow_a"),
        ("trace4", "workflow_b"),
        ("trace5", "workflow_b"),
    ]

    for trace_id, name in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: name},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_b"},
        "values": {"COUNT": 2},
    }


def test_query_trace_metrics_count_by_multiple_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_count_by_multiple_dimensions")

    traces_data = [
        ("trace1", TraceStatus.OK, "workflow_a"),
        ("trace2", TraceStatus.OK, "workflow_a"),
        ("trace3", TraceStatus.ERROR, "workflow_a"),
        ("trace4", TraceStatus.OK, "workflow_b"),
        ("trace5", TraceStatus.ERROR, "workflow_b"),
    ]

    for trace_id, status, name in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=status,
            tags={TraceTagKey.TRACE_NAME: name},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[TraceMetricDimensionKey.TRACE_STATUS, TraceMetricDimensionKey.TRACE_NAME],
    )

    assert len(result) == 4
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            TraceMetricDimensionKey.TRACE_STATUS: "ERROR",
            TraceMetricDimensionKey.TRACE_NAME: "workflow_a",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            TraceMetricDimensionKey.TRACE_STATUS: "ERROR",
            TraceMetricDimensionKey.TRACE_NAME: "workflow_b",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            TraceMetricDimensionKey.TRACE_STATUS: "OK",
            TraceMetricDimensionKey.TRACE_NAME: "workflow_a",
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[3]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            TraceMetricDimensionKey.TRACE_STATUS: "OK",
            TraceMetricDimensionKey.TRACE_NAME: "workflow_b",
        },
        "values": {"COUNT": 1},
    }


def test_query_trace_metrics_latency_avg(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_latency_avg")

    traces_data = [
        ("trace1", "workflow_a", 100),
        ("trace2", "workflow_a", 200),
        ("trace3", "workflow_a", 300),
        ("trace4", "workflow_b", 150),
        ("trace5", "workflow_b", 250),
    ]

    for trace_id, name, duration in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=duration,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: name},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.LATENCY,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.LATENCY,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": {"AVG": 200.0},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.LATENCY,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_b"},
        "values": {"AVG": 200.0},
    }


@pytest.mark.parametrize(
    "percentile_value",
    [50.0, 75.0, 90.0, 95.0, 99.0],
)
def test_query_trace_metrics_latency_percentiles(
    store: SqlAlchemyStore,
    percentile_value: float,
):
    exp_id = store.create_experiment(f"test_latency_percentile_{percentile_value}")

    traces_data = [
        ("trace1", "workflow_a", 100),
        ("trace2", "workflow_a", 200),
        ("trace3", "workflow_a", 300),
        ("trace4", "workflow_b", 100),
        ("trace5", "workflow_b", 200),
    ]

    for trace_id, name, duration in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=duration,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: name},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(
                aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
            )
        ],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    # Calculate expected values based on database type
    expected_workflow_a = _get_expected_percentile_value(
        store, percentile_value, 100.0, 300.0, [100.0, 200.0, 300.0]
    )
    expected_workflow_b = _get_expected_percentile_value(
        store, percentile_value, 100.0, 200.0, [100.0, 200.0]
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.LATENCY,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": {f"P{percentile_value}": expected_workflow_a},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.LATENCY,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_b"},
        "values": {f"P{percentile_value}": expected_workflow_b},
    }


def test_query_trace_metrics_latency_multiple_aggregations(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_latency_multiple_aggregations")

    traces_data = [
        ("trace1", "workflow_a", 100),
        ("trace2", "workflow_a", 200),
        ("trace3", "workflow_a", 300),
        ("trace4", "workflow_a", 400),
        ("trace5", "workflow_a", 500),
    ]

    for trace_id, name, duration in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=duration,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: name},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.AVG),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=95),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=99),
        ],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    # Calculate expected percentile value based on database type
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    expected_p95 = _get_expected_percentile_value(store, 95.0, 100.0, 500.0, values)
    expected_p99 = _get_expected_percentile_value(store, 99.0, 100.0, 500.0, values)

    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.LATENCY,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": {"AVG": 300.0, "P95": expected_p95, "P99": expected_p99},
    }


def test_query_trace_metrics_with_time_interval(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_time_interval")

    base_time = 1577836800000  # 2020-01-01 00:00:00 UTC in milliseconds
    hour_ms = 60 * 60 * 1000

    traces_data = [
        ("trace1", base_time, 100),
        ("trace2", base_time + 10 * 60 * 1000, 200),
        ("trace3", base_time + hour_ms, 150),
        ("trace4", base_time + hour_ms + 30 * 60 * 1000, 250),
        ("trace5", base_time + 2 * hour_ms, 300),
    ]

    for trace_id, timestamp, duration in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp,
            execution_duration=duration,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time,
        end_time_ms=base_time + 3 * hour_ms,
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(base_time / 1000, tz=timezone.utc).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time + hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time + 2 * hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 1},
    }


def test_query_trace_metrics_with_time_interval_and_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_time_interval_and_dimensions")

    base_time = 1577836800000  # 2020-01-01 00:00:00 UTC in milliseconds
    hour_ms = 60 * 60 * 1000

    traces_data = [
        ("trace1", base_time, TraceStatus.OK, 100),
        ("trace2", base_time + 10 * 60 * 1000, TraceStatus.ERROR, 200),
        ("trace3", base_time + hour_ms, TraceStatus.OK, 150),
        ("trace4", base_time + hour_ms, TraceStatus.ERROR, 250),
    ]

    for trace_id, timestamp, status, duration in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp,
            execution_duration=duration,
            state=status,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[TraceMetricDimensionKey.TRACE_STATUS],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time,
        end_time_ms=base_time + 2 * hour_ms,
    )

    assert len(result) == 4
    time_bucket_1 = datetime.fromtimestamp(base_time / 1000, tz=timezone.utc).isoformat()
    time_bucket_2 = datetime.fromtimestamp(
        (base_time + hour_ms) / 1000, tz=timezone.utc
    ).isoformat()
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_1,
            TraceMetricDimensionKey.TRACE_STATUS: "ERROR",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_1,
            TraceMetricDimensionKey.TRACE_STATUS: "OK",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_2,
            TraceMetricDimensionKey.TRACE_STATUS: "ERROR",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[3]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_2,
            TraceMetricDimensionKey.TRACE_STATUS: "OK",
        },
        "values": {"COUNT": 1},
    }


def test_query_trace_metrics_with_status_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_status_filter")

    traces_data = [
        ("trace1", TraceStatus.OK),
        ("trace2", TraceStatus.OK),
        ("trace3", TraceStatus.OK),
        ("trace4", TraceStatus.ERROR),
        ("trace5", TraceStatus.ERROR),
    ]

    for trace_id, status in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=status,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'OK'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 3},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'ERROR'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


def test_query_trace_metrics_with_source_run_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_source_run_filter")

    traces_data = [
        ("trace1", "run_123"),
        ("trace2", "run_123"),
        ("trace3", "run_456"),
        ("trace4", "run_456"),
        ("trace5", None),  # No source run
    ]

    for trace_id, source_run in traces_data:
        metadata = {TraceMetadataKey.SOURCE_RUN: source_run} if source_run else {}
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
            trace_metadata=metadata,
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run_123'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run_456'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


def test_query_trace_metrics_with_multiple_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_multiple_filters")

    traces_data = [
        ("trace1", TraceStatus.OK, "run_123"),
        ("trace2", TraceStatus.OK, "run_123"),
        ("trace3", TraceStatus.ERROR, "run_123"),
        ("trace4", TraceStatus.OK, "run_456"),
        ("trace5", TraceStatus.ERROR, "run_456"),
    ]

    for trace_id, status, source_run in traces_data:
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=status,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
            trace_metadata={TraceMetadataKey.SOURCE_RUN: source_run},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[
            "trace.status = 'OK'",
            f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run_123'",
        ],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


def test_query_trace_metrics_with_tag_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_tag_filter")

    traces_data = [
        ("trace1", "model_v1"),
        ("trace2", "model_v1"),
        ("trace3", "model_v2"),
        ("trace4", "model_v2"),
        ("trace5", None),  # No model tag
    ]

    for trace_id, model_version in traces_data:
        tags = {TraceTagKey.TRACE_NAME: "test_trace", "tag1": "value1"}
        if model_version:
            tags["model.version"] = model_version

        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags=tags,
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.tag.tag1 = 'value1'"],
    )
    assert len(result) == 1
    assert asdict(result[0])["values"] == {"COUNT": 5}

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.tag.`model.version` = 'model_v1'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.tag.`model.version` = 'model_v2'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


@pytest.mark.parametrize(
    ("filter_string", "error_match"),
    [
        ("status = 'OK'", r"Invalid identifier 'status'"),
        ("trace.status != 'OK'", r"Invalid comparator: '!=', only '=' operator is supported"),
        ("trace.unsupported_field = 'value'", r"Invalid entity 'unsupported_field' specified"),
        ("span.status = 'OK'", r"Filtering by span is only supported for SPANS view type"),
        (
            "assessment.type = 'feedback'",
            r"Filtering by assessment is only supported for ASSESSMENTS view type",
        ),
    ],
)
def test_query_trace_metrics_with_invalid_filter(
    store: SqlAlchemyStore, filter_string: str, error_match: str
):
    exp_id = store.create_experiment("test_with_invalid_filter")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    with pytest.raises(MlflowException, match=error_match):
        store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
            filters=[filter_string],
        )


@pytest.fixture
def traces_with_token_usage_setup(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_traces_with_token_usage")
    traces_data = [
        ("trace1", "workflow_a", 100, 50, 150),
        ("trace2", "workflow_a", 200, 100, 300),
        ("trace3", "workflow_a", 150, 75, 225),
        ("trace4", "workflow_b", 300, 150, 450),
        ("trace5", "workflow_b", 250, 125, 375),
    ]

    for trace_id, name, input_tokens, output_tokens, total_tokens in traces_data:
        token_usage = {
            TraceMetricKey.INPUT_TOKENS: input_tokens,
            TraceMetricKey.OUTPUT_TOKENS: output_tokens,
            TraceMetricKey.TOTAL_TOKENS: total_tokens,
        }
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: name},
            trace_metadata={TraceMetadataKey.TOKEN_USAGE: json.dumps(token_usage)},
        )
        store.start_trace(trace_info)
    return exp_id, store


def test_query_trace_metrics_total_tokens_sum(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.SUM)],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": {"SUM": 675},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_b"},
        "values": {"SUM": 825},
    }


def test_query_trace_metrics_total_tokens_avg(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": {"AVG": 225.0},
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_b"},
        "values": {"AVG": 412.5},
    }


def test_query_trace_metrics_total_tokens_percentiles(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup
    percentiles = [50, 75, 90, 95, 99]

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=p)
            for p in percentiles
        ],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
    )

    # Calculate expected values based on database type
    workflow_a_values = [150, 225, 300]
    workflow_b_values = [375, 450]

    expected_workflow_a_values = {
        f"P{p}": _get_expected_percentile_value(
            store, p, min(workflow_a_values), max(workflow_a_values), workflow_a_values
        )
        for p in percentiles
    }
    expected_workflow_b_values = {
        f"P{p}": _get_expected_percentile_value(
            store, p, min(workflow_b_values), max(workflow_b_values), workflow_b_values
        )
        for p in percentiles
    }

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_a"},
        "values": expected_workflow_a_values,
    }
    assert asdict(result[1]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {TraceMetricDimensionKey.TRACE_NAME: "workflow_b"},
        "values": expected_workflow_b_values,
    }


def test_query_trace_metrics_total_tokens_no_dimensions(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.SUM),
            MetricAggregation(aggregation_type=AggregationType.AVG),
        ],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {},
        "values": {"SUM": 1500, "AVG": 300.0},
    }


def test_query_trace_metrics_total_tokens_without_token_usage(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_total_tokens_without_token_usage")

    for i in range(3):
        trace_info = TraceInfo(
            trace_id=f"trace{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: "test_trace"},
        )
        store.start_trace(trace_info)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.SUM)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": TraceMetricKey.TOTAL_TOKENS,
        "dimensions": {},
        "values": {"SUM": None},
    }


def test_query_span_metrics_count_no_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_count_no_dimensions")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    spans = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=1000000000),
        create_test_span("trace1", "span2", span_id=2, span_type="CHAIN", start_ns=1100000000),
        create_test_span("trace1", "span3", span_id=3, span_type="LLM", start_ns=1200000000),
        create_test_span("trace1", "span4", span_id=4, span_type="TOOL", start_ns=1300000000),
        create_test_span("trace1", "span5", span_id=5, span_type="LLM", start_ns=1400000000),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {},
        "values": {"COUNT": 5},
    }


def test_query_span_metrics_count_by_span_type(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_count_by_span_type")

    # Create a trace first
    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different types
    spans = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=1000000000),
        create_test_span("trace1", "span2", span_id=2, span_type="LLM", start_ns=1100000000),
        create_test_span("trace1", "span3", span_id=3, span_type="LLM", start_ns=1200000000),
        create_test_span("trace1", "span4", span_id=4, span_type="CHAIN", start_ns=1300000000),
        create_test_span("trace1", "span5", span_id=5, span_type="CHAIN", start_ns=1400000000),
        create_test_span("trace1", "span6", span_id=6, span_type="TOOL", start_ns=1500000000),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_TYPE],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "CHAIN"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "LLM"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[2]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "TOOL"},
        "values": {"COUNT": 1},
    }


def test_query_span_metrics_with_time_interval(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_with_time_interval")

    # Base time in nanoseconds (2020-01-01 00:00:00 UTC)
    base_time_ns = 1577836800000000000
    hour_ns = 60 * 60 * 1_000_000_000

    # Create a trace
    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=base_time_ns // 1_000_000,  # Convert to milliseconds
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans at different times
    spans = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=base_time_ns),
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="LLM",
            start_ns=base_time_ns + 10 * 60 * 1_000_000_000,
        ),
        create_test_span(
            "trace1", "span3", span_id=3, span_type="LLM", start_ns=base_time_ns + hour_ns
        ),
        create_test_span(
            "trace1",
            "span4",
            span_id=4,
            span_type="LLM",
            start_ns=base_time_ns + hour_ns + 30 * 60 * 1_000_000_000,
        ),
        create_test_span(
            "trace1", "span5", span_id=5, span_type="LLM", start_ns=base_time_ns + 2 * hour_ns
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time_ns // 1_000_000,
        end_time_ms=(base_time_ns + 3 * hour_ns) // 1_000_000,
    )

    base_time_ms = base_time_ns // 1_000_000
    hour_ms = 60 * 60 * 1000

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(base_time_ms / 1000, tz=timezone.utc).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + 2 * hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 1},
    }


def test_query_span_metrics_with_time_interval_and_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_with_time_interval_and_dimensions")

    # Base time in nanoseconds (2020-01-01 00:00:00 UTC)
    base_time_ns = 1577836800000000000
    hour_ns = 60 * 60 * 1_000_000_000

    # Create a trace
    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=base_time_ns // 1_000_000,
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans at different times with different types
    spans = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=base_time_ns),
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="CHAIN",
            start_ns=base_time_ns + 10 * 60 * 1_000_000_000,
        ),
        create_test_span(
            "trace1", "span3", span_id=3, span_type="LLM", start_ns=base_time_ns + hour_ns
        ),
        create_test_span(
            "trace1", "span4", span_id=4, span_type="CHAIN", start_ns=base_time_ns + hour_ns
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_TYPE],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time_ns // 1_000_000,
        end_time_ms=(base_time_ns + 2 * hour_ns) // 1_000_000,
    )

    base_time_ms = base_time_ns // 1_000_000
    hour_ms = 60 * 60 * 1000
    time_bucket_1 = datetime.fromtimestamp(base_time_ms / 1000, tz=timezone.utc).isoformat()
    time_bucket_2 = datetime.fromtimestamp(
        (base_time_ms + hour_ms) / 1000, tz=timezone.utc
    ).isoformat()

    assert len(result) == 4
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_1,
            SpanMetricDimensionKey.SPAN_TYPE: "CHAIN",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_1,
            SpanMetricDimensionKey.SPAN_TYPE: "LLM",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_2,
            SpanMetricDimensionKey.SPAN_TYPE: "CHAIN",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[3]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_2,
            SpanMetricDimensionKey.SPAN_TYPE: "LLM",
        },
        "values": {"COUNT": 1},
    }


def test_query_span_metrics_with_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_with_filters")

    # Create traces with different statuses
    trace1 = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace1)

    trace2 = TraceInfo(
        trace_id="trace2",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.ERROR,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace2)

    # Create spans for trace1 (OK status)
    spans_trace1 = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=1000000000),
        create_test_span("trace1", "span2", span_id=2, span_type="LLM", start_ns=1100000000),
        create_test_span("trace1", "span3", span_id=3, span_type="CHAIN", start_ns=1200000000),
    ]
    store.log_spans(exp_id, spans_trace1)

    # Create spans for trace2 (ERROR status)
    spans_trace2 = [
        create_test_span("trace2", "span4", span_id=4, span_type="LLM", start_ns=1300000000),
        create_test_span("trace2", "span5", span_id=5, span_type="CHAIN", start_ns=1400000000),
    ]
    store.log_spans(exp_id, spans_trace2)

    # Query spans only for traces with OK status
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'OK'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {},
        "values": {"COUNT": 3},
    }

    # Query spans grouped by type for traces with ERROR status
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_TYPE],
        filters=["trace.status = 'ERROR'"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "CHAIN"},
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "LLM"},
        "values": {"COUNT": 1},
    }


def test_query_span_metrics_across_multiple_traces(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_across_multiple_traces")

    # Create multiple traces
    for i in range(3):
        trace_info = TraceInfo(
            trace_id=f"trace{i}",
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: f"workflow_{i}"},
        )
        store.start_trace(trace_info)

        # Create spans for each trace
        spans = [
            create_test_span(
                f"trace{i}",
                f"span{i}_1",
                span_id=i * 10 + 1,
                span_type="LLM",
                start_ns=1000000000 + i * 100000000,
            ),
            create_test_span(
                f"trace{i}",
                f"span{i}_2",
                span_id=i * 10 + 2,
                span_type="CHAIN",
                start_ns=1100000000 + i * 100000000,
            ),
        ]
        store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_TYPE],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "CHAIN"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "LLM"},
        "values": {"COUNT": 3},
    }


def test_query_span_metrics_with_span_status_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_with_status_filter")

    trace1 = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace1)

    trace2 = TraceInfo(
        trace_id="trace2",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.ERROR,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace2)

    spans_ok = [
        create_test_span(
            "trace1",
            "span1",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            status=trace_api.StatusCode.OK,
        ),
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="LLM",
            start_ns=1100000000,
            status=trace_api.StatusCode.OK,
        ),
        create_test_span(
            "trace1",
            "span3",
            span_id=3,
            span_type="CHAIN",
            start_ns=1200000000,
            status=trace_api.StatusCode.OK,
        ),
        create_test_span(
            "trace1",
            "span3",
            span_id=3,
            span_type="CHAIN",
            start_ns=1200000000,
            status=trace_api.StatusCode.OK,
        ),
    ]
    store.log_spans(exp_id, spans_ok)

    spans_error = [
        create_test_span(
            "trace2",
            "span4",
            span_id=4,
            span_type="LLM",
            start_ns=1300000000,
            status=trace_api.StatusCode.ERROR,
        ),
        create_test_span(
            "trace2",
            "span5",
            span_id=5,
            span_type="CHAIN",
            start_ns=1400000000,
            status=trace_api.StatusCode.ERROR,
        ),
    ]
    store.log_spans(exp_id, spans_error)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_TYPE],
        filters=["span.status = 'OK'"],
    )

    # Should only count spans from trace1 (OK status)
    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "CHAIN"},
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_TYPE: "LLM"},
        "values": {"COUNT": 2},
    }


def test_query_span_metrics_with_span_name_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_with_name_filter")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different names
    spans = [
        create_test_span(
            "trace1", "generate_response", span_id=1, span_type="LLM", start_ns=1000000000
        ),
        create_test_span(
            "trace1", "generate_response", span_id=2, span_type="LLM", start_ns=1100000000
        ),
        create_test_span(
            "trace1", "process_input", span_id=3, span_type="CHAIN", start_ns=1200000000
        ),
        create_test_span(
            "trace1", "validate_output", span_id=4, span_type="TOOL", start_ns=1300000000
        ),
    ]
    store.log_spans(exp_id, spans)

    # Query spans with span.name filter
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["span.name = 'generate_response'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


def test_query_span_metrics_with_span_type_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_with_type_filter")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    spans = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=1000000000),
        create_test_span("trace1", "span2", span_id=2, span_type="TOOL", start_ns=1100000000),
        create_test_span("trace1", "span3", span_id=3, span_type="CHAIN", start_ns=1200000000),
        create_test_span("trace1", "span4", span_id=4, span_type="TOOL", start_ns=1300000000),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["span.type = 'TOOL'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


@pytest.mark.parametrize(
    ("filter_string", "error_match"),
    [
        ("status = 'OK'", r"Invalid identifier 'status'"),
        ("span.status != 'OK'", r"Invalid comparator: '!=', only '=' operator is supported"),
        ("span.invalid_field = 'value'", r"Invalid entity 'invalid_field' specified"),
        ("span.status.extra = 'value'", r"does not require a key"),
        ("span.name LIKE 'test%'", r"only '=' operator is supported"),
    ],
)
def test_query_span_metrics_invalid_filters(
    store: SqlAlchemyStore, filter_string: str, error_match: str
):
    exp_id = store.create_experiment("test_span_invalid_filters")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    spans = [
        create_test_span("trace1", "span1", span_id=1, span_type="LLM", start_ns=1000000000),
    ]
    store.log_spans(exp_id, spans)

    with pytest.raises(MlflowException, match=error_match):
        store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
            filters=[filter_string],
        )


def test_query_span_metrics_count_by_span_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_count_by_span_name")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different names
    spans = [
        create_test_span(
            "trace1", "generate_response", span_id=1, span_type="LLM", start_ns=1000000000
        ),
        create_test_span(
            "trace1", "generate_response", span_id=2, span_type="LLM", start_ns=1100000000
        ),
        create_test_span(
            "trace1", "generate_response", span_id=3, span_type="LLM", start_ns=1200000000
        ),
        create_test_span(
            "trace1", "process_input", span_id=4, span_type="CHAIN", start_ns=1300000000
        ),
        create_test_span(
            "trace1", "process_input", span_id=5, span_type="CHAIN", start_ns=1400000000
        ),
        create_test_span(
            "trace1", "validate_output", span_id=6, span_type="TOOL", start_ns=1500000000
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_NAME],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "generate_response"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "process_input"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "validate_output"},
        "values": {"COUNT": 1},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_NAME],
        filters=["span.type = 'TOOL'"],
    )
    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "validate_output"},
        "values": {"COUNT": 1},
    }


def test_query_span_metrics_count_by_span_name_and_type(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_count_by_span_name_and_type")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different names and types
    spans = [
        create_test_span("trace1", "llm_call", span_id=1, span_type="LLM", start_ns=1000000000),
        create_test_span("trace1", "llm_call", span_id=2, span_type="LLM", start_ns=1100000000),
        create_test_span("trace1", "tool_call", span_id=3, span_type="TOOL", start_ns=1200000000),
        create_test_span("trace1", "tool_call", span_id=4, span_type="TOOL", start_ns=1300000000),
        create_test_span("trace1", "chain_call", span_id=5, span_type="CHAIN", start_ns=1400000000),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.SPAN_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[SpanMetricDimensionKey.SPAN_NAME, SpanMetricDimensionKey.SPAN_TYPE],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            SpanMetricDimensionKey.SPAN_NAME: "chain_call",
            SpanMetricDimensionKey.SPAN_TYPE: "CHAIN",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            SpanMetricDimensionKey.SPAN_NAME: "llm_call",
            SpanMetricDimensionKey.SPAN_TYPE: "LLM",
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": SpanMetricKey.SPAN_COUNT,
        "dimensions": {
            SpanMetricDimensionKey.SPAN_NAME: "tool_call",
            SpanMetricDimensionKey.SPAN_TYPE: "TOOL",
        },
        "values": {"COUNT": 2},
    }


def test_query_span_metrics_latency_avg(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_latency_avg")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different latencies (end_ns - start_ns in nanoseconds)
    # Latency in ms = (end_ns - start_ns) / 1_000_000
    spans = [
        # 100ms latency
        create_test_span(
            "trace1",
            "span1",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,
        ),
        # 200ms latency
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,
        ),
        # 300ms latency
        create_test_span(
            "trace1",
            "span3",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3300000000,
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {},
        "values": {"AVG": 200.0},  # (100 + 200 + 300) / 3 = 200
    }


def test_query_span_metrics_latency_avg_by_span_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_latency_avg_by_span_name")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different names and latencies
    spans = [
        # generate_response: 100ms, 200ms, 300ms -> avg = 200ms
        create_test_span(
            "trace1",
            "generate_response",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,
        ),
        create_test_span(
            "trace1",
            "generate_response",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,
        ),
        create_test_span(
            "trace1",
            "generate_response",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3300000000,
        ),
        # process_input: 50ms, 150ms -> avg = 100ms
        create_test_span(
            "trace1",
            "process_input",
            span_id=4,
            span_type="CHAIN",
            start_ns=4000000000,
            end_ns=4050000000,
        ),
        create_test_span(
            "trace1",
            "process_input",
            span_id=5,
            span_type="CHAIN",
            start_ns=5000000000,
            end_ns=5150000000,
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[SpanMetricDimensionKey.SPAN_NAME],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "generate_response"},
        "values": {"AVG": 200.0},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "process_input"},
        "values": {"AVG": 100.0},
    }


def test_query_span_metrics_latency_by_span_status(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_latency_by_span_status")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different statuses and latencies
    spans = [
        # OK status: 100ms, 200ms -> avg = 150ms
        create_test_span(
            "trace1",
            "span1",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,
            status=trace_api.StatusCode.OK,
        ),
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,
            status=trace_api.StatusCode.OK,
        ),
        # ERROR status: 50ms, 150ms -> avg = 100ms
        create_test_span(
            "trace1",
            "span3",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3050000000,
            status=trace_api.StatusCode.ERROR,
        ),
        create_test_span(
            "trace1",
            "span4",
            span_id=4,
            span_type="LLM",
            start_ns=4000000000,
            end_ns=4150000000,
            status=trace_api.StatusCode.ERROR,
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[SpanMetricDimensionKey.SPAN_STATUS],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {SpanMetricDimensionKey.SPAN_STATUS: "ERROR"},
        "values": {"AVG": 100.0},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {SpanMetricDimensionKey.SPAN_STATUS: "OK"},
        "values": {"AVG": 150.0},
    }


@pytest.mark.parametrize(
    "percentile_value",
    [50.0, 75.0, 90.0, 95.0, 99.0],
)
def test_query_span_metrics_latency_percentiles(
    store: SqlAlchemyStore,
    percentile_value: float,
):
    exp_id = store.create_experiment(f"test_span_latency_percentile_{percentile_value}")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different latencies: 100ms, 200ms, 300ms, 400ms, 500ms
    spans = [
        create_test_span(
            "trace1",
            "span1",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,  # 100ms
        ),
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,  # 200ms
        ),
        create_test_span(
            "trace1",
            "span3",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3300000000,  # 300ms
        ),
        create_test_span(
            "trace1",
            "span4",
            span_id=4,
            span_type="LLM",
            start_ns=4000000000,
            end_ns=4400000000,  # 400ms
        ),
        create_test_span(
            "trace1",
            "span5",
            span_id=5,
            span_type="LLM",
            start_ns=5000000000,
            end_ns=5500000000,  # 500ms
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(
                aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
            )
        ],
    )

    # Calculate expected percentile value
    latency_values = [100.0, 200.0, 300.0, 400.0, 500.0]
    expected_percentile = _get_expected_percentile_value(
        store, percentile_value, 100.0, 500.0, latency_values
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {},
        "values": {f"P{percentile_value}": expected_percentile},
    }


def test_query_span_metrics_latency_percentiles_by_span_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_latency_percentiles_by_span_name")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different names and latencies
    spans = [
        # generate_response: 100ms, 200ms, 300ms
        create_test_span(
            "trace1",
            "generate_response",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,
        ),
        create_test_span(
            "trace1",
            "generate_response",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,
        ),
        create_test_span(
            "trace1",
            "generate_response",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3300000000,
        ),
        # process_input: 50ms, 150ms
        create_test_span(
            "trace1",
            "process_input",
            span_id=4,
            span_type="CHAIN",
            start_ns=4000000000,
            end_ns=4050000000,
        ),
        create_test_span(
            "trace1",
            "process_input",
            span_id=5,
            span_type="CHAIN",
            start_ns=5000000000,
            end_ns=5150000000,
        ),
    ]
    store.log_spans(exp_id, spans)

    percentile_value = 50.0
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(
                aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
            )
        ],
        dimensions=[SpanMetricDimensionKey.SPAN_NAME],
    )

    # Calculate expected percentile values
    gen_resp_values = [100.0, 200.0, 300.0]
    proc_input_values = [50.0, 150.0]

    expected_gen_resp = _get_expected_percentile_value(
        store, percentile_value, 100.0, 300.0, gen_resp_values
    )
    expected_proc_input = _get_expected_percentile_value(
        store, percentile_value, 50.0, 150.0, proc_input_values
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "generate_response"},
        "values": {f"P{percentile_value}": expected_gen_resp},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {SpanMetricDimensionKey.SPAN_NAME: "process_input"},
        "values": {f"P{percentile_value}": expected_proc_input},
    }


def test_query_span_metrics_latency_multiple_aggregations(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_latency_multiple_aggregations")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with latencies: 100ms, 200ms, 300ms, 400ms, 500ms
    spans = [
        create_test_span(
            "trace1",
            "span1",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,
        ),
        create_test_span(
            "trace1",
            "span2",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,
        ),
        create_test_span(
            "trace1",
            "span3",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3300000000,
        ),
        create_test_span(
            "trace1",
            "span4",
            span_id=4,
            span_type="LLM",
            start_ns=4000000000,
            end_ns=4400000000,
        ),
        create_test_span(
            "trace1",
            "span5",
            span_id=5,
            span_type="LLM",
            start_ns=5000000000,
            end_ns=5500000000,
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.AVG),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=50),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=95),
        ],
    )

    latency_values = [100.0, 200.0, 300.0, 400.0, 500.0]
    expected_p50 = _get_expected_percentile_value(store, 50.0, 100.0, 500.0, latency_values)
    expected_p95 = _get_expected_percentile_value(store, 95.0, 100.0, 500.0, latency_values)

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {},
        "values": {"AVG": 300.0, "P50": expected_p50, "P95": expected_p95},
    }


def test_query_span_metrics_latency_by_span_name_and_status(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_span_latency_by_span_name_and_status")

    trace_info = TraceInfo(
        trace_id="trace1",
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create spans with different names, statuses, and latencies
    spans = [
        # generate_response + OK: 100ms, 200ms -> avg = 150ms
        create_test_span(
            "trace1",
            "generate_response",
            span_id=1,
            span_type="LLM",
            start_ns=1000000000,
            end_ns=1100000000,
            status=trace_api.StatusCode.OK,
        ),
        create_test_span(
            "trace1",
            "generate_response",
            span_id=2,
            span_type="LLM",
            start_ns=2000000000,
            end_ns=2200000000,
            status=trace_api.StatusCode.OK,
        ),
        # generate_response + ERROR: 300ms
        create_test_span(
            "trace1",
            "generate_response",
            span_id=3,
            span_type="LLM",
            start_ns=3000000000,
            end_ns=3300000000,
            status=trace_api.StatusCode.ERROR,
        ),
        # process_input + OK: 50ms
        create_test_span(
            "trace1",
            "process_input",
            span_id=4,
            span_type="CHAIN",
            start_ns=4000000000,
            end_ns=4050000000,
            status=trace_api.StatusCode.OK,
        ),
    ]
    store.log_spans(exp_id, spans)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[SpanMetricDimensionKey.SPAN_NAME, SpanMetricDimensionKey.SPAN_STATUS],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {
            SpanMetricDimensionKey.SPAN_NAME: "generate_response",
            SpanMetricDimensionKey.SPAN_STATUS: "ERROR",
        },
        "values": {"AVG": 300.0},
    }
    assert asdict(result[1]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {
            SpanMetricDimensionKey.SPAN_NAME: "generate_response",
            SpanMetricDimensionKey.SPAN_STATUS: "OK",
        },
        "values": {"AVG": 150.0},
    }
    assert asdict(result[2]) == {
        "metric_name": SpanMetricKey.LATENCY,
        "dimensions": {
            SpanMetricDimensionKey.SPAN_NAME: "process_input",
            SpanMetricDimensionKey.SPAN_STATUS: "OK",
        },
        "values": {"AVG": 50.0},
    }


def test_query_assessment_metrics_count_no_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_count_no_dimensions")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments = [
        Feedback(
            trace_id=trace_id,
            name="correctness",
            value=True,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user1@test.com"
            ),
        ),
        Feedback(
            trace_id=trace_id,
            name="relevance",
            value=0.8,
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"),
        ),
        Expectation(
            trace_id=trace_id,
            name="expected_output",
            value="Hello World",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test_suite"),
        ),
    ]

    for assessment in assessments:
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {},
        "values": {"COUNT": 3},
    }


def test_query_assessment_metrics_count_by_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_count_by_name")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create assessments with different names
    assessments_data = [
        ("correctness", True),
        ("correctness", False),
        ("relevance", 0.9),
        ("relevance", 0.8),
        ("relevance", 0.7),
        ("quality", "high"),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "quality"},
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "relevance"},
        "values": {"COUNT": 3},
    }


def test_query_assessment_metrics_count_by_value_and_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_count_by_value")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create assessments with different values
    assessments_data = [
        ("correctness", True),
        ("correctness", True),
        ("correctness", False),
        ("quality", "high"),
        ("quality", "high"),
        ("quality", "low"),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[
            AssessmentMetricDimensionKey.ASSESSMENT_NAME,
            AssessmentMetricDimensionKey.ASSESSMENT_VALUE,
        ],
    )

    # Values are stored as JSON strings
    assert len(result) == 4
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness",
            AssessmentMetricDimensionKey.ASSESSMENT_VALUE: json.dumps(False),
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness",
            AssessmentMetricDimensionKey.ASSESSMENT_VALUE: json.dumps(True),
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "quality",
            AssessmentMetricDimensionKey.ASSESSMENT_VALUE: json.dumps("high"),
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[3]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "quality",
            AssessmentMetricDimensionKey.ASSESSMENT_VALUE: json.dumps("low"),
        },
        "values": {"COUNT": 1},
    }


def test_query_assessment_metrics_with_time_interval(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_with_time_interval")

    # Base time in milliseconds (2020-01-01 00:00:00 UTC)
    base_time_ms = 1577836800000
    hour_ms = 60 * 60 * 1000

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=base_time_ms,
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create assessments at different times
    assessment_times = [
        base_time_ms,
        base_time_ms + 10 * 60 * 1000,  # +10 minutes
        base_time_ms + hour_ms,  # +1 hour
        base_time_ms + hour_ms + 30 * 60 * 1000,  # +1.5 hours
        base_time_ms + 2 * hour_ms,  # +2 hours
    ]

    for i, timestamp in enumerate(assessment_times):
        assessment = Feedback(
            trace_id=trace_id,
            name=f"quality_{i}",
            value=True,
            create_time_ms=timestamp,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time_ms,
        end_time_ms=base_time_ms + 3 * hour_ms,
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(base_time_ms / 1000, tz=timezone.utc).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + 2 * hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 1},
    }


def test_query_assessment_metrics_with_time_interval_and_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_with_time_interval_and_dimensions")

    # Base time in milliseconds (2020-01-01 00:00:00 UTC)
    base_time_ms = 1577836800000
    hour_ms = 60 * 60 * 1000

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=base_time_ms,
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create assessments at different times with different names
    assessments_data = [
        (base_time_ms, "correctness"),
        (base_time_ms + 10 * 60 * 1000, "relevance"),
        (base_time_ms + hour_ms, "correctness"),
        (base_time_ms + hour_ms, "relevance"),
    ]

    for timestamp, name in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=True,
            create_time_ms=timestamp,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time_ms,
        end_time_ms=base_time_ms + 2 * hour_ms,
    )

    time_bucket_1 = datetime.fromtimestamp(base_time_ms / 1000, tz=timezone.utc).isoformat()
    time_bucket_2 = datetime.fromtimestamp(
        (base_time_ms + hour_ms) / 1000, tz=timezone.utc
    ).isoformat()

    assert len(result) == 4
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_1,
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_1,
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "relevance",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_2,
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[3]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {
            "time_bucket": time_bucket_2,
            AssessmentMetricDimensionKey.ASSESSMENT_NAME: "relevance",
        },
        "values": {"COUNT": 1},
    }


def test_query_assessment_metrics_with_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_with_filters")

    # Create traces with different statuses
    trace_id1 = f"tr-{uuid.uuid4().hex}"
    trace1 = TraceInfo(
        trace_id=trace_id1,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace1)

    trace_id2 = f"tr-{uuid.uuid4().hex}"
    trace2 = TraceInfo(
        trace_id=trace_id2,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.ERROR,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace2)

    # Create assessments for trace1 (OK status)
    for i in range(3):
        assessment = Feedback(
            trace_id=trace_id1,
            name=f"quality_{i}",
            value=True,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    # Create assessments for trace2 (ERROR status)
    for i in range(2):
        assessment = Feedback(
            trace_id=trace_id2,
            name=f"error_check_{i}",
            value=False,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    # Query assessments only for traces with OK status
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'OK'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {},
        "values": {"COUNT": 3},
    }

    # Query assessments for traces with ERROR status
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'ERROR'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


def test_query_assessment_metrics_across_multiple_traces(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_across_multiple_traces")

    # Create multiple traces
    for i in range(3):
        trace_id = f"tr-{uuid.uuid4().hex}"
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=get_current_time_millis(),
            execution_duration=100,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: f"workflow_{i}"},
        )
        store.start_trace(trace_info)

        # Create assessments for each trace
        assessments_data = [
            ("correctness", True),
            ("relevance", 0.9),
        ]
        for name, value in assessments_data:
            assessment = Feedback(
                trace_id=trace_id,
                name=name,
                value=value,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
                ),
            )
            store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "relevance"},
        "values": {"COUNT": 3},
    }


def test_query_assessment_value_avg_by_name(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_avg_by_name")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data = [
        ("accuracy", 0.8),
        ("accuracy", 0.9),
        ("accuracy", 0.85),
        ("precision", 0.7),
        ("precision", 0.75),
        ("quality", "high"),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "accuracy"},
        "values": {"AVG": pytest.approx(0.85, abs=0.01)},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "precision"},
        "values": {"AVG": pytest.approx(0.725, abs=0.01)},
    }
    # non-numeric value result should be None
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "quality"},
        "values": {"AVG": None},
    }


def test_query_assessment_value_with_boolean_values(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_with_boolean")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data = [
        ("correctness", True),
        ("correctness", True),
        ("correctness", False),
        ("correctness", True),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "correctness"},
        "values": {"AVG": pytest.approx(3 / 4, abs=0.01)},
    }


@pytest.mark.parametrize(
    "percentile_value",
    [50.0, 75.0, 90.0, 95.0, 99.0],
)
def test_query_assessment_value_percentiles(
    store: SqlAlchemyStore,
    percentile_value: float,
):
    exp_id = store.create_experiment(f"test_assessment_value_p{percentile_value}")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data_accuracy = [
        ("accuracy", 0.1),
        ("accuracy", 0.2),
        ("accuracy", 0.3),
        ("accuracy", 0.4),
        ("accuracy", 0.5),
    ]
    assessments_data_score = [
        ("score", 10.0),
        ("score", 20.0),
        ("score", 30.0),
    ]

    for name, value in assessments_data_accuracy + assessments_data_score:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[
            MetricAggregation(
                aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
            )
        ],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    accuracy_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    score_values = [10.0, 20.0, 30.0]

    expected_accuracy = pytest.approx(
        _get_expected_percentile_value(
            store, percentile_value, min(accuracy_values), max(accuracy_values), accuracy_values
        ),
        abs=0.01,
    )
    expected_score = pytest.approx(
        _get_expected_percentile_value(
            store, percentile_value, min(score_values), max(score_values), score_values
        ),
        abs=0.01,
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "accuracy"},
        "values": {f"P{percentile_value}": expected_accuracy},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "score"},
        "values": {f"P{percentile_value}": expected_score},
    }


def test_query_assessment_value_mixed_types(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_mixed_types")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data = [
        ("rating", 5.0),
        ("rating", 4.5),
        ("rating", 3.0),
        ("status", "good"),
        ("status", "bad"),
        ("passed", True),
        ("passed", False),
        ("passed", True),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "passed"},
        "values": {"AVG": pytest.approx(2.0 / 3.0, abs=0.01)},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "rating"},
        "values": {"AVG": pytest.approx(12.5 / 3.0, abs=0.01)},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "status"},
        "values": {"AVG": None},
    }


def test_query_assessment_value_multiple_aggregations(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_multiple_aggs")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data = [
        ("score", 10.0),
        ("score", 20.0),
        ("score", 30.0),
        ("score", 40.0),
        ("score", 50.0),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.AVG),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=50),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=95),
        ],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected_p50 = pytest.approx(
        _get_expected_percentile_value(store, 50.0, 10.0, 50.0, values), abs=0.01
    )
    expected_p95 = pytest.approx(
        _get_expected_percentile_value(store, 95.0, 10.0, 50.0, values), abs=0.01
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "score"},
        "values": {"AVG": 30.0, "P50": expected_p50, "P95": expected_p95},
    }


@pytest.mark.parametrize(
    "assessment_type",
    [Feedback, Expectation],
)
def test_query_assessment_value_no_dimensions(
    store: SqlAlchemyStore, assessment_type: type[Assessment]
):
    exp_id = store.create_experiment("test_assessment_value_no_dimensions")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data = [
        ("accuracy", 0.8),
        ("accuracy", 0.9),
        ("precision", 0.7),
        ("recall", 0.85),
    ]

    for name, value in assessments_data:
        assessment = assessment_type(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {},
        "values": {"AVG": pytest.approx(3.25 / 4, abs=0.01)},
    }


def test_query_assessment_value_with_time_bucket(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_time_bucket")

    # Base time in milliseconds (2020-01-01 00:00:00 UTC)
    base_time_ms = 1577836800000
    hour_ms = 60 * 60 * 1000

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=base_time_ms,
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create assessments with numeric values at different times
    assessment_data = [
        # Hour 0: avg should be (0.8 + 0.9) / 2 = 0.85
        (base_time_ms, "accuracy", 0.8),
        (base_time_ms + 10 * 60 * 1000, "accuracy", 0.9),
        # Hour 1: avg should be (0.7 + 0.75) / 2 = 0.725
        (base_time_ms + hour_ms, "precision", 0.7),
        (base_time_ms + hour_ms + 30 * 60 * 1000, "precision", 0.75),
        # Hour 2: avg should be 0.95
        (base_time_ms + 2 * hour_ms, "recall", 0.95),
    ]

    for timestamp, name, value in assessment_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            create_time_ms=timestamp,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        time_interval_seconds=3600,
        start_time_ms=base_time_ms,
        end_time_ms=base_time_ms + 3 * hour_ms,
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(base_time_ms / 1000, tz=timezone.utc).isoformat()
        },
        "values": {"AVG": pytest.approx(0.85, abs=0.01)},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"AVG": pytest.approx(0.725, abs=0.01)},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + 2 * hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"AVG": pytest.approx(0.95, abs=0.01)},
    }


@pytest.mark.parametrize(
    "assessment_type",
    [Feedback, Expectation],
)
def test_query_assessment_invalid_values(store: SqlAlchemyStore, assessment_type: type[Assessment]):
    exp_id = store.create_experiment("test_assessment_invalid_values")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessments_data = [
        ("string_score", json.dumps("test")),
        ("list_score", json.dumps([1, 2, 3])),
        ("dict_score", json.dumps({"a": 1, "b": 2})),
    ]

    for name, value in assessments_data:
        assessment = assessment_type(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "dict_score"},
        "values": {"AVG": None},
    }
    assert asdict(result[1]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "list_score"},
        "values": {"AVG": None},
    }
    assert asdict(result[2]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "string_score"},
        "values": {"AVG": None},
    }


def test_query_assessment_value_with_null_value(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_null_value")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessment = Feedback(
        trace_id=trace_id,
        name="score",
        value=None,
        error=AssessmentError(
            error_message="Null value",
            error_code="NULL_VALUE",
        ),
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"),
    )
    store.create_assessment(assessment)
    assessment = Feedback(
        trace_id=trace_id,
        name="score",
        value=12,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"),
    )
    store.create_assessment(assessment)

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=["assessment_name"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {"assessment_name": "score"},
        "values": {"AVG": 12},
    }


def test_query_assessment_value_with_assessment_name_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_value_with_filter")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create multiple assessments with different names
    assessments_data = [
        ("accuracy", 0.8),
        ("accuracy", 0.9),
        ("precision", 0.7),
        ("recall", 0.85),
    ]

    for name, value in assessments_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    # Query with assessment_name filter
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
        filters=["assessment.name = 'accuracy'"],
    )

    # Should only return accuracy results
    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_VALUE,
        "dimensions": {AssessmentMetricDimensionKey.ASSESSMENT_NAME: "accuracy"},
        "values": {"AVG": pytest.approx(0.85, abs=0.01)},
    }


def test_query_assessment_with_type_filter(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_with_type_filter")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create feedback assessments
    for i in range(3):
        assessment = Feedback(
            trace_id=trace_id,
            name=f"feedback_{i}",
            value=0.8,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    # Create expectation assessments
    for i in range(2):
        assessment = Expectation(
            trace_id=trace_id,
            name=f"expectation_{i}",
            value="expected",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test_suite"),
        )
        store.create_assessment(assessment)

    # Query with assessment.type = 'feedback'
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["assessment.type = 'feedback'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {},
        "values": {"COUNT": 3},
    }

    # Query with assessment.type = 'expectation'
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["assessment.type = 'expectation'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


def test_query_assessment_with_combined_name_and_type_filters(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_assessment_combined_filters")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    # Create feedback assessments with different names
    feedback_data = [
        ("accuracy", 0.8),
        ("accuracy", 0.9),
        ("precision", 0.7),
    ]
    for name, value in feedback_data:
        assessment = Feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"
            ),
        )
        store.create_assessment(assessment)

    # Create expectation assessments
    expectation_data = [
        ("accuracy", "expected_value"),
        ("recall", "expected_value"),
    ]
    for name, value in expectation_data:
        assessment = Expectation(
            trace_id=trace_id,
            name=name,
            value=value,
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test_suite"),
        )
        store.create_assessment(assessment)

    # Query with both type and name filters - should only return feedback with name 'accuracy'
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[
            "assessment.type = 'feedback'",
            "assessment.name = 'accuracy'",
        ],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": AssessmentMetricKey.ASSESSMENT_COUNT,
        "dimensions": {},
        "values": {"COUNT": 2},
    }


@pytest.mark.parametrize(
    ("filter_string", "error_match"),
    [
        ("name = 'accuracy'", r"Invalid identifier 'name'"),
        (
            "assessment.name != 'accuracy'",
            r"Invalid comparator: '!=', only '=' operator is supported",
        ),
        ("assessment.invalid_field = 'value'", r"Invalid entity 'invalid_field' specified"),
        ("assessment.name.extra = 'value'", r"does not require a key"),
        ("assessment.type LIKE 'feed%'", r"only '=' operator is supported"),
        ("assessment.value = '0.8'", r"Invalid entity 'value' specified"),
    ],
)
def test_query_assessments_metrics_invalid_filters(
    store: SqlAlchemyStore, filter_string: str, error_match: str
):
    exp_id = store.create_experiment("test_assessment_invalid_filters")

    trace_id = f"tr-{uuid.uuid4().hex}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=get_current_time_millis(),
        execution_duration=100,
        state=TraceStatus.OK,
        tags={TraceTagKey.TRACE_NAME: "test_trace"},
    )
    store.start_trace(trace_info)

    assessment = Feedback(
        trace_id=trace_id,
        name="accuracy",
        value=0.8,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user@test.com"),
    )
    store.create_assessment(assessment)

    with pytest.raises(MlflowException, match=error_match):
        store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
            filters=[filter_string],
        )
