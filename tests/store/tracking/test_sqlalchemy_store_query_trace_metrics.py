import json
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from mlflow.entities import trace_location
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import POSTGRES
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import TokenUsageKey, TraceMetadataKey, TraceTagKey
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["status"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {"status": "ERROR"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": "trace",
        "dimensions": {"status": "OK"},
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["name"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {"name": "workflow_a"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[1]) == {
        "metric_name": "trace",
        "dimensions": {"name": "workflow_b"},
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["status", "name"],
    )

    assert len(result) == 4
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {"status": "ERROR", "name": "workflow_a"},
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": "trace",
        "dimensions": {"status": "ERROR", "name": "workflow_b"},
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": "trace",
        "dimensions": {"status": "OK", "name": "workflow_a"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[3]) == {
        "metric_name": "trace",
        "dimensions": {"status": "OK", "name": "workflow_b"},
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
        metric_name="latency",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=["name"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "latency",
        "dimensions": {"name": "workflow_a"},
        "values": {"AVG": 200.0},
    }
    assert asdict(result[1]) == {
        "metric_name": "latency",
        "dimensions": {"name": "workflow_b"},
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
        metric_name="latency",
        aggregations=[
            MetricAggregation(
                aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
            )
        ],
        dimensions=["name"],
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
        "metric_name": "latency",
        "dimensions": {"name": "workflow_a"},
        "values": {f"P{percentile_value}": expected_workflow_a},
    }
    assert asdict(result[1]) == {
        "metric_name": "latency",
        "dimensions": {"name": "workflow_b"},
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
        metric_name="latency",
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.AVG),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=95),
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=99),
        ],
        dimensions=["name"],
    )

    # Calculate expected percentile value based on database type
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    expected_p95 = _get_expected_percentile_value(store, 95.0, 100.0, 500.0, values)
    expected_p99 = _get_expected_percentile_value(store, 99.0, 100.0, 500.0, values)

    assert asdict(result[0]) == {
        "metric_name": "latency",
        "dimensions": {"name": "workflow_a"},
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time,
        end_time_ms=base_time + 3 * hour_ms,
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(base_time / 1000, tz=timezone.utc).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": "trace",
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time + hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": "trace",
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["status"],
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
        "metric_name": "trace",
        "dimensions": {
            "time_bucket": time_bucket_1,
            "status": "ERROR",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": "trace",
        "dimensions": {
            "time_bucket": time_bucket_1,
            "status": "OK",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": "trace",
        "dimensions": {
            "time_bucket": time_bucket_2,
            "status": "ERROR",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[3]) == {
        "metric_name": "trace",
        "dimensions": {
            "time_bucket": time_bucket_2,
            "status": "OK",
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'OK'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {},
        "values": {"COUNT": 3},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'ERROR'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run_123'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {},
        "values": {"COUNT": 2},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run_456'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=[
            "trace.status = 'OK'",
            f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run_123'",
        ],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
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
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.tag.tag1 = 'value1'"],
    )
    assert len(result) == 1
    assert asdict(result[0])["values"] == {"COUNT": 5}

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.tag.`model.version` = 'model_v1'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {},
        "values": {"COUNT": 2},
    }

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="trace",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.tag.`model.version` = 'model_v2'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "trace",
        "dimensions": {},
        "values": {"COUNT": 2},
    }


@pytest.mark.parametrize(
    ("filter_string", "error_match"),
    [
        ("status = 'OK'", r"Filter must start with 'trace\.' prefix"),
        ("trace.status != 'OK'", r"Only '=' operator is supported for trace metrics"),
        ("trace.unsupported_field = 'value'", r"Invalid attribute key"),
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
            metric_name="trace",
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
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: total_tokens,
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
        metric_name="total_tokens",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.SUM)],
        dimensions=["name"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "total_tokens",
        "dimensions": {"name": "workflow_a"},
        "values": {"SUM": 675},
    }
    assert asdict(result[1]) == {
        "metric_name": "total_tokens",
        "dimensions": {"name": "workflow_b"},
        "values": {"SUM": 825},
    }


def test_query_trace_metrics_total_tokens_avg(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="total_tokens",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        dimensions=["name"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "total_tokens",
        "dimensions": {"name": "workflow_a"},
        "values": {"AVG": 225.0},
    }
    assert asdict(result[1]) == {
        "metric_name": "total_tokens",
        "dimensions": {"name": "workflow_b"},
        "values": {"AVG": 412.5},
    }


def test_query_trace_metrics_total_tokens_percentiles(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup
    percentiles = [50, 75, 90, 95, 99]

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="total_tokens",
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=p)
            for p in percentiles
        ],
        dimensions=["name"],
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
        "metric_name": "total_tokens",
        "dimensions": {"name": "workflow_a"},
        "values": expected_workflow_a_values,
    }
    assert asdict(result[1]) == {
        "metric_name": "total_tokens",
        "dimensions": {"name": "workflow_b"},
        "values": expected_workflow_b_values,
    }


def test_query_trace_metrics_total_tokens_no_dimensions(traces_with_token_usage_setup):
    exp_id, store = traces_with_token_usage_setup

    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name="total_tokens",
        aggregations=[
            MetricAggregation(aggregation_type=AggregationType.SUM),
            MetricAggregation(aggregation_type=AggregationType.AVG),
        ],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "total_tokens",
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
        metric_name="total_tokens",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.SUM)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "total_tokens",
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
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "span",
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
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["span_type"],
    )

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "CHAIN"},
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "LLM"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[2]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "TOOL"},
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
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        time_interval_seconds=3600,  # 1 hour
        start_time_ms=base_time_ns // 1_000_000,
        end_time_ms=(base_time_ns + 3 * hour_ns) // 1_000_000,
    )

    base_time_ms = base_time_ns // 1_000_000
    hour_ms = 60 * 60 * 1000

    assert len(result) == 3
    assert asdict(result[0]) == {
        "metric_name": "span",
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(base_time_ms / 1000, tz=timezone.utc).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[1]) == {
        "metric_name": "span",
        "dimensions": {
            "time_bucket": datetime.fromtimestamp(
                (base_time_ms + hour_ms) / 1000, tz=timezone.utc
            ).isoformat()
        },
        "values": {"COUNT": 2},
    }
    assert asdict(result[2]) == {
        "metric_name": "span",
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
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["span_type"],
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
        "metric_name": "span",
        "dimensions": {
            "time_bucket": time_bucket_1,
            "span_type": "CHAIN",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": "span",
        "dimensions": {
            "time_bucket": time_bucket_1,
            "span_type": "LLM",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[2]) == {
        "metric_name": "span",
        "dimensions": {
            "time_bucket": time_bucket_2,
            "span_type": "CHAIN",
        },
        "values": {"COUNT": 1},
    }
    assert asdict(result[3]) == {
        "metric_name": "span",
        "dimensions": {
            "time_bucket": time_bucket_2,
            "span_type": "LLM",
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
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        filters=["trace.status = 'OK'"],
    )

    assert len(result) == 1
    assert asdict(result[0]) == {
        "metric_name": "span",
        "dimensions": {},
        "values": {"COUNT": 3},
    }

    # Query spans grouped by type for traces with ERROR status
    result = store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["span_type"],
        filters=["trace.status = 'ERROR'"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "CHAIN"},
        "values": {"COUNT": 1},
    }
    assert asdict(result[1]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "LLM"},
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
        metric_name="span",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=["span_type"],
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "CHAIN"},
        "values": {"COUNT": 3},
    }
    assert asdict(result[1]) == {
        "metric_name": "span",
        "dimensions": {"span_type": "LLM"},
        "values": {"COUNT": 3},
    }
