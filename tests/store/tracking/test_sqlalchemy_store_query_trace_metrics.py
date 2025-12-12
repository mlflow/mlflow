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
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.utils.time import get_current_time_millis

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
