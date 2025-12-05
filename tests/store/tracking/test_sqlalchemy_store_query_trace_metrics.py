from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from mlflow.entities import trace_location
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_metrics import AggregationType, MetricsViewType, TimeGranularity
from mlflow.entities.trace_status import TraceStatus
from mlflow.store.db.db_types import MSSQL, POSTGRES
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import TraceTagKey
from mlflow.utils.time import get_current_time_millis

pytestmark = pytest.mark.notrackingurimock


def _get_expected_percentile_value(
    store: SqlAlchemyStore, percentile: float, min_val: float, max_val: float, values: list[float]
) -> float:
    """
    Calculate expected percentile value based on database type.

    PostgreSQL and MSSQL use linear interpolation (PERCENTILE_CONT).
    MySQL and SQLite use min + percentile * (max - min) approximation.
    """
    db_type = store._get_dialect()
    if db_type in (POSTGRES, MSSQL):
        # Linear interpolation for PERCENTILE_CONT
        # For a sorted list, percentile index = percentile * (n - 1)
        sorted_values = sorted(values)
        n = len(sorted_values)
        index = percentile * (n - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, n - 1)
        fraction = index - lower_idx
        return sorted_values[lower_idx] + fraction * (
            sorted_values[upper_idx] - sorted_values[lower_idx]
        )
    else:
        # Approximation for MySQL and SQLite
        return min_val + percentile * (max_val - min_val)


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
        view_type=MetricsViewType.TRACES,
        metric_name="trace",
        aggregation_types=[AggregationType.COUNT],
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
        view_type=MetricsViewType.TRACES,
        metric_name="trace",
        aggregation_types=[AggregationType.COUNT],
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
        view_type=MetricsViewType.TRACES,
        metric_name="trace",
        aggregation_types=[AggregationType.COUNT],
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
        view_type=MetricsViewType.TRACES,
        metric_name="trace",
        aggregation_types=[AggregationType.COUNT],
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
        view_type=MetricsViewType.TRACES,
        metric_name="latency",
        aggregation_types=[AggregationType.AVG],
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
    "agg_type",
    [
        AggregationType.P50,
        AggregationType.P75,
        AggregationType.P90,
        AggregationType.P95,
        AggregationType.P99,
    ],
)
def test_query_trace_metrics_latency_percentiles(
    store: SqlAlchemyStore,
    agg_type: AggregationType,
):
    exp_id = store.create_experiment(f"test_latency_percentile_{agg_type.value}")

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
        view_type=MetricsViewType.TRACES,
        metric_name="latency",
        aggregation_types=[agg_type],
        dimensions=["name"],
    )

    # Calculate expected values based on database type
    percentile = agg_type.map_to_percentile()
    expected_workflow_a = _get_expected_percentile_value(
        store, percentile, 100.0, 300.0, [100.0, 200.0, 300.0]
    )
    expected_workflow_b = _get_expected_percentile_value(
        store, percentile, 100.0, 200.0, [100.0, 200.0]
    )

    assert len(result) == 2
    assert asdict(result[0]) == {
        "metric_name": "latency",
        "dimensions": {"name": "workflow_a"},
        "values": {agg_type.value: expected_workflow_a},
    }
    assert asdict(result[1]) == {
        "metric_name": "latency",
        "dimensions": {"name": "workflow_b"},
        "values": {agg_type.value: expected_workflow_b},
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
        view_type=MetricsViewType.TRACES,
        metric_name="latency",
        aggregation_types=[
            AggregationType.AVG,
            AggregationType.P50,
            AggregationType.P90,
            AggregationType.P99,
        ],
        dimensions=["name"],
    )

    # Calculate expected percentile values based on database type
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    expected_p50 = _get_expected_percentile_value(store, 0.5, 100.0, 500.0, values)
    expected_p90 = _get_expected_percentile_value(store, 0.9, 100.0, 500.0, values)
    expected_p99 = _get_expected_percentile_value(store, 0.99, 100.0, 500.0, values)

    assert len(result) == 1
    assert result[0].metric_name == "latency"
    assert result[0].dimensions == {"name": "workflow_a"}
    assert result[0].values["AVG"] == 300.0
    assert result[0].values["P50"] == expected_p50
    assert result[0].values["P90"] == expected_p90
    assert result[0].values["P99"] == expected_p99


def test_query_trace_metrics_with_time_granularity(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_time_granularity")

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
        view_type=MetricsViewType.TRACES,
        metric_name="trace",
        aggregation_types=[AggregationType.COUNT],
        time_granularity=TimeGranularity.HOUR,
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


def test_query_trace_metrics_with_time_granularity_and_dimensions(store: SqlAlchemyStore):
    exp_id = store.create_experiment("test_with_time_granularity_and_dimensions")

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
        view_type=MetricsViewType.TRACES,
        metric_name="trace",
        aggregation_types=[AggregationType.COUNT],
        dimensions=["status"],
        time_granularity=TimeGranularity.HOUR,
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
