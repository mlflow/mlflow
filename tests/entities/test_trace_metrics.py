import pytest

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricDataPoint,
    MetricsViewType,
    TimeGranularity,
)
from mlflow.protos import service_pb2 as pb


@pytest.mark.parametrize(
    ("view_type", "expected_proto"),
    [
        (MetricsViewType.TRACES, pb.MetricsViewType.TRACES),
        (MetricsViewType.SPANS, pb.MetricsViewType.SPANS),
        (MetricsViewType.ASSESSMENTS, pb.MetricsViewType.ASSESSMENTS),
    ],
)
def test_trace_metrics_view_type(view_type: MetricsViewType, expected_proto: pb.MetricsViewType):
    assert view_type.to_proto() == expected_proto


@pytest.mark.parametrize(
    ("aggregation_type", "expected_proto"),
    [
        (AggregationType.COUNT, pb.AggregationType.COUNT),
        (AggregationType.SUM, pb.AggregationType.SUM),
        (AggregationType.AVG, pb.AggregationType.AVG),
        (AggregationType.P50, pb.AggregationType.P50),
        (AggregationType.P75, pb.AggregationType.P75),
        (AggregationType.P90, pb.AggregationType.P90),
        (AggregationType.P95, pb.AggregationType.P95),
        (AggregationType.P99, pb.AggregationType.P99),
    ],
)
def test_trace_metrics_aggregation_type_to_proto(
    aggregation_type: AggregationType, expected_proto: pb.AggregationType
):
    assert aggregation_type.to_proto() == expected_proto


@pytest.mark.parametrize(
    ("time_granularity", "expected_proto"),
    [
        (TimeGranularity.MINUTE, pb.TimeGranularity.MINUTE),
        (TimeGranularity.HOUR, pb.TimeGranularity.HOUR),
        (TimeGranularity.DAY, pb.TimeGranularity.DAY),
        (TimeGranularity.WEEK, pb.TimeGranularity.WEEK),
        (TimeGranularity.MONTH, pb.TimeGranularity.MONTH),
    ],
)
def test_trace_metrics_time_granularity_to_proto(
    time_granularity: TimeGranularity, expected_proto: pb.TimeGranularity
):
    assert time_granularity.to_proto() == expected_proto


def test_trace_metrics_metric_data_point_from_proto():
    metric_data_point_proto = pb.MetricDataPoint(
        dimensions={"trace_status": "OK", "time_bucket": "2025-11-13"},
        metric_name="trace",
        values={"count": "150"},
    )
    assert MetricDataPoint.from_proto(metric_data_point_proto) == MetricDataPoint(
        dimensions={"trace_status": "OK", "time_bucket": "2025-11-13"},
        metric_name="trace",
        values={"count": "150"},
    )
