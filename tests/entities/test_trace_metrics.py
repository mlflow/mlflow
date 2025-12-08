import pytest

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricDataPoint,
    MetricsAggregation,
    MetricsViewType,
)
from mlflow.protos import service_pb2 as pb


@pytest.mark.parametrize(
    ("view_type", "expected_proto"),
    zip(MetricsViewType, pb.MetricsViewType.values(), strict=True),
)
def test_trace_metrics_view_type(view_type: MetricsViewType, expected_proto: pb.MetricsViewType):
    assert view_type.to_proto() == expected_proto


@pytest.mark.parametrize(
    ("aggregation_type", "expected_proto"),
    zip(AggregationType, pb.AggregationType.values(), strict=True),
)
def test_trace_metrics_aggregation_type_to_proto(
    aggregation_type: AggregationType, expected_proto: pb.AggregationType
):
    assert aggregation_type.to_proto() == expected_proto


def test_metrics_aggregation_to_proto_without_percentile():
    aggregation = MetricsAggregation(aggregation_type=AggregationType.AVG)
    proto = aggregation.to_proto()
    assert proto.aggregation_type == pb.AggregationType.AVG
    assert not proto.HasField("percentile_value")


@pytest.mark.parametrize(
    ("percentile_value"),
    [50.0, 75.0, 90.0, 95.0, 99.0, 99.9],
)
def test_metrics_aggregation_percentile_values(percentile_value: float):
    aggregation = MetricsAggregation(
        aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
    )
    proto = aggregation.to_proto()
    assert proto.percentile_value == percentile_value


def test_metrics_aggregation_percentile_requires_value():
    with pytest.raises(ValueError, match="Percentile value is required for PERCENTILE aggregation"):
        MetricsAggregation(aggregation_type=AggregationType.PERCENTILE)


@pytest.mark.parametrize("percentile_value", [-1.0, -0.1, 100.1, 101.0, 1000.0])
def test_metrics_aggregation_percentile_value_out_of_range(percentile_value: float):
    with pytest.raises(ValueError, match="Percentile value must be between 0 and 100"):
        MetricsAggregation(
            aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
        )


@pytest.mark.parametrize("percentile_value", [0.0, 0.1, 50.0, 99.9, 100.0])
def test_metrics_aggregation_percentile_value_valid_range(percentile_value: float):
    aggregation = MetricsAggregation(
        aggregation_type=AggregationType.PERCENTILE, percentile_value=percentile_value
    )
    assert aggregation.percentile_value == percentile_value


@pytest.mark.parametrize(
    "agg_type",
    [t for t in AggregationType if t is not AggregationType.PERCENTILE],
)
def test_metrics_aggregation_non_percentile_with_value_raises(agg_type: AggregationType):
    with pytest.raises(
        ValueError, match="Percentile value is only allowed for PERCENTILE aggregation"
    ):
        MetricsAggregation(aggregation_type=agg_type, percentile_value=50.0)


def test_trace_metrics_metric_data_point_from_proto():
    metric_data_point_proto = pb.MetricDataPoint(
        metric_name="latency",
        dimensions={"status": "OK"},
        values={"avg": 150.5, "p99": 200},
    )
    assert MetricDataPoint.from_proto(metric_data_point_proto) == MetricDataPoint(
        metric_name="latency",
        dimensions={"status": "OK"},
        values={"avg": 150.5, "p99": 200},
    )
