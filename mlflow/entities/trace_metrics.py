from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos import service_pb2 as pb


class MetricsViewType(str, Enum):
    TRACES = "TRACES"
    SPANS = "SPANS"
    ASSESSMENTS = "ASSESSMENTS"

    def __str__(self) -> str:
        return self.value

    def to_proto(self):
        return pb.MetricsViewType.Value(self)


class AggregationType(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    PERCENTILE = "PERCENTILE"
    MIN = "MIN"
    MAX = "MAX"

    def __str__(self) -> str:
        return self.value

    def to_proto(self):
        return pb.AggregationType.Value(self)


@dataclass
class MetricsAggregation(_MlflowObject):
    aggregation_type: AggregationType
    percentile_value: float | None = None

    def __post_init__(self):
        if self.aggregation_type == AggregationType.PERCENTILE:
            if self.percentile_value is None:
                raise ValueError("Percentile value is required for PERCENTILE aggregation")
            if self.percentile_value > 100 or self.percentile_value < 0:
                raise ValueError(
                    f"Percentile value must be between 0 and 100, got {self.percentile_value}"
                )
        elif self.percentile_value is not None:
            raise ValueError(
                "Percentile value is only allowed for PERCENTILE aggregation type, "
                f"got {self.aggregation_type}"
            )

    def to_proto(self) -> pb.MetricsAggregation:
        proto = pb.MetricsAggregation()
        proto.aggregation_type = self.aggregation_type.to_proto()
        if self.percentile_value is not None:
            proto.percentile_value = self.percentile_value
        return proto


@dataclass
class MetricDataPoint(_MlflowObject):
    metric_name: str
    dimensions: dict[str, str]
    values: dict[str, float]

    @classmethod
    def from_proto(cls, proto: pb.MetricDataPoint) -> "MetricDataPoint":
        return cls(
            metric_name=proto.metric_name,
            dimensions=dict(proto.dimensions),
            values=dict(proto.values),
        )
