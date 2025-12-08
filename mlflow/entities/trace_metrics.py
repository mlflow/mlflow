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
    values: dict[str, str]

    @classmethod
    def from_proto(cls, proto: pb.MetricDataPoint) -> "MetricDataPoint":
        return cls(
            metric_name=proto.metric_name,
            dimensions=dict(proto.dimensions),
            values=dict(proto.values),
        )
