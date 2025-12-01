from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos import service_pb2 as pb


class MetricsViewType(str, Enum):
    TRACES = "TRACES"
    SPANS = "SPANS"
    ASSESSMENTS = "ASSESSMENTS"

    def to_proto(self):
        return pb.MetricsViewType.Value(self)


class AggregationType(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    P50 = "P50"
    P75 = "P75"
    P90 = "P90"
    P95 = "P95"
    P99 = "P99"

    def to_proto(self):
        return pb.AggregationType.Value(self)


class TimeGranularity(str, Enum):
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"

    def to_proto(self):
        return pb.TimeGranularity.Value(self)


@dataclass
class MetricDataPoint(_MlflowObject):
    dimensions: dict[str, str]
    metric_name: str
    values: dict[str, str]

    @classmethod
    def from_proto(cls, proto: pb.MetricDataPoint) -> "MetricDataPoint":
        return cls(
            dimensions=dict(proto.dimensions),
            metric_name=proto.metric_name,
            values=dict(proto.values),
        )
