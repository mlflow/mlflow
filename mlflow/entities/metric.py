from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Metric as ProtoMetric


class Metric(_MLflowObject):
    """
    Metric object.
    """

    def __init__(self, key, value, timestamp):
        self._key = key
        self._value = value
        self._timestamp = timestamp

    @property
    def key(self):
        """String key corresponding to the metric name."""
        return self._key

    @property
    def value(self):
        """Float value of the metric."""
        return self._value

    @property
    def timestamp(self):
        """Metric timestamp as an integer (milliseconds since the Unix epoch)."""
        return self._timestamp

    def to_proto(self):
        metric = ProtoMetric()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        return metric

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value, proto.timestamp)
