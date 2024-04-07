from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Metric as ProtoMetric
from mlflow.protos.service_pb2 import MetricWithRunId as ProtoMetricWithRunId


class Metric(_MlflowObject):
    """
    Metric object.
    """

    def __init__(self, key, value, timestamp, step):
        self._key = key
        self._value = value
        self._timestamp = timestamp
        self._step = step

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

    @property
    def step(self):
        """Integer metric step (x-coordinate)."""
        return self._step

    def to_proto(self):
        metric = ProtoMetric()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        metric.step = self.step
        return metric

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value, proto.timestamp, proto.step)

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__dict__ == __o.__dict__

        return False

    def __hash__(self):
        return hash((self._key, self._value, self._timestamp, self._step))


class MetricWithRunId(Metric):
    def __init__(self, metric: Metric, run_id):
        super().__init__(
            key=metric.key,
            value=metric.value,
            timestamp=metric.timestamp,
            step=metric.step,
        )
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "run_id": self.run_id,
        }

    def to_proto(self):
        metric = ProtoMetricWithRunId()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        metric.step = self.step
        metric.run_id = self.run_id
        return metric
