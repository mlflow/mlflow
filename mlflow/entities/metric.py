from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
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

    def to_dictionary(self):
        """
        Convert the Metric object to a dictionary.

        Returns:
            dict: The Metric object represented as a dictionary.
        """
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "step": self.step,
        }

    @classmethod
    def from_dictionary(cls, metric_dict):
        """
        Create a Metric object from a dictionary.

        Args:
            metric_dict (dict): Dictionary containing metric information.

        Returns:
            Metric: The Metric object created from the dictionary.
        """
        required_keys = ["key", "value", "timestamp", "step"]
        missing_keys = [key for key in required_keys if key not in metric_dict]
        if missing_keys:
            raise MlflowException(
                f"Missing required keys {missing_keys} in metric dictionary",
                INVALID_PARAMETER_VALUE,
            )

        return cls(**metric_dict)


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
