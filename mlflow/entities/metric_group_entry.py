from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import MetricGroupEntry as ProtoMetricGroupEntry


class MetricGroupEntry(_MLflowObject):
    """
    Metric group entry object.
    """

    def __init__(self, params, values, timestamp):
        self._params = params
        self._values = values
        self._timestamp = timestamp

    @property
    def params(self):
        """List of string param values for the current entry."""
        return self._params

    @property
    def values(self):
        """List of float metric values for the current entry."""
        return self._values

    @property
    def timestamp(self):
        """Metric group entry timestamp as an integer (milliseconds since the Unix epoch)."""
        return self._timestamp

    def to_proto(self):
        metric_group_entry = ProtoMetricGroupEntry()
        metric_group_entry.params.extend(self.params)
        metric_group_entry.values.extend(self.values)
        metric_group_entry.timestamp = self.timestamp
        return metric_group_entry

    @classmethod
    def from_proto(cls, proto):
        metric_group_entry = cls(
            proto.params,
            proto.values,
            proto.timestamp
        )
        return metric_group_entry

    @classmethod
    def _properties(cls):
        # TODO: Hard coding this list of props for now. There has to be a clearer way...
        return ["params", "values", "timestamp"]
