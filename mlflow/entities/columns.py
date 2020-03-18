from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import ListAllColumns


class Columns(_MLflowObject):
    """
    Columns object.
    """

    def __init__(self, metrics, params, tags):
        super(Columns, self).__init__()
        self._metrics = metrics
        self._params = params
        self._tags = tags

    @property
    def metrics(self):
        return self._metrics

    @property
    def params(self):
        return self._params

    @property
    def tags(self):
        return self._tags

    @classmethod
    def from_proto(cls, proto):
        columns = cls(proto.metrics,
                      proto.params,
                      proto.tags)
        return columns

    def to_proto(self):
        columns = ListAllColumns.Response()
        columns.metrics.extend(self.metrics)
        columns.params.extend(self.params)
        columns.tags.extend(self.tags)
        return columns
