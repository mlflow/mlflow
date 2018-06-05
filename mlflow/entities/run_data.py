from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.protos.service_pb2 import RunData as ProtoRunData


class RunData(_MLflowObject):
    def __init__(self, metrics=None, params=None):
        self._metrics = []
        self._params = []
        if metrics is not None:
            for m in metrics:
                self.add_metric(m)
        if params is not None:
            for p in params:
                self.add_param(p)

    @property
    def metrics(self):
        return self._metrics

    @property
    def params(self):
        return self._params

    def add_metric(self, metric):
        self._metrics.append(metric)

    def add_param(self, param):
        self._params.append(param)

    def to_proto(self):
        run_data = ProtoRunData()
        run_data.metrics.extend([m.to_proto() for m in self.metrics])
        run_data.params.extend([p.to_proto() for p in self.params])
        return run_data

    @classmethod
    def from_proto(cls, proto):
        run_data = cls()
        # iterate proto and add metrics and params
        for proto_metric in proto.metrics:
            run_data.add_metric(Metric.from_proto(proto_metric))
        for proto_param in proto.params:
            run_data.add_param(Param.from_proto(proto_param))

        return run_data

    @classmethod
    def from_dictionary(cls, the_dict):
        run_data = cls()
        for p in the_dict.get("metrics", []):
            run_data.add_metric(p)
        for p in the_dict.get("params", []):
            run_data.add_param(p)
        return run_data

    @classmethod
    def _properties(cls):
        # TODO: Hard coding this list of props for now. There has to be a clearer way...
        return ["metrics", "params"]
