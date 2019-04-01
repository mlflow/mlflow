from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.protos.service_pb2 import RunData as ProtoRunData


class RunData(_MLflowObject):
    """
    Run data (metrics and parameters).
    """
    def __init__(self, metrics=None, params=None, tags=None):
        """
        Construct a new :py:class:`mlflow.entities.RunData` instance.
        :param metrics: List of :py:class:`mlflow.entities.Metric`. We expect a ingl
        :param params:
        :param tags:
        """
        self._metrics = metrics or {}
        self._params = params or {}
        self._tags = tags or {}

    @property
    def metrics(self):
        """
        Dictionary of string key -> :py:class:`mlflow.entities.Metric` for the current run.
        For each metric key, the maximum metric value at the maximum timestamp is returned.
        """
        return self._metrics

    @property
    def params(self):
        """Dictionary of param key (string) -> param value (string) for the current run."""
        return self._params

    @property
    def tags(self):
        """Dictionary of tag key (string) -> tag value (string) for the current run."""
        return self._tags

    def _add_metric(self, metric):
        if isinstance(metric, dict):
            metric = Metric(metric['key'], metric['value'], metric['timestamp'])
        self._metrics[metric.key] = metric

    def _add_param(self, param):
        if isinstance(param, dict):
            param = Param(param['key'], param['value'])
        self._params[param.key] = param.value

    def _add_tag(self, tag):
        if isinstance(tag, dict):
            tag = RunTag(tag['key'], tag['value'])
        self._tags[tag.key] = tag.value

    def to_proto(self):
        run_data = ProtoRunData()
        run_data.metrics.extend([m.to_proto() for m in self.metrics])
        run_data.params.extend([p.to_proto() for p in self.params])
        run_data.tags.extend([t.to_proto() for t in self.tags])
        return run_data

    def to_dictionary(self):
        return {p: [dict(val) for val in getattr(self, p)] for p in RunData._properties()}

    @classmethod
    def from_proto(cls, proto):
        run_data = cls()
        # iterate proto and add metrics, params, and tags
        for proto_metric in proto.metrics:
            run_data._add_metric(Metric.from_proto(proto_metric))
        for proto_param in proto.params:
            run_data._add_param(Param.from_proto(proto_param))
        for proto_tag in proto.tags:
            run_data._add_tag(RunTag.from_proto(proto_tag))

        return run_data

    @classmethod
    def from_dictionary(cls, the_dict):
        run_data = cls()
        for p in the_dict.get("metrics", []):
            run_data._add_metric(p)
        for p in the_dict.get("params", []):
            run_data._add_param(p)
        for t in the_dict.get("tags", []):
            run_data._add_tag(t)
        return run_data
