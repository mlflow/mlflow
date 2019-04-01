from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.protos.service_pb2 import RunData as ProtoRunData, Param as ProtoParam,\
    RunTag as ProtoRunTag


class RunData(_MLflowObject):
    """
    Run data (metrics and parameters).
    """
    def __init__(self, metrics=None, params=None, tags=None):
        """
        Construct a new :py:class:`mlflow.entities.RunData` instance.
        :param metrics: Dictionary of metric key (string) to :py:class:`mlflow.entities.Metric`.
        :param params: Dictionary of param key (string) to param value (string).
        :param tags: Dictionary of tag key (string) to tag value (string).
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
        run_data.metrics.extend([m.to_proto() for m in self.metrics.values()])
        run_data.params.extend([ProtoParam(key=key, value=val) for key, val in self.params.items()])
        run_data.tags.extend([ProtoRunTag(key=key, value=val) for key, val in self.tags.items()])
        return run_data

    def to_dictionary(self):
        return {
            "metrics": {key: dict(metric) for key, metric in self.metrics.items()},
            "params": self.params,
            "tags": self.tags,
        }

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
        for m_obj in the_dict.get("metrics", {}).values():
            run_data._add_metric(m_obj)
        run_data._params = the_dict.get("params", {})
        run_data._tags = the_dict.get("tags", [])
        return run_data
