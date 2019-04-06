from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.protos.service_pb2 import RunData as ProtoRunData, Param as ProtoParam,\
    RunTag as ProtoRunTag, Metric as ProtoMetric


class RunData(_MLflowObject):
    """
    Run data (metrics and parameters).
    """
    def __init__(self, metrics=None, params=None, tags=None):
        """
        Construct a new :py:class:`mlflow.entities.RunData` instance.
        :param metrics: List of :py:class:`mlflow.entities.Metric`.
        :param params: List of :py:class:`mlflow.entities.Param`.
        :param tags: List of :py:class:`mlflow.entities.RunTag`.
        """
        self._metrics = {metric.key: metric.value for metric in metrics}
        self._metric_objs = metrics or []
        self._params = {param.key: param.value for param in params}
        self._tags = {tag.key: tag.value for tag in tags}

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
        self._metrics[metric.key] = metric.value
        self._metric_objs.append(metric)


    def _add_param(self, param):
        self._params[param.key] = param.value

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    def to_proto(self):
        run_data = ProtoRunData()
        run_data.metrics.extend([m.to_proto() for m in self._metric_objs])
        run_data.params.extend([ProtoParam(key=key, value=val) for key, val in self.params.items()])
        run_data.tags.extend([ProtoRunTag(key=key, value=val) for key, val in self.tags.items()])
        return run_data

    def to_dictionary(self):
        return {
            "metrics": self.metrics,
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
            # Handle dicts since the "metrics" dictionary might either be a map of metric keys
            # to Metric objects (generated via dict(run_data)) or a map of metric keys to dicts
            # representing metric objects (generated via run_data.to_dictionary())
            if isinstance(m_obj, dict):
                m_obj = Metric(m_obj['key'], m_obj['value'], m_obj['timestamp'])
            run_data._add_metric(m_obj)
        run_data._params = the_dict.get("params", {})
        run_data._tags = the_dict.get("tags", [])
        return run_data
