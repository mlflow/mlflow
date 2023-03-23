from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Input as ProtoInput


class Input(_MLflowObject):
    """Tag object associated with an experiment."""

    def __init__(self, dataset_uuid, experiment_id, name, digest):
        self._dataset_uuid = dataset_uuid,
        self._experiment_id = experiment_id,
        self._name = name,
        self._digest = digest

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self):
        """String name of the tag."""
        return self._key

    @property
    def value(self):
        """String value of the tag."""
        return self._value

    def to_proto(self):
        dataset = ProtoDataset()
        param.key = self.key
        param.value = self.value
        return param

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value)
