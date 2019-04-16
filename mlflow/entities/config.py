from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Config as ProtoConfig


class Config(_MLflowObject):
    """
    Config object.
    """

    def __init__(self, key, value):
        self._key = key
        self._value = value

    @property
    def key(self):
        """String key corresponding to the parameter name."""
        return self._key

    @property
    def value(self):
        """String value of the parameter."""
        return self._value

    def to_proto(self):
        config = ProtoConfig()
        config.key = self.key
        config.value = self.value
        return config

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value)
