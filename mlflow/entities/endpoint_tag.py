from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import SecretRouteTag as ProtoSecretRouteTag


class EndpointTag(_MlflowObject):
    """Tag object associated with a secret endpoint."""

    def __init__(self, key, value):
        self._key = key
        self._value = value

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
        return ProtoSecretRouteTag(key=self.key, value=self.value)

    @classmethod
    def from_proto(cls, proto):
        return cls(key=proto.key, value=proto.value)
