from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos import service_pb2 as pb2


class LoggedModelTag(_MlflowObject):
    """Tag object associated with a Model."""

    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __eq__(self, other):
        if type(other) is type(self):
            # TODO deep equality here?
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
        return pb2.LoggedModelTag(key=self._key, value=self._value)

    @classmethod
    def from_proto(cls, proto):
        return cls(key=proto.key, value=proto.value)
