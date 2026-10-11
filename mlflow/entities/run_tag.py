from __future__ import annotations

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag


class RunTag(_MlflowObject):
    """Tag object associated with a run."""

    def __init__(self, key: str, value: str) -> None:
        self._key = key
        self._value = value

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            # TODO deep equality here?
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self) -> str:
        """String name of the tag."""
        return self._key

    @property
    def value(self) -> str:
        """String value of the tag."""
        return self._value

    def to_proto(self) -> ProtoRunTag:
        param = ProtoRunTag()
        param.key = self.key
        param.value = self.value
        return param

    @classmethod
    def from_proto(cls, proto: ProtoRunTag) -> RunTag:
        return cls(proto.key, proto.value)
