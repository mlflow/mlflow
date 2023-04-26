from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import InputTag as ProtoInputTag
from mlflow.utils.annotations import experimental


@experimental
class InputTag(_MLflowObject):
    """Input tag object associated with a dataset."""

    def __init__(self, key: str, value: str) -> None:
        self._key = key
        self._value = value

    def __eq__(self, other: _MLflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self) -> str:
        """String name of the input tag."""
        return self._key

    @property
    def value(self) -> str:
        """String value of the input tag."""
        return self._value

    def to_proto(self):
        tag = ProtoInputTag()
        tag.key = self.key
        tag.value = self.value
        return tag

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value)
