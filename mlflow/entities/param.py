from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Param as ProtoParam


class Param(_MLflowObject):
    """
    Param object for python client. Backend stores will hydrate this object in APIs.
    """

    def __init__(self, key, value):
        self._key = key
        self._value = value

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    def to_proto(self):
        param = ProtoParam()
        param.key = self.key
        param.value = self.value
        return param

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value)

    @classmethod
    def _properties(cls):
        # TODO: Hard coding this list of props for now. There has to be a clearer way...
        return ["key", "value"]
