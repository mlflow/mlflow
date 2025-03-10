import sys

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos import service_pb2 as pb2


class LoggedModelParameter(_MlflowObject):
    """
    MLflow entity representing a parameter of a Model.
    """

    def __init__(self, key, value):
        if "pyspark.ml" in sys.modules:
            import pyspark.ml.param

            if isinstance(key, pyspark.ml.param.Param):
                key = key.name
                value = str(value)
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

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self._key == __o._key

        return False

    def __hash__(self):
        return hash(self._key)

    def to_proto(self):
        return pb2.LoggedModelParameter(key=self._key, value=self._value)

    @classmethod
    def from_proto(cls, proto):
        return cls(key=proto.key, value=proto.value)
