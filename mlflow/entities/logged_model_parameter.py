from __future__ import annotations

import sys

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import LoggedModelParameter as ProtoLoggedModelParameter


class LoggedModelParameter(_MlflowObject):
    """
    MLflow entity representing a parameter of a Model.
    """

    def __init__(self, key: str, value: str) -> None:
        if "pyspark.ml" in sys.modules:
            import pyspark.ml.param

            if isinstance(key, pyspark.ml.param.Param):
                key = key.name
                value = str(value)
        self._key = key
        self._value = value

    @property
    def key(self) -> str:
        """String key corresponding to the parameter name."""
        return self._key

    @property
    def value(self) -> str:
        """String value of the parameter."""
        return self._value

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self._key == __o._key

        return False

    def __hash__(self) -> int:
        return hash(self._key)

    def to_proto(self) -> ProtoLoggedModelParameter:
        return ProtoLoggedModelParameter(key=self._key, value=self._value)

    @classmethod
    def from_proto(cls, proto: ProtoLoggedModelParameter) -> LoggedModelParameter:
        return cls(key=proto.key, value=proto.value)
