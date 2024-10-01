from enum import Enum

from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import LoggedModelStatus


class ModelStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.Model`."""

    UNSPECIFIED = "UNSPECIFIED"
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"

    def to_proto(self):
        if self == ModelStatus.UNSPECIFIED:
            return LoggedModelStatus.LOGGED_MODEL_STATUS_UNSPECIFIED
        elif self == ModelStatus.PENDING:
            return LoggedModelStatus.LOGGED_MODEL_PENDING
        elif self == ModelStatus.READY:
            return LoggedModelStatus.LOGGED_MODEL_READY
        elif self == ModelStatus.FAILED:
            return LoggedModelStatus.LOGGED_MODEL_UPLOAD_FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {self}")

    @classmethod
    def from_proto(cls, proto):
        if proto == LoggedModelStatus.LOGGED_MODEL_STATUS_UNSPECIFIED:
            return ModelStatus.UNSPECIFIED
        elif proto == LoggedModelStatus.LOGGED_MODEL_PENDING:
            return ModelStatus.PENDING
        elif proto == LoggedModelStatus.LOGGED_MODEL_READY:
            return ModelStatus.READY
        elif proto == LoggedModelStatus.LOGGED_MODEL_FAILED:
            return ModelStatus.FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {proto}")
