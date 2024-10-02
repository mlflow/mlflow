from enum import Enum

from mlflow.exceptions import MlflowException
from mlflow.protos import service_pb2 as pb2


class LoggedModelStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.Model`."""

    UNSPECIFIED = "UNSPECIFIED"
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"

    def to_proto(self):
        if self == LoggedModelStatus.UNSPECIFIED:
            return pb2.pb2.LoggedModelStatus.LOGGED_MODEL_STATUS_UNSPECIFIED
        elif self == LoggedModelStatus.PENDING:
            return pb2.LoggedModelStatus.LOGGED_MODEL_PENDING
        elif self == LoggedModelStatus.READY:
            return pb2.LoggedModelStatus.LOGGED_MODEL_READY
        elif self == LoggedModelStatus.FAILED:
            return pb2.LoggedModelStatus.LOGGED_MODEL_UPLOAD_FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {self}")

    @classmethod
    def from_proto(cls, proto):
        if proto == pb2.LoggedModelStatus.LOGGED_MODEL_STATUS_UNSPECIFIED:
            return LoggedModelStatus.UNSPECIFIED
        elif proto == pb2.LoggedModelStatus.LOGGED_MODEL_PENDING:
            return LoggedModelStatus.PENDING
        elif proto == pb2.LoggedModelStatus.LOGGED_MODEL_READY:
            return LoggedModelStatus.READY
        elif proto == pb2.LoggedModelStatus.LOGGED_MODEL_FAILED:
            return LoggedModelStatus.FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {proto}")
