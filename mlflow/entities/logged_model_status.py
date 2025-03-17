from enum import Enum

from mlflow.exceptions import MlflowException
from mlflow.protos import service_pb2 as pb2


class LoggedModelStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.LoggedModel`."""

    UNSPECIFIED = "UNSPECIFIED"
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"

    def __str__(self):
        return self.value

    def to_proto(self):
        if self == LoggedModelStatus.UNSPECIFIED:
            return pb2.LoggedModelStatus.LOGGED_MODEL_STATUS_UNSPECIFIED
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

    @classmethod
    def from_int(self, status_int: int) -> "LoggedModelStatus":
        if status_int == 0:
            return LoggedModelStatus.UNSPECIFIED
        elif status_int == 1:
            return LoggedModelStatus.PENDING
        elif status_int == 2:
            return LoggedModelStatus.READY
        elif status_int == 3:
            return LoggedModelStatus.FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {status_int}")

    def to_int(self) -> int:
        if self == LoggedModelStatus.UNSPECIFIED:
            return 0
        elif self == LoggedModelStatus.PENDING:
            return 1
        elif self == LoggedModelStatus.READY:
            return 2
        elif self == LoggedModelStatus.FAILED:
            return 3

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {self}")
