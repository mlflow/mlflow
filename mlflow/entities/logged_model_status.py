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

    @staticmethod
    def is_finalized(status) -> bool:
        """
        Determines whether or not a LoggedModelStatus is a finalized status.
        A finalized status indicates that no further status updates will occur.
        """
        return status in [LoggedModelStatus.READY, LoggedModelStatus.FAILED]

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
        elif proto == pb2.LoggedModelStatus.LOGGED_MODEL_UPLOAD_FAILED:
            return LoggedModelStatus.FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown model status: {proto}")

    @classmethod
    def from_int(cls, status_int: int) -> "LoggedModelStatus":
        if status_int == 0:
            return cls.UNSPECIFIED
        elif status_int == 1:
            return cls.PENDING
        elif status_int == 2:
            return cls.READY
        elif status_int == 3:
            return cls.FAILED

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
