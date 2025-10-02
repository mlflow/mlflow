from enum import Enum

from mlflow.exceptions import MlflowException


class JobStatus(Enum):
    """Enum for status of a Job."""

    PENDING = 0
    RUNNING = 1
    SUCCEEDED = 2
    FAILED = 3
    TIMEOUT = 4

    @classmethod
    def from_int(cls, status_int: int) -> "JobStatus":
        """Convert integer status to JobStatus enum."""
        try:
            return JobStatus(status_int)
        except ValueError as e:
            raise MlflowException.invalid_parameter_value(str(e))

    @classmethod
    def from_str(cls, status_str: str) -> "JobStatus":
        """Convert string status to JobStatus enum."""
        try:
            return JobStatus[status_str]
        except ValueError as e:
            raise MlflowException.invalid_parameter_value(str(e))

    def to_int(self) -> int:
        """Convert JobStatus enum to integer."""
        return self.value

    def __str__(self):
        return self.name

    @staticmethod
    def is_finalized(status: "JobStatus") -> bool:
        """
        Determines whether or not a JobStatus is a finalized status.
        A finalized status indicates that no further status updates will occur.
        """
        return status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.TIMEOUT]
