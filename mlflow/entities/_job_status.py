from enum import Enum

from mlflow.exceptions import MlflowException


class JobStatus(str, Enum):
    """Enum for status of a Job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def from_int(cls, status_int: int) -> "JobStatus":
        """Convert integer status to JobStatus enum."""
        try:
            return next(e for i, e in enumerate(JobStatus) if i == status_int)
        except StopIteration:
            raise MlflowException.invalid_parameter_value(
                f"The value {status_int} can't be converted to JobStatus enum value."
            )

    @classmethod
    def from_str(cls, status_str: str) -> "JobStatus":
        """Convert string status to JobStatus enum."""
        try:
            return JobStatus[status_str]
        except KeyError:
            raise MlflowException.invalid_parameter_value(
                f"The string '{status_str}' can't be converted to JobStatus enum value."
            )

    def to_int(self) -> int:
        """Convert JobStatus enum to integer."""
        return next(i for i, e in enumerate(JobStatus) if e == self)

    def __str__(self):
        return self.name

    @staticmethod
    def is_finalized(status: "JobStatus") -> bool:
        """
        Determines whether or not a JobStatus is a finalized status.
        A finalized status indicates that no further status updates will occur.
        """
        return status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.TIMEOUT]
