from enum import Enum

from mlflow.exceptions import MlflowException


class JobStatus(str, Enum):
    """Enum for status of a Job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_int(cls, status_int: int) -> "JobStatus":
        """Convert integer status to JobStatus enum."""
        if status_int == 0:
            return cls.PENDING
        elif status_int == 1:
            return cls.RUNNING
        elif status_int == 2:
            return cls.DONE
        elif status_int == 3:
            return cls.FAILED

        raise MlflowException.invalid_parameter_value(f"Unknown job status: {status_int}")

    def to_int(self) -> int:
        """Convert JobStatus enum to integer."""
        if self == JobStatus.PENDING:
            return 0
        elif self == JobStatus.RUNNING:
            return 1
        elif self == JobStatus.DONE:
            return 2
        elif self == JobStatus.FAILED:
            return 3

        raise MlflowException.invalid_parameter_value(f"Unknown job status: {self}")

    @staticmethod
    def is_finalized(status: "JobStatus") -> bool:
        """
        Determines whether or not a JobStatus is a finalized status.
        A finalized status indicates that no further status updates will occur.
        """
        return status in [JobStatus.DONE, JobStatus.FAILED]
