from enum import Enum

from mlflow.exceptions import MlflowException
from mlflow.protos.jobs_pb2 import JobStatus as ProtoJobStatus


class JobStatus(str, Enum):
    """Enum for status of a Job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    NEEDS_RECOVERY = "NEEDS_RECOVERY"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELED = "CANCELED"

    @classmethod
    def from_int(cls, status_int: int) -> "JobStatus":
        """Convert integer status to JobStatus enum."""
        mapping = {
            0: JobStatus.PENDING,
            1: JobStatus.RUNNING,
            2: JobStatus.SUCCEEDED,
            3: JobStatus.FAILED,
            4: JobStatus.TIMEOUT,
            5: JobStatus.CANCELED,
            6: JobStatus.NEEDS_RECOVERY,
        }
        if status := mapping.get(status_int):
            return status
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
        return {
            JobStatus.PENDING: 0,
            JobStatus.RUNNING: 1,
            JobStatus.SUCCEEDED: 2,
            JobStatus.FAILED: 3,
            JobStatus.TIMEOUT: 4,
            JobStatus.CANCELED: 5,
            JobStatus.NEEDS_RECOVERY: 6,
        }[self]

    def to_proto(self) -> int:
        """Convert JobStatus enum to proto JobStatus enum value."""
        mapping = {
            JobStatus.PENDING: ProtoJobStatus.JOB_STATUS_PENDING,
            JobStatus.RUNNING: ProtoJobStatus.JOB_STATUS_IN_PROGRESS,
            JobStatus.NEEDS_RECOVERY: ProtoJobStatus.JOB_STATUS_NEEDS_RECOVERY,
            JobStatus.SUCCEEDED: ProtoJobStatus.JOB_STATUS_COMPLETED,
            JobStatus.FAILED: ProtoJobStatus.JOB_STATUS_FAILED,
            JobStatus.TIMEOUT: ProtoJobStatus.JOB_STATUS_FAILED,  # No TIMEOUT in proto
            JobStatus.CANCELED: ProtoJobStatus.JOB_STATUS_CANCELED,
        }
        return mapping.get(self, ProtoJobStatus.JOB_STATUS_UNSPECIFIED)

    def __str__(self):
        return self.name

    @staticmethod
    def is_finalized(status: "JobStatus") -> bool:
        """
        Determines whether or not a JobStatus is a finalized status.
        A finalized status indicates that no further status updates will occur.
        """
        return status in [
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.TIMEOUT,
            JobStatus.CANCELED,
        ]
