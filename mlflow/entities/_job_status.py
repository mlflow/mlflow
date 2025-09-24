from enum import Enum


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
        return JobStatus(status_int)

    def to_int(self) -> int:
        """Convert JobStatus enum to integer."""
        return self.value

    def __str__(self):
        # `super().__str__()` returns value like `JobStatus.PENDING`
        # strip the redundant `JobStatus.` prefix.
        return super().__str__().split(".")[1]

    @staticmethod
    def is_finalized(status: "JobStatus") -> bool:
        """
        Determines whether or not a JobStatus is a finalized status.
        A finalized status indicates that no further status updates will occur.
        """
        return status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.TIMEOUT]
