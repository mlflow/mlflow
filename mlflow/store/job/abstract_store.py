from abc import ABCMeta, abstractmethod

from mlflow.entities._job import JobStatus
from mlflow.utils.annotations import developer_stable


@developer_stable
class AbstractJobStore:
    """
    Abstract class that defines API interfaces for storing Job metadata.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Empty constructor. This is deliberately not marked as abstract, else every derived class
        would be forced to create one.
        """

    @abstractmethod
    def create_job(self, function: str, params: str) -> str:
        """
        Create a new job with the specified function and parameters.

        Args:
            function: The function name to execute
            params: The job parameters as a string

        Returns:
            The job ID (UUID4 string)
        """

    @abstractmethod
    def start_job(self, job_id: str) -> None:
        """
        Start a job by setting its status to RUNNING.

        Args:
            job_id: The ID of the job to start
        """

    @abstractmethod
    def reset_job(self, job_id: str) -> None:
        """
        Reset a job by setting its status to PENDING.

        Args:
            job_id: The ID of the job to re-enqueue.
        """

    @abstractmethod
    def finish_job(self, job_id: str, result: str) -> None:
        """
        Finish a job by setting its status to DONE and setting the result.

        Args:
            job_id: The ID of the job to finish
            result: The job result as a string
        """

    @abstractmethod
    def fail_job(self, job_id: str, error: str) -> None:
        """
        Fail a job by setting its status to FAILED and setting the error message.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string
        """

    @abstractmethod
    def retry_or_fail_job(self, job_id: str, error: str) -> int | None:
        """
        If the job retry_count is less than maximum allowed retry count,
        increase the retry_count and reset the job to PENDING status,
        otherwise set the job to FAIL status and fill the job's error field.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string

        Returns:
            If the job is allowed to retry, returns the retry count,
            otherwise returns None.
        """

    @abstractmethod
    def list_jobs(
        self,
        function: str | None = None,
        status: JobStatus | None = None,
        begin_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ):
        """
        List jobs based on the provided filters.

        Args:
            function: Filter by function name (exact match)
            status: Filter by job status (PENDING, RUNNING, DONE, FAILED)
            begin_timestamp: Filter jobs created after this timestamp (inclusive)
            end_timestamp: Filter jobs created before this timestamp (inclusive)

        Returns:
            List of Job entities that match the filters, order by creation time (newest first)
        """

    @abstractmethod
    def get_job(self, job_id: str):
        """
        Get a job by its ID.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            Job entity

        Raises:
            MlflowException: If job with the given ID is not found
        """
