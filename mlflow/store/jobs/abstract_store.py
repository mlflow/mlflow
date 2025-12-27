from abc import ABC, abstractmethod
from typing import Any, Iterator

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.utils.annotations import developer_stable


@developer_stable
class AbstractJobStore(ABC):
    """
    Abstract class that defines API interfaces for storing Job metadata.
    """

    @abstractmethod
    def create_job(self, job_name: str, params: str, timeout: float | None = None) -> Job:
        """
        Create a new job with the specified function and parameters.

        Args:
            job_name: The static job name that identifies the decorated job function
            params: The job parameters that are serialized as a JSON string
            timeout: The job execution timeout in seconds

        Returns:
            Job entity instance
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
    def mark_job_timed_out(self, job_id: str) -> None:
        """
        Set a job status to Timeout.

        Args:
            job_id: The ID of the job
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
        increment the retry_count and reset the job to PENDING status,
        otherwise set the job to FAILED status and fill the job's error field.

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
        job_name: str | None = None,
        statuses: list[JobStatus] | None = None,
        begin_timestamp: int | None = None,
        end_timestamp: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[Job]:
        """
        List jobs based on the provided filters.

        Args:
            job_name: Filter by job name (exact match)
            statuses: Filter by a list of job status (PENDING, RUNNING, DONE, FAILED, TIMEOUT)
            begin_timestamp: Filter jobs created after this timestamp (inclusive)
            end_timestamp: Filter jobs created before this timestamp (inclusive)
            params: Filter jobs by matching job params dict with the provided params dict
                e.g., if `params` is ``{'a': 3, 'b': 4}``, it can match the following job params:
                ``{'a': 3, 'b': 4}``, ``{'a': 3, 'b': 4, 'c': 5}``, but it does not match the
                following job params: ``{'a': 3, 'b': 6}``, ``{'a': 3, 'c': 5}``.

        Returns:
            Iterator of Job entities that match the filters, ordered by creation time (oldest first)
        """

    @abstractmethod
    def get_job(self, job_id: str) -> Job:
        """
        Get a job by its ID.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            Job entity

        Raises:
            MlflowException: If job with the given ID is not found
        """

    @abstractmethod
    def cancel_job(self, job_id: str) -> Job:
        """
        Cancel a job by its ID.

        Args:
            job_id: The ID of the job to cancel

        Returns:
            Job entity

        Raises:
            MlflowException: If job with the given ID is not found
        """

    @abstractmethod
    def delete_jobs(self, older_than: int = 0, job_ids: list[str] | None = None) -> list[str]:
        """
        Delete finalized jobs based on the provided filters. Used by ``mlflow gc``.

        Only jobs with finalized status (SUCCEEDED, FAILED, TIMEOUT, CANCELED) are
        eligible for deletion.

        Behavior:
            - No filters: Deletes all finalized jobs.
            - Only ``older_than``: Deletes finalized jobs older than the threshold.
            - Only ``job_ids``: Deletes only the specified finalized jobs.
            - Both filters: Deletes finalized jobs matching both conditions.

        Args:
            older_than: Time threshold in milliseconds. Jobs with creation_time
                older than (current_time - older_than) are eligible for deletion.
                A value of 0 disables this filter.
            job_ids: List of specific job IDs to delete. If None, all finalized jobs
                (subject to older_than filter) are eligible for deletion.

        Returns:
            List of job IDs that were deleted.
        """
