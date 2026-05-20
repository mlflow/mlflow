from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterator

from mlflow.entities._job import Job, JobProgress
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import developer_stable


class JobTerminalStateUpdateException(MlflowException):
    """Raised when attempting to update a job that is already terminal."""

    def __init__(self, job_id: str, status: str):
        super().__init__(
            f"The Job {job_id} is already finalized with status: {status}, it can't be updated.",
            error_code=INVALID_PARAMETER_VALUE,
        )


class JobUpdateStatus(str, Enum):
    """Outcome of a conditional job state transition."""

    APPLIED = "APPLIED"
    WRONG_STATE = "WRONG_STATE"


@developer_stable
class AbstractJobStore(ABC):
    """
    Abstract class that defines API interfaces for storing Job metadata.
    """

    @property
    def supports_workspaces(self) -> bool:
        """Return whether workspaces are supported by this job store."""
        return False

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
    def claim_job(self, job_id: str, lease_duration: float | None = None) -> JobUpdateStatus:
        """
        Conditionally claim a pending job for execution.

        Args:
            job_id: The ID of the job to claim
            lease_duration: Optional lease duration in seconds. When provided,
                the store should persist the corresponding lease expiry while
                transitioning the row to ``RUNNING``.

        Returns:
            ``APPLIED`` if the job transitioned from ``PENDING`` to
            ``RUNNING``, ``WRONG_STATE`` if the row exists but was no longer
            claimable
        """

    @abstractmethod
    def renew_job_lease(self, job_id: str, lease_duration: float) -> JobUpdateStatus:
        """
        Renew the short-lived lease for a running job.

        Args:
            job_id: The ID of the job whose lease should be renewed
            lease_duration: New lease duration in seconds, measured from the
                time the renewal is persisted.

        Returns:
            ``APPLIED`` if the lease was renewed, ``WRONG_STATE`` if the row
            exists but is no longer in a renewable state
        """

    @abstractmethod
    def retry_job(self, job_id: str) -> int:
        """
        Transition a running job back to ``PENDING`` for another attempt.

        Args:
            job_id: The ID of the job to retry

        Returns:
            The incremented retry count
        """

    @abstractmethod
    def mark_job_needs_recovery(self, job_id: str) -> JobUpdateStatus:
        """
        Transition a stale running job to ``NEEDS_RECOVERY``.

        Args:
            job_id: The ID of the job to mark for recovery

        Returns:
            ``APPLIED`` if the transition succeeded, ``WRONG_STATE`` if the
            row exists but is no longer in a recoverable running state
        """

    @abstractmethod
    def reattach_job(self, job_id: str, lease_duration: float | None = None) -> JobUpdateStatus:
        """
        Transition a recovery-owned job back to ``RUNNING``.

        Args:
            job_id: The ID of the job to reattach to active monitoring
            lease_duration: Optional new lease duration in seconds to apply
                when the job returns to ``RUNNING``.

        Returns:
            ``APPLIED`` if the transition succeeded, ``WRONG_STATE`` if the
            row exists but is no longer in ``NEEDS_RECOVERY``
        """

    @abstractmethod
    def requeue_job(self, job_id: str) -> JobUpdateStatus:
        """
        Transition a recovery-owned job back to ``PENDING`` without incrementing retries.

        Args:
            job_id: The ID of the job to requeue

        Returns:
            ``APPLIED`` if the transition succeeded, ``WRONG_STATE`` if the
            row exists but is no longer in ``NEEDS_RECOVERY``
        """

    @abstractmethod
    def report_job_result(
        self,
        job_id: str,
        status: JobStatus,
        result: str | None = None,
        error_message: str | None = None,
        is_transient_error: bool = False,
    ) -> None:
        """
        Persist a terminal job outcome for framework-owned execution.

        Args:
            job_id: The ID of the job to terminalize
            status: Terminal status to persist
            result: Serialized successful result payload, when applicable
            error_message: Human-readable terminal error payload, when
                applicable.
            is_transient_error: Executor-side retry classification. Stores do
                not persist this flag today, but callers may use this signature
                to match the job-result contract.
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
    def update_status_details(self, job_id: str, status_details: dict[str, Any]) -> None:
        """
        Update job status details.

        Merges the provided status details with existing job status details.

        Args:
            job_id: The ID of the job to update
            status_details: Status details to merge into existing job status details
        """

    def update_job_progress(
        self,
        job_id: str,
        message: str | None = None,
        progress: JobProgress | None = None,
    ) -> None:
        """
        Update structured job progress fields.

        This is intentionally separate from ``update_status_details()`` so the
        legacy free-form status-details channel can evolve independently from the
        structured progress contract.

        Stores that support structured job progress should override this method.

        Args:
            job_id: The ID of the job to update
            message: Human-readable plain-text progress message. ``None`` leaves
                the existing value unchanged.
            progress: Structured machine-readable progress payload. ``None``
                leaves the existing value unchanged.
        """
        # Progress reporting is optional for job stores. Stores that support it
        # should override this method; others may treat progress updates as a no-op.
        return None

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
