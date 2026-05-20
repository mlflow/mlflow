import json
import logging
import math
import threading
import uuid
from typing import Any, Iterator, NoReturn

import sqlalchemy

from mlflow.entities._job import Job, JobProgress
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.db.utils import (
    _get_managed_session_maker,
    _safe_initialize_tables,
    create_sqlalchemy_engine_with_retry,
)
from mlflow.store.jobs.abstract_store import (
    AbstractJobStore,
    JobTerminalStateUpdateException,
    JobUpdateStatus,
)
from mlflow.store.tracking.dbmodels.models import SqlJob
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

sqlalchemy.orm.configure_mappers()

_LIST_JOB_PAGE_SIZE = 100
_logger = logging.getLogger(__name__)


class SqlAlchemyJobStore(AbstractJobStore):
    """
    SQLAlchemy compliant backend store for storing Job metadata.
    This store interacts with SQL store using SQLAlchemy abstractions defined
    for MLflow Job entities.
    """

    # Class-level cache for SQLAlchemy engines to prevent connection pool leaks
    # when multiple store instances are created with the same database URI.
    _engine_map: dict[str, sqlalchemy.engine.Engine] = {}
    _engine_map_lock = threading.Lock()

    @classmethod
    def _get_or_create_engine(cls, db_uri: str) -> sqlalchemy.engine.Engine:
        """Get a cached engine or create a new one for the given database URI."""
        if db_uri not in cls._engine_map:
            with cls._engine_map_lock:
                if db_uri not in cls._engine_map:
                    cls._engine_map[db_uri] = create_sqlalchemy_engine_with_retry(db_uri)
        return cls._engine_map[db_uri]

    def __init__(self, db_uri):
        """
        Create a database backed store.

        Args:
            db_uri: The SQLAlchemy database URI string to connect to the database.
        """
        super().__init__()
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = self._get_or_create_engine(db_uri)
        _safe_initialize_tables(self.engine)

        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = _get_managed_session_maker(SessionMaker, self.db_type)

    def _get_active_workspace(self) -> str:
        """
        Get the active workspace name.

        In single-tenant mode, always returns DEFAULT_WORKSPACE_NAME.
        Workspace-aware subclasses override this to enforce isolation.
        """
        return DEFAULT_WORKSPACE_NAME

    def _get_query(self, session, model):
        """
        Return a query for ``model``.
        Workspace-aware subclasses override this to enforce scoping.
        """
        return session.query(model)

    def _with_workspace_field(self, instance):
        """
        Allow subclasses to populate model fields (e.g., workspace metadata) on ORM instances.
        """
        if hasattr(instance, "workspace") and getattr(instance, "workspace", None) is None:
            instance.workspace = DEFAULT_WORKSPACE_NAME
        return instance

    @staticmethod
    def _lease_expiration_time(lease_duration: float | None) -> int | None:
        if lease_duration is None:
            return None
        if not math.isfinite(lease_duration) or lease_duration <= 0:
            raise MlflowException.invalid_parameter_value(
                "`lease_duration` must be a finite positive number."
            )
        return get_current_time_millis() + int(lease_duration * 1000)

    @staticmethod
    def _transient_field_update_values() -> dict[Any, Any]:
        # Runtime-only fields (lease, progress, auth tokens) cleared when a job
        # reaches a terminal or re-queued state to avoid retaining stale
        # metadata and reduce stored row size.
        return {
            SqlJob.lease_expires_at: None,
            SqlJob.status_message: None,
            SqlJob.progress: None,
            SqlJob.progress_updated_at: None,
            SqlJob.token_hash: None,
            SqlJob.scoped_permissions: None,
        }

    @classmethod
    def _terminal_update_values(
        cls, status: JobStatus, payload: str | None, update_time: int
    ) -> dict[Any, Any]:
        return {
            SqlJob.status: status.to_int(),
            SqlJob.result: payload,
            SqlJob.last_update_time: update_time,
            **cls._transient_field_update_values(),
        }

    @classmethod
    def _pending_update_values(
        cls, update_time: int, *, increment_retry: bool = False
    ) -> dict[Any, Any]:
        values: dict[Any, Any] = {
            SqlJob.status: JobStatus.PENDING.to_int(),
            SqlJob.result: None,
            SqlJob.last_update_time: update_time,
            **cls._transient_field_update_values(),
        }
        if increment_retry:
            values[SqlJob.retry_count] = SqlJob.retry_count + 1
        return values

    def _conditional_status_update(
        self,
        session,
        job_id: str,
        current_statuses: tuple[JobStatus, ...],
        values: dict[Any, Any],
        additional_filters: tuple[Any, ...] = (),
    ) -> JobUpdateStatus:
        rows_updated = (
            self
            ._get_query(session, SqlJob)
            .filter(
                SqlJob.id == job_id,
                SqlJob.status.in_([status.to_int() for status in current_statuses]),
                *additional_filters,
            )
            .update(values, synchronize_session=False)
        )
        if rows_updated > 0:
            return JobUpdateStatus.APPLIED

        # Distinguish "job not found" from "job exists but not in the
        # required status". Raises MlflowException if the job does not exist.
        exists = (
            self
            ._get_query(session, SqlJob)
            .filter(SqlJob.id == job_id)
            .with_entities(SqlJob.id)
            .first()
        )
        if exists is None:
            raise MlflowException(
                f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
            )
        return JobUpdateStatus.WRONG_STATE

    def _raise_invalid_transition(
        self,
        session,
        job_id: str,
        action: str,
        current_statuses: tuple[JobStatus, ...],
    ) -> NoReturn:
        job = self._get_sql_job(session, job_id, populate_existing=True)
        current_status = JobStatus.from_int(job.status)
        if JobStatus.is_finalized(current_status):
            raise MlflowException.invalid_parameter_value(
                "The Job "
                f"{job_id} is already finalized with status: {current_status}, "
                "it can't be updated."
            )
        expected_statuses = ", ".join(status.name for status in current_statuses)
        raise MlflowException.invalid_parameter_value(
            f"Job {job_id} is in {current_status} state, cannot {action} "
            f"(must be {expected_statuses})"
        )

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
        with self.ManagedSessionMaker(read_only=False) as session:
            job_id = str(uuid.uuid4())
            creation_time = get_current_time_millis()

            job = self._with_workspace_field(
                SqlJob(
                    id=job_id,
                    creation_time=creation_time,
                    job_name=job_name,
                    params=params,
                    timeout=timeout,
                    status=JobStatus.PENDING.to_int(),
                    result=None,
                    last_update_time=creation_time,
                )
            )

            session.add(job)
            session.flush()
            return job.to_mlflow_entity()

    def _try_claim_job(
        self, session, job_id: str, lease_duration: float | None = None
    ) -> JobUpdateStatus:
        update_time = get_current_time_millis()
        return self._conditional_status_update(
            session,
            job_id,
            (JobStatus.PENDING,),
            {
                SqlJob.status: JobStatus.RUNNING.to_int(),
                SqlJob.lease_expires_at: self._lease_expiration_time(lease_duration),
                SqlJob.last_update_time: update_time,
            },
        )

    def claim_job(self, job_id: str, lease_duration: float | None = None) -> JobUpdateStatus:
        with self.ManagedSessionMaker(read_only=False) as session:
            return self._try_claim_job(session, job_id, lease_duration)

    def renew_job_lease(self, job_id: str, lease_duration: float) -> JobUpdateStatus:
        if lease_duration is None:
            raise MlflowException.invalid_parameter_value(
                "`lease_duration` must be provided when renewing a job lease."
            )
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            return self._conditional_status_update(
                session,
                job_id,
                (JobStatus.RUNNING,),
                {
                    SqlJob.lease_expires_at: self._lease_expiration_time(lease_duration),
                    SqlJob.last_update_time: update_time,
                },
            )

    def retry_job(self, job_id: str) -> int:
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.RUNNING,),
                    self._pending_update_values(update_time, increment_retry=True),
                )
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(session, job_id, "retry", (JobStatus.RUNNING,))

            updated_job = self._get_sql_job(session, job_id, populate_existing=True)
            return updated_job.retry_count

    def mark_job_needs_recovery(self, job_id: str) -> JobUpdateStatus:
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            return self._conditional_status_update(
                session,
                job_id,
                (JobStatus.RUNNING,),
                {
                    SqlJob.status: JobStatus.NEEDS_RECOVERY.to_int(),
                    SqlJob.lease_expires_at: None,
                    SqlJob.last_update_time: update_time,
                },
            )

    def reattach_job(self, job_id: str, lease_duration: float | None = None) -> JobUpdateStatus:
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            return self._conditional_status_update(
                session,
                job_id,
                (JobStatus.NEEDS_RECOVERY,),
                {
                    SqlJob.status: JobStatus.RUNNING.to_int(),
                    SqlJob.lease_expires_at: self._lease_expiration_time(lease_duration),
                    SqlJob.last_update_time: update_time,
                },
            )

    def requeue_job(self, job_id: str) -> JobUpdateStatus:
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            return self._conditional_status_update(
                session,
                job_id,
                (JobStatus.NEEDS_RECOVERY,),
                self._pending_update_values(update_time),
            )

    def _try_report_job_result(
        self,
        session,
        job_id: str,
        status: JobStatus,
        payload: str | None,
    ) -> JobUpdateStatus:
        update_time = get_current_time_millis()
        return self._conditional_status_update(
            session,
            job_id,
            (JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
            self._terminal_update_values(status, payload, update_time),
        )

    def report_job_result(
        self,
        job_id: str,
        status: JobStatus,
        result: str | None = None,
        error_message: str | None = None,
        is_transient_error: bool = False,
    ) -> None:
        # Keep this parameter to match the job-result/API contract. Remote
        # executors send transient-error classification through the API layer,
        # but the SQL store does not persist it; higher layers use it to decide
        # whether the job should be retried.
        del is_transient_error

        if not JobStatus.is_finalized(status):
            raise MlflowException.invalid_parameter_value(
                "`report_job_result()` requires a terminal JobStatus."
            )

        if status == JobStatus.SUCCEEDED:
            if error_message is not None:
                raise MlflowException.invalid_parameter_value(
                    "Succeeded job results must leave `error_message` unset."
                )
            payload = result
        elif status == JobStatus.FAILED:
            if error_message is None or result is not None:
                raise MlflowException.invalid_parameter_value(
                    "Failed job results must set `error_message` and leave `result` unset."
                )
            payload = error_message
        elif status == JobStatus.TIMEOUT:
            if result is not None:
                raise MlflowException.invalid_parameter_value(
                    "Timed-out job results must leave `result` unset; `error_message` is optional."
                )
            payload = error_message
        elif status == JobStatus.CANCELED:
            if result is not None or error_message is not None:
                _logger.warning(
                    "Ignoring payload for canceled job result update for job_id=%s.",
                    job_id,
                )
            payload = None
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unsupported terminal JobStatus for result reporting: {status}"
            )

        with self.ManagedSessionMaker(read_only=False) as session:
            if (
                self._try_report_job_result(session, job_id, status, payload)
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(
                    session,
                    job_id,
                    "report result",
                    (JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                )

    def start_job(self, job_id: str) -> None:
        """
        Start a job by setting its status to RUNNING.
        Only succeeds if the job is currently in PENDING state.

        Args:
            job_id: The ID of the job to start

        Raises:
            MlflowException: If job is not in PENDING state or doesn't exist
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            if self._try_claim_job(session, job_id) == JobUpdateStatus.APPLIED:
                return
            self._raise_invalid_transition(session, job_id, "start", (JobStatus.PENDING,))

    def reset_job(self, job_id: str) -> None:
        """
        Reset a job by setting its status to PENDING.

        Args:
            job_id: The ID of the job to re-enqueue.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                    self._pending_update_values(update_time),
                )
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(
                    session,
                    job_id,
                    "reset",
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                )

    def finish_job(self, job_id: str, result: str) -> None:
        """
        Finish a job by setting its status to DONE and setting the result.

        Args:
            job_id: The ID of the job to finish
            result: The job result as a string
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                    self._terminal_update_values(JobStatus.SUCCEEDED, result, update_time),
                )
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(
                    session,
                    job_id,
                    "finish",
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                )

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Fail a job by setting its status to FAILED and setting the error message.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                    self._terminal_update_values(JobStatus.FAILED, error, update_time),
                )
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(
                    session,
                    job_id,
                    "fail",
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                )

    def mark_job_timed_out(self, job_id: str) -> None:
        """
        Set a job status to Timeout.

        Args:
            job_id: The ID of the job
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                    self._terminal_update_values(JobStatus.TIMEOUT, None, update_time),
                )
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(
                    session,
                    job_id,
                    "time out",
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                )

    def retry_or_fail_job(self, job_id: str, error: str) -> int | None:
        """
        Retry or fail a running job based on the configured retry budget.

        If the running job retry_count is less than the maximum allowed retry
        count, increment the retry_count and reset the job to PENDING status.
        Otherwise set the running job to FAILED status and fill the job's error
        field.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string

        Returns:
            If the job is allowed to retry, returns the retry count,
            otherwise returns None.

        Raises:
            MlflowException: If the job is not in RUNNING state, is already
                finalized, or does not exist.
        """
        from mlflow.environment_variables import MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES

        max_retries = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get()

        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.RUNNING,),
                    self._pending_update_values(update_time, increment_retry=True),
                    additional_filters=(SqlJob.retry_count < max_retries,),
                )
                == JobUpdateStatus.APPLIED
            ):
                updated_job = self._get_sql_job(session, job_id, populate_existing=True)
                return updated_job.retry_count

            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.RUNNING,),
                    self._terminal_update_values(JobStatus.FAILED, error, update_time),
                    additional_filters=(SqlJob.retry_count >= max_retries,),
                )
                == JobUpdateStatus.APPLIED
            ):
                return None

            self._raise_invalid_transition(session, job_id, "fail", (JobStatus.RUNNING,))

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
            params: Filter jobs by matching job params dict with the provided params dict.
                e.g., if `params` is ``{'a': 3, 'b': 4}``, it can match the following job params:
                ``{'a': 3, 'b': 4}``, ``{'a': 3, 'b': 4, 'c': 5}``, but it does not match the
                following job params: ``{'a': 3, 'b': 6}``, ``{'a': 3, 'c': 5}``.

        Returns:
            Iterator of Job entities that match the filters, ordered by creation time (oldest first)
        """
        offset = 0

        def filter_by_params(job_params: dict[str, Any]) -> bool:
            for key in params:
                if key in job_params:
                    if job_params[key] != params[key]:
                        return False
                else:
                    return False
            return True

        while True:
            with self.ManagedSessionMaker() as session:
                # Select all columns needed for Job entity
                query = self._get_query(session, SqlJob)

                # Apply filters
                if job_name is not None:
                    query = query.filter(SqlJob.job_name == job_name)

                if statuses:
                    query = query.filter(
                        SqlJob.status.in_([status.to_int() for status in statuses])
                    )

                if begin_timestamp is not None:
                    query = query.filter(SqlJob.creation_time >= begin_timestamp)

                if end_timestamp is not None:
                    query = query.filter(SqlJob.creation_time <= end_timestamp)

                # Order by creation time (oldest first) and apply pagination
                jobs = (
                    query
                    .order_by(SqlJob.creation_time)
                    .offset(offset)
                    .limit(_LIST_JOB_PAGE_SIZE)
                    .all()
                )

                # If no jobs returned, we've reached the end
                if not jobs:
                    break

                # Yield each job
                if params:
                    for job in jobs:
                        if filter_by_params(json.loads(job.params)):
                            yield job.to_mlflow_entity()
                else:
                    for job in jobs:
                        yield job.to_mlflow_entity()

                # If we got fewer jobs than page_size, we've reached the end
                if len(jobs) < _LIST_JOB_PAGE_SIZE:
                    break

                # Move to next page
                offset += _LIST_JOB_PAGE_SIZE

    def _get_sql_job(self, session, job_id, *, populate_existing: bool = False) -> SqlJob:
        query = self._get_query(session, SqlJob).filter(SqlJob.id == job_id)
        if populate_existing:
            query = query.populate_existing()
        job = query.one_or_none()
        if job is None:
            raise MlflowException(
                f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
            )
        return job

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
        with self.ManagedSessionMaker() as session:
            job = self._get_sql_job(session, job_id)
            return job.to_mlflow_entity()

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
        current_time = get_current_time_millis()
        time_threshold = current_time - older_than

        finalized_statuses = [
            JobStatus.SUCCEEDED.to_int(),
            JobStatus.FAILED.to_int(),
            JobStatus.TIMEOUT.to_int(),
            JobStatus.CANCELED.to_int(),
        ]

        with self.ManagedSessionMaker(read_only=False) as session:
            query = self._get_query(session, SqlJob).filter(SqlJob.status.in_(finalized_statuses))

            if job_ids:
                query = query.filter(SqlJob.id.in_(job_ids))

            if older_than > 0:
                query = query.filter(SqlJob.creation_time < time_threshold)

            ids_to_delete = [job.id for job in query.all()]

            if ids_to_delete:
                self._get_query(session, SqlJob).filter(SqlJob.id.in_(ids_to_delete)).delete(
                    synchronize_session=False
                )

            return ids_to_delete

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
        with self.ManagedSessionMaker(read_only=False) as session:
            update_time = get_current_time_millis()
            if (
                self._conditional_status_update(
                    session,
                    job_id,
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                    self._terminal_update_values(JobStatus.CANCELED, None, update_time),
                )
                != JobUpdateStatus.APPLIED
            ):
                self._raise_invalid_transition(
                    session,
                    job_id,
                    "cancel",
                    (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY),
                )
            return self._get_sql_job(session, job_id).to_mlflow_entity()

    def update_status_details(self, job_id: str, status_details: dict[str, Any]) -> None:
        """
        Update job status details.

        Merges the provided status details with existing job status details. For the same
        key, the new value will overwrite the existing value.

        Args:
            job_id: The ID of the job to update
            status_details: Status details to merge into existing job status details
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            job = self._get_sql_job(session, job_id)

            if JobStatus.is_finalized(JobStatus.from_int(job.status)):
                raise JobTerminalStateUpdateException(job_id, JobStatus.from_int(job.status))

            # Merge new status details with existing
            current_details = job.status_details or {}
            current_details.update(status_details)
            job.status_details = current_details
            job.last_update_time = get_current_time_millis()

    def update_job_progress(
        self,
        job_id: str,
        message: str | None = None,
        progress: JobProgress | None = None,
    ) -> None:
        """
        Update structured progress fields for an in-flight job.

        Args:
            job_id: The ID of the job to update
            message: Human-readable plain-text progress message. ``None`` leaves
                the existing value unchanged.
            progress: Structured machine-readable progress payload. ``None``
                leaves the existing value unchanged.
        """
        if message is None and progress is None:
            return

        with self.ManagedSessionMaker(read_only=False) as session:
            job = self._get_sql_job(session, job_id)

            if JobStatus.is_finalized(JobStatus.from_int(job.status)):
                raise JobTerminalStateUpdateException(job_id, JobStatus.from_int(job.status))

            update_time = get_current_time_millis()
            if message is not None:
                job.status_message = message
            if progress is not None:
                job.progress = progress.to_dict()
            job.progress_updated_at = update_time
            job.last_update_time = update_time
