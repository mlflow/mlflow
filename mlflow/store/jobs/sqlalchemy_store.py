import json
import threading
import uuid
from typing import Any, Iterator

import sqlalchemy

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.db.utils import (
    _get_managed_session_maker,
    _safe_initialize_tables,
    create_sqlalchemy_engine_with_retry,
)
from mlflow.store.jobs.abstract_store import AbstractJobStore
from mlflow.store.tracking.dbmodels.models import SqlJob
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri

sqlalchemy.orm.configure_mappers()

_LIST_JOB_PAGE_SIZE = 100


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
        with self.ManagedSessionMaker() as session:
            job_id = str(uuid.uuid4())
            creation_time = get_current_time_millis()

            job = SqlJob(
                id=job_id,
                creation_time=creation_time,
                job_name=job_name,
                params=params,
                timeout=timeout,
                status=JobStatus.PENDING.to_int(),
                result=None,
                last_update_time=creation_time,
            )

            session.add(job)
            session.flush()
            return job.to_mlflow_entity()

    def _update_job(self, job_id: str, new_status: JobStatus, result: str | None = None) -> Job:
        with self.ManagedSessionMaker() as session:
            job = self._get_sql_job(session, job_id)

            if JobStatus.is_finalized(job.status):
                raise MlflowException(
                    f"The Job {job_id} is already finalized with status: {job.status}, "
                    "it can't be updated."
                )

            job.status = new_status.to_int()
            if result is not None:
                job.result = result
            job.last_update_time = get_current_time_millis()
            return job.to_mlflow_entity()

    def start_job(self, job_id: str) -> None:
        """
        Start a job by setting its status to RUNNING.
        Only succeeds if the job is currently in PENDING state.

        Args:
            job_id: The ID of the job to start

        Raises:
            MlflowException: If job is not in PENDING state or doesn't exist
        """
        with self.ManagedSessionMaker() as session:
            # Atomic update: only transition from PENDING to RUNNING
            rows_updated = (
                session.query(SqlJob)
                .filter(SqlJob.id == job_id, SqlJob.status == JobStatus.PENDING.to_int())
                .update(
                    {
                        SqlJob.status: JobStatus.RUNNING.to_int(),
                        SqlJob.last_update_time: get_current_time_millis(),
                    }
                )
            )

            if rows_updated == 0:
                job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
                if job is None:
                    raise MlflowException(
                        f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                    )
                raise MlflowException(
                    f"Job {job_id} is in {JobStatus.from_int(job.status)} state, "
                    "cannot start (must be PENDING)"
                )

    def reset_job(self, job_id: str) -> None:
        """
        Reset a job by setting its status to PENDING.

        Args:
            job_id: The ID of the job to re-enqueue.
        """
        self._update_job(job_id, JobStatus.PENDING)

    def finish_job(self, job_id: str, result: str) -> None:
        """
        Finish a job by setting its status to DONE and setting the result.

        Args:
            job_id: The ID of the job to finish
            result: The job result as a string
        """
        self._update_job(job_id, JobStatus.SUCCEEDED, result)

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Fail a job by setting its status to FAILED and setting the error message.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string
        """
        self._update_job(job_id, JobStatus.FAILED, error)

    def mark_job_timed_out(self, job_id: str) -> None:
        """
        Set a job status to Timeout.

        Args:
            job_id: The ID of the job
        """
        self._update_job(job_id, JobStatus.TIMEOUT)

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
        from mlflow.environment_variables import MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES

        max_retries = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get()

        with self.ManagedSessionMaker() as session:
            job = self._get_sql_job(session, job_id)

            if job.retry_count >= max_retries:
                job.status = JobStatus.FAILED.to_int()
                job.result = error
                return None
            job.retry_count += 1
            job.status = JobStatus.PENDING.to_int()
            job.last_update_time = get_current_time_millis()
            return job.retry_count

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
                query = session.query(SqlJob)

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
                    query.order_by(SqlJob.creation_time)
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

    def _get_sql_job(self, session, job_id) -> SqlJob:
        job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
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
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )
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

        with self.ManagedSessionMaker() as session:
            query = session.query(SqlJob).filter(SqlJob.status.in_(finalized_statuses))

            if job_ids:
                query = query.filter(SqlJob.id.in_(job_ids))

            if older_than > 0:
                query = query.filter(SqlJob.creation_time < time_threshold)

            ids_to_delete = [job.id for job in query.all()]

            if ids_to_delete:
                session.query(SqlJob).filter(SqlJob.id.in_(ids_to_delete)).delete(
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
        return self._update_job(job_id, JobStatus.CANCELED)
