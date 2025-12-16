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

    def create_job(self, function_fullname: str, params: str, timeout: float | None = None) -> Job:
        """
        Create a new job with the specified function and parameters.

        Args:
            function_fullname: The full name of the function to execute
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
                function_fullname=function_fullname,
                params=params,
                timeout=timeout,
                status=JobStatus.PENDING.to_int(),
                result=None,
                last_update_time=creation_time,
            )

            session.add(job)
            session.flush()
            return job.to_mlflow_entity()

    def _update_job(self, job_id: str, new_status: JobStatus, result: str | None = None) -> None:
        with self.ManagedSessionMaker() as session:
            job = self._get_sql_job(session, job_id)

            job.status = new_status.to_int()
            if result is not None:
                job.result = result
            job.last_update_time = get_current_time_millis()

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
        function_fullname: str | None = None,
        statuses: list[JobStatus] | None = None,
        begin_timestamp: int | None = None,
        end_timestamp: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[Job]:
        """
        List jobs based on the provided filters.

        Args:
            function_fullname: Filter by function full name (exact match)
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
                if function_fullname is not None:
                    query = query.filter(SqlJob.function_fullname == function_fullname)

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
