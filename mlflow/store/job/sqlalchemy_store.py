import uuid

import sqlalchemy

import mlflow.store.db.utils
from mlflow.entities._job import Job, JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.job.abstract_store import AbstractJobStore
from mlflow.store.tracking.dbmodels.models import SqlJob
from mlflow.utils.time import get_current_time_millis

sqlalchemy.orm.configure_mappers()


class SqlAlchemyJobStore(AbstractJobStore):
    """
    SQLAlchemy compliant backend store for storing Job metadata.
    This store interacts with SQL store using SQLAlchemy abstractions defined for MLflow Job entities.
    """

    def __init__(self, db_uri):
        """
        Create a database backed store.

        Args:
            db_uri: The SQLAlchemy database URI string to connect to the database.
        """
        super().__init__()
        self.db_uri = db_uri
        self.engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(db_uri)
        if not mlflow.store.db.utils._all_tables_exist(self.engine):
            mlflow.store.db.utils._initialize_tables(self.engine)
        import sqlalchemy.orm

        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(
            SessionMaker, self.engine.dialect.name
        )

    def create_job(self, function: str, params: str) -> str:
        """
        Create a new job with the specified function and parameters.

        Args:
            function: The function name to execute
            params: The job parameters as a string

        Returns:
            The job ID (UUID4 string)
        """
        with self.ManagedSessionMaker() as session:
            job_id = str(uuid.uuid4())
            creation_time = get_current_time_millis()

            job = SqlJob(
                id=job_id,
                creation_time=creation_time,
                function=function,
                params=params,
                status=JobStatus.PENDING.to_int(),
                result=None,
            )

            session.add(job)
            session.flush()
            return job_id

    def start_job(self, job_id: str) -> None:
        """
        Start a job by setting its status to RUNNING.

        Args:
            job_id: The ID of the job to start
        """
        with self.ManagedSessionMaker() as session:
            job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )

            job.status = JobStatus.RUNNING.to_int()

    def reset_job(self, job_id: str) -> None:
        """
        Reset a job by setting its status to PENDING.

        Args:
            job_id: The ID of the job to re-enqueue.
        """
        with self.ManagedSessionMaker() as session:
            job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )

            job.status = JobStatus.PENDING.to_int()

    def finish_job(self, job_id: str, result: str) -> None:
        """
        Finish a job by setting its status to DONE and setting the result.

        Args:
            job_id: The ID of the job to finish
            result: The job result as a string
        """
        with self.ManagedSessionMaker() as session:
            job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )

            job.status = JobStatus.DONE.to_int()
            job.result = result

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Fail a job by setting its status to FAILED and setting the error message.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string
        """
        with self.ManagedSessionMaker() as session:
            job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )

            job.status = JobStatus.FAILED.to_int()
            job.result = error

    def retry_or_fail_job(self, job_id: str, error: str) -> bool:
        """
        If the job retry_count is less than maximum allowed retry count,
        increase the retry_count and reset the job to PENDING status,
        otherwise set the job to FAIL status and fill the job's error field.

        Args:
            job_id: The ID of the job to fail
            error: The error message as a string

        Returns:
            If the job is retried, returns `True` otherwise returns `False`
        """
        from mlflow.environment_variables import MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES

        max_retries = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get()

        with self.ManagedSessionMaker() as session:
            job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )

            if job.retry_count >= max_retries:
                job.status = JobStatus.FAILED.to_int()
                job.result = error
                return False
            job.retry_count += 1
            job.status = JobStatus.PENDING.to_int()
            return True

    def list_jobs(
        self,
        function: str | None = None,
        status: JobStatus | None = None,
        begin_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ) -> list[Job]:
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
        with self.ManagedSessionMaker() as session:
            # Select all columns needed for Job entity
            query = session.query(SqlJob)

            # Apply filters
            if function is not None:
                query = query.filter(SqlJob.function == function)

            if status is not None:
                query = query.filter(SqlJob.status == status.to_int())

            if begin_timestamp is not None:
                query = query.filter(SqlJob.creation_time >= begin_timestamp)

            if end_timestamp is not None:
                query = query.filter(SqlJob.creation_time <= end_timestamp)

            # Order by creation time (newest first) and return Job entities
            jobs = query.order_by(SqlJob.creation_time.desc()).all()
            return [job.to_mlflow_entity() for job in jobs]

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
            job = session.query(SqlJob).filter(SqlJob.id == job_id).one_or_none()
            if job is None:
                raise MlflowException(
                    f"Job with ID {job_id} not found", error_code=RESOURCE_DOES_NOT_EXIST
                )
            return job.to_mlflow_entity()
