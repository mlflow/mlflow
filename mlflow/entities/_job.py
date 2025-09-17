from mlflow.entities._job_status import JobStatus
from mlflow.entities._mlflow_object import _MlflowObject


class Job(_MlflowObject):
    """
    MLflow entity representing a Job.
    """

    def __init__(
        self,
        job_id: str,
        creation_time: int,
        function: str,
        params: str,
        status: JobStatus,
        result: str | None,
        retry_count: int = 0,
    ):
        super().__init__()
        self._job_id = job_id
        self._creation_time = creation_time
        self._function = function
        self._params = params
        self._status = status
        self._result = result
        self._retry_count = retry_count

    @property
    def job_id(self) -> str:
        """String containing job ID."""
        return self._job_id

    @property
    def creation_time(self) -> int:
        """Creation timestamp of the job, in number of milliseconds since the UNIX epoch."""
        return self._creation_time

    @property
    def function(self) -> str:
        """
        String containing the fully-qualified function name, in the form of `<module_name>.<function_name>`
        """
        return self._function

    @property
    def params(self) -> str:
        """
        String containing the job serialized parameters in JSON format.
        For example, `{"a": 3, "b": 4}` represents two params: `a` with value 3 and `b` with value 4.
        """
        return self._params

    @property
    def status(self) -> JobStatus:
        """
        One of the values in :py:class:`mlflow.entities._job_status.JobStatus`
        describing the status of the job.
        """
        return self._status

    @property
    def result(self) -> str | None:
        """String containing the job result or error message."""
        return self._result

    @property
    def retry_count(self) -> int:
        """Integer containing the job retry count"""
        return self._retry_count

    def __repr__(self) -> str:
        return f"<Job(job_id={self.job_id}, function={self.function})>"
