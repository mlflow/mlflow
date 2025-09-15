from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities._job_status import JobStatus


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
        status: JobStatus | int,
        result: str | None = None,
    ):
        super().__init__()
        self._job_id = job_id
        self._creation_time = creation_time
        self._function = function
        self._params = params
        self._status = (
            status if isinstance(status, JobStatus) else JobStatus.from_int(status)
        )
        self._result = result

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
        String containing the function full name, in the form of `<module_name>.<function_name>`
        """
        return self._function

    @property
    def params(self) -> str:
        """
        String containing the job serialized parameters in JSON format.
        e.g. `{'a': 3, 'b': 4}` represents there are 2 params `a` with value 3 and
        `b` with value 4.
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, Job):
            return False
        return (
            self.job_id == other.job_id
            and self.creation_time == other.creation_time
            and self.function == other.function
            and self.params == other.params
            and self.status == other.status
            and self.result == other.result
        )

    def __repr__(self) -> str:
        return f"<Job({self.job_id}, {self.function}, {self.status.value})>"
