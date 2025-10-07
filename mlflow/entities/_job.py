import json
from typing import Any

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
        function_fullname: str,
        params: str,
        timeout: float | None,
        status: JobStatus,
        result: str | None,
        retry_count: int,
        last_update_time: int,
    ):
        super().__init__()
        self._job_id = job_id
        self._creation_time = creation_time
        self._function_fullname = function_fullname
        self._params = params
        self._timeout = timeout
        self._status = status
        self._result = result
        self._retry_count = retry_count
        self._last_update_time = last_update_time

    @property
    def job_id(self) -> str:
        """String containing job ID."""
        return self._job_id

    @property
    def creation_time(self) -> int:
        """Creation timestamp of the job, in number of milliseconds since the UNIX epoch."""
        return self._creation_time

    @property
    def function_fullname(self) -> str:
        """
        String containing the fully-qualified function name,
        in the form of `<module_name>.<function_name>`
        """
        return self._function_fullname

    @property
    def params(self) -> str:
        """
        String containing the job serialized parameters in JSON format.
        For example, `{"a": 3, "b": 4}` represents two params:
        `a` with value 3 and `b` with value 4.
        """
        return self._params

    @property
    def timeout(self) -> float | None:
        """
        Job execution timeout in seconds.
        """
        return self._timeout

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
    def parsed_result(self) -> Any:
        """
        Return the parsed result.
        If job status is SUCCEEDED, the parsed result is the
        job function returned value
        If job status is FAILED, the parsed result is the error string.
        Otherwise, the parsed result is None.
        """
        if self.status == JobStatus.SUCCEEDED:
            return json.loads(self.result)
        return self.result

    @property
    def retry_count(self) -> int:
        """Integer containing the job retry count"""
        return self._retry_count

    @property
    def last_update_time(self) -> int:
        """Last update timestamp of the job, in number of milliseconds since the UNIX epoch."""
        return self._last_update_time

    def __repr__(self) -> str:
        return f"<Job(job_id={self.job_id}, function_fullname={self.function_fullname})>"
