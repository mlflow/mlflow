from typing import Any
from types import FunctionType

import json
from mlflow.entities._job import JobStatus
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.exceptions import MlflowException


_tracking_store = _get_tracking_store()
if not isinstance(_tracking_store, SqlAlchemyStore):
    raise MlflowException.invalid_parameter_value(
        "If enabling MLflow job execution, mlflow server must configure "
        "--backend-store-uri to a database URI."
    )


def submit_job(function, **params: Any):
    """
    Submit a job to the job queue.
    The job is ensured to be scheduled to execute once.
    If Mlflow server crashes when the job is in pending / running status,
    when the Mlflow server restarts, the job will be scheduled again.

    Args:
        function: The job funtion, it must be a python global function,
            and all params and return value must be JSON-serializable.
        **params: The params to be passed to the job function.

    Returns:
        The unique job id. You can call `query_job` API by the `job_id` to get
        the job status and result.
    """
    from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_JOB_EXECUTION
    from mlflow.server.job.job_runner import huey_task_exec_job

    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        raise MlflowException(
            "Mlflow server job execution feature is not enabled, please set "
            "environment variable 'MLFLOW_SERVER_ENABLE_JOB_EXECUTION' to 'true' to enable it."
        )

    if not (
        isinstance(function, FunctionType) and
        "." not in function.__qualname__
    ):
        raise MlflowException("The job function must be a python global function.")
    serialized_params = json.dumps(params)
    func_fullname = f"{function.__module__}.{function.__name__}"
    job_id = _tracking_store.create_job(func_fullname, serialized_params)

    # enqueue job
    huey_task_exec_job(job_id, function, params)

    return job_id


def query_job(job_id: str) -> tuple[JobStatus, Any]:
    """
    Query the job status and result by the job id.

    Args:
        job_id: The job id.

    Returns:
        A tuple of (status, result)
        status value is one of PENDING / RUNNING / DONE / FAILED.
        If status is PENDING / RUNNING, result is None.
        If status is DONE, result is the job function returned value.
        If status is FAILED, result is the error message.
    """
    job = _tracking_store.get_job(job_id)
    status = job.status
    result = job.result
    if status == JobStatus.DONE:
        result = json.loads(result)
    return status, result
