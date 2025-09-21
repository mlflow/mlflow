import json
import logging
import os
import shutil
import sys
import threading
import time
from types import FunctionType
from typing import Any, Callable

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.server.handlers import _get_job_store
from mlflow.environment_variables import MLFLOW_SERVER_JOB_MAX_PARALLELISM


_logger = logging.getLogger(__name__)


class TransientError(RuntimeError):
    """
    Raise `TransientError` in a job to trigger job retry
    """

    def __init__(self, origin_error: Exception):
        super().__init__()
        self._origin_error = origin_error

    @property
    def origin_error(self) -> Exception:
        return self._origin_error


def submit_job(
    function: Callable[..., Any], params: dict[str, Any], timeout: int | None = None
) -> Job:
    """
    Submit a job to the job queue. The job is executed at most once.
    If the MLflow server crashes while the job is pending or running,
    it is rescheduled on restart.

    Note:
        This is a server-side API and requires the MLflow server to configure
        the backend store URI to a database URI.

    Note:
        This is a server-side API, and it requires MLflow server configures
        backend store URI to database URI.

    Args:
        function: The job function, it must be a python module-level function,
            and all params and return value must be JSON-serializable.
            The function can raise `TransientError` in order to trigger
            job retry, you can set `MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES`
            to configure maximum allowed retries for transient errors
            and set `MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY` to
            configure base retry delay in seconds.
        params: The params to be passed to the job function.
        timeout: (optional) the job execution timeout, default None (no timeout)

    Returns:
        The Job entity. You can call `query_job` API by the `job.job_id` to get
        the job status and result.
    """
    from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_JOB_EXECUTION
    from mlflow.server.job.job_runner import huey_task_exec_job

    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        raise MlflowException(
            "Mlflow server job execution feature is not enabled, please set "
            "environment variable 'MLFLOW_SERVER_ENABLE_JOB_EXECUTION' to 'true' to enable it."
        )

    if not (isinstance(function, FunctionType) and "." not in function.__qualname__):
        raise MlflowException("The job function must be a python global function.")

    job_store = _get_job_store()
    serialized_params = json.dumps(params)
    func_fullname = f"{function.__module__}.{function.__name__}"
    job = job_store.create_job(func_fullname, serialized_params, timeout)

    # enqueue job
    huey_task_exec_job(job.job_id, function, params, timeout)

    return job


def query_job(job_id: str) -> tuple[JobStatus, Any]:
    """
    Query the job status and result by the job id.

    Note:
        This is a server-side API, and it requires MLflow server configures
        backend store URI to database URI.

    Args:
        job_id: The job id.

    Returns:
        A tuple of (status, result)
        status value is one of PENDING / RUNNING / DONE / FAILED.
        If status is PENDING / RUNNING, result is None.
        If status is DONE, result is the job function returned value.
        If status is FAILED, result is the error message.
    """
    job_store = _get_job_store()
    job = job_store.get_job(job_id)
    status = job.status
    result = job.result
    if status == JobStatus.DONE:
        result = json.loads(result)
    return status, result


def _start_job_runner(
    env_map: dict[str, str],
    max_job_parallelism: int,
    server_proc_pid: int,
    start_new_runner: bool,
) -> "subprocess.Popen":
    from mlflow.utils.process import _exec_cmd

    return _exec_cmd(
        [
            sys.executable,
            shutil.which("huey_consumer.py"),
            "mlflow.server.job.job_runner.huey",
            "-w",
            str(max_job_parallelism),
        ],
        capture_output=False,
        synchronous=False,
        extra_env={
            **env_map,
            "_IS_MLFLOW_JOB_RUNNER": "1",
            "MLFLOW_SERVER_PID": str(server_proc_pid),
            "_START_NEW_MLFLOW_JOB_RUNNER": ("1" if start_new_runner else "0"),
        },
    )


def _reinit_huey_queue() -> None:
    from mlflow.server.job.job_runner import _init_huey_queue

    _init_huey_queue()


def _launch_job_backend(
    backend_store_uri: str,
    env_map: dict[str, str],
    server_proc_pid: int,
) -> None:
    from mlflow.utils.uri import extract_db_type_from_uri

    if extract_db_type_from_uri(backend_store_uri) is None:
        _logger.warning(
            f"Job store requires a database backend store URI but got {backend_store_uri}, "
            "skip launching the job scheduler."
        )
        return

    max_job_parallelism = MLFLOW_SERVER_JOB_MAX_PARALLELISM.get()

    if not max_job_parallelism:
        max_job_parallelism = os.cpu_count() or 1

    def _start_job_runner_fn() -> None:
        start_new_runner = True
        while True:
            # start Mlflow job runner process
            # Put it inside the loop to ensure the job runner process alive
            job_runner_proc = _start_job_runner(
                env_map, max_job_parallelism, server_proc_pid, start_new_runner
            )
            job_runner_proc.wait()
            start_new_runner = False
            time.sleep(1)

    # start job runner.
    threading.Thread(target=_start_job_runner_fn, daemon=True).start()
