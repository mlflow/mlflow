import importlib
import json
import os
import signal
import threading
import time
from typing import Any, Callable

import cloudpickle
from huey import SqliteHuey
from huey.exceptions import RetryTask
from huey.serializer import Serializer

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
)
from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.server.handlers import _get_job_store

huey_instance = None


def _exponential_backoff_retry(retry_count: int) -> None:
    # We can support more retry strategies (e.g. exponential backoff) in future
    base_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get()
    max_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get()
    delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
    raise RetryTask(delay=delay)


def _exec_job(
    job_id: str, function: Callable[..., Any], params: dict[str, Any], timeout: float | None
) -> None:
    from mlflow.server.jobs.util import execute_function_with_timeout

    job_store = _get_job_store()
    job_store.start_job(job_id)

    try:
        job_result = execute_function_with_timeout(function, params, timeout)

        if job_result.succeeded:
            job_store.finish_job(job_id, job_result.result)
        else:
            if job_result.is_transient_error:
                # For transient errors, if the retry count is less than max allowed count,
                # trigger task retry by raising `RetryTask` exception.
                retry_count = job_store.retry_or_fail_job(job_id, job_result.error)
                if retry_count is not None:
                    _exponential_backoff_retry(retry_count)
            else:
                job_store.fail_job(job_id, job_result.error)
    except TimeoutError:
        job_store.mark_job_timed_out(job_id)


huey_task_exec_job = None


def _init_huey_queue():
    global huey_instance, huey_task_exec_job

    class CloudPickleSerializer(Serializer):
        def serialize(self, data):
            return cloudpickle.dumps(data)

        def deserialize(self, data):
            return cloudpickle.loads(data)

    huey_instance = SqliteHuey(
        filename=os.environ[HUEY_STORAGE_PATH_ENV_VAR],
        results=False,
        serializer=CloudPickleSerializer(),
    )
    huey_task_exec_job = huey_instance.task()(_exec_job)


_init_huey_queue()


def _start_watcher_to_kill_job_runner_if_mlflow_server_dies(check_interval: float = 1.0) -> None:
    from mlflow.server.jobs.util import is_process_alive

    mlflow_server_pid = int(os.environ.get("MLFLOW_SERVER_PID"))

    def watcher():
        while True:
            if not is_process_alive(mlflow_server_pid):
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(check_interval)

    t = threading.Thread(target=watcher, daemon=True, name="job-runner-watcher")
    t.start()


def _load_function(fullname: str) -> Callable[..., Any]:
    match fullname.split("."):
        case [*module_parts, func_name] if module_parts:
            module_name = ".".join(module_parts)
        case _:
            raise ValueError(f"Invalid function fullname: {fullname!r}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _enqueue_unfinished_jobs() -> None:
    job_store = _get_job_store()

    if os.environ.get("_START_NEW_MLFLOW_JOB_RUNNER") == "1":
        # If `_START_NEW_MLFLOW_JOB_RUNNER` is `1`,
        # it is the case that a new MLflow server instance is launched,
        # and a new mlflow job runner process is launched,
        # so that queue is empty,
        # we need to enqueue all pending / running jobs that are recorded
        # in MLflow job store.
        status_list = [JobStatus.PENDING, JobStatus.RUNNING]
    else:
        # If `_START_NEW_MLFLOW_JOB_RUNNER` is `0`,
        # it is the case that mlflow job runner process crashes and restarts,
        # in the case the pending jobs are already in the job queue
        # (they are loaded from the job queue persisted storage),
        # we only need to enqueue the lost running jobs.
        status_list = [JobStatus.RUNNING]

    unfinished_jobs = job_store.list_jobs(statuses=status_list)

    for job in unfinished_jobs:
        if job.status == JobStatus.RUNNING:
            job_store.reset_job(job.job_id)  # reset the job status to PENDING

        params = json.loads(job.params)
        function = _load_function(job.function_fullname)
        timeout = job.timeout
        # enqueue job
        huey_task_exec_job(job.job_id, function, params, timeout)


if os.environ.get("_IS_MLFLOW_JOB_RUNNER") == "1":
    # This module is launched as huey consumer by command like
    # `huey_consumer.py server.job.job_runner`
    # the huey consumer will automatically poll the queue and schedule tasks
    # when initializing the huey consumer, we need to set up watcher thread and
    # enqueue unfinished jobs
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()
    _enqueue_unfinished_jobs()
