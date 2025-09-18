import errno
import importlib
import json
import os
import signal
import threading
import time
from typing import Any, Callable

import cloudpickle
import requests
from huey import SqliteHuey
from huey.exceptions import RetryTask
from huey.serializer import Serializer

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_INTERVAL
from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.server.handlers import _get_job_store


huey = None


_TRANSIENT_ERRORS = (
    requests.Timeout,
    requests.ConnectionError,
)


def _raise_retry(retry_count: int) -> None:
    # We can support more retry strategies (e.g. exponential backoff) in future
    raise RetryTask(delay=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_INTERVAL.get())


def _exec_job(job_id: str, function: Callable, params: dict[str, Any]) -> None:
    job_store = _get_job_store()
    job_store.start_job(job_id)
    try:
        result = function(**params)
        serialized_result = json.dumps(result)
        job_store.finish_job(job_id, serialized_result)
    except _TRANSIENT_ERRORS as e:
        # For transient errors, if the retry count is less than max allowed count,
        # trigger task retry by raising `RetryTask` exception.
        retry_count = job_store.retry_or_fail_job(job_id, repr(e))
        if retry_count is not None:
            _raise_retry(retry_count)
    except Exception as e:
        job_store.fail_job(job_id, repr(e))


huey_task_exec_job = None


def _init_huey_queue():
    global huey, huey_task_exec_job

    class CloudPickleSerializer(Serializer):
        def serialize(self, data):
            return cloudpickle.dumps(data)

        def deserialize(self, data):
            return cloudpickle.loads(data)

    huey = SqliteHuey(
        filename=os.environ[HUEY_STORAGE_PATH_ENV_VAR],
        results=False,
        serializer=CloudPickleSerializer(),
    )
    huey_task_exec_job = huey.task()(_exec_job)


_init_huey_queue()


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # doesn't actually kill
    except OSError as e:
        if e.errno == errno.ESRCH:  # No such process
            return False
        elif e.errno == errno.EPERM:  # Process exists, but no permission
            return True
        else:
            raise
    else:
        return True


def _start_watcher_to_kill_job_runner_if_mlflow_server_dies(check_interval=1.0):
    mlflow_server_pid = int(os.environ.get("MLFLOW_SERVER_PID"))

    def watcher():
        while True:
            if not _is_process_alive(mlflow_server_pid):
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(check_interval)

    t = threading.Thread(target=watcher, daemon=True)
    t.start()


def _load_function(fullname: str):
    module_name, func_name = fullname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _enqueue_pending_running_jobs():
    job_store = _get_job_store()

    pending_jobs = job_store.list_jobs(status=JobStatus.PENDING)
    running_jobs = job_store.list_jobs(status=JobStatus.RUNNING)

    for job in running_jobs:
        job_store.reset_job(job.job_id)  # reset the job status to PENDING

    pending_jobs = pending_jobs + running_jobs

    for job in pending_jobs:
        params = json.loads(job.params)
        function = _load_function(job.function)
        # enqueue job
        huey_task_exec_job(job.job_id, function, params)


if os.environ.get("_IS_MLFLOW_JOB_RUNNER") == "1":
    # This module is launched as huey consumer by command like
    # `huey_consumer.py server.job.job_runner`
    # the huey consumer will automatically poll the queue and schedule tasks
    # when initializing the huey consumer, we need to set up watcher thread and
    # enqueue unfinished jobs
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()
    _enqueue_pending_running_jobs()
