import requests
from typing import Any, Callable
import importlib
import json
import os
import errno
import sys
import threading
import time
import json
import signal
from huey import SqliteHuey
from huey.exceptions import RetryTask
from huey.serializer import Serializer
import cloudpickle
import tempfile
from mlflow.entities._job import JobStatus
from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.exceptions import MlflowException


def _create_huey_instance():
    class CloudPickleSerializer(Serializer):
        def serialize(self, data):
            return cloudpickle.dumps(data)

        def deserialize(self, data):
            return cloudpickle.loads(data)

    return SqliteHuey(
        filename=os.environ[HUEY_STORAGE_PATH_ENV_VAR],
        results=False,
        serializer=CloudPickleSerializer()
    )


huey = _create_huey_instance()


_TRANSIENT_ERRORS = (
    requests.Timeout,
    requests.ConnectionError,
)


@huey.task()
def huey_task_exec_job(job_id: str, function: Callable, params: dict[str, Any]) -> None:
    tracking_store = _get_tracking_store()
    tracking_store.start_job(job_id)
    try:
        result = function(**params)
        serialized_result = json.dumps(result)
        tracking_store.finish_job(job_id, serialized_result)
    except _TRANSIENT_ERRORS as e:
        # For transient errors, if the retry count is less than max allowed count,
        # trigger task retry by raising `RetryTask` exception.
        do_retry = tracking_store.retry_or_fail_job(job_id, repr(e))
        if do_retry:
            raise RetryTask(delay=10)
    except Exception as e:
        tracking_store.fail_job(job_id, repr(e))


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
    mlflow_server_pid = int(os.environ["MLFLOW_SERVER_PID"])

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
    tracking_store = _get_tracking_store()

    pending_jobs = tracking_store.list_jobs(status=JobStatus.PENDING)
    running_jobs = tracking_store.list_jobs(status=JobStatus.RUNNING)

    for job in running_jobs:
        tracking_store.reset_job(job.job_id)  # reset the job status to PENDING

    pending_jobs = pending_jobs + running_jobs

    for job in pending_jobs:
        params = json.loads(job.params)
        function = _load_function(job.function)
        # enqueue job
        huey_task_exec_job(job.job_id, function, params)


if os.environ.get("_IS_MLFLOW_JOB_RUNNER") == "1":
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()
    _enqueue_pending_running_jobs()
