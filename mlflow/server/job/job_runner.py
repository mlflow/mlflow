from typing import Any, Callable
import os
import errno
import sys
import threading
import time
import json
import signal
from huey import SqliteHuey
from huey.serializer import Serializer
import cloudpickle
from mlflow.server.job import job_functions
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.exceptions import MlflowException


def _create_huey_instance():
    from mlflow.server import MLFLOW_HUEY_STORAGE_PATH

    class CloudPickleSerializer(Serializer):
        def serialize(self, data):
            return cloudpickle.dumps(data)

        def deserialize(self, data):
            return cloudpickle.loads(data)

    return SqliteHuey(
        filename=MLFLOW_HUEY_STORAGE_PATH,
        results=False,
        serializer=CloudPickleSerializer()
    )


huey = _create_huey_instance()


@huey.task()
def exec_job(job_id: str, function: Callable, params: dict[str, Any]) -> None:
    tracking_store = _get_tracking_store()
    tracking_store.start_job(job_id)
    try:
        result = function(**params)
        serialized_result = json.dumps(result)
        tracking_store.finish_job(job_id, serialized_result)
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


def _kill_job_runner_if_mlflow_server_dies(check_interval=1.0):
    mlflow_server_pid = int(os.environ["MLFLOW_SERVER_PID"])

    def watcher():
        while True:
            if not _is_process_alive(mlflow_server_pid):
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(check_interval)

    t = threading.Thread(target=watcher, daemon=True)
    t.start()


if os.environ.get("_IS_MLFLOW_JOB_RUNNER") == "1":
    _kill_job_runner_if_mlflow_server_dies()
