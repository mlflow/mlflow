"""
This module is used for launching the job runner process.

The job runner will:
* enqueue all unfinished huey tasks when MLflow server is down last time.
* Watch the `_MLFLOW_HUEY_STORAGE_PATH` path,
  if new files (named like `XXX.mlflow-huey-store`) are created,
  it means a new Huey queue is created, then the job runner
  launches an individual Huey consumer process for each Huey queue.
  See module `mlflow/server/jobs/_huey_consumer.py` for details of Huey consumer.
"""

import logging
import os

from mlflow.exceptions import MlflowException
from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.server.jobs import _ALLOWED_JOB_FUNCTION_LIST
from mlflow.server.jobs.utils import (
    _enqueue_unfinished_jobs,
    _launch_huey_consumer,
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies,
)

if __name__ == "__main__":
    logger = logging.getLogger("mlflow.server.jobs._job_runner")
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()
    _enqueue_unfinished_jobs()

    huey_store_path = os.environ[HUEY_STORAGE_PATH_ENV_VAR]

    for job_fn_fullname in _ALLOWED_JOB_FUNCTION_LIST:
        try:
            _launch_huey_consumer(job_fn_fullname)
        except Exception as e:
            logging.warning(
                f"Launch Huey consumer for {job_fn_fullname} jobs failed, root cause: {repr(e)}"
            )
