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

import os
import time

from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.server.jobs.utils import (
    _enqueue_unfinished_jobs,
    _launch_huey_consumer,
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies,
)

if __name__ == "__main__":
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()
    _enqueue_unfinished_jobs()

    huey_store_path = os.environ[HUEY_STORAGE_PATH_ENV_VAR]

    seen_huey_files = set()
    huey_file_suffix = ".mlflow-huey-store"

    while True:
        time.sleep(0.5)
        current_huey_files = set(os.listdir(huey_store_path))
        new_huey_files = current_huey_files - seen_huey_files

        for huey_file in new_huey_files:
            if huey_file.endswith(huey_file_suffix):
                job_fn_fullname = huey_file[: -len(huey_file_suffix)]
                _launch_huey_consumer(job_fn_fullname)

        seen_huey_files = current_huey_files
