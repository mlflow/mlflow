"""
This module is used for launching the job runner process.

The job runner will:
* enqueue all unfinished huey tasks when MLflow server is down last time.
* Watch the `_MLFLOW_HUEY_STORAGE_PATH` path,
  if new files (named like `XXX.mlflow-huey-store`) are created,
  it means a new Huey queue is created, then the job runner
  launches an individual Huey consumer process for each Huey queue.
  See module `mlflow/server/jobs/_huey_consumer.py` for details of Huey consumer.
* Initialize periodic tasks on a dedicated Huey instance.
"""

import logging
import os
import time

from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.server.jobs.utils import (
    _enqueue_unfinished_jobs,
    _job_name_to_fn_fullname_map,
    _launch_huey_consumer,
    _launch_periodic_tasks_consumer,
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies,
)

if __name__ == "__main__":
    logger = logging.getLogger("mlflow.server.jobs._job_runner")
    server_up_time = int(time.time() * 1000)
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()

    huey_store_path = os.environ[HUEY_STORAGE_PATH_ENV_VAR]

    for job_name in _job_name_to_fn_fullname_map:
        try:
            _launch_huey_consumer(job_name)
        except Exception as e:
            logging.warning(f"Launch Huey consumer for {job_name} jobs failed, root cause: {e!r}")

    # Launch dedicated consumer for periodic tasks
    # (periodic tasks are registered when the consumer starts up)
    _launch_periodic_tasks_consumer()

    time.sleep(10)  # wait for huey consumer launching
    _enqueue_unfinished_jobs(server_up_time)
