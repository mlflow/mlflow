"""
This module is used for launching the periodic tasks Huey consumer.

This is a dedicated consumer that only runs periodic tasks (like the online scoring scheduler).
It is launched by the job runner and runs in a separate process from job execution consumers.
"""

import threading

from mlflow.server.jobs.logging_utils import configure_logging_for_jobs
from mlflow.server.jobs.utils import (
    HUEY_PERIODIC_TASKS_INSTANCE_KEY,
    _exit_when_orphaned,
    _get_or_init_huey_instance,
    register_periodic_tasks,
)

# Configure Python logging to suppress noisy job logs
configure_logging_for_jobs()

# Ensure the subprocess is killed when parent process dies.
# The huey consumer's parent process is `_job_runner` process,
# if `_job_runner` process is died, it means the MLflow server exits.
threading.Thread(
    target=_exit_when_orphaned,
    name="exit_when_orphaned",
    daemon=True,
).start()

huey_instance = _get_or_init_huey_instance(HUEY_PERIODIC_TASKS_INSTANCE_KEY).instance

# Register periodic tasks with this dedicated instance
register_periodic_tasks(huey_instance)
