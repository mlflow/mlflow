"""
This module is used for launching the periodic tasks Huey consumer.

This is a dedicated consumer that only runs periodic tasks (like the online scoring scheduler).
It is launched by the job runner and runs in a separate process from job execution consumers.
"""

import logging
import threading

from mlflow.server.jobs._log_filters import SuppressOnlineScoringFilter

# Suppress online scoring logs from huey - use filter since setLevel gets overridden by huey
_filter = SuppressOnlineScoringFilter()
logging.getLogger("huey").addFilter(_filter)
logging.getLogger("huey.consumer").addFilter(_filter)
logging.getLogger("huey.consumer.Scheduler").addFilter(_filter)

# Suppress alembic INFO logs
logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

from mlflow.server.jobs.utils import (
    HUEY_PERIODIC_TASKS_INSTANCE_KEY,
    _exit_when_orphaned,
    _get_or_init_huey_instance,
    register_periodic_tasks,
)

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
