"""
This module is used for launching Huey consumer

the command is like:

```
export _MLFLOW_HUEY_STORAGE_PATH={huey_store_dir}
export _MLFLOW_HUEY_INSTANCE_KEY={huey_instance_key}
huey_consumer.py mlflow.server.jobs.huey_consumer.huey_instance -w {max_workers}
```

It launches the Huey consumer that polls tasks from the huey storage file path
`{huey_store_dir}/{huey_instance_key}.mlflow-huey-store`
and schedules the job execution continuously.
"""

import logging
import os
import threading

from mlflow.server.constants import MLFLOW_HUEY_INSTANCE_KEY

# Filter to suppress huey logs for online scoring jobs only
ONLINE_SCORING_JOB_NAMES = ("run_online_trace_scorer", "run_online_session_scorer")

# Check if this consumer is for an online scoring job based on the instance key
_is_online_scoring_consumer = os.environ.get(MLFLOW_HUEY_INSTANCE_KEY) in ONLINE_SCORING_JOB_NAMES


class OnlineScoringLogFilter(logging.Filter):
    """Filter that suppresses INFO logs for online scoring job consumers."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Only filter if this is an online scoring consumer
        if not _is_online_scoring_consumer:
            return True
        # Only filter INFO level logs
        return record.levelno != logging.INFO


# Add filter to huey and alembic loggers to suppress online scoring job logs
_filter = OnlineScoringLogFilter()
logging.getLogger("huey").addFilter(_filter)
logging.getLogger("alembic.runtime.migration").addFilter(_filter)
from mlflow.server.jobs.utils import (
    _exit_when_orphaned,
    _get_or_init_huey_instance,
)

# ensure the subprocess is killed when parent process dies.
# The huey consumer's parent process is `_job_runner` process,
# if `_job_runner` process is died, it means the MLflow server exits.
threading.Thread(
    target=_exit_when_orphaned,
    name="exit_when_orphaned",
    daemon=True,
).start()

huey_instance = _get_or_init_huey_instance(os.environ[MLFLOW_HUEY_INSTANCE_KEY]).instance
