"""
This module is used for launching Huey consumer

the command is like:

```
export _MLFLOW_HUEY_STORAGE_PATH={huey_store_dir}
export _MLFLOW_HUEY_INSTANCE_KEY={huey_instance_key}
huey_consumer.py mlflow.server.jobs.huey_consumer.huey_instance -w {max_workers}
```

It launches the Huey consumer that polls tasks from the huey storage file path
`{huey_store_dir}/mlflow-huey-store.{huey_instance_key}`
and schedules the job execution continuously.
"""

import os
from mlflow.server.jobs.util import (
    _get_or_init_huey_instance,
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies,
)

_start_watcher_to_kill_job_runner_if_mlflow_server_dies()

huey_instance = _get_or_init_huey_instance(
    os.environ["_MLFLOW_HUEY_INSTANCE_KEY"]
).instance
