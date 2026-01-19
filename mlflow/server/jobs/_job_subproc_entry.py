"""
This module is used for launching subprocess to execute the job function.

If the job has timeout setting, or the job has pip requirements dependencies,
or the job has extra environment variables setting,
the job is executed as a subprocess.
"""

import importlib
import json
import os
import threading

from mlflow.server.jobs.logging_utils import configure_logging_for_jobs
from mlflow.server.jobs.utils import JobResult, _exit_when_orphaned, _load_function

# Configure Python logging to suppress noisy job logs
configure_logging_for_jobs()

if __name__ == "__main__":
    # ensure the subprocess is killed when parent process dies.
    threading.Thread(
        target=_exit_when_orphaned,
        name="exit_when_orphaned",
        daemon=True,
    ).start()

    params = json.loads(os.environ["_MLFLOW_SERVER_JOB_PARAMS"])
    function = _load_function(os.environ["_MLFLOW_SERVER_JOB_FUNCTION_FULLNAME"])
    result_dump_path = os.environ["_MLFLOW_SERVER_JOB_RESULT_DUMP_PATH"]
    transient_error_classes_path = os.environ["_MLFLOW_SERVER_JOB_TRANSIENT_ERROR_ClASSES_PATH"]

    with open(transient_error_classes_path) as f:
        content = f.read()

    transient_error_classes = []
    for cls_str in content.split("\n"):
        if not cls_str:
            continue
        *module_parts, cls_name = cls_str.split(".")
        module = importlib.import_module(".".join(module_parts))
        transient_error_classes.append(getattr(module, cls_name))

    try:
        value = function(**params)
        job_result = JobResult(
            succeeded=True,
            result=json.dumps(value),
        )
        job_result.dump(result_dump_path)
    except Exception as e:
        JobResult.from_error(e, transient_error_classes).dump(result_dump_path)
