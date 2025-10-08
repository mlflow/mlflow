"""
This module is used for launching subprocess to execute the job function.

If the job has timeout setting, or the job has pip requirements dependencies,
or the job has extra environment variables setting,
the job is executed as a subprocess.
"""

import json
import os
import threading

import cloudpickle

from mlflow.server.jobs.utils import JobResult, _exit_when_orphaned, _load_function

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

    try:
        with open(transient_error_classes_path, "rb") as f:
            transient_error_classes = cloudpickle.load(f)
        value = function(**params)
        job_result = JobResult(
            succeeded=True,
            result=json.dumps(value),
        )
        job_result.dump(result_dump_path)
    except Exception as e:
        JobResult.from_error(e, transient_error_classes).dump(result_dump_path)
