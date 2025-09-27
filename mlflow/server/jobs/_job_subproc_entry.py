"""
This module is used for launching subprocess to execute the job function.

If the job has timeout setting, or the job has pip requirements dependencies,
or the job has extra environment variables setting,
the job is executed as a subprocess.
"""

import os
import json
from mlflow.server.jobs.util import JobResult, _load_function


if __name__ == "__main__":
    params = json.loads(os.environ["_MLFLOW_SERVER_JOB_PARAMS"])
    function = _load_function(os.environ["_MLFLOW_SERVER_JOB_FUNCTION_FULLNAME"])
    result_dump_path = os.environ["_MLFLOW_SERVER_JOB_RESULT_DUMP_PATH"]
    try:
        value = function(**params)
        job_result = JobResult(
            succeeded=True,
            result=json.dumps(value),
        )
        job_result.dump(result_dump_path)
    except Exception as e:
        JobResult.from_error(e).dump(result_dump_path)
