"""
This module is used for launching subprocess to execute the job function.

If the job has timeout setting, or the job has pip requirements dependencies,
or the job has extra environment variables setting,
the job is executed as a subprocess.
"""

import importlib
import json
import logging
import os
import threading
import traceback
from contextlib import nullcontext

import cloudpickle

from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.server.jobs.logging_utils import configure_logging_for_jobs
from mlflow.server.jobs.utils import JobResult, _exit_when_orphaned, _load_function
from mlflow.utils.workspace_context import WorkspaceContext

_logger = logging.getLogger(__name__)
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

    workspace = os.environ.get(MLFLOW_WORKSPACE.name)
    ctx = WorkspaceContext(workspace) if workspace else nullcontext()

    transient_error_classes = []
    try:
        with open(transient_error_classes_path, "rb") as f:
            transient_error_classes = cloudpickle.load(f)
        if transient_error_classes is None:
            transient_error_classes = []
    except Exception:
        with open(transient_error_classes_path) as f:
            content = f.read()

        for cls_str in content.split("\n"):
            if not cls_str:
                continue
            *module_parts, cls_name = cls_str.split(".")
            module = importlib.import_module(".".join(module_parts))
            transient_error_classes.append(getattr(module, cls_name))

    try:
        with ctx:
            value = function(**params)
        job_result = JobResult(
            succeeded=True,
            result=json.dumps(value),
        )
        job_result.dump(result_dump_path)
    except Exception as e:
        _logger.error(
            f"Job function {os.environ['_MLFLOW_SERVER_JOB_FUNCTION_FULLNAME']} failed with "
            f"error:\n{traceback.format_exc()}"
        )
        JobResult.from_error(e, transient_error_classes).dump(result_dump_path)
