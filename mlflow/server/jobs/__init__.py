import json
import logging
import os
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable, ParamSpec, TypeVar

from mlflow.entities._job import Job as JobEntity
from mlflow.exceptions import MlflowException
from mlflow.server.handlers import _get_job_store
from mlflow.utils.environment import _PythonEnv

_logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


_ALLOWED_JOB_FUNCTION_LIST = [
    # Putting all allowed job function in the list
]

if allowed_job_function_list_env := os.environ.get("_MLFLOW_ALLOWED_JOB_FUNCTION_LIST"):
    _ALLOWED_JOB_FUNCTION_LIST += allowed_job_function_list_env.split(",")


class TransientError(RuntimeError):
    """
    Raise `TransientError` in a job to trigger job retry
    """

    def __init__(self, origin_error: Exception):
        super().__init__()
        self._origin_error = origin_error

    @property
    def origin_error(self) -> Exception:
        return self._origin_error


@dataclass
class JobFunctionMetadata:
    fn_fullname: str
    max_workers: int
    transient_error_classes: list[type[Exception]] | None = None
    python_env: _PythonEnv | None = None


def job(
    max_workers: int,
    transient_error_classes: list[type[Exception]] | None = None,
    python_version: str | None = None,
    pip_requirements: list[str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    The decorator for the custom job function for setting max parallel workers that
    the job function can use.
    Each job is executed in an individual subprocess.

    Args:
        max_workers: The maximum number of workers that are allowed to run the jobs
            using this job function.
        transient_error_classes: (optional) Specify a list of classes that are regarded as
            transient error classes.
        python_version: (optional) The required python version to run the job function.
        pip_requirements: (optional) The required pip requirements to run the job function,
            relative file references such as "-r requirements.txt" are not supported.
    """
    from mlflow.utils import PYTHON_VERSION
    from mlflow.utils.requirements_utils import _parse_requirements
    from mlflow.version import VERSION

    if not python_version and not pip_requirements:
        python_env = None
    else:
        python_version = python_version or PYTHON_VERSION
        try:
            pip_requirements = [
                req.req_str for req in _parse_requirements(pip_requirements, is_constraint=False)
            ]
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Invalid pip_requirements for job function: {pip_requirements}, "
                f"parsing error: {e!r}"
            )
        if mlflow_home := os.environ.get("MLFLOW_HOME"):
            # Append MLflow dev version dependency (for testing)
            pip_requirements += [mlflow_home]
        else:
            pip_requirements += [f"mlflow=={VERSION}"]

        python_env = _PythonEnv(
            python=python_version,
            dependencies=pip_requirements,
        )

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        fn._job_fn_metadata = JobFunctionMetadata(
            fn_fullname=f"{fn.__module__}.{fn.__name__}",
            max_workers=max_workers,
            transient_error_classes=transient_error_classes,
            python_env=python_env,
        )
        return fn

    return decorator


def submit_job(
    function: Callable[..., Any],
    params: dict[str, Any],
    timeout: float | None = None,
) -> JobEntity:
    """
    Submit a job to the job queue. The job is executed at most once.
    If the MLflow server crashes while the job is pending or running,
    it is rescheduled on restart.

    Note:
        This is a server-side API and requires the MLflow server to configure
        the backend store URI to a database URI.

    Args:
        function: The job function, it must be a python module-level function,
            and all params and return value must be JSON-serializable.
            The function can raise `TransientError` in order to trigger
            job retry, or you can annotate a list of transient error classes
            by ``@job(..., transient_error_classes=[...])``.
            You can set `MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES`
            to configure maximum allowed retries for transient errors
            and set `MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY` to
            configure base retry delay in seconds.

            The function must be decorated by `mlflow.server.jobs.job_function` decorator.
        params: The params to be passed to the job function.
        timeout: (optional) The job execution timeout, default None (no timeout)

    Returns:
        The job entity. You can call `get_job` API by the job id to get
        the updated job entity.
    """
    from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_JOB_EXECUTION
    from mlflow.server.jobs.utils import (
        _check_requirements,
        _get_or_init_huey_instance,
        _validate_function_parameters,
    )

    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        raise MlflowException(
            "Mlflow server job execution feature is not enabled, please set "
            "environment variable 'MLFLOW_SERVER_ENABLE_JOB_EXECUTION' to 'true' to enable it."
        )

    _check_requirements()

    if not (isinstance(function, FunctionType) and "." not in function.__qualname__):
        raise MlflowException("The job function must be a python global function.")

    func_fullname = f"{function.__module__}.{function.__name__}"

    if func_fullname not in _ALLOWED_JOB_FUNCTION_LIST:
        raise MlflowException.invalid_parameter_value(
            f"The function {func_fullname} is not in the allowed job function list"
        )

    if not hasattr(function, "_job_fn_metadata"):
        raise MlflowException(
            f"The job function {func_fullname} is not decorated by "
            "'mlflow.server.jobs.job_function'."
        )

    if not isinstance(params, dict):
        raise MlflowException.invalid_parameter_value(
            "When calling 'submit_job', the 'params' argument must be a dict."
        )
    # Validate that required parameters are provided
    _validate_function_parameters(function, params)

    job_store = _get_job_store()
    serialized_params = json.dumps(params)
    job = job_store.create_job(func_fullname, serialized_params, timeout)

    # enqueue job
    _get_or_init_huey_instance(func_fullname).submit_task(
        job.job_id,
        function,
        params,
        timeout,
    )

    return job


def get_job(job_id: str) -> JobEntity:
    """
    Get the job entity by the job id.

    Note:
        This is a server-side API, and it requires MLflow server configures
        backend store URI to database URI.

    Args:
        job_id: The job id.

    Returns:
        The job entity. If the job does not exist, error is raised.
    """
    job_store = _get_job_store()
    return job_store.get_job(job_id)
