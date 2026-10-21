"""Per-job executor backend selection."""

from typing import Any

from mlflow.environment_variables import (
    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
    MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.scorer_utils import is_online_scoring_job
from mlflow.server.jobs.executor_registry import get_executor_registry


def select_executor_backend(
    *,
    is_custom_scorer: bool,
    job_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Select the executor backend name for a job submission.

    Custom scorer jobs use ``MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND`` when it is set. Setting it
    means custom scorer code runs only in that backend, so a custom scorer job that the backend
    does not support (see ``AbstractJobExecutor.supports_job``) is refused rather than run on
    ``MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND``. The exception is an online scoring job, which runs on
    the default backend and runs its custom scorers only in job containers of its own, or skips
    them. All other jobs use the default backend.
    """
    default_backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
    custom_backend = MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get()
    if not (is_custom_scorer and custom_backend):
        return default_backend
    if job_name is not None and not get_executor_registry().get(custom_backend).supports_job(
        job_name, params or {}
    ):
        if is_online_scoring_job(job_name):
            return default_backend
        raise MlflowException.invalid_parameter_value(
            "This job includes code-based scorers, but the "
            f"{custom_backend!r} executor backend that runs code-based scorers on this server "
            "(MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND) cannot run it, and the server does not "
            "run code-based scorers anywhere else, so the job was not started. Remove the "
            "code-based scorers from the job, or run them with mlflow.genai.evaluate() instead."
        )
    return custom_backend
