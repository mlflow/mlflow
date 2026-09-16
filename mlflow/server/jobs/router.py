"""Per-job executor backend selection."""

from mlflow.environment_variables import (
    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
    MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
)


def select_executor_backend(*, is_custom_scorer: bool) -> str:
    """Select the executor backend name for a job submission.

    Custom scorer jobs use ``MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND`` when it is set;
    otherwise all jobs use ``MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND``.
    """
    default_backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
    custom_backend = MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get()
    if is_custom_scorer and custom_backend:
        return custom_backend
    return default_backend
