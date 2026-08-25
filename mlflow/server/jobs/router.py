"""Per-job executor backend selection."""

from mlflow.environment_variables import (
    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
    MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
)
from mlflow.server.jobs.executor_registry import JobExecutorRegistry


class JobExecutorRouter:
    """Selects the executor backend name for a job submission.

    Owns backend-name selection only: it never inspects job payloads. Callers pass a
    pre-computed ``is_custom_scorer`` signal (see
    ``mlflow.genai.scorers.scorer_utils.params_contain_custom_scorer_code``).
    """

    def __init__(self, registry: JobExecutorRegistry) -> None:
        self._registry = registry

    def select(self, job_name: str, *, is_custom_scorer: bool) -> str:
        default_backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
        custom_backend = MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get()
        if is_custom_scorer and custom_backend:
            return custom_backend
        return default_backend
