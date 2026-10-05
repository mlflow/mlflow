"""Per-job executor backend selection."""

import logging
from typing import Any

from mlflow.environment_variables import (
    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
    MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
)
from mlflow.server.jobs.executor_registry import get_executor_registry

_logger = logging.getLogger(__name__)


def select_executor_backend(
    *,
    is_custom_scorer: bool,
    job_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Select the executor backend name for a job submission.

    Custom scorer jobs use ``MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND`` when it is set, unless
    ``job_name`` is given and that backend does not support the job (see
    ``AbstractJobExecutor.supports_job``), in which case the job stays on
    ``MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND``. All other jobs use the default backend.
    """
    default_backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
    custom_backend = MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get()
    if not (is_custom_scorer and custom_backend):
        return default_backend
    if job_name is not None and not get_executor_registry().get(custom_backend).supports_job(
        job_name, params or {}
    ):
        _logger.info(
            "Custom scorer job %s runs on the %r executor backend because the %r backend does not "
            "support it.",
            job_name,
            default_backend,
            custom_backend,
        )
        return default_backend
    return custom_backend
