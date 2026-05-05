"""Built-in local subprocess job executor."""

from typing import Any

from mlflow.server.jobs.executor import (
    AbstractJobExecutor,
    JobExecutionContext,
    JobExecutorConfig,
    JobRecoveryResult,
    JobResult,
)
from mlflow.utils.environment import _PythonEnv


class LocalJobExecutor(AbstractJobExecutor):
    """Default executor that runs jobs as local subprocesses.

    Args:
        config: Framework-level executor configuration.
    """

    def __init__(self, config: JobExecutorConfig) -> None:
        super().__init__(config)

    def submit_job(
        self,
        job_id: str,
        job_name: str,
        fn_fullname: str,
        params: dict[str, Any],
        context: JobExecutionContext,
        python_env: _PythonEnv | None = None,
        timeout: float | None = None,
    ) -> None:
        raise NotImplementedError("LocalJobExecutor.submit_job() is not implemented yet.")

    def wait_for_job(self, job_id: str) -> JobResult:
        raise NotImplementedError("LocalJobExecutor.wait_for_job() is not implemented yet.")

    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError("LocalJobExecutor.cancel_job() is not implemented yet.")

    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        raise NotImplementedError("LocalJobExecutor.recover_jobs() is not implemented yet.")
