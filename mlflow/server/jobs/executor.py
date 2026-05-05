"""Executor interface and supporting types for the job execution framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from mlflow.entities._job_status import JobStatus
from mlflow.utils.environment import _PythonEnv


@dataclass
class JobExecutorConfig:
    """Framework-level configuration shared across executor instances."""

    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    max_retries: int = 3
    default_timeout: float = 3600.0
    job_lease_ttl: float = 60.0
    completed_job_ttl: float = 86400.0


@dataclass
class JobResult:
    """Result returned by an executor's ``wait_for_job()`` method."""

    status: JobStatus
    result: str | None = None
    error_message: str | None = None
    is_transient_error: bool = False


@dataclass
class JobExecutionContext:
    """Execution metadata the framework passes to the selected backend."""

    job_id: str
    tracking_uri: str
    gateway_uri: str | None = None
    token: str | None = None
    workspace: str | None = None


@dataclass
class JobRecoveryResult:
    """Per-job recovery outcome returned by ``recover_jobs()``."""

    job_id: str
    action: Literal["reattach", "requeue", "fail"]
    error_message: str | None = None


class AbstractJobExecutor(ABC):
    """Backend contract for job executor plugins."""

    def __init__(self, config: JobExecutorConfig) -> None:
        self._config = config

    @property
    def config(self) -> JobExecutorConfig:
        """The executor's framework configuration."""
        return self._config

    def start_executor(self) -> None:
        """Called during MLflow server startup.

        Subclasses may override this to acquire background resources.
        """

    def stop_executor(self) -> None:
        """Called during MLflow server shutdown for graceful cleanup.

        Subclasses may override this to release background resources.
        """

    @abstractmethod
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
        """Submit a job for execution.

        Args:
            job_id: Unique job identifier.
            job_name: Static name of the decorated job function.
            fn_fullname: Fully-qualified Python module path of the job function.
            params: Job parameters (JSON-serialisable).
            context: Execution metadata including tracking URI and auth.
            python_env: Optional Python environment specification.
            timeout: Optional per-job timeout in seconds.
        """

    @abstractmethod
    def wait_for_job(self, job_id: str) -> JobResult:
        """Block until the job reaches a terminal state.

        Returns:
            A ``JobResult`` with a terminal status.
        """

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Request cancellation of backend work for the given job."""

    @abstractmethod
    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        """Determine recovery action for jobs whose monitoring loop disappeared.

        Args:
            unfinished_job_ids: Job IDs that were in-flight when the previous
                monitoring loop stopped.

        Returns:
            One ``JobRecoveryResult`` per job indicating whether to reattach,
            requeue, or fail.
        """

    @property
    def remote_execution(self) -> bool:
        """Whether this executor runs jobs through the remote execution contract.

        When ``True``, the framework generates scoped tokens and requires
        Gateway-backed model URIs. Defaults to ``False``.
        """
        return False

    def check_requirements(self) -> None:
        """Optional fail-fast validation run during server startup.

        Subclasses may override this to verify that external dependencies
        are available (e.g. Docker daemon, Kubernetes API). Raise
        ``MlflowException`` if requirements are not met.
        """
