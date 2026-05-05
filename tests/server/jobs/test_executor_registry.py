from typing import Any
from unittest import mock

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import (
    AbstractJobExecutor,
    JobExecutionContext,
    JobExecutorConfig,
    JobRecoveryResult,
    JobResult,
)
from mlflow.server.jobs.executor_registry import (
    JobExecutorRegistry,
    _build_executor_config_from_env,
    _register_default_executors,
    get_executor_registry,
    shutdown_executor_registry,
    validate_executor_config,
)
from mlflow.utils.environment import _PythonEnv


class StubExecutor(AbstractJobExecutor):
    """Minimal concrete executor for testing."""

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
        pass

    def wait_for_job(self, job_id: str) -> JobResult:
        return JobResult(status=JobStatus.SUCCEEDED)

    def cancel_job(self, job_id: str) -> None:
        pass

    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        return []


class FailingRequirementsExecutor(StubExecutor):
    """Executor whose check_requirements raises."""

    def check_requirements(self) -> None:
        raise MlflowException("Docker daemon is not running")


# ---------------------------------------------------------------------------
# JobExecutorRegistry unit tests
# ---------------------------------------------------------------------------


def test_register_and_get():
    registry = JobExecutorRegistry()
    executor = StubExecutor(registry.config)
    registry.register("test-backend", executor)

    assert registry.get("test-backend") is executor
    assert "test-backend" in registry.get_registered_names()


def test_register_duplicate_name_raises():
    registry = JobExecutorRegistry()
    executor = StubExecutor(registry.config)
    registry.register("dup", executor)

    with pytest.raises(MlflowException, match="already registered"):
        registry.register("dup", StubExecutor(registry.config))


def test_get_unknown_name_raises():
    registry = JobExecutorRegistry()

    with pytest.raises(MlflowException, match="No job executor backend registered"):
        registry.get("nonexistent")


def test_get_unknown_name_lists_available():
    registry = JobExecutorRegistry()
    registry.register("alpha", StubExecutor(registry.config))

    with pytest.raises(MlflowException, match="alpha"):
        registry.get("nonexistent")


def test_validate_backends_success():
    registry = JobExecutorRegistry()
    registry.register("local", StubExecutor(registry.config))

    registry.validate_backends("local")


def test_validate_backends_failing_requirements():
    registry = JobExecutorRegistry()
    registry.register("docker", FailingRequirementsExecutor(registry.config))

    with pytest.raises(MlflowException, match="Docker daemon is not running"):
        registry.validate_backends("docker")


def test_validate_backends_unknown_name():
    registry = JobExecutorRegistry()

    with pytest.raises(MlflowException, match="No job executor backend registered"):
        registry.validate_backends("missing")


def test_get_registered_names_sorted():
    registry = JobExecutorRegistry()
    registry.register("zulu", StubExecutor(registry.config))
    registry.register("alpha", StubExecutor(registry.config))

    assert registry.get_registered_names() == ["alpha", "zulu"]


def test_custom_config_propagated():
    config = JobExecutorConfig(max_retries=10, default_timeout=7200.0)
    registry = JobExecutorRegistry(config)
    registry.register("test", StubExecutor(config))

    assert registry.get("test").config.max_retries == 10
    assert registry.get("test").config.default_timeout == 7200.0


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------


def test_register_default_executors():
    from mlflow.server.jobs.local_executor import LocalJobExecutor

    registry = JobExecutorRegistry()
    _register_default_executors(registry)

    assert "local" in registry.get_registered_names()
    assert isinstance(registry.get("local"), LocalJobExecutor)


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


def _make_entry_point(name: str, executor_cls: type):
    """Create a mock entry point that loads to the given class."""
    ep = mock.Mock()
    ep.name = name
    ep.value = f"{executor_cls.__module__}:{executor_cls.__qualname__}"
    ep.load.return_value = executor_cls
    return ep


def test_discover_and_register_from_entry_points():
    ep = _make_entry_point("remote", StubExecutor)

    registry = JobExecutorRegistry()
    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[ep]):
        registry.discover_and_register()

    assert "remote" in registry.get_registered_names()
    assert isinstance(registry.get("remote"), StubExecutor)


def test_discover_entry_point_conflicting_with_builtin_raises():
    ep = _make_entry_point("local", StubExecutor)

    registry = JobExecutorRegistry()
    _register_default_executors(registry)

    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[ep]):
        with pytest.raises(MlflowException, match="conflicts with an already-registered"):
            registry.discover_and_register()


def test_discover_non_executor_class_skips_with_warning():
    ep = _make_entry_point("bad", dict)

    registry = JobExecutorRegistry()
    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[ep]):
        registry.discover_and_register()

    assert "bad" not in registry.get_registered_names()


def test_discover_import_failure_skips_with_warning():
    ep = mock.Mock()
    ep.name = "broken"
    ep.value = "nonexistent.module:Cls"
    ep.load.side_effect = ImportError("No module named 'nonexistent'")

    registry = JobExecutorRegistry()
    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[ep]):
        registry.discover_and_register()

    assert "broken" not in registry.get_registered_names()


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------


def test_build_executor_config_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY", "2")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY", "30")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", "7")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT", "120")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_LEASE_TTL", "45")
    monkeypatch.setenv("MLFLOW_SERVER_COMPLETED_JOB_TTL", "600")

    config = _build_executor_config_from_env()

    assert config.retry_base_delay == 2
    assert config.retry_max_delay == 30
    assert config.max_retries == 7
    assert config.default_timeout == 120
    assert config.job_lease_ttl == 45
    assert config.completed_job_ttl == 600


# ---------------------------------------------------------------------------
# Lazy global singleton
# ---------------------------------------------------------------------------


def test_get_executor_registry_lazy_init(monkeypatch):
    import mlflow.server.jobs.executor_registry as mod

    mod._global_registry = None
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)

    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[]):
        registry = get_executor_registry()

    assert "local" in registry.get_registered_names()

    # Second call returns the same cached instance
    assert get_executor_registry() is registry

    shutdown_executor_registry()


def test_shutdown_executor_registry_is_idempotent():
    shutdown_executor_registry()
    shutdown_executor_registry()


# ---------------------------------------------------------------------------
# Early validation (called from _run_server)
# ---------------------------------------------------------------------------


def test_validate_executor_config_succeeds(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)

    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[]):
        validate_executor_config()


def test_validate_executor_config_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "nonexistent")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)

    with mock.patch("mlflow.server.jobs.executor_registry.get_entry_points", return_value=[]):
        with pytest.raises(MlflowException, match="No job executor backend registered"):
            validate_executor_config()


# ---------------------------------------------------------------------------
# AbstractJobExecutor defaults
# ---------------------------------------------------------------------------


def test_executor_remote_execution_defaults_to_false():
    executor = StubExecutor(JobExecutorConfig())
    assert executor.remote_execution is False


def test_executor_check_requirements_default_is_noop():
    executor = StubExecutor(JobExecutorConfig())
    executor.check_requirements()
