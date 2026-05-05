"""Executor plugin discovery, registration, and validation."""

import logging
import threading
import warnings

from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import AbstractJobExecutor, JobExecutorConfig
from mlflow.utils.plugins import get_entry_points

_logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "mlflow.job_executors"
DEFAULT_EXECUTOR_BACKEND = "local"

_global_registry_lock = threading.Lock()


class JobExecutorRegistry:
    """Discovers, instantiates, validates, and manages executor backends."""

    def __init__(self, config: JobExecutorConfig | None = None) -> None:
        self._config = config or JobExecutorConfig()
        self._executors: dict[str, AbstractJobExecutor] = {}

    @property
    def config(self) -> JobExecutorConfig:
        """The shared executor configuration."""
        return self._config

    def discover_and_register(self) -> None:
        """Load executor plugins from installed entry points.

        Raises:
            MlflowException: If two entry points declare the same backend name
                or collide with an already-registered built-in.
        """
        discovered = get_entry_points(ENTRY_POINT_GROUP)

        for ep in discovered:
            # Built-in backends are not overridable by entry-point plugins.
            if ep.name in self._executors:
                raise MlflowException(
                    f"Job executor backend name '{ep.name}' from entry point "
                    f"'{ep.value}' conflicts with an already-registered backend."
                )

            try:
                executor_cls = ep.load()
            except Exception as exc:
                warnings.warn(
                    f"Failure attempting to register job executor '{ep.name}': {exc}",
                    stacklevel=2,
                )
                continue

            if not (
                isinstance(executor_cls, type) and issubclass(executor_cls, AbstractJobExecutor)
            ):
                warnings.warn(
                    f"Failure attempting to register job executor '{ep.name}': "
                    f"entry point does not resolve to a subclass of AbstractJobExecutor.",
                    stacklevel=2,
                )
                continue

            try:
                executor = executor_cls(self._config)
            except Exception as exc:
                warnings.warn(
                    f"Failure attempting to register job executor '{ep.name}': "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                continue

            self._executors[ep.name] = executor
            _logger.debug("Registered job executor backend '%s' from '%s'", ep.name, ep.value)

        _logger.info(
            "Available backends after discovery: %s",
            ", ".join(sorted(self._executors)) or "(none)",
        )

    def register(self, name: str, executor: AbstractJobExecutor) -> None:
        """Manually register an executor backend."""
        if name in self._executors:
            raise MlflowException(f"Job executor backend '{name}' is already registered.")
        self._executors[name] = executor

    def get(self, name: str) -> AbstractJobExecutor:
        """Return the executor registered under the given backend name.

        Raises:
            MlflowException: If no executor is registered with that name.
        """
        if name not in self._executors:
            available = ", ".join(sorted(self._executors)) or "(none)"
            raise MlflowException(
                f"No job executor backend registered with name '{name}'. "
                f"Available backends: {available}"
            )
        return self._executors[name]

    def get_registered_names(self) -> list[str]:
        """Return all registered backend names in sorted order."""
        return sorted(self._executors)

    def validate_backends(self, *backend_names: str) -> None:
        """Run fail-fast validation for the given configured backend names."""
        for name in backend_names:
            executor = self.get(name)
            try:
                executor.check_requirements()
            except MlflowException:
                raise
            except Exception as e:
                raise MlflowException(
                    f"Requirements check failed for job executor backend '{name}': {e}"
                ) from e
            _logger.info("Job executor backend '%s' passed requirements check", name)


_global_registry: JobExecutorRegistry | None = None


def _register_default_executors(registry: JobExecutorRegistry) -> None:
    """Register built-in executor backends."""
    from mlflow.server.jobs.local_executor import LocalJobExecutor

    registry.register(DEFAULT_EXECUTOR_BACKEND, LocalJobExecutor(registry.config))


def _get_configured_backend_names() -> list[str]:
    """Return the list of configured backend names from environment variables."""
    from mlflow.environment_variables import (
        MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
        MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
    )

    default_backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
    custom_scorer_backend = MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get()

    backends = [default_backend]
    if custom_scorer_backend and custom_scorer_backend != default_backend:
        backends.append(custom_scorer_backend)
    return backends


def _build_validated_registry() -> tuple[JobExecutorRegistry, list[str]]:
    """Build a registry with built-ins, plugins, and validated backends."""
    config = _build_executor_config_from_env()
    registry = JobExecutorRegistry(config)
    _register_default_executors(registry)
    registry.discover_and_register()

    backends = _get_configured_backend_names()
    registry.validate_backends(*backends)
    return registry, backends


def get_executor_registry() -> JobExecutorRegistry:
    """Return the global executor registry, creating it on first access.

    On the first call in a given process the registry is built, built-in
    backends are registered, entry-point plugins are discovered, and configured
    backends are validated. Subsequent calls return the cached instance.
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                registry, _ = _build_validated_registry()
                _global_registry = registry
    return _global_registry


def validate_executor_config() -> None:
    """Validate configured executor backends and their requirements.

    Intended to be called from ``_run_server()`` in the parent launcher
    process for early fail-fast feedback. This builds a temporary registry,
    registers built-in backends, discovers entry-point plugins, and validates
    that configured backend names can be resolved and satisfy their runtime
    requirements.
    """
    _build_validated_registry()


def _build_executor_config_from_env() -> JobExecutorConfig:
    """Build ``JobExecutorConfig`` from the current environment variables."""
    from mlflow.environment_variables import (
        MLFLOW_SERVER_COMPLETED_JOB_TTL,
        MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT,
        MLFLOW_SERVER_JOB_LEASE_TTL,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
    )

    return JobExecutorConfig(
        retry_base_delay=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get(),
        retry_max_delay=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get(),
        max_retries=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get(),
        default_timeout=MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT.get(),
        job_lease_ttl=MLFLOW_SERVER_JOB_LEASE_TTL.get(),
        completed_job_ttl=MLFLOW_SERVER_COMPLETED_JOB_TTL.get(),
    )


def shutdown_executor_registry() -> None:
    """Clear the global registry.

    Safe to call even if the registry was never initialised.
    """
    global _global_registry
    if _global_registry is not None:
        _global_registry = None
