"""Executor plugin discovery, registration, and validation."""

import logging
import math
import threading
import warnings

from mlflow.environment_variables import MLFLOW_SERVER_ENABLE_JOB_EXECUTION
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import (
    _MIN_JOB_LEASE_TTL,
    AbstractJobExecutor,
    JobExecutorConfig,
)
from mlflow.utils.plugins import get_entry_points

_logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "mlflow.job_executors"
DEFAULT_EXECUTOR_BACKEND = "local"
DOCKER_EXECUTOR_BACKEND = "docker"

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
            if executor.remote_execution:
                # Remote execution (running a job off the server box and handing it credentials
                # through a scoped token) is not implemented yet: no in-tree executor sets
                # remote_execution=True, and the framework cannot yet provision a remote job's
                # auth. Refuse to start with such a backend configured, so that adding a remote
                # executor has to remove this guard and, at that point, wire up the direct-provider
                # model validation in submit_job that it gates.
                raise MlflowException.invalid_parameter_value(
                    f"Job executor backend '{name}' declares remote_execution=True, which is not "
                    "supported yet. Configure a local executor backend instead."
                )
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
    """Register built-in executor backends.

    Registering a backend does not touch its dependencies; only configured backends are validated
    (see ``validate_backends``), so an unused ``docker`` backend never contacts a Docker daemon.
    """
    from mlflow.server.jobs.docker_executor import DockerJobExecutor
    from mlflow.server.jobs.local_executor import LocalJobExecutor

    registry.register(DEFAULT_EXECUTOR_BACKEND, LocalJobExecutor(registry.config))
    registry.register(DOCKER_EXECUTOR_BACKEND, DockerJobExecutor(registry.config))


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


def _executor_engine_in_use() -> bool:
    """Whether jobs from this process run on the executor engine, so it has a registry to build."""
    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        return False
    # Imported here: importing the job utils loads the job functions, and this module can be
    # imported while one of their modules (for example mlflow.genai.scorers.job) is still loading.
    from mlflow.server.jobs.utils import _check_requirements, get_job_execution_engine

    try:
        _check_requirements()
        return get_job_execution_engine() == "executor"
    except MlflowException:
        # Job execution is unavailable in this process; submit_job reports why when it is used.
        return False


def _build_registry_or_log() -> None:
    try:
        get_executor_registry()
    except Exception as e:
        _logger.warning(
            "Could not check the job executor backends when this process started; job "
            "submissions from it will check them again: %s",
            e,
            exc_info=not isinstance(e, MlflowException),
        )


def warm_executor_registry_in_background() -> threading.Thread | None:
    """Build and check this process's executor registry now, when jobs use the executor engine.

    ``get_executor_registry`` builds the registry the first time a process needs it, and caches it
    only when every configured backend passes its checks. If a backend is unavailable at that
    moment (for example the Docker daemon), the process cannot submit any job until the backend
    is back. These server processes, started by ``mlflow server``, build it when they start, just
    after the server checked the same backends, so that a backend that becomes unavailable later
    does not stop them from submitting jobs:

    - the periodic task process, which runs the online scoring scheduler, when it starts;
    - each uvicorn worker, at startup (the FastAPI lifespan);
    - each gunicorn or waitress worker, on its first request (the Flask app has no startup hook).

    The registry is built in a daemon thread, so a slow check (for example a Docker daemon that
    does not answer) does not delay the caller. A job submitted meanwhile waits for it, because
    ``get_executor_registry`` builds the registry under a lock. A failed check is logged, not
    raised: the process then builds the registry on first use, as before. That is also what
    happens in a process started while the backend is unavailable.

    Whether the executor engine is used is checked first, on the calling thread, and no thread is
    started otherwise. A thread that is importing modules when the process forks (as
    ``multiprocessing`` does on Linux) can leave the child process deadlocked, so most processes
    should not start one.
    """
    if not _executor_engine_in_use():
        return None
    thread = threading.Thread(
        target=_build_registry_or_log, name="MLflow-warm-executor-registry", daemon=True
    )
    thread.start()
    return thread


def validate_executor_config() -> None:
    """Validate configured executor backends and their requirements.

    Intended to be called from ``_run_server()`` in the parent launcher
    process for early fail-fast feedback. This builds a temporary registry,
    registers built-in backends, discovers entry-point plugins, and validates
    that configured backend names can be resolved and satisfy their runtime
    requirements.
    """
    from mlflow.environment_variables import (
        MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
        MLFLOW_SERVER_JOB_EXECUTION_ENGINE,
    )
    from mlflow.server.jobs.utils import get_job_execution_engine

    custom_scorer_backend = MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get()
    if (
        custom_scorer_backend
        and custom_scorer_backend != DEFAULT_EXECUTOR_BACKEND
        and get_job_execution_engine() != "executor"
    ):
        # Only the executor engine runs jobs on a backend. Without it, custom scorers would
        # silently run in the server's local job runner instead of the configured backend (for
        # example outside the docker sandbox), even when the default backend names the same one,
        # so refuse to start.
        raise MlflowException.invalid_parameter_value(
            f"MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND={custom_scorer_backend!r} requires "
            f"{MLFLOW_SERVER_JOB_EXECUTION_ENGINE.name}=executor."
        )
    _build_validated_registry()


def _build_executor_config_from_env() -> JobExecutorConfig:
    """Build ``JobExecutorConfig`` from the current environment variables."""
    from mlflow.environment_variables import (
        MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT,
        MLFLOW_SERVER_JOB_LEASE_TTL,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
    )

    job_lease_ttl = MLFLOW_SERVER_JOB_LEASE_TTL.get()
    if not math.isfinite(job_lease_ttl) or job_lease_ttl < _MIN_JOB_LEASE_TTL:
        raise MlflowException.invalid_parameter_value(
            f"MLFLOW_SERVER_JOB_LEASE_TTL must be finite and at least "
            f"{_MIN_JOB_LEASE_TTL} seconds, but got {job_lease_ttl}."
        )

    return JobExecutorConfig(
        retry_base_delay=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get(),
        retry_max_delay=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get(),
        max_retries=MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get(),
        default_timeout=MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT.get(),
        job_lease_ttl=job_lease_ttl,
    )


def shutdown_executor_registry() -> None:
    """Clear the global registry.

    Safe to call even if the registry was never initialised.
    """
    global _global_registry
    if _global_registry is not None:
        _global_registry = None
