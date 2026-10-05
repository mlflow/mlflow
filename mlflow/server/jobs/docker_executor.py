"""Experimental Docker job executor backend that runs custom scorer code without network access.

The job runs in a hardened container with networking disabled (``network_mode="none"``), so the
scorer code cannot reach the MLflow server, the backend database, other services on the host, or
the internet. Data moves through bind-mounted files instead: before the run the host writes the
job's inputs (for example the traces to score) to a read-only input directory, the container writes
its result to an output directory, and the host then validates the result and records it (for
example by logging assessments). Only jobs that can be split this way are supported (see
``supports_job``); other custom scorer jobs stay on the default backend.

Limits of this mode:

- scorers must be pure compute: anything that needs the network fails inside the container;
- the image must already contain MLflow and the scorer's dependencies (nothing is installed at
  container start);
- the Docker daemon must run on the same host as the MLflow server, because inputs and results are
  exchanged through bind mounts.
"""

import json
import logging
import math
import os
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import mlflow
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_JOB_EXTRA_LABELS,
    MLFLOW_JOB_IMAGE,
    MLFLOW_JOBS_DOCKER_HOST,
)
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import (
    AbstractJobExecutor,
    JobExecutionContext,
    JobExecutorConfig,
    JobRecoveryResult,
    JobResult,
)
from mlflow.server.jobs.utils import (
    _JOB_ENTRY_MODULE,
    MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR,
    MLFLOW_SERVER_JOB_PARAMS_ENV_VAR,
    MLFLOW_SERVER_JOB_RESULT_DUMP_PATH_ENV_VAR,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_CLASSES_PATH_ENV_VAR,
    _load_function,
    _SubprocessJobResult,
)
from mlflow.server.sandbox.container import (
    _PIDS_LIMIT,
    _ensure_image,
    _is_read_timeout,
    _kill_quietly,
    _remove_quietly,
)
from mlflow.utils.environment import _PythonEnv

_logger = logging.getLogger(__name__)

LABEL_JOB_ID = "mlflow.job_id"
LABEL_JOB_NAME = "mlflow.job_name"

# Job types this backend can run, mapped to the host-side function that plans the run (see
# ``DockerJobPlan``). Keyed by job name so this module does not import the job modules eagerly.
_JOB_PLANNERS = {
    "invoke_scorer": "mlflow.genai.scorers.job._plan_invoke_scorer_job_for_docker",
}
# Optional per job type check of a job's parameters, for job types that this backend can run only
# for some parameters. A job it rejects runs on the default backend instead.
_JOB_SUPPORT_CHECKS = {
    "invoke_scorer": "mlflow.genai.scorers.job._docker_supports_invoke_scorer_params",
}

_CONTAINER_INPUT_DIR = "/job/input"
_CONTAINER_OUTPUT_DIR = "/job/output"
_RESULT_FILE = "result.json"
_TRANSIENT_ERROR_CLASSES_FILE = "transient_error_classes"

# Default resource limits for a job container, overridable per job type with
# ``MLFLOW_JOB_RESOURCE_LIMITS_<job_name>`` (see ``MLFLOW_JOB_IMAGE``).
_DEFAULT_MEMORY_BYTES = 1024**3
_DEFAULT_NANO_CPUS = 1_000_000_000

# The container writes the result file, so its size is untrusted; refuse to load anything larger.
_MAX_RESULT_BYTES = 64 * 1024**2

# How much of the container's logs to keep in a failure message, and how much the Docker daemon
# keeps on disk for a job container.
_LOG_TAIL_LINES = 50
_LOG_TAIL_BYTES = 4000
_LOG_CONFIG = {"type": "json-file", "config": {"max-size": "10m", "max-file": "1"}}

_MEMORY_SUFFIXES = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "k": 1000,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}


@dataclass
class DockerJobPlan:
    """How to run one job in a container that has no network access.

    Produced on the host by the job type's planner (see ``_JOB_PLANNERS``) after it has written the
    job's inputs to the input directory.

    Args:
        fn_fullname: Fully qualified name of the function the container runs. It reads its inputs
            from the input directory and returns a JSON-serializable value.
        params: Keyword arguments for that function.
        finalize: Runs on the host after the container exits successfully. It receives the value
            the container returned, which is untrusted, and returns the job's final result. It
            must validate that value and perform any writes the container could not (for example
            logging assessments).
    """

    fn_fullname: str
    params: dict[str, Any]
    finalize: Callable[[Any], Any]


@dataclass
class _DockerJob:
    workdir: tempfile.TemporaryDirectory
    deadline: float
    timeout: float
    plan: DockerJobPlan | None = None
    container_id: str | None = None
    cancel_requested: bool = False


def _parse_cpu(value: str) -> int:
    """Convert a CPU quantity such as ``"500m"`` or ``"2"`` to Docker nano-CPUs."""
    text = str(value).strip()
    cpus = float(text[:-1]) / 1000 if text.endswith("m") else float(text)
    if not math.isfinite(cpus) or cpus <= 0:
        raise ValueError(f"CPU limit must be positive, got {value!r}")
    return int(cpus * 1_000_000_000)


def _parse_memory(value: str) -> int:
    """Convert a memory quantity such as ``"512Mi"``, ``"1G"``, or ``"1048576"`` to bytes."""
    text = str(value).strip()
    for suffix, factor in sorted(_MEMORY_SUFFIXES.items(), key=lambda item: -len(item[0])):
        if text.endswith(suffix):
            amount = float(text[: -len(suffix)]) * factor
            break
    else:
        amount = float(text)
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"Memory limit must be positive, got {value!r}")
    return int(amount)


def _resource_limits(job_name: str) -> tuple[int, int]:
    """Return ``(memory_bytes, nano_cpus)`` for a job type, applying any configured override."""
    raw = os.environ.get(f"MLFLOW_JOB_RESOURCE_LIMITS_{job_name}") or os.environ.get(
        f"MLFLOW_JOB_RESOURCE_LIMITS_{job_name.upper()}"
    )
    if not raw:
        return _DEFAULT_MEMORY_BYTES, _DEFAULT_NANO_CPUS
    try:
        limits = json.loads(raw)
        if not isinstance(limits, dict):
            raise ValueError("expected a JSON object")
        memory = _parse_memory(limits["memory"]) if "memory" in limits else _DEFAULT_MEMORY_BYTES
        cpus = _parse_cpu(limits["cpu"]) if "cpu" in limits else _DEFAULT_NANO_CPUS
    except (ValueError, TypeError) as e:
        raise MlflowException.invalid_parameter_value(
            f"Invalid MLFLOW_JOB_RESOURCE_LIMITS_{job_name}: {e}. Expected a JSON object such as "
            '{"cpu": "500m", "memory": "1Gi"}.'
        ) from e
    return memory, cpus


def _container_labels(job_id: str, job_name: str) -> dict[str, str]:
    labels = {}
    if raw := MLFLOW_JOB_EXTRA_LABELS.get():
        try:
            extra = json.loads(raw)
        except ValueError as e:
            raise MlflowException.invalid_parameter_value(
                f"MLFLOW_JOB_EXTRA_LABELS must be a JSON object of strings: {e}"
            ) from e
        if not isinstance(extra, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra.items()
        ):
            raise MlflowException.invalid_parameter_value(
                "MLFLOW_JOB_EXTRA_LABELS must be a JSON object with string keys and values."
            )
        labels.update(extra)
    # The job labels are applied last so they cannot be overridden: recovery relies on them.
    labels[LABEL_JOB_ID] = job_id
    labels[LABEL_JOB_NAME] = job_name
    return labels


def _default_image() -> str:
    return f"mlflow-job:{mlflow.__version__}"


_RESULT_FIELDS = {f.name for f in fields(_SubprocessJobResult)}


def _load_result(path: Path) -> _SubprocessJobResult:
    """Load the container's result file, treating it and its contents as untrusted.

    The container can write anything to its output directory, so the result must be a regular file
    (not a symlink to a host file, a FIFO, or a device) no larger than ``_MAX_RESULT_BYTES``.
    """
    try:
        # O_NONBLOCK keeps opening a FIFO from blocking; it has no effect on regular files.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        raise MlflowException(f"The job container's result could not be opened: {e}") from e
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise MlflowException("The job container's result is not a regular file.")
        with os.fdopen(fd, "rb", closefd=False) as fp:
            data = fp.read(_MAX_RESULT_BYTES + 1)
    except OSError as e:
        raise MlflowException(f"The job container's result could not be read: {e}") from e
    finally:
        os.close(fd)
    if len(data) > _MAX_RESULT_BYTES:
        raise MlflowException(
            f"The job container's result is larger than {_MAX_RESULT_BYTES} bytes."
        )
    try:
        raw = json.loads(data)
    except (ValueError, RecursionError) as e:
        raise MlflowException(f"The job container wrote a malformed result: {e}") from e
    if (
        not isinstance(raw, dict)
        or not set(raw) <= _RESULT_FIELDS
        or not isinstance(raw.get("succeeded"), bool)
        or not isinstance(raw.get("result"), (str, type(None)))
        or not isinstance(raw.get("is_transient_error"), (bool, type(None)))
        or not isinstance(raw.get("error"), (str, type(None)))
    ):
        raise MlflowException("The job container wrote a malformed result.")
    return _SubprocessJobResult(
        succeeded=raw["succeeded"],
        result=raw.get("result"),
        is_transient_error=raw.get("is_transient_error"),
        error=raw.get("error"),
    )


def _log_tail(container) -> str:
    try:
        logs = container.logs(tail=_LOG_TAIL_LINES)
    except Exception:
        return ""
    return logs.decode("utf-8", errors="replace")[-_LOG_TAIL_BYTES:]


class DockerJobExecutor(AbstractJobExecutor):
    """Runs custom scorer jobs in Docker containers that have no network access.

    Experimental: see the module docstring for the limits of this mode.
    """

    def __init__(self, config: JobExecutorConfig) -> None:
        super().__init__(config)
        self._client = None
        self._client_lock = threading.Lock()
        self._jobs: dict[str, _DockerJob] = {}
        self._lock = threading.RLock()
        self._stopped = False

    def supports_job(self, job_name: str, params: dict[str, Any]) -> bool:
        if job_name not in _JOB_PLANNERS:
            return False
        check = _JOB_SUPPORT_CHECKS.get(job_name)
        return check is None or _load_function(check)(params)

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                import docker

                host = MLFLOW_JOBS_DOCKER_HOST.get()
                self._client = docker.DockerClient(base_url=host) if host else docker.from_env()
            return self._client

    def check_requirements(self) -> None:
        if os.name == "nt":
            raise MlflowException(
                "The docker job executor requires a POSIX host running Docker or Podman."
            )
        # Fail at startup rather than on every job if a container setting is malformed.
        for job_name in _JOB_PLANNERS:
            _resource_limits(job_name)
        _container_labels("", "")
        try:
            import docker  # noqa: F401
        except ImportError as e:
            raise MlflowException(
                f"The docker job executor requires the 'docker' package: {e}"
            ) from e
        try:
            self._get_client().ping()
        except Exception as e:
            raise MlflowException(
                f"The docker job executor cannot reach the Docker daemon: {e}"
            ) from e

    def start_executor(self) -> None:
        self.check_requirements()
        self._prepare_image()
        _logger.warning(
            "The docker job executor runs custom scorers without network access. Only on-demand "
            "scorer runs made only of custom scorers use it; other jobs with custom scorers, such "
            "as evaluation jobs, keep running on the default executor backend."
        )
        if os.getuid() == 0:
            _logger.warning(
                "The MLflow server runs as root, so docker job containers also run as root. Run "
                "the server as a non-root user to run job containers as a non-root user."
            )

    def stop_executor(self) -> None:
        # A job container cannot be resumed after the runner stops (its inputs and the host-side
        # step that records its result live in this process), so stop them rather than let them
        # run on unattended. ``wait_for_job`` then raises for these jobs instead of reporting a
        # result, so the runner leaves them for recovery, which requeues them on the next start.
        with self._lock:
            self._stopped = True
            jobs = list(self._jobs.items())
        for _, job in jobs:
            self._remove_container(job, kill=True)

    def _image(self) -> str:
        return MLFLOW_JOB_IMAGE.get() or _default_image()

    def _prepare_image(self) -> None:
        import docker.errors

        client = self._get_client()
        if configured := MLFLOW_JOB_IMAGE.get():
            try:
                client.images.get(configured)
            except docker.errors.ImageNotFound:
                _logger.info("Pulling job image %s", configured)
                client.images.pull(configured)
        else:
            _ensure_image(client, _default_image())

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
        if context.job_id != job_id:
            raise MlflowException.invalid_parameter_value(
                f"Job ID mismatch: context.job_id={context.job_id!r} does not match {job_id!r}"
            )
        if job_name not in _JOB_PLANNERS:
            raise MlflowException.invalid_parameter_value(
                f"The docker job executor does not support job type {job_name!r}."
            )
        if python_env is not None:
            raise MlflowException.invalid_parameter_value(
                "The docker job executor cannot install a job's Python environment because job "
                "containers have no network access. Provide an image that already contains the "
                "job's dependencies."
            )

        effective_timeout = timeout if timeout is not None else self.config.default_timeout
        job = _DockerJob(
            workdir=tempfile.TemporaryDirectory(prefix="mlflow-docker-job-"),
            deadline=time.monotonic() + effective_timeout,
            timeout=effective_timeout,
        )
        with self._lock:
            if self._stopped:
                job.workdir.cleanup()
                raise MlflowException("DockerJobExecutor is stopped and cannot accept new jobs.")
            if job_id in self._jobs:
                job.workdir.cleanup()
                raise MlflowException.invalid_parameter_value(
                    f"Job {job_id!r} is already being managed by DockerJobExecutor."
                )
            self._jobs[job_id] = job

        try:
            self._start_container(job_id, job_name, fn_fullname, params, job)
        except Exception:
            self._finalize(job_id, job)
            raise

    def _start_container(
        self,
        job_id: str,
        job_name: str,
        fn_fullname: str,
        params: dict[str, Any],
        job: _DockerJob,
    ) -> None:
        input_dir = Path(job.workdir.name, "input")
        output_dir = Path(job.workdir.name, "output")
        input_dir.mkdir()
        output_dir.mkdir()

        planner = _load_function(_JOB_PLANNERS[job_name])
        job.plan = planner(
            params=params, input_dir=input_dir, container_input_dir=_CONTAINER_INPUT_DIR
        )

        transient_error_classes = _load_function(
            fn_fullname
        )._job_fn_metadata.transient_error_classes
        (input_dir / _TRANSIENT_ERROR_CLASSES_FILE).write_text(
            "".join(f"{cls.__module__}.{cls.__name__}\n" for cls in transient_error_classes or [])
        )

        environment = {
            MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR: job.plan.fn_fullname,
            MLFLOW_SERVER_JOB_PARAMS_ENV_VAR: json.dumps(job.plan.params),
            MLFLOW_SERVER_JOB_RESULT_DUMP_PATH_ENV_VAR: f"{_CONTAINER_OUTPUT_DIR}/{_RESULT_FILE}",
            MLFLOW_SERVER_JOB_TRANSIENT_ERROR_CLASSES_PATH_ENV_VAR: (
                f"{_CONTAINER_INPUT_DIR}/{_TRANSIENT_ERROR_CLASSES_FILE}"
            ),
            # The operator enabled custom scorers for this server (otherwise the job could not have
            # been submitted), and the container is where their code is meant to run.
            "MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS": "true",
            # There is nowhere to send scorer traces or telemetry without network access.
            "MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING": "false",
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "DO_NOT_TRACK": "true",
            # The root filesystem is read-only; give tools that write under HOME a writable one.
            "HOME": "/tmp",
        }

        with self._lock:
            if self._stopped or self._jobs.get(job_id) is not job or job.cancel_requested:
                return
            try:
                container = self._run_container(
                    job_id, job_name, input_dir, output_dir, environment
                )
            except Exception:
                # docker-py creates the container before starting it and does not remove it if
                # the start fails, and no container ID was recorded, so find it by its label.
                self._remove_job_containers(job_id)
                raise
            job.container_id = container.id
        _logger.info(
            "Started docker job container %s for job %s (%s)", container.short_id, job_id, job_name
        )

    def _run_container(
        self,
        job_id: str,
        job_name: str,
        input_dir: Path,
        output_dir: Path,
        environment: dict[str, str],
    ):
        import docker.types

        memory, nano_cpus = _resource_limits(job_name)
        return self._get_client().containers.run(
            self._image(),
            command=["python", "-m", _JOB_ENTRY_MODULE],
            detach=True,
            labels=_container_labels(job_id, job_name),
            network_mode="none",
            mem_limit=memory,
            memswap_limit=memory,
            nano_cpus=nano_cpus,
            pids_limit=_PIDS_LIMIT,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            tmpfs={"/tmp": ""},
            log_config=_LOG_CONFIG,
            # Caps every file the job writes, including in the bind-mounted output directory,
            # which has no quota of its own.
            ulimits=[
                docker.types.Ulimit(name="fsize", soft=_MAX_RESULT_BYTES, hard=_MAX_RESULT_BYTES)
            ],
            # Run as the server's user so the result file is owned by the server, not root. If the
            # server runs as root the container does too (``start_executor`` warns about it).
            user=f"{os.getuid()}:{os.getgid()}",
            working_dir=_CONTAINER_OUTPUT_DIR,
            environment=environment,
            volumes={
                str(input_dir): {"bind": _CONTAINER_INPUT_DIR, "mode": "ro"},
                str(output_dir): {"bind": _CONTAINER_OUTPUT_DIR, "mode": "rw"},
            },
        )

    def wait_for_job(self, job_id: str) -> JobResult:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise MlflowException.invalid_parameter_value(
                f"Unknown job ID for DockerJobExecutor: {job_id!r}"
            )
        try:
            return self._wait(job_id, job)
        finally:
            self._finalize(job_id, job)

    def _wait(self, job_id: str, job: _DockerJob) -> JobResult:
        if job.container_id is None:
            if job.cancel_requested:
                return JobResult(status=JobStatus.CANCELED)
            self._raise_if_stopped(job_id)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=f"The container for job {job_id!r} was never started.",
            )

        timed_out = JobResult(
            status=JobStatus.TIMEOUT,
            error_message=f"Docker job {job_id!r} timed out after {job.timeout} seconds.",
        )
        container = None
        try:
            container = self._get_client().containers.get(job.container_id)
            # The deadline also covers the host-side trace prefetch, which may have used it all.
            # A zero timeout would be rejected by the HTTP client rather than time out.
            remaining = job.deadline - time.monotonic()
            if remaining <= 0:
                _kill_quietly(container)
                return timed_out
            outcome = container.wait(timeout=remaining)
        except Exception as e:
            if container is not None:
                _kill_quietly(container)
            if job.cancel_requested:
                return JobResult(status=JobStatus.CANCELED)
            if _is_read_timeout(e):
                return timed_out
            self._raise_if_stopped(job_id)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=f"Lost track of the container for job {job_id!r}: {e}",
            )

        if job.cancel_requested:
            return JobResult(status=JobStatus.CANCELED)
        self._raise_if_stopped(job_id)

        exit_code = outcome.get("StatusCode")
        result_path = Path(job.workdir.name, "output", _RESULT_FILE)
        if not os.path.lexists(result_path):
            logs = _log_tail(container)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=(
                    f"The container for job {job_id!r} exited with code {exit_code} without "
                    f"writing a result. Container logs:\n{logs}"
                ),
            )

        try:
            subprocess_result = _load_result(result_path)
        except MlflowException as e:
            return JobResult(status=JobStatus.FAILED, error_message=e.message)
        if not subprocess_result.succeeded:
            return JobResult(
                status=JobStatus.FAILED,
                error_message=subprocess_result.error,
                is_transient_error=bool(subprocess_result.is_transient_error),
            )

        try:
            value = json.loads(subprocess_result.result) if subprocess_result.result else None
            final = job.plan.finalize(value)
        except Exception as e:
            _logger.exception("Failed to record the result of docker job %s", job_id)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=f"Failed to record the result of docker job {job_id!r}: {e!r}",
            )
        return JobResult(status=JobStatus.SUCCEEDED, result=json.dumps(final))

    def _raise_if_stopped(self, job_id: str) -> None:
        if self._stopped:
            raise MlflowException(
                f"The docker job executor stopped while job {job_id!r} was running; the job will "
                "be recovered on the next start."
            )

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.cancel_requested = True
            container_id = job.container_id
        if container_id is not None:
            try:
                self._get_client().containers.get(container_id).kill()
            except Exception:
                _logger.debug("Could not kill container for job %s", job_id, exc_info=True)

    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        # A container left by a previous runner cannot be resumed: the host-side step that records
        # its result lived in that process. Remove any leftover container and requeue the job.
        client = self._get_client()
        for job_id in unfinished_job_ids:
            try:
                leftovers = client.containers.list(
                    all=True, filters={"label": f"{LABEL_JOB_ID}={job_id}"}
                )
            except Exception:
                _logger.warning(
                    "Could not list leftover containers for job %s", job_id, exc_info=True
                )
                continue
            for container in leftovers:
                _kill_quietly(container)
                _remove_quietly(container)
        return [JobRecoveryResult(job_id=job_id, action="requeue") for job_id in unfinished_job_ids]

    def _remove_job_containers(self, job_id: str) -> None:
        try:
            containers = self._get_client().containers.list(
                all=True, filters={"label": f"{LABEL_JOB_ID}={job_id}"}
            )
        except Exception:
            _logger.debug("Could not list containers for job %s", job_id, exc_info=True)
            return
        for container in containers:
            _remove_quietly(container)

    def _remove_container(self, job: _DockerJob, kill: bool = False) -> None:
        if job.container_id is None:
            return
        try:
            container = self._get_client().containers.get(job.container_id)
        except Exception:
            return
        if kill:
            _kill_quietly(container)
        _remove_quietly(container)

    def _finalize(self, job_id: str, job: _DockerJob) -> None:
        with self._lock:
            if self._jobs.get(job_id) is job:
                self._jobs.pop(job_id, None)
        self._remove_container(job)
        job.workdir.cleanup()
