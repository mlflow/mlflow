"""A single job container that runs without network access, shared by the docker job executor
backend and by jobs that run part of their work in a container.

A job is split into a host side and a container side (see ``DockerJobPlan``). The host writes the
job's inputs to an input directory, the container reads them and writes its result to an output
directory, and the host then checks that untrusted result and records it. See the docker executor
module (``mlflow.server.jobs.docker_executor``) for the limits of this mode.
"""

import json
import logging
import math
import os
import stat
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import mlflow
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_JOB_EXTRA_LABELS
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import JobResult
from mlflow.server.jobs.utils import (
    _JOB_ENTRY_MODULE,
    MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR,
    MLFLOW_SERVER_JOB_PARAMS_ENV_VAR,
    MLFLOW_SERVER_JOB_RESULT_DUMP_PATH_ENV_VAR,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_CLASSES_PATH_ENV_VAR,
    _load_function,
    _SubprocessJobResult,
    get_job_fn_fullname,
)
from mlflow.server.sandbox.container import (
    _PIDS_LIMIT,
    _is_read_timeout,
    _kill_quietly,
    _remove_quietly,
)

_logger = logging.getLogger(__name__)

LABEL_JOB_ID = "mlflow.job_id"
LABEL_JOB_NAME = "mlflow.job_name"

_CONTAINER_INPUT_DIR = "/job/input"
_CONTAINER_OUTPUT_DIR = "/job/output"
_RESULT_FILE = "result.json"
_TRANSIENT_ERROR_CLASSES_FILE = "transient_error_classes"

# Default resource limits for a job container. A job type can declare its own with
# ``@job(resource_limits=...)``, and ``MLFLOW_JOB_RESOURCE_LIMITS_<job_name>`` overrides both (see
# ``MLFLOW_JOB_IMAGE``).
_DEFAULT_MEMORY_BYTES = 1024**3
_DEFAULT_NANO_CPUS = 1_000_000_000

# The container writes the result file, so its size is untrusted; refuse to load anything larger.
# Parsing JSON can take far more memory than its size (around 20x for many small numbers), and the
# host parses the result before checking it, so this cap also bounds that memory per job. Results
# are a few assessments per trace, so this leaves ample room.
_MAX_RESULT_BYTES = 8 * 1024**2
# Caps every file the job writes (the ``fsize`` ulimit), including scratch files under /tmp and in
# the output directory, which has no quota of its own. Larger than the result cap so a scorer's own
# files are not limited by it, and a too-large result fails on the host with a clear error.
_MAX_FILE_BYTES = 64 * 1024**2
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

    Produced on the host after the job's inputs have been written to the input directory.

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


def _resource_limits(job_name: str, declared: dict[str, str] | None = None) -> tuple[int, int]:
    """Return ``(memory_bytes, nano_cpus)`` for a job type.

    Each limit comes from ``MLFLOW_JOB_RESOURCE_LIMITS_<job_name>`` if set there, else from the
    limits the job type declares with ``@job(resource_limits=...)`` (``declared``), else from the
    defaults.
    """
    raw = os.environ.get(f"MLFLOW_JOB_RESOURCE_LIMITS_{job_name}") or os.environ.get(
        f"MLFLOW_JOB_RESOURCE_LIMITS_{job_name.upper()}"
    )
    limits = dict(declared or {})
    try:
        if raw:
            overrides = json.loads(raw)
            if not isinstance(overrides, dict):
                raise ValueError("expected a JSON object")
            limits.update(overrides)
        memory = _parse_memory(limits["memory"]) if "memory" in limits else _DEFAULT_MEMORY_BYTES
        cpus = _parse_cpu(limits["cpu"]) if "cpu" in limits else _DEFAULT_NANO_CPUS
    except (ValueError, TypeError) as e:
        raise MlflowException.invalid_parameter_value(
            f"Invalid resource limits for job type {job_name!r} (from "
            f"MLFLOW_JOB_RESOURCE_LIMITS_{job_name} or the job's declared resource_limits): {e}. "
            'Expected an object such as {"cpu": "500m", "memory": "1Gi"}.'
        ) from e
    return memory, cpus


def _declared_resource_limits(job_name: str) -> dict[str, str] | None:
    """The resource limits a job type declares with ``@job(resource_limits=...)``, if known."""
    try:
        fn_fullname = get_job_fn_fullname(job_name)
    except MlflowException:
        return None
    return _load_function(fn_fullname)._job_fn_metadata.resource_limits


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
            f"The job container's result is larger than {_MAX_RESULT_BYTES} bytes. Score fewer "
            "traces per job (for on-demand single-turn scorer runs, lower "
            "MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE)."
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


def make_job_container_dirs(workdir: Path) -> tuple[Path, Path]:
    """Create the ``(input_dir, output_dir)`` a job container mounts, under ``workdir``."""
    input_dir = workdir / "input"
    output_dir = workdir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


def write_transient_error_classes(
    input_dir: Path, transient_error_classes: list[type[Exception]] | None
) -> None:
    """Write the exception classes the container should report as transient errors."""
    (input_dir / _TRANSIENT_ERROR_CLASSES_FILE).write_text(
        "".join(f"{cls.__module__}.{cls.__name__}\n" for cls in transient_error_classes or [])
    )


def job_container_environment(plan: DockerJobPlan) -> dict[str, str]:
    """The environment of a job container that runs ``plan``'s container side."""
    return {
        MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR: plan.fn_fullname,
        MLFLOW_SERVER_JOB_PARAMS_ENV_VAR: json.dumps(plan.params),
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


def run_job_container(
    client,
    image: str,
    job_id: str,
    job_name: str,
    input_dir: Path,
    output_dir: Path,
    environment: dict[str, str],
    memory: int,
    nano_cpus: int,
):
    """Create and start a hardened job container with no network access, and return it."""
    import docker.types

    return client.containers.run(
        image,
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
        ulimits=[docker.types.Ulimit(name="fsize", soft=_MAX_FILE_BYTES, hard=_MAX_FILE_BYTES)],
        # Run as the server's user so the result file is owned by the server, not root. If the
        # server runs as root the container does too (the docker executor warns about it).
        user=f"{os.getuid()}:{os.getgid()}",
        working_dir=_CONTAINER_OUTPUT_DIR,
        environment=environment,
        volumes={
            str(input_dir): {"bind": _CONTAINER_INPUT_DIR, "mode": "ro"},
            str(output_dir): {"bind": _CONTAINER_OUTPUT_DIR, "mode": "rw"},
        },
    )


def remove_job_containers(client, job_id: str) -> None:
    """Remove every container labeled with ``job_id``, for example one that failed to start.

    So a job must run at most one container at a time.
    """
    try:
        containers = client.containers.list(all=True, filters={"label": f"{LABEL_JOB_ID}={job_id}"})
    except Exception:
        _logger.debug("Could not list containers for job %s", job_id, exc_info=True)
        return
    for container in containers:
        _remove_quietly(container)


def job_result_from_container(
    container, exit_code: int | None, output_dir: Path, job_id: str, plan: DockerJobPlan
) -> JobResult:
    """The result of a job whose container exited: its untrusted result, checked and finalized."""
    result_path = output_dir / _RESULT_FILE
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
        final = plan.finalize(value)
    except Exception as e:
        _logger.exception("Failed to record the result of docker job %s", job_id)
        return JobResult(
            status=JobStatus.FAILED,
            error_message=f"Failed to record the result of docker job {job_id!r}: {e!r}",
        )
    return JobResult(status=JobStatus.SUCCEEDED, result=json.dumps(final))


def run_plan_in_container(
    client,
    image: str,
    *,
    job_id: str,
    job_name: str,
    plan: DockerJobPlan,
    input_dir: Path,
    output_dir: Path,
    timeout: float,
    transient_error_classes: list[type[Exception]] | None = None,
    resource_limits: dict[str, str] | None = None,
) -> JobResult:
    """Run ``plan``'s container side in a job container and wait for it, then finalize its result.

    For work that runs part of a job in a container, synchronously. ``input_dir`` and
    ``output_dir`` come from ``make_job_container_dirs``, with the plan's inputs already written to
    the input directory. The result's ``result`` is the finalized value as JSON.

    ``image`` must already be present: a missing image is pulled inside the start, which counts
    against ``timeout``. ``timeout`` bounds starting the container and waiting for it to exit. The
    steps after that are not part of it: fetching the logs of a container that wrote no result and
    removing the container are bounded only by the Docker client's request timeout, and
    ``plan.finalize`` runs on the host with no time limit, so a caller with a deadline must leave
    time for them. The container is removed before this returns, but not if this process is killed
    while it runs; the caller must then remove it (it is labeled with ``job_id``).
    """
    timed_out = JobResult(
        status=JobStatus.TIMEOUT,
        error_message=f"Docker job {job_id!r} timed out after {timeout} seconds.",
    )
    if timeout <= 0:
        return timed_out
    write_transient_error_classes(input_dir, transient_error_classes)
    memory, nano_cpus = _resource_limits(job_name, resource_limits)
    environment = job_container_environment(plan)
    deadline = time.monotonic() + timeout
    try:
        container = run_job_container(
            client,
            image,
            job_id,
            job_name,
            input_dir,
            output_dir,
            environment,
            memory,
            nano_cpus,
        )
    except Exception:
        # docker-py creates the container before starting it and does not remove it if the start
        # fails, so find it by its label.
        remove_job_containers(client, job_id)
        raise
    try:
        # A zero timeout would be rejected by the HTTP client rather than time out.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_quietly(container)
            return timed_out
        try:
            outcome = container.wait(timeout=remaining)
        except Exception as e:
            _kill_quietly(container)
            if _is_read_timeout(e):
                return timed_out
            return JobResult(
                status=JobStatus.FAILED,
                error_message=f"Lost track of the container for job {job_id!r}: {e}",
            )
        return job_result_from_container(
            container, outcome.get("StatusCode"), output_dir, job_id, plan
        )
    finally:
        _remove_quietly(container)
