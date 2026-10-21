"""Experimental Docker job executor backend that runs custom scorer code without network access.

The job runs in a hardened container with networking disabled (``network_mode="none"``), so the
scorer code cannot reach the MLflow server, the backend database, other services on the host, or
the internet. Data moves through bind-mounted files instead: before the run the host writes the
job's inputs (for example the traces to score) to a read-only input directory, the container writes
its result to an output directory, and the host then validates the result and records it (for
example by logging assessments). Only jobs that can be split this way are supported (see
``supports_job``); other custom scorer jobs are refused, except online scoring jobs, which
run on the default backend and start job containers of their own.

Limits of this mode:

- scorers must be pure compute: anything that needs the network fails inside the container;
- the image must already contain MLflow and the scorer's dependencies (nothing is installed at
  container start);
- the Docker daemon must run on the same host as the MLflow server, because inputs and results are
  exchanged through bind mounts;
- the output directory, a host temporary directory, has no size limit of its own: each file the
  job writes is capped (see ``job_container._MAX_FILE_BYTES``), but not the number of files, so a
  malicious scorer can fill the host disk (or memory, where the temporary directory is a tmpfs)
  until the job times out;
- stopping the MLflow server does not stop job containers that are running: the job runner exits
  without stopping them, and their time limit is enforced from the host. They are removed when the
  job runner next starts (or with ``docker rm -f $(docker ps -aq --filter label=mlflow.job_id)``).
"""

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import (
    AbstractJobExecutor,
    JobExecutionContext,
    JobExecutorConfig,
    JobRecoveryResult,
    JobResult,
)
from mlflow.server.jobs.job_container import (
    _CONTAINER_INPUT_DIR,
    LABEL_JOB_ID,
    DockerJobPlan,
    _container_labels,
    _declared_resource_limits,
    _resource_limits,
    job_container_environment,
    job_docker_client,
    job_image,
    job_result_from_container,
    make_job_container_dirs,
    prepare_job_image,
    remove_job_containers,
    run_job_container,
    write_transient_error_classes,
)
from mlflow.server.jobs.utils import _load_function
from mlflow.server.sandbox.container import (
    _is_read_timeout,
    _kill_quietly,
    _remove_quietly,
)
from mlflow.utils.environment import _PythonEnv

_logger = logging.getLogger(__name__)

# Job types this backend can run, mapped to the host-side function that plans the run (see
# ``DockerJobPlan``). Keyed by job name so this module does not import the job modules eagerly.
_JOB_PLANNERS = {
    "invoke_scorer": "mlflow.genai.scorers.job._plan_invoke_scorer_job_for_docker",
}
# Optional per job type check of a job's parameters, for job types that this backend can run only
# for some parameters. A custom scorer job it rejects is refused, except an online scoring job (see
# ``select_executor_backend``).
_JOB_SUPPORT_CHECKS = {
    "invoke_scorer": "mlflow.genai.scorers.job._docker_supports_invoke_scorer_params",
}

# Online scoring jobs that run custom scorers in job containers of their own (see
# ``mlflow.genai.scorers.online.container_scoring``) rather than through this executor.
_ONLINE_CONTAINER_JOB_NAMES = ("run_online_trace_scorer", "run_online_session_scorer")

# How long ``stop_executor`` waits for container starts in progress before giving up on them, which
# bounds how long that wait can delay shutdown when the Docker daemon hangs.
_STOP_WAIT_FOR_STARTS_SECONDS = 30.0


@dataclass
class _DockerJob:
    workdir: tempfile.TemporaryDirectory
    deadline: float
    timeout: float
    plan: DockerJobPlan | None = None
    container_id: str | None = None
    cancel_requested: bool = False
    starting: bool = False


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
        # Notified when a container start finishes, for ``stop_executor``.
        self._start_finished = threading.Condition(self._lock)
        self._stopped = False

    def supports_job(self, job_name: str, params: dict[str, Any]) -> bool:
        if job_name not in _JOB_PLANNERS:
            return False
        check = _JOB_SUPPORT_CHECKS.get(job_name)
        return check is None or _load_function(check)(params)

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = job_docker_client()
            return self._client

    def check_requirements(self) -> None:
        if os.name == "nt":
            raise MlflowException(
                "The docker job executor requires a POSIX host running Docker or Podman."
            )
        # Fail at startup rather than on every job if a container setting is malformed.
        for job_name in (*_JOB_PLANNERS, *_ONLINE_CONTAINER_JOB_NAMES):
            _resource_limits(job_name, _declared_resource_limits(job_name))
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
            "The docker job executor runs custom scorers without network access. On-demand scorer "
            "runs made only of custom scorers use it, and online scoring jobs run their custom "
            "scorers in job containers too; other jobs with custom scorers, such as evaluation "
            "jobs, keep running them on the default executor backend."
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
            # The process may exit right after this returns, so wait for starts in progress: their
            # containers would otherwise run on unattended until the next start recovers them.
            if not self._start_finished.wait_for(
                lambda: not any(job.starting for job in self._jobs.values()),
                timeout=_STOP_WAIT_FOR_STARTS_SECONDS,
            ):
                _logger.warning(
                    "Docker job containers were still starting after %s seconds; they will be "
                    "removed when the job runner next starts.",
                    _STOP_WAIT_FOR_STARTS_SECONDS,
                )
            jobs = list(self._jobs.items())
        for _, job in jobs:
            self._remove_container(job, kill=True)

    def _image(self) -> str:
        return job_image()

    def _prepare_image(self) -> None:
        prepare_job_image(self._get_client())

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
        input_dir, output_dir = make_job_container_dirs(Path(job.workdir.name))

        planner = _load_function(_JOB_PLANNERS[job_name])
        job.plan = planner(
            params=params, input_dir=input_dir, container_input_dir=_CONTAINER_INPUT_DIR
        )

        fn_meta = _load_function(fn_fullname)._job_fn_metadata
        write_transient_error_classes(input_dir, fn_meta.transient_error_classes)
        memory, nano_cpus = _resource_limits(job_name, fn_meta.resource_limits)
        environment = job_container_environment(job.plan)

        with self._lock:
            if self._stopped or self._jobs.get(job_id) is not job or job.cancel_requested:
                return
            job.starting = True
        # Not under the lock: these daemon calls can take a while, and the scheduler's cancel_job
        # and stop_executor also take it, so a slow or hung daemon would stall scheduling and
        # shutdown.
        try:
            container = run_job_container(
                self._get_client(),
                self._image(),
                job_id,
                job_name,
                input_dir,
                output_dir,
                environment,
                memory,
                nano_cpus,
            )
        except Exception:
            with self._lock:
                job.starting = False
                self._start_finished.notify_all()
            # docker-py creates the container before starting it and does not remove it if the
            # start fails, and no container ID was recorded, so find it by its label.
            self._remove_job_containers(job_id)
            raise
        with self._lock:
            job.container_id = container.id
            job.starting = False
            self._start_finished.notify_all()
            # A cancel or shutdown during the start found no container to kill, so kill it now.
            # wait_for_job then reports the job as canceled or leaves it for recovery. Defensive:
            # the job runner never finalizes a job before submit_job returns, but a job finalized
            # meanwhile would leave its container with no one to remove it.
            orphaned = self._jobs.get(job_id) is not job
            kill = orphaned or self._stopped or job.cancel_requested
        if kill:
            _kill_quietly(container)
            if orphaned:
                _remove_quietly(container)
            _logger.info(
                "Killed docker job container %s for job %s right after it started: the job was "
                "canceled or finalized, or the executor stopped",
                container.short_id,
                job_id,
            )
            return
        _logger.info(
            "Started docker job container %s for job %s (%s)", container.short_id, job_id, job_name
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

        return job_result_from_container(
            container,
            outcome.get("StatusCode"),
            Path(job.workdir.name, "output"),
            job_id,
            job.plan,
        )

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
            client = self._get_client()
        except Exception:
            _logger.debug("Could not list containers for job %s", job_id, exc_info=True)
            return
        remove_job_containers(client, job_id)

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
