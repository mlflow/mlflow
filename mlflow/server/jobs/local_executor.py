"""Built-in local subprocess job executor."""

import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_GATEWAY_URI, MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import (
    AbstractJobExecutor,
    JobExecutionContext,
    JobExecutorConfig,
    JobRecoveryResult,
    JobResult,
)
from mlflow.server.jobs.utils import (
    JobResult as LegacyJobResult,
)
from mlflow.server.jobs.utils import (
    _load_function,
    _prepare_job_subprocess,
    _PreparedJobSetupCommand,
)
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.process import ShellCommandException, _exec_cmd


class _LocalJobSubmissionCanceled(Exception):
    pass


@dataclass
class _LocalJobProcess:
    process: subprocess.Popen | None
    temporary_directory: tempfile.TemporaryDirectory
    result_path: str
    function_fullname: str
    timeout: float
    deadline: float | None
    cancel_requested: bool = False


class LocalJobExecutor(AbstractJobExecutor):
    """Default executor that runs jobs as local subprocesses.

    Args:
        config: Framework-level executor configuration.
    """

    def __init__(self, config: JobExecutorConfig) -> None:
        super().__init__(config)
        self._processes: dict[str, _LocalJobProcess] = {}
        self._lock = threading.RLock()
        self._stopped = False

    def _effective_timeout(self, timeout: float | None) -> float:
        return timeout if timeout is not None else self.config.default_timeout

    def _normalize_python_env(self, python_env: _PythonEnv | None) -> _PythonEnv | None:
        if python_env is None:
            return None

        return _PythonEnv(
            python=PYTHON_VERSION,
            build_dependencies=list(python_env.build_dependencies),
            dependencies=list(python_env.dependencies),
        )

    def _finalize_record(self, job_id: str, record: _LocalJobProcess) -> None:
        with self._lock:
            if self._processes.get(job_id) is record:
                self._processes.pop(job_id, None)
        record.temporary_directory.cleanup()

    def _kill_process(self, process: subprocess.Popen | None) -> bool:
        if process is None or process.returncode is not None:
            return False
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
                return True
        except ProcessLookupError:
            return False
        return False

    def _reap_process(self, process: subprocess.Popen | None) -> None:
        if process is None:
            return
        try:
            process.wait()
        except ChildProcessError:
            pass

    def _run_setup_command(
        self,
        job_id: str,
        record: _LocalJobProcess,
        setup_command: _PreparedJobSetupCommand,
    ) -> None:
        with self._lock:
            if self._processes.get(job_id) is not record or record.cancel_requested:
                raise _LocalJobSubmissionCanceled

            process = _exec_cmd(
                setup_command.command,
                cwd=setup_command.cwd,
                extra_env=setup_command.extra_env,
                capture_output=False,
                synchronous=False,
                start_new_session=True,
            )
            record.process = process

        returncode = process.wait()
        with self._lock:
            if record.process is process:
                record.process = None
            is_active = self._processes.get(job_id) is record
            cancel_requested = record.cancel_requested

        if not is_active or cancel_requested:
            raise _LocalJobSubmissionCanceled
        if returncode != 0:
            raise ShellCommandException.from_completed_process(
                subprocess.CompletedProcess(process.args, returncode)
            )

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

        effective_timeout = self._effective_timeout(timeout)
        record = _LocalJobProcess(
            process=None,
            temporary_directory=tempfile.TemporaryDirectory(),
            result_path="",
            function_fullname=fn_fullname,
            timeout=effective_timeout,
            deadline=None,
        )
        with self._lock:
            if self._stopped:
                record.temporary_directory.cleanup()
                raise MlflowException("LocalJobExecutor is stopped and cannot accept new jobs.")
            if job_id in self._processes:
                record.temporary_directory.cleanup()
                raise MlflowException.invalid_parameter_value(
                    f"Job {job_id!r} is already being managed by LocalJobExecutor."
                )
            self._processes[job_id] = record

        try:
            function = _load_function(fn_fullname)
            if not hasattr(function, "_job_fn_metadata"):
                raise MlflowException.invalid_parameter_value(
                    f"The job function {fn_fullname} is not decorated by 'mlflow.server.jobs.job'."
                )
            transient_error_classes = function._job_fn_metadata.transient_error_classes

            normalized_python_env = self._normalize_python_env(python_env)
            extra_envs = {
                MLFLOW_TRACKING_URI.name: context.tracking_uri,
                MLFLOW_GATEWAY_URI.name: context.gateway_uri or context.tracking_uri,
            }

            with self._lock:
                if self._processes.get(job_id) is not record or record.cancel_requested:
                    raise _LocalJobSubmissionCanceled

                prepared_subprocess = _prepare_job_subprocess(
                    function_fullname=fn_fullname,
                    params=params,
                    python_env=normalized_python_env,
                    transient_error_classes=transient_error_classes,
                    tmpdir=record.temporary_directory.name,
                    job_id=job_id,
                    job_name=job_name,
                    workspace=context.workspace,
                    extra_envs=extra_envs,
                )
                record.result_path = prepared_subprocess.result_path

            for setup_command in prepared_subprocess.setup_commands:
                self._run_setup_command(job_id, record, setup_command)

            with self._lock:
                if self._processes.get(job_id) is not record or record.cancel_requested:
                    raise _LocalJobSubmissionCanceled

                record.deadline = time.monotonic() + effective_timeout
                process = subprocess.Popen(
                    prepared_subprocess.command,
                    env=prepared_subprocess.env,
                    start_new_session=True,
                )
                record.process = process
        except _LocalJobSubmissionCanceled:
            return
        except Exception:
            self._finalize_record(job_id, record)
            raise

    def wait_for_job(self, job_id: str) -> JobResult:
        with self._lock:
            record = self._processes.get(job_id)

        if record is None:
            raise MlflowException.invalid_parameter_value(
                f"Unknown job ID for LocalJobExecutor: {job_id!r}"
            )

        process = record.process
        if process is None:
            if record.cancel_requested:
                self._finalize_record(job_id, record)
                return JobResult(status=JobStatus.CANCELED)

            self._finalize_record(job_id, record)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=(
                    f"The local subprocess for job {job_id!r} "
                    f"({record.function_fullname}) was never started."
                ),
            )

        remaining_timeout = max(record.deadline - time.monotonic(), 0.0)
        timed_out = False
        try:
            process.wait(timeout=remaining_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True

        if timed_out:
            with self._lock:
                cancel_requested = record.cancel_requested
                self._kill_process(process)
            status = JobStatus.CANCELED if cancel_requested else JobStatus.TIMEOUT
            self._reap_process(process)
            self._finalize_record(job_id, record)
            return JobResult(
                status=status,
                error_message=(
                    None
                    if cancel_requested
                    else (
                        f"Local job {job_id!r} ({record.function_fullname}) "
                        f"timed out after {record.timeout} seconds."
                    )
                ),
            )

        with self._lock:
            cancel_requested = record.cancel_requested

        if cancel_requested:
            self._finalize_record(job_id, record)
            return JobResult(status=JobStatus.CANCELED)

        if process.returncode != 0:
            self._finalize_record(job_id, record)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=(
                    "The subprocess that executes job function "
                    f"{record.function_fullname} exited with error code {process.returncode}"
                ),
            )

        try:
            legacy_result = LegacyJobResult.load(record.result_path)
        except Exception as exc:
            self._finalize_record(job_id, record)
            return JobResult(
                status=JobStatus.FAILED,
                error_message=(
                    "Failed to load the local subprocess result for "
                    f"{record.function_fullname}: {exc!r}"
                ),
            )

        self._finalize_record(job_id, record)
        if legacy_result.succeeded:
            return JobResult(status=JobStatus.SUCCEEDED, result=legacy_result.result)

        return JobResult(
            status=JobStatus.FAILED,
            error_message=legacy_result.error,
            is_transient_error=bool(legacy_result.is_transient_error),
        )

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            record = self._processes.get(job_id)
            if record is None or record.cancel_requested:
                return

            record.cancel_requested = True
            process = record.process
            cancellation_delivered = self._kill_process(process)
            if record.deadline is not None and process is not None and not cancellation_delivered:
                record.cancel_requested = False

    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        recovery_results = []
        for job_id in unfinished_job_ids:
            with self._lock:
                record = self._processes.get(job_id)
                if record is not None:
                    record.cancel_requested = True
                    self._kill_process(record.process)

            if record is not None:
                self._reap_process(record.process)
                self._finalize_record(job_id, record)

            recovery_results.append(JobRecoveryResult(job_id=job_id, action="requeue"))
        return recovery_results

    def stop_executor(self) -> None:
        with self._lock:
            self._stopped = True
            records = list(self._processes.items())
            for _, record in records:
                record.cancel_requested = True
                self._kill_process(record.process)

        for job_id, record in records:
            self._reap_process(record.process)
            self._finalize_record(job_id, record)

    def check_requirements(self) -> None:
        if os.name == "nt":
            raise MlflowException("MLflow job backend does not support Windows system.")
