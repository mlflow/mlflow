"""Built-in local subprocess job executor."""

import os
import select
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
    _kill_process,
    _load_function,
    _normalize_python_env,
    _prepare_job_subprocess,
    _PreparedJobSetupCommand,
    _reap_process,
    _SubprocessJobResult,
)
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.process import ShellCommandException, _exec_cmd


class _LocalJobSubmissionCanceled(Exception):
    pass


class _LocalJobSubmissionTimedOut(Exception):
    """Raised internally when the hard deadline elapses during environment setup."""


_STDERR_TAIL_MAX_CHARS = 4000


class _BoundedStderrReader:
    """Drain a subprocess's stderr into a bounded in-memory tail.

    A background thread reads the stream continuously so a job that emits a large
    volume of stderr can neither deadlock ``process.wait()`` (as an unread PIPE
    would once its buffer fills) nor force the server to hold all of it in memory.
    Only the trailing ``max_bytes`` bytes are retained.

    The drain loop polls the raw fd with ``select`` on a short interval and reads
    with ``os.read`` rather than blocking in ``stream.read()``. This keeps the
    thread interruptible: if a job spawns a descendant that inherits the PIPE
    write end and outlives the job (so the pipe never reaches EOF), ``close()``
    can stop the reader within one poll interval instead of stalling. Closing the
    stream to interrupt a blocked ``BufferedReader.read`` is deliberately avoided:
    the close would itself block on the buffered-IO lock the in-flight read holds,
    so it would hang until the descendant exited rather than unblocking anything.
    """

    _POLL_INTERVAL = 0.5

    def __init__(self, stream: Any, max_bytes: int = _STDERR_TAIL_MAX_CHARS * 4) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._buffer = bytearray()
        self._truncated = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._drain, name="local-job-stderr-reader", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        try:
            fd = self._stream.fileno()
        except (OSError, ValueError, AttributeError):
            return
        if not isinstance(fd, int):
            # A mocked stream (in tests) whose fileno() is not a real descriptor.
            return
        try:
            while True:
                try:
                    ready, _, _ = select.select([fd], [], [], self._POLL_INTERVAL)
                except (OSError, ValueError):
                    break
                if ready:
                    # Drain everything currently available before honoring a stop
                    # request, so stderr the job wrote right before exiting is not
                    # dropped when close()/tail() races the final read.
                    try:
                        chunk = os.read(fd, 4096)
                    except (OSError, ValueError):
                        break
                    if not chunk:
                        break  # EOF: every write end (job and descendants) closed.
                    with self._lock:
                        self._buffer.extend(chunk)
                        overflow = len(self._buffer) - self._max_bytes
                        if overflow > 0:
                            del self._buffer[:overflow]
                            self._truncated = True
                    continue
                # No data right now: stop if asked, otherwise keep polling.
                if self._stop.is_set():
                    break
        finally:
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        """Stop the reader and wait for it, so no daemon thread lingers."""
        self._stop.set()
        self._thread.join(timeout=5)

    def tail(self, max_chars: int = _STDERR_TAIL_MAX_CHARS) -> str:
        """Return the trailing captured stderr, stopping the reader first."""
        # The process has exited by the time this is called. In the common case
        # the pipe is already at EOF and the reader has finished; close() just
        # joins it. If a descendant still holds the write end, close() stops the
        # reader within one poll interval rather than waiting for that descendant.
        self.close()
        with self._lock:
            data = bytes(self._buffer)
            truncated = self._truncated
        content = data.decode("utf-8", errors="replace").strip()
        if len(content) > max_chars:
            content = content[-max_chars:]
            truncated = True
        if truncated:
            return "...(truncated)...\n" + content
        return content


@dataclass
class _LocalJobProcess:
    active_process: subprocess.Popen | None
    temporary_directory: tempfile.TemporaryDirectory
    result_path: str
    function_fullname: str
    timeout: float
    # Hard deadline (monotonic clock) armed at submit-entry so the timeout covers
    # environment setup in addition to job execution. Always set at construction.
    deadline: float
    stderr_reader: _BoundedStderrReader | None = None
    cancel_requested: bool = False
    timed_out: bool = False
    # Set True only once the job subprocess (not a setup subprocess) has been
    # launched. Used by cancel_job to decide whether a failed cancellation should
    # roll back cancel_requested.
    job_process_launched: bool = False


class LocalJobExecutor(AbstractJobExecutor):
    """Default executor that runs jobs as local subprocesses.

    Args:
        config: Framework-level executor configuration.
    """

    def __init__(self, config: JobExecutorConfig) -> None:
        super().__init__(config)
        self._processes: dict[str, _LocalJobProcess] = {}
        # Protects:
        # - self._processes
        # - self._stopped
        # - mutable fields on _LocalJobProcess stored in self._processes
        # Never hold while blocking on process.wait().
        self._state_lock = threading.RLock()
        self._stopped = False

    def _effective_timeout(self, timeout: float | None) -> float:
        return timeout if timeout is not None else self.config.default_timeout

    def _finalize_record(self, job_id: str, record: _LocalJobProcess) -> None:
        with self._state_lock:
            if self._processes.get(job_id) is record:
                self._processes.pop(job_id, None)
        if record.stderr_reader is not None:
            record.stderr_reader.close()
        record.temporary_directory.cleanup()

    def _run_setup_command(
        self,
        job_id: str,
        record: _LocalJobProcess,
        setup_command: _PreparedJobSetupCommand,
    ) -> None:
        with self._state_lock:
            if self._processes.get(job_id) is not record or record.cancel_requested:
                raise _LocalJobSubmissionCanceled
            if time.monotonic() >= record.deadline:
                record.timed_out = True
                raise _LocalJobSubmissionTimedOut

            process = _exec_cmd(
                setup_command.command,
                cwd=setup_command.cwd,
                extra_env=setup_command.extra_env,
                capture_output=False,
                synchronous=False,
                start_new_session=True,
            )
            record.active_process = process

        # The hard deadline was armed at submission start, so it bounds the total
        # setup + execution time. Enforce the remaining budget on this setup phase.
        remaining_timeout = max(record.deadline - time.monotonic(), 0.0)
        try:
            returncode = process.wait(timeout=remaining_timeout)
        except subprocess.TimeoutExpired:
            with self._state_lock:
                record.timed_out = True
                _kill_process(process)
                if record.active_process is process:
                    record.active_process = None
            _reap_process(process)
            raise _LocalJobSubmissionTimedOut

        with self._state_lock:
            if record.active_process is process:
                record.active_process = None
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
            active_process=None,
            temporary_directory=tempfile.TemporaryDirectory(),
            result_path="",
            function_fullname=fn_fullname,
            timeout=effective_timeout,
            # Arm the hard deadline before any setup work so the timeout covers
            # environment setup in addition to job execution.
            deadline=time.monotonic() + effective_timeout,
        )
        with self._state_lock:
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

            normalized_python_env = _normalize_python_env(python_env)
            extra_envs = {
                MLFLOW_TRACKING_URI.name: context.tracking_uri,
                MLFLOW_GATEWAY_URI.name: context.gateway_uri or context.tracking_uri,
            }

            with self._state_lock:
                if self._processes.get(job_id) is not record or record.cancel_requested:
                    raise _LocalJobSubmissionCanceled

            # _prepare_job_subprocess only reads this record's own temporary
            # directory and never mutates shared executor state, so it runs
            # outside _state_lock. Holding the lock across its file I/O (and the
            # module import already performed above) would serialize other jobs'
            # submit/cancel behind this job's setup. A cancel that races in is
            # still honored: _run_setup_command re-checks cancellation under the
            # lock before launching anything.
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

            with self._state_lock:
                if self._processes.get(job_id) is not record or record.cancel_requested:
                    raise _LocalJobSubmissionCanceled
                if time.monotonic() >= record.deadline:
                    record.timed_out = True
                    raise _LocalJobSubmissionTimedOut

                # Capture the job subprocess stderr so a diagnostic tail can be
                # surfaced if the subprocess crashes. A background reader drains
                # the pipe into a bounded buffer, which both avoids deadlocking
                # process.wait() (an unread PIPE stalls once its buffer fills) and
                # caps how much stderr the server holds in memory.
                process = subprocess.Popen(
                    prepared_subprocess.command,
                    env=prepared_subprocess.env,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                record.stderr_reader = _BoundedStderrReader(process.stderr)
                record.active_process = process
                record.job_process_launched = True
        except _LocalJobSubmissionCanceled:
            return
        except _LocalJobSubmissionTimedOut:
            # The record is intentionally left in place (with timed_out=True) so
            # wait_for_job can report a TIMEOUT result to the caller.
            return
        except Exception:
            self._finalize_record(job_id, record)
            raise

    def wait_for_job(self, job_id: str) -> JobResult:
        with self._state_lock:
            record = self._processes.get(job_id)

        if record is None:
            raise MlflowException.invalid_parameter_value(
                f"Unknown job ID for LocalJobExecutor: {job_id!r}"
            )

        process = record.active_process
        if process is None:
            if record.cancel_requested:
                self._finalize_record(job_id, record)
                return JobResult(status=JobStatus.CANCELED)

            if record.timed_out:
                self._finalize_record(job_id, record)
                return JobResult(
                    status=JobStatus.TIMEOUT,
                    error_message=(
                        f"Local job {job_id!r} ({record.function_fullname}) timed out after "
                        f"{record.timeout} seconds during environment setup."
                    ),
                )

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
            with self._state_lock:
                cancel_requested = record.cancel_requested
                _kill_process(process)
            status = JobStatus.CANCELED if cancel_requested else JobStatus.TIMEOUT
            _reap_process(process)
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

        with self._state_lock:
            cancel_requested = record.cancel_requested

        if cancel_requested:
            self._finalize_record(job_id, record)
            return JobResult(status=JobStatus.CANCELED)

        if process.returncode != 0:
            # Read the captured stderr before _finalize_record joins the reader.
            stderr_tail = record.stderr_reader.tail() if record.stderr_reader else ""
            self._finalize_record(job_id, record)
            error_message = (
                "The subprocess that executes job function "
                f"{record.function_fullname} exited with error code {process.returncode}"
            )
            if stderr_tail:
                error_message += f"\nSubprocess stderr (tail):\n{stderr_tail}"
            return JobResult(status=JobStatus.FAILED, error_message=error_message)

        try:
            subprocess_result = _SubprocessJobResult.load(record.result_path)
        except Exception as exc:
            stderr_tail = record.stderr_reader.tail() if record.stderr_reader else ""
            self._finalize_record(job_id, record)
            error_message = (
                "Failed to load the local subprocess result for "
                f"{record.function_fullname}: {exc!r}"
            )
            if stderr_tail:
                error_message += f"\nSubprocess stderr (tail):\n{stderr_tail}"
            return JobResult(status=JobStatus.FAILED, error_message=error_message)

        self._finalize_record(job_id, record)
        if subprocess_result.succeeded:
            return JobResult(status=JobStatus.SUCCEEDED, result=subprocess_result.result)

        return JobResult(
            status=JobStatus.FAILED,
            error_message=subprocess_result.error,
            is_transient_error=bool(subprocess_result.is_transient_error),
        )

    def cancel_job(self, job_id: str) -> None:
        with self._state_lock:
            record = self._processes.get(job_id)
            if record is None or record.cancel_requested:
                return

            record.cancel_requested = True
            process = record.active_process
            cancellation_delivered = _kill_process(process)
            # Only roll back the cancellation when a live job subprocess failed to
            # receive the signal. During environment setup active_process may be a
            # transient setup subprocess (or briefly None between setup steps), so
            # a rolled-back flag there would let submit_job go on to launch the job
            # after cancel_job returned, leaking a process the caller asked to stop.
            # Latch cancellation until the job subprocess itself is running.
            if record.job_process_launched and process is not None and not cancellation_delivered:
                record.cancel_requested = False

    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        """Best-effort cleanup for unfinished local jobs.

        LocalJobExecutor cannot reattach to prior subprocesses. Recovery therefore
        always returns ``action="requeue"`` after killing/reaping any still-tracked
        local process and removing its in-memory record.
        """
        recovery_results = []
        for job_id in unfinished_job_ids:
            with self._state_lock:
                record = self._processes.get(job_id)
                if record is not None:
                    record.cancel_requested = True
                    _kill_process(record.active_process)

            if record is not None:
                _reap_process(record.active_process)
                self._finalize_record(job_id, record)

            recovery_results.append(JobRecoveryResult(job_id=job_id, action="requeue"))
        return recovery_results

    def stop_executor(self) -> None:
        with self._state_lock:
            self._stopped = True
            records = list(self._processes.items())
            for _, record in records:
                record.cancel_requested = True
                _kill_process(record.active_process)

        for job_id, record in records:
            _reap_process(record.active_process)
            self._finalize_record(job_id, record)

    def check_requirements(self) -> None:
        if os.name == "nt":
            raise MlflowException("MLflow job backend does not support Windows system.")
