import errno
import hashlib
import importlib
import inspect
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_LOGGING_LEVEL,
    MLFLOW_SERVER_JOB_EXECUTION_ENGINE,
    MLFLOW_SERVER_JOB_FLUSH_PERIODIC_LOCKS_ON_STARTUP,
    MLFLOW_SERVER_JOB_HUEY_REDIS_URL,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
    MLFLOW_WORKSPACE,
)
from mlflow.exceptions import MlflowException
from mlflow.server.constants import HUEY_STORAGE_PATH_ENV_VAR, MLFLOW_SERVER_UP_TIME
from mlflow.tracing.trace_archival_service import run_trace_archival_scheduler
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.import_hooks import register_post_import_hook
from mlflow.utils.process import _exec_cmd
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

if TYPE_CHECKING:
    from huey import Huey

    from mlflow.entities._job import Job
    from mlflow.store.jobs.abstract_store import AbstractJobStore

_logger = logging.getLogger(__name__)

# Reserved Huey instance key for periodic tasks
HUEY_PERIODIC_TASKS_INSTANCE_KEY = "periodic_tasks"

# Environment variable names for job execution
MLFLOW_SERVER_JOB_NAME_ENV_VAR = "_MLFLOW_SERVER_JOB_NAME"
MLFLOW_SERVER_JOB_ID_ENV_VAR = "_MLFLOW_SERVER_JOB_ID"
MLFLOW_SERVER_JOB_PARAMS_ENV_VAR = "_MLFLOW_SERVER_JOB_PARAMS"
MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR = "_MLFLOW_SERVER_JOB_FUNCTION_FULLNAME"
MLFLOW_SERVER_JOB_RESULT_DUMP_PATH_ENV_VAR = "_MLFLOW_SERVER_JOB_RESULT_DUMP_PATH"
MLFLOW_SERVER_JOB_TRANSIENT_ERROR_CLASSES_PATH_ENV_VAR = (
    "_MLFLOW_SERVER_JOB_TRANSIENT_ERROR_CLASSES_PATH"
)
MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR = "_MLFLOW_ORIGINAL_PARENT_PID"

# Number of worker threads for the periodic tasks consumer
PERIODIC_TASKS_WORKER_COUNT = 5


def _exponential_backoff_retry(retry_count: int) -> None:
    from huey.exceptions import RetryTask

    # We can support more retry strategies (e.g. exponential backoff) in future
    base_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get()
    max_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get()
    delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
    raise RetryTask(delay=delay)


@dataclass
class _SubprocessJobResult:
    succeeded: bool
    result: str | None = None  # serialized JSON string
    is_transient_error: bool | None = None
    error: str | None = None

    @classmethod
    def from_error(
        cls, e: Exception, transient_error_classes: list[type[Exception]] | None = None
    ) -> "_SubprocessJobResult":
        from mlflow.server.jobs import TransientError

        if isinstance(e, TransientError):
            return _SubprocessJobResult(
                succeeded=False, is_transient_error=True, error=repr(e.origin_error)
            )

        if transient_error_classes:
            if e.__class__ in transient_error_classes:
                return _SubprocessJobResult(succeeded=False, is_transient_error=True, error=repr(e))

        return _SubprocessJobResult(
            succeeded=False,
            is_transient_error=False,
            error=repr(e),
        )

    def dump(self, path: str) -> None:
        with open(path, "w") as fp:
            json.dump(asdict(self), fp)

    @classmethod
    def load(cls, path: str) -> "_SubprocessJobResult":
        with open(path) as fp:
            return _SubprocessJobResult(**json.load(fp))


def _kill_own_process_group() -> None:
    """SIGKILL the caller's process group when the caller leads that group.

    Local-executor job subprocesses are started with ``start_new_session=True``,
    so each leads its own process group; killing that group tears down any
    descendants the job spawned. ``killpg`` delivers ``SIGKILL`` to the caller
    too, so callers that also need to guarantee termination of a non-leader
    process should still fall back to ``os._exit`` afterwards.
    """
    try:
        pgid = os.getpgid(0)
    except OSError:
        return
    if pgid == os.getpid():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass


def _exit_when_orphaned(poll_interval: float = 1) -> None:
    raw_parent_pid = os.environ.get(MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR)
    try:
        parent_pid = int(raw_parent_pid) if raw_parent_pid is not None else 0
    except (TypeError, ValueError):
        parent_pid = 0
    if parent_pid <= 0:
        parent_pid = os.getppid()
    while True:
        current_parent_pid = os.getppid()
        # Detect orphaning by comparing against the original parent pid rather than
        # testing ``current_parent_pid == 1``: when the original parent legitimately
        # runs as PID 1 (e.g. the MLflow server as a container's init process) a
        # healthy child also observes ``getppid() == 1``, and treating that as
        # orphaned would kill a running job's whole process group.
        if current_parent_pid != parent_pid:
            # Kill the whole process group so descendants spawned by an orphaned
            # job do not outlive it. For non-leaders this is a no-op and os._exit
            # still terminates this process.
            _kill_own_process_group()
            os._exit(1)
        time.sleep(poll_interval)


def _normalize_python_env(python_env: _PythonEnv | None) -> _PythonEnv | None:
    """Pin the interpreter to the server's Python version, preserving deps.

    Returns a copy so callers never mutate the caller-supplied ``_PythonEnv``.
    """
    if python_env is None:
        return None

    return _PythonEnv(
        python=PYTHON_VERSION,
        build_dependencies=list(python_env.build_dependencies),
        dependencies=list(python_env.dependencies),
    )


def _kill_process(process: subprocess.Popen | None) -> bool:
    """SIGKILL the process group led by ``process``.

    Returns True only when a signal was delivered to a still-running group leader.
    """
    if process is None or process.returncode is not None:
        return False
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGKILL)
            return True
    except ProcessLookupError:
        return False
    return False


def _reap_process(process: subprocess.Popen | None) -> None:
    """Wait on ``process`` to release its zombie entry, ignoring double-reap."""
    if process is None:
        return
    try:
        process.wait()
    except ChildProcessError:
        pass


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # doesn't actually kill
    except OSError as e:
        if e.errno == errno.ESRCH:  # No such process
            return False
        elif e.errno == errno.EPERM:  # Process exists, but no permission
            return True
        else:
            raise
    else:
        return True


def _start_huey_consumer_proc(
    huey_instance_key: str,
    max_job_parallelism: int,
):
    from mlflow.server.constants import MLFLOW_HUEY_INSTANCE_KEY
    from mlflow.utils.process import _exec_cmd

    cmd = [
        sys.executable,
        "-m",
        "huey.bin.huey_consumer",
        "mlflow.server.jobs._huey_consumer.huey_instance",
        "-w",
        str(max_job_parallelism),
    ]

    # Add quiet flag unless DEBUG logging is explicitly requested,
    # to suppress noisy huey consumer logs (e.g., Scheduler, Executing messages)
    log_level = (MLFLOW_LOGGING_LEVEL.get() or "INFO").upper()
    if log_level != "DEBUG":
        cmd.append("-q")

    return _exec_cmd(
        cmd,
        capture_output=False,
        synchronous=False,
        extra_env={
            MLFLOW_HUEY_INSTANCE_KEY: huey_instance_key,
            MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR: str(os.getpid()),
        },
    )


_JOB_ENTRY_MODULE = "mlflow.server.jobs._job_subproc_entry"


_JOB_ENV_SETUP_MODULE = "mlflow.server.jobs._job_env_setup"


_JOB_STATUS_POLL_INTERVAL = 1


@dataclass(frozen=True)
class _PreparedJobSetupCommand:
    command: list[str]
    cwd: str | None = None
    extra_env: dict[str, str] | None = None


@dataclass(frozen=True)
class _PreparedJobSubprocess:
    command: list[str]
    env: dict[str, str]
    result_path: str
    setup_commands: tuple[_PreparedJobSetupCommand, ...] = ()


def _prepare_job_subprocess(
    function_fullname: str,
    params: dict[str, Any],
    python_env: _PythonEnv | None,
    transient_error_classes: list[type[Exception]] | None,
    tmpdir: str,
    job_id: str,
    job_name: str,
    workspace: str | None,
    extra_envs: dict[str, str] | None = None,
) -> _PreparedJobSubprocess:
    from mlflow.server.jobs._job_env_setup import _get_incomplete_marker_path
    from mlflow.utils.process import _join_commands
    from mlflow.utils.virtualenv import (
        _get_mlflow_virtualenv_root,
        _get_virtualenv_activate_cmd,
        _get_virtualenv_name,
    )

    setup_commands = []
    if python_env is not None:
        if shutil.which("uv") is None:
            raise MlflowException(
                "The job requires 'uv' to create an isolated Python environment, "
                "but 'uv' is not installed."
            )

        virtual_envs_root_path = Path(_get_mlflow_virtualenv_root())
        env_name = _get_virtualenv_name(python_env, None)
        env_dir = virtual_envs_root_path / env_name
        activate_cmd = _get_virtualenv_activate_cmd(env_dir)

        if not env_dir.exists() or _get_incomplete_marker_path(env_dir).exists():
            _logger.info(f"Creating or repairing a python virtual environment in {env_dir}.")
            requirements_file = Path(tmpdir) / "requirements.txt"
            requirements_file.write_text("\n".join(python_env.dependencies))
            setup_command = [
                sys.executable,
                "-m",
                _JOB_ENV_SETUP_MODULE,
                "--env-dir",
                str(env_dir),
                "--python-version",
                python_env.python,
                "--requirements-file",
                str(requirements_file),
            ]
            # Install build dependencies (e.g. pinned pip/setuptools/wheel)
            # before the regular dependencies, mirroring the canonical MLflow
            # environment-restore ordering.
            if python_env.build_dependencies:
                build_requirements_file = Path(tmpdir) / "build-requirements.txt"
                build_requirements_file.write_text("\n".join(python_env.build_dependencies))
                setup_command += ["--build-requirements-file", str(build_requirements_file)]
            # Record the original parent pid so the setup subprocess can exit (and
            # release its environment lock) if the server dies and it is reparented.
            setup_commands.append(
                _PreparedJobSetupCommand(
                    command=setup_command,
                    extra_env={MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR: str(os.getpid())},
                )
            )
        else:
            _logger.debug(f"The python environment {env_dir} already exists.")

        job_cmd = _join_commands(activate_cmd, f"exec python -m {_JOB_ENTRY_MODULE}")
    else:
        job_cmd = [sys.executable, "-m", _JOB_ENTRY_MODULE]

    result_file = str(Path(tmpdir) / "result.json")
    transient_error_classes_file = str(Path(tmpdir) / "transient_error_classes")
    transient_error_classes = transient_error_classes or []
    with open(transient_error_classes_file, "w") as f:
        for cls in transient_error_classes:
            f.write(f"{cls.__module__}.{cls.__name__}\n")

    job_env = {
        **os.environ,
        MLFLOW_SERVER_JOB_NAME_ENV_VAR: job_name,
        MLFLOW_SERVER_JOB_ID_ENV_VAR: job_id,
        MLFLOW_SERVER_JOB_PARAMS_ENV_VAR: json.dumps(params),
        MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR: function_fullname,
        MLFLOW_SERVER_JOB_RESULT_DUMP_PATH_ENV_VAR: result_file,
        MLFLOW_SERVER_JOB_TRANSIENT_ERROR_CLASSES_PATH_ENV_VAR: transient_error_classes_file,
        MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR: str(os.getpid()),
        **(extra_envs or {}),
    }

    if workspace is None:
        job_env.pop(MLFLOW_WORKSPACE.name, None)
    else:
        job_env[MLFLOW_WORKSPACE.name] = workspace

    return _PreparedJobSubprocess(
        command=job_cmd,
        env=job_env,
        result_path=result_file,
        setup_commands=tuple(setup_commands),
    )


def _exec_job_in_subproc(
    function_fullname: str,
    params: dict[str, Any],
    python_env: _PythonEnv | None,
    transient_error_classes: list[type[Exception]] | None,
    timeout: float | None,
    tmpdir: str,
    job_store: "AbstractJobStore",
    job_id: str,
    job_name: str,
    workspace: str | None,
    extra_envs: dict[str, str] | None = None,
) -> _SubprocessJobResult | None:
    """
    Executes the job function in a subprocess,
    If the job execution time exceeds timeout, the subprocess is killed and return None,
    otherwise return `_SubprocessJobResult` instance,
    """
    prepared_subprocess = _prepare_job_subprocess(
        function_fullname=function_fullname,
        params=params,
        python_env=python_env,
        transient_error_classes=transient_error_classes,
        tmpdir=tmpdir,
        job_id=job_id,
        job_name=job_name,
        workspace=workspace,
        extra_envs=extra_envs,
    )

    for setup_command in prepared_subprocess.setup_commands:
        _exec_cmd(
            setup_command.command,
            cwd=setup_command.cwd,
            extra_env=setup_command.extra_env,
            capture_output=False,
        )

    with subprocess.Popen(
        prepared_subprocess.command,
        env=prepared_subprocess.env,
    ) as popen:
        beg_time = time.time()
        while popen.poll() is None:
            time.sleep(_JOB_STATUS_POLL_INTERVAL)

            job_status = job_store.get_job(job_id).status
            if job_status == JobStatus.CANCELED:
                popen.kill()
                return None

            if timeout is not None:
                if beg_time + timeout <= time.time():
                    # timeout
                    popen.kill()
                    job_store.mark_job_timed_out(job_id)
                    return None

        if popen.returncode == 0:
            return _SubprocessJobResult.load(prepared_subprocess.result_path)

        return _SubprocessJobResult.from_error(
            RuntimeError(
                f"The subprocess that executes job function {function_fullname} "
                f"exited with error code {popen.returncode}"
            )
        )


def _compute_exclusive_lock_key(job_name: str, params: dict[str, Any]) -> str:
    """
    Compute a lock key based on job name and params hash.

    Args:
        job_name: Name of the job.
        params: Parameter dictionary to use for the lock key.

    Returns:
        Lock key string.
    """
    params_json = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha256(params_json.encode()).hexdigest()[:16]
    return f"{job_name}:{params_hash}"


def _compute_job_lock_key(
    job_name: str, params: dict[str, Any], exclusive: bool | list[str], workspace: str | None
) -> str:
    """Compute the exclusive lock key for a job.

    Shared by both execution engines (the Huey ``_exec_job`` and the executor scheduler) so their
    key derivation cannot drift: ``exclusive=True`` locks on all params, a list of names locks on
    only those params, and the key is namespaced by workspace when workspaces are enabled. Callers
    check ``exclusive`` is truthy before calling.
    """
    lock_params = (
        {k: v for k, v in params.items() if k in exclusive}
        if isinstance(exclusive, list)
        else params
    )
    lock_key_job_name = job_name
    if MLFLOW_ENABLE_WORKSPACES.get():
        lock_key_job_name = f"{workspace or DEFAULT_WORKSPACE_NAME}:{job_name}"
    return _compute_exclusive_lock_key(lock_key_job_name, lock_params)


def _exec_job(
    job_id: str,
    workspace: str | None,
    job_name: str,
    params: dict[str, Any],
    timeout: float | None,
    exclusive: bool | list[str] = False,
    extra_envs: dict[str, str] | None = None,
) -> None:
    """
    Execute a job in a subprocess.

    Args:
        job_id: Unique identifier for the job.
        workspace: Workspace associated with the job.
        job_name: Name of the job function to execute.
        params: Parameters to pass to the job function.
        timeout: Maximum execution time in seconds, or None for no timeout.
        exclusive: If True, only one instance of this job with the same params can run
            at a time. If a list of parameter names, only those parameters are considered
            for exclusivity.
        extra_envs: Optional dictionary of additional environment variables to set
            before executing the job.
    """
    from mlflow.server.handlers import _get_job_store

    workspace_ctx = WorkspaceContext(workspace) if workspace else nullcontext()
    with workspace_ctx:
        job_store = _get_job_store()

        # If exclusive, acquire lock based on job_name + hash(params)
        # If lock is already held, TaskLockedException is raised and job is skipped
        lock = None
        if exclusive:
            from huey.exceptions import TaskLockedException

            huey_instance = _get_or_init_huey_instance(job_name).instance
            lock_key = _compute_job_lock_key(job_name, params, exclusive, workspace)
            lock = huey_instance.lock_task(lock_key)
            try:
                lock.acquire()
            except TaskLockedException:
                _logger.info(f"Skipping job {job_id} - exclusive lock {lock_key} already held")
                job_store.cancel_job(job_id)
                return
        else:
            lock = None

        job_started = False
        try:
            job_store.start_job(job_id)
            job_started = True

            fn_fullname = get_job_fn_fullname(job_name)
            function = _load_function(fn_fullname)
            fn_metadata = function._job_fn_metadata

            with tempfile.TemporaryDirectory() as tmpdir:
                job_result = _exec_job_in_subproc(
                    fn_metadata.fn_fullname,
                    params,
                    fn_metadata.python_env,
                    fn_metadata.transient_error_classes,
                    timeout,
                    tmpdir,
                    job_store,
                    job_id,
                    job_name,
                    workspace,
                    extra_envs,
                )

            if job_result is None:
                return

            if job_result.succeeded:
                job_store.finish_job(job_id, job_result.result)
                return

            if job_result.is_transient_error:
                # For transient errors, if the retry count is less than max allowed count,
                # trigger task retry by raising `RetryTask` exception.
                retry_count = job_store.retry_or_fail_job(job_id, job_result.error)
                if retry_count is not None:
                    _exponential_backoff_retry(retry_count)
            else:
                _logger.error(f"Job {job_id} ({job_name}) failed with error: {job_result.error}")
                job_store.fail_job(job_id, job_result.error)
        except Exception as exc:
            # If start_job succeeded but a subsequent step raises an unexpected error,
            # fail the job so it doesn't remain stuck in RUNNING state.
            # Note: RetryTask is raised intentionally by _exponential_backoff_retry to
            # schedule a Huey retry, not a real error - skip fail_job in that case.
            from huey.exceptions import RetryTask

            if job_started and not isinstance(exc, RetryTask):
                _logger.error(
                    f"Job {job_id} ({job_name}) encountered an unexpected error: {exc!r}",
                    exc_info=True,
                )
                try:
                    job_store.fail_job(job_id, repr(exc))
                except Exception as fail_exc:
                    _logger.error(
                        f"Job {job_id} ({job_name}) failed to transition to FAILED state via "
                        f"fail_job: {fail_exc!r}",
                        exc_info=True,
                    )
            raise
        finally:
            if lock is not None:
                lock.release()


@dataclass
class HueyInstance:
    instance: "Huey"
    submit_task: Callable[..., Any]


# Each job function has an individual execution pool, each execution pool
# is managed by a Huey instance.
# The `_huey_instance_map` stores the map, the key is the job function fullname,
# and the value is the `HueyInstance` object.
_huey_instance_map: dict[str, HueyInstance] = {}
_huey_instance_map_lock = threading.RLock()


def _get_huey_redis_url() -> str | None:
    return MLFLOW_SERVER_JOB_HUEY_REDIS_URL.get()


def _should_flush_periodic_locks() -> bool:
    configured = MLFLOW_SERVER_JOB_FLUSH_PERIODIC_LOCKS_ON_STARTUP.get()
    if configured is not None:
        return configured
    return _get_huey_redis_url() is None


def _get_or_init_huey_instance(instance_key: str):
    from huey import RedisHuey, SqliteHuey
    from huey.serializer import Serializer

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return {
                    "__type__": "datetime",
                    "value": obj.isoformat(),
                }
            return super().default(obj)

    def json_loader_object_hook(d):
        if d.get("__type__") == "datetime":
            return datetime.fromisoformat(d["value"])
        return d

    class JsonSerializer(Serializer):
        def serialize(self, data):
            # Huey passes two types of data through the serializer:
            # 1. Message objects (task data) - have ._asdict() method
            # 2. Plain data (e.g., lock values like '1') - no ._asdict() method
            # We need to handle both cases for exclusive job locks to work.
            data_dict = data._asdict() if hasattr(data, "_asdict") else data
            return json.dumps(data_dict, cls=CustomJSONEncoder).encode("utf-8")

        def deserialize(self, data):
            from huey.registry import Message

            decoded = json.loads(data.decode("utf-8"), object_hook=json_loader_object_hook)
            # Message objects have specific structure: {"id": ..., "name": ..., ...}
            # Only reconstruct as Message when that structure exists.
            # Plain data (like lock values) should be returned as-is.
            if isinstance(decoded, dict) and "id" in decoded and "name" in decoded:
                return Message(**decoded)
            else:
                return decoded

    def _get_huey_storage_config(key: str) -> tuple[str, dict[str, str]]:
        if storage_url := _get_huey_redis_url():
            return "redis", {"url": storage_url, "name": key}

        huey_store_file = os.path.join(
            os.environ[HUEY_STORAGE_PATH_ENV_VAR],
            f"{key}.mlflow-huey-store",
        )
        return "sqlite", {"filename": huey_store_file, "name": key}

    with _huey_instance_map_lock:
        if instance_key not in _huey_instance_map:
            _logger.debug(f"Creating huey instance for {instance_key}")
            storage_type, storage_kwargs = _get_huey_storage_config(instance_key)
            if storage_type == "redis":
                huey_instance = RedisHuey(
                    results=False,
                    serializer=JsonSerializer(),
                    **storage_kwargs,
                )
            else:
                huey_instance = SqliteHuey(
                    results=False,
                    serializer=JsonSerializer(),
                    **storage_kwargs,
                )
            huey_submit_task_fn = huey_instance.task(retries=0)(_exec_job)
            _huey_instance_map[instance_key] = HueyInstance(
                instance=huey_instance,
                submit_task=huey_submit_task_fn,
            )
        return _huey_instance_map[instance_key]


def _launch_huey_consumer(job_name: str) -> None:
    _logger.debug(f"Starting huey consumer for job function {job_name}")

    fn_fullname = get_job_fn_fullname(job_name)
    job_fn = _load_function(fn_fullname)

    if not hasattr(job_fn, "_job_fn_metadata"):
        raise MlflowException.invalid_parameter_value(
            f"The job function {job_name} is not decorated by 'mlflow.server.jobs.job_function'."
        )

    max_job_parallelism = job_fn._job_fn_metadata.max_workers

    def _huey_consumer_thread() -> None:
        while True:
            # start MLflow job runner process
            # Put it inside the loop to ensure the job runner process alive
            job_runner_proc = _start_huey_consumer_proc(
                job_name,
                max_job_parallelism,
            )
            job_runner_proc.wait()
            time.sleep(1)

    # start job runner.
    threading.Thread(
        target=_huey_consumer_thread,
        name=f"MLflow-huey-consumer-{job_name}-watcher",
        daemon=False,
    ).start()


def _launch_periodic_tasks_consumer() -> None:
    """
    Launch a dedicated Huey consumer for periodic tasks.
    This consumer runs scheduled tasks like the online scoring scheduler.
    """
    _logger.debug("Starting dedicated Huey consumer for periodic tasks")

    def _huey_consumer_thread() -> None:
        while True:
            job_runner_proc = _start_periodic_tasks_consumer_proc()
            job_runner_proc.wait()
            time.sleep(1)

    threading.Thread(
        target=_huey_consumer_thread,
        name="MLflow-huey-consumer-periodic-tasks-watcher",
        daemon=False,
    ).start()


def _start_periodic_tasks_consumer_proc():
    cmd = [
        sys.executable,
        "-m",
        "huey.bin.huey_consumer",
        "mlflow.server.jobs._periodic_tasks_consumer.huey_instance",
        "-w",
        str(PERIODIC_TASKS_WORKER_COUNT),
    ]

    # Add quiet flag unless DEBUG logging is explicitly requested,
    # to suppress noisy huey consumer logs (e.g., Scheduler, Executing messages)
    log_level = (MLFLOW_LOGGING_LEVEL.get() or "INFO").upper()
    if log_level != "DEBUG":
        cmd.append("-q")

    # SQLite needs stale-lock recovery after a crash. Redis is shared across replicas,
    # so flushing locks at startup could remove locks held by another live instance.
    if _should_flush_periodic_locks():
        cmd.append("-f")

    return _exec_cmd(
        cmd,
        capture_output=False,
        synchronous=False,
        extra_env={MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR: str(os.getpid())},
    )


def _launch_job_runner(env_map, server_proc_pid):
    server_up_time = str(int(time.time() * 1000))
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow.server.jobs._job_runner",
        ],
        env={
            **os.environ,
            **env_map,
            "MLFLOW_SERVER_PID": str(server_proc_pid),
            MLFLOW_SERVER_UP_TIME: server_up_time,
        },
    )


# MLFLOW_SERVER_JOB_EXECUTION_ENGINE is an opt-in: unset -> the default engine (Huey today), or
# "executor" to opt into the executor framework. "huey" is intentionally not settable by name
# (see get_job_execution_engine), so nothing has to change when Huey is retired.
_DEFAULT_JOB_EXECUTION_ENGINE = "huey"
_EXECUTOR_JOB_EXECUTION_ENGINE = "executor"


def get_job_execution_engine() -> str:
    """Return the job execution engine to use.

    ``MLFLOW_SERVER_JOB_EXECUTION_ENGINE`` is an opt-in switch: leave it unset to use the default
    engine (currently Huey), or set it to ``"executor"`` to route job execution through the
    ``AbstractJobExecutor`` framework. It is intentionally *not* settable to ``"huey"`` — pinning
    the current default by name would silently break once Huey is retired — so any explicit value
    other than ``"executor"`` is rejected rather than silently accepted.
    """
    if not MLFLOW_SERVER_JOB_EXECUTION_ENGINE.is_set():
        return _DEFAULT_JOB_EXECUTION_ENGINE
    engine = MLFLOW_SERVER_JOB_EXECUTION_ENGINE.get().strip().lower()
    if engine != _EXECUTOR_JOB_EXECUTION_ENGINE:
        raise MlflowException.invalid_parameter_value(
            f"{MLFLOW_SERVER_JOB_EXECUTION_ENGINE.name} may only be set to "
            f"{_EXECUTOR_JOB_EXECUTION_ENGINE!r} to opt into the executor engine; unset it to use "
            f"the default engine (got {engine!r})."
        )
    return engine


def _launch_executor_runner(env_map, server_proc_pid):
    server_up_time = str(int(time.time() * 1000))
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow.server.jobs._executor_runner",
        ],
        env={
            **os.environ,
            **env_map,
            "MLFLOW_SERVER_PID": str(server_proc_pid),
            MLFLOW_SERVER_UP_TIME: server_up_time,
        },
    )


def _launch_job_execution_runner(env_map, server_proc_pid):
    """Launch the job runner for the configured execution engine."""
    if get_job_execution_engine() == "executor":
        return _launch_executor_runner(env_map, server_proc_pid)
    return _launch_job_runner(env_map, server_proc_pid)


def _start_watcher_to_kill_job_runner_if_mlflow_server_dies(check_interval: float = 1.0) -> None:
    mlflow_server_pid = int(os.environ.get("MLFLOW_SERVER_PID"))

    def watcher():
        while True:
            if not is_process_alive(mlflow_server_pid):
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(check_interval)

    t = threading.Thread(target=watcher, daemon=True, name="job-runner-watcher")
    t.start()


def _load_function(fullname: str) -> Callable[..., Any]:
    match fullname.split("."):
        case [*module_parts, func_name] if module_parts:
            module_name = ".".join(module_parts)
        case _:
            raise MlflowException.invalid_parameter_value(
                f"Invalid function fullname format: {fullname}"
            )
    try:
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    except ModuleNotFoundError:
        # Module doesn't exist
        raise MlflowException.invalid_parameter_value(
            f"Module not found for function '{fullname}'",
        )
    except AttributeError:
        # error_code is INVALID_PARAMETER_VALUE but this is an attribute lookup failure
        raise MlflowException.invalid_parameter_value(
            f"Function not found in module for '{fullname}'",
            error_class="ATTRIBUTE_NOT_FOUND",
        )


def _workspace_contexts_for_recovery() -> list[ContextManager[str | None]]:
    """
    Determine the set of workspace contexts that may contain unfinished jobs.

    When workspaces are disabled, this returns only a ``nullcontext``. Otherwise, it queries the
    configured workspace store to enumerate all defined workspaces so the job runner can resume
    tasks for each tenant.
    """
    if not MLFLOW_ENABLE_WORKSPACES.get():
        return [nullcontext()]

    from mlflow.server.workspace_helpers import _get_workspace_store  # avoid circular import

    store = _get_workspace_store()
    return [WorkspaceContext(workspace.name) for workspace in store.list_workspaces()]


def _for_each_unfinished_job(
    job_store: "AbstractJobStore",
    statuses: list[JobStatus],
    end_timestamp: int,
    handler: "Callable[[Job, str | None], None]",
) -> None:
    """Apply ``handler`` to each unfinished job created at/before ``end_timestamp``, per workspace.

    Shared by both engines' startup recovery. It owns the per-workspace iteration and the
    launch-time bound (so a job submitted to the freshly started server is never touched); the
    status set and the per-job action stay with each caller, since the engines act differently
    (Huey resets then re-enqueues; the executor only resets and lets the scheduler re-claim).
    """
    for workspace_ctx in _workspace_contexts_for_recovery():
        with workspace_ctx as workspace:
            for job in list(job_store.list_jobs(statuses=statuses, end_timestamp=end_timestamp)):
                handler(job, workspace)


def _enqueue_unfinished_jobs(server_launching_timestamp: int) -> None:
    from mlflow.server.handlers import _get_job_store

    job_store = _get_job_store()

    def _reset_and_enqueue(job: "Job", workspace: str | None) -> None:
        if job.status in {JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY}:
            job_store.reset_job(job.job_id)  # reset the job status to PENDING

        params = json.loads(job.params)
        timeout = job.timeout
        # Only propagate workspace to subprocess when workspaces are enabled
        if MLFLOW_ENABLE_WORKSPACES.get():
            job_workspace = job.workspace or workspace or DEFAULT_WORKSPACE_NAME
        else:
            job_workspace = None
        # Look up exclusive flag from function metadata
        fn_fullname = get_job_fn_fullname(job.job_name)
        fn_metadata = _load_function(fn_fullname)._job_fn_metadata
        # enqueue job
        _get_or_init_huey_instance(job.job_name).submit_task(
            job.job_id,
            job_workspace,
            job.job_name,
            params,
            timeout,
            fn_metadata.exclusive,
        )

    _for_each_unfinished_job(
        job_store,
        [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY],
        server_launching_timestamp,
        _reset_and_enqueue,
    )


def _validate_function_parameters(function: Callable[..., Any], params: dict[str, Any]) -> None:
    """Validate that the provided parameters match the function's required arguments.

    Args:
        function: The function to validate parameters against
        params: Dictionary of parameters provided for the function

    Raises:
        MlflowException: If required parameters are missing
    """
    sig = inspect.signature(function)

    # Get all required parameters (no default value)
    # Exclude VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs) parameters
    required_params = [
        name
        for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # Check for missing required parameters
    if missing_params := [param for param in required_params if param not in params]:
        raise MlflowException.invalid_parameter_value(
            f"Missing required parameters for function '{function.__name__}': {missing_params}. "
            f"Expected parameters: {list(sig.parameters.keys())}"
        )


def _check_requirements(backend_store_uri: str | None = None) -> None:
    from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR
    from mlflow.utils.uri import extract_db_type_from_uri

    if os.name == "nt":
        raise MlflowException("MLflow job backend does not support Windows system.")

    backend_store_uri = backend_store_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR)
    if not backend_store_uri:
        raise MlflowException(
            "MLflow job backend requires a database backend store URI but "
            "'--backend-store-uri' is not set"
        )
    try:
        extract_db_type_from_uri(backend_store_uri)
    except MlflowException:
        raise MlflowException(
            f"MLflow job backend requires a database backend store URI but got {backend_store_uri}"
        )


# The map from job name to the job function's fullname.
_job_name_to_fn_fullname_map = {}


def get_job_fn_fullname(job_name: str):
    if job_name not in _job_name_to_fn_fullname_map:
        raise MlflowException.invalid_parameter_value(f"Invalid job name: {job_name}")
    return _job_name_to_fn_fullname_map[job_name]


def _build_job_name_to_fn_fullname_map():
    from mlflow.server.jobs import _SUPPORTED_JOB_FUNCTION_LIST

    for fn_fullname in set(_SUPPORTED_JOB_FUNCTION_LIST):
        try:
            fn_meta = _load_function(fn_fullname)._job_fn_metadata
            if exist_fullname := _job_name_to_fn_fullname_map.get(fn_meta.name):
                if exist_fullname != fn_fullname:
                    _logger.warning(
                        f"The 2 job functions {fn_fullname} and {exist_fullname} have the same "
                        f"job name {fn_meta.name}, this is not allowed, skip loading function "
                        f"{fn_fullname}."
                    )
            else:
                _job_name_to_fn_fullname_map[fn_meta.name] = fn_fullname
        except Exception as e:
            _logger.warning(f"loading job function {fn_fullname} failed: {e!r}", exc_info=True)


register_post_import_hook(lambda m: _build_job_name_to_fn_fullname_map(), __name__)


def register_periodic_tasks(huey_instance) -> None:
    """
    Register all periodic tasks with the given huey instance.

    Args:
        huey_instance: The huey instance to register tasks with.
    """
    from huey import crontab

    @huey_instance.periodic_task(crontab(minute="*/1"))
    # Prevent concurrent execution if scheduler takes longer than 1 minute.
    @huey_instance.lock_task("online-scoring-scheduler-lock")
    def online_scoring_scheduler():
        """Runs every minute to fetch active scorer configs and submit scoring jobs."""
        from mlflow.genai.scorers.job import run_online_scoring_scheduler

        try:
            run_online_scoring_scheduler()
        except Exception as e:
            _logger.exception(f"Online scoring scheduler failed: {e!r}")

    _logger.info("Registered online_scoring_scheduler periodic task (runs every 1 minute)")

    @huey_instance.periodic_task(crontab(minute="*/1"))
    # Prevent concurrent execution if scheduler takes longer than 1 minute.
    @huey_instance.lock_task("trace-archival-scheduler-lock")
    def trace_archival_scheduler():
        """Runs every minute and delegates scheduling cadence to the archival service."""
        try:
            run_trace_archival_scheduler()
        except Exception as e:
            _logger.exception(f"Trace archival scheduler failed: {e!r}")

    _logger.info(
        "Registered trace_archival_scheduler periodic task (polls every 1 minute and "
        "no-ops when trace archival is disabled or unconfigured)"
    )
