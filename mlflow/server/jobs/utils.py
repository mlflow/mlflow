import errno
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cloudpickle

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
)
from mlflow.exceptions import MlflowException
from mlflow.server.constants import HUEY_STORAGE_PATH_ENV_VAR
from mlflow.utils.environment import _PythonEnv

if TYPE_CHECKING:
    import huey

_logger = logging.getLogger(__name__)


def _exponential_backoff_retry(retry_count: int) -> None:
    from huey.exceptions import RetryTask

    # We can support more retry strategies (e.g. exponential backoff) in future
    base_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get()
    max_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get()
    delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
    raise RetryTask(delay=delay)


@dataclass
class JobResult:
    succeeded: bool
    result: str | None = None  # serialized JSON string
    is_transient_error: bool | None = None
    error: str | None = None

    @classmethod
    def from_error(
        cls, e: Exception, transient_error_classes: list[type[Exception]] | None = None
    ) -> "JobResult":
        from mlflow.server.jobs import TransientError

        if isinstance(e, TransientError):
            return JobResult(succeeded=False, is_transient_error=True, error=repr(e.origin_error))

        if transient_error_classes:
            if e.__class__ in transient_error_classes:
                return JobResult(succeeded=False, is_transient_error=True, error=repr(e))

        return JobResult(
            succeeded=False,
            is_transient_error=False,
            error=repr(e),
        )

    def dump(self, path: str) -> None:
        with open(path, "w") as fp:
            json.dump(asdict(self), fp)

    @classmethod
    def load(cls, path: str) -> "JobResult":
        with open(path) as fp:
            return JobResult(**json.load(fp))


def _exit_when_orphaned(poll_interval: float = 1) -> None:
    while True:
        if os.getppid() == 1:
            os._exit(1)
        time.sleep(poll_interval)


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

    return _exec_cmd(
        [
            sys.executable,
            shutil.which("huey_consumer.py"),
            "mlflow.server.jobs._huey_consumer.huey_instance",
            "-w",
            str(max_job_parallelism),
        ],
        capture_output=False,
        synchronous=False,
        extra_env={
            MLFLOW_HUEY_INSTANCE_KEY: huey_instance_key,
        },
    )


_JOB_ENTRY_MODULE = "mlflow.server.jobs._job_subproc_entry"


def _exec_job_in_subproc(
    function_fullname: str,
    params: dict[str, Any],
    python_env: _PythonEnv | None,
    transient_error_classes: list[type[Exception]] | None,
    timeout: float | None,
    tmpdir: str,
) -> JobResult | None:
    """
    Executes the job function in a subprocess,
    If the job execution time exceeds timeout, the subprocess is killed and return None,
    otherwise return `JobResult` instance,
    """
    from mlflow.utils.process import _exec_cmd, _join_commands
    from mlflow.utils.virtualenv import (
        _get_mlflow_virtualenv_root,
        _get_uv_env_creation_command,
        _get_virtualenv_activate_cmd,
        _get_virtualenv_extra_env_vars,
        _get_virtualenv_name,
    )

    if python_env is not None:
        # set up virtual python environment
        virtual_envs_root_path = Path(_get_mlflow_virtualenv_root())
        env_name = _get_virtualenv_name(python_env, None)
        env_dir = virtual_envs_root_path / env_name
        activate_cmd = _get_virtualenv_activate_cmd(env_dir)

        if not env_dir.exists():
            _logger.info(f"Creating a python virtual environment in {env_dir}.")
            # create python environment
            env_creation_cmd = _get_uv_env_creation_command(env_dir, python_env.python)
            _exec_cmd(env_creation_cmd, capture_output=False)

            # install dependencies
            tmp_req_file = "requirements.txt"
            (Path(tmpdir) / tmp_req_file).write_text("\n".join(python_env.dependencies))
            cmd = _join_commands(activate_cmd, f"uv pip install -r {tmp_req_file}")
            _exec_cmd(
                cmd,
                cwd=tmpdir,
                extra_env=_get_virtualenv_extra_env_vars(),
                capture_output=False,
            )
        else:
            _logger.debug(f"The python environment {env_dir} already exists.")

        job_cmd = _join_commands(activate_cmd, f"exec python -m {_JOB_ENTRY_MODULE}")
    else:
        job_cmd = [sys.executable, "-m", _JOB_ENTRY_MODULE]

    result_file = str(Path(tmpdir) / "result.json")
    transient_error_classes_file = str(Path(tmpdir) / "transient_error_classes.pkl")
    with open(transient_error_classes_file, "wb") as f:
        cloudpickle.dump(transient_error_classes, f)

    with subprocess.Popen(
        job_cmd,
        env={
            **os.environ,
            "_MLFLOW_SERVER_JOB_PARAMS": json.dumps(params),
            "_MLFLOW_SERVER_JOB_FUNCTION_FULLNAME": function_fullname,
            "_MLFLOW_SERVER_JOB_RESULT_DUMP_PATH": result_file,
            "_MLFLOW_SERVER_JOB_TRANSIENT_ERROR_ClASSES_PATH": transient_error_classes_file,
        },
    ) as popen:
        try:
            popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            popen.kill()
            return None

        if popen.returncode == 0:
            return JobResult.load(result_file)

        return JobResult.from_error(
            RuntimeError(
                f"The subprocess that executes job function {function_fullname} "
                f"exists with error code {popen.returncode}"
            )
        )


def _exec_job(
    job_id: str,
    function: Callable[..., Any],
    params: dict[str, Any],
    timeout: float | None,
) -> None:
    from mlflow.server.handlers import _get_job_store

    job_store = _get_job_store()
    job_store.start_job(job_id)

    fn_metadata = function._job_fn_metadata

    with tempfile.TemporaryDirectory() as tmpdir:
        job_result = _exec_job_in_subproc(
            fn_metadata.fn_fullname,
            params,
            fn_metadata.python_env,
            fn_metadata.transient_error_classes,
            timeout,
            tmpdir,
        )

    if job_result is None:
        job_store.mark_job_timed_out(job_id)
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
        job_store.fail_job(job_id, job_result.error)


@dataclass
class HueyInstance:
    instance: "huey.SqliteHuey"
    submit_task: Callable[..., Any]


# Each job function has an individual execution pool, each execution pool
# is managed by a Huey instance.
# The `_huey_instance_map` stores the map, the key is the job function fullname,
# and the value is the `HueyInstance` object.
_huey_instance_map: dict[str, HueyInstance] = {}
_huey_instance_map_lock = threading.RLock()


def _get_or_init_huey_instance(instance_key: str):
    from huey import SqliteHuey
    from huey.serializer import Serializer

    class CloudPickleSerializer(Serializer):
        def serialize(self, data):
            return cloudpickle.dumps(data)

        def deserialize(self, data):
            return cloudpickle.loads(data)

    with _huey_instance_map_lock:
        if instance_key not in _huey_instance_map:
            _logger.info(f"Creating huey instance for {instance_key}")
            huey_store_file = os.path.join(
                os.environ[HUEY_STORAGE_PATH_ENV_VAR], f"{instance_key}.mlflow-huey-store"
            )
            huey_instance = SqliteHuey(
                filename=huey_store_file,
                results=False,
                serializer=CloudPickleSerializer(),
            )
            huey_submit_task_fn = huey_instance.task(retries=0)(_exec_job)
            _huey_instance_map[instance_key] = HueyInstance(
                instance=huey_instance,
                submit_task=huey_submit_task_fn,
            )
        return _huey_instance_map[instance_key]


def _launch_huey_consumer(job_fn_fullname: str) -> None:
    _logger.info(f"Starting huey consumer for job function {job_fn_fullname}")
    job_fn = _load_function(job_fn_fullname)

    if not hasattr(job_fn, "_job_fn_metadata"):
        raise MlflowException.invalid_parameter_value(
            f"The job function {job_fn_fullname} is not decorated by "
            "'mlflow.server.jobs.job_function'."
        )

    max_job_parallelism = job_fn._job_fn_metadata.max_workers

    def _huey_consumer_thread() -> None:
        while True:
            # start MLflow job runner process
            # Put it inside the loop to ensure the job runner process alive
            job_runner_proc = _start_huey_consumer_proc(
                job_fn_fullname,
                max_job_parallelism,
            )
            job_runner_proc.wait()
            time.sleep(1)

    # start job runner.
    threading.Thread(
        target=_huey_consumer_thread,
        name=f"MLflow-huey-consumer-{job_fn_fullname}-watcher",
        daemon=False,
    ).start()


def _launch_job_runner(env_map, server_proc_pid):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow.server.jobs._job_runner",
        ],
        env={**os.environ, **env_map, "MLFLOW_SERVER_PID": str(server_proc_pid)},
    )


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
        # Function doesn't exist in the module
        raise MlflowException.invalid_parameter_value(
            f"Function not found in module for '{fullname}'",
        )


def _enqueue_unfinished_jobs(server_launching_timestamp: int) -> None:
    from mlflow.server.handlers import _get_job_store

    job_store = _get_job_store()

    unfinished_jobs = job_store.list_jobs(
        statuses=[JobStatus.PENDING, JobStatus.RUNNING],
        # filter out jobs created after the server is launched.
        end_timestamp=server_launching_timestamp,
    )

    for job in unfinished_jobs:
        if job.status == JobStatus.RUNNING:
            job_store.reset_job(job.job_id)  # reset the job status to PENDING

        params = json.loads(job.params)
        function = _load_function(job.function_fullname)
        timeout = job.timeout
        # enqueue job
        _get_or_init_huey_instance(job.function_fullname).submit_task(
            job.job_id, function, params, timeout
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

    if shutil.which("uv") is None:
        raise MlflowException("MLflow job backend requires 'uv' but it is not installed.")

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
