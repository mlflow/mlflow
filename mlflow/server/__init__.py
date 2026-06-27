import importlib
import logging
import os
import secrets
import shlex
import signal
import sys
import tempfile
import types
import warnings
from pathlib import Path

_logger = logging.getLogger("mlflow.server")

from mlflow.environment_variables import (
    _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN,
    _MLFLOW_SGI_NAME,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
)
from mlflow.exceptions import MlflowException
from mlflow.server.constants import (
    ARTIFACT_ROOT_ENV_VAR,
    ARTIFACTS_DESTINATION_ENV_VAR,
    ARTIFACTS_ONLY_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
    PROMETHEUS_EXPORTER_ENV_VAR,
    READ_REPLICA_BACKEND_STORE_URI_ENV_VAR,
    REGISTRY_STORE_URI_ENV_VAR,
    SECRETS_CACHE_MAX_SIZE_ENV_VAR,
    SECRETS_CACHE_TTL_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
)
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.process import _exec_cmd

REL_STATIC_DIR = "js/build"


def _find_app(app_name: str) -> str:
    apps = get_entry_points("mlflow.app")
    for app in apps:
        if app.name == app_name:
            return app.value

    raise MlflowException(
        f"Failed to find app '{app_name}'. Available apps: {[a.name for a in apps]}"
    )


def _is_factory(app: str) -> bool:
    module, obj_name = app.rsplit(":", 1)
    mod = importlib.import_module(module)
    obj = getattr(mod, obj_name)
    return isinstance(obj, types.FunctionType)


def get_app_client(app_name: str, *args, **kwargs):
    clients = get_entry_points("mlflow.app.client")
    for client in clients:
        if client.name == app_name:
            cls = client.load()
            return cls(*args, **kwargs)

    raise MlflowException(
        f"Failed to find client for '{app_name}'. Available clients: {[c.name for c in clients]}"
    )


def _build_gunicorn_command(gunicorn_opts, host, port, workers, app_name):
    bind_address = f"{host}:{port}"
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
    if not any("-k" in o or "--worker-class" in o for o in opts):
        opts.extend(["-k", "uvicorn.workers.UvicornWorker"])
    return [
        sys.executable,
        "-m",
        "gunicorn",
        *opts,
        "-b",
        bind_address,
        "-w",
        str(workers),
        app_name,
    ]


_UVICORN_LOG_CONFIG = Path(__file__).parent / "uvicorn_log_config.yaml"


def _build_uvicorn_command(
    uvicorn_opts, host, port, workers, app_name, env_file=None, is_factory=False
):
    opts = shlex.split(uvicorn_opts) if uvicorn_opts else []
    if not any(o == "--log-config" or o.startswith("--log-config=") for o in opts):
        opts.extend(["--log-config", str(_UVICORN_LOG_CONFIG)])
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        *opts,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    if env_file:
        cmd.extend(["--env-file", env_file])
    if is_factory:
        cmd.append("--factory")
    cmd.append(app_name)
    return cmd


def _run_server(
    *,
    file_store_path,
    read_replica_backend_store_uri=None,
    registry_store_uri,
    default_artifact_root,
    serve_artifacts,
    artifacts_only,
    artifacts_destination,
    host,
    port,
    static_prefix=None,
    workers=None,
    gunicorn_opts=None,
    waitress_opts=None,
    expose_prometheus=None,
    app_name=None,
    uvicorn_opts=None,
    env_file=None,
    secrets_cache_ttl=None,
    secrets_cache_max_size=None,
):
    env_map = {}
    if file_store_path:
        env_map[BACKEND_STORE_URI_ENV_VAR] = file_store_path
    if read_replica_backend_store_uri:
        env_map[READ_REPLICA_BACKEND_STORE_URI_ENV_VAR] = read_replica_backend_store_uri
    if registry_store_uri:
        env_map[REGISTRY_STORE_URI_ENV_VAR] = registry_store_uri
    if default_artifact_root:
        env_map[ARTIFACT_ROOT_ENV_VAR] = default_artifact_root
    if serve_artifacts:
        env_map[SERVE_ARTIFACTS_ENV_VAR] = "true"
    if artifacts_only:
        env_map[ARTIFACTS_ONLY_ENV_VAR] = "true"
    if artifacts_destination:
        env_map[ARTIFACTS_DESTINATION_ENV_VAR] = artifacts_destination
    if static_prefix:
        env_map[STATIC_PREFIX_ENV_VAR] = static_prefix

    if expose_prometheus:
        env_map[PROMETHEUS_EXPORTER_ENV_VAR] = expose_prometheus

    if secrets_cache_ttl is not None:
        env_map[SECRETS_CACHE_TTL_ENV_VAR] = str(secrets_cache_ttl)
    if secrets_cache_max_size is not None:
        env_map[SECRETS_CACHE_MAX_SIZE_ENV_VAR] = str(secrets_cache_max_size)

    if secret_key := MLFLOW_FLASK_SERVER_SECRET_KEY.get():
        env_map[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = secret_key

    using_gunicorn = gunicorn_opts is not None
    using_uvicorn = not using_gunicorn

    if waitress_opts is not None:
        warnings.warn(
            "--waitress-opts is deprecated and will be removed in a future release. "
            "The server now uses uvicorn by default. Use '--uvicorn-opts' instead.",
            FutureWarning,
            stacklevel=2,
        )
        using_uvicorn = True

    if using_uvicorn:
        env_map[_MLFLOW_SGI_NAME.name] = "uvicorn"
    elif using_gunicorn:
        env_map[_MLFLOW_SGI_NAME.name] = "gunicorn"

    if app_name is None:
        is_factory = False
        app = "mlflow.server.fastapi_app:app"
    else:
        app = _find_app(app_name)
        is_factory = _is_factory(app)
        use_factory_syntax = is_factory and not using_uvicorn
        app = f"{app}()" if use_factory_syntax else app

    if using_uvicorn:
        full_command = _build_uvicorn_command(
            uvicorn_opts, host, port, workers or 4, app, env_file, is_factory
        )
    elif using_gunicorn:
        if sys.platform == "win32":
            raise MlflowException(
                "Gunicorn is not supported on Windows. "
                "Please use uvicorn (default) or specify '--uvicorn-opts'."
            )
        warnings.warn(
            "We recommend using uvicorn for improved performance. "
            "Please use uvicorn by default or specify '--uvicorn-opts' "
            "instead of '--gunicorn-opts'.",
            FutureWarning,
            stacklevel=2,
        )
        full_command = _build_gunicorn_command(gunicorn_opts, host, port, workers or 4, app)
    else:
        raise MlflowException("No server configuration specified.")

    job_execution_enabled = False
    if MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        from mlflow.server.jobs.utils import _check_requirements

        try:
            _check_requirements(file_store_path)
            job_execution_enabled = True
        except Exception as e:
            _logger.warning(
                f"MLflow job execution requirements not met ({e!s}). "
                "Server will start without job execution support. "
                "Errors will be surfaced at job invocation time."
            )

    if app_name == "basic-auth" and job_execution_enabled:
        env_map[_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name] = secrets.token_hex(32)

    if job_execution_enabled:
        env_map[HUEY_STORAGE_PATH_ENV_VAR] = (
            tempfile.mkdtemp(dir="/dev/shm") if os.path.exists("/dev/shm") else tempfile.mkdtemp()
        )

    server_proc = _exec_cmd(
        full_command,
        extra_env=env_map,
        capture_output=False,
        synchronous=False,
    )

    def _forward_signal(signum, _frame):
        if server_proc.poll() is not None:
            return
        try:
            server_proc.send_signal(signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    if job_execution_enabled:
        from mlflow.environment_variables import MLFLOW_GATEWAY_URI, MLFLOW_TRACKING_URI
        from mlflow.server.jobs.utils import _launch_job_runner

        server_uri = f"http://{host}:{port}"
        job_env = {
            **env_map,
            MLFLOW_TRACKING_URI.name: server_uri,
        }
        if not MLFLOW_GATEWAY_URI.is_set():
            job_env[MLFLOW_GATEWAY_URI.name] = server_uri
        _launch_job_runner(job_env, server_proc.pid)

    server_proc.wait()
