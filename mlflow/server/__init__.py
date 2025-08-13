import importlib
import importlib.metadata
import os
import shlex
import sys
import textwrap
import types
import warnings

from flask import Flask, Response, send_from_directory
from packaging.version import Version

from mlflow.environment_variables import MLFLOW_FLASK_SERVER_SECRET_KEY
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.handlers import (
    STATIC_PREFIX_ENV_VAR,
    _add_static_prefix,
    create_promptlab_run_handler,
    gateway_proxy_handler,
    get_artifact_handler,
    get_logged_model_artifact_handler,
    get_metric_history_bulk_handler,
    get_metric_history_bulk_interval_handler,
    get_model_version_artifact_handler,
    get_trace_artifact_handler,
    search_datasets_handler,
    upload_artifact_handler,
)
from mlflow.utils.os import is_windows
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION

# NB: These are internal environment variables used for communication between
# the cli and the forked gunicorn processes.
BACKEND_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_FILE_STORE"
REGISTRY_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_REGISTRY_STORE"
ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"
ARTIFACTS_DESTINATION_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_DESTINATION"
PROMETHEUS_EXPORTER_ENV_VAR = "prometheus_multiproc_dir"
SERVE_ARTIFACTS_ENV_VAR = "_MLFLOW_SERVER_SERVE_ARTIFACTS"
ARTIFACTS_ONLY_ENV_VAR = "_MLFLOW_SERVER_ARTIFACTS_ONLY"

REL_STATIC_DIR = "js/build"

app = Flask(__name__, static_folder=REL_STATIC_DIR)
IS_FLASK_V1 = Version(importlib.metadata.version("flask")) < Version("2.0")


for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)

if os.getenv(PROMETHEUS_EXPORTER_ENV_VAR):
    from mlflow.server.prometheus_exporter import activate_prometheus_exporter

    prometheus_metrics_path = os.getenv(PROMETHEUS_EXPORTER_ENV_VAR)
    if not os.path.exists(prometheus_metrics_path):
        os.makedirs(prometheus_metrics_path)
    activate_prometheus_exporter(app)


# Provide a health check endpoint to ensure the application is responsive
@app.route("/health")
def health():
    return "OK", 200


# Provide an endpoint to query the version of mlflow running on the server
@app.route("/version")
def version():
    return VERSION, 200


# Serve the "get-artifact" route.
@app.route(_add_static_prefix("/get-artifact"))
def serve_artifacts():
    return get_artifact_handler()


# Serve the "model-versions/get-artifact" route.
@app.route(_add_static_prefix("/model-versions/get-artifact"))
def serve_model_version_artifact():
    return get_model_version_artifact_handler()


# Serve the "metrics/get-history-bulk" route.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/metrics/get-history-bulk"))
def serve_get_metric_history_bulk():
    return get_metric_history_bulk_handler()


# Serve the "metrics/get-history-bulk-interval" route.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval"))
def serve_get_metric_history_bulk_interval():
    return get_metric_history_bulk_interval_handler()


# Serve the "experiments/search-datasets" route.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/experiments/search-datasets"), methods=["POST"])
def serve_search_datasets():
    return search_datasets_handler()


# Serve the "runs/create-promptlab-run" route.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/runs/create-promptlab-run"), methods=["POST"])
def serve_create_promptlab_run():
    return create_promptlab_run_handler()


@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/gateway-proxy"), methods=["POST", "GET"])
def serve_gateway_proxy():
    return gateway_proxy_handler()


@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/upload-artifact"), methods=["POST"])
def serve_upload_artifact():
    return upload_artifact_handler()


# Serve the "/get-trace-artifact" route to allow frontend to fetch trace artifacts
# and render them in the Trace UI. The request body should contain the request_id
# of the trace.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/get-trace-artifact"), methods=["GET"])
def serve_get_trace_artifact():
    return get_trace_artifact_handler()


@app.route(
    _add_static_prefix("/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files"),
    methods=["GET"],
)
def serve_get_logged_model_artifact(model_id: str):
    return get_logged_model_artifact_handler(model_id)


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
# The files are hashed based on source code, so ok to send Cache-Control headers via max_age.
@app.route(_add_static_prefix("/static-files/<path:path>"))
def serve_static_file(path):
    if IS_FLASK_V1:
        return send_from_directory(app.static_folder, path, cache_timeout=2419200)
    else:
        return send_from_directory(app.static_folder, path, max_age=2419200)


# Serve the index.html for the React App for all other routes.
@app.route(_add_static_prefix("/"))
def serve():
    if os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")

    text = textwrap.dedent(
        """
    Unable to display MLflow UI - landing page (index.html) not found.

    You are very likely running the MLflow server using a source installation of the Python MLflow
    package.

    If you are a developer making MLflow source code changes and intentionally running a source
    installation of MLflow, you can view the UI by running the Javascript dev server:
    https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#running-the-javascript-dev-server

    Otherwise, uninstall MLflow via 'pip uninstall mlflow', reinstall an official MLflow release
    from PyPI via 'pip install mlflow', and rerun the MLflow server.
    """
    )
    return Response(text, mimetype="text/plain")


def _find_app(app_name: str) -> str:
    apps = get_entry_points("mlflow.app")
    for app in apps:
        if app.name == app_name:
            return app.value

    raise MlflowException(
        f"Failed to find app '{app_name}'. Available apps: {[a.name for a in apps]}"
    )


def _is_factory(app: str) -> bool:
    """
    Returns True if the given app is a factory function, False otherwise.

    Args:
        app: The app to check, e.g. "mlflow.server.app:app
    """
    module, obj_name = app.rsplit(":", 1)
    mod = importlib.import_module(module)
    obj = getattr(mod, obj_name)
    return isinstance(obj, types.FunctionType)


def get_app_client(app_name: str, *args, **kwargs):
    """
    Instantiate a client provided by an app.

    Args:
        app_name: The app name defined in `setup.py`, e.g., "basic-auth".
        args: Additional arguments passed to the app client constructor.
        kwargs: Additional keyword arguments passed to the app client constructor.

    Returns:
        An app client instance.
    """
    clients = get_entry_points("mlflow.app.client")
    for client in clients:
        if client.name == app_name:
            cls = client.load()
            return cls(*args, **kwargs)

    raise MlflowException(
        f"Failed to find client for '{app_name}'. Available clients: {[c.name for c in clients]}"
    )


def _build_waitress_command(waitress_opts, host, port, app_name, is_factory):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return [
        sys.executable,
        "-m",
        "waitress",
        *opts,
        f"--host={host}",
        f"--port={port}",
        "--ident=mlflow",
        *(["--call"] if is_factory else []),
        app_name,
    ]


def _build_gunicorn_command(gunicorn_opts, host, port, workers, app_name):
    bind_address = f"{host}:{port}"
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
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


def _build_uvicorn_command(uvicorn_opts, host, port, workers, app_name):
    """Build command to run uvicorn server."""
    opts = shlex.split(uvicorn_opts) if uvicorn_opts else []
    return [
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
        app_name,
    ]


def _run_server(
    *,
    file_store_path,
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
):
    """
    Run the MLflow server, wrapping it in gunicorn, uvicorn, or waitress on windows

    Args:
        static_prefix: If set, the index.html asset will be served from the path static_prefix.
                       If left None, the index.html asset will be served from the root path.
        uvicorn_opts: Additional options for uvicorn server.

    Returns:
        None
    """
    env_map = {}
    if file_store_path:
        env_map[BACKEND_STORE_URI_ENV_VAR] = file_store_path
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

    secret_key = MLFLOW_FLASK_SERVER_SECRET_KEY.get()
    if secret_key:
        env_map[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = secret_key

    # Determine which server we're using (only one should be true)
    using_gunicorn = gunicorn_opts is not None
    using_waitress = waitress_opts is not None
    using_uvicorn = not using_gunicorn and not using_waitress

    if app_name is None:
        is_factory = False
        # For uvicorn, use the FastAPI app; for gunicorn/waitress, use the Flask app
        app = "mlflow.server.fastapi_app:app" if using_uvicorn else f"{__name__}:app"
    else:
        app = _find_app(app_name)
        is_factory = _is_factory(app)
        # `waitress` doesn't support `()` syntax for factory functions.
        # Instead, we need to use the `--call` flag.
        # Don't use () syntax if we're using uvicorn
        use_factory_syntax = not is_windows() and is_factory and not using_uvicorn
        app = f"{app}()" if use_factory_syntax else app

    # Determine which server to use
    if using_uvicorn:
        # Use uvicorn (default when no specific server options are provided)
        full_command = _build_uvicorn_command(uvicorn_opts, host, port, workers or 4, app)
    elif using_waitress:
        # Use waitress if explicitly requested
        warnings.warn(
            "We recommend using uvicorn for improved performance. "
            "Please use uvicorn by default or specify '--uvicorn-opts' "
            "instead of '--waitress-opts'.",
            FutureWarning,
            stacklevel=2,
        )
        full_command = _build_waitress_command(waitress_opts, host, port, app, is_factory)
    elif using_gunicorn:
        # Use gunicorn if explicitly requested
        if sys.platform == "win32":
            raise MlflowException(
                "Gunicorn is not supported on Windows. "
                "Please use uvicorn (default) or specify '--waitress-opts'."
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
        # This shouldn't happen given the logic in CLI, but handle it just in case
        raise MlflowException("No server configuration specified.")
    _exec_cmd(full_command, extra_env=env_map, capture_output=False)
