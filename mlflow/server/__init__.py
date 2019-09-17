import os
import shlex
import sys

from flask import Flask, send_from_directory

from mlflow.server import handlers
from mlflow.server.handlers import get_artifact_handler, STATIC_PREFIX_ENV_VAR, _add_static_prefix
from mlflow.utils.process import exec_cmd

# NB: These are intenrnal environment variables used for communication between
# the cli and the forked gunicorn processes.
BACKEND_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_FILE_STORE"
ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"

REL_STATIC_DIR = "js/build"

app = Flask(__name__, static_folder=REL_STATIC_DIR)
STATIC_DIR = os.path.join(app.root_path, REL_STATIC_DIR)


for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)


# Serve the "get-artifact" route.
@app.route(_add_static_prefix('/get-artifact'))
def serve_artifacts():
    return get_artifact_handler()


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
@app.route(_add_static_prefix('/static-files/<path:path>'))
def serve_static_file(path):
    return send_from_directory(STATIC_DIR, path)


# Serve the index.html for the React App for all other routes.
@app.route(_add_static_prefix('/'))
def serve():
    return send_from_directory(STATIC_DIR, 'index.html')


def _build_waitress_command(waitress_opts, host, port):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return ['waitress-serve'] + \
        opts + [
            "--host=%s" % host,
            "--port=%s" % port,
            "--ident=mlflow",
            "mlflow.server:app"
    ]


def _build_gunicorn_command(gunicorn_opts, host, port, workers):
    bind_address = "%s:%s" % (host, port)
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
    return ["gunicorn"] + opts + ["-b", bind_address, "-w", "%s" % workers, "mlflow.server:app"]


def _run_server(file_store_path, default_artifact_root, host, port, static_prefix=None,
                workers=None, gunicorn_opts=None, waitress_opts=None):
    """
    Run the MLflow server, wrapping it in gunicorn or waitress on windows
    :param static_prefix: If set, the index.html asset will be served from the path static_prefix.
                          If left None, the index.html asset will be served from the root path.
    :return: None
    """
    env_map = {}
    if file_store_path:
        env_map[BACKEND_STORE_URI_ENV_VAR] = file_store_path
    if default_artifact_root:
        env_map[ARTIFACT_ROOT_ENV_VAR] = default_artifact_root
    if static_prefix:
        env_map[STATIC_PREFIX_ENV_VAR] = static_prefix

    # TODO: eventually may want waitress on non-win32
    if sys.platform == 'win32':
        full_command = _build_waitress_command(waitress_opts, host, port)
    else:
        full_command = _build_gunicorn_command(gunicorn_opts, host, port, workers or 4)
    exec_cmd(full_command, env=env_map, stream_output=True)
