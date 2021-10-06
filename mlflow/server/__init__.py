import os
import shlex
import sys
import textwrap
import boto3
import smart_open

from flask import Flask, send_from_directory, Response, request, jsonify

from mlflow.server import handlers
from mlflow.server.handlers import (
    get_artifact_handler,
    STATIC_PREFIX_ENV_VAR,
    _add_static_prefix,
    get_model_version_artifact_handler,
)
from mlflow.utils.process import exec_cmd
import logging

logging.basicConfig(level=logging.DEBUG)

for name in logging.root.manager.loggerDict:
    if name.startswith("smart_open.s3") or name.startswith("flask"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger(name)
        logger.disabled = True

# NB: These are intenrnal environment variables used for communication between
# the cli and the forked gunicorn processes.
BACKEND_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_FILE_STORE"
ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"
PROMETHEUS_EXPORTER_ENV_VAR = "prometheus_multiproc_dir"

REL_STATIC_DIR = "js/build"

app = Flask(__name__, static_folder=REL_STATIC_DIR)
STATIC_DIR = os.path.join(app.root_path, REL_STATIC_DIR)


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


# Serve the "get-artifact" route.
@app.route(_add_static_prefix("/get-artifact"))
def serve_artifacts():
    return get_artifact_handler()


def _upload_to_s3(stream, bucket_name, key, chunk_size=10 * 1024 ** 2):
    import time

    # smart_open:
    # https://github.com/RaRe-Technologies/smart_open

    # smart_open performs a multi part upload (MPU):
    # https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html

    # Google Cloud Storage storage supports MPU:
    # https://cloud.google.com/storage/docs/multipart-uploads

    # Azure Blob Storage also supports MPU:
    # https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-block-blobs--append-blobs--and-page-blobs

    url = f"s3://{bucket_name}/{key}"
    transport_params = {
        "client": boto3.client("s3"),
        # Size of each part (default: 50 MB)
        # Doc: https://github.com/RaRe-Technologies/smart_open/blob/develop/howto.md#how-to-write-to-s3-efficiently
        "min_part_size": chunk_size,
    }
    start = time.time()
    with smart_open.open(url, "wb", transport_params=transport_params) as fout:
        while True:
            chunk = stream.read(chunk_size)
            if len(chunk) == 0:
                break
            fout.write(chunk)
    return time.time() - start


@app.route(_add_static_prefix("/artifacts/upload"), methods=["POST"])
def _upload_artifact():
    print("called")
    bucket_name = request.args.get("bucket_name")
    key = request.args.get("key")
    duration = _upload_to_s3(request.stream, bucket_name, key)
    return jsonify({"duration": duration})


def _read_from_s3(bucket_name, key, chunk_size=8192):
    url = f"s3://{bucket_name}/{key}"
    transport_params = {"client": boto3.client("s3")}
    # smart_open performs a multi part upload
    with smart_open.open(url, "rb", transport_params=transport_params) as f:
        while True:
            chunk = f.read(chunk_size)
            if len(chunk) == 0:
                break
            yield chunk


# Serve the "get-artifact" route.
@app.route(_add_static_prefix("/artifacts/get"), methods=["GET"])
def stream_get_artifact():
    bucket_name = request.args.get("bucket_name")
    key = request.args.get("key")
    return Response(_read_from_s3(bucket_name, key))


# Serve the "model-versions/get-artifact" route.
@app.route(_add_static_prefix("/model-versions/get-artifact"))
def serve_model_version_artifact():
    return get_model_version_artifact_handler()


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
@app.route(_add_static_prefix("/static-files/<path:path>"))
def serve_static_file(path):
    return send_from_directory(STATIC_DIR, path)


# Serve the index.html for the React App for all other routes.
@app.route(_add_static_prefix("/"))
def serve():
    if os.path.exists(os.path.join(STATIC_DIR, "index.html")):
        return send_from_directory(STATIC_DIR, "index.html")

    text = textwrap.dedent(
        """
    Unable to display MLflow UI - landing page (index.html) not found.

    You are very likely running the MLflow server using a source installation of the Python MLflow
    package.

    If you are a developer making MLflow source code changes and intentionally running a source
    installation of MLflow, you can view the UI by running the Javascript dev server:
    https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.rst#running-the-javascript-dev-server

    Otherwise, uninstall MLflow via 'pip uninstall mlflow', reinstall an official MLflow release
    from PyPI via 'pip install mlflow', and rerun the MLflow server.
    """
    )
    return Response(text, mimetype="text/plain")


def _build_waitress_command(waitress_opts, host, port):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return (
        ["waitress-serve"]
        + opts
        + ["--host=%s" % host, "--port=%s" % port, "--ident=mlflow", "mlflow.server:app"]
    )


def _build_gunicorn_command(gunicorn_opts, host, port, workers):
    bind_address = "%s:%s" % (host, port)
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
    return (
        ["gunicorn"]
        + opts
        + [
            "-b",
            bind_address,
            "-w",
            "%s" % workers,
            "mlflow.server:app",
            "--timeout",
            "120",
            "--reload",
        ]
    )


def _run_server(
    file_store_path,
    default_artifact_root,
    host,
    port,
    static_prefix=None,
    workers=None,
    gunicorn_opts=None,
    waitress_opts=None,
    expose_prometheus=None,
):
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

    if expose_prometheus:
        env_map[PROMETHEUS_EXPORTER_ENV_VAR] = expose_prometheus

    # TODO: eventually may want waitress on non-win32
    if sys.platform == "win32":
        full_command = _build_waitress_command(waitress_opts, host, port)
    else:
        full_command = _build_gunicorn_command(gunicorn_opts, host, port, workers or 4)
    exec_cmd(full_command, env=env_map, stream_output=True)
