import entrypoints
import os
import shlex
import sys
import textwrap

from flask import Flask, send_from_directory, Response, request, redirect, jsonify
from werkzeug.security import check_password_hash

from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.handlers import (
    get_artifact_handler,
    get_metric_history_bulk_handler,
    STATIC_PREFIX_ENV_VAR,
    _add_static_prefix,
    get_model_version_artifact_handler,
)
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
    JWTManager,
    set_access_cookies,
)
from . import db

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
app.config["DEBUG"] = True
app.config["JWT_SECRET_KEY"] = "secret"
jwt = JWTManager(app)

# Bcrypt is set as default SECURITY_PASSWORD_HASH, which requires a salt
# Generate a good salt using: secrets.SystemRandom().getrandbits(128)
app.config["SECURITY_PASSWORD_SALT"] = os.environ.get(
    "SECURITY_PASSWORD_SALT", "146585145368132386173505678016728509634"
)


db.init_app(app)
with app.app_context():
    db.init_tables()

STATIC_DIR = os.path.join(app.root_path, REL_STATIC_DIR)


for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)


@app.route("/ajax-api/2.0/mlflow/auth/signup", methods=["POST"])
def signup():
    email = request.json.get("email", None)
    password = request.json.get("password", None)
    db.create_user(email, password)
    return jsonify({"message": "User created successfully"}), 200


@app.route("/ajax-api/2.0/mlflow/auth/signin", methods=["POST"])
def singin():
    email = request.json.get("email", None)
    password = request.json.get("password", None)
    user = db.get_user_by_email(email)
    if user is None or not check_password_hash(user.password, password):
        return jsonify({"error": "error"}), 401

    access_token = create_access_token(identity=email)
    return jsonify(access_token=access_token)


@app.route("/api/2.0/mlflow/auth/experiments/<experiment_id>/permissions", methods=["GET"])
@jwt_required()
def _list_experiment_permissions(experiment_id):
    user = db.get_user_by_email(get_jwt_identity())
    perm = db.get_experiment_permission(experiment_id, user.id)
    if perm is None or not handlers.get_permission(perm.permission).can_manage():
        return "You do not have access to view permissions of this experiment", 403

    # Make permissions serializable
    perms = [
        {
            "user_id": p.user_id,
            "permission": p.permission,
        }
        for p in db.list_experiment_permissions(experiment_id)
    ]

    return {"permissions": perms}, 200


@app.route("/api/2.0/mlflow/auth/experiments/<experiment_id>/permissions", methods=["PUT"])
def _create_experiment_permissions(experiment_id):
    perm = db.get_experiment_permission(request.authorization.username, experiment_id)
    if perm is None or not handlers.get_permission(perm.permission).can_manage():
        return "You do not have access to create permissions of this experiment", 403

    db.create_experiment_permission(request.json["user"], experiment_id, request.json["permission"])
    return "OK", 200


@app.route("/api/2.0/mlflow/auth/experiments/<experiment_id>/permissions", methods=["POST"])
def _update_experiment_permissions(experiment_id):
    perm = db.get_experiment_permission(request.authorization.username, experiment_id)
    if perm is None or not handlers.get_permission(perm.permission).can_manage():
        return "You do not have access to update permissions of this experiment", 403

    db.update_e(request.json["user"], experiment_id, request.json["permission"])
    return "OK", 200


@app.route("/api/2.0/mlflow/auth/experiments/<experiment_id>/permissions", methods=["DELETE"])
def _delete_experiment_permissions(experiment_id):
    perm = db.get_experiment_permission(request.authorization.username, experiment_id)
    if perm is None or not handlers.get_permission(perm.permission).can_manage():
        return "You do not have permission to delete permissions of this experiment", 403

    db.delete_experiment_permission(request.json["user"], experiment_id)
    return "OK", 200


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


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
@app.route(_add_static_prefix("/static-files/<path:path>"))
def serve_static_file(path):
    return send_from_directory(STATIC_DIR, path)


# Serve the index.html for the React App for all other routes.
@app.route(_add_static_prefix("/"))
def mlflow():
    if os.path.exists(os.path.join(STATIC_DIR, "index.html")):
        return send_from_directory(STATIC_DIR, "index.html")

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


def _get_app_name() -> str:
    """Search for plugins for custom mlflow app, otherwise return default."""
    apps = list(entrypoints.get_group_all("mlflow.app"))
    # Default, nothing installed
    if len(apps) == 0:
        return f"{__name__}:app"
    # Cannot install more than one
    if len(apps) > 1:
        raise MlflowException(
            "Multiple server plugins detected. "
            "Only one server plugin may be installed. "
            f"Detected plugins: {', '.join([f'{a.module_name}.{a.object_name}' for a in apps])}"
        )
    # Has a plugin installed
    plugin_app = apps[0]
    return f"{plugin_app.module_name}:{plugin_app.object_name}"


def _build_waitress_command(waitress_opts, host, port, app_name):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return (
        ["waitress-serve"]
        + opts
        + ["--host=%s" % host, "--port=%s" % port, "--ident=mlflow", app_name]
    )


def _build_gunicorn_command(gunicorn_opts, host, port, workers, app_name):
    bind_address = f"{host}:{port}"
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
    return ["gunicorn"] + opts + ["-b", bind_address, "-w", "%s" % workers, app_name]


def _run_server(
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

    app_name = _get_app_name()
    # TODO: eventually may want waitress on non-win32
    if sys.platform == "win32":
        full_command = _build_waitress_command(waitress_opts, host, port, app_name)
    else:
        full_command = _build_gunicorn_command(gunicorn_opts, host, port, workers or 4, app_name)
    _exec_cmd(full_command, extra_env=env_map, capture_output=False)
