"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging
import uuid
import os
from pathlib import Path
from flask import Flask, request, make_response, Response, redirect, flash, render_template_string

from mlflow import get_run
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.permissions import get_permission, Permission
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
    _get_rest_path,
    _get_tracking_store,
    _get_request_message,
    catch_mlflow_exception,
    get_endpoints,
    message_to_json,
)
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
)
from mlflow.protos.service_pb2 import (
    CreateExperiment,
    GetExperiment,
    GetRun,
    SearchRuns,
    ListArtifacts,
    GetMetricHistory,
    CreateRun,
    UpdateRun,
    LogMetric,
    LogParam,
    SetTag,
    SearchExperiments,
    DeleteExperiment,
    RestoreExperiment,
    RestoreRun,
    DeleteRun,
    UpdateExperiment,
    LogBatch,
    DeleteTag,
    SetExperimentTag,
    GetExperimentByName,
    LogModel,
)

_AUTH_CONFIG_PATH_ENV_VAR = "MLFLOW_AUTH_CONFIG_PATH"

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def _get_auth_config_path():
    return os.environ.get(
        _AUTH_CONFIG_PATH_ENV_VAR, (Path(__file__).parent / "basic_auth.ini").resolve()
    )


auth_config_path = _get_auth_config_path()
auth_config = read_auth_config(auth_config_path)
store = SqlAlchemyStore()


class ROUTES:
    HOME = "/"
    USERS = _get_rest_path("/mlflow/users")
    SIGNUP = "/signup"
    CREATE_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/create")
    GET_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/get")
    UPDATE_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/update")
    DELETE_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/delete")
    CREATE_REGISTERED_MODEL_PERMISSION = _get_rest_path(
        "/mlflow/registered-models/permissions/create"
    )
    GET_REGISTERED_MODEL_PERMISSION = _get_rest_path("/mlflow/registered-models/permissions/get")
    UPDATE_REGISTERED_MODEL_PERMISSION = _get_rest_path(
        "/mlflow/registered-models/permissions/update"
    )
    DELETE_REGISTERED_MODEL_PERMISSION = _get_rest_path(
        "/mlflow/registered-models/permissions/delete"
    )


UNPROTECTED_ROUTES = [ROUTES.USERS, ROUTES.SIGNUP]


def is_unprotected_route(path: str) -> bool:
    if path.startswith(("/static", "/favicon.ico")):
        return True
    return path in UNPROTECTED_ROUTES


def make_basic_auth_response() -> Response:
    res = make_response()
    res.status_code = 401
    res.set_data(
        "You are not authenticated. Please set the environment variables "
        f"{_TRACKING_USERNAME_ENV_VAR} and {_TRACKING_PASSWORD_ENV_VAR}."
    )
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res


def make_forbidden_response() -> Response:
    res = make_response("Permission denied")
    res.status_code = 403
    return res


def _get_request_param(param: str) -> str:
    if request.method == "GET":
        args = request.args
    elif request.method == "POST":
        args = request.json
    else:
        _logger.error(f"Unsupported HTTP method '{request.method}'")
        raise NotImplementedError()

    _logger.info(f"Getting request param (method={request.method}): {str(args)}")
    if param not in args:
        raise ValueError()
    return args[param]


def _get_experiment_id() -> str:
    return _get_request_param("experiment_id")


def _get_run_id() -> str:
    return _get_request_param("run_id")


def _get_permission_from_experiment_id() -> Permission:
    _logger.info("getting permission")
    experiment_id = _get_experiment_id()
    _logger.info(f"exp id: {experiment_id}")
    user = store.get_user(request.authorization.username)
    _logger.info(f"user: {user}")
    perm = store.get_experiment_permission(experiment_id, user.id)
    _logger.info(f"perm: {perm}")
    perm = perm.permission if perm else auth_config.default_permission
    return get_permission(perm)


def _get_permission_from_run_id() -> Permission:
    run_id = _get_run_id()
    user = store.get_user(request.authorization.username)
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run = get_run(run_id)
    experiment_id = run.info.experiment_id
    perm = store.get_experiment_permission(experiment_id, user.id)
    perm = perm.permission if perm else auth_config.default_permission
    return get_permission(perm)


def validate_can_read_experiment():
    return _get_permission_from_experiment_id().can_read


def validate_can_update_experiment():
    return _get_permission_from_experiment_id().can_update


def validate_can_delete_experiment():
    return _get_permission_from_experiment_id().can_delete


def validate_can_manage_experiment():
    # return _get_permission_from_experiment_id().can_manage
    p = _get_permission_from_experiment_id()
    _logger.info(p)
    return p.can_manage


def validate_can_read_run():
    return _get_permission_from_run_id().can_read


def validate_can_update_run():
    return _get_permission_from_run_id().can_update


def validate_can_delete_run():
    return _get_permission_from_run_id().can_delete


def validate_can_manage_run():
    return _get_permission_from_run_id().can_manage


BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    GetExperiment: validate_can_read_experiment,
    GetExperimentByName: validate_can_read_experiment,
    UpdateExperiment: validate_can_update_experiment,
    DeleteExperiment: validate_can_delete_experiment,
    RestoreExperiment: validate_can_delete_experiment,
    SetExperimentTag: validate_can_update_experiment,
    SearchExperiments: validate_can_read_experiment,
    # Routes for runs
    CreateRun: validate_can_update_experiment,
    GetRun: validate_can_read_run,
    UpdateRun: validate_can_update_run,
    DeleteRun: validate_can_delete_run,
    RestoreRun: validate_can_delete_run,
    SearchRuns: validate_can_read_run,
    ListArtifacts: validate_can_read_run,
    GetMetricHistory: validate_can_read_run,
    LogMetric: validate_can_update_run,
    LogParam: validate_can_update_run,
    SetTag: validate_can_update_run,
    DeleteTag: validate_can_update_run,
    LogModel: validate_can_update_run,
    LogBatch: validate_can_update_run,
}


def get_before_request_handler(request_class):
    return BEFORE_REQUEST_HANDLERS.get(request_class)


BEFORE_REQUEST_VALIDATORS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_before_request_handler)
    for method in methods
}

BEFORE_REQUEST_VALIDATORS.update(
    {
        (ROUTES.GET_EXPERIMENT_PERMISSION, "GET"): validate_can_manage_experiment,
        (ROUTES.CREATE_EXPERIMENT_PERMISSION, "PUT"): validate_can_manage_experiment,
        (ROUTES.UPDATE_EXPERIMENT_PERMISSION, "POST"): validate_can_manage_experiment,
        (ROUTES.DELETE_EXPERIMENT_PERMISSION, "DELETE"): validate_can_manage_experiment,
    }
)


def _before_request():
    if is_unprotected_route(request.path):
        return

    _user = request.authorization.username if request.authorization else None
    _logger.debug(f"before_request: {request.method} {request.path} (user: {_user})")

    if request.authorization is None:
        return make_basic_auth_response()

    username = request.authorization.username
    password = request.authorization.password
    if not store.authenticate_user(username, password):
        # let user attempt login again
        return make_basic_auth_response()

    # authorization
    validator = BEFORE_REQUEST_VALIDATORS.get((request.path, request.method))
    if validator:
        _logger.info(f"Calling validator: {validator.__name__}")
        if not validator():
            return make_forbidden_response()


def _after_request(resp):
    # TODO: Implement post-request logic
    return resp


def signup():
    # TODO: add css
    return render_template_string(
        r"""
<form action="{{ users_route }}" method="post">
  Username:
  <br>
  <input type=text name=username>
  <br>
  Password:
  <br>
  <input type=password name=password>
  <br>
  <br>
  <input type="submit" value="Signup">
</form>
<style>
.alert.error {
  color: red;
}
</style>
{% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
    {% for category, message in messages %}
      <p class="alert {{ category }}">{{ message }}</p>
    {% endfor %}
  {% endif %}
{% endwith %}
""",
        users_route=ROUTES.USERS,
    )


@catch_mlflow_exception
def create_user():
    content_type = request.headers.get("Content-Type")
    if content_type == "application/x-www-form-urlencoded":
        username = request.form["username"]
        password = request.form["password"]
    elif content_type == "application/json":
        username = request.json["username"]
        password = request.json["password"]
    else:
        return make_response(f"Invalid content type: '{content_type}'", 400)

    if store.has_user(username):
        flash(f"Username has already been taken: '{username}'", category="error")
        return redirect(ROUTES.SIGNUP)

    store.create_user(username, password)
    flash(f"Successfully signed up user: '{username}'")
    return redirect(ROUTES.HOME)


@catch_mlflow_exception
def create_experiment_permission(experiment_id: str, user_id: int, permission: Permission):
    return store.create_experiment_permission(experiment_id, user_id, permission.name)


@catch_mlflow_exception
def get_experiment_permission() -> str:
    experiment_id = _get_experiment_id()
    user_id = int(_get_request_param("user_id"))
    return store.get_experiment_permission(experiment_id, user_id).permission


@catch_mlflow_exception
def update_experiment_permission(experiment_id: str, user_id: int, permission: Permission):
    store.update_experiment_permission(experiment_id, user_id, permission.name)


@catch_mlflow_exception
def delete_experiment_permission():
    experiment_id = _get_experiment_id()
    user_id = int(_get_request_param("user_id"))
    store.delete_experiment_permission(experiment_id, user_id)


def _enable_auth(app: Flask):
    """
    Enables authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    """
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )
    # secret key required for flashing
    if not app.secret_key:
        app.secret_key = str(uuid.uuid4())

    _logger.debug("Database URI: %s", auth_config.database_uri)
    store.init_db(auth_config.database_uri)

    app.add_url_rule(
        rule=ROUTES.SIGNUP,
        view_func=signup,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=ROUTES.USERS,
        view_func=create_user,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=ROUTES.CREATE_EXPERIMENT_PERMISSION,
        view_func=create_experiment_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=ROUTES.GET_EXPERIMENT_PERMISSION,
        view_func=get_experiment_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=ROUTES.UPDATE_EXPERIMENT_PERMISSION,
        view_func=update_experiment_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=ROUTES.DELETE_EXPERIMENT_PERMISSION,
        view_func=delete_experiment_permission,
        methods=["POST"],
    )

    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
