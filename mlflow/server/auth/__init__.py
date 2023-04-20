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
from typing import Callable

from flask import Flask, request, make_response, Response, redirect, flash, render_template_string

from mlflow import get_run, MlflowException
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.permissions import get_permission, Permission, MANAGE
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
    _get_rest_path,
    catch_mlflow_exception,
    get_endpoints,
)
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
)
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.protos.service_pb2 import (
    GetExperiment,
    GetRun,
    ListArtifacts,
    GetMetricHistory,
    CreateRun,
    UpdateRun,
    LogMetric,
    LogParam,
    SetTag,
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
from mlflow.protos.model_registry_pb2 import (
    GetRegisteredModel,
    DeleteRegisteredModel,
    UpdateRegisteredModel,
    RenameRegisteredModel,
    GetLatestVersions,
    CreateModelVersion,
    GetModelVersion,
    DeleteModelVersion,
    UpdateModelVersion,
    TransitionModelVersionStage,
    GetModelVersionDownloadUri,
    SetRegisteredModelTag,
    DeleteRegisteredModelTag,
    SetModelVersionTag,
    DeleteModelVersionTag,
    SetRegisteredModelAlias,
    DeleteRegisteredModelAlias,
    GetModelVersionByAlias,
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
        raise MlflowException(
            f"Unsupported HTTP method '{request.method}'",
            BAD_REQUEST,
        )

    if param not in args:
        raise MlflowException(
            f"Missing value for required parameter '{param}'. "
            "See the API docs for more information about request parameters.",
            INVALID_PARAMETER_VALUE,
        )
    return args[param]


def _get_permission_from_store_or_default(store_func: Callable[[], str]) -> Permission:
    """
    Attempts to get permission from store,
    and returns default permission if no record is found.
    """
    try:
        perm = store_func()
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            perm = auth_config.default_permission
        else:
            raise
    return get_permission(perm)


def _get_permission_from_experiment_id() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    user = store.get_user(request.authorization.username)
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, user.id).permission
    )


def _get_permission_from_run_id() -> Permission:
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run_id = _get_request_param("run_id")
    run = get_run(run_id)
    experiment_id = run.info.experiment_id
    user = store.get_user(request.authorization.username)
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, user.id).permission
    )


def _get_permission_from_registered_model_name() -> Permission:
    name = _get_request_param("name")
    user = store.get_user(request.authorization.username)
    return _get_permission_from_store_or_default(
        lambda: store.get_registered_model_permission(name, user.id).permission
    )


def validate_can_read_experiment():
    return _get_permission_from_experiment_id().can_read


def validate_can_update_experiment():
    return _get_permission_from_experiment_id().can_update


def validate_can_delete_experiment():
    return _get_permission_from_experiment_id().can_delete


def validate_can_manage_experiment():
    return _get_permission_from_experiment_id().can_manage


def validate_can_read_run():
    return _get_permission_from_run_id().can_read


def validate_can_update_run():
    return _get_permission_from_run_id().can_update


def validate_can_delete_run():
    return _get_permission_from_run_id().can_delete


def validate_can_manage_run():
    return _get_permission_from_run_id().can_manage


def validate_can_read_registered_model():
    return _get_permission_from_registered_model_name().can_read


def validate_can_update_registered_model():
    return _get_permission_from_registered_model_name().can_update


def validate_can_delete_registered_model():
    return _get_permission_from_registered_model_name().can_delete


def validate_can_manage_registered_model():
    return _get_permission_from_registered_model_name().can_manage


BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    GetExperiment: validate_can_read_experiment,
    GetExperimentByName: validate_can_read_experiment,
    DeleteExperiment: validate_can_delete_experiment,
    RestoreExperiment: validate_can_delete_experiment,
    UpdateExperiment: validate_can_update_experiment,
    SetExperimentTag: validate_can_update_experiment,
    # Routes for runs
    CreateRun: validate_can_update_experiment,
    GetRun: validate_can_read_run,
    DeleteRun: validate_can_delete_run,
    RestoreRun: validate_can_delete_run,
    UpdateRun: validate_can_update_run,
    LogMetric: validate_can_update_run,
    LogBatch: validate_can_update_run,
    LogModel: validate_can_update_run,
    SetTag: validate_can_update_run,
    DeleteTag: validate_can_update_run,
    LogParam: validate_can_update_run,
    GetMetricHistory: validate_can_read_run,
    ListArtifacts: validate_can_read_run,
    # Routes for model registry
    GetRegisteredModel: validate_can_read_registered_model,
    DeleteRegisteredModel: validate_can_delete_registered_model,
    UpdateRegisteredModel: validate_can_update_registered_model,
    RenameRegisteredModel: validate_can_update_registered_model,
    GetLatestVersions: validate_can_read_registered_model,
    CreateModelVersion: validate_can_update_registered_model,
    GetModelVersion: validate_can_read_registered_model,
    DeleteModelVersion: validate_can_delete_registered_model,
    UpdateModelVersion: validate_can_update_registered_model,
    TransitionModelVersionStage: validate_can_update_registered_model,
    GetModelVersionDownloadUri: validate_can_read_registered_model,
    SetRegisteredModelTag: validate_can_update_registered_model,
    DeleteRegisteredModelTag: validate_can_update_registered_model,
    SetModelVersionTag: validate_can_update_registered_model,
    DeleteModelVersionTag: validate_can_delete_registered_model,
    SetRegisteredModelAlias: validate_can_update_registered_model,
    DeleteRegisteredModelAlias: validate_can_delete_registered_model,
    GetModelVersionByAlias: validate_can_read_registered_model,
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
        (ROUTES.CREATE_EXPERIMENT_PERMISSION, "POST"): validate_can_manage_experiment,
        (ROUTES.UPDATE_EXPERIMENT_PERMISSION, "POST"): validate_can_manage_experiment,
        (ROUTES.DELETE_EXPERIMENT_PERMISSION, "POST"): validate_can_manage_experiment,
        (ROUTES.GET_REGISTERED_MODEL_PERMISSION, "GET"): validate_can_manage_registered_model,
        (ROUTES.CREATE_REGISTERED_MODEL_PERMISSION, "POST"): validate_can_manage_registered_model,
        (ROUTES.UPDATE_REGISTERED_MODEL_PERMISSION, "POST"): validate_can_manage_registered_model,
        (ROUTES.DELETE_REGISTERED_MODEL_PERMISSION, "POST"): validate_can_manage_registered_model,
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

    # admins don't need to be authorized
    user = store.get_user(username)
    if user.is_admin:
        _logger.debug(f"Admin (username={username}) authorization not required")
        return

    # authorization
    validator = BEFORE_REQUEST_VALIDATORS.get((request.path, request.method))
    if validator:
        _logger.debug(f"Calling validator: {validator.__name__}")
        if not validator():
            return make_forbidden_response()
    else:
        _logger.debug(f"No validator found for {(request.path, request.method)}")


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


def create_root_user(username, password):
    if not store.has_user(username):
        store.create_user(username, password, is_admin=True)
        _logger.info(
            f"Created root user '{username}'. "
            "It is recommended that you set a new password as soon as possible."
        )


@catch_mlflow_exception
def create_experiment_permission():
    experiment_id = _get_request_param("experiment_id")
    username = _get_request_param("username")
    permission = _get_request_param("permission")
    ep = store.create_experiment_permission(experiment_id, username, permission)
    return make_response({"experiment_permission": ep.to_json()})


@catch_mlflow_exception
def get_experiment_permission():
    experiment_id = _get_request_param("experiment_id")
    username = _get_request_param("username")
    ep = store.get_experiment_permission(experiment_id, username)
    return make_response({"experiment_permission": ep.to_json()})


@catch_mlflow_exception
def update_experiment_permission():
    experiment_id = _get_request_param("experiment_id")
    username = _get_request_param("username")
    permission = _get_request_param("permission")
    store.update_experiment_permission(experiment_id, username, permission)
    return make_response({})


@catch_mlflow_exception
def delete_experiment_permission():
    experiment_id = _get_request_param("experiment_id")
    username = _get_request_param("username")
    store.delete_experiment_permission(experiment_id, username)
    return make_response({})


@catch_mlflow_exception
def create_registered_model_permission():
    name = _get_request_param("name")
    username = _get_request_param("username")
    permission = _get_request_param("permission")
    rmp = store.create_registered_model_permission(name, username, permission)
    return make_response({"registered_model_permission": rmp.to_json()})


@catch_mlflow_exception
def get_registered_model_permission():
    name = _get_request_param("name")
    username = _get_request_param("username")
    rmp = store.get_registered_model_permission(name, username)
    return make_response({"registered_model_permission": rmp.to_json()})


@catch_mlflow_exception
def update_registered_model_permission():
    name = _get_request_param("name")
    username = _get_request_param("username")
    permission = _get_request_param("permission")
    store.update_registered_model_permission(name, username, permission)
    return make_response({})


@catch_mlflow_exception
def delete_registered_model_permission():
    name = _get_request_param("name")
    username = _get_request_param("username")
    store.delete_registered_model_permission(name, username)
    return make_response({})


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
    create_root_user(auth_config.root_username, auth_config.root_password)

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
    app.add_url_rule(
        rule=ROUTES.CREATE_REGISTERED_MODEL_PERMISSION,
        view_func=create_registered_model_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=ROUTES.GET_REGISTERED_MODEL_PERMISSION,
        view_func=get_registered_model_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=ROUTES.UPDATE_REGISTERED_MODEL_PERMISSION,
        view_func=update_registered_model_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=ROUTES.DELETE_REGISTERED_MODEL_PERMISSION,
        view_func=delete_registered_model_permission,
        methods=["POST"],
    )

    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
