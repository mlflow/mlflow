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

from flask import Flask, request, make_response, Response, flash, render_template_string

from mlflow import get_run, MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.logo import MLFLOW_LOGO
from mlflow.server.auth.permissions import get_permission, Permission, MANAGE
from mlflow.server.auth.routes import (
    HOME,
    SIGNUP,
    CREATE_USER,
    GET_USER,
    UPDATE_USER_PASSWORD,
    UPDATE_USER_ADMIN,
    DELETE_USER,
    CREATE_EXPERIMENT_PERMISSION,
    GET_EXPERIMENT_PERMISSION,
    UPDATE_EXPERIMENT_PERMISSION,
    DELETE_EXPERIMENT_PERMISSION,
    CREATE_REGISTERED_MODEL_PERMISSION,
    GET_REGISTERED_MODEL_PERMISSION,
    UPDATE_REGISTERED_MODEL_PERMISSION,
    DELETE_REGISTERED_MODEL_PERMISSION,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
    _get_request_message,
    _get_tracking_store,
    _get_model_registry_store,
    catch_mlflow_exception,
    get_endpoints,
)
from mlflow.store.entities import PagedList
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
    CreateExperiment,
    SearchExperiments,
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
    CreateRegisteredModel,
    SearchRegisteredModels,
)
from mlflow.utils.proto_json_utils import parse_dict, message_to_json
from mlflow.utils.search_utils import SearchUtils

_AUTH_CONFIG_PATH_ENV_VAR = "MLFLOW_AUTH_CONFIG_PATH"

_logger = logging.getLogger(__name__)


def _get_auth_config_path():
    return os.environ.get(
        _AUTH_CONFIG_PATH_ENV_VAR, (Path(__file__).parent / "basic_auth.ini").resolve()
    )


auth_config_path = _get_auth_config_path()
auth_config = read_auth_config(auth_config_path)
store = SqlAlchemyStore()


UNPROTECTED_ROUTES = [CREATE_USER, SIGNUP]


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
    elif request.method in ("POST", "PATCH", "DELETE"):
        args = request.json
    else:
        raise MlflowException(
            f"Unsupported HTTP method '{request.method}'",
            BAD_REQUEST,
        )

    if param not in args:
        # Special handling for run_id
        if param == "run_id":
            return _get_request_param("run_uuid")
        raise MlflowException(
            f"Missing value for required parameter '{param}'. "
            "See the API docs for more information about request parameters.",
            INVALID_PARAMETER_VALUE,
        )
    return args[param]


def _get_permission_from_store_or_default(store_permission_func: Callable[[], str]) -> Permission:
    """
    Attempts to get permission from store,
    and returns default permission if no record is found.
    """
    try:
        perm = store_permission_func()
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            perm = auth_config.default_permission
        else:
            raise
    return get_permission(perm)


def _get_permission_from_experiment_id() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    username = request.authorization.username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def _get_permission_from_experiment_name() -> Permission:
    experiment_name = _get_request_param("experiment_name")
    store_exp = _get_tracking_store().get_experiment_by_name(experiment_name)
    if store_exp is None:
        raise MlflowException(
            f"Could not find experiment with name {experiment_name}",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    username = request.authorization.username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(store_exp.experiment_id, username).permission
    )


def _get_permission_from_run_id() -> Permission:
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run_id = _get_request_param("run_id")
    run = get_run(run_id)
    experiment_id = run.info.experiment_id
    username = request.authorization.username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def _get_permission_from_registered_model_name() -> Permission:
    name = _get_request_param("name")
    username = request.authorization.username
    return _get_permission_from_store_or_default(
        lambda: store.get_registered_model_permission(name, username).permission
    )


def validate_can_read_experiment():
    return _get_permission_from_experiment_id().can_read


def validate_can_read_experiment_by_name():
    return _get_permission_from_experiment_name().can_read


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


def sender_is_admin():
    """Validate if the sender is admin"""
    username = request.authorization.username
    return store.get_user(username).is_admin


def username_is_sender():
    """Validate if the request username is the sender"""
    username = _get_request_param("username")
    sender = request.authorization.username
    return username == sender


def validate_can_read_user():
    return username_is_sender()


def validate_can_update_user_password():
    return username_is_sender()


def validate_can_update_user_admin():
    # only admins can update, but admins won't reach this validator
    return False


def validate_can_delete_user():
    # only admins can delete, but admins won't reach this validator
    return False


BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    GetExperiment: validate_can_read_experiment,
    GetExperimentByName: validate_can_read_experiment_by_name,
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
        (GET_USER, "GET"): validate_can_read_user,
        (UPDATE_USER_PASSWORD, "PATCH"): validate_can_update_user_password,
        (UPDATE_USER_ADMIN, "PATCH"): validate_can_update_user_admin,
        (DELETE_USER, "DELETE"): validate_can_delete_user,
        (GET_EXPERIMENT_PERMISSION, "GET"): validate_can_manage_experiment,
        (CREATE_EXPERIMENT_PERMISSION, "POST"): validate_can_manage_experiment,
        (UPDATE_EXPERIMENT_PERMISSION, "PATCH"): validate_can_manage_experiment,
        (DELETE_EXPERIMENT_PERMISSION, "DELETE"): validate_can_manage_experiment,
        (GET_REGISTERED_MODEL_PERMISSION, "GET"): validate_can_manage_registered_model,
        (CREATE_REGISTERED_MODEL_PERMISSION, "POST"): validate_can_manage_registered_model,
        (UPDATE_REGISTERED_MODEL_PERMISSION, "PATCH"): validate_can_manage_registered_model,
        (DELETE_REGISTERED_MODEL_PERMISSION, "DELETE"): validate_can_manage_registered_model,
    }
)


@catch_mlflow_exception
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
    if sender_is_admin():
        _logger.debug(f"Admin (username={username}) authorization not required")
        return

    # authorization
    if validator := BEFORE_REQUEST_VALIDATORS.get((request.path, request.method)):
        _logger.debug(f"Calling validator: {validator.__name__}")
        if not validator():
            return make_forbidden_response()
    else:
        _logger.debug(f"No validator found for {(request.path, request.method)}")


def set_can_manage_experiment_permission(resp: Response):
    response_message = CreateExperiment.Response()
    parse_dict(resp.json, response_message)
    experiment_id = response_message.experiment_id
    username = request.authorization.username
    store.create_experiment_permission(experiment_id, username, MANAGE.name)


def set_can_manage_registered_model_permission(resp: Response):
    response_message = CreateRegisteredModel.Response()
    parse_dict(resp.json, response_message)
    name = response_message.registered_model.name
    username = request.authorization.username
    store.create_registered_model_permission(name, username, MANAGE.name)


def filter_search_experiments(resp: Response):
    if sender_is_admin():
        return

    response_message = SearchExperiments.Response()
    parse_dict(resp.json, response_message)

    # fetch permissions
    username = request.authorization.username
    perms = store.list_experiment_permissions(username)
    can_read = {p.experiment_id: get_permission(p.permission).can_read for p in perms}
    default_can_read = get_permission(auth_config.default_permission).can_read

    # filter out unreadable
    for e in list(response_message.experiments):
        if not can_read.get(e.experiment_id, default_can_read):
            response_message.experiments.remove(e)

    # re-fetch to fill max results
    request_message = _get_request_message(SearchExperiments())
    while (
        len(response_message.experiments) < request_message.max_results
        and response_message.next_page_token != ""
    ):
        refetched: PagedList[Experiment] = _get_tracking_store().search_experiments(
            view_type=request_message.view_type,
            max_results=request_message.max_results,
            order_by=request_message.order_by,
            filter_string=request_message.filter,
            page_token=response_message.next_page_token,
        )
        refetched = refetched[: request_message.max_results - len(response_message.experiments)]
        if len(refetched) == 0:
            response_message.next_page_token = ""
            break

        refetched_readable_proto = [
            e.to_proto() for e in refetched if can_read.get(e.experiment_id, default_can_read)
        ]
        response_message.experiments.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)


def filter_search_registered_models(resp: Response):
    if sender_is_admin():
        return

    response_message = SearchRegisteredModels.Response()
    parse_dict(resp.json, response_message)

    # fetch permissions
    username = request.authorization.username
    perms = store.list_registered_model_permissions(username)
    can_read = {p.name: get_permission(p.permission).can_read for p in perms}
    default_can_read = get_permission(auth_config.default_permission).can_read

    # filter out unreadable
    for rm in list(response_message.registered_models):
        if not can_read.get(rm.name, default_can_read):
            response_message.registered_models.remove(rm)

    # re-fetch to fill max results
    request_message = _get_request_message(SearchRegisteredModels())
    while (
        len(response_message.registered_models) < request_message.max_results
        and response_message.next_page_token != ""
    ):
        refetched: PagedList[
            RegisteredModel
        ] = _get_model_registry_store().search_registered_models(
            filter_string=request_message.filter,
            max_results=request_message.max_results,
            order_by=request_message.order_by,
            page_token=response_message.next_page_token,
        )
        refetched = refetched[
            : request_message.max_results - len(response_message.registered_models)
        ]
        if len(refetched) == 0:
            response_message.next_page_token = ""
            break

        refetched_readable_proto = [
            rm.to_proto() for rm in refetched if can_read.get(rm.name, default_can_read)
        ]
        response_message.registered_models.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)


AFTER_REQUEST_PATH_HANDLERS = {
    CreateExperiment: set_can_manage_experiment_permission,
    CreateRegisteredModel: set_can_manage_registered_model_permission,
    SearchExperiments: filter_search_experiments,
    SearchRegisteredModels: filter_search_registered_models,
}


def get_after_request_handler(request_class):
    return AFTER_REQUEST_PATH_HANDLERS.get(request_class)


AFTER_REQUEST_HANDLERS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_after_request_handler)
    for method in methods
}


@catch_mlflow_exception
def _after_request(resp: Response):
    _logger.debug(f"after_request: {request.method} {request.path}")
    if 400 <= resp.status_code < 600:
        return resp

    if handler := AFTER_REQUEST_HANDLERS.get((request.path, request.method)):
        _logger.debug(f"Calling after request handler: {handler.__name__}")
        handler(resp)
    return resp


def create_admin_user(username, password):
    if not store.has_user(username):
        store.create_user(username, password, is_admin=True)
        _logger.info(
            f"Created admin user '{username}'. "
            "It is recommended that you set a new password as soon as possible "
            f"on {UPDATE_USER_PASSWORD}."
        )


def alert(href: str):
    return render_template_string(
        r"""
<script type = "text/javascript">
{% with messages = get_flashed_messages() %}
  {% if messages %}
    {% for message in messages %}
      alert("{{ message }}");
    {% endfor %}
  {% endif %}
{% endwith %}
      window.location.href = "{{ href }}";
</script>
""",
        href=href,
    )


def signup():
    return render_template_string(
        r"""
<style>
  form {
    background-color: #F5F5F5;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
    padding: 20px;
    max-width: 400px;
    margin: 0 auto;
    font-family: Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }

  input[type=text], input[type=password] {
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
    box-sizing: border-box;
  }
  input[type=submit] {
    background-color: rgb(34, 114, 180);
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
  }

  input[type=submit]:hover {
    background-color: rgb(14, 83, 139);
  }

  .logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
  }

  .logo {
    max-width: 150px;
    margin-right: 10px;
  }
</style>

<form action="{{ users_route }}" method="post">
  <div class="logo-container">
    {% autoescape false %}
    {{ mlflow_logo }}
    {% endautoescape %}
  </div>
  <label for="username">Username:</label>
  <br>
  <input type="text" id="username" name="username">
  <br>
  <label for="password">Password:</label>
  <br>
  <input type="password" id="password" name="password">
  <br>
  <br>
  <input type="submit" value="Sign up">
</form>
""",
        mlflow_logo=MLFLOW_LOGO,
        users_route=CREATE_USER,
    )


@catch_mlflow_exception
def create_user():
    content_type = request.headers.get("Content-Type")
    if content_type == "application/x-www-form-urlencoded":
        username = request.form["username"]
        password = request.form["password"]

        if store.has_user(username):
            flash(f"Username has already been taken: {username}")
            return alert(href=SIGNUP)

        store.create_user(username, password)
        flash(f"Successfully signed up user: {username}")
        return alert(href=HOME)
    elif content_type == "application/json":
        username = _get_request_param("username")
        password = _get_request_param("password")

        user = store.create_user(username, password)
        return make_response({"user": user.to_json()})
    else:
        return make_response(f"Invalid content type: '{content_type}'", 400)


@catch_mlflow_exception
def get_user():
    username = _get_request_param("username")
    user = store.get_user(username)
    return make_response({"user": user.to_json()})


@catch_mlflow_exception
def update_user_password():
    username = _get_request_param("username")
    password = _get_request_param("password")
    store.update_user(username, password=password)
    return make_response({})


@catch_mlflow_exception
def update_user_admin():
    username = _get_request_param("username")
    is_admin_str = _get_request_param("is_admin").lower()
    if is_admin_str == "true":
        is_admin = True
    elif is_admin_str == "false":
        is_admin = False
    else:
        raise MlflowException(
            f"Invalid parameter 'is_admin': '{is_admin_str}', "
            "must be either 'true' or 'false' (case insensitive).",
            INVALID_PARAMETER_VALUE,
        )
    store.update_user(username, is_admin=is_admin)
    return make_response({})


@catch_mlflow_exception
def delete_user():
    username = _get_request_param("username")
    store.delete_user(username)
    return make_response({})


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
    create_admin_user(auth_config.admin_username, auth_config.admin_password)

    app.add_url_rule(
        rule=SIGNUP,
        view_func=signup,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=CREATE_USER,
        view_func=create_user,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=GET_USER,
        view_func=get_user,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=UPDATE_USER_PASSWORD,
        view_func=update_user_password,
        methods=["PATCH"],
    )
    app.add_url_rule(
        rule=UPDATE_USER_ADMIN,
        view_func=update_user_admin,
        methods=["PATCH"],
    )
    app.add_url_rule(
        rule=DELETE_USER,
        view_func=delete_user,
        methods=["DELETE"],
    )
    app.add_url_rule(
        rule=CREATE_EXPERIMENT_PERMISSION,
        view_func=create_experiment_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=GET_EXPERIMENT_PERMISSION,
        view_func=get_experiment_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=UPDATE_EXPERIMENT_PERMISSION,
        view_func=update_experiment_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        rule=DELETE_EXPERIMENT_PERMISSION,
        view_func=delete_experiment_permission,
        methods=["DELETE"],
    )
    app.add_url_rule(
        rule=CREATE_REGISTERED_MODEL_PERMISSION,
        view_func=create_registered_model_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=GET_REGISTERED_MODEL_PERMISSION,
        view_func=get_registered_model_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=UPDATE_REGISTERED_MODEL_PERMISSION,
        view_func=update_registered_model_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        rule=DELETE_REGISTERED_MODEL_PERMISSION,
        view_func=delete_registered_model_permission,
        methods=["DELETE"],
    )

    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
