"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging

from flask import Flask, request, make_response, Response

from mlflow import get_run
from mlflow.server import app
from mlflow.server.auth import store
from mlflow.server.auth.config import app_config
from mlflow.server.auth.permissions import Permission, get_permission
from mlflow.server.handlers import (
    _get_tracking_store,
    _get_request_message,
    get_endpoints,
    message_to_json,
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


_logger = logging.getLogger(__name__)


class ROUTES:
    CREATE_EXPERIMENT_PERMISSION = "/mlflow/experiments/permissions/create"
    READ_EXPERIMENT_PERMISSION = "/mlflow/experiments/permissions/read"
    UPDATE_EXPERIMENT_PERMISSION = "/mlflow/experiments/permissions/update"
    DELETE_EXPERIMENT_PERMISSION = "/mlflow/experiments/permissions/delete"
    USERS = "/users"
    SIGNUP = "/signup"


PROTECTED_ROUTES = []


def is_protected_route(path: str) -> bool:
    if path.startswith("/static-files"):
        return True
    return path in PROTECTED_ROUTES


def make_forbidden_response() -> Response:
    res = make_response("Permission denied")
    res.status_code = 403
    return res


def make_invalid_login_response() -> Response:
    res = make_response("Invalid username or password")
    res.status_code = 403
    return res


def make_basic_auth_response() -> Response:
    res = make_response()
    res.status_code = 401
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res


def _get_request_param(param: str) -> str:
    if request.method == "GET":
        return request.args[param]
    elif request.method == "POST":
        return request.json[param]
    else:
        raise NotImplementedError()


def get_experiment_id() -> str:
    return _get_request_param("experiment_id")


def get_run_id() -> str:
    return _get_request_param("run_id")


def get_permission_from_experiment_id() -> Permission:
    experiment_id = get_experiment_id()
    user = store.get_user(request.authorization.username)
    perm = store.get_experiment_permission(experiment_id, user.id)
    perm = perm.permission if perm else app_config.default_permission
    return get_permission(perm)


def get_permission_from_run_id() -> Permission:
    run_id = get_run_id()
    user = store.get_user(request.authorization.username)
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run = get_run(run_id)
    experiment_id = run.info.experiment_id
    perm = store.get_experiment_permission(experiment_id, user.id)
    perm = perm.permission if perm else app_config.default_permission
    return get_permission(perm)


def validate_can_read_experiment():
    return get_permission_from_experiment_id().can_read


def validate_can_update_experiment():
    return get_permission_from_experiment_id().can_update


def validate_can_delete_experiment():
    return get_permission_from_experiment_id().can_delete


def validate_can_manage_experiment():
    return get_permission_from_experiment_id().can_manage


def validate_can_read_run():
    return get_permission_from_run_id().can_read


def validate_can_update_run():
    return get_permission_from_run_id().can_update


def validate_can_delete_run():
    return get_permission_from_run_id().can_delete


def validate_can_manage_run():
    return get_permission_from_run_id().can_manage


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
    for http_path, handler, method in get_endpoints(get_before_request_handler)
}
BEFORE_REQUEST_VALIDATORS.update(
    {
        (ROUTES.READ_EXPERIMENT_PERMISSION, "GET"): validate_can_manage_experiment,
        (ROUTES.CREATE_EXPERIMENT_PERMISSION, "PUT"): validate_can_manage_experiment,
        (ROUTES.UPDATE_EXPERIMENT_PERMISSION, "POST"): validate_can_manage_experiment,
        (ROUTES.DELETE_EXPERIMENT_PERMISSION, "DELETE"): validate_can_manage_experiment,
    }
)


def _before_request():
    # TODO: Implement authentication
    # TODO: Implement authorization
    # TODO: remove
    _logger.info("before_request: %s %s %s", request.method, request.path, request.authorization)
    if not is_protected_route(request.path):
        return

    if request.authorization is None:
        return make_basic_auth_response()

    username = request.authorization.username
    password = request.authorization.password
    if not store.authenticate_user(username, password):
        return make_invalid_login_response()

    # authorization
    validator = BEFORE_REQUEST_VALIDATORS.get((request.path, request.method))
    if validator:
        # TODO: remove
        _logger.info(f"Calling validator: {validator.__name__}")
        if not validator():
            return make_forbidden_response()


def _after_request(resp):
    # TODO: Implement post-request logic
    return resp


def signup():
    # TODO: add css
    return """
<form action="/users" method="post">
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
"""


def _enable_auth(app: Flask):
    """
    Enables authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    """
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )
    # TODO: remove
    _logger.info("Database URI: %s", app_config.database_uri)
    app.config["SQLALCHEMY_DATABASE_URI"] = app_config.database_uri

    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
