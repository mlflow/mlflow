"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

from __future__ import annotations

import functools
import importlib
import logging
import re
from typing import Any, Callable

import sqlalchemy
from flask import (
    Flask,
    Request,
    Response,
    flash,
    jsonify,
    make_response,
    render_template_string,
    request,
)
from werkzeug.datastructures import Authorization

from mlflow import MlflowException
from mlflow.entities import Experiment
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.model_registry import RegisteredModel
from mlflow.environment_variables import (
    _MLFLOW_SGI_NAME,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    RenameRegisteredModel,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.service_pb2 import (
    CreateExperiment,
    # Routes for logged models
    CreateLoggedModel,
    CreateRun,
    DeleteExperiment,
    DeleteExperimentTag,
    DeleteLoggedModel,
    DeleteLoggedModelTag,
    DeleteRun,
    DeleteScorer,
    DeleteTag,
    FinalizeLoggedModel,
    GetExperiment,
    GetExperimentByName,
    GetLoggedModel,
    GetMetricHistory,
    GetRun,
    GetScorer,
    ListArtifacts,
    ListScorers,
    ListScorerVersions,
    LogBatch,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogParam,
    RegisterScorer,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SearchLoggedModels,
    SetExperimentTag,
    SetLoggedModelTags,
    SetTag,
    UpdateExperiment,
    UpdateRun,
)
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.logo import MLFLOW_LOGO
from mlflow.server.auth.permissions import MANAGE, Permission, get_permission
from mlflow.server.auth.routes import (
    CREATE_EXPERIMENT_PERMISSION,
    CREATE_PROMPTLAB_RUN,
    CREATE_REGISTERED_MODEL_PERMISSION,
    CREATE_SCORER_PERMISSION,
    CREATE_USER,
    CREATE_USER_UI,
    DELETE_EXPERIMENT_PERMISSION,
    DELETE_REGISTERED_MODEL_PERMISSION,
    DELETE_SCORER_PERMISSION,
    DELETE_USER,
    GATEWAY_PROXY,
    GET_ARTIFACT,
    GET_EXPERIMENT_PERMISSION,
    GET_METRIC_HISTORY_BULK,
    GET_METRIC_HISTORY_BULK_INTERVAL,
    GET_MODEL_VERSION_ARTIFACT,
    GET_REGISTERED_MODEL_PERMISSION,
    GET_SCORER_PERMISSION,
    GET_TRACE_ARTIFACT,
    GET_USER,
    HOME,
    SEARCH_DATASETS,
    SIGNUP,
    UPDATE_EXPERIMENT_PERMISSION,
    UPDATE_REGISTERED_MODEL_PERMISSION,
    UPDATE_SCORER_PERMISSION,
    UPDATE_USER_ADMIN,
    UPDATE_USER_PASSWORD,
    UPLOAD_ARTIFACT,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.handlers import (
    _get_model_registry_store,
    _get_request_message,
    _get_tracking_store,
    catch_mlflow_exception,
    get_endpoints,
)
from mlflow.store.entities import PagedList
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX
from mlflow.utils.search_utils import SearchUtils

try:
    from flask_wtf.csrf import CSRFProtect
except ImportError as e:
    raise ImportError(
        "The MLflow basic auth app requires the Flask-WTF package to perform CSRF "
        "validation. Please run `pip install mlflow[auth]` to install it."
    ) from e

_logger = logging.getLogger(__name__)

auth_config = read_auth_config()
store = SqlAlchemyStore()


def is_unprotected_route(path: str) -> bool:
    return path.startswith(("/static", "/favicon.ico", "/health"))


def make_basic_auth_response() -> Response:
    res = make_response(
        "You are not authenticated. Please see "
        "https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow "
        "on how to authenticate."
    )
    res.status_code = 401
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res


def make_forbidden_response() -> Response:
    res = make_response("Permission denied")
    res.status_code = 403
    return res


def _get_request_param(param: str) -> str:
    if request.method == "GET":
        args = request.args
    elif request.method in ("POST", "PATCH"):
        args = request.json
    elif request.method == "DELETE":
        args = request.json if request.is_json else request.args
    else:
        raise MlflowException(
            f"Unsupported HTTP method '{request.method}'",
            BAD_REQUEST,
        )

    args = args | (request.view_args or {})
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
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


_EXPERIMENT_ID_PATTERN = re.compile(r"^(\d+)/")


def _get_experiment_id_from_view_args():
    if artifact_path := request.view_args.get("artifact_path"):
        if m := _EXPERIMENT_ID_PATTERN.match(artifact_path):
            return m.group(1)
    return None


def _get_permission_from_experiment_id_artifact_proxy() -> Permission:
    if experiment_id := _get_experiment_id_from_view_args():
        username = authenticate_request().username
        return _get_permission_from_store_or_default(
            lambda: store.get_experiment_permission(experiment_id, username).permission
        )
    return get_permission(auth_config.default_permission)


def _get_permission_from_experiment_name() -> Permission:
    experiment_name = _get_request_param("experiment_name")
    store_exp = _get_tracking_store().get_experiment_by_name(experiment_name)
    if store_exp is None:
        raise MlflowException(
            f"Could not find experiment with name {experiment_name}",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(store_exp.experiment_id, username).permission
    )


def _get_permission_from_run_id() -> Permission:
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run_id = _get_request_param("run_id")
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def _get_permission_from_model_id() -> Permission:
    # logged model permissions inherit from parent resource (experiment)
    model_id = _get_request_param("model_id")
    model = _get_tracking_store().get_logged_model(model_id)
    experiment_id = model.experiment_id
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def _get_permission_from_registered_model_name() -> Permission:
    name = _get_request_param("name")
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_registered_model_permission(name, username).permission
    )


def _get_permission_from_scorer_name() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    name = _get_request_param("name")
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_scorer_permission(experiment_id, name, username).permission
    )


def _get_permission_from_scorer_permission_request() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    scorer_name = _get_request_param("scorer_name")
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_scorer_permission(experiment_id, scorer_name, username).permission
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


def validate_can_read_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_read


def validate_can_update_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_update


def validate_can_delete_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_manage


# Runs
def validate_can_read_run():
    return _get_permission_from_run_id().can_read


def validate_can_update_run():
    return _get_permission_from_run_id().can_update


def validate_can_delete_run():
    return _get_permission_from_run_id().can_delete


def validate_can_manage_run():
    return _get_permission_from_run_id().can_manage


# Logged models
def validate_can_read_logged_model():
    return _get_permission_from_model_id().can_read


def validate_can_update_logged_model():
    return _get_permission_from_model_id().can_update


def validate_can_delete_logged_model():
    return _get_permission_from_model_id().can_delete


def validate_can_manage_logged_model():
    return _get_permission_from_model_id().can_manage


# Registered models
def validate_can_read_registered_model():
    return _get_permission_from_registered_model_name().can_read


def validate_can_update_registered_model():
    return _get_permission_from_registered_model_name().can_update


def validate_can_delete_registered_model():
    return _get_permission_from_registered_model_name().can_delete


def validate_can_manage_registered_model():
    return _get_permission_from_registered_model_name().can_manage


# Scorers
def validate_can_read_scorer():
    return _get_permission_from_scorer_name().can_read


def validate_can_update_scorer():
    return _get_permission_from_scorer_name().can_update


def validate_can_delete_scorer():
    return _get_permission_from_scorer_name().can_delete


def validate_can_manage_scorer():
    return _get_permission_from_scorer_name().can_manage


def validate_can_manage_scorer_permission():
    return _get_permission_from_scorer_permission_request().can_manage


def sender_is_admin():
    """Validate if the sender is admin"""
    username = authenticate_request().username
    return store.get_user(username).is_admin


def filter_experiment_ids(experiment_ids: list[str]) -> list[str]:
    """
    Filter experiment IDs to only include those the user has read access to.

    Args:
        experiment_ids: List of experiment IDs to filter

    Returns:
        Filtered list of experiment IDs the user can read
    """
    if not auth_config:
        return experiment_ids

    try:
        if sender_is_admin():
            return experiment_ids

        username = authenticate_request().username
        perms = store.list_experiment_permissions(username)
        can_read = {p.experiment_id: get_permission(p.permission).can_read for p in perms}
        default_can_read = get_permission(auth_config.default_permission).can_read

        return [exp_id for exp_id in experiment_ids if can_read.get(exp_id, default_can_read)]
    except (RuntimeError, AttributeError):
        # Auth system not fully initialized, skip filtering
        return experiment_ids


def username_is_sender():
    """Validate if the request username is the sender"""
    username = _get_request_param("username")
    sender = authenticate_request().username
    return username == sender


def validate_can_read_user():
    return username_is_sender()


def validate_can_create_user():
    # only admins can create user, but admins won't reach this validator
    return False


def validate_can_update_user_password():
    return username_is_sender()


def validate_can_update_user_admin():
    # only admins can update, but admins won't reach this validator
    return False


def validate_can_delete_user():
    # only admins can delete, but admins won't reach this validator
    return False


def _get_permission_from_run_id_or_uuid() -> Permission:
    """
    Get permission for Flask routes that use either run_id or run_uuid parameter.
    """
    run_id = request.args.get("run_id") or request.args.get("run_uuid")
    if not run_id:
        raise MlflowException(
            "Request must specify run_id or run_uuid parameter",
            INVALID_PARAMETER_VALUE,
        )
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def validate_can_read_run_artifact():
    """Checks READ permission on run artifacts."""
    return _get_permission_from_run_id_or_uuid().can_read


def validate_can_update_run_artifact():
    """Checks UPDATE permission on run artifacts."""
    return _get_permission_from_run_id_or_uuid().can_update


def _get_permission_from_model_version() -> Permission:
    """
    Get permission for model version artifacts.
    Model versions inherit permissions from their registered model.
    """
    name = request.args.get("name")
    if not name:
        raise MlflowException(
            "Request must specify name parameter",
            INVALID_PARAMETER_VALUE,
        )
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_registered_model_permission(name, username).permission
    )


def validate_can_read_model_version_artifact():
    """Checks READ permission on model version artifacts."""
    return _get_permission_from_model_version().can_read


def _get_permission_from_trace_request_id() -> Permission:
    """
    Get permission for trace artifacts.
    Traces inherit permissions from their parent run/experiment.
    """
    request_id = request.args.get("request_id")
    if not request_id:
        raise MlflowException(
            "Request must specify request_id parameter",
            INVALID_PARAMETER_VALUE,
        )
    # Get the trace to find its experiment
    trace = _get_tracking_store().get_trace_info(request_id)
    experiment_id = trace.experiment_id
    username = authenticate_request().username
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def validate_can_read_trace_artifact():
    """Checks READ permission on trace artifacts."""
    return _get_permission_from_trace_request_id().can_read


def validate_can_read_metric_history_bulk(run_ids=None):
    """Checks READ permission on all requested runs.

    Args:
        run_ids: Optional list of run IDs to validate. If not provided,
            extracts 'run_id' from request args (for GetMetricHistoryBulk endpoint).
    """
    if run_ids is None:
        run_ids = request.args.to_dict(flat=False).get("run_id", [])
    if not run_ids:
        raise MlflowException(
            "GetMetricHistoryBulk request must specify at least one run_id.",
            INVALID_PARAMETER_VALUE,
        )

    username = authenticate_request().username
    tracking_store = _get_tracking_store()

    for run_id in run_ids:
        run = tracking_store.get_run(run_id)
        experiment_id = run.info.experiment_id
        permission = _get_permission_from_store_or_default(
            lambda eid=experiment_id: store.get_experiment_permission(eid, username).permission
        )
        if not permission.can_read:
            return False

    return True


def validate_can_read_metric_history_bulk_interval():
    """Checks READ permission on all requested runs for the bulk interval endpoint."""
    run_ids = request.args.to_dict(flat=False).get("run_ids", [])
    if not run_ids:
        raise MlflowException(
            "GetMetricHistoryBulkInterval request must specify at least one run_id.",
            INVALID_PARAMETER_VALUE,
        )
    return validate_can_read_metric_history_bulk(run_ids)


def validate_can_search_datasets():
    """Checks READ permission on all requested experiments."""
    if request.method == "POST":
        data = request.json
        experiment_ids = data.get("experiment_ids", [])
    else:
        experiment_ids = request.args.getlist("experiment_ids")

    if not experiment_ids:
        raise MlflowException(
            "SearchDatasets request must specify at least one experiment_id.",
            INVALID_PARAMETER_VALUE,
        )

    username = authenticate_request().username

    # Check permission for each experiment
    for experiment_id in experiment_ids:
        permission = _get_permission_from_store_or_default(
            lambda eid=experiment_id: store.get_experiment_permission(eid, username).permission
        )
        if not permission.can_read:
            return False

    return True


def validate_can_create_promptlab_run():
    """Checks UPDATE permission on the experiment."""
    data = request.json
    experiment_id = data.get("experiment_id")
    if not experiment_id:
        raise MlflowException(
            "CreatePromptlabRun request must specify experiment_id.",
            INVALID_PARAMETER_VALUE,
        )

    username = authenticate_request().username
    permission = _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )
    return permission.can_update


def validate_gateway_proxy():
    """
    Allows gateway proxy requests without permission checks.
    This endpoint proxies to external services that handle their own authorization.
    """
    return True


BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    GetExperiment: validate_can_read_experiment,
    GetExperimentByName: validate_can_read_experiment_by_name,
    DeleteExperiment: validate_can_delete_experiment,
    RestoreExperiment: validate_can_delete_experiment,
    UpdateExperiment: validate_can_update_experiment,
    SetExperimentTag: validate_can_update_experiment,
    DeleteExperimentTag: validate_can_update_experiment,
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
    # Routes for scorers
    RegisterScorer: validate_can_update_experiment,
    ListScorers: validate_can_read_experiment,
    GetScorer: validate_can_read_scorer,
    DeleteScorer: validate_can_delete_scorer,
    ListScorerVersions: validate_can_read_scorer,
}


def get_before_request_handler(request_class):
    return BEFORE_REQUEST_HANDLERS.get(request_class)


def _re_compile_path(path: str) -> re.Pattern:
    """
    Convert a path with angle brackets to a regex pattern. For example,
    "/api/2.0/experiments/<experiment_id>" becomes "/api/2.0/experiments/([^/]+)".
    """
    return re.compile(re.sub(r"<([^>]+)>", r"([^/]+)", path))


BEFORE_REQUEST_VALIDATORS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_before_request_handler)
    for method in methods
}

# Auth-related routes
BEFORE_REQUEST_VALIDATORS.update(
    {
        (SIGNUP, "GET"): validate_can_create_user,
        (GET_USER, "GET"): validate_can_read_user,
        (CREATE_USER, "POST"): validate_can_create_user,
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
        (GET_SCORER_PERMISSION, "GET"): validate_can_manage_scorer_permission,
        (CREATE_SCORER_PERMISSION, "POST"): validate_can_manage_scorer_permission,
        (UPDATE_SCORER_PERMISSION, "PATCH"): validate_can_manage_scorer_permission,
        (DELETE_SCORER_PERMISSION, "DELETE"): validate_can_manage_scorer_permission,
    }
)

# Flask routes (no proto mapping)
BEFORE_REQUEST_VALIDATORS.update(
    {
        (GET_ARTIFACT, "GET"): validate_can_read_run_artifact,
        (UPLOAD_ARTIFACT, "POST"): validate_can_update_run_artifact,
        (GET_MODEL_VERSION_ARTIFACT, "GET"): validate_can_read_model_version_artifact,
        (GET_TRACE_ARTIFACT, "GET"): validate_can_read_trace_artifact,
        (GET_METRIC_HISTORY_BULK, "GET"): validate_can_read_metric_history_bulk,
        (GET_METRIC_HISTORY_BULK_INTERVAL, "GET"): validate_can_read_metric_history_bulk_interval,
        (SEARCH_DATASETS, "POST"): validate_can_search_datasets,
        (CREATE_PROMPTLAB_RUN, "POST"): validate_can_create_promptlab_run,
        (GATEWAY_PROXY, "GET"): validate_gateway_proxy,
        (GATEWAY_PROXY, "POST"): validate_gateway_proxy,
    }
)


LOGGED_MODEL_BEFORE_REQUEST_HANDLERS = {
    CreateLoggedModel: validate_can_update_experiment,
    GetLoggedModel: validate_can_read_logged_model,
    DeleteLoggedModel: validate_can_delete_logged_model,
    FinalizeLoggedModel: validate_can_update_logged_model,
    DeleteLoggedModelTag: validate_can_delete_logged_model,
    SetLoggedModelTags: validate_can_update_logged_model,
    LogLoggedModelParamsRequest: validate_can_update_logged_model,
}


def get_logged_model_before_request_handler(request_class):
    return LOGGED_MODEL_BEFORE_REQUEST_HANDLERS.get(request_class)


LOGGED_MODEL_BEFORE_REQUEST_VALIDATORS = {
    # Paths for logged models contains path parameters (e.g. /mlflow/logged-models/<model_id>)
    (_re_compile_path(http_path), method): handler
    for http_path, handler, methods in get_endpoints(get_logged_model_before_request_handler)
    for method in methods
}


def _is_proxy_artifact_path(path: str) -> bool:
    return path.startswith(f"{_REST_API_PATH_PREFIX}/mlflow-artifacts/artifacts/")


def _get_proxy_artifact_validator(
    method: str, view_args: dict[str, Any] | None
) -> Callable[[], bool] | None:
    if view_args is None:
        return validate_can_read_experiment_artifact_proxy  # List

    return {
        "GET": validate_can_read_experiment_artifact_proxy,  # Download
        "PUT": validate_can_update_experiment_artifact_proxy,  # Upload
        "DELETE": validate_can_delete_experiment_artifact_proxy,  # Delete
    }.get(method)


def authenticate_request() -> Authorization | Response:
    """Use configured authorization function to get request authorization."""
    auth_func = get_auth_func(auth_config.authorization_function)
    return auth_func()


@functools.lru_cache(maxsize=None)
def get_auth_func(authorization_function: str) -> Callable[[], Authorization | Response]:
    """
    Import and return the specified authorization function.

    Args:
        authorization_function: A string of the form "module.submodule:auth_func"
    """
    mod_name, fn_name = authorization_function.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, fn_name)


def authenticate_request_basic_auth() -> Authorization | Response:
    """Authenticate the request using basic auth."""
    if request.authorization is None:
        return make_basic_auth_response()

    username = request.authorization.username
    password = request.authorization.password
    if store.authenticate_user(username, password):
        return request.authorization
    else:
        # let user attempt login again
        return make_basic_auth_response()


def _find_validator(req: Request) -> Callable[[], bool] | None:
    """
    Finds the validator matching the request path and method.
    """
    if "/mlflow/logged-models" in req.path:
        # logged model routes are not registered in the app
        # so we need to check them manually
        return next(
            (
                v
                for (pat, method), v in LOGGED_MODEL_BEFORE_REQUEST_VALIDATORS.items()
                if pat.fullmatch(req.path) and method == req.method
            ),
            None,
        )
    else:
        return BEFORE_REQUEST_VALIDATORS.get((req.path, req.method))


@catch_mlflow_exception
def _before_request():
    if is_unprotected_route(request.path):
        return

    authorization = authenticate_request()
    if isinstance(authorization, Response):
        return authorization
    elif not isinstance(authorization, Authorization):
        raise MlflowException(
            f"Unsupported result type from {auth_config.authorization_function}: "
            f"'{type(authorization).__name__}'",
            INTERNAL_ERROR,
        )

    # admins don't need to be authorized
    if sender_is_admin():
        return

    # authorization
    if validator := _find_validator(request):
        if not validator():
            return make_forbidden_response()
    elif _is_proxy_artifact_path(request.path):
        if validator := _get_proxy_artifact_validator(request.method, request.view_args):
            if not validator():
                return make_forbidden_response()


def set_can_manage_experiment_permission(resp: Response):
    response_message = CreateExperiment.Response()
    parse_dict(resp.json, response_message)
    experiment_id = response_message.experiment_id
    username = authenticate_request().username
    store.create_experiment_permission(experiment_id, username, MANAGE.name)


def set_can_manage_registered_model_permission(resp: Response):
    response_message = CreateRegisteredModel.Response()
    parse_dict(resp.json, response_message)
    name = response_message.registered_model.name
    username = authenticate_request().username
    store.create_registered_model_permission(name, username, MANAGE.name)


def delete_can_manage_registered_model_permission(resp: Response):
    """
    Delete registered model permission when the model is deleted.

    We need to do this because the primary key of the registered model is the name,
    unlike the experiment where the primary key is experiment_id (UUID). Therefore,
    we have to delete the permission record when the model is deleted otherwise it
    conflicts with the new model registered with the same name.
    """
    # Get model name from request context because it's not available in the response
    name = request.get_json(force=True, silent=True)["name"]
    username = authenticate_request().username
    store.delete_registered_model_permission(name, username)


def filter_search_experiments(resp: Response):
    if sender_is_admin():
        return

    response_message = SearchExperiments.Response()
    parse_dict(resp.json, response_message)

    # fetch permissions
    username = authenticate_request().username
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


def filter_search_logged_models(resp: Response) -> None:
    """
    Filter out unreadable logged models from the search results.
    """
    from mlflow.utils.search_utils import SearchLoggedModelsPaginationToken as Token

    if sender_is_admin():
        return

    response_proto = SearchLoggedModels.Response()
    parse_dict(resp.json, response_proto)

    # fetch permissions
    username = authenticate_request().username
    perms = store.list_experiment_permissions(username)
    can_read = {p.experiment_id: get_permission(p.permission).can_read for p in perms}
    default_can_read = get_permission(auth_config.default_permission).can_read

    # Remove unreadable models
    for m in list(response_proto.models):
        if not can_read.get(m.info.experiment_id, default_can_read):
            response_proto.models.remove(m)

    request_proto = _get_request_message(SearchLoggedModels())
    max_results = request_proto.max_results
    # These parameters won't change in the loop
    params = {
        "experiment_ids": list(request_proto.experiment_ids),
        "filter_string": request_proto.filter or None,
        "order_by": (
            [
                {
                    "field_name": ob.field_name,
                    "ascending": ob.ascending,
                    "dataset_name": ob.dataset_name,
                    "dataset_digest": ob.dataset_digest,
                }
                for ob in request_proto.order_by
            ]
            if request_proto.order_by
            else None
        ),
    }
    next_page_token = response_proto.next_page_token or None
    tracking_store = _get_tracking_store()
    while len(response_proto.models) < max_results and next_page_token is not None:
        batch: PagedList[LoggedModel] = tracking_store.search_logged_models(
            max_results=max_results, page_token=next_page_token, **params
        )
        is_last_page = batch.token is None
        offset = Token.decode(next_page_token).offset if next_page_token else 0
        last_index = len(batch) - 1
        for index, model in enumerate(batch):
            if not can_read.get(model.experiment_id, default_can_read):
                continue
            response_proto.models.append(model.to_proto())
            if len(response_proto.models) >= max_results:
                next_page_token = (
                    None
                    if is_last_page and index == last_index
                    else Token(offset=offset + index + 1, **params).encode()
                )
                break
        else:
            # If we reach here, it means we have not reached the max results.
            next_page_token = (
                None if is_last_page else Token(offset=offset + max_results, **params).encode()
            )

    if next_page_token:
        response_proto.next_page_token = next_page_token
    resp.data = message_to_json(response_proto)


def filter_search_registered_models(resp: Response):
    if sender_is_admin():
        return

    response_message = SearchRegisteredModels.Response()
    parse_dict(resp.json, response_message)

    # fetch permissions
    username = authenticate_request().username
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
        refetched: PagedList[RegisteredModel] = (
            _get_model_registry_store().search_registered_models(
                filter_string=request_message.filter,
                max_results=request_message.max_results,
                order_by=request_message.order_by,
                page_token=response_message.next_page_token,
            )
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


def rename_registered_model_permission(resp: Response):
    """
    A model registry can be assigned to multiple users with different permissions.

    Changing the model registry name must be propagated to all users.
    """
    # get registry model name before update
    data = request.get_json(force=True, silent=True)
    store.rename_registered_model_permissions(data.get("name"), data.get("new_name"))


def set_can_manage_scorer_permission(resp: Response):
    response_message = RegisterScorer.Response()
    parse_dict(resp.json, response_message)
    experiment_id = response_message.experiment_id
    name = response_message.name
    username = authenticate_request().username
    store.create_scorer_permission(experiment_id, name, username, MANAGE.name)


def delete_scorer_permissions_cascade(resp: Response):
    data = request.get_json(force=True, silent=True)
    experiment_id = data.get("experiment_id")
    name = data.get("name")
    if experiment_id and name:
        store.delete_scorer_permissions_for_scorer(experiment_id, name)


AFTER_REQUEST_PATH_HANDLERS = {
    CreateExperiment: set_can_manage_experiment_permission,
    CreateRegisteredModel: set_can_manage_registered_model_permission,
    DeleteRegisteredModel: delete_can_manage_registered_model_permission,
    SearchExperiments: filter_search_experiments,
    SearchLoggedModels: filter_search_logged_models,
    SearchRegisteredModels: filter_search_registered_models,
    RenameRegisteredModel: rename_registered_model_permission,
    RegisterScorer: set_can_manage_scorer_permission,
    DeleteScorer: delete_scorer_permissions_cascade,
}


def get_after_request_handler(request_class):
    return AFTER_REQUEST_PATH_HANDLERS.get(request_class)


AFTER_REQUEST_HANDLERS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_after_request_handler)
    for method in methods
    if handler is not None and "/graphql" not in http_path
}


@catch_mlflow_exception
def _after_request(resp: Response):
    if 400 <= resp.status_code < 600:
        return resp

    if handler := AFTER_REQUEST_HANDLERS.get((request.path, request.method)):
        handler(resp)
    return resp


def create_admin_user(username, password):
    if not store.has_user(username):
        try:
            store.create_user(username, password, is_admin=True)
            _logger.info(
                f"Created admin user '{username}'. "
                "It is recommended that you set a new password as soon as possible "
                f"on {UPDATE_USER_PASSWORD}."
            )
        except MlflowException as e:
            if isinstance(e.__cause__, sqlalchemy.exc.IntegrityError):
                # When multiple workers are starting up at the same time, it's possible
                # that they try to create the admin user at the same time and one of them
                # will succeed while the others will fail with an IntegrityError.
                return
            raise


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
  <input type="hidden" name="csrf_token" value="{{ csrf_token() }}"/>
  <div class="logo-container">
    {% autoescape false %}
    {{ mlflow_logo }}
    {% endautoescape %}
  </div>
  <label for="username">Username:</label>
  <br>
  <input type="text" id="username" name="username" minlength="4">
  <br>
  <label for="password">Password:</label>
  <br>
  <input type="password" id="password" name="password" minlength="12">
  <br>
  <br>
  <input type="submit" value="Sign up">
</form>
""",
        mlflow_logo=MLFLOW_LOGO,
        users_route=CREATE_USER_UI,
    )


@catch_mlflow_exception
def create_user_ui(csrf):
    csrf.protect()
    content_type = request.headers.get("Content-Type")
    if content_type == "application/x-www-form-urlencoded":
        username = request.form["username"]
        password = request.form["password"]

        if not username or not password:
            message = "Username and password cannot be empty."
            return make_response(message, 400)

        if store.has_user(username):
            flash(f"Username has already been taken: {username}")
            return alert(href=SIGNUP)

        store.create_user(username, password)
        flash(f"Successfully signed up user: {username}")
        return alert(href=HOME)
    else:
        message = "Invalid content type. Must be application/x-www-form-urlencoded"
        return make_response(message, 400)


@catch_mlflow_exception
def create_user():
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        username = _get_request_param("username")
        password = _get_request_param("password")

        if not username or not password:
            message = "Username and password cannot be empty."
            return make_response(message, 400)

        user = store.create_user(username, password)
        return jsonify({"user": user.to_json()})
    else:
        message = "Invalid content type. Must be application/json"
        return make_response(message, 400)


@catch_mlflow_exception
def get_user():
    username = _get_request_param("username")
    user = store.get_user(username)
    return jsonify({"user": user.to_json()})


@catch_mlflow_exception
def update_user_password():
    username = _get_request_param("username")
    password = _get_request_param("password")
    store.update_user(username, password=password)
    return make_response({})


@catch_mlflow_exception
def update_user_admin():
    username = _get_request_param("username")
    is_admin = _get_request_param("is_admin")
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
    return jsonify({"experiment_permission": ep.to_json()})


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


@catch_mlflow_exception
def create_scorer_permission():
    experiment_id = _get_request_param("experiment_id")
    scorer_name = _get_request_param("scorer_name")
    username = _get_request_param("username")
    permission = _get_request_param("permission")
    sp = store.create_scorer_permission(experiment_id, scorer_name, username, permission)
    return jsonify({"scorer_permission": sp.to_json()})


@catch_mlflow_exception
def get_scorer_permission():
    experiment_id = _get_request_param("experiment_id")
    scorer_name = _get_request_param("scorer_name")
    username = _get_request_param("username")
    sp = store.get_scorer_permission(experiment_id, scorer_name, username)
    return make_response({"scorer_permission": sp.to_json()})


@catch_mlflow_exception
def update_scorer_permission():
    experiment_id = _get_request_param("experiment_id")
    scorer_name = _get_request_param("scorer_name")
    username = _get_request_param("username")
    permission = _get_request_param("permission")
    store.update_scorer_permission(experiment_id, scorer_name, username, permission)
    return make_response({})


@catch_mlflow_exception
def delete_scorer_permission():
    experiment_id = _get_request_param("experiment_id")
    scorer_name = _get_request_param("scorer_name")
    username = _get_request_param("username")
    store.delete_scorer_permission(experiment_id, scorer_name, username)
    return make_response({})


# =============================================================================
# GraphQL Authorization
# =============================================================================

_auth_initialized = False


def is_auth_enabled() -> bool:
    return _auth_initialized


def _graphql_get_permission_for_experiment(experiment_id: str, username: str) -> Permission:
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def _graphql_get_permission_for_run(run_id: str, username: str) -> Permission:
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    return _get_permission_from_store_or_default(
        lambda: store.get_experiment_permission(experiment_id, username).permission
    )


def _graphql_get_permission_for_model(model_name: str, username: str) -> Permission:
    return _get_permission_from_store_or_default(
        lambda: store.get_registered_model_permission(model_name, username).permission
    )


def _graphql_can_read_experiment(experiment_id: str, username: str) -> bool:
    return _graphql_get_permission_for_experiment(experiment_id, username).can_read


def _graphql_can_read_run(run_id: str, username: str) -> bool:
    return _graphql_get_permission_for_run(run_id, username).can_read


def _graphql_can_read_model(model_name: str, username: str) -> bool:
    return _graphql_get_permission_for_model(model_name, username).can_read


class GraphQLAuthorizationMiddleware:
    """
    Graphene middleware that enforces per-object authorization for GraphQL queries.

    This middleware checks user permissions before resolving protected fields.
    It integrates with MLflow's basic-auth permission system.
    """

    PROTECTED_FIELDS = {
        "mlflowGetExperiment",
        "mlflowGetRun",
        "mlflowListArtifacts",
        "mlflowGetMetricHistoryBulkInterval",
        "mlflowSearchRuns",
        "mlflowSearchDatasets",
    }

    def resolve(self, next, root, info, **args):
        """
        Middleware resolve function called for every field resolution.

        Args:
            next: The next resolver in the chain
            root: The root value object
            info: GraphQL resolve info containing field name and context
            args: Field arguments as keyword arguments

        Returns:
            The resolved value or an error response
        """
        field_name = info.field_name

        if field_name not in self.PROTECTED_FIELDS:
            return next(root, info, **args)

        try:
            authorization = authenticate_request()
            if isinstance(authorization, Response):
                return None
            username = authorization.username

            if store.get_user(username).is_admin:
                return next(root, info, **args)
        except Exception:
            _logger.warning("GraphQL authorization failed: auth system error", exc_info=True)
            return None

        try:
            if not self._check_authorization(field_name, args, username):
                _logger.debug(f"GraphQL authorization denied for {field_name} by user {username}")
                return None
        except MlflowException:
            return None
        except Exception:
            _logger.warning(f"GraphQL authorization error for {field_name}", exc_info=True)
            return None

        return next(root, info, **args)

    def _check_authorization(self, field_name: str, args: dict[str, Any], username: str) -> bool:
        """
        Check if the user is authorized to access the requested field.

        Args:
            field_name: The GraphQL field being resolved
            args: The field arguments
            username: The authenticated username

        Returns:
            True if authorized, False otherwise
        """
        input_obj = args.get("input")
        if input_obj is None:
            # No input means no specific resource to check
            return True

        if field_name == "mlflowGetExperiment":
            if experiment_id := getattr(input_obj, "experiment_id", None):
                return _graphql_can_read_experiment(experiment_id, username)

        elif field_name in ("mlflowGetRun", "mlflowListArtifacts"):
            if run_id := (
                getattr(input_obj, "run_id", None) or getattr(input_obj, "run_uuid", None)
            ):
                return _graphql_can_read_run(run_id, username)

        elif field_name == "mlflowGetMetricHistoryBulkInterval":
            run_ids = getattr(input_obj, "run_ids", None) or []
            for run_id in run_ids:
                if not _graphql_can_read_run(run_id, username):
                    return False

        elif field_name in ("mlflowSearchRuns", "mlflowSearchDatasets"):
            if experiment_ids := (getattr(input_obj, "experiment_ids", None) or []):
                readable_ids = [
                    exp_id
                    for exp_id in experiment_ids
                    if _graphql_can_read_experiment(exp_id, username)
                ]
                if not readable_ids:
                    return False
                input_obj.experiment_ids = readable_ids

        return True


def get_graphql_authorization_middleware():
    """
    Get the GraphQL authorization middleware instance if auth is enabled.

    Returns:
        A list containing the middleware instance if auth is enabled,
        empty list otherwise. Suitable for passing to schema.execute(middleware=...).
    """
    if not MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH.get():
        return []
    if not is_auth_enabled():
        return []
    return [GraphQLAuthorizationMiddleware()]


def create_app(app: Flask = app):
    """
    A factory to enable authentication and authorization for the MLflow server.

    Args:
        app: The Flask app to enable authentication and authorization for.

    Returns:
        The app with authentication and authorization enabled.
    """
    global _auth_initialized

    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )

    # a secret key is required for flashing, and also for
    # CSRF protection. it's important that this is a static key,
    # otherwise CSRF validation won't work across workers.
    secret_key = MLFLOW_FLASK_SERVER_SECRET_KEY.get()
    if not secret_key:
        raise MlflowException(
            "A static secret key needs to be set for CSRF protection. Please set the "
            "`MLFLOW_FLASK_SERVER_SECRET_KEY` environment variable before starting the "
            "server. For example:\n\n"
            "export MLFLOW_FLASK_SERVER_SECRET_KEY='my-secret-key'\n\n"
            "If you are using multiple servers, please ensure this key is consistent between "
            "them, in order to prevent validation issues."
        )
    app.secret_key = secret_key

    # we only need to protect the CREATE_USER_UI route, since that's
    # the only browser-accessible route. the rest are client / REST
    # APIs that do not have access to the CSRF token for validation
    app.config["WTF_CSRF_CHECK_DEFAULT"] = False
    csrf = CSRFProtect()
    csrf.init_app(app)

    store.init_db(auth_config.database_uri)
    create_admin_user(auth_config.admin_username, auth_config.admin_password)

    _auth_initialized = True

    app.add_url_rule(
        rule=SIGNUP,
        view_func=signup,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=CREATE_USER_UI,
        view_func=lambda: create_user_ui(csrf),
        methods=["POST"],
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
    app.add_url_rule(
        rule=CREATE_SCORER_PERMISSION,
        view_func=create_scorer_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        rule=GET_SCORER_PERMISSION,
        view_func=get_scorer_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=UPDATE_SCORER_PERMISSION,
        view_func=update_scorer_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        rule=DELETE_SCORER_PERMISSION,
        view_func=delete_scorer_permission,
        methods=["DELETE"],
    )

    app.before_request(_before_request)
    app.after_request(_after_request)

    if _MLFLOW_SGI_NAME.get() == "uvicorn":
        return create_fastapi_app(app)
    else:
        return app
