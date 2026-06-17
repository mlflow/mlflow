"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

from __future__ import annotations

import base64
import functools
import hmac
import importlib
import json
import logging
import re
import secrets
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from typing import Any, Awaitable, Callable

import sqlalchemy
from cachetools import TTLCache
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from flask import (
    Flask,
    Request,
    Response,
    flash,
    g,
    jsonify,
    make_response,
    render_template_string,
    request,
)
from starlette.requests import Request as StarletteRequest
from werkzeug.datastructures import Authorization

from mlflow import MlflowException
from mlflow.entities import Experiment
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.model_registry import RegisteredModel
from mlflow.environment_variables import (
    _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN,
    _MLFLOW_SGI_NAME,
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_RBAC_SEED_DEFAULT_ROLES,
    MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH,
)
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.protos.label_schemas_pb2 import (
    CreateLabelSchema,
    DeleteLabelSchema,
    GetLabelSchema,
    GetLabelSchemaByName,
    ListLabelSchemas,
    UpdateLabelSchema,
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
    SearchModelVersions,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.review_queues_pb2 import (
    AddItemsToReviewQueue,
    CreateReviewQueue,
    DeleteReviewQueue,
    GetOrCreateUserQueue,
    GetReviewQueue,
    GetReviewQueueByName,
    ListReviewQueueItems,
    ListReviewQueues,
    RemoveItemsFromReviewQueue,
    SetReviewQueueItemStatus,
    UpdateReviewQueue,
)
from mlflow.protos.service_pb2 import (
    AttachModelToGatewayEndpoint,
    BatchGetTraceInfos,
    BatchGetTraces,
    CalculateTraceFilterCorrelation,
    CancelPromptOptimizationJob,
    CreateAssessment,
    CreateExperiment,
    CreateGatewayBudgetPolicy,
    CreateGatewayEndpoint,
    CreateGatewayEndpointBinding,
    CreateGatewayModelDefinition,
    CreateGatewaySecret,
    CreateLoggedModel,
    CreatePromptOptimizationJob,
    CreateRun,
    CreateWorkspace,
    DeleteAssessment,
    DeleteExperiment,
    DeleteExperimentTag,
    DeleteGatewayBudgetPolicy,
    DeleteGatewayEndpoint,
    DeleteGatewayEndpointBinding,
    DeleteGatewayEndpointTag,
    DeleteGatewayModelDefinition,
    DeleteGatewaySecret,
    DeleteLoggedModel,
    DeleteLoggedModelTag,
    DeletePromptOptimizationJob,
    DeleteRun,
    DeleteScorer,
    DeleteTag,
    DeleteTraces,
    DeleteTracesV3,
    DeleteTraceTag,
    DeleteTraceTagV3,
    DeleteWorkspace,
    DetachModelFromGatewayEndpoint,
    EndTrace,
    FinalizeLoggedModel,
    GetAssessmentRequest,
    GetExperiment,
    GetExperimentByName,
    GetGatewayEndpoint,
    GetGatewayModelDefinition,
    GetGatewaySecretInfo,
    GetLoggedModel,
    GetMetricHistory,
    GetPromptOptimizationJob,
    GetRun,
    GetScorer,
    GetTrace,
    GetTraceInfo,
    GetTraceInfoV3,
    GetWorkspace,
    LinkPromptsToTrace,
    LinkTracesToRun,
    ListArtifacts,
    ListGatewayEndpointBindings,
    ListLoggedModelArtifacts,
    ListScorers,
    ListScorerVersions,
    ListWorkspaces,
    LogBatch,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogParam,
    QueryTraceMetrics,
    RegisterScorer,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SearchLoggedModels,
    SearchPromptOptimizationJobs,
    SearchTraces,
    SearchTracesV3,
    SetExperimentTag,
    SetGatewayEndpointTag,
    SetLoggedModelTags,
    SetTag,
    SetTraceTag,
    SetTraceTagV3,
    StartTrace,
    StartTraceV3,
    UpdateAssessment,
    UpdateExperiment,
    UpdateGatewayBudgetPolicy,
    UpdateGatewayEndpoint,
    UpdateGatewayModelDefinition,
    UpdateGatewaySecret,
    UpdateRun,
    UpdateWorkspace,
)
from mlflow.protos.service_pb2 import (
    GetGatewayBudgetPolicy as GetGatewayBudgetPolicy,
)
from mlflow.protos.service_pb2 import (
    ListGatewayBudgetPolicies as ListGatewayBudgetPolicies,
)
from mlflow.protos.service_pb2 import (
    ListGatewayEndpoints as ListGatewayEndpoints,
)
from mlflow.protos.service_pb2 import (
    ListGatewayModelDefinitions as ListGatewayModelDefinitions,
)
from mlflow.protos.service_pb2 import (
    ListGatewaySecretInfos as ListGatewaySecretInfos,
)
from mlflow.protos.webhooks_pb2 import (
    CreateWebhook,
    DeleteWebhook,
    GetWebhook,
    ListWebhooks,
    TestWebhook,
    UpdateWebhook,
    WebhookService,
)
from mlflow.server import app
from mlflow.server.asgi_utils import get_routed_asgi_path
from mlflow.server.auth.config import DEFAULT_AUTHORIZATION_FUNCTION, read_auth_config
from mlflow.server.auth.entities import GetUserPermissionResult, User
from mlflow.server.auth.logo import MLFLOW_LOGO
from mlflow.server.auth.permissions import (
    MANAGE,
    NO_PERMISSIONS,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_GATEWAY_ENDPOINT,
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
    RESOURCE_TYPE_GATEWAY_SECRET,
    RESOURCE_TYPE_REGISTERED_MODEL,
    RESOURCE_TYPE_SCORER,
    RESOURCE_TYPE_WORKSPACE,
    USE,
    Permission,
    _validate_resource_type,
    get_permission,
)
from mlflow.server.auth.permissions import (
    max_permission as max_permission,
)
from mlflow.server.auth.routes import (
    ADD_ROLE_PERMISSION,
    AJAX_ADD_ROLE_PERMISSION,
    AJAX_ASSIGN_ROLE,
    AJAX_CREATE_ROLE,
    AJAX_CREATE_USER,
    AJAX_DELETE_ROLE,
    AJAX_DELETE_USER,
    AJAX_GET_CURRENT_USER,
    AJAX_GET_ROLE,
    AJAX_GET_USER,
    AJAX_GET_USER_PERMISSION,
    AJAX_GRANT_USER_PERMISSION,
    AJAX_LIST_CURRENT_USER_PERMISSIONS,
    AJAX_LIST_ROLE_PERMISSIONS,
    AJAX_LIST_ROLE_USERS,
    AJAX_LIST_ROLES,
    AJAX_LIST_USER_PERMISSIONS,
    AJAX_LIST_USER_ROLES,
    AJAX_LIST_USERS,
    AJAX_REMOVE_ROLE_PERMISSION,
    AJAX_REVOKE_USER_PERMISSION,
    AJAX_UNASSIGN_ROLE,
    AJAX_UPDATE_ROLE,
    AJAX_UPDATE_ROLE_PERMISSION,
    AJAX_UPDATE_USER_ADMIN,
    AJAX_UPDATE_USER_PASSWORD,
    ASSIGN_ROLE,
    CREATE_PROMPTLAB_RUN,
    CREATE_ROLE,
    CREATE_USER,
    CREATE_USER_UI,
    DELETE_ROLE,
    DELETE_USER,
    GATEWAY_PROVIDER_CONFIG,
    GATEWAY_PROXY,
    GATEWAY_SECRETS_CONFIG,
    GATEWAY_SUPPORTED_MODELS,
    GATEWAY_SUPPORTED_PROVIDERS,
    GET_ARTIFACT,
    GET_CURRENT_USER,
    GET_METRIC_HISTORY_BULK,
    GET_METRIC_HISTORY_BULK_INTERVAL,
    GET_MODEL_VERSION_ARTIFACT,
    GET_ROLE,
    GET_TRACE_ARTIFACT,
    GET_TRACE_ARTIFACT_V3,
    GET_USER,
    GET_USER_PERMISSION,
    GRANT_USER_PERMISSION,
    HOME,
    INVOKE_SCORER,
    LIST_CURRENT_USER_PERMISSIONS,
    LIST_ROLE_PERMISSIONS,
    LIST_ROLE_USERS,
    LIST_ROLES,
    LIST_USER_PERMISSIONS,
    LIST_USER_ROLES,
    LIST_USERS,
    REMOVE_ROLE_PERMISSION,
    REVOKE_USER_PERMISSION,
    SEARCH_DATASETS,
    SIGNUP,
    UNASSIGN_ROLE,
    UPDATE_ROLE,
    UPDATE_ROLE_PERMISSION,
    UPDATE_USER_ADMIN,
    UPDATE_USER_PASSWORD,
    UPLOAD_ARTIFACT,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.server.handlers import (
    _add_static_prefix,
    _get_ajax_path,
    _get_model_registry_store,
    _get_request_message,
    _get_tracking_store,
    catch_mlflow_exception,
    get_endpoints,
    get_service_endpoints,
)
from mlflow.server.handlers import (
    _disable_if_workspaces_disabled as _disable_if_workspaces_disabled,
)
from mlflow.server.jobs import get_job
from mlflow.server.workspace_helpers import _get_workspace_store
from mlflow.store.entities import PagedList
from mlflow.store.workspace.utils import get_default_workspace_optional
from mlflow.utils import workspace_context
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

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

# Cache for resource_id -> workspace_name mapping. The relationship between a resource
# (experiment, registered model) and its workspace is immutable.
_RESOURCE_WORKSPACE_CACHE: TTLCache[str, str | None] = TTLCache(
    maxsize=auth_config.workspace_cache_max_size,
    ttl=auth_config.workspace_cache_ttl_seconds,
)

# Cache for successful basic-auth credential checks. Keys derive from the password via
# HMAC-SHA256 using a per-process secret, so the plaintext password is not stored in the
# cache key *and* the digest is not usable for offline dictionary attack if process memory
# is ever compromised (an attacker without the HMAC key cannot recompute the digest).
# Skipping the PBKDF2 hash comparison inside ``store.authenticate_user`` on cache hits is
# the dominant cost saving — a single check_password_hash call costs tens of milliseconds
# by design.
_USER_AUTH_CACHE: TTLCache[tuple[str, bytes], User] | None = (
    TTLCache(
        maxsize=auth_config.auth_cache_max_size,
        ttl=auth_config.auth_cache_ttl_seconds,
    )
    if auth_config.auth_cache_ttl_seconds > 0
    else None
)
# cachetools.TTLCache is not thread-safe — Flask handlers run under gunicorn's thread
# pool, so every touch of the cache needs to hold this lock.
_USER_AUTH_CACHE_LOCK = threading.Lock()
# Random per-process key for the HMAC that turns the password into a cache-key digest.
# Regenerated at every server start; the cache is ephemeral anyway (process-local, TTL'd),
# so invalidating the key on restart costs nothing beyond a one-time re-auth for each
# active credential.
_USER_AUTH_CACHE_HMAC_KEY = secrets.token_bytes(32)


def _auth_cache_key(username: str, password: str) -> tuple[str, bytes]:
    digest = hmac.new(_USER_AUTH_CACHE_HMAC_KEY, password.encode("utf-8"), "sha256").digest()
    return (username, digest)


def _authenticate_cached(username: str, password: str) -> User | None:
    """Run basic-auth verification with the credential cache in front of it.

    Used by both the Flask (``authenticate_request_basic_auth``) and FastAPI
    (``_authenticate_fastapi_request``) auth paths so neither pays the PBKDF2
    cost twice for the same credential within ``auth_cache_ttl_seconds``.

    Returns the ``User`` on success, or ``None`` when the credential is invalid
    or the user has been deleted between ``authenticate_user`` and ``get_user``.
    """
    if _USER_AUTH_CACHE is None:
        if not store.authenticate_user(username, password):
            return None
        try:
            return store.get_user(username)
        except MlflowException:
            return None

    key = _auth_cache_key(username, password)
    with _USER_AUTH_CACHE_LOCK:
        cached = _USER_AUTH_CACHE.get(key)
    if cached is not None:
        return cached

    # Keep the PBKDF2 comparison outside the lock so concurrent verifications for
    # *different* credentials still run in parallel.
    if not store.authenticate_user(username, password):
        return None
    try:
        user = store.get_user(username)
    except MlflowException:
        # User was deleted between authenticate_user and get_user — treat as auth
        # failure and don't cache anything.
        return None
    with _USER_AUTH_CACHE_LOCK:
        _USER_AUTH_CACHE[key] = user
    return user


def _invalidate_user_auth_cache(username: str) -> None:
    """Drop every cached credential for ``username``.

    Called from user-mutation routes (password change, admin flag change, deletion)
    so those changes take effect immediately instead of after ``auth_cache_ttl_seconds``.
    """
    if _USER_AUTH_CACHE is None:
        return
    with _USER_AUTH_CACHE_LOCK:
        for key in [k for k in _USER_AUTH_CACHE if k[0] == username]:
            _USER_AUTH_CACHE.pop(key, None)


_UNPROTECTED_PATH_PREFIXES = ("/static", "/favicon.ico", "/health")


def is_unprotected_route(path: str) -> bool:
    # When ``_MLFLOW_STATIC_PREFIX`` is set, the health/static routes are
    # actually served from e.g. ``/mlflow/health``, not ``/health``. Match
    # both the unprefixed and the prefixed forms so health checks don't end
    # up requiring auth on prefixed deployments.
    prefixed = tuple(_add_static_prefix(p) for p in _UNPROTECTED_PATH_PREFIXES)
    return path.startswith(_UNPROTECTED_PATH_PREFIXES) or path.startswith(prefixed)


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
        # Coerce null/empty/non-dict JSON bodies to {} so callers get a 400, not
        # a 500 from the dict-merge below.
        body = request.get_json(silent=True)
        args = body if isinstance(body, dict) else {}
    elif request.method == "DELETE":
        if request.is_json:
            body = request.get_json(silent=True)
            args = body if isinstance(body, dict) else {}
        else:
            args = request.args
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


def _get_int_request_param(param: str) -> int:
    """
    Extract an integer request parameter or raise ``INVALID_PARAMETER_VALUE``.

    Wraps ``_get_request_param`` so non-numeric input produces a 400 instead of bubbling
    up a ``ValueError`` and surfacing as a 500.
    """
    return _coerce_int_param(param, _get_request_param(param))


def _coerce_int_param(param: str, raw: object) -> int:
    """
    Convert an already-extracted parameter value to ``int`` or raise
    ``INVALID_PARAMETER_VALUE`` on non-numeric input. Used by call sites that pick
    the parameter themselves (e.g. branching on which of several optional keys is
    present) instead of going through ``_get_request_param``.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise MlflowException.invalid_parameter_value(
            f"Parameter '{param}' must be an integer. Got: {raw!r}"
        )


def _user_inherits_default_workspace_grant(workspace_name: str) -> bool:
    """
    True if the request workspace is the configured default workspace *and* the
    auth server is opted into auto-granting ``default_permission`` there
    (``auth_config.grant_default_workspace_access``). Used as a fallback for the
    resource-permission resolver and create-gate so deployments that relied on
    the pre-simplification implicit "default workspace is open" behavior keep
    working when configured.
    """
    if not auth_config.grant_default_workspace_access:
        return False
    default_workspace, _ = get_default_workspace_optional(_get_workspace_store())
    return default_workspace is not None and workspace_name == default_workspace.name


def _get_role_permission_or_default(
    role_permission_func: Callable[[], Permission | None],
) -> Permission:
    """Fold the role-derived permission against ``default_permission`` as a floor.

    ``NO_PERMISSIONS`` is preserved rather than max'd against ``default_permission``
    — it's the resolver's "user has no presence in this workspace" signal (no role
    matches in the resource's workspace and it isn't an autograted default workspace).
    That's the only place the workspace boundary lives in this chain; lifting it via
    the floor would silently leak ``default_permission`` (e.g. READ) into every
    workspace the user has no role in. ``None`` (workspaces disabled, no grant) still
    falls through to ``default_permission`` as the safety net.
    """
    perm = role_permission_func()
    default = get_permission(auth_config.default_permission)
    if perm is None:
        # Workspaces disabled, no grant matched.
        return default
    if perm.name == NO_PERMISSIONS.name:
        # Workspace-boundary deny — see docstring.
        return perm
    return get_permission(max_permission(perm.name, default.name))


def _user_can_create_in_workspace() -> bool:
    """
    True if the current request can create new resources in the request's
    workspace. Always allows when workspaces are disabled. Otherwise requires
    a workspace-wide grant whose level has ``can_use`` (i.e. USE or MANAGE under
    the simplified two-tier workspace model). Resource-specific grants don't
    confer create rights — only workspace-wide grants do.

    Querying with ``resource_type='workspace'`` restricts the resolver to the
    unified workspace-wide grant slot (``rp.resource_type='workspace'`` with
    ``rp.resource_pattern='*'``); resource-specific grants on concrete types
    don't satisfy this lookup.

    Also honors ``auth_config.grant_default_workspace_access``: when enabled and
    the request workspace is the default workspace, an ungranted user inherits
    ``default_permission`` and can create iff that permission carries ``can_use``.
    """
    if not MLFLOW_ENABLE_WORKSPACES.get():
        return True

    workspace_name = workspace_context.get_request_workspace()
    if workspace_name is None:
        return False

    user = store.get_user(authenticate_request().username)
    perm = store.get_role_permission_for_resource(user.id, "workspace", "*", workspace_name)
    if perm is not None and perm.can_use:
        return True
    if perm is None and _user_inherits_default_workspace_grant(workspace_name):
        return get_permission(auth_config.default_permission).can_use
    return False


def _get_resource_workspace(
    resource_id: str,
    fetcher: Callable[[str], Any],
    resource_label: str,
    silent: bool = False,
) -> str | None:
    """
    Get the workspace name for a resource, using a cache to avoid repeated lookups.

    The resource->workspace relationship is immutable, so caching is safe.

    Args:
        silent: When True, suppress the lookup-failure warning. Set by
            non-authorization callers (e.g. listing endpoints) where a
            ``None`` return is an expected outcome for deleted resources
            rather than a security-relevant error.
    """
    # Use a cache key that includes the resource_label to avoid collisions between
    # experiments and registered models that might have the same ID/name.
    workspace_scope = (
        workspace_context.get_request_workspace() if MLFLOW_ENABLE_WORKSPACES.get() else None
    )
    cache_key = (
        f"{resource_label}:{workspace_scope}:{resource_id}" if workspace_scope is not None else None
    )

    if cache_key is not None and cache_key in _RESOURCE_WORKSPACE_CACHE:
        return _RESOURCE_WORKSPACE_CACHE[cache_key]

    try:
        resource = fetcher(resource_id)
        workspace_name = getattr(resource, "workspace", None)
    except MlflowException as e:
        if not silent:
            _logger.warning(
                "Failed to determine workspace for %s '%s': %s. Denying access for security.",
                resource_label,
                resource_id,
                e,
            )
        workspace_name = None

    if cache_key is None:
        cache_key = (
            f"{resource_label}:{workspace_name}:{resource_id}"
            if workspace_name is not None
            else f"{resource_label}:{resource_id}"
        )

    _RESOURCE_WORKSPACE_CACHE[cache_key] = workspace_name
    return workspace_name


def _get_permission_from_experiment_id() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    username = authenticate_request().username
    return _get_experiment_permission(experiment_id, username)


def _role_permission_for(
    username: str,
    resource_type: str,
    resource_key: str,
    workspace_lookup_id: str,
    workspace_fetcher: Callable[[str], Any],
    workspace_label: str,
) -> Callable[[], Permission | None]:
    """
    Build a callable that resolves a user's role-based permission on a specific resource,
    for use as ``role_permission_func`` in ``_get_role_permission_or_default``.

    ``resource_key`` is the lookup key for ``role_permissions`` (may differ from the
    workspace-resolution id for composite resources, e.g. scorers use
    ``SqlAlchemyStore._scorer_pattern(experiment_id, scorer_name)`` as the role key
    but resolve the workspace via the parent experiment).
    """

    def _role_perm() -> Permission | None:
        user = store.get_user(username)
        workspace_name = _get_resource_workspace(
            workspace_lookup_id, workspace_fetcher, workspace_label
        )
        if workspace_name is None:
            # Workspace lookup failed — when workspaces are enabled, deny by returning
            # NO_PERMISSIONS (security: don't let resource_not_found silently become a
            # default-permission grant). When disabled, fall through to the default.
            return NO_PERMISSIONS if MLFLOW_ENABLE_WORKSPACES.get() else None
        perm = store.get_role_permission_for_resource(
            user.id, resource_type, resource_key, workspace_name
        )
        if perm is not None:
            return perm
        # No grant in the resolved workspace. With workspaces disabled, fall through
        # to the configured default. With workspaces enabled, deny — *unless* the
        # operator opted into ``grant_default_workspace_access`` and this is the
        # default workspace, in which case the user inherits ``default_permission``
        # so deployments that relied on the implicit auto-grant pre-simplification
        # don't lose resource-level access.
        if not MLFLOW_ENABLE_WORKSPACES.get():
            return None
        if _user_inherits_default_workspace_grant(workspace_name):
            return get_permission(auth_config.default_permission)
        return NO_PERMISSIONS

    return _role_perm


def _role_permission_for_known_workspace(
    username: str,
    resource_type: str,
    resource_key: str,
    workspace_name: str | None,
) -> Callable[[], Permission | None]:
    """Like ``_role_permission_for`` but with workspace already resolved.

    Avoids the ``workspace_fetcher`` DB round-trip when the caller already
    holds the resource object (e.g. ``_get_permission_from_registered_model_or_prompt_name``).
    """

    def _role_perm() -> Permission | None:
        if workspace_name is None:
            return NO_PERMISSIONS if MLFLOW_ENABLE_WORKSPACES.get() else None
        user = store.get_user(username)
        perm = store.get_role_permission_for_resource(
            user.id, resource_type, resource_key, workspace_name
        )
        if perm is not None:
            return perm
        if not MLFLOW_ENABLE_WORKSPACES.get():
            return None
        if _user_inherits_default_workspace_grant(workspace_name):
            return get_permission(auth_config.default_permission)
        return NO_PERMISSIONS

    return _role_perm


def _get_experiment_permission(experiment_id: str, username: str) -> Permission:
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


_EXPERIMENT_ID_PATTERN = re.compile(r"^(\d+)/")


def _get_experiment_id_from_view_args():
    # For download/upload/delete artifact endpoints, artifact_path is a URL path parameter.
    # For the list-artifacts endpoint, the path is a query parameter named "path".
    if artifact_path := (request.view_args.get("artifact_path") or request.args.get("path")):
        if m := _EXPERIMENT_ID_PATTERN.match(artifact_path):
            return m.group(1)
    return None


def _get_permission_from_experiment_id_artifact_proxy() -> Permission:
    username = authenticate_request().username

    if experiment_id := _get_experiment_id_from_view_args():
        return _get_role_permission_or_default(
            _role_permission_for(
                username=username,
                resource_type="experiment",
                resource_key=experiment_id,
                workspace_lookup_id=experiment_id,
                workspace_fetcher=_get_tracking_store().get_experiment,
                workspace_label="experiment",
            ),
        )

    if MLFLOW_ENABLE_WORKSPACES.get():
        if workspace_name := workspace_context.get_request_workspace():
            user = store.get_user(username)
            perm = store.get_role_permission_for_resource(user.id, "workspace", "*", workspace_name)
            if perm is not None:
                return perm
            # Honor the default-workspace auto-grant when configured.
            if _user_inherits_default_workspace_grant(workspace_name):
                return get_permission(auth_config.default_permission)
        return NO_PERMISSIONS

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

    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=store_exp.experiment_id,
            workspace_lookup_id=store_exp.experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _get_permission_from_run_id() -> Permission:
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run_id = _get_request_param("run_id")
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _get_permission_from_model_id() -> Permission:
    # logged model permissions inherit from parent resource (experiment)
    model_id = _get_request_param("model_id")
    model = _get_tracking_store().get_logged_model(model_id)
    experiment_id = model.experiment_id
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _get_permission_from_prompt_optimization_job_id() -> Permission:
    # prompt optimization job permissions inherit from parent resource (experiment)
    job_id = _get_request_param("job_id")
    job_entity = get_job(job_id)
    params = json.loads(job_entity.params)
    experiment_id = params.get("experiment_id")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _get_permission_from_registered_model_name() -> Permission:
    name = _get_request_param("name")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="registered_model",
            resource_key=name,
            workspace_lookup_id=name,
            workspace_fetcher=_get_model_registry_store().get_registered_model,
            workspace_label="registered model",
        ),
    )


def _get_permission_from_prompt_name() -> Permission:
    # Grant lookup is namespaced under ``"prompt"`` so a registered_model grant
    # on the same name does not satisfy a prompt request, and vice versa.
    # Workspace resolution reuses the registry's ``get_registered_model``
    # (returns both shapes).
    name = _get_request_param("name")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="prompt",
            resource_key=name,
            workspace_lookup_id=name,
            workspace_fetcher=_get_model_registry_store().get_registered_model,
            workspace_label="prompt",
        ),
    )


def _get_permission_from_registered_model_or_prompt_name() -> Permission:
    """Resolve permission for a shared model-registry route in a single DB round-trip.

    Fetches the ``RegisteredModel`` once, classifies it as prompt or model via
    ``._is_prompt()``, and resolves the workspace from the same object — avoiding
    the separate classify fetch that ``_request_targets_prompt`` would add.
    """
    name = _get_request_param("name")
    username = authenticate_request().username
    workspace_name = None
    resource_type = "registered_model"
    try:
        rm = _get_model_registry_store().get_registered_model(name)
        resource_type = "prompt" if rm._is_prompt() else "registered_model"
        workspace_name = getattr(rm, "workspace", None)
    except MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
    return _get_role_permission_or_default(
        _role_permission_for_known_workspace(username, resource_type, name, workspace_name)
    )


def _get_permission_from_scorer_name() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    name = _get_request_param("name")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="scorer",
            resource_key=store._scorer_pattern(experiment_id, name),
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _get_permission_from_scorer_permission_request() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    scorer_name = _get_request_param("scorer_name")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="scorer",
            resource_key=store._scorer_pattern(experiment_id, scorer_name),
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _get_permission_from_gateway_secret_id() -> Permission:
    secret_id = _get_request_param("secret_id")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="gateway_secret",
            resource_key=secret_id,
            workspace_lookup_id=secret_id,
            workspace_fetcher=lambda sid: _get_tracking_store().get_secret_info(secret_id=sid),
            workspace_label="gateway secret",
        ),
    )


def _get_permission_from_gateway_endpoint_id() -> Permission:
    endpoint_id = _get_request_param("endpoint_id")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="gateway_endpoint",
            resource_key=endpoint_id,
            workspace_lookup_id=endpoint_id,
            workspace_fetcher=lambda eid: _get_tracking_store().get_gateway_endpoint(
                endpoint_id=eid
            ),
            workspace_label="gateway endpoint",
        ),
    )


def _get_permission_from_gateway_model_definition_id() -> Permission:
    model_definition_id = _get_request_param("model_definition_id")
    username = authenticate_request().username
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="gateway_model_definition",
            resource_key=model_definition_id,
            workspace_lookup_id=model_definition_id,
            workspace_fetcher=lambda mdid: _get_tracking_store().get_gateway_model_definition(
                model_definition_id=mdid
            ),
            workspace_label="gateway model definition",
        ),
    )


def validate_can_read_experiment():
    return _get_permission_from_experiment_id().can_read


def validate_can_read_scorer_list():
    # ``ListScorers`` accepts an optional ``experiment_id``. When set, gate
    # on the experiment read permission as usual; when empty, the request is
    # a cross-experiment listing and ``AFTER_REQUEST_PATH_HANDLERS`` does the
    # per-row RBAC filtering, so the route itself is open to any authenticated
    # caller.
    args = request.args if request.method == "GET" else (request.get_json(silent=True) or {})
    if not args.get("experiment_id"):
        return True
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


# Prompt optimization jobs
def validate_can_read_prompt_optimization_job():
    return _get_permission_from_prompt_optimization_job_id().can_read


def validate_can_update_prompt_optimization_job():
    return _get_permission_from_prompt_optimization_job_id().can_update


def validate_can_delete_prompt_optimization_job():
    return _get_permission_from_prompt_optimization_job_id().can_delete


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


# Prompts
def validate_can_read_prompt():
    return _get_permission_from_prompt_name().can_read


def validate_can_update_prompt():
    return _get_permission_from_prompt_name().can_update


def validate_can_delete_prompt():
    return _get_permission_from_prompt_name().can_delete


def validate_can_manage_prompt():
    return _get_permission_from_prompt_name().can_manage


def _request_targets_prompt() -> bool:
    """Classify a shared registered-model request as targeting a prompt.

    Reads the ``mlflow.prompt.is_prompt`` tag from the **persisted** entity, not
    the request body — trusting the body would let a caller with
    ``(prompt, foo, MANAGE)`` spoof the tag on a non-CREATE registered-model
    route and escalate. Missing names and ``RESOURCE_DOES_NOT_EXIST`` fall
    through to the registered-model path; other errors propagate so a broken
    registry doesn't silently flip the auth namespace.
    """
    name = _request_params().get("name")
    if not name:
        return False
    try:
        rm = _get_model_registry_store().get_registered_model(name)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return False
        raise
    return rm._is_prompt()


def _validate_can_read_registered_model_or_prompt():
    return _get_permission_from_registered_model_or_prompt_name().can_read


def _validate_can_update_registered_model_or_prompt():
    return _get_permission_from_registered_model_or_prompt_name().can_update


def _validate_can_delete_registered_model_or_prompt():
    return _get_permission_from_registered_model_or_prompt_name().can_delete


def _validate_can_manage_registered_model_or_prompt():
    return _get_permission_from_registered_model_or_prompt_name().can_manage


def validate_can_create_experiment() -> bool:
    return _user_can_create_in_workspace()


def validate_can_create_registered_model() -> bool:
    return _user_can_create_in_workspace()


def validate_can_view_workspace() -> bool:
    if not MLFLOW_ENABLE_WORKSPACES.get():
        return True

    username = authenticate_request().username

    workspace_name = request.view_args.get("workspace_name") if request.view_args else None
    if workspace_name is None:
        return False

    if username is None:
        return False

    if auth_config.grant_default_workspace_access:
        default_workspace, _ = get_default_workspace_optional(_get_workspace_store())
        if default_workspace and workspace_name == default_workspace.name:
            return True

    names = set(store.list_accessible_workspace_names(username))

    return workspace_name in names


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


def _is_workspace_admin(user_id: int, workspace: str) -> bool:
    return store.is_workspace_admin(user_id, workspace)


def _request_params() -> dict[str, object]:
    """Return the request's params dict (body for POST/PATCH/DELETE, args for GET)."""
    if request.method == "GET":
        return dict(request.args)
    if request.method in ("POST", "PATCH"):
        return dict(request.get_json(silent=True) or {})
    if request.method == "DELETE":
        if request.is_json:
            return dict(request.get_json(silent=True) or {})
        return dict(request.args)
    return {}


def _get_role_workspace_from_request() -> str | None:
    """
    Resolve the workspace the request is targeting for role-authorization purposes.

    Requests identify a role either directly (``role_id``), indirectly via a role
    permission (``role_permission_id``), or by supplying ``workspace`` on create.
    Returns ``None`` if the referenced role/role_permission does not exist — callers
    (validators) should treat that as unauthorized rather than leaking existence via
    a 404.
    """
    params = _request_params()
    try:
        if "role_id" in params:
            return store.get_role(_coerce_int_param("role_id", params["role_id"])).workspace
        if "role_permission_id" in params:
            rp = store.get_role_permission(
                _coerce_int_param("role_permission_id", params["role_permission_id"])
            )
            return store.get_role(rp.role_id).workspace
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        raise
    if "workspace" in params:
        workspace = params["workspace"]
        if not isinstance(workspace, str) or not workspace.strip():
            raise MlflowException.invalid_parameter_value(
                "Parameter 'workspace' must be a non-empty string."
            )
        return workspace
    raise MlflowException.invalid_parameter_value(
        "Request must include one of: role_id, role_permission_id, workspace."
    )


def validate_can_manage_roles():
    username = authenticate_request().username
    user = store.get_user(username)
    if user.is_admin:
        return True
    workspace = _get_role_workspace_from_request()
    if workspace is None:
        return False
    return _is_workspace_admin(user.id, workspace)


def validate_can_view_roles():
    username = authenticate_request().username
    user = store.get_user(username)
    if user.is_admin:
        return True
    workspace = _get_role_workspace_from_request()
    if workspace is None:
        return False
    return store.user_has_any_role_in_workspace(user.id, workspace)


def validate_can_list_roles():
    """
    Authorization for the ``/mlflow/roles/list`` endpoint. The endpoint accepts a
    repeated ``workspace`` query param: zero workspaces lists across the system
    (admin-only), one or more scopes the listing to those workspaces. Non-admins
    must hold a role in *every* workspace they request; only super admins can list
    unscoped.
    """
    username = authenticate_request().username
    user = store.get_user(username)
    if user.is_admin:
        return True
    requested = {
        w.strip() for w in request.args.getlist("workspace") if isinstance(w, str) and w.strip()
    }
    if not requested:
        return False
    # Single batch query — avoids N round-trips when multiple workspaces are
    # requested and dedups duplicates implicitly via the set.
    return requested <= store.list_user_present_workspaces(user.id)


def validate_can_view_user_roles():
    username = authenticate_request().username
    user = store.get_user(username)
    if user.is_admin:
        return True
    target_username = _get_request_param("username")
    if username == target_username:
        return True
    # WP admins can view user roles for users in their workspaces.
    # If the target user does not exist, the handler will raise RESOURCE_DOES_NOT_EXIST;
    # treat this as "not authorized" here rather than letting the validator raise.
    if not store.has_user(target_username):
        return False
    target_user = store.get_user(target_username)
    return store.is_workspace_admin_of_any_of_users_workspaces(user.id, target_user.id)


# Unified per-user permission convenience APIs. Replace the deprecated
# per-resource endpoints with a uniform ``(resource_type, resource_id)`` shape;
# preserve per-resource MANAGE delegation (gated on resource-level MANAGE).


def _scorer_lookup_keys(resource_id: str) -> tuple[str, str]:
    """Split ``<experiment_id>/<url_quote(scorer_name)>`` into (experiment_id, full_pattern)."""
    experiment_id, sep, _ = resource_id.partition("/")
    if not sep:
        raise MlflowException(
            "Invalid scorer resource_id. Expected '<experiment_id>/<scorer_name>'.",
            INVALID_PARAMETER_VALUE,
        )
    return experiment_id, resource_id


def _reject_workspace_resource_type(resource_type: str) -> None:
    # Workspace-tier grants have their own surface (set_workspace_permission /
    # delete_workspace_permission); rejecting here gives every code path the
    # same 400 message regardless of which gate fires first.
    if resource_type == RESOURCE_TYPE_WORKSPACE:
        raise MlflowException(
            "resource_type 'workspace' is not supported by the per-user permission "
            "convenience APIs. Use set_workspace_permission / "
            "delete_workspace_permission for workspace-wide grants.",
            INVALID_PARAMETER_VALUE,
        )


# Maps each resource_type to ``(workspace_label, workspace_fetcher_factory)``.
# Factory is invoked lazily so tests can patch the underlying store.
_RESOURCE_WORKSPACE_FETCHER: dict[str, tuple[str, Callable[[], Callable[[str], Any]]]] = {
    RESOURCE_TYPE_EXPERIMENT: ("experiment", lambda: _get_tracking_store().get_experiment),
    RESOURCE_TYPE_REGISTERED_MODEL: (
        "registered model",
        lambda: _get_model_registry_store().get_registered_model,
    ),
    RESOURCE_TYPE_GATEWAY_SECRET: (
        "gateway secret",
        lambda: lambda sid: _get_tracking_store().get_secret_info(secret_id=sid),
    ),
    RESOURCE_TYPE_GATEWAY_ENDPOINT: (
        "gateway endpoint",
        lambda: lambda eid: _get_tracking_store().get_gateway_endpoint(endpoint_id=eid),
    ),
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION: (
        "gateway model definition",
        lambda: (
            lambda mdid: _get_tracking_store().get_gateway_model_definition(
                model_definition_id=mdid
            )
        ),
    ),
}


@dataclass(frozen=True)
class _ResourceDispatch:
    """Lookup keys for resolving ``(resource_type, resource_id)`` against the
    role-permission store + the workspace fetcher.

    Scorers are the only resource_type whose role-lookup key differs from the
    workspace-lookup id (key = ``<exp_id>/<scorer_name>``, lookup = ``exp_id``).
    """

    resource_key: str
    workspace_lookup_id: str
    workspace_fetcher: Callable[[str], Any]
    workspace_label: str


def _resource_dispatch_keys(resource_type: str, resource_id: str) -> _ResourceDispatch | None:
    """Return the dispatch keys for ``(resource_type, resource_id)``, or ``None``
    if the type isn't supported by the per-user convenience APIs.
    """
    if resource_type == RESOURCE_TYPE_SCORER:
        experiment_id, scorer_pattern = _scorer_lookup_keys(resource_id)
        return _ResourceDispatch(
            resource_key=scorer_pattern,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        )
    spec = _RESOURCE_WORKSPACE_FETCHER.get(resource_type)
    if spec is None:
        return None
    label, fetcher_factory = spec
    return _ResourceDispatch(
        resource_key=resource_id,
        workspace_lookup_id=resource_id,
        workspace_fetcher=fetcher_factory(),
        workspace_label=label,
    )


def _resolve_user_permission_for_resource(
    username: str, resource_type: str, resource_id: str
) -> Permission:
    """Resolve effective permission via the same code path as the runtime check
    (``_get_permission_from_*`` family) so ``get_user_permission`` can't drift
    from real authorization decisions.
    """
    _reject_workspace_resource_type(resource_type)
    _validate_resource_type(resource_type)
    dispatch = _resource_dispatch_keys(resource_type, resource_id)
    if dispatch is None:
        raise MlflowException(
            f"resource_type '{resource_type}' is not supported by the per-user "
            "permission convenience APIs.",
            INVALID_PARAMETER_VALUE,
        )
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type=resource_type,
            resource_key=dispatch.resource_key,
            workspace_lookup_id=dispatch.workspace_lookup_id,
            workspace_fetcher=dispatch.workspace_fetcher,
            workspace_label=dispatch.workspace_label,
        ),
    )


def validate_can_manage_resource() -> bool:
    """Dispatcher validator for ``grant_user_permission`` / ``revoke_user_permission``.
    Gates on the same per-resource MANAGE check the legacy ``validate_can_manage_*``
    validators used.
    """
    resource_type = _get_request_param("resource_type")
    resource_id = _get_request_param("resource_id")
    requester = authenticate_request().username
    return _resolve_user_permission_for_resource(requester, resource_type, resource_id).can_manage


def _workspace_for_resource(resource_type: str, resource_id: str) -> str | None:
    """Resolve the workspace name owning ``(resource_type, resource_id)``, or None
    if the resource can't be located. Fail-closed: any lookup failure returns
    None so authorization gates deny rather than leak across workspaces.
    """
    try:
        dispatch = _resource_dispatch_keys(resource_type, resource_id)
    except MlflowException:
        # Malformed scorer id — deny rather than 500.
        return None
    if dispatch is None:
        return None
    return _get_resource_workspace(
        dispatch.workspace_lookup_id,
        dispatch.workspace_fetcher,
        dispatch.workspace_label,
        silent=True,
    )


def validate_can_get_user_permission() -> bool:
    """Gate ``get_user_permission``: admin / self / workspace MANAGE in the
    resource's workspace. Scoping to the resource workspace closes the
    cross-workspace probe (workspace-A admin asking about a workspace-B resource).
    """
    # Reject workspace upfront so admin and non-admin paths produce the same
    # 400 message rather than a 400 for admin and a 403 for non-admin.
    resource_type = _get_request_param("resource_type")
    _reject_workspace_resource_type(resource_type)
    requester = authenticate_request().username
    requester_user = store.get_user(requester)
    if requester_user.is_admin:
        return True
    target_username = _get_request_param("username")
    if requester == target_username:
        return True
    if not store.has_user(target_username):
        return False
    resource_id = _get_request_param("resource_id")
    workspace_name = _workspace_for_resource(resource_type, resource_id)
    if workspace_name is None:
        return False
    return store.is_workspace_admin(requester_user.id, workspace_name)


def _entity_is_prompt(entity) -> bool:
    """True if a ``RegisteredModel`` / ``ModelVersion`` entity is prompt-flagged.

    Unifies the two response-filtering paths in ``filter_search_*`` — initial
    response rows arrive as protos (via ``parse_dict``), refetched rows arrive
    as ORM entities (``PagedList[RegisteredModel]`` / ``[ModelVersion]``).
    Both ORM entities expose ``_is_prompt()``; protos don't, so we fall back
    to scanning the repeated ``.tags`` field for the prompt marker.
    """
    if hasattr(entity, "_is_prompt"):
        return entity._is_prompt()
    return any(t.key == IS_PROMPT_TAG_KEY and t.value.lower() == "true" for t in entity.tags)


def _rm_or_prompt_read_predicate(username: str) -> Callable[[Any], bool]:
    """Build a ``p(entity) -> bool`` for filtering shared registered-model /
    model-version search responses. Classifies each row by its
    ``mlflow.prompt.is_prompt`` tag and consults the matching grant namespace.
    """
    can_read_rm = _role_based_read_predicate(username, "registered_model")
    can_read_prompt = _role_based_read_predicate(username, "prompt")

    def can_read(entity) -> bool:
        return (can_read_prompt if _entity_is_prompt(entity) else can_read_rm)(entity.name)

    return can_read


def _role_based_read_predicate(username: str, resource_type: str) -> Callable[[str], bool]:
    """
    Build a ``p(resource_id) -> bool`` predicate from ``username``'s role
    grants in the active workspace. Max-style: any positive grant (specific or
    wildcard) wins; ``NO_PERMISSIONS`` rows are ignored. Falls back to
    ``default_permission.can_read`` when workspaces are disabled, otherwise to
    deny.
    """
    workspace_name = (
        workspace_context.get_request_workspace()
        if MLFLOW_ENABLE_WORKSPACES.get()
        else DEFAULT_WORKSPACE_NAME
    )
    if workspace_name is None:
        return lambda _resource_id: False

    user = store.get_user(username)
    readable: set[str] = set()
    wildcard_can_read = False
    for resource_pattern, permission in store.list_role_grants_for_user_in_workspace(
        user.id, workspace_name, resource_type
    ):
        if not get_permission(permission).can_read:
            continue
        if resource_pattern == "*":
            wildcard_can_read = True
        else:
            readable.add(resource_pattern)

    default_can_read = get_permission(auth_config.default_permission).can_read
    fallback = default_can_read if not MLFLOW_ENABLE_WORKSPACES.get() else False

    def predicate(resource_id: str) -> bool:
        return resource_id in readable or wildcard_can_read or fallback

    return predicate


def filter_experiment_ids(experiment_ids: list[str]) -> list[str]:
    """
    Filter experiment IDs to only include those the user has read access to.

    Called from ``search_runs_impl`` before the tracking store query. When workspaces
    are enabled, the tracking store subsequently filters to the active workspace, so we
    only consult role grants in that workspace here — experiments outside it would be
    rejected anyway.

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
        predicate = _role_based_read_predicate(authenticate_request().username, "experiment")
        return [exp_id for exp_id in experiment_ids if predicate(exp_id)]
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


def validate_can_list_users():
    # Any workspace member may list users (the reviewer-assignment UI needs the
    # roster); listing grants no access on its own. Workspace-scoped so the roster
    # isn't leaked across workspaces.
    return _user_can_create_in_workspace()


def validate_can_create_user():
    # Workspace admins may need to seed an account before assigning it a role
    # in a workspace they manage; creating a user grants no access on its own.
    # Deletion stays super-admin-only (see ``validate_can_delete_user``).
    # (Super admins short-circuit in ``_before_request`` and never reach this validator.)
    user = store.get_user(authenticate_request().username)
    return bool(store.list_workspace_admin_workspaces(user.id))


def validate_can_update_user_password():
    return username_is_sender()


def validate_can_update_user_admin():
    # only admins can update, but admins won't reach this validator
    return False


def validate_can_delete_user():
    # only admins can delete, but admins won't reach this validator
    return False


def validate_can_read_gateway_secret():
    return _get_permission_from_gateway_secret_id().can_read


def validate_can_update_gateway_secret():
    return _get_permission_from_gateway_secret_id().can_update


def validate_can_delete_gateway_secret():
    return _get_permission_from_gateway_secret_id().can_delete


def validate_can_manage_gateway_secret():
    return _get_permission_from_gateway_secret_id().can_manage


def validate_can_read_gateway_endpoint():
    return _get_permission_from_gateway_endpoint_id().can_read


def validate_can_delete_gateway_endpoint():
    return _get_permission_from_gateway_endpoint_id().can_delete


def validate_can_manage_gateway_endpoint():
    return _get_permission_from_gateway_endpoint_id().can_manage


def validate_can_read_gateway_model_definition():
    return _get_permission_from_gateway_model_definition_id().can_read


def validate_can_delete_gateway_model_definition():
    return _get_permission_from_gateway_model_definition_id().can_delete


def validate_can_manage_gateway_model_definition():
    return _get_permission_from_gateway_model_definition_id().can_manage


def validate_can_create_gateway_model_definition():
    """
    Validate that the user can create a gateway model definition.
    This requires USE permission on the referenced secret.
    """
    body = request.json or {}
    secret_id = body.get("secret_id")
    if not secret_id:
        # If no secret is provided, allow creation (will fail in handler)
        return True

    username = authenticate_request().username
    permission = _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="gateway_secret",
            resource_key=secret_id,
            workspace_lookup_id=secret_id,
            workspace_fetcher=lambda sid: _get_tracking_store().get_secret_info(secret_id=sid),
            workspace_label="gateway secret",
        ),
    )
    return permission.can_use


def validate_can_update_gateway_model_definition():
    """
    Validate that the user can update a gateway model definition.
    This requires UPDATE permission on the model definition AND
    USE permission on any new secret being referenced.
    """
    # First check update permission on the model definition
    if not _get_permission_from_gateway_model_definition_id().can_update:
        return False

    # If updating the secret, check USE permission on the new secret
    body = request.json or {}
    secret_id = body.get("secret_id")
    if not secret_id:
        # No secret being changed, just return True
        return True

    username = authenticate_request().username
    permission = _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="gateway_secret",
            resource_key=secret_id,
            workspace_lookup_id=secret_id,
            workspace_fetcher=lambda sid: _get_tracking_store().get_secret_info(secret_id=sid),
            workspace_label="gateway secret",
        ),
    )
    return permission.can_use


def _validate_can_use_model_definitions(model_configs: list[dict[str, Any]]) -> bool:
    """
    Helper to validate USE permission on all model definitions in model_configs.
    Returns True if all model definitions have USE permission, False otherwise.
    """
    if not model_configs:
        return True

    model_def_ids = [
        config.get("model_definition_id")
        for config in model_configs
        if config.get("model_definition_id")
    ]

    if not model_def_ids:
        return True

    username = authenticate_request().username
    for model_def_id in model_def_ids:
        permission = _get_role_permission_or_default(
            _role_permission_for(
                username=username,
                resource_type="gateway_model_definition",
                resource_key=model_def_id,
                workspace_lookup_id=model_def_id,
                workspace_fetcher=lambda mdid: _get_tracking_store().get_gateway_model_definition(
                    model_definition_id=mdid
                ),
                workspace_label="gateway model definition",
            ),
        )
        if not permission.can_use:
            return False

    return True


def _validate_can_use_model_definitions_for_create(model_configs: list[dict[str, Any]]) -> bool:
    """
    Create-only helper that enforces workspace USE permission when no model definitions
    are provided, otherwise validates USE permission on referenced model definitions.
    """
    if not model_configs or not any(config.get("model_definition_id") for config in model_configs):
        if not MLFLOW_ENABLE_WORKSPACES.get():
            return True
        workspace_name = workspace_context.get_request_workspace()
        if workspace_name is None:
            return False
        username = authenticate_request().username
        user = store.get_user(username)
        workspace_perm = store.get_role_permission_for_resource(
            user.id, "workspace", "*", workspace_name
        )
        if workspace_perm is not None and workspace_perm.can_use:
            return True
        # Honor ``grant_default_workspace_access``: an ungranted user in the
        # default workspace inherits ``default_permission`` and can create iff
        # that permission carries ``can_use``.
        if workspace_perm is None and _user_inherits_default_workspace_grant(workspace_name):
            return get_permission(auth_config.default_permission).can_use
        return False

    return _validate_can_use_model_definitions(model_configs)


def validate_can_create_gateway_endpoint():
    """
    Validate that the user can create a gateway endpoint.
    This requires USE permission on all referenced model definitions.
    """
    body = request.json or {}
    model_configs = body.get("model_configs", [])
    return _validate_can_use_model_definitions_for_create(model_configs)


def validate_can_update_gateway_endpoint():
    """
    Validate that the user can update a gateway endpoint.
    This requires UPDATE permission on the endpoint AND
    USE permission on any new model definitions being referenced.
    """
    if not _get_permission_from_gateway_endpoint_id().can_update:
        return False

    body = request.json or {}
    model_configs = body.get("model_configs", [])
    return _validate_can_use_model_definitions(model_configs)


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
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
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
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="registered_model",
            resource_key=name,
            workspace_lookup_id=name,
            workspace_fetcher=_get_model_registry_store().get_registered_model,
            workspace_label="registered model",
        ),
    )


def validate_can_read_model_version_artifact():
    """Checks READ permission on model version artifacts."""
    return _get_permission_from_model_version().can_read


def _get_permission_from_trace_request_id() -> Permission:
    request_id = request.args.get("request_id")
    if not request_id:
        raise MlflowException(
            "Request must specify request_id parameter",
            INVALID_PARAMETER_VALUE,
        )
    trace = _get_tracking_store().get_trace_info(request_id)
    return _get_experiment_permission(trace.experiment_id, authenticate_request().username)


def validate_can_read_trace_artifact():
    """Checks READ permission on trace artifacts."""
    return _get_permission_from_trace_request_id().can_read


def _get_permission_from_trace(trace_id: str, username: str) -> Permission:
    try:
        trace = _get_tracking_store().get_trace_info(trace_id)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return NO_PERMISSIONS
        raise
    return _get_experiment_permission(trace.experiment_id, username)


def validate_can_read_trace_by_request_id():
    return _get_permission_from_trace(
        _get_request_param("request_id"), authenticate_request().username
    ).can_read


def validate_can_read_trace_by_trace_id():
    return _get_permission_from_trace(
        _get_request_param("trace_id"), authenticate_request().username
    ).can_read


def validate_can_search_traces():
    experiment_ids = request.args.to_dict(flat=False).get("experiment_ids", [])
    username = authenticate_request().username
    return bool(experiment_ids) and all(
        _get_experiment_permission(eid, username).can_read for eid in experiment_ids
    )


def validate_can_search_traces_v3():
    locations = (request.json or {}).get("locations", [])
    # Only mlflow_experiment locations carry an experiment_id we can permission-check;
    # inference_table and other future location types don't map to a local experiment so
    # they are intentionally excluded and requests containing only those locations are
    # denied (fail-closed) via the bool(experiment_ids) guard below.
    experiment_ids = [
        eid
        for loc in locations
        if isinstance(loc, dict)
        if isinstance(ml_exp := loc.get("mlflow_experiment"), dict)
        if (eid := ml_exp.get("experiment_id"))
    ]
    username = authenticate_request().username
    return bool(experiment_ids) and all(
        _get_experiment_permission(eid, username).can_read for eid in experiment_ids
    )


def validate_can_batch_get_traces():
    if request.method == "GET":
        trace_ids = request.args.to_dict(flat=False).get("trace_ids", [])
    else:
        trace_ids = (request.json or {}).get("trace_ids", [])
    username = authenticate_request().username
    tracking_store = _get_tracking_store()
    try:
        experiment_ids = {tracking_store.get_trace_info(tid).experiment_id for tid in trace_ids}
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return False
        raise
    return bool(experiment_ids) and all(
        _get_experiment_permission(eid, username).can_read for eid in experiment_ids
    )


def validate_can_delete_traces():
    return _get_experiment_permission(
        _get_request_param("experiment_id"), authenticate_request().username
    ).can_delete


def validate_can_update_trace_by_trace_id():
    return _get_permission_from_trace(
        _get_request_param("trace_id"), authenticate_request().username
    ).can_update


def validate_can_update_trace_by_request_id():
    return _get_permission_from_trace(
        _get_request_param("request_id"), authenticate_request().username
    ).can_update


def validate_can_read_traces_by_experiment_ids():
    experiment_ids = (request.json or {}).get("experiment_ids", [])
    username = authenticate_request().username
    return bool(experiment_ids) and all(
        _get_experiment_permission(eid, username).can_read for eid in experiment_ids
    )


def validate_can_start_trace_v3():
    body = request.json or {}
    match body:
        case {
            "trace": {
                "trace_info": {"trace_location": {"mlflow_experiment": {"experiment_id": str(eid)}}}
            }
        } if eid:
            return _get_experiment_permission(eid, authenticate_request().username).can_update
        case _:
            return False


def validate_can_link_traces_to_run():
    tracking_store = _get_tracking_store()
    username = authenticate_request().username
    run_id = _get_request_param("run_id")
    try:
        run = tracking_store.get_run(run_id)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return False
        raise
    if not _get_experiment_permission(run.info.experiment_id, username).can_update:
        return False
    trace_ids = (request.json or {}).get("trace_ids", [])
    try:
        trace_experiment_ids = {
            tracking_store.get_trace_info(tid).experiment_id for tid in trace_ids
        }
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return False
        raise
    return bool(trace_experiment_ids) and all(
        _get_experiment_permission(eid, username).can_read for eid in trace_experiment_ids
    )


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
        permission = _get_role_permission_or_default(
            _role_permission_for(
                username=username,
                resource_type="experiment",
                resource_key=experiment_id,
                workspace_lookup_id=experiment_id,
                workspace_fetcher=_get_tracking_store().get_experiment,
                workspace_label="experiment",
            ),
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
        permission = _get_role_permission_or_default(
            _role_permission_for(
                username=username,
                resource_type="experiment",
                resource_key=experiment_id,
                workspace_lookup_id=experiment_id,
                workspace_fetcher=_get_tracking_store().get_experiment,
                workspace_label="experiment",
            ),
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
    permission = _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )
    return permission.can_update


def validate_gateway_proxy():
    """
    Allows gateway proxy requests without permission checks.
    This endpoint proxies to external services that handle their own authorization.
    """
    return True


# Review queues & label schemas
#
# Permissions inherit from the parent experiment (like runs / logged models).
# Creating a queue requires EDIT (the creator owns it); updating or deleting one
# requires MANAGE or EDIT-with-ownership (reassigning a queue's owner is
# MANAGE-only); managing label schemas (create / update / delete) requires
# MANAGE; routing work into a queue requires EDIT; reviewing through a queue
# (set / reopen status) requires EDIT plus membership in the queue's assigned-user
# pool; reads require experiment READ, with per-queue visibility narrowed by
# ``filter_list_review_queues``.
def _get_permission_from_review_queue_id() -> Permission:
    queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
    username = authenticate_request().username
    return _get_experiment_permission(queue.experiment_id, username)


def _get_permission_from_label_schema_id() -> Permission:
    schema = _get_tracking_store().get_label_schema(_get_request_param("schema_id"))
    username = authenticate_request().username
    return _get_experiment_permission(schema.experiment_id, username)


def _review_queue_has_member(queue, username: str) -> bool:
    # Assigned users are normalized at write time, so normalize only the incoming name.
    target = (username or "").strip().lower()
    return target in queue.users


def _is_review_queue_owner(queue, username: str) -> bool:
    # ``created_by`` is stored case-preserved; compare case-insensitively.
    owner = (queue.created_by or "").strip().lower()
    return bool(owner) and owner == (username or "").strip().lower()


def _can_own_or_manage_review_queue(queue, username: str) -> bool:
    """Owner-level access to a queue: experiment MANAGE, or experiment EDIT and
    you own the queue (``created_by``). Ownership amplifies EDIT — it is never a
    substitute for it.
    """
    perm = _get_experiment_permission(queue.experiment_id, username)
    if perm.can_manage:
        return True
    return perm.can_update and _is_review_queue_owner(queue, username)


def _can_delete_or_prune_review_queue(queue, username: str) -> bool:
    """Whether the user may delete the queue or remove its items (un-assign work).

    A manager may act on any queue; an EDIT owner only on their own CUSTOM queue (a
    USER queue's lifecycle is a manager's responsibility, never its assignee's).
    """
    from mlflow.genai.review_queues import ReviewQueueType

    perm = _get_experiment_permission(queue.experiment_id, username)
    if perm.can_manage:
        return True
    return (
        perm.can_update
        and _is_review_queue_owner(queue, username)
        and queue.queue_type == ReviewQueueType.CUSTOM
    )


def _parse_update_review_queue_request() -> UpdateReviewQueue:
    """Parse the ``UpdateReviewQueue`` request body once for the gate to inspect.

    Field presence is read from the parsed proto, not raw JSON keys, so protobuf
    JSON's camelCase aliases (e.g. ``newOwner``) are detected the same way the
    handler reads them; a raw-key scan would miss the camelCase form.
    """
    body = request.get_json(silent=True)
    message = UpdateReviewQueue()
    parse_dict(body if isinstance(body, dict) else {}, message)
    return message


def _registered_username_match(name: object) -> str | None:
    """The registered user whose name equals ``name`` (case-insensitive), or None.

    User queues are named after their user (lowercased by ``normalize_user``), so a
    custom queue or a rename that takes a username shadows that user's personal
    queue. The match is intentionally case-insensitive — broader than the store's
    case-sensitive name uniqueness — so a look-alike like ``"Alice"`` is rejected
    too, not just the exact ``"alice"`` that would hard-collide and lock the user out.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    target = name.strip().lower()
    return next(
        (u.username for u in store.list_users() if u.username.strip().lower() == target), None
    )


def _reject_create_review_queue_shadowing_user():
    """Reject creating a CUSTOM queue whose name is a registered username."""
    from mlflow.genai.review_queues import ReviewQueueType

    body = request.get_json(silent=True)
    message = CreateReviewQueue()
    parse_dict(body if isinstance(body, dict) else {}, message)
    # Only CUSTOM queues choose an arbitrary name; a USER queue *is* its username.
    # Compare the raw proto enum (an unset queue_type is 0/UNSPECIFIED, which
    # `from_proto` would reject) so a non-CUSTOM request just skips the check.
    if message.queue_type != ReviewQueueType.CUSTOM.to_proto():
        return
    if (matched := _registered_username_match(message.name)) is not None:
        raise MlflowException.invalid_parameter_value(
            f"'{matched}' is a registered user. A custom review queue cannot use a "
            "username as its name; assign that user to a queue instead.",
        )


def _reject_rename_review_queue_shadowing_user(queue, message):
    """Reject renaming a CUSTOM queue onto a registered username.

    User queues can't be renamed at all (the store rejects that), so this only
    concerns a CUSTOM queue being renamed onto a username.
    """
    from mlflow.genai.review_queues import ReviewQueueType

    if queue.queue_type != ReviewQueueType.CUSTOM or not message.HasField("name"):
        return
    if (matched := _registered_username_match(message.name)) is not None:
        raise MlflowException.invalid_parameter_value(
            f"'{matched}' is a registered user. A custom review queue cannot be "
            "renamed to a username; assign that user to a queue instead.",
        )


def validate_can_create_review_queue():
    # Creating (and thereby owning) a queue requires experiment EDIT.
    permission = _get_permission_from_experiment_id().can_update
    # A custom queue may not take a registered username, which would shadow that
    # user's personal queue. Only enforced once the caller is authorized.
    if permission:
        _reject_create_review_queue_shadowing_user()
    return permission


def validate_can_update_review_queue():
    # Editing a queue's shape (name / users / schemas) is allowed to a manager or
    # the owning EDIT user. Reassigning the owner (``new_owner``) is MANAGE-only —
    # an owner cannot transfer their own queue.
    username = authenticate_request().username
    queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
    message = _parse_update_review_queue_request()
    if message.HasField("new_owner"):
        permission = _get_experiment_permission(queue.experiment_id, username).can_manage
    else:
        permission = _can_own_or_manage_review_queue(queue, username)
    # A rename can't take a registered username either (same shadowing concern).
    if permission:
        _reject_rename_review_queue_shadowing_user(queue, message)
    return permission


def enforce_review_queue_name_not_username():
    """Reject a review-queue create/rename that shadows a username, for admins.

    A custom queue (or a rename) named after a registered user shadows that user's
    personal queue, so this is a data-integrity rule that applies to *every* caller,
    not a permission. Non-admins hit it inside the validators above, after the
    permission gate. Admins bypass validators entirely, so ``_before_request`` calls
    this for them. A no-op for every endpoint other than create/update review queue.
    """
    # Resolve via the dispatcher (not a raw path lookup) so this stays correct if
    # these routes ever gain a path parameter, matching how non-admins are routed.
    validator = _find_validator(request)
    if validator is validate_can_create_review_queue:
        _reject_create_review_queue_shadowing_user()
    elif validator is validate_can_update_review_queue:
        queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
        _reject_rename_review_queue_shadowing_user(queue, _parse_update_review_queue_request())


def validate_can_remove_items_from_review_queue():
    username = authenticate_request().username
    queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
    return _can_delete_or_prune_review_queue(queue, username)


def validate_can_delete_review_queue():
    username = authenticate_request().username
    queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
    return _can_delete_or_prune_review_queue(queue, username)


def validate_can_add_items_to_review_queue():
    # Adding items (flag-for-review) is open to any EDITor, unlike removing them.
    return _get_permission_from_review_queue_id().can_update


def validate_can_get_or_create_user_queue():
    return _get_permission_from_experiment_id().can_update


def validate_can_view_review_queue():
    # Detail-tier read: experiment READ plus MANAGE, owner, or membership. Mirrors
    # the row predicate in ``filter_list_review_queues``.
    username = authenticate_request().username
    queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
    perm = _get_experiment_permission(queue.experiment_id, username)
    if not perm.can_read:
        return False
    if perm.can_manage or _review_queue_has_member(queue, username):
        return True
    return perm.can_update and _is_review_queue_owner(queue, username)


def validate_can_view_review_queue_by_name():
    experiment_id = _get_request_param("experiment_id")
    username = authenticate_request().username
    perm = _get_experiment_permission(experiment_id, username)
    if not perm.can_read:
        return False
    if perm.can_manage:
        return True
    queue = _get_tracking_store().get_review_queue_by_name(
        experiment_id, name=_get_request_param("name")
    )
    return _review_queue_has_member(queue, username) or (
        perm.can_update and _is_review_queue_owner(queue, username)
    )


def validate_can_review_queue_item():
    # Submitting / reopening review work: experiment EDIT plus membership in the
    # queue's assigned-user pool (even a manager must assign themselves first).
    # Fetch the queue once and resolve the experiment permission from it.
    username = authenticate_request().username
    queue = _get_tracking_store().get_review_queue(_get_request_param("queue_id"))
    perm = _get_experiment_permission(queue.experiment_id, username)
    return perm.can_update and _review_queue_has_member(queue, username)


def validate_can_create_label_schema():
    return _get_permission_from_experiment_id().can_manage


def validate_can_read_label_schema():
    return _get_permission_from_label_schema_id().can_read


def validate_can_manage_label_schema():
    return _get_permission_from_label_schema_id().can_manage


def filter_list_review_queues(resp: Response) -> None:
    """Narrow a ``ListReviewQueues`` response to queues the caller may see.

    A server admin or any user with experiment EDIT (or MANAGE) sees every
    queue (the list tier is intentionally broad — clicking into a queue is
    separately gated by ``validate_can_view_review_queue``). A READ-only user
    sees only queues they are assigned to (their personal queue plus any custom
    queue whose assigned-user pool contains them).
    """
    if sender_is_admin():
        return

    response_message = ListReviewQueues.Response()
    parse_dict(resp.json, response_message)

    username = authenticate_request().username
    # One shared experiment, so resolve the grant once: EDIT/MANAGE see all rows,
    # READ-only users see only queues they're assigned to.
    experiment_id = _get_request_param("experiment_id")
    perm = _get_experiment_permission(experiment_id, username)
    if perm.can_update:
        return

    visible = [q for q in response_message.review_queues if _review_queue_has_member(q, username)]
    response_message.ClearField("review_queues")
    response_message.review_queues.extend(visible)
    resp.data = message_to_json(response_message)


BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    CreateExperiment: validate_can_create_experiment,
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
    # Routes for model registry (shared with prompts — dispatch via
    # `_get_permission_from_registered_model_or_prompt_name`).
    CreateRegisteredModel: validate_can_create_registered_model,
    GetRegisteredModel: _validate_can_read_registered_model_or_prompt,
    DeleteRegisteredModel: _validate_can_delete_registered_model_or_prompt,
    UpdateRegisteredModel: _validate_can_update_registered_model_or_prompt,
    RenameRegisteredModel: _validate_can_update_registered_model_or_prompt,
    GetLatestVersions: _validate_can_read_registered_model_or_prompt,
    CreateModelVersion: _validate_can_update_registered_model_or_prompt,
    GetModelVersion: _validate_can_read_registered_model_or_prompt,
    DeleteModelVersion: _validate_can_delete_registered_model_or_prompt,
    UpdateModelVersion: _validate_can_update_registered_model_or_prompt,
    TransitionModelVersionStage: _validate_can_update_registered_model_or_prompt,
    GetModelVersionDownloadUri: _validate_can_read_registered_model_or_prompt,
    SetRegisteredModelTag: _validate_can_update_registered_model_or_prompt,
    DeleteRegisteredModelTag: _validate_can_update_registered_model_or_prompt,
    SetModelVersionTag: _validate_can_update_registered_model_or_prompt,
    DeleteModelVersionTag: _validate_can_delete_registered_model_or_prompt,
    SetRegisteredModelAlias: _validate_can_update_registered_model_or_prompt,
    DeleteRegisteredModelAlias: _validate_can_delete_registered_model_or_prompt,
    GetModelVersionByAlias: _validate_can_read_registered_model_or_prompt,
    # Routes for scorers
    RegisterScorer: validate_can_update_experiment,
    ListScorers: validate_can_read_scorer_list,
    GetScorer: validate_can_read_scorer,
    DeleteScorer: validate_can_delete_scorer,
    ListScorerVersions: validate_can_read_scorer,
    # Routes for gateway secrets
    GetGatewaySecretInfo: validate_can_read_gateway_secret,
    UpdateGatewaySecret: validate_can_update_gateway_secret,
    DeleteGatewaySecret: validate_can_delete_gateway_secret,
    # Routes for gateway endpoints
    CreateGatewayEndpoint: validate_can_create_gateway_endpoint,
    GetGatewayEndpoint: validate_can_read_gateway_endpoint,
    UpdateGatewayEndpoint: validate_can_update_gateway_endpoint,
    DeleteGatewayEndpoint: validate_can_delete_gateway_endpoint,
    # Routes for gateway model definitions
    CreateGatewayModelDefinition: validate_can_create_gateway_model_definition,
    GetGatewayModelDefinition: validate_can_read_gateway_model_definition,
    UpdateGatewayModelDefinition: validate_can_update_gateway_model_definition,
    DeleteGatewayModelDefinition: validate_can_delete_gateway_model_definition,
    # Routes for gateway budget policies
    CreateGatewayBudgetPolicy: sender_is_admin,
    UpdateGatewayBudgetPolicy: sender_is_admin,
    DeleteGatewayBudgetPolicy: sender_is_admin,
    # Routes for gateway endpoint-model mappings
    AttachModelToGatewayEndpoint: validate_can_update_gateway_endpoint,
    DetachModelFromGatewayEndpoint: validate_can_update_gateway_endpoint,
    # Routes for gateway endpoint bindings
    CreateGatewayEndpointBinding: validate_can_update_gateway_endpoint,
    DeleteGatewayEndpointBinding: validate_can_update_gateway_endpoint,
    ListGatewayEndpointBindings: validate_can_read_gateway_endpoint,
    # Routes for gateway endpoint tags
    SetGatewayEndpointTag: validate_can_update_gateway_endpoint,
    DeleteGatewayEndpointTag: validate_can_update_gateway_endpoint,
    # Routes for prompt optimization jobs
    CreatePromptOptimizationJob: validate_can_update_experiment,
    GetPromptOptimizationJob: validate_can_read_prompt_optimization_job,
    SearchPromptOptimizationJobs: validate_can_read_experiment,
    CancelPromptOptimizationJob: validate_can_update_prompt_optimization_job,
    DeletePromptOptimizationJob: validate_can_delete_prompt_optimization_job,
    # Routes for traces
    StartTrace: validate_can_update_experiment,
    StartTraceV3: validate_can_start_trace_v3,
    EndTrace: validate_can_update_trace_by_request_id,
    GetTraceInfo: validate_can_read_trace_by_request_id,
    GetTraceInfoV3: validate_can_read_trace_by_trace_id,
    GetTrace: validate_can_read_trace_by_trace_id,
    SearchTraces: validate_can_search_traces,
    SearchTracesV3: validate_can_search_traces_v3,
    BatchGetTraces: validate_can_batch_get_traces,
    BatchGetTraceInfos: validate_can_batch_get_traces,
    DeleteTraces: validate_can_delete_traces,
    DeleteTracesV3: validate_can_delete_traces,
    SetTraceTag: validate_can_update_trace_by_request_id,
    SetTraceTagV3: validate_can_update_trace_by_trace_id,
    DeleteTraceTag: validate_can_update_trace_by_request_id,
    DeleteTraceTagV3: validate_can_update_trace_by_trace_id,
    LinkTracesToRun: validate_can_link_traces_to_run,
    LinkPromptsToTrace: validate_can_update_trace_by_trace_id,
    CalculateTraceFilterCorrelation: validate_can_read_traces_by_experiment_ids,
    QueryTraceMetrics: validate_can_read_traces_by_experiment_ids,
    CreateAssessment: validate_can_update_trace_by_trace_id,
    GetAssessmentRequest: validate_can_read_trace_by_trace_id,
    UpdateAssessment: validate_can_update_trace_by_trace_id,
    DeleteAssessment: validate_can_update_trace_by_trace_id,
    # Routes for review queues
    CreateReviewQueue: validate_can_create_review_queue,
    GetReviewQueue: validate_can_view_review_queue,
    GetReviewQueueByName: validate_can_view_review_queue_by_name,
    GetOrCreateUserQueue: validate_can_get_or_create_user_queue,
    ListReviewQueues: validate_can_read_experiment,
    UpdateReviewQueue: validate_can_update_review_queue,
    DeleteReviewQueue: validate_can_delete_review_queue,
    AddItemsToReviewQueue: validate_can_add_items_to_review_queue,
    RemoveItemsFromReviewQueue: validate_can_remove_items_from_review_queue,
    ListReviewQueueItems: validate_can_view_review_queue,
    SetReviewQueueItemStatus: validate_can_review_queue_item,
    # Routes for label schemas (review questions)
    CreateLabelSchema: validate_can_create_label_schema,
    GetLabelSchema: validate_can_read_label_schema,
    GetLabelSchemaByName: validate_can_read_experiment,
    ListLabelSchemas: validate_can_read_experiment,
    UpdateLabelSchema: validate_can_manage_label_schema,
    DeleteLabelSchema: validate_can_manage_label_schema,
    # Workspace routes
    ListWorkspaces: None,
    CreateWorkspace: sender_is_admin,
    GetWorkspace: validate_can_view_workspace,
    UpdateWorkspace: sender_is_admin,
    DeleteWorkspace: sender_is_admin,
}


def get_before_request_handler(request_class):
    return BEFORE_REQUEST_HANDLERS.get(request_class)


@functools.lru_cache(maxsize=None)
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
    if "/scorers/online-config" not in http_path
}

# Auth-related routes
BEFORE_REQUEST_VALIDATORS.update({
    (SIGNUP, "GET"): validate_can_create_user,
    (GET_USER, "GET"): validate_can_read_user,
    (AJAX_GET_USER, "GET"): validate_can_read_user,
    # /current returns only the authenticated user's own identity — any
    # authenticated user may read it.
    (GET_CURRENT_USER, "GET"): lambda: True,
    (AJAX_GET_CURRENT_USER, "GET"): lambda: True,
    # Same goes for /current/permissions.
    (LIST_CURRENT_USER_PERMISSIONS, "GET"): lambda: True,
    (AJAX_LIST_CURRENT_USER_PERMISSIONS, "GET"): lambda: True,
    (LIST_USERS, "GET"): validate_can_list_users,
    (AJAX_LIST_USERS, "GET"): validate_can_list_users,
    (CREATE_USER, "POST"): validate_can_create_user,
    (AJAX_CREATE_USER, "POST"): validate_can_create_user,
    (UPDATE_USER_PASSWORD, "PATCH"): validate_can_update_user_password,
    (AJAX_UPDATE_USER_PASSWORD, "PATCH"): validate_can_update_user_password,
    (UPDATE_USER_ADMIN, "PATCH"): validate_can_update_user_admin,
    (AJAX_UPDATE_USER_ADMIN, "PATCH"): validate_can_update_user_admin,
    (DELETE_USER, "DELETE"): validate_can_delete_user,
    (AJAX_DELETE_USER, "DELETE"): validate_can_delete_user,
})

# Role management routes (RBAC)
BEFORE_REQUEST_VALIDATORS.update({
    (CREATE_ROLE, "POST"): validate_can_manage_roles,
    (AJAX_CREATE_ROLE, "POST"): validate_can_manage_roles,
    (GET_ROLE, "GET"): validate_can_view_roles,
    (AJAX_GET_ROLE, "GET"): validate_can_view_roles,
    (LIST_ROLES, "GET"): validate_can_list_roles,
    (AJAX_LIST_ROLES, "GET"): validate_can_list_roles,
    (UPDATE_ROLE, "PATCH"): validate_can_manage_roles,
    (AJAX_UPDATE_ROLE, "PATCH"): validate_can_manage_roles,
    (DELETE_ROLE, "DELETE"): validate_can_manage_roles,
    (AJAX_DELETE_ROLE, "DELETE"): validate_can_manage_roles,
    (ADD_ROLE_PERMISSION, "POST"): validate_can_manage_roles,
    (AJAX_ADD_ROLE_PERMISSION, "POST"): validate_can_manage_roles,
    (REMOVE_ROLE_PERMISSION, "DELETE"): validate_can_manage_roles,
    (AJAX_REMOVE_ROLE_PERMISSION, "DELETE"): validate_can_manage_roles,
    (LIST_ROLE_PERMISSIONS, "GET"): validate_can_view_roles,
    (AJAX_LIST_ROLE_PERMISSIONS, "GET"): validate_can_view_roles,
    (UPDATE_ROLE_PERMISSION, "PATCH"): validate_can_manage_roles,
    (AJAX_UPDATE_ROLE_PERMISSION, "PATCH"): validate_can_manage_roles,
    (ASSIGN_ROLE, "POST"): validate_can_manage_roles,
    (AJAX_ASSIGN_ROLE, "POST"): validate_can_manage_roles,
    (UNASSIGN_ROLE, "DELETE"): validate_can_manage_roles,
    (AJAX_UNASSIGN_ROLE, "DELETE"): validate_can_manage_roles,
    (LIST_USER_ROLES, "GET"): validate_can_view_user_roles,
    (AJAX_LIST_USER_ROLES, "GET"): validate_can_view_user_roles,
    # Same authorization shape as ``LIST_USER_ROLES``: super admins
    # bypass; self can view their own grants; workspace admins can
    # view grants for users in workspaces they administer.
    (LIST_USER_PERMISSIONS, "GET"): validate_can_view_user_roles,
    (AJAX_LIST_USER_PERMISSIONS, "GET"): validate_can_view_user_roles,
    (LIST_ROLE_USERS, "GET"): validate_can_manage_roles,
    (AJAX_LIST_ROLE_USERS, "GET"): validate_can_manage_roles,
    # Unified per-user grant convenience APIs. ``grant`` / ``revoke`` gate on
    # per-resource MANAGE on the target resource (preserving the per-resource
    # MANAGE delegation the legacy endpoints offered, permanently). ``check``
    # uses the same self/admin/workspace-admin shape as ``LIST_USER_ROLES``.
    (GRANT_USER_PERMISSION, "POST"): validate_can_manage_resource,
    (AJAX_GRANT_USER_PERMISSION, "POST"): validate_can_manage_resource,
    (REVOKE_USER_PERMISSION, "POST"): validate_can_manage_resource,
    (AJAX_REVOKE_USER_PERMISSION, "POST"): validate_can_manage_resource,
    (GET_USER_PERMISSION, "GET"): validate_can_get_user_permission,
    (AJAX_GET_USER_PERMISSION, "GET"): validate_can_get_user_permission,
})

# Flask routes (no proto mapping)
BEFORE_REQUEST_VALIDATORS.update({
    (GET_ARTIFACT, "GET"): validate_can_read_run_artifact,
    (UPLOAD_ARTIFACT, "POST"): validate_can_update_run_artifact,
    (GET_MODEL_VERSION_ARTIFACT, "GET"): validate_can_read_model_version_artifact,
    (GET_TRACE_ARTIFACT, "GET"): validate_can_read_trace_artifact,
    (GET_TRACE_ARTIFACT_V3, "GET"): validate_can_read_trace_artifact,
    (GET_METRIC_HISTORY_BULK, "GET"): validate_can_read_metric_history_bulk,
    (GET_METRIC_HISTORY_BULK_INTERVAL, "GET"): validate_can_read_metric_history_bulk_interval,
    (SEARCH_DATASETS, "POST"): validate_can_search_datasets,
    (CREATE_PROMPTLAB_RUN, "POST"): validate_can_create_promptlab_run,
    (GATEWAY_PROXY, "GET"): validate_gateway_proxy,
    (GATEWAY_PROXY, "POST"): validate_gateway_proxy,
    (INVOKE_SCORER, "POST"): validate_gateway_proxy,
})

# Trace endpoints with path parameters (e.g. /mlflow/traces/<request_id>/tags) require
# regex matching — the BEFORE_REQUEST_VALIDATORS exact-match lookup won't find them when
# the real request path contains an actual trace/request ID instead of the template name.
TRACE_PARAMETERIZED_BEFORE_REQUEST_VALIDATORS = {
    (_re_compile_path(path), method): handler
    for (path, method), handler in BEFORE_REQUEST_VALIDATORS.items()
    if "<" in path and "/mlflow/traces/" in path
}

LOGGED_MODEL_BEFORE_REQUEST_HANDLERS = {
    CreateLoggedModel: validate_can_update_experiment,
    GetLoggedModel: validate_can_read_logged_model,
    DeleteLoggedModel: validate_can_delete_logged_model,
    FinalizeLoggedModel: validate_can_update_logged_model,
    DeleteLoggedModelTag: validate_can_delete_logged_model,
    SetLoggedModelTags: validate_can_update_logged_model,
    ListLoggedModelArtifacts: validate_can_read_logged_model,
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
# The AJAX artifact download endpoint is a plain Flask route with a path parameter, so it
# can't go in routes.py/BEFORE_REQUEST_VALIDATORS (exact match) and must be added here.
LOGGED_MODEL_BEFORE_REQUEST_VALIDATORS[
    (
        _re_compile_path(_get_ajax_path("/mlflow/logged-models/<model_id>/artifacts/files")),
        "GET",
    )
] = validate_can_read_logged_model

WEBHOOK_BEFORE_REQUEST_HANDLERS = {
    CreateWebhook: sender_is_admin,
    GetWebhook: sender_is_admin,
    ListWebhooks: sender_is_admin,
    UpdateWebhook: sender_is_admin,
    DeleteWebhook: sender_is_admin,
    TestWebhook: sender_is_admin,
}


def get_webhook_before_request_handler(request_class):
    return WEBHOOK_BEFORE_REQUEST_HANDLERS.get(request_class)


WEBHOOK_BEFORE_REQUEST_VALIDATORS = {
    # Paths for webhooks contain path parameters (e.g. /mlflow/webhooks/<webhook_id>)
    (_re_compile_path(http_path), method): handler
    for http_path, handler, methods in get_service_endpoints(
        WebhookService, get_webhook_before_request_handler
    )
    for method in methods
}

_AJAX_API_PATH_PREFIX = "/ajax-api/2.0"


_AJAX_API_PATH_PREFIX = "/ajax-api/2.0"


def _is_proxy_artifact_path(path: str) -> bool:
    # MlflowArtifactsService endpoints are registered at both /api/2.0/... and /ajax-api/2.0/...
    # paths (see handlers._get_paths), so we need to check both prefixes for auth validation.
    prefixes = [
        f"{_REST_API_PATH_PREFIX}/mlflow-artifacts/artifacts",
        f"{_AJAX_API_PATH_PREFIX}/mlflow-artifacts/artifacts",
        f"{_REST_API_PATH_PREFIX}/mlflow-artifacts/mpu/",
        f"{_AJAX_API_PATH_PREFIX}/mlflow-artifacts/mpu/",
    ]
    return any(path.startswith(prefix) for prefix in prefixes)


def _get_proxy_artifact_validator(
    method: str, view_args: dict[str, Any] | None
) -> Callable[[], bool] | None:
    if view_args is None:
        return validate_can_read_experiment_artifact_proxy  # List

    return {
        "GET": validate_can_read_experiment_artifact_proxy,  # Download
        "PUT": validate_can_update_experiment_artifact_proxy,  # Upload
        "DELETE": validate_can_delete_experiment_artifact_proxy,  # Delete
        "POST": validate_can_update_experiment_artifact_proxy,  # Multipart upload
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
    # When the cache is disabled, don't pay the extra get_user round-trip that
    # _authenticate_cached does for the sake of cache-population — the Flask
    # path only cares about the yes/no auth decision.
    if _USER_AUTH_CACHE is None:
        if store.authenticate_user(username, password):
            return request.authorization
    elif _authenticate_cached(username, password):
        return request.authorization
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

    if "/mlflow/webhooks" in req.path:
        # Webhook routes contain path parameters (e.g., /mlflow/webhooks/<webhook_id>)
        # so we need regex matching
        return next(
            (
                v
                for (pat, method), v in WEBHOOK_BEFORE_REQUEST_VALIDATORS.items()
                if pat.fullmatch(req.path) and method == req.method
            ),
            None,
        )

    if validator := BEFORE_REQUEST_VALIDATORS.get((req.path, req.method)):
        return validator

    # Trace routes with path parameters (e.g. /mlflow/traces/<request_id>/tags).
    # Unknown paths under this prefix are denied (fail-closed) rather than skipped.
    if "/mlflow/traces/" in req.path:
        validator = next(
            (
                v
                for (pat, method), v in TRACE_PARAMETERIZED_BEFORE_REQUEST_VALIDATORS.items()
                if pat.fullmatch(req.path) and method == req.method
            ),
            None,
        )
        return validator if validator is not None else lambda: False

    return None


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

    # Expose the authenticated user to handlers (e.g. to stamp a review-queue owner).
    g.mlflow_authenticated_user = authorization.username

    # admins don't need to be authorized, but data-integrity rules still apply to
    # them. A custom review queue (or rename) may not shadow a username; admins skip
    # the validators, so the guard runs here for them (non-admins hit it post-gate
    # inside the validators).
    if sender_is_admin():
        enforce_review_queue_name_not_username()
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
    store.grant_user_permission(username, "experiment", experiment_id, MANAGE.name)


def set_can_manage_registered_model_permission(resp: Response):
    # ``CreateRegisteredModel`` is shared with prompt creation; the response
    # carries the persisted ``mlflow.prompt.is_prompt`` tag, so we can classify
    # the entity authoritatively here and grant MANAGE in the matching
    # namespace. Granting unconditionally on ``registered_model`` would leave
    # the prompt creator without ``(prompt, name, MANAGE)`` and lock them out
    # of the entity they just created via the prompt-side validators.
    response_message = CreateRegisteredModel.Response()
    parse_dict(resp.json, response_message)
    name = response_message.registered_model.name
    resource_type = (
        "prompt" if _entity_is_prompt(response_message.registered_model) else "registered_model"
    )
    username = authenticate_request().username
    store.grant_user_permission(username, resource_type, name, MANAGE.name)


def delete_can_manage_registered_model_permission(resp: Response):
    """
    Sweep registered-model and prompt grants when the entity is deleted.

    The model registry's primary key is the entity name (unlike experiments
    which use a UUID), so a future entity with the same name would otherwise
    inherit stale grants. ``DeleteRegisteredModel`` is shared between
    registered models and prompts on the REST surface; the entity is already
    gone by the time this after-request hook runs, so we cannot classify it
    now. Names are unique within the registry, so exactly one of the two
    sweeps applies and the other is a no-op.
    """
    # ``silent=True`` returns ``None`` on missing / unparsable bodies; the
    # ``or {}`` guard prevents a ``TypeError`` from leaking out as a 500.
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name")
    if not name:
        raise MlflowException(
            "Missing value for required parameter 'name'.",
            INVALID_PARAMETER_VALUE,
        )
    store.delete_grants_for_resource("registered_model", name, workspace_scoped=True)
    store.delete_grants_for_resource("prompt", name, workspace_scoped=True)


# ---- Role management handlers (RBAC) ----


@catch_mlflow_exception
def create_role():
    name = _get_request_param("name")
    workspace = _get_request_param("workspace")
    if not isinstance(name, str) or not name.strip():
        raise MlflowException.invalid_parameter_value("Role name cannot be empty.")
    if not isinstance(workspace, str) or not workspace.strip():
        raise MlflowException.invalid_parameter_value("Workspace cannot be empty.")
    body = request.get_json(silent=True) or {}
    description = body.get("description")
    if description is not None and not isinstance(description, str):
        raise MlflowException.invalid_parameter_value("Role description must be a string or null.")
    role = store.create_role(name, workspace, description)
    return jsonify({"role": role.to_json()})


@catch_mlflow_exception
def get_role():
    role_id = _get_int_request_param("role_id")
    role = store.get_role(role_id)
    return jsonify({"role": role.to_json()})


@catch_mlflow_exception
def list_roles():
    # Repeated ``workspace`` scopes the listing. When omitted, fall back to cross-
    # workspace listing (admin-only — enforced by validate_can_list_roles).
    workspaces = request.args.getlist("workspace")
    for w in workspaces:
        if not isinstance(w, str) or not w.strip():
            raise MlflowException.invalid_parameter_value(
                "Parameter 'workspace' must be a non-empty string when provided."
            )
    roles = store.list_roles(workspaces) if workspaces else store.list_roles()
    return jsonify({"roles": [r.to_json() for r in roles]})


@catch_mlflow_exception
def update_role():
    role_id = _get_int_request_param("role_id")
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    description = body.get("description")
    if name is None and description is None:
        raise MlflowException.invalid_parameter_value(
            "At least one of 'name' or 'description' must be provided to update a role."
        )
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise MlflowException.invalid_parameter_value("Role name cannot be empty.")
    if description is not None and not isinstance(description, str):
        raise MlflowException.invalid_parameter_value("Role description must be a string.")
    role = store.update_role(role_id, name=name, description=description)
    return jsonify({"role": role.to_json()})


@catch_mlflow_exception
def delete_role():
    role_id = _get_int_request_param("role_id")
    store.delete_role(role_id)
    return make_response({})


@catch_mlflow_exception
def add_role_permission():
    role_id = _get_int_request_param("role_id")
    resource_type = _get_request_param("resource_type")
    resource_pattern = _get_request_param("resource_pattern")
    permission = _get_request_param("permission")
    rp = store.add_role_permission(role_id, resource_type, resource_pattern, permission)
    return jsonify({"role_permission": rp.to_json()})


@catch_mlflow_exception
def remove_role_permission():
    role_permission_id = _get_int_request_param("role_permission_id")
    store.remove_role_permission(role_permission_id)
    return make_response({})


@catch_mlflow_exception
def list_role_permissions():
    role_id = _get_int_request_param("role_id")
    perms = store.list_role_permissions(role_id)
    return jsonify({"role_permissions": [p.to_json() for p in perms]})


@catch_mlflow_exception
def update_role_permission():
    role_permission_id = _get_int_request_param("role_permission_id")
    permission = _get_request_param("permission")
    rp = store.update_role_permission(role_permission_id, permission)
    return jsonify({"role_permission": rp.to_json()})


@catch_mlflow_exception
def assign_role():
    username = _get_request_param("username")
    role_id = _get_int_request_param("role_id")
    user = store.get_user(username)
    assignment = store.assign_role_to_user(user.id, role_id)
    return jsonify({"assignment": assignment.to_json()})


@catch_mlflow_exception
def unassign_role():
    username = _get_request_param("username")
    role_id = _get_int_request_param("role_id")
    user = store.get_user(username)
    store.unassign_role_from_user(user.id, role_id)
    return make_response({})


@catch_mlflow_exception
def list_user_roles():
    username = _get_request_param("username")
    user = store.get_user(username)
    roles = store.list_user_roles(user.id)

    # Filter the response to match the caller's authorization scope so we don't leak
    # role/workspace membership outside what the caller can see:
    # - Self or super admin: see all of the target's roles.
    # - Workspace admin: see only roles in workspaces where the caller is a WP admin.
    # Fetch the requester's admin workspaces once rather than querying per role.
    requester = authenticate_request().username
    requester_user = store.get_user(requester)
    if not (requester_user.is_admin or requester == username):
        admin_workspaces = store.list_workspace_admin_workspaces(requester_user.id)
        roles = [r for r in roles if r.workspace in admin_workspaces]

    return jsonify({"roles": [r.to_json() for r in roles]})


@catch_mlflow_exception
def list_role_users():
    role_id = _get_int_request_param("role_id")
    assignments = store.list_role_users(role_id)
    return jsonify({"assignments": [a.to_json() for a in assignments]})


def filter_list_workspaces(resp: Response) -> None:
    if sender_is_admin():
        return

    username = authenticate_request().username
    response_message = ListWorkspaces.Response()
    parse_dict(resp.json, response_message)

    allowed: set[str] = set()
    if username is not None:
        allowed = set(store.list_accessible_workspace_names(username))
        if auth_config.grant_default_workspace_access:
            default_workspace, _ = get_default_workspace_optional(_get_workspace_store())
            if default_workspace:
                allowed.add(default_workspace.name)

    filtered = [ws for ws in response_message.workspaces if ws.name in allowed]
    response_message.ClearField("workspaces")
    response_message.workspaces.extend(filtered)

    resp.data = message_to_json(response_message)


# Default roles seeded into every new workspace when
# ``MLFLOW_RBAC_SEED_DEFAULT_ROLES`` is on. ``CreateWorkspace`` is gated to
# ``sender_is_admin``, so the creator is always a super-admin whose ``is_admin``
# flag already bypasses RBAC checks — we therefore don't assign the creator to
# any of these roles. The two roles exist as ready-made scaffolding for the
# admin to hand out to other users.
#
# Workspace permissions in the simplified model collapse to two tiers, both
# stored in the unified ``resource_type='workspace'`` slot:
# - ``admin`` (MANAGE on ``('workspace', '*')``): full authority, including
#   role/user management within the workspace.
# - ``user`` (USE on ``('workspace', '*')``): read every resource in the workspace
#   plus the ability to create new experiments / registered models. The
#   creator-as-owner mechanism then grants the creator MANAGE on what they
#   create — so users manage their own resources without gaining write or
#   delete on resources owned by others.
_WORKSPACE_GRANT = ("workspace", "*")
_DEFAULT_WORKSPACE_ROLES = (
    (
        "admin",
        _WORKSPACE_GRANT,
        MANAGE.name,
        "Full MANAGE authority over the workspace.",
    ),
    (
        "user",
        _WORKSPACE_GRANT,
        USE.name,
        (
            "Read every resource in the workspace and create new experiments "
            "and registered models; the creator-as-owner mechanism grants "
            "MANAGE on what you create, with no write or delete access to "
            "resources owned by other users."
        ),
    ),
)


def _seed_default_workspace_roles(resp: Response) -> None:
    """After a successful ``CreateWorkspace``, seed default RBAC roles into the new
    workspace. Partial failures are logged rather than raised — the workspace
    creation has already succeeded at this point.

    No creator-assignment logic: the ``before_request`` handler gates
    ``CreateWorkspace`` to super-admins (via ``sender_is_admin``), and super-admins
    already bypass RBAC checks via their ``is_admin`` flag. The seeded roles are
    therefore a convenience for the admin to hand out to other users, not
    something the creator needs assigned to themselves.
    """
    if not MLFLOW_RBAC_SEED_DEFAULT_ROLES.get():
        return

    response_message = CreateWorkspace.Response()
    parse_dict(resp.json, response_message)
    workspace_name = response_message.workspace.name

    for role_name, (
        resource_type,
        resource_pattern,
    ), permission, description in _DEFAULT_WORKSPACE_ROLES:
        try:
            role = store.create_role(
                name=role_name, workspace=workspace_name, description=description
            )
        except MlflowException as e:
            _logger.error(
                "Failed to create default role '%s' for workspace '%s': %s",
                role_name,
                workspace_name,
                e,
            )
            continue
        try:
            store.add_role_permission(role.id, resource_type, resource_pattern, permission)
        except MlflowException as e:
            _logger.error(
                "Failed to add permission to default role '%s' for workspace '%s': %s. "
                "Rolling back the orphan role.",
                role_name,
                workspace_name,
                e,
            )
            # Remove the orphan role so the workspace doesn't end up with a named
            # role that grants nothing. Best-effort: log on failure.
            try:
                store.delete_role(role.id)
            except MlflowException as delete_err:
                _logger.error(
                    "Failed to roll back orphan role '%s' (id=%s) for workspace '%s': %s",
                    role_name,
                    role.id,
                    workspace_name,
                    delete_err,
                )


def _cleanup_workspace_permissions(resp: Response) -> None:
    # This handler runs only on successful DELETE responses. Cleanup failures are logged
    # instead of raised because the workspace deletion has already succeeded at this point.
    workspace_name = request.view_args.get("workspace_name") if request.view_args else None
    if not workspace_name:
        return

    try:
        store.delete_workspace_permissions_for_workspace(workspace_name)
    except MlflowException as e:
        _logger.error(
            "Failed to delete workspace permissions for workspace '%s': %s",
            workspace_name,
            e,
        )

    try:
        store.delete_roles_for_workspace(workspace_name)
    except MlflowException as e:
        _logger.error(
            "Failed to delete roles for workspace '%s': %s",
            workspace_name,
            e,
        )


def filter_search_experiments(resp: Response):
    if sender_is_admin():
        return

    response_message = SearchExperiments.Response()
    parse_dict(resp.json, response_message)

    username = authenticate_request().username
    can_read = _role_based_read_predicate(username, "experiment")
    # filter out unreadable
    for e in list(response_message.experiments):
        if not can_read(e.experiment_id):
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

        refetched_readable_proto = [e.to_proto() for e in refetched if can_read(e.experiment_id)]
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

    username = authenticate_request().username
    can_read = _role_based_read_predicate(username, "experiment")
    # Remove unreadable models
    for m in list(response_proto.models):
        if not can_read(m.info.experiment_id):
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
            if not can_read(model.experiment_id):
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

    username = authenticate_request().username
    # The registered-model REST surface is shared with prompts; classify each
    # row by its ``mlflow.prompt.is_prompt`` tag and check the correct grant
    # namespace. Without this, a user holding only a ``(prompt, foo, READ)``
    # grant would have prompt ``foo`` silently filtered out of the response.
    can_read = _rm_or_prompt_read_predicate(username)

    # filter out unreadable
    for rm in list(response_message.registered_models):
        if not can_read(rm):
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

        # ``can_read`` accepts both protos and ORM entities; reuse it here so
        # refetched ORM rows go through the same classification as the initial
        # JSON-parsed proto rows above.
        refetched_readable_proto = [rm.to_proto() for rm in refetched if can_read(rm)]
        response_message.registered_models.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)


def filter_search_model_versions(resp: Response):
    if sender_is_admin():
        return

    response_message = SearchModelVersions.Response()
    parse_dict(resp.json, response_message)

    username = authenticate_request().username
    # Prompt versions and model versions share the same REST surface; classify
    # each row by its ``mlflow.prompt.is_prompt`` tag so a prompt-version
    # carrying a ``(prompt, name, READ)`` grant isn't dropped on the floor.
    can_read = _rm_or_prompt_read_predicate(username)

    # filter out model versions whose parent model is unreadable
    for mv in list(response_message.model_versions):
        if not can_read(mv):
            response_message.model_versions.remove(mv)

    resp.data = message_to_json(response_message)


def rename_registered_model_permission(resp: Response):
    """
    Propagate a registered-model rename to RBAC grants.

    ``RenameRegisteredModel`` is shared between registered models and prompts;
    sweep both namespaces so a prompt rename doesn't orphan its
    ``(prompt, old_name, ...)`` grants. Names are unique within the registry,
    so exactly one of the two renames applies and the other is a no-op.
    """
    # ``silent=True`` returns ``None`` on missing / unparsable bodies; ``or
    # {}`` plus the explicit value checks below prevent ``None`` from
    # propagating to ``resource_pattern`` and silently rewriting rows.
    data = request.get_json(force=True, silent=True) or {}
    old_name = data.get("name")
    new_name = data.get("new_name")
    if not old_name or not new_name:
        raise MlflowException(
            "Missing value for required parameter 'name' or 'new_name'.",
            INVALID_PARAMETER_VALUE,
        )
    store.rename_grants_for_resource("registered_model", old_name, new_name, workspace_scoped=True)
    store.rename_grants_for_resource("prompt", old_name, new_name, workspace_scoped=True)


def set_can_manage_scorer_permission(resp: Response):
    response_message = RegisterScorer.Response()
    parse_dict(resp.json, response_message)
    experiment_id = response_message.experiment_id
    name = response_message.name
    username = authenticate_request().username
    # ``grant_user_permission`` is upsert, so re-registration is a no-op
    # rather than an error — no try/except needed.
    pattern = store._scorer_pattern(experiment_id, name)
    store.grant_user_permission(username, "scorer", pattern, MANAGE.name)


def delete_scorer_permissions_cascade(resp: Response):
    data = request.get_json(force=True, silent=True)
    experiment_id = data.get("experiment_id")
    name = data.get("name")
    if experiment_id and name:
        pattern = store._scorer_pattern(experiment_id, name)
        store.delete_grants_for_resource("scorer", pattern)


def set_can_manage_gateway_secret_permission(resp: Response):
    response_message = CreateGatewaySecret.Response()
    parse_dict(resp.json, response_message)
    secret_id = response_message.secret.secret_id
    username = authenticate_request().username
    store.grant_user_permission(username, "gateway_secret", secret_id, MANAGE.name)


def delete_gateway_secret_permissions_cascade(resp: Response):
    data = request.get_json(force=True, silent=True)
    if secret_id := data.get("secret_id"):
        store.delete_grants_for_resource("gateway_secret", secret_id)


def set_can_manage_gateway_endpoint_permission(resp: Response):
    response_message = CreateGatewayEndpoint.Response()
    parse_dict(resp.json, response_message)
    endpoint_id = response_message.endpoint.endpoint_id
    username = authenticate_request().username
    store.grant_user_permission(username, "gateway_endpoint", endpoint_id, MANAGE.name)


def delete_gateway_endpoint_permissions_cascade(resp: Response):
    data = request.get_json(force=True, silent=True)
    if endpoint_id := data.get("endpoint_id"):
        store.delete_grants_for_resource("gateway_endpoint", endpoint_id)


def set_can_manage_gateway_model_definition_permission(resp: Response):
    response_message = CreateGatewayModelDefinition.Response()
    parse_dict(resp.json, response_message)
    model_definition_id = response_message.model_definition.model_definition_id
    username = authenticate_request().username
    store.grant_user_permission(
        username, "gateway_model_definition", model_definition_id, MANAGE.name
    )


def delete_gateway_model_definition_permissions_cascade(resp: Response):
    data = request.get_json(force=True, silent=True)
    if model_definition_id := data.get("model_definition_id"):
        store.delete_grants_for_resource("gateway_model_definition", model_definition_id)


def filter_list_scorers(resp: Response) -> None:
    """Filter cross-experiment ``ListScorers`` responses to rows the caller can read.

    Single-experiment requests are already gated by ``validate_can_read_scorer_list``
    (which delegates to ``validate_can_read_experiment``); cross-experiment requests
    (empty ``experiment_id``) skip that gate so the response can carry scorers from
    multiple experiments. This filter applies the experiment + scorer read
    predicates per row so the picker doesn't leak names the caller has no grant on.
    """
    if sender_is_admin():
        return

    response_message = ListScorers.Response()
    parse_dict(resp.json, response_message)

    username = authenticate_request().username
    can_read_experiment = _role_based_read_predicate(username, "experiment")
    can_read_scorer = _role_based_read_predicate(username, "scorer")
    for scorer in list(response_message.scorers):
        exp_id = str(scorer.experiment_id)
        if not can_read_experiment(exp_id):
            response_message.scorers.remove(scorer)
            continue
        if not can_read_scorer(store._scorer_pattern(exp_id, scorer.scorer_name)):
            response_message.scorers.remove(scorer)
    resp.data = message_to_json(response_message)


AFTER_REQUEST_PATH_HANDLERS = {
    CreateExperiment: set_can_manage_experiment_permission,
    CreateRegisteredModel: set_can_manage_registered_model_permission,
    DeleteRegisteredModel: delete_can_manage_registered_model_permission,
    SearchExperiments: filter_search_experiments,
    SearchLoggedModels: filter_search_logged_models,
    SearchModelVersions: filter_search_model_versions,
    SearchRegisteredModels: filter_search_registered_models,
    RenameRegisteredModel: rename_registered_model_permission,
    RegisterScorer: set_can_manage_scorer_permission,
    ListScorers: filter_list_scorers,
    DeleteScorer: delete_scorer_permissions_cascade,
    ListReviewQueues: filter_list_review_queues,
    CreateGatewaySecret: set_can_manage_gateway_secret_permission,
    DeleteGatewaySecret: delete_gateway_secret_permissions_cascade,
    CreateGatewayEndpoint: set_can_manage_gateway_endpoint_permission,
    DeleteGatewayEndpoint: delete_gateway_endpoint_permissions_cascade,
    CreateGatewayModelDefinition: set_can_manage_gateway_model_definition_permission,
    DeleteGatewayModelDefinition: delete_gateway_model_definition_permissions_cascade,
    ListWorkspaces: filter_list_workspaces,
    CreateWorkspace: _seed_default_workspace_roles,
    DeleteWorkspace: _cleanup_workspace_permissions,
}


def get_after_request_handler(request_class):
    return AFTER_REQUEST_PATH_HANDLERS.get(request_class)


_AJAX_GATEWAY_PATHS = frozenset([
    GATEWAY_SUPPORTED_PROVIDERS,
    GATEWAY_SUPPORTED_MODELS,
    GATEWAY_PROVIDER_CONFIG,
    GATEWAY_SECRETS_CONFIG,
    INVOKE_SCORER,
])

AFTER_REQUEST_HANDLERS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_after_request_handler)
    for method in methods
    if handler is not None
    and "/graphql" not in http_path
    and "/scorers/online-config" not in http_path
    and "/mlflow/server-info" not in http_path
    and http_path not in _AJAX_GATEWAY_PATHS
}

# Precompile workspace parameterized paths for after-request handlers.
WORKSPACE_PARAMETERIZED_AFTER_REQUEST_HANDLERS = {
    (_re_compile_path(path), method): handler
    for (path, method), handler in AFTER_REQUEST_HANDLERS.items()
    if "<" in path and "/workspaces/" in path
}


@catch_mlflow_exception
def _after_request(resp: Response):
    if 400 <= resp.status_code < 600:
        return resp

    handler = AFTER_REQUEST_HANDLERS.get((request.path, request.method))
    if handler is None and "/workspaces/" in request.path:
        # Fallback to regex matching for workspace paths.
        for (path, method), candidate in WORKSPACE_PARAMETERIZED_AFTER_REQUEST_HANDLERS.items():
            if method != request.method:
                continue
            if path.fullmatch(request.path):
                handler = candidate
                break

    if handler is not None:
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


# Must match the admin_password shipped in mlflow/server/auth/basic_auth.ini.
_DEFAULT_ADMIN_PASSWORD = "password1234"


def _warn_if_default_admin_password(password):
    if password == _DEFAULT_ADMIN_PASSWORD:
        _logger.warning(
            "The MLflow basic auth admin account is using the default password shipped "
            "in basic_auth.ini. Change it before exposing this server beyond localhost. "
            "To override, set the MLFLOW_AUTH_CONFIG_PATH environment variable to point "
            "to a custom basic_auth.ini with a non-default admin_password, or update the "
            f"password via {UPDATE_USER_PASSWORD} after startup."
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
    if not request.is_json:
        return make_response("Invalid content type. Must be application/json", 400)

    username = _get_request_param("username")
    password = _get_request_param("password")

    if not username or not password:
        return make_response("Username and password cannot be empty.", 400)

    user = store.create_user(username, password)
    return jsonify({"user": user.to_json()})


@catch_mlflow_exception
def get_user():
    username = _get_request_param("username")
    user = store.get_user(username)
    return jsonify({"user": user.to_json()})


@catch_mlflow_exception
def list_users():
    # Eager-load each user's roles in one batch so the admin Users tab doesn't
    # have to fan out per-user requests. Per-user roles are scoped to what the
    # caller is allowed to see, mirroring ``list_user_roles``:
    # - super admin / self → every role
    # - workspace admin → roles in workspaces they administer
    users_with_roles = store.list_users_with_roles()
    requester = authenticate_request().username
    requester_user = store.get_user(requester)
    admin_workspaces: set[str] | None = (
        None
        if requester_user.is_admin
        else store.list_workspace_admin_workspaces(requester_user.id)
    )
    response_users = []
    for u, roles in users_with_roles:
        if admin_workspaces is None or u.username == requester:
            visible_roles = roles
        else:
            visible_roles = [r for r in roles if r.workspace in admin_workspaces]
        response_users.append({
            "id": u.id,
            "username": u.username,
            "is_admin": u.is_admin,
            "roles": [r.to_json() for r in visible_roles],
        })
    return jsonify({"users": response_users})


@catch_mlflow_exception
def get_current_user():
    # HTTP Basic Auth doesn't set any identifying cookie, so the frontend has
    # no way to know *which* user the browser authenticated as. This endpoint
    # returns minimal identity info (no password hash, no permission arrays)
    # for the currently authenticated user.
    #
    # ``is_basic_auth`` lets the frontend gate Basic-Auth-only affordances
    # (the logout XHR trick and the change-password modal) on deployments
    # that swap in a custom ``authorization_function``.
    username = authenticate_request().username
    user = store.get_user(username)
    is_basic_auth = auth_config.authorization_function == DEFAULT_AUTHORIZATION_FUNCTION
    return jsonify({
        "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin},
        "is_basic_auth": is_basic_auth,
    })


@dataclass
class _UserRolePermissionRow:
    """One row of ``GET /users/permissions/list``: a single grant on one of the
    user's roles, enriched with role identity so the frontend can split the
    "Direct permissions" view (rows whose ``role_name`` matches the synthetic
    ``__user_<id>__`` pattern) from the "Role permissions" view.
    """

    role_id: int
    role_name: str
    workspace: str
    resource_type: str
    resource_pattern: str
    permission: str


def _list_user_role_permissions(username: str) -> tuple[bool, list[_UserRolePermissionRow]]:
    """Flatten every role-derived permission grant the user holds.

    Returns ``(is_admin, rows)`` where ``rows`` covers all roles the user is
    assigned to (including the synthetic ``__user_<id>__`` role used by the
    ``grant`` / ``revoke`` convenience APIs). The ``workspace`` of each row is
    the role's workspace; because resource-level grants are written into roles
    scoped to the resource's workspace, that's equivalent to the resource's
    workspace.
    """
    user = store.get_user(username)
    rows = [
        _UserRolePermissionRow(
            role_id=role.id,
            role_name=role.name,
            workspace=role.workspace,
            resource_type=rp.resource_type,
            resource_pattern=rp.resource_pattern,
            permission=rp.permission,
        )
        for role in store.list_user_roles(user.id)
        for rp in role.permissions
    ]
    return user.is_admin, rows


@catch_mlflow_exception
def list_current_user_permissions():
    # Sender == target. Returns every permission grant across every role the
    # user holds, plus ``is_admin`` at the top level so the frontend can show
    # admin status without a second call to ``/users/get``.
    username = authenticate_request().username
    is_admin, rows = _list_user_role_permissions(username)
    return jsonify({"is_admin": is_admin, "permissions": [asdict(r) for r in rows]})


@catch_mlflow_exception
def list_user_permissions():
    # Admin / self / workspace-admin-of-target view of one user's permissions
    # across every role they hold. Workspace admins see only rows in workspaces
    # they administer; super admins and self see everything. Mirrors
    # ``list_user_roles``'s response-scoping.
    username = _get_request_param("username")
    is_admin, rows = _list_user_role_permissions(username)

    requester = authenticate_request().username
    requester_user = store.get_user(requester)
    if not (requester_user.is_admin or requester == username):
        admin_workspaces = store.list_workspace_admin_workspaces(requester_user.id)
        rows = [r for r in rows if r.workspace in admin_workspaces]

    return jsonify({"is_admin": is_admin, "permissions": [asdict(r) for r in rows]})


@catch_mlflow_exception
def update_user_password():
    username = _get_request_param("username")
    password = _get_request_param("password")
    # Self-service flows re-assert current_password so a hijacked browser
    # session can't silently rotate it. Admin paths skip this check.
    sender = authenticate_request()
    sender_username = getattr(sender, "username", None)
    if sender_username == username:
        body = request.get_json(silent=True) or {}
        current_password = body.get("current_password")
        if not current_password:
            raise MlflowException(
                "Current password is required when changing your own password.",
                INVALID_PARAMETER_VALUE,
            )
        if not store.authenticate_user(username, current_password):
            raise MlflowException(
                "Current password does not match.",
                INVALID_PARAMETER_VALUE,
            )
        # Self-service only: equality avoids re-running bcrypt, and admin
        # paths skip it so they don't leak a same-vs-different oracle.
        if password == current_password:
            raise MlflowException(
                "New password must differ from the current password.",
                INVALID_PARAMETER_VALUE,
            )
    store.update_user(username, password=password)
    _invalidate_user_auth_cache(username)
    return make_response({})


@catch_mlflow_exception
def update_user_admin():
    username = _get_request_param("username")
    is_admin = _get_request_param("is_admin")
    store.update_user(username, is_admin=is_admin)
    _invalidate_user_auth_cache(username)
    return make_response({})


@catch_mlflow_exception
def delete_user():
    username = _get_request_param("username")
    # Admins cannot delete their own account — like blocking ``root`` from
    # ``userdel root`` on Unix. Without this guard the request still succeeds
    # but leaves the caller in a broken state (browser still has Basic Auth
    # creds for a now-missing user, so every subsequent request 401s).
    #
    # ``authenticate_request()`` can return a ``Response`` (401 challenge)
    # rather than an ``Authorization`` object, so guard the ``.username``
    # access with ``getattr`` — same pattern ``update_user_password`` uses.
    sender = authenticate_request()
    sender_username = getattr(sender, "username", None)
    if username == sender_username:
        raise MlflowException(
            "Users cannot delete their own account. Ask another admin to delete this user instead.",
            BAD_REQUEST,
        )
    store.delete_user(username)
    _invalidate_user_auth_cache(username)
    return make_response({})


# =============================================================================
# Unified per-user permission convenience handlers
# =============================================================================


# Workspace rejection + resource_type / permission validation live on the store
# layer (sqlalchemy_store.grant/revoke_user_resource_permission); the handlers
# defer so both paths share one source of truth.


@catch_mlflow_exception
def grant_user_permission():
    username = _get_request_param("username")
    resource_type = _get_request_param("resource_type")
    resource_id = _get_request_param("resource_id")
    permission = _get_request_param("permission")
    store.get_user(username)
    store.grant_user_resource_permission(username, resource_type, resource_id, permission)
    return make_response({})


@catch_mlflow_exception
def revoke_user_permission():
    username = _get_request_param("username")
    resource_type = _get_request_param("resource_type")
    resource_id = _get_request_param("resource_id")
    store.get_user(username)
    store.revoke_user_resource_permission(username, resource_type, resource_id)
    return make_response({})


@catch_mlflow_exception
def get_user_permission():
    username = _get_request_param("username")
    resource_type = _get_request_param("resource_type")
    resource_id = _get_request_param("resource_id")
    # Unknown *users* and unsupported resource_types raise 4xx; unknown *resources*
    # (e.g. nonexistent experiment_id) intentionally return ``allowed=False`` —
    # matches the deny-by-default semantics of the runtime authorization check.
    store.get_user(username)
    permission = _resolve_user_permission_for_resource(username, resource_type, resource_id)
    # ``allowed`` mirrors ``can_use`` (regular access tier). READ alone is not
    # sufficient — callers needing a different cut inspect ``permission`` directly.
    return make_response(
        GetUserPermissionResult(allowed=permission.can_use, permission=permission.name).to_json()
    )


# =============================================================================
# GraphQL Authorization
# =============================================================================

_auth_initialized = False


def is_auth_enabled() -> bool:
    return _auth_initialized


def _graphql_get_permission_for_experiment(experiment_id: str, username: str) -> Permission:
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _graphql_get_permission_for_run(run_id: str, username: str) -> Permission:
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="experiment",
            resource_key=experiment_id,
            workspace_lookup_id=experiment_id,
            workspace_fetcher=_get_tracking_store().get_experiment,
            workspace_label="experiment",
        ),
    )


def _graphql_get_permission_for_model(model_name: str, username: str) -> Permission:
    return _get_role_permission_or_default(
        _role_permission_for(
            username=username,
            resource_type="registered_model",
            resource_key=model_name,
            workspace_lookup_id=model_name,
            workspace_fetcher=_get_model_registry_store().get_registered_model,
            workspace_label="registered model",
        ),
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
        "mlflowSearchModelVersions",
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

        result = next(root, info, **args)
        return self._post_resolve(field_name, result, username) if result is not None else None

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

    def _post_resolve(self, field_name: str, result, username: str):
        """Apply post-resolution filtering on GraphQL results."""
        if field_name == "mlflowSearchModelVersions":
            return self._filter_model_versions_result(result, username)
        return result

    def _filter_model_versions_result(self, result, username: str):
        """Filter model versions the user doesn't have read access to."""
        can_read = _role_based_read_predicate(username, "registered_model")
        if hasattr(result, "model_versions") and result.model_versions is not None:
            filtered = [mv for mv in result.model_versions if can_read(mv.name)]
            del result.model_versions[:]
            result.model_versions.extend(filtered)
        return result


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


# Routes that need request body to extract endpoint name for validation
_ROUTES_NEEDING_BODY = frozenset((
    "/gateway/mlflow/v1/chat/completions",
    "/gateway/openai/v1/chat/completions",
    "/gateway/openai/v1/embeddings",
    "/gateway/openai/v1/responses",
    "/gateway/anthropic/v1/messages",
))


def _authenticate_fastapi_request(request: StarletteRequest) -> User | None:
    """
    Authenticate request using Basic Auth.

    External clients send real username/password credentials. Server-spawned job
    subprocesses (e.g., online scoring) send the internal gateway token as the
    password; when it matches, the user is trusted without calling
    ``store.authenticate_user()``.

    Args:
        request: The Starlette/FastAPI Request object.

    Returns:
        User object if authentication succeeds, None otherwise.
    """
    if "Authorization" not in request.headers:
        return None

    request_path = get_routed_asgi_path(request)
    auth = request.headers["Authorization"]
    try:
        scheme, credentials = auth.split()
        if scheme.lower() != "basic":
            return None
        decoded = base64.b64decode(credentials).decode("ascii")
        username, _, password = decoded.partition(":")

        # Check if this is a trusted internal request from a job subprocess.
        # The server generates a random token at startup and passes it to workers
        # via _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN. When the password matches that
        # token, we trust the username without calling store.authenticate_user().
        # Restrict to /gateway/ routes only so the token cannot be used as a
        # master password on other endpoints (e.g. /v1/traces, /ajax-api/).
        internal_token = _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.get()
        if (
            internal_token
            and request_path.startswith("/gateway/")
            and secrets.compare_digest(password, internal_token)
        ):
            return store.get_user(username)

        return _authenticate_cached(username, password)
    except Exception:
        return None


def _extract_gateway_endpoint_name(path: str, body: dict[str, Any] | None) -> str | None:
    """Extract endpoint name from gateway routes."""
    # Pattern 1: /gateway/{endpoint_name}/mlflow/invocations
    if match := re.match(r"^/gateway/([^/]+)/mlflow/invocations$", path):
        return match.group(1)

    # Pattern 2-6: Passthrough routes (endpoint in request body as "model")
    if path in _ROUTES_NEEDING_BODY:
        if body:
            return body.get("model")
        return None

    # Pattern 7-8: Gemini routes (endpoint in URL path)
    if match := re.match(r"^/gateway/gemini/v1beta/models/([^/:]+):generateContent$", path):
        return match.group(1)
    if match := re.match(r"^/gateway/gemini/v1beta/models/([^/:]+):streamGenerateContent$", path):
        return match.group(1)

    # Pattern 9: Raw proxy route (/gateway/proxy/{endpoint_name}/{path:path})
    if match := re.match(r"^/gateway/proxy/([^/]+)/", path):
        return match.group(1)

    return None


def _validate_gateway_use_permission(endpoint_name: str, username: str) -> bool:
    """Check if the user has USE permission on the gateway endpoint."""
    # TODO: we need to query endpoint ID by name from the database.
    # Revisit the mutability of the endpoint name if it causes latency issues.
    try:
        tracking_store = _get_tracking_store()
        endpoint = tracking_store.get_gateway_endpoint(name=endpoint_name)
        endpoint_id = endpoint.endpoint_id

        permission = _get_role_permission_or_default(
            _role_permission_for(
                username=username,
                resource_type="gateway_endpoint",
                resource_key=endpoint_id,
                workspace_lookup_id=endpoint_id,
                workspace_fetcher=lambda eid: _get_tracking_store().get_gateway_endpoint(
                    endpoint_id=eid
                ),
                workspace_label="gateway endpoint",
            ),
        )
        return permission.can_use
    except MlflowException:
        return False


def _get_gateway_validator(path: str) -> Callable[[str, StarletteRequest], Awaitable[bool]] | None:
    """
    Get a validator function for gateway routes.

    Args:
        path: The request path.

    Returns:
        An async validator function that takes (username, request) and returns
        True if authorized, or None if no validation is needed for this route.
    """

    async def validator(username: str, request: StarletteRequest) -> bool:
        body = None
        if path in _ROUTES_NEEDING_BODY:
            try:
                body = await request.json()
                # Cache parsed body in request.state so route handlers can reuse it
                # (request body can only be read once in Starlette/FastAPI)
                request.state.cached_body = body
            except Exception as e:
                raise MlflowException(f"Invalid JSON payload: {e}", error_code=BAD_REQUEST)

        endpoint_name = _extract_gateway_endpoint_name(path, body)
        if endpoint_name is None:
            raise MlflowException("No endpoint name found", error_code=BAD_REQUEST)

        return _validate_gateway_use_permission(endpoint_name, username)

    return validator


def _get_require_authentication_validator() -> Callable[[str, StarletteRequest], Awaitable[bool]]:
    """
    Get a validator that requires authentication but grants access to any authenticated user.

    Returns:
        An async validator function that always returns True.
    """

    async def validator(username: str, request: StarletteRequest) -> bool:
        return True

    return validator


def _get_otel_validator(
    path: str,
) -> Callable[[str, StarletteRequest], Awaitable[bool]]:
    """
    Get a validator for OpenTelemetry trace ingestion routes.
    """

    async def validator(username: str, request: StarletteRequest) -> bool:
        experiment_id = request.headers.get("x-mlflow-experiment-id")
        if not experiment_id:
            raise MlflowException(
                "Missing required header: X-Mlflow-Experiment-Id", error_code=BAD_REQUEST
            )
        return _get_experiment_permission(experiment_id, username).can_update

    return validator


def _find_fastapi_validator(path: str) -> Callable[[str, StarletteRequest], Awaitable[bool]] | None:
    """
    Find the validator for a FastAPI route that bypasses Flask.

    This mirrors the _find_validator pattern used in Flask's _before_request,
    returning a validator function for routes that need permission checks.

    Args:
        path: The request path.

    Returns:
        An async validator function that takes (username, request) and returns
        True if authorized, or None if the route is handled by Flask (WSGI).
    """
    if path.startswith("/gateway/"):
        return _get_gateway_validator(path)

    if path.startswith("/v1/traces"):
        return _get_otel_validator(path)

    if path.startswith("/ajax-api/3.0/jobs"):
        return _get_require_authentication_validator()

    if path.startswith("/ajax-api/3.0/mlflow/assistant"):
        return _get_require_authentication_validator()

    return None


def add_fastapi_permission_middleware(app: FastAPI) -> None:
    """
    Add permission middleware to FastAPI app for routes not handled by Flask.

    This middleware mirrors the high-level logic of ``_before_request`` for routes that are
    served directly by FastAPI (e.g., ``/gateway/`` routes) and thus bypass Flask's
    ``before_request`` hooks. It follows the same authorization flow:

    1. Skip unprotected routes
    2. Find the appropriate validator for the route
    3. Reject if custom authorization_function is configured (not supported for FastAPI routes)
    4. Authenticate the request
    5. Allow admins full access
    6. Run the validator

    Args:
        app: The FastAPI application instance.
    """

    @app.middleware("http")
    async def fastapi_permission_middleware(request, call_next):
        path = get_routed_asgi_path(request)

        # Skip unprotected routes
        if is_unprotected_route(path):
            return await call_next(request)

        # Find validator for this route
        validator = _find_fastapi_validator(path)
        if validator is None:
            return await call_next(request)

        # Check for custom authorization_function (only affects routes with validators)
        if auth_config.authorization_function != DEFAULT_AUTHORIZATION_FUNCTION:
            return PlainTextResponse(
                f"Custom authorization_function '{auth_config.authorization_function}' is not "
                f"supported for FastAPI routes (e.g., /gateway/ endpoints). Only the default "
                f"Basic Auth function is supported. Please use "
                f"'{DEFAULT_AUTHORIZATION_FUNCTION}' or disable the AI Gateway feature.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        # Authenticate user
        user = _authenticate_fastapi_request(request)
        if user is None:
            return PlainTextResponse(
                "You are not authenticated. Please see "
                "https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow "
                "on how to authenticate.",
                status_code=HTTPStatus.UNAUTHORIZED,
                headers={"WWW-Authenticate": 'Basic realm="mlflow"'},
            )

        # Store user info in request state for downstream handlers (e.g., gateway tracing)
        request.state.username = user.username
        request.state.user_id = user.id

        # Admins have full access
        if user.is_admin:
            return await call_next(request)

        # Run the validator
        try:
            if not await validator(user.username, request):
                return PlainTextResponse(
                    "Permission denied",
                    status_code=HTTPStatus.FORBIDDEN,
                )
        except MlflowException as e:
            return PlainTextResponse(
                e.message,
                status_code=e.get_http_status_code(),
            )

        return await call_next(request)


# Role management routes (RBAC). Each route is exposed at both the REST path (Python
# client) and the AJAX path (MLflow frontend). Registration loop lives inside create_app.
_RBAC_ROUTES: list[tuple[Callable[[], Any], str, str, str]] = [
    (create_role, "POST", CREATE_ROLE, AJAX_CREATE_ROLE),
    (get_role, "GET", GET_ROLE, AJAX_GET_ROLE),
    (list_roles, "GET", LIST_ROLES, AJAX_LIST_ROLES),
    (update_role, "PATCH", UPDATE_ROLE, AJAX_UPDATE_ROLE),
    (delete_role, "DELETE", DELETE_ROLE, AJAX_DELETE_ROLE),
    (add_role_permission, "POST", ADD_ROLE_PERMISSION, AJAX_ADD_ROLE_PERMISSION),
    (remove_role_permission, "DELETE", REMOVE_ROLE_PERMISSION, AJAX_REMOVE_ROLE_PERMISSION),
    (list_role_permissions, "GET", LIST_ROLE_PERMISSIONS, AJAX_LIST_ROLE_PERMISSIONS),
    (update_role_permission, "PATCH", UPDATE_ROLE_PERMISSION, AJAX_UPDATE_ROLE_PERMISSION),
    (assign_role, "POST", ASSIGN_ROLE, AJAX_ASSIGN_ROLE),
    (unassign_role, "DELETE", UNASSIGN_ROLE, AJAX_UNASSIGN_ROLE),
    (list_user_roles, "GET", LIST_USER_ROLES, AJAX_LIST_USER_ROLES),
    (list_role_users, "GET", LIST_ROLE_USERS, AJAX_LIST_ROLE_USERS),
    # Unified per-user permission convenience APIs under /mlflow/users/permissions/*.
    # ``grant`` / ``revoke`` mutate state (POST); ``check`` resolves the user's
    # effective permission without touching the DB (GET).
    (grant_user_permission, "POST", GRANT_USER_PERMISSION, AJAX_GRANT_USER_PERMISSION),
    (revoke_user_permission, "POST", REVOKE_USER_PERMISSION, AJAX_REVOKE_USER_PERMISSION),
    (get_user_permission, "GET", GET_USER_PERMISSION, AJAX_GET_USER_PERMISSION),
]


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

    store.init_db(
        auth_config.database_uri,
        read_db_uri=auth_config.read_database_uri,
    )
    create_admin_user(auth_config.admin_username, auth_config.admin_password)
    _warn_if_default_admin_password(auth_config.admin_password)

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
    for rule in [CREATE_USER, AJAX_CREATE_USER]:
        app.add_url_rule(
            rule=rule,
            view_func=create_user,
            methods=["POST"],
        )
    for rule in [GET_USER, AJAX_GET_USER]:
        app.add_url_rule(
            rule=rule,
            view_func=get_user,
            methods=["GET"],
        )
    for rule in [LIST_USERS, AJAX_LIST_USERS]:
        app.add_url_rule(
            rule=rule,
            view_func=list_users,
            methods=["GET"],
        )
    for rule in [GET_CURRENT_USER, AJAX_GET_CURRENT_USER]:
        app.add_url_rule(
            rule=rule,
            view_func=get_current_user,
            methods=["GET"],
        )
    for rule in [LIST_CURRENT_USER_PERMISSIONS, AJAX_LIST_CURRENT_USER_PERMISSIONS]:
        app.add_url_rule(
            rule=rule,
            view_func=list_current_user_permissions,
            methods=["GET"],
        )
    for rule in [LIST_USER_PERMISSIONS, AJAX_LIST_USER_PERMISSIONS]:
        app.add_url_rule(
            rule=rule,
            view_func=list_user_permissions,
            methods=["GET"],
        )
    for rule in [UPDATE_USER_PASSWORD, AJAX_UPDATE_USER_PASSWORD]:
        app.add_url_rule(
            rule=rule,
            view_func=update_user_password,
            methods=["PATCH"],
        )
    for rule in [UPDATE_USER_ADMIN, AJAX_UPDATE_USER_ADMIN]:
        app.add_url_rule(
            rule=rule,
            view_func=update_user_admin,
            methods=["PATCH"],
        )
    for rule in [DELETE_USER, AJAX_DELETE_USER]:
        app.add_url_rule(
            rule=rule,
            view_func=delete_user,
            methods=["DELETE"],
        )
    # Role management routes (RBAC) — see _RBAC_ROUTES at module scope.
    for view_func, method, rest_path, ajax_path in _RBAC_ROUTES:
        for path in (rest_path, ajax_path):
            app.add_url_rule(rule=path, view_func=view_func, methods=[method])

    app.before_request(_before_request)
    app.after_request(_after_request)

    if _MLFLOW_SGI_NAME.get() == "uvicorn":
        fastapi_app = create_fastapi_app(app)
        add_fastapi_permission_middleware(fastapi_app)
        return fastapi_app
    else:
        return app
