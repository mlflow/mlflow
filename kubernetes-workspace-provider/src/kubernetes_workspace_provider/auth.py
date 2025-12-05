"""Kubernetes-backed authorization plugin for the MLflow tracking server."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from typing import Iterable, NamedTuple

import werkzeug
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from flask import Flask, Response, g, has_request_context, request
from kubernetes import client, config
from kubernetes.client import AuthorizationV1Api
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from mlflow.environment_variables import _MLFLOW_SGI_NAME
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.mlflow_artifacts_pb2 import (
    AbortMultipartUpload,
    CompleteMultipartUpload,
    CreateMultipartUpload,
    DeleteArtifact,
    DownloadArtifact,
    UploadArtifact,
)
from mlflow.protos.mlflow_artifacts_pb2 import (
    ListArtifacts as ListArtifactsMlflowArtifacts,
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
    AddDatasetToExperiments,
    BatchGetTraces,
    CalculateTraceFilterCorrelation,
    CreateAssessment,
    CreateDataset,
    CreateExperiment,
    CreateLoggedModel,
    CreateRun,
    CreateWorkspace,
    DeleteAssessment,
    DeleteDataset,
    DeleteDatasetTag,
    DeleteExperiment,
    DeleteExperimentTag,
    DeleteLoggedModel,
    DeleteLoggedModelTag,
    DeleteRun,
    DeleteScorer,
    DeleteTag,
    DeleteTraces,
    DeleteTracesV3,
    DeleteTraceTag,
    DeleteTraceTagV3,
    DeleteWorkspace,
    EndTrace,
    FinalizeLoggedModel,
    GetAssessmentRequest,
    GetDataset,
    GetDatasetExperimentIds,
    GetDatasetRecords,
    GetExperiment,
    GetExperimentByName,
    GetLoggedModel,
    GetMetricHistory,
    GetMetricHistoryBulkInterval,
    GetRun,
    GetScorer,
    GetTraceInfo,
    GetTraceInfoV3,
    GetWorkspace,
    LinkTracesToRun,
    ListArtifacts,
    ListLoggedModelArtifacts,
    ListScorers,
    ListScorerVersions,
    ListWorkspaces,
    LogBatch,
    LogInputs,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogOutputs,
    LogParam,
    RegisterScorer,
    RemoveDatasetFromExperiments,
    RestoreExperiment,
    RestoreRun,
    SearchDatasets,
    SearchEvaluationDatasets,
    SearchExperiments,
    SearchLoggedModels,
    SearchRuns,
    SearchTraces,
    SearchTracesV3,
    SetDatasetTags,
    SetExperimentTag,
    SetLoggedModelTags,
    SetTag,
    SetTraceTag,
    SetTraceTagV3,
    StartTrace,
    StartTraceV3,
    UpdateAssessment,
    UpdateExperiment,
    UpdateRun,
    UpdateWorkspace,
    UpsertDatasetRecords,
)
from mlflow.protos.webhooks_pb2 import (
    CreateWebhook,
    DeleteWebhook,
    GetWebhook,
    ListWebhooks,
    TestWebhook,
    UpdateWebhook,
)
from mlflow.server import app as mlflow_app
from mlflow.server import handlers as mlflow_handlers
from mlflow.server.fastapi_app import FASTAPI_NATIVE_PREFIXES, create_fastapi_app
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR, get_endpoints
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils import workspace_context

if not hasattr(werkzeug, "__version__"):  # pragma: no cover - compatibility shim
    werkzeug.__version__ = "werkzeug"

DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_USERNAME_CLAIM = "sub"
DEFAULT_AUTH_GROUP = "mlflow.kubeflow.org"
DEFAULT_REMOTE_USER_HEADER = "x-remote-user"
DEFAULT_REMOTE_GROUPS_HEADER = "x-remote-groups"
DEFAULT_REMOTE_GROUPS_SEPARATOR = "|"

CACHE_TTL_ENV = "MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS"
USERNAME_CLAIM_ENV = "MLFLOW_K8S_AUTH_USERNAME_CLAIM"
AUTHORIZATION_MODE_ENV = "MLFLOW_K8S_AUTH_AUTHORIZATION_MODE"
REMOTE_USER_HEADER_ENV = "MLFLOW_K8S_AUTH_REMOTE_USER_HEADER"
REMOTE_GROUPS_HEADER_ENV = "MLFLOW_K8S_AUTH_REMOTE_GROUPS_HEADER"
REMOTE_GROUPS_SEPARATOR_ENV = "MLFLOW_K8S_AUTH_REMOTE_GROUPS_SEPARATOR"

_UNPROTECTED_PATH_PREFIXES = ("/static", "/favicon.ico", "/health", "/build")
_UNPROTECTED_PATHS = {
    "/",
    "/metrics",
    "/api/2.0/mlflow/server-features",
    "/ajax-api/2.0/mlflow/server-features",
}
RESOURCE_EXPERIMENTS = "experiments"
RESOURCE_REGISTERED_MODELS = "registeredmodels"
RESOURCE_WORKSPACES = "workspaces"
RESOURCE_JOBS = "jobs"
_ALLOWED_RESOURCES = {
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
    RESOURCE_WORKSPACES,
    RESOURCE_JOBS,
}
_WORKSPACE_PERMISSION_RESOURCE_PRIORITY = (
    RESOURCE_EXPERIMENTS,
    RESOURCE_REGISTERED_MODELS,
    RESOURCE_JOBS,
)
_FASTAPI_AUTH_PREFIXES = (OTLP_TRACES_PATH, "/ajax-api/3.0")

_WORKSPACE_REQUIRED_ERROR_MESSAGE = "Workspace context is required for this request."
_WORKSPACE_MUTATION_DENIED_MESSAGE = (
    "Workspace create, update, and delete operations are not supported when MLflow "
    "workspaces map to Kubernetes namespaces."
)

K8S_GRAPHQL_OPERATION_RESOURCE_MAP = {
    # Experiment / Run surfaces
    "MlflowGetExperimentQuery": RESOURCE_EXPERIMENTS,
    "GetExperiment": RESOURCE_EXPERIMENTS,
    "GetRun": RESOURCE_EXPERIMENTS,
    "MlflowGetRunQuery": RESOURCE_EXPERIMENTS,
    "SearchRuns": RESOURCE_EXPERIMENTS,
    "MlflowSearchRunsQuery": RESOURCE_EXPERIMENTS,
    "GetMetricHistoryBulkInterval": RESOURCE_EXPERIMENTS,
    "MlflowGetMetricHistoryBulkIntervalQuery": RESOURCE_EXPERIMENTS,
    # Model Registry surfaces
    "SearchModelVersions": RESOURCE_REGISTERED_MODELS,
    "MlflowSearchModelVersionsQuery": RESOURCE_REGISTERED_MODELS,
    "GetModelVersion": RESOURCE_REGISTERED_MODELS,
    "MlflowGetModelVersionQuery": RESOURCE_REGISTERED_MODELS,
    "GetRegisteredModel": RESOURCE_REGISTERED_MODELS,
    "MlflowGetRegisteredModelQuery": RESOURCE_REGISTERED_MODELS,
    "SearchRegisteredModels": RESOURCE_REGISTERED_MODELS,
    "MlflowSearchRegisteredModelsQuery": RESOURCE_REGISTERED_MODELS,
}

K8S_GRAPHQL_OPERATION_VERB_MAP: dict[str, str] = {
    # Experiment / Run surfaces
    "MlflowGetExperimentQuery": "get",
    "GetExperiment": "get",
    "GetRun": "get",
    "MlflowGetRunQuery": "get",
    "SearchRuns": "list",
    "MlflowSearchRunsQuery": "list",
    "GetMetricHistoryBulkInterval": "get",
    "MlflowGetMetricHistoryBulkIntervalQuery": "get",
    # Model Registry surfaces
    "SearchModelVersions": "list",
    "MlflowSearchModelVersionsQuery": "list",
    "GetModelVersion": "get",
    "MlflowGetModelVersionQuery": "get",
    "GetRegisteredModel": "get",
    "MlflowGetRegisteredModelQuery": "get",
    "SearchRegisteredModels": "list",
    "MlflowSearchRegisteredModelsQuery": "list",
}


_logger = logging.getLogger(__name__)


class AuthorizationMode(str, Enum):
    SELF_SUBJECT_ACCESS_REVIEW = "self_subject_access_review"
    SUBJECT_ACCESS_REVIEW = "subject_access_review"


class _ReadWriteLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer or self._waiting_writers > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        with self._condition:
            self._writer = False
            self._condition.notify_all()


class _CacheEntry(NamedTuple):
    allowed: bool
    expires_at: float


class _AuthorizationCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[tuple[str, str, str, str], _CacheEntry] = {}
        self._lock = _ReadWriteLock()

    def get(self, key: tuple[str, str, str, str]) -> bool | None:
        observed_expiration: float | None = None

        self._lock.acquire_read()
        try:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at > time.time():
                return entry.allowed
            observed_expiration = entry.expires_at
        finally:
            self._lock.release_read()

        self._lock.acquire_write()
        try:
            current = self._entries.get(key)
            if current is not None:
                if observed_expiration is None or current.expires_at <= observed_expiration:
                    self._entries.pop(key, None)
        finally:
            self._lock.release_write()
        return None

    def set(self, key: tuple[str, str, str, str], allowed: bool) -> None:
        self._lock.acquire_write()
        try:
            self._entries[key] = _CacheEntry(
                allowed=allowed, expires_at=time.time() + self._ttl_seconds
            )
        finally:
            self._lock.release_write()


@dataclass(frozen=True)
class _RequestIdentity:
    token: str | None = None
    user: str | None = None
    groups: tuple[str, ...] = ()

    def subject_hash(
        self,
        mode: AuthorizationMode,
        *,
        missing_user_label: str = "Remote user header",
    ) -> str:
        if mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
            if not self.token:
                raise MlflowException(
                    "Bearer token is required for SelfSubjectAccessReview mode.",
                    error_code=databricks_pb2.UNAUTHENTICATED,
                )
            return sha256(self.token.encode("utf-8")).hexdigest()

        user = (self.user or "").strip()
        if not user:
            raise MlflowException(
                f"{missing_user_label} is required for SubjectAccessReview mode.",
                error_code=databricks_pb2.UNAUTHENTICATED,
            )
        # Use the null byte as a delimiter so user/group names cannot collide accidentally.
        normalized_groups = "\x00".join(sorted(self.groups))
        serialized = "\x00".join([user, normalized_groups])
        return sha256(serialized.encode("utf-8")).hexdigest()


class AuthorizationRule(NamedTuple):
    verb: str | None
    resource: str | None = None
    override_run_user: bool = False
    apply_workspace_filter: bool = False
    requires_workspace: bool = True
    deny: bool = False


def _experiments_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_EXPERIMENTS, **kwargs)


def _registered_models_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_REGISTERED_MODELS, **kwargs)


def _jobs_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_JOBS, **kwargs)


def _workspaces_rule(verb: str | None, **kwargs) -> AuthorizationRule:
    return AuthorizationRule(verb, resource=RESOURCE_WORKSPACES, **kwargs)


def _normalize_resource_name(resource: str | None) -> str | None:
    if resource is None:
        return None
    normalized = resource.replace("_", "")
    if normalized not in _ALLOWED_RESOURCES:
        raise ValueError(f"Unsupported RBAC resource '{resource}'")
    return normalized


_AUTH_RULES: dict[tuple[str, str], AuthorizationRule] = {}
_AUTH_REGEX_RULES: list[tuple[re.Pattern[str], str, AuthorizationRule]] = []
_HANDLER_RULES: dict[object, AuthorizationRule] = {}
_RULES_COMPILED = False


REQUEST_AUTHORIZATION_RULES: dict[type, AuthorizationRule] = {
    # Datasets and assessments
    AddDatasetToExperiments: _experiments_rule("update"),
    CreateAssessment: _experiments_rule("update"),
    CreateDataset: _experiments_rule("update"),
    DeleteAssessment: _experiments_rule("delete"),
    DeleteDataset: _experiments_rule("delete"),
    DeleteDatasetTag: _experiments_rule("delete"),
    GetAssessmentRequest: _experiments_rule("get"),
    GetDataset: _experiments_rule("get"),
    GetDatasetExperimentIds: _experiments_rule("list"),
    GetDatasetRecords: _experiments_rule("list"),
    RemoveDatasetFromExperiments: _experiments_rule("update"),
    SearchDatasets: _experiments_rule("list"),
    SearchEvaluationDatasets: _experiments_rule("list"),
    SetDatasetTags: _experiments_rule("update"),
    UpdateAssessment: _experiments_rule("update"),
    UpsertDatasetRecords: _experiments_rule("update"),
    # Experiments
    CreateExperiment: _experiments_rule("create"),
    GetExperiment: _experiments_rule("get"),
    GetExperimentByName: _experiments_rule("get"),
    DeleteExperiment: _experiments_rule("delete"),
    RestoreExperiment: _experiments_rule("update"),
    UpdateExperiment: _experiments_rule("update"),
    SetExperimentTag: _experiments_rule("update"),
    DeleteExperimentTag: _experiments_rule("delete"),
    SearchExperiments: _experiments_rule("list"),
    # Runs
    CreateRun: _experiments_rule("update", override_run_user=True),
    GetRun: _experiments_rule("get"),
    DeleteRun: _experiments_rule("delete"),
    RestoreRun: _experiments_rule("update"),
    UpdateRun: _experiments_rule("update"),
    LogMetric: _experiments_rule("update"),
    LogBatch: _experiments_rule("update"),
    LogModel: _experiments_rule("update"),
    SetTag: _experiments_rule("update"),
    DeleteTag: _experiments_rule("delete"),
    LogParam: _experiments_rule("update"),
    GetMetricHistory: _experiments_rule("list"),
    ListArtifacts: _experiments_rule("list"),
    SearchLoggedModels: _experiments_rule("list"),
    # Logged models
    CreateLoggedModel: _experiments_rule("update"),
    GetLoggedModel: _experiments_rule("get"),
    DeleteLoggedModel: _experiments_rule("delete"),
    FinalizeLoggedModel: _experiments_rule("update"),
    DeleteLoggedModelTag: _experiments_rule("delete"),
    SetLoggedModelTags: _experiments_rule("update"),
    LogLoggedModelParamsRequest: _experiments_rule("update"),
    # Model registry
    CreateRegisteredModel: _registered_models_rule("create"),
    GetRegisteredModel: _registered_models_rule("get"),
    DeleteRegisteredModel: _registered_models_rule("delete"),
    UpdateRegisteredModel: _registered_models_rule("update"),
    RenameRegisteredModel: _registered_models_rule("update"),
    GetLatestVersions: _registered_models_rule("list"),
    CreateModelVersion: _registered_models_rule("update"),
    GetModelVersion: _registered_models_rule("get"),
    DeleteModelVersion: _registered_models_rule("delete"),
    UpdateModelVersion: _registered_models_rule("update"),
    TransitionModelVersionStage: _registered_models_rule("update"),
    GetModelVersionDownloadUri: _registered_models_rule("get"),
    SetRegisteredModelTag: _registered_models_rule("update"),
    DeleteRegisteredModelTag: _registered_models_rule("delete"),
    SetModelVersionTag: _registered_models_rule("update"),
    DeleteModelVersionTag: _registered_models_rule("delete"),
    SetRegisteredModelAlias: _registered_models_rule("update"),
    DeleteRegisteredModelAlias: _registered_models_rule("delete"),
    GetModelVersionByAlias: _registered_models_rule("get"),
    SearchRegisteredModels: _registered_models_rule("list"),
    # Scorers
    DeleteScorer: _experiments_rule("delete"),
    GetScorer: _experiments_rule("get"),
    ListScorers: _experiments_rule("list"),
    ListScorerVersions: _experiments_rule("list"),
    RegisterScorer: _experiments_rule("update"),
    # Traces
    BatchGetTraces: _experiments_rule("list"),
    CalculateTraceFilterCorrelation: _experiments_rule("list"),
    DeleteTraceTag: _experiments_rule("delete"),
    DeleteTraceTagV3: _experiments_rule("delete"),
    DeleteTraces: _experiments_rule("delete"),
    DeleteTracesV3: _experiments_rule("delete"),
    EndTrace: _experiments_rule("update"),
    GetTraceInfo: _experiments_rule("get"),
    GetTraceInfoV3: _experiments_rule("get"),
    LinkTracesToRun: _experiments_rule("update"),
    SearchTraces: _experiments_rule("list"),
    SearchTracesV3: _experiments_rule("list"),
    SetTraceTag: _experiments_rule("update"),
    SetTraceTagV3: _experiments_rule("update"),
    StartTrace: _experiments_rule("update"),
    StartTraceV3: _experiments_rule("update"),
    # Runs extras
    GetMetricHistoryBulkInterval: _experiments_rule("list"),
    LogInputs: _experiments_rule("update"),
    LogOutputs: _experiments_rule("update"),
    SearchRuns: _experiments_rule("list"),
    # Logged models extras
    ListLoggedModelArtifacts: _experiments_rule("list"),
    # Artifacts service
    AbortMultipartUpload: _experiments_rule("delete"),
    CompleteMultipartUpload: _experiments_rule("update"),
    CreateMultipartUpload: _experiments_rule("update"),
    DeleteArtifact: _experiments_rule("delete"),
    DownloadArtifact: _experiments_rule("get"),
    ListArtifactsMlflowArtifacts: _experiments_rule("list"),
    UploadArtifact: _experiments_rule("update"),
    # Webhooks
    CreateWebhook: _registered_models_rule("update"),
    DeleteWebhook: _registered_models_rule("delete"),
    GetWebhook: _registered_models_rule("get"),
    ListWebhooks: _registered_models_rule("list"),
    TestWebhook: _registered_models_rule("update"),
    UpdateWebhook: _registered_models_rule("update"),
    # Workspaces
    # ListWorkspaces omits a direct RBAC verb/namespace check because a single
    # SelfSubjectAccessReview cannot cover the full list. The response is instead filtered per
    # namespace via accessible_workspaces.
    ListWorkspaces: _workspaces_rule(None, apply_workspace_filter=True, requires_workspace=False),
    GetWorkspace: _workspaces_rule(None, requires_workspace=False),
    CreateWorkspace: _workspaces_rule("create", deny=True, requires_workspace=False),
    UpdateWorkspace: _workspaces_rule("update", deny=True, requires_workspace=False),
    DeleteWorkspace: _workspaces_rule("delete", deny=True, requires_workspace=False),
}


PATH_AUTHORIZATION_RULES: dict[tuple[str, str], AuthorizationRule] = {
    ("/api/2.0/mlflow/model-versions/search", "GET"): _registered_models_rule("list"),
    ("/ajax-api/2.0/mlflow/model-versions/search", "GET"): _registered_models_rule("list"),
    ("/graphql", "GET"): _experiments_rule("get"),
    ("/graphql", "POST"): _experiments_rule("get"),
    ("/version", "GET"): AuthorizationRule(None),
    ("/api/2.0/mlflow/gateway-proxy", "GET"): _experiments_rule("get"),
    ("/api/2.0/mlflow/gateway-proxy", "POST"): _experiments_rule("update"),
    ("/ajax-api/2.0/mlflow/gateway-proxy", "GET"): _experiments_rule("get"),
    ("/ajax-api/2.0/mlflow/gateway-proxy", "POST"): _experiments_rule("update"),
    ("/get-artifact", "GET"): _experiments_rule("get"),
    ("/model-versions/get-artifact", "GET"): _registered_models_rule("get"),
    ("/ajax-api/2.0/mlflow/upload-artifact", "POST"): _experiments_rule("update"),
    ("/ajax-api/2.0/mlflow/get-trace-artifact", "GET"): _experiments_rule("get"),
    ("/ajax-api/2.0/mlflow/metrics/get-history-bulk", "GET"): _experiments_rule("list"),
    (
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
        "GET",
    ): _experiments_rule("list"),
    ("/ajax-api/2.0/mlflow/runs/create-promptlab-run", "POST"): _experiments_rule("update"),
    (
        "/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files",
        "GET",
    ): _experiments_rule("get"),
    ("/api/2.0/mlflow/experiments/search-datasets", "POST"): _experiments_rule("list"),
    ("/ajax-api/2.0/mlflow/experiments/search-datasets", "POST"): _experiments_rule("list"),
    # Trace retrieval endpoints (REST v3)
    ("/api/3.0/mlflow/traces/get", "GET"): _experiments_rule("get"),
    ("/ajax-api/3.0/mlflow/traces/get", "GET"): _experiments_rule("get"),
    # OTEL trace ingestion endpoint
    (OTLP_TRACES_PATH, "POST"): _experiments_rule("update"),
    # Job API endpoints (FastAPI router)
    ("/ajax-api/3.0/jobs", "POST"): _jobs_rule("create"),
    ("/ajax-api/3.0/jobs/", "POST"): _jobs_rule("create"),
    ("/ajax-api/3.0/jobs/<job_id>", "GET"): _jobs_rule("get"),
    ("/ajax-api/3.0/jobs/<job_id>/", "GET"): _jobs_rule("get"),
    ("/ajax-api/3.0/jobs/search", "POST"): _jobs_rule("list"),
    ("/ajax-api/3.0/jobs/search/", "POST"): _jobs_rule("list"),
}


def _build_graphql_operation_rules() -> dict[str, AuthorizationRule]:
    rules: dict[str, AuthorizationRule] = {}
    resource_operations = set(K8S_GRAPHQL_OPERATION_RESOURCE_MAP)
    verb_operations = set(K8S_GRAPHQL_OPERATION_VERB_MAP)

    missing_verbs = resource_operations - verb_operations
    extra_verbs = verb_operations - resource_operations
    if missing_verbs or extra_verbs:
        details = []
        if missing_verbs:
            details.append(f"missing verbs for {sorted(missing_verbs)}")
        if extra_verbs:
            details.append(f"unexpected verbs for {sorted(extra_verbs)}")
        raise ValueError("GraphQL operation mappings are inconsistent: " + "; ".join(details))

    for operation_name in resource_operations:
        resource = K8S_GRAPHQL_OPERATION_RESOURCE_MAP[operation_name]
        verb = K8S_GRAPHQL_OPERATION_VERB_MAP[operation_name]
        rules[operation_name] = AuthorizationRule(
            verb, resource=_normalize_resource_name(resource) or RESOURCE_EXPERIMENTS
        )
    return rules


GRAPHQL_OPERATION_RULES: dict[str, AuthorizationRule] = _build_graphql_operation_rules()
_DEFAULT_GRAPHQL_RULE = AuthorizationRule("get", resource=RESOURCE_EXPERIMENTS)


def _unwrap_handler(handler):
    while hasattr(handler, "__wrapped__"):
        handler = handler.__wrapped__
    return handler


def _load_kubernetes_configuration() -> client.Configuration:
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    try:
        return client.Configuration.get_default_copy()
    except AttributeError:  # pragma: no cover - fallback for older client versions
        return client.Configuration()


def _create_api_client_for_subject_access_reviews() -> client.ApiClient:
    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except ConfigException as exc:  # pragma: no cover - depends on env
            raise MlflowException(
                "Failed to load Kubernetes configuration for authorization plugin",
                error_code=databricks_pb2.INVALID_STATE,
            ) from exc
    return client.ApiClient()


def _get_static_prefix() -> str | None:
    prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, None)
    if prefix:
        return prefix

    return None


def _strip_static_prefix(path: str) -> str:
    prefix = _get_static_prefix()
    if not prefix:
        return path

    prefix = prefix.rstrip("/")
    if prefix and path.startswith(prefix):
        stripped = path[len(prefix) :]
        return stripped if stripped.startswith("/") else f"/{stripped}"
    return path


@lru_cache(maxsize=None)
def _re_compile_path(path: str) -> re.Pattern[str]:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith("path:"):
            return "(.+)"
        return "([^/]+)"

    return re.compile(re.sub(r"<([^>]+)>", _replace, path))


def _is_unprotected_path(path: str) -> bool:
    return (
        any(path.startswith(prefix) for prefix in _UNPROTECTED_PATH_PREFIXES)
        or path in _UNPROTECTED_PATHS
    )


_TEMPLATE_TOKEN_PATTERN = re.compile(r"<[^>]+>|{[^}]+}")


def _fastapi_path_to_template(path: str) -> str:
    """Convert FastAPI-style `{param}` segments into Flask-style `<param>` tokens."""
    return re.sub(r"{([^}]+)}", r"<\1>", path)


def _templated_path_to_probe(path: str, placeholder: str = "probe") -> str:
    """Replace templated segments with a concrete placeholder for matching."""
    return _TEMPLATE_TOKEN_PATTERN.sub(placeholder, path)


def _is_fastapi_protected_path(path: str) -> bool:
    return path.startswith(_FASTAPI_AUTH_PREFIXES)


def _parse_jwt_subject(token: str, claim: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_segment = parts[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        payload = json.loads(decoded)
        value = payload.get(claim)
        return value if isinstance(value, str) and value else None
    except Exception as exc:
        _logger.error(
            "Failed to extract claim '%s' from JWT payload: %s",
            claim,
            exc,
            exc_info=True,
        )
        return None


@dataclass
class _AuthorizationResult:
    identity: _RequestIdentity
    rule: AuthorizationRule
    username: str | None

    @property
    def token(self) -> str | None:
        return self.identity.token


def _resolve_bearer_token(
    authorization_header: str | None, forwarded_access_token: str | None
) -> str:
    if authorization_header:
        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
        # fall through to forwarded token if available

    if forwarded_access_token:
        token = forwarded_access_token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if token:
            return token

    if authorization_header:
        raise MlflowException(
            "Authorization header must be in the format 'Bearer <token>'.",
            error_code=databricks_pb2.UNAUTHENTICATED,
        )

    raise MlflowException(
        "Missing Authorization header or X-Forwarded-Access-Token header.",
        error_code=databricks_pb2.UNAUTHENTICATED,
    )


def _parse_remote_groups(
    header_value: str | None, separator: str = DEFAULT_REMOTE_GROUPS_SEPARATOR
) -> tuple[str, ...]:
    if not header_value:
        return ()

    tokens = [header_value] if not separator else header_value.split(separator)
    return tuple(token.strip() for token in tokens if token and token.strip())


def _extract_workspace_scope_from_request(rule: AuthorizationRule) -> str | None:
    """
    Attempt to recover the workspace name from the active Flask request.

    Workspace CRUD endpoints encode the target workspace name in either the route parameters
    (e.g., ``/workspaces/<workspace_name>``) or, for creation, within the JSON payload. These
    endpoints do not require the workspace context header, so we fall back to parsing the request
    when the context is absent.
    """

    if not has_request_context():
        return None

    view_args = getattr(request, "view_args", None)
    if isinstance(view_args, dict):
        candidate = view_args.get("workspace_name")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate:
                return candidate

    if rule.resource != RESOURCE_WORKSPACES:
        return None

    if (request.method or "").upper() == "POST":
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            candidate = payload.get("name")
            if isinstance(candidate, str):
                candidate = candidate.strip()
                if candidate:
                    return candidate

    return None


def _authorize_request(
    *,
    authorization_header: str | None,
    forwarded_access_token: str | None,
    remote_user_header_value: str | None,
    remote_groups_header_value: str | None,
    path: str,
    method: str,
    authorizer: KubernetesAuthorizer,
    config_values: KubernetesAuthConfig,
    workspace: str | None,
) -> _AuthorizationResult:
    """
    Resolve the caller identity and ensure the MLflow request is permitted.

    Depending on the configured authorization mode, the caller is represented either by a bearer
    token (validated via `SelfSubjectAccessReview`) or by proxy-provided username/group headers that
    are evaluated through `SubjectAccessReview`. The resolved AuthorizationRule determines which
    Kubernetes resource/verb combination must be authorized within the workspace context. Any
    failures surface as `MlflowException` instances so HTTP handlers can relay a structured error.

    Returns:
        _AuthorizationResult: Includes the normalized identity, matched authorization rule, and the
            username derived from the token or proxy headers (used to override run ownership).

    Raises:
        MlflowException: If authentication information is missing/invalid, the workspace context is
            required but absent, or Kubernetes denies the requested access.
    """
    if config_values.authorization_mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
        token = _resolve_bearer_token(authorization_header, forwarded_access_token)
        identity = _RequestIdentity(token=token)
        username = _parse_jwt_subject(token, config_values.username_claim)
    else:
        remote_user = (remote_user_header_value or "").strip()
        if not remote_user:
            raise MlflowException(
                f"Missing required '{config_values.user_header}' header for "
                "SubjectAccessReview mode.",
                error_code=databricks_pb2.UNAUTHENTICATED,
            )
        groups = _parse_remote_groups(remote_groups_header_value, config_values.groups_separator)
        identity = _RequestIdentity(user=remote_user, groups=groups)
        username = remote_user

    workspace_name = None
    if isinstance(workspace, str):
        workspace_name = workspace.strip() or None

    rule = _find_authorization_rule(path, method)
    if rule is None:
        raise MlflowException(
            f"Endpoint '{method} {path}' is not covered by Kubernetes authorization.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )

    if rule.deny:
        raise MlflowException(
            _WORKSPACE_MUTATION_DENIED_MESSAGE,
            error_code=databricks_pb2.PERMISSION_DENIED,
        )

    if not workspace_name and rule.resource == RESOURCE_WORKSPACES:
        workspace_name = _extract_workspace_scope_from_request(rule)

    if not workspace_name and rule.requires_workspace:
        raise MlflowException(
            _WORKSPACE_REQUIRED_ERROR_MESSAGE,
            error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
        )

    resource_type = rule.resource
    if rule.verb is not None:
        if not resource_type:
            raise MlflowException(
                f"Authorization rule for '{method} {path}' is missing an RBAC resource mapping.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        allowed = authorizer.is_allowed(identity, resource_type, rule.verb, workspace_name)
        if not allowed:
            raise MlflowException(
                "Permission denied for requested operation.",
                error_code=databricks_pb2.PERMISSION_DENIED,
            )
    elif rule.resource == RESOURCE_WORKSPACES and not rule.deny and not rule.apply_workspace_filter:
        if not workspace_name:
            raise MlflowException(
                _WORKSPACE_REQUIRED_ERROR_MESSAGE,
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        if not authorizer.can_access_workspace(identity, workspace_name, verb="get"):
            raise MlflowException(
                "Permission denied for requested operation.",
                error_code=databricks_pb2.PERMISSION_DENIED,
            )
    elif rule.apply_workspace_filter:
        pass  # Authorization is handled via response filtering
    else:
        raise MlflowException(
            f"Authorization rule for '{method} {path}' is missing a verb or other "
            "required configuration.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )

    return _AuthorizationResult(identity=identity, rule=rule, username=username)


class KubernetesAuthorizer:
    def __init__(
        self, config_values: KubernetesAuthConfig, group: str = DEFAULT_AUTH_GROUP
    ) -> None:
        self._group = group
        self._cache = _AuthorizationCache(config_values.cache_ttl_seconds)
        self._mode = config_values.authorization_mode
        self._base_configuration: client.Configuration | None = None
        self._sar_api_client: client.ApiClient | None = None
        self._user_header_label = (
            f"Header '{config_values.user_header}'"
            if config_values.user_header
            else "Remote user header"
        )

        if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
            self._base_configuration = _load_kubernetes_configuration()
        else:
            self._sar_api_client = _create_api_client_for_subject_access_reviews()

    def _build_api_client_with_token(self, token: str) -> client.ApiClient:
        if self._base_configuration is None:
            raise MlflowException(
                "SelfSubjectAccessReview mode is not initialized with base configuration.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        base = self._base_configuration
        configuration = client.Configuration()
        # Make a copy of the Kubernetes client without credential information
        configuration.host = base.host
        configuration.ssl_ca_cert = base.ssl_ca_cert
        configuration.verify_ssl = base.verify_ssl
        configuration.proxy = base.proxy
        configuration.no_proxy = base.no_proxy
        configuration.proxy_headers = base.proxy_headers
        configuration.safe_chars_for_path_param = base.safe_chars_for_path_param
        configuration.connection_pool_maxsize = base.connection_pool_maxsize
        configuration.assert_hostname = getattr(base, "assert_hostname", None)
        configuration.retries = getattr(base, "retries", None)
        configuration.cert_file = None
        configuration.key_file = None
        configuration.username = None
        configuration.password = None
        configuration.refresh_api_key_hook = None
        configuration.api_key = {"authorization": token}
        configuration.api_key_prefix = {"authorization": "Bearer"}
        return client.ApiClient(configuration)

    def _submit_self_subject_access_review(
        self,
        token: str,
        resource: str,
        verb: str,
        namespace: str,
    ) -> bool:
        body = client.V1SelfSubjectAccessReview(
            spec=client.V1SelfSubjectAccessReviewSpec(
                resource_attributes=client.V1ResourceAttributes(
                    group=self._group,
                    resource=resource,
                    verb=verb,
                    namespace=namespace,
                )
            )
        )

        api_client = self._build_api_client_with_token(token)
        try:
            authorization_api = AuthorizationV1Api(api_client)
            response = authorization_api.create_self_subject_access_review(body)  # type: ignore[call-arg]
        finally:
            api_client.close()

        status = getattr(response, "status", None)
        allowed = getattr(status, "allowed", None)
        if allowed is None:
            raise MlflowException(
                "Unexpected Kubernetes SelfSubjectAccessReview response structure",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        return bool(allowed)

    def _submit_subject_access_review(
        self,
        user: str,
        groups: tuple[str, ...],
        resource: str,
        verb: str,
        namespace: str,
    ) -> bool:
        body = client.V1SubjectAccessReview(
            spec=client.V1SubjectAccessReviewSpec(
                user=user,
                groups=list(groups) if groups else None,
                resource_attributes=client.V1ResourceAttributes(
                    group=self._group,
                    resource=resource,
                    verb=verb,
                    namespace=namespace,
                ),
            )
        )

        if self._sar_api_client is None:
            raise MlflowException(
                "SubjectAccessReview mode requires a Kubernetes client.",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )

        authorization_api = AuthorizationV1Api(self._sar_api_client)
        response = authorization_api.create_subject_access_review(body)  # type: ignore[call-arg]

        status = getattr(response, "status", None)
        allowed = getattr(status, "allowed", None)
        if allowed is None:
            raise MlflowException(
                "Unexpected Kubernetes SubjectAccessReview response structure",
                error_code=databricks_pb2.INTERNAL_ERROR,
            )
        return bool(allowed)

    def is_allowed(
        self,
        identity: _RequestIdentity,
        resource_type: str,
        verb: str,
        namespace: str,
    ) -> bool:
        resource = resource_type.replace("_", "")
        identity_hash = identity.subject_hash(
            self._mode, missing_user_label=self._user_header_label
        )
        cache_key = (identity_hash, namespace, resource, verb)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
                allowed = self._submit_self_subject_access_review(
                    identity.token or "", resource, verb, namespace
                )
            else:
                allowed = self._submit_subject_access_review(
                    identity.user or "", identity.groups, resource, verb, namespace
                )
        except ApiException as exc:  # pragma: no cover - depends on live cluster
            if exc.status == 401:
                raise MlflowException(
                    "Authentication with the Kubernetes API failed. The provided token may be "
                    "invalid or expired.",
                    error_code=databricks_pb2.UNAUTHENTICATED,
                ) from exc
            if exc.status == 403:
                if self._mode == AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW:
                    message = (
                        "The Kubernetes service account is not permitted to perform "
                        "SelfSubjectAccessReview. Grant the required authorization."
                    )
                else:
                    message = (
                        "The Kubernetes service account is not permitted to perform "
                        "SubjectAccessReview. Grant 'create' on subjectaccessreviews."
                    )
                raise MlflowException(
                    message,
                    error_code=databricks_pb2.PERMISSION_DENIED,
                ) from exc
            raise MlflowException(
                f"Failed to perform Kubernetes authorization check: {exc}",
                error_code=databricks_pb2.INTERNAL_ERROR,
            ) from exc

        self._cache.set(cache_key, allowed)
        _logger.debug(
            "Access review evaluated subject_hash=%s resource=%s namespace=%s verb=%s allowed=%s",
            identity_hash,
            resource,
            namespace,
            verb,
            allowed,
        )
        return allowed

    def accessible_workspaces(self, identity: _RequestIdentity, names: Iterable[str]) -> set[str]:
        accessible: set[str] = set()
        subject_hash = identity.subject_hash(self._mode, missing_user_label=self._user_header_label)
        for workspace_name in names:
            if self.can_access_workspace(identity, workspace_name, verb="list"):
                accessible.add(workspace_name)
            else:
                _logger.debug(
                    "Workspace %s excluded for subject_hash=%s; no list permission detected",
                    workspace_name,
                    subject_hash,
                )
        return accessible

    def can_access_workspace(
        self, identity: _RequestIdentity, workspace_name: str, verb: str = "get"
    ) -> bool:
        """Check if the identity can access the workspace via any priority resource.

        Iterates through experiments, registeredmodels, and jobs resources to find if
        the identity has the specified permission on any of them for the given workspace
        (namespace).

        Args:
            identity: The request identity containing token or user/groups.
            workspace_name: The workspace (namespace) to check access for.
            verb: The Kubernetes RBAC verb to check (default: "get").

        Returns:
            True if the identity has the permission on any priority resource.
        """
        subject_hash = identity.subject_hash(self._mode, missing_user_label=self._user_header_label)
        for resource in _WORKSPACE_PERMISSION_RESOURCE_PRIORITY:
            if self.is_allowed(identity, resource, verb, workspace_name):
                _logger.debug(
                    "Workspace %s accessible for subject_hash=%s via resource=%s verb=%s",
                    workspace_name,
                    subject_hash,
                    resource,
                    verb,
                )
                return True
        return False


@dataclass(frozen=True)
class KubernetesAuthConfig:
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    username_claim: str = DEFAULT_USERNAME_CLAIM
    authorization_mode: AuthorizationMode = AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW
    user_header: str = DEFAULT_REMOTE_USER_HEADER
    groups_header: str = DEFAULT_REMOTE_GROUPS_HEADER
    groups_separator: str = DEFAULT_REMOTE_GROUPS_SEPARATOR

    @classmethod
    def from_env(cls) -> "KubernetesAuthConfig":
        ttl_env = os.environ.get(CACHE_TTL_ENV)
        username_claim = os.environ.get(USERNAME_CLAIM_ENV, DEFAULT_USERNAME_CLAIM)
        mode_env = os.environ.get(
            AUTHORIZATION_MODE_ENV, AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW.value
        )
        user_header = os.environ.get(REMOTE_USER_HEADER_ENV, DEFAULT_REMOTE_USER_HEADER)
        groups_header = os.environ.get(REMOTE_GROUPS_HEADER_ENV, DEFAULT_REMOTE_GROUPS_HEADER)
        groups_separator = os.environ.get(
            REMOTE_GROUPS_SEPARATOR_ENV, DEFAULT_REMOTE_GROUPS_SEPARATOR
        )

        cache_ttl_seconds = DEFAULT_CACHE_TTL_SECONDS
        if ttl_env:
            try:
                cache_ttl_seconds = float(ttl_env)
                if cache_ttl_seconds <= 0:
                    raise ValueError
            except ValueError as exc:
                raise MlflowException(
                    f"Environment variable {CACHE_TTL_ENV} must be a positive number if set",
                    error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
                ) from exc

        try:
            authorization_mode = AuthorizationMode(mode_env.strip().lower())
        except ValueError as exc:
            valid_modes = ", ".join(mode.value for mode in AuthorizationMode)
            raise MlflowException(
                f"Environment variable {AUTHORIZATION_MODE_ENV} must be one of: {valid_modes}",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            ) from exc

        user_header = user_header.strip()
        groups_header = groups_header.strip()
        if not user_header:
            raise MlflowException(
                f"Environment variable {REMOTE_USER_HEADER_ENV} cannot be empty",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        if not groups_header:
            raise MlflowException(
                f"Environment variable {REMOTE_GROUPS_HEADER_ENV} cannot be empty",
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )

        return cls(
            cache_ttl_seconds=cache_ttl_seconds,
            username_claim=username_claim,
            authorization_mode=authorization_mode,
            user_header=user_header,
            groups_header=groups_header,
            groups_separator=groups_separator or DEFAULT_REMOTE_GROUPS_SEPARATOR,
        )


def _compile_authorization_rules() -> None:
    global _RULES_COMPILED
    if _RULES_COMPILED:
        return

    # Rebuild every cache/artifact so reconfiguration (e.g., during tests) is deterministic.
    _HANDLER_RULES.clear()

    exact_rules: dict[tuple[str, str], AuthorizationRule] = {}
    regex_rules: list[tuple[re.Pattern[str], str, AuthorizationRule]] = []
    uncovered: list[tuple[str, str]] = []

    def _get_request_authorization_handler(request_class):
        # Record the AuthorizationRule associated with the concrete Flask handler so we can
        # reference it later when iterating through Flask endpoints.
        handler = mlflow_handlers.get_handler(request_class)
        rule = REQUEST_AUTHORIZATION_RULES.get(request_class)
        if handler is not None and rule is not None:
            _HANDLER_RULES[_unwrap_handler(handler)] = rule
        return handler

    # Inspect the protobuf-driven Flask routes and copy over authorization metadata.
    for path, handler, methods in get_endpoints(_get_request_authorization_handler):
        if not path:
            continue

        canonical_path = _strip_static_prefix(path)
        if _is_unprotected_path(canonical_path):
            continue

        base_handler = _unwrap_handler(handler)
        rule = _HANDLER_RULES.get(base_handler)
        if rule is None:
            # If a protobuf route lacks a handler-derived rule, fall back to the explicit
            # PATH_AUTHORIZATION_RULES definition; otherwise flag it as uncovered.
            if all(
                PATH_AUTHORIZATION_RULES.get((canonical_path, method)) is not None
                for method in methods
            ):
                continue
            uncovered.extend((canonical_path, method) for method in methods)
            continue

        for method in methods:
            # Regex patterns are required for templated paths; literal paths can be matched exactly.
            if "<" in canonical_path:
                regex_rules.append((_re_compile_path(canonical_path), method, rule))
            else:
                exact_rules[(canonical_path, method)] = rule

    # Include custom Flask routes (e.g., get-artifact) that aren't part of the protobuf services.
    for rule in mlflow_app.url_map.iter_rules():
        view_func = mlflow_app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue

        canonical_path = _strip_static_prefix(rule.rule)
        if _is_unprotected_path(canonical_path):
            continue

        base_handler = _unwrap_handler(view_func)
        if base_handler in _HANDLER_RULES:
            continue

        methods = {m for m in (rule.methods or set()) if m not in {"HEAD", "OPTIONS"}}
        for method in methods:
            # These custom routes rely exclusively on PATH_AUTHORIZATION_RULES; track any gaps.
            if PATH_AUTHORIZATION_RULES.get((canonical_path, method)) is None:
                uncovered.append((canonical_path, method))

    # Explicit allowlist entries (with and without templated segments) always win.
    for (path, method), rule in PATH_AUTHORIZATION_RULES.items():
        if "<" in path:
            regex_rules.append((_re_compile_path(path), method, rule))
        else:
            exact_rules[(path, method)] = rule

    if uncovered:
        formatted = ", ".join(f"{method} {path}" for path, method in uncovered)
        raise MlflowException(
            "Kubernetes auth plugin cannot determine authorization mapping for endpoints: "
            f"{formatted}. Update the plugin allow list or verb mapping.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )

    # Persist the computed lookup tables so _find_authorization_rule can use them.
    _AUTH_RULES.update(exact_rules)
    _AUTH_REGEX_RULES.extend(regex_rules)
    _RULES_COMPILED = True


def _validate_fastapi_route_authorization(fastapi_app: FastAPI) -> None:
    """Ensure all protected FastAPI routes are covered by authorization rules."""
    missing: list[tuple[str, str]] = []

    for route in getattr(fastapi_app, "routes", []):
        if not isinstance(route, APIRoute):
            continue
        methods = getattr(route, "methods", set()) or set()
        canonical_path = _strip_static_prefix(route.path or "")
        if not _is_fastapi_protected_path(canonical_path):
            continue
        template_path = _fastapi_path_to_template(canonical_path)
        # Use a concrete probe path so _find_authorization_rule follows the same regex path
        # matching logic that real requests do.
        probe_path = _templated_path_to_probe(template_path)

        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            if _find_authorization_rule(probe_path, method) is None:
                missing.append((method, canonical_path))

    if missing:
        formatted = ", ".join(f"{method} {path}" for method, path in missing)
        raise MlflowException(
            "Kubernetes auth plugin is missing authorization rules for FastAPI endpoints: "
            f"{formatted}. Update PATH_AUTHORIZATION_RULES before enabling the plugin.",
            error_code=databricks_pb2.INTERNAL_ERROR,
        )


def _find_authorization_rule(request_path: str, method: str) -> AuthorizationRule | None:
    canonical_path = _strip_static_prefix(request_path or "")

    rule = _AUTH_RULES.get((canonical_path, method))
    if rule is not None:
        # Special handling for GraphQL operations
        if canonical_path.endswith("/graphql"):
            try:
                payload = request.get_json(silent=True) or {}
            except Exception:
                payload = {}
            operation_name = payload.get("operationName")
            if not operation_name:
                return rule
            gql_rule = GRAPHQL_OPERATION_RULES.get(operation_name)
            if gql_rule is None:
                _logger.warning(
                    "GraphQL operation '%s' does not have an explicit Kubernetes authorization "
                    "mapping; defaulting to read-only access.",
                    operation_name,
                )
                return _DEFAULT_GRAPHQL_RULE
            return gql_rule
        return rule

    for pattern, pattern_method, candidate in _AUTH_REGEX_RULES:
        if pattern_method == method and pattern.fullmatch(canonical_path):
            return candidate

    return None


# MLflow has some APIs that are through Flask and some through FastAPI. This middleware is only
# responsible for the FastAPI APIs. When MLflow is running under uvicorn, the FastAPI app wraps the
# entire Flask app, so we need to be selective about which requests the FastAPI middleware
# processes.
class KubernetesAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Kubernetes-based authorization."""

    def __init__(self, app, authorizer: KubernetesAuthorizer, config_values: KubernetesAuthConfig):
        super().__init__(app)
        self.authorizer = authorizer
        self.config_values = config_values

    async def dispatch(self, request: Request, call_next):
        """Process each request through the authorization pipeline."""
        path = str(request.url.path or "")
        if not path.startswith(FASTAPI_NATIVE_PREFIXES):
            # Skip if not a FastAPI route handled by this middleware.
            return await call_next(request)

        canonical_path = _strip_static_prefix(path)

        # Skip authentication for unprotected paths
        if _is_unprotected_path(canonical_path):
            return await call_next(request)

        workspace_name = workspace_context.get_request_workspace()

        # Check permissions if verb is specified
        try:
            _authorize_request(
                authorization_header=request.headers.get("Authorization"),
                forwarded_access_token=request.headers.get("X-Forwarded-Access-Token"),
                remote_user_header_value=request.headers.get(self.config_values.user_header),
                remote_groups_header_value=request.headers.get(self.config_values.groups_header),
                path=path,
                method=request.method,
                authorizer=self.authorizer,
                config_values=self.config_values,
                workspace=workspace_name,
            )
        except MlflowException as exc:
            if (
                workspace_name is None
                and exc.error_code
                == databricks_pb2.ErrorCode.Name(databricks_pb2.INVALID_PARAMETER_VALUE)
                and exc.message == _WORKSPACE_REQUIRED_ERROR_MESSAGE
            ):
                exc = MlflowException(
                    _WORKSPACE_REQUIRED_ERROR_MESSAGE,
                    error_code=databricks_pb2.INTERNAL_ERROR,
                )
            return JSONResponse(
                status_code=exc.get_http_status_code(),
                content={"error": {"code": exc.error_code, "message": exc.message}},
            )

        # Continue with the request
        return await call_next(request)


def _override_run_user(username: str) -> None:
    """Rewrite the request payload so MLflow sees the authenticated user as run owner."""
    if not request.mimetype or "json" not in request.mimetype.lower():
        return

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return

    payload["user_id"] = username
    data = json.dumps(payload).encode("utf-8")

    # Reset cached JSON with proper structure expected by Werkzeug
    # The keys are boolean values for the 'silent' parameter
    request._cached_json = {True: payload, False: payload}  # type: ignore[attr-defined]
    request._cached_data = data  # type: ignore[attr-defined]
    request.environ["wsgi.input"] = io.BytesIO(data)
    request.environ["CONTENT_LENGTH"] = str(len(data))
    request.environ["CONTENT_TYPE"] = "application/json"


def create_app(app: Flask = mlflow_app) -> Flask:
    """Enable Kubernetes-based authorization for the MLflow tracking server."""

    global _logger
    parent_logger = getattr(app, "logger", logging.getLogger("mlflow"))
    _logger = parent_logger
    _logger.info("Kubernetes authorization plugin initialized")

    config_values = KubernetesAuthConfig.from_env()
    authorizer = KubernetesAuthorizer(config_values=config_values)

    _compile_authorization_rules()

    @app.before_request
    def _k8s_auth_before_request():
        path = request.path or ""
        if _is_unprotected_path(_strip_static_prefix(path)):
            return None

        try:
            auth_result = _authorize_request(
                authorization_header=request.headers.get("Authorization"),
                forwarded_access_token=request.headers.get("X-Forwarded-Access-Token"),
                remote_user_header_value=request.headers.get(config_values.user_header),
                remote_groups_header_value=request.headers.get(config_values.groups_header),
                path=path,
                method=request.method,
                authorizer=authorizer,
                config_values=config_values,
                workspace=workspace_context.get_request_workspace(),
            )
        except MlflowException as exc:
            response = Response(mimetype="application/json")
            response.set_data(exc.serialize_as_json())
            response.status_code = exc.get_http_status_code()
            return response

        if auth_result.username and auth_result.rule.override_run_user:
            _override_run_user(auth_result.username)

        # These can be used in after_request hooks to modify the response.
        g.mlflow_k8s_identity = auth_result.identity
        g.mlflow_k8s_apply_workspace_filter = auth_result.rule.apply_workspace_filter

        return None

    @app.after_request
    def _k8s_auth_after_request(response: Response):
        try:
            should_filter = getattr(g, "mlflow_k8s_apply_workspace_filter", False)
            identity = getattr(g, "mlflow_k8s_identity", None)
            can_filter_response = (
                response.mimetype == "application/json" and response.status_code < 400
            )
            if not (should_filter and identity and can_filter_response):
                return response

            try:
                payload = json.loads(response.get_data(as_text=True))
            except Exception:
                payload = None

            if not isinstance(payload, dict):
                return response

            workspaces = payload.get("workspaces")
            if not isinstance(workspaces, list):
                return response

            workspace_names = [ws.get("name") for ws in workspaces if isinstance(ws, dict)]
            accessible = authorizer.accessible_workspaces(
                identity, [name for name in workspace_names if isinstance(name, str)]
            )
            payload["workspaces"] = [
                ws for ws in workspaces if isinstance(ws, dict) and ws.get("name") in accessible
            ]
            response.set_data(json.dumps(payload))
            response.headers["Content-Length"] = str(len(response.get_data()))
        finally:
            for attr in (
                "mlflow_k8s_identity",
                "mlflow_k8s_apply_workspace_filter",
            ):
                if hasattr(g, attr):
                    delattr(g, attr)

        return response

    if _MLFLOW_SGI_NAME.get() == "uvicorn":
        fastapi_app = create_fastapi_app(app)

        # Add Kubernetes auth middleware to FastAPI
        # Important: This must be added AFTER security middleware but BEFORE routes
        # to ensure proper middleware ordering
        #
        # Note: The KubernetesAuthMiddleware only handles FastAPI-specific routes
        # (e.g. OTEL and Job API endpoints). All other routes are handled by the Flask
        # auth handlers defined above. This is because when running under uvicorn,
        # the FastAPI app wraps the entire Flask app, so we need to be selective
        # about which requests the FastAPI middleware processes.
        fastapi_app.add_middleware(
            KubernetesAuthMiddleware,
            authorizer=authorizer,
            config_values=config_values,
        )
        _validate_fastapi_route_authorization(fastapi_app)
        return fastapi_app
    return app


__all__ = [
    "create_app",
    "K8S_GRAPHQL_OPERATION_RESOURCE_MAP",
    "K8S_GRAPHQL_OPERATION_VERB_MAP",
]
