from __future__ import annotations

import logging
import os

from flask import Response, request

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_WORKSPACE_URI
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.store.workspace.utils import get_default_workspace_optional
from mlflow.tracking._workspace import context as workspace_context
from mlflow.tracking._workspace import utils as workspace_utils
from mlflow.tracking._workspace.registry import get_workspace_store
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

_logger = logging.getLogger(__name__)

_workspace_store = None


def resolve_workspace_from_header(header_workspace: str | None) -> Workspace | None:
    """
    Resolve (and validate) the active workspace given an optional header value.

    When ``header_workspace`` is None or empty, the default workspace is used (if configured).
    Returns None if no workspace can be resolved.
    """
    store = _get_workspace_store()
    header_workspace = header_workspace.strip() if header_workspace else None

    if header_workspace:
        return store.get_workspace(header_workspace)

    workspace, _ = get_default_workspace_optional(store, logger=_logger)
    return workspace


def _get_workspace_store(workspace_uri: str | None = None, tracking_uri: str | None = None):
    """
    Resolve and cache the workspace store configured for this server process.

    The store is constructed on first invocation using the provided arguments (or their
    environment-derived defaults) and memoized for all subsequent calls, regardless of any new
    ``workspace_uri`` / ``tracking_uri`` values supplied later.
    """
    if not MLFLOW_ENABLE_WORKSPACES.get():
        raise MlflowException(
            "Workspace APIs are not available: multi-tenancy is not enabled on this server",
            databricks_pb2.FEATURE_DISABLED,
        )

    global _workspace_store
    if _workspace_store is not None:
        return _workspace_store

    from mlflow.server import BACKEND_STORE_URI_ENV_VAR

    resolved_tracking_uri = tracking_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR)
    resolved_workspace_uri = workspace_utils.resolve_workspace_uri(
        workspace_uri, tracking_uri=resolved_tracking_uri
    )
    if resolved_workspace_uri is None:
        raise MlflowException.invalid_parameter_value(
            "Workspace URI could not be resolved. Provide --workspace-store-uri or set "
            f"{MLFLOW_WORKSPACE_URI.name}."
        )

    _workspace_store = get_workspace_store(workspace_uri=resolved_workspace_uri)
    return _workspace_store


def _resolve_request_workspace() -> Workspace | None:
    """Determine the workspace for the current Flask request."""

    if not MLFLOW_ENABLE_WORKSPACES.get():
        return None

    return resolve_workspace_from_header(request.headers.get(WORKSPACE_HEADER_NAME))


def _workspace_error_response(exc: Exception) -> Response:
    if isinstance(exc, MlflowException):
        mlflow_exc = exc
    else:
        mlflow_exc = MlflowException(str(exc), error_code=databricks_pb2.INTERNAL_ERROR)

    response = Response(mimetype="application/json")
    response.set_data(mlflow_exc.serialize_as_json())
    response.status_code = mlflow_exc.get_http_status_code()
    return response


def workspace_before_request_handler():
    if not MLFLOW_ENABLE_WORKSPACES.get():
        header_workspace = request.headers.get(WORKSPACE_HEADER_NAME)
        header_workspace = header_workspace.strip() if header_workspace else None
        if header_workspace:
            return _workspace_error_response(
                MlflowException(
                    "Workspace APIs are not available: multi-tenancy is not enabled on this server",
                    error_code=databricks_pb2.FEATURE_DISABLED,
                )
            )
        return None

    try:
        workspace = _resolve_request_workspace()
    except MlflowException as exc:
        return _workspace_error_response(exc)
    except Exception as exc:
        _logger.exception("Unexpected error while resolving workspace")
        return _workspace_error_response(exc)

    workspace_name = workspace.name if workspace else None
    workspace_context.set_current_workspace(workspace_name)
    return None


def workspace_teardown_request_handler(_exc):
    if MLFLOW_ENABLE_WORKSPACES.get():
        workspace_context.clear_workspace()


__all__ = [
    "resolve_workspace_from_header",
    "_get_workspace_store",
    "workspace_before_request_handler",
    "workspace_teardown_request_handler",
]
