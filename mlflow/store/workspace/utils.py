from __future__ import annotations

import logging

from mlflow.entities import Workspace
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2

_INVALID_PARAMETER_VALUE_NAME = databricks_pb2.ErrorCode.Name(
    databricks_pb2.INVALID_PARAMETER_VALUE
)


def get_default_workspace_optional(
    workspace_store, request, *, logger: logging.Logger | None = None
) -> tuple[Workspace | None, bool]:
    """
    Attempt to resolve a default workspace from the provider without bubbling opt-out errors.

    Providers can signal that default workspace resolution is unsupported by raising either
    ``NotImplementedError`` or ``MlflowException`` with ``INVALID_PARAMETER_VALUE``. This helper
    normalizes those cases and returns ``(None, False)`` so callers can decide how to proceed.

    Args:
        workspace_store: Workspace store exposing ``get_default_workspace``.
        request: Request object (may be ``None`` when not applicable).
        logger: Optional logger used for debug messaging.

    Returns:
        Tuple of (workspace or None, supports_default_workspace flag).
    """
    if workspace_store is None:
        return None, False

    provider_name = type(workspace_store).__name__
    active_logger = logger

    try:
        workspace = workspace_store.get_default_workspace(request)
    except NotImplementedError:
        if active_logger:
            active_logger.debug(
                "Workspace provider %s does not implement default workspace resolution",
                provider_name,
            )
        return None, False
    except MlflowException as exc:
        if exc.error_code == _INVALID_PARAMETER_VALUE_NAME:
            if active_logger:
                active_logger.debug(
                    "Workspace provider %s does not support default workspace resolution: %s",
                    provider_name,
                    exc,
                )
            return None, False
        raise

    return workspace, True
