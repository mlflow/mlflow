from __future__ import annotations

import logging

from mlflow.entities import Workspace
from mlflow.protos import databricks_pb2

_INVALID_PARAMETER_VALUE_CODE = databricks_pb2.INVALID_PARAMETER_VALUE
_INVALID_PARAMETER_VALUE_NAME = databricks_pb2.ErrorCode.Name(_INVALID_PARAMETER_VALUE_CODE)


_logger = logging.getLogger(__name__)


def get_default_workspace_optional(workspace_store) -> tuple[Workspace | None, bool]:
    """
    Attempt to resolve a default workspace from the provider without bubbling opt-out errors.

    Providers can signal that default workspace resolution is unsupported by raising
    ``NotImplementedError``. This helper normalizes that case and returns ``(None, False)`` so
    callers can decide how to proceed.

    Args:
        workspace_store: Workspace store exposing ``get_default_workspace``.

    Returns:
        Tuple of (workspace or None, supports_default_workspace flag).
    """
    if workspace_store is None:
        return None, False

    provider_name = type(workspace_store).__name__

    try:
        workspace = workspace_store.get_default_workspace()
    except NotImplementedError:
        _logger.debug(
            "Workspace provider %s does not implement default workspace resolution",
            provider_name,
        )
        return None, False

    return workspace, True
