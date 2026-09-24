from __future__ import annotations

import logging
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import NOT_IMPLEMENTED

_logger = logging.getLogger(__name__)


def is_serving_artifacts() -> bool:
    # `mlflow.server.handlers` imports most of the server, including the modules that import
    # this package, so it is imported where it is used, as the other registry APIs do.
    from mlflow.server.handlers import _is_serving_proxied_artifacts

    return _is_serving_proxied_artifacts()


def require_artifact_serving() -> None:
    """
    Fail with a stable capability error when this deployment cannot store skill content.

    Uploaded content is written to, and later pulled through, the server's proxied artifact
    storage. A server started without ``--serve-artifacts`` has nowhere to put it, so
    MLflow-backed registration is refused up front rather than failing part-way. External
    ``git``/``oci``/``zip`` sources have no such requirement.
    """
    if not is_serving_artifacts():
        raise MlflowException(
            "This MLflow server does not serve artifacts, so skill content cannot be stored in "
            "MLflow artifact storage. Register the skill from a git, oci, or zip source, or "
            "start the server with --serve-artifacts.",
            error_code=NOT_IMPLEMENTED,
        )


def _repo_and_scoped_path(artifact_path: str):
    from mlflow.server.handlers import (
        _get_artifact_repo_mlflow_artifacts,
        _get_workspace_scoped_repo_path_if_enabled,
    )

    # Recorded paths are workspace-relative, like every `mlflow-artifacts:/` URI a client
    # sees; the workspace prefix is applied only when touching the underlying repository.
    scoped = _get_workspace_scoped_repo_path_if_enabled(artifact_path)
    return _get_artifact_repo_mlflow_artifacts(), scoped


def store_skill_tree(local_dir: Path, artifact_path: str) -> None:
    """Write the validated tree at ``local_dir`` to ``artifact_path`` in artifact storage."""
    repo, scoped = _repo_and_scoped_path(artifact_path)
    repo.log_artifacts(str(local_dir), artifact_path=scoped)


def delete_artifact_tree_best_effort(artifact_path: str) -> bool:
    """
    Delete ``artifact_path`` from artifact storage; never raises.

    A failure only leaves unreferenced bytes behind, which is an accepted leak, so it is logged
    and reported through the return value instead of failing the operation that already
    succeeded in the database.
    """
    try:
        repo, scoped = _repo_and_scoped_path(artifact_path)
        repo.delete_artifacts(scoped)
    except Exception as e:
        _logger.warning(
            "Failed to delete skill artifacts at '%s'; the content is no longer referenced by "
            "the registry and can be removed manually: %s",
            artifact_path,
            e,
        )
        return False
    return True
