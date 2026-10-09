from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from mlflow.server.skill_registry.artifacts import (
    delete_artifact_tree_best_effort,
    is_serving_artifacts,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_logger = logging.getLogger(__name__)


def delete_skill(
    name: str,
    organization: str = "",
    before_commit: Callable[[Session], None] | None = None,
) -> None:
    """
    Hard-delete a skill and reclaim the artifact content its versions owned.

    The store checks live references, captures the owned artifact paths, and commits the row
    deletion in one transaction. Only after that commit are the captured paths deleted, each
    one best-effort: a cleanup failure is logged and never restores rows or fails the delete.
    Content a version merely references, such as a tree that belongs to an imported agent
    plugin, is never among the captured paths, even when this was its last reference.

    The optional ``before_commit`` callback runs inside the store's deletion transaction.
    Authorization uses it to revoke grants before the parent identity can be reused; a
    callback failure rolls back the deletion and prevents artifact cleanup.
    """
    from mlflow.server.handlers import _get_tracking_store

    owned_paths = _get_tracking_store().delete_skill_and_collect_artifacts(
        name, organization, before_commit=before_commit
    )
    if not owned_paths:
        return
    if not is_serving_artifacts():
        _logger.warning(
            "Skill '%s' owned %d artifact path(s) but this server does not serve artifacts, so "
            "they were left in place: %s",
            name,
            len(owned_paths),
            ", ".join(owned_paths),
        )
        return
    for path in owned_paths:
        delete_artifact_tree_best_effort(path)
