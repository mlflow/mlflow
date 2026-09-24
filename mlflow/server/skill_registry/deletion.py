from __future__ import annotations

import logging

from mlflow.server.skill_registry.artifacts import (
    delete_artifact_tree_best_effort,
    is_serving_artifacts,
)

_logger = logging.getLogger(__name__)


def delete_skill(name: str, organization: str = "") -> None:
    """
    Hard-delete a skill and reclaim the artifact content its versions owned.

    The store checks live references, captures the owned artifact paths, and commits the row
    deletion in one transaction. Only after that commit are the captured paths deleted, each
    one best-effort: a cleanup failure is logged and never restores rows or fails the delete.
    Content a version merely references, such as a tree that belongs to an imported agent
    plugin, is never among the captured paths, even when this was its last reference.
    """
    from mlflow.server.handlers import _get_tracking_store

    owned_paths = _get_tracking_store().delete_skill_and_collect_artifacts(name, organization)
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
