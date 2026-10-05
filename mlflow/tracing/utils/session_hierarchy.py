"""Utilities for hierarchical (multi-level) session IDs.

A hierarchical session ID is a compact JSON array of level values, outermost level
first, e.g. ``["trip-2","ep-2","dom-2","turn-5"]``. The level names for an
experiment are declared in the ``mlflow.experiment.sessionHierarchy`` experiment
tag with the JSON value ``{"version": 1, "levels": [{"name": "trip"}, ...]}``.
"""

import json
import logging
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracing.utils import serialize_session_id
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_SESSION_HIERARCHY

_logger = logging.getLogger(__name__)


def parse_session_hierarchy_levels(tag_value: str | None) -> list[str] | None:
    """Parse the ``mlflow.experiment.sessionHierarchy`` tag value into level names.

    Args:
        tag_value: Raw tag value. Expected to be JSON like
            ``{"version": 1, "levels": [{"name": "trip"}, {"name": "episode"}]}``.

    Returns:
        The list of level names, or None if the tag is absent or invalid. ``version``
        must be an int and ``levels`` a non-empty list of objects with unique,
        non-empty string names.
    """
    if not tag_value:
        return None
    try:
        data = json.loads(tag_value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        return None
    levels = data.get("levels")
    if not isinstance(levels, list) or not levels:
        return None
    names = []
    for level in levels:
        if not isinstance(level, dict):
            return None
        name = level.get("name")
        if not isinstance(name, str) or not name:
            return None
        names.append(name)
    if len(set(names)) != len(names):
        return None
    return names


def parse_session_path(value: Any) -> list[str] | None:
    """Return the level values if ``value`` is a hierarchical session ID.

    A hierarchical session ID is a compact JSON array of strings. Any other value
    (a plain session ID string, a non-array JSON document, an empty array, or a
    non-string value) returns None.
    """
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data and all(isinstance(item, str) for item in data):
        return data
    return None


def session_group_key(session_value: str, level_index: int) -> str:
    """Compute the session grouping key at the given hierarchy level.

    Args:
        session_value: Raw ``mlflow.trace.session`` value.
        level_index: 0-based position of the hierarchy level to group by.

    Returns:
        For a hierarchical session path with more than ``level_index`` elements,
        the serialized prefix of the first ``level_index + 1`` elements. For a
        plain session ID or a path shorter than the level, the original value
        unchanged, so the trace forms its own group.
    """
    path = parse_session_path(session_value)
    if path is not None and len(path) > level_index:
        return serialize_session_id(path[: level_index + 1])
    return session_value


@experimental(version="3.17.0")
def set_session_hierarchy(levels: list[str], experiment_id: str | None = None) -> None:
    """Declare the session hierarchy levels for an experiment.

    Writes the ``mlflow.experiment.sessionHierarchy`` tag so that scorers with a
    ``session_level`` can group traces by a hierarchy level instead of the full
    session value.

    Args:
        levels: Non-empty list of unique, non-empty level names, outermost first
            (e.g. ``["trip", "episode", "domain"]``).
        experiment_id: The experiment to tag. Defaults to the current experiment.
    """
    if (
        not isinstance(levels, list)
        or not levels
        or not all(isinstance(level, str) and level for level in levels)
        or len(set(levels)) != len(levels)
    ):
        raise MlflowException.invalid_parameter_value(
            f"`levels` must be a non-empty list of unique, non-empty level names, got {levels!r}."
        )

    experiment_id = experiment_id or _get_experiment_id()
    tag_value = json.dumps(
        {"version": 1, "levels": [{"name": level} for level in levels]},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    MlflowClient().set_experiment_tag(experiment_id, MLFLOW_EXPERIMENT_SESSION_HIERARCHY, tag_value)
