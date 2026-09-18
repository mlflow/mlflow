"""Search-text projection builders for the skill registry.

Each skill and agent-plugin version stores a ``search_text`` column that
concatenates several discovery fields into one value.  A single
``LIKE``/``ILIKE`` on this column matches across all discovery fields without
requiring ``OR`` (which MLflow ``filter_string`` does not support).

These helpers *build* the projection string.  They do **not** persist it —
callers in the concrete search stories handle persistence.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.store.tracking.dbmodels.models import (
        SqlAgentPluginVersion,
        SqlSkill,
    )

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    # str() guards against non-string values from untrusted manifest JSON.
    s = str(text)
    return _WHITESPACE_RUN.sub(" ", s).strip() if s else ""


def build_skill_search_text(name: str, description: str | None = None) -> str:
    """Build the ``search_text`` projection for a skill from its name and description."""
    parts = [_normalize(name), _normalize(description)]
    return " ".join(part for part in parts if part)


def build_agent_plugin_version_search_text(
    name: str,
    parent_description: str | None = None,
    organization: str = "",
    manifest_description: str | None = None,
    keywords: list[str] | None = None,
    author_name: str | None = None,
) -> str:
    """Build the ``search_text`` projection for an agent-plugin version.

    Covers name, mutable parent description, organization, and the version's
    manifest description, keywords, and author name.
    """
    parts = [
        _normalize(name),
        _normalize(parent_description),
        _normalize(organization),
        _normalize(manifest_description),
    ]
    if keywords:
        parts.extend(_normalize(kw) for kw in keywords)
    parts.append(_normalize(author_name))
    return " ".join(part for part in parts if part)


def recompute_skill_search_text(skill_row: SqlSkill) -> str:
    """Rebuild ``search_text`` from an ``SqlSkill`` row, e.g. after its description changes."""
    return build_skill_search_text(name=skill_row.name, description=skill_row.description)


def recompute_agent_plugin_version_search_text(
    version_row: SqlAgentPluginVersion,
    parent_description: str | None = None,
) -> str:
    """Convenience wrapper for an ``SqlAgentPluginVersion`` ORM row."""
    # plugin_json is stored manifest data, so ignore fields that do not have the
    # shape the RFC defines rather than failing or iterating a string's characters.
    plugin_json = version_row.plugin_json if isinstance(version_row.plugin_json, dict) else {}
    keywords = plugin_json.get("keywords")
    author = plugin_json.get("author")
    return build_agent_plugin_version_search_text(
        name=version_row.name,
        parent_description=parent_description,
        organization=version_row.organization,
        manifest_description=plugin_json.get("description"),
        keywords=keywords if isinstance(keywords, list) else None,
        author_name=author.get("name") if isinstance(author, dict) else None,
    )
