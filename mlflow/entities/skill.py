from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from typing_extensions import NotRequired

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class RegistryIcon(TypedDict):
    """Icon descriptor for registry presentation metadata.

    Same shape as RFC-0004's ``MCPIcon`` so UIs share one icon renderer
    across the MCP, skill, and agent plugin registries.
    """

    src: str
    sizes: NotRequired[list[str]]
    mimeType: NotRequired[str]
    theme: NotRequired[str]


class SkillStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"

    def __str__(self):
        return self.value


def _aliases_to_dict(aliases: Any) -> dict[str, Any]:
    """Normalize an aliases payload to an ``{alias: version}`` dict.

    REST responses expose aliases as a list of ``{"alias": ..., "version": ...}``
    objects; accept that list (and an already-normalized dict) and return the dict
    shape the entity fields use.
    """
    if not aliases:
        return {}
    if isinstance(aliases, dict):
        return aliases
    return {a["alias"]: a["version"] for a in aliases}


@experimental(version="3.16.0")
@dataclass
class Skill:
    name: str
    organization: str = ""
    description: str | None = None
    # mutable presentation metadata; returned as stored (null when unset)
    icons: list[RegistryIcon] | None = None
    workspace: str | None = None
    status: SkillStatus | None = None  # read-only, derived by producers
    tags: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, int] = field(default_factory=dict)  # read-only
    latest_version: int | None = None  # read-only
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse Skill response: expected a dictionary"
            )
        try:
            return cls(
                name=data["name"],
                organization=data.get("organization", ""),
                description=data.get("description"),
                icons=data.get("icons"),
                workspace=data.get("workspace"),
                status=SkillStatus(data["status"]) if data.get("status") else None,
                tags=data.get("tags") or {},
                aliases=_aliases_to_dict(data.get("aliases")),
                latest_version=data.get("latest_version"),
                created_by=data.get("created_by"),
                last_updated_by=data.get("last_updated_by"),
                creation_timestamp=data.get("creation_timestamp"),
                last_updated_timestamp=data.get("last_updated_timestamp"),
            )
        except KeyError as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse Skill response: missing required field {e}"
            ) from None
        except (ValueError, TypeError, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse Skill response: {e}"
            ) from None
