from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import (
    GitSource,
    OCISource,
    SkillSourceType,
    ZipSource,
    source_from_dict,
    source_to_dict,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@experimental(version="3.16.0")
@dataclass
class AgentPluginVersion:
    name: str
    version: str
    organization: str = ""
    plugin_json: dict[str, Any] = field(default_factory=dict)  # immutable after creation
    source: GitSource | OCISource | ZipSource | str | None = None  # immutable after creation
    source_type: SkillSourceType | None = None
    status: SkillStatus = SkillStatus.ACTIVE
    tags: dict[str, str] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)  # read-only member list
    aliases: list[str] = field(default_factory=list)  # read-only
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "organization": self.organization,
            "plugin_json": self.plugin_json,
            "source": source_to_dict(self.source),
            "source_type": str(self.source_type) if self.source_type is not None else None,
            "status": str(self.status),
            "tags": self.tags,
            "skills": self.skills,
            "aliases": self.aliases,
            "workspace": self.workspace,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentPluginVersion:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse AgentPluginVersion response: expected a dictionary"
            )
        try:
            source_type = SkillSourceType(data["source_type"]) if data.get("source_type") else None
            return cls(
                name=data["name"],
                version=data["version"],
                organization=data.get("organization", ""),
                plugin_json=data.get("plugin_json") or {},
                source=source_from_dict(data.get("source"), source_type),
                source_type=source_type,
                status=SkillStatus(data["status"]) if data.get("status") else SkillStatus.ACTIVE,
                tags=data.get("tags") or {},
                skills=data.get("skills") or [],
                aliases=data.get("aliases") or [],
                workspace=data.get("workspace"),
                created_by=data.get("created_by"),
                last_updated_by=data.get("last_updated_by"),
                creation_timestamp=data.get("creation_timestamp"),
                last_updated_timestamp=data.get("last_updated_timestamp"),
            )
        except KeyError as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse AgentPluginVersion response: missing required field {e}"
            ) from None
        except (ValueError, TypeError, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse AgentPluginVersion response: {e}"
            ) from None
