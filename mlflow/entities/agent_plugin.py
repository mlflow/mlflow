from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow.entities.skill import SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@experimental(version="3.16.0")
@dataclass
class AgentPlugin:
    name: str
    organization: str = ""
    description: str | None = None
    workspace: str | None = None
    status: SkillStatus | None = None  # read-only, derived by producers
    tags: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)  # read-only (alias -> version string)
    latest_version: str | None = None  # read-only
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "organization": self.organization,
            "description": self.description,
            "workspace": self.workspace,
            "status": str(self.status) if self.status is not None else None,
            "tags": self.tags,
            "aliases": self.aliases,
            "latest_version": self.latest_version,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentPlugin:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse AgentPlugin response: expected a dictionary"
            )
        try:
            return cls(
                name=data["name"],
                organization=data.get("organization", ""),
                description=data.get("description"),
                workspace=data.get("workspace"),
                status=SkillStatus(data["status"]) if data.get("status") else None,
                tags=data.get("tags") or {},
                aliases=data.get("aliases") or {},
                latest_version=data.get("latest_version"),
                created_by=data.get("created_by"),
                last_updated_by=data.get("last_updated_by"),
                creation_timestamp=data.get("creation_timestamp"),
                last_updated_timestamp=data.get("last_updated_timestamp"),
            )
        except KeyError as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse AgentPlugin response: missing required field {e}"
            ) from None
        except (ValueError, TypeError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse AgentPlugin response: {e}"
            ) from None
