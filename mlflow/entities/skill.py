from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class SkillStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"

    def __str__(self):
        return self.value


@experimental(version="3.16.0")
@dataclass
class Skill:
    name: str
    organization: str = ""
    description: str | None = None
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
                f"Failed to parse Skill response: missing required field {e}"
            ) from None
        except (ValueError, TypeError, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse Skill response: {e}"
            ) from None
