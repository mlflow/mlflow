from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import (
    GitSource,
    OCISource,
    SkillSourceType,
    ZipSource,
    build_source,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@experimental(version="3.16.0")
@dataclass
class SkillVersion:
    name: str
    version: int
    organization: str = ""
    source: GitSource | OCISource | ZipSource | str | None = None  # immutable after creation
    source_type: SkillSourceType | None = None
    digest: str | None = None
    status: SkillStatus = SkillStatus.ACTIVE
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)  # read-only
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillVersion:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse SkillVersion response: expected a dictionary"
            )
        try:
            source_type = SkillSourceType(data["source_type"]) if data.get("source_type") else None
            return cls(
                name=data["name"],
                version=data["version"],
                organization=data.get("organization", ""),
                source=build_source(
                    source_type, data.get("source"), data.get("ref"), data.get("subpath")
                ),
                source_type=source_type,
                digest=data.get("digest"),
                status=SkillStatus(data["status"]) if data.get("status") else SkillStatus.ACTIVE,
                tags=data.get("tags") or {},
                aliases=data.get("aliases") or [],
                workspace=data.get("workspace"),
                created_by=data.get("created_by"),
                last_updated_by=data.get("last_updated_by"),
                creation_timestamp=data.get("creation_timestamp"),
                last_updated_timestamp=data.get("last_updated_timestamp"),
            )
        except KeyError as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse SkillVersion response: missing required field {e}"
            ) from None
        except (ValueError, TypeError, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse SkillVersion response: {e}"
            ) from None
