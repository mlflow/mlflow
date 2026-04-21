from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import pydantic


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_schema(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_schema(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_schema(v) for v in value]
    if isinstance(value, pydantic.BaseModel):
        return value.__class__.model_json_schema()
    if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
        return value.model_json_schema()
    return value


@dataclass
class AgentInfo:
    """
    Metadata contract for agent discovery via the ``/agent/info`` endpoint.
    """

    name: str | None = None
    use_case: str | None = None
    mlflow_version: str | None = None
    agent_api: str | None = None
    description: str | None = None
    version: str | None = None
    metadata: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: _normalize_schema(value)
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }
