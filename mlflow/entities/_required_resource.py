from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# These type strings must match the RESOURCE_TYPE_* constants in
# mlflow.server.auth.permissions (e.g. RESOURCE_TYPE_GATEWAY_ENDPOINT,
# RESOURCE_TYPE_PROMPT). We can't import them directly due to circular
# dependency (mlflow.entities -> mlflow.server -> mlflow.genai -> mlflow.entities).
# Alignment is enforced by tests/genai/scorers/test_resources.py::test_resource_types_match_rbac.
RequiredResourceType = Literal["gateway_endpoint", "prompt"]

VALID_REQUIRED_RESOURCE_TYPES: frozenset[str] = frozenset({"gateway_endpoint", "prompt"})


@dataclass(frozen=True)
class RequiredResource:
    """MLflow-managed resource a job needs at runtime.

    The job execution framework reads these declarations to compute
    least-privilege scoped permissions for remote execution. The ``type``
    values align with MLflow RBAC resource types defined in
    ``mlflow.server.auth.permissions``.
    """

    type: RequiredResourceType
    # Resource identifier — a name (e.g. "my-endpoint") or a URI
    # (e.g. "prompts:/tool-grounded/1") depending on the resource type.
    name: str

    def __post_init__(self):
        if self.type not in VALID_REQUIRED_RESOURCE_TYPES:
            raise ValueError(
                f"Unknown required resource type: {self.type!r}. "
                f"Supported types: {sorted(VALID_REQUIRED_RESOURCE_TYPES)}"
            )
        if not self.name or not self.name.strip():
            raise ValueError("RequiredResource name must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "name": self.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequiredResource:
        return cls(type=data["type"], name=data["name"])
