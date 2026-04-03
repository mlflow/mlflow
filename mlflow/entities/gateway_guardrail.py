from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.scorer import ScorerVersion
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class GuardrailStage(str, Enum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"

    def __str__(self) -> str:
        return self.value


class GuardrailAction(str, Enum):
    VALIDATION = "VALIDATION"
    SANITIZATION = "SANITIZATION"

    def __str__(self) -> str:
        return self.value


@dataclass
class GatewayGuardrail(_MlflowObject):
    guardrail_id: str
    name: str
    scorer: ScorerVersion
    stage: GuardrailStage
    action: GuardrailAction
    created_at: int
    last_updated_at: int
    action_endpoint_name: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    workspace: str | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)
        if isinstance(self.stage, str):
            self.stage = GuardrailStage(self.stage)
        if isinstance(self.action, str):
            self.action = GuardrailAction(self.action)


@dataclass
class GatewayGuardrailConfig(_MlflowObject):
    """Junction between a guardrail and a gateway endpoint, with ordering."""

    endpoint_id: str
    guardrail_id: str
    execution_order: int | None
    created_at: int
    guardrail: GatewayGuardrail | None = None
    created_by: str | None = None
    workspace: str | None = None
