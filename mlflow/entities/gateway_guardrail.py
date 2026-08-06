from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.scorer import ScorerVersion
from mlflow.protos.service_pb2 import GatewayGuardrail as ProtoGatewayGuardrail
from mlflow.protos.service_pb2 import GatewayGuardrailConfig as ProtoGatewayGuardrailConfig
from mlflow.protos.service_pb2 import GuardrailAction as ProtoGuardrailAction
from mlflow.protos.service_pb2 import GuardrailStage as ProtoGuardrailStage
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class GuardrailStage(str, Enum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: ProtoGuardrailStage) -> GuardrailStage:
        return cls(ProtoGuardrailStage.Name(proto))

    def to_proto(self) -> ProtoGuardrailStage:
        return ProtoGuardrailStage.Value(self.value)


class GuardrailAction(str, Enum):
    VALIDATION = "VALIDATION"
    SANITIZATION = "SANITIZATION"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: ProtoGuardrailAction) -> GuardrailAction:
        return cls(ProtoGuardrailAction.Name(proto))

    def to_proto(self) -> ProtoGuardrailAction:
        return ProtoGuardrailAction.Value(self.value)


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

    def to_proto(self):
        proto = ProtoGatewayGuardrail()
        proto.guardrail_id = self.guardrail_id
        proto.name = self.name
        proto.scorer.CopyFrom(self.scorer.to_proto())
        proto.stage = self.stage.to_proto()
        proto.action = self.action.to_proto()
        if self.action_endpoint_name:
            proto.action_endpoint_id = self.action_endpoint_name
        proto.created_by = self.created_by or ""
        proto.created_at = self.created_at
        proto.last_updated_by = self.last_updated_by or ""
        proto.last_updated_at = self.last_updated_at
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            guardrail_id=proto.guardrail_id,
            name=proto.name,
            scorer=ScorerVersion.from_proto(proto.scorer),
            stage=GuardrailStage.from_proto(proto.stage),
            action=GuardrailAction.from_proto(proto.action),
            action_endpoint_name=proto.action_endpoint_id or None,
            created_by=proto.created_by or None,
            created_at=proto.created_at,
            last_updated_by=proto.last_updated_by or None,
            last_updated_at=proto.last_updated_at,
        )


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

    def to_proto(self):
        proto = ProtoGatewayGuardrailConfig()
        proto.endpoint_id = self.endpoint_id
        proto.guardrail_id = self.guardrail_id
        if self.execution_order is not None:
            proto.execution_order = self.execution_order
        if self.guardrail is not None:
            proto.guardrail.CopyFrom(self.guardrail.to_proto())
        proto.created_by = self.created_by or ""
        proto.created_at = self.created_at
        return proto

    @classmethod
    def from_proto(cls, proto):
        guardrail = None
        if proto.HasField("guardrail"):
            guardrail = GatewayGuardrail.from_proto(proto.guardrail)
        return cls(
            endpoint_id=proto.endpoint_id,
            guardrail_id=proto.guardrail_id,
            execution_order=proto.execution_order if proto.HasField("execution_order") else None,
            guardrail=guardrail,
            created_at=proto.created_at,
            created_by=proto.created_by or None,
        )
