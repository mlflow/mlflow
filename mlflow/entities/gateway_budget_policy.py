from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import BudgetAction as ProtoBudgetAction
from mlflow.protos.service_pb2 import BudgetDurationUnit as ProtoBudgetDurationUnit
from mlflow.protos.service_pb2 import BudgetTargetType as ProtoBudgetTargetType
from mlflow.protos.service_pb2 import BudgetUnit as ProtoBudgetUnit
from mlflow.protos.service_pb2 import GatewayBudgetPolicy as ProtoGatewayBudgetPolicy
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class BudgetDurationUnit(str, Enum):
    """Duration unit for budget policy fixed windows."""

    MINUTES = "MINUTES"
    HOURS = "HOURS"
    DAYS = "DAYS"
    MONTHS = "MONTHS"

    @classmethod
    def from_proto(cls, proto: ProtoBudgetDurationUnit) -> BudgetDurationUnit | None:
        try:
            return cls(ProtoBudgetDurationUnit.Name(proto))
        except ValueError:
            return None

    def to_proto(self) -> ProtoBudgetDurationUnit:
        return ProtoBudgetDurationUnit.Value(self.value)


class BudgetTargetScope(str, Enum):
    """Target scope for a budget policy."""

    GLOBAL = "GLOBAL"
    WORKSPACE = "WORKSPACE"

    @classmethod
    def from_proto(cls, proto: ProtoBudgetTargetType) -> BudgetTargetType | None:
        try:
            return cls(ProtoBudgetTargetType.Name(proto))
        except ValueError:
            return None

    def to_proto(self) -> ProtoBudgetTargetType:
        return ProtoBudgetTargetType.Value(self.value)


class BudgetAction(str, Enum):
    """Action to take when a budget is exceeded."""

    ALERT = "ALERT"
    REJECT = "REJECT"

    @classmethod
    def from_proto(cls, proto: ProtoBudgetAction) -> BudgetAction | None:
        try:
            return cls(ProtoBudgetAction.Name(proto))
        except ValueError:
            return None

    def to_proto(self) -> ProtoBudgetAction:
        return ProtoBudgetAction.Value(self.value)


class BudgetUnit(str, Enum):
    """Budget measurement unit."""

    USD = "USD"

    @classmethod
    def from_proto(cls, proto: ProtoBudgetUnit) -> BudgetUnit | None:
        try:
            return cls(ProtoBudgetUnit.Name(proto))
        except ValueError:
            return None

    def to_proto(self) -> ProtoBudgetUnit:
        return ProtoBudgetUnit.Value(self.value)


@dataclass
class GatewayBudgetPolicy(_MlflowObject):
    """
    Represents a budget policy for the AI Gateway.

    Budget policies set limits with fixed time windows,
    supporting global or per-workspace scoping.

    Args:
        budget_policy_id: Unique identifier for this budget policy.
        budget_unit: Budget measurement unit (e.g. USD).
        budget_amount: Budget limit amount.
        duration_unit: Unit of time window (MINUTES, HOURS, DAYS, MONTHS).
        duration_value: Length of the window in units of duration_unit.
        target_scope: Scope of the budget (GLOBAL or WORKSPACE).
        budget_action: Action when budget is exceeded (ALERT, REJECT).
        created_at: Timestamp (milliseconds) when the policy was created.
        last_updated_at: Timestamp (milliseconds) when the policy was last updated.
        created_by: User ID who created the policy.
        last_updated_by: User ID who last updated the policy.
        workspace: Workspace that owns the policy.
    """

    budget_policy_id: str
    budget_unit: BudgetUnit
    budget_amount: float
    duration_unit: BudgetDurationUnit
    duration_value: int
    target_scope: BudgetTargetScope
    budget_action: BudgetAction
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
    workspace: str | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)
        if isinstance(self.budget_unit, str):
            self.budget_unit = BudgetUnit(self.budget_unit)
        if isinstance(self.duration_unit, str):
            self.duration_unit = BudgetDurationUnit(self.duration_unit)
        if isinstance(self.target_scope, str):
            self.target_scope = BudgetTargetScope(self.target_scope)
        if isinstance(self.budget_action, str):
            self.budget_action = BudgetAction(self.budget_action)

    def to_proto(self):
        proto = ProtoGatewayBudgetPolicy()
        proto.budget_policy_id = self.budget_policy_id
        proto.budget_unit = self.budget_unit.to_proto()
        proto.budget_amount = self.budget_amount
        proto.duration_unit = self.duration_unit.to_proto()
        proto.duration_value = self.duration_value
        proto.target_type = self.target_type.to_proto()
        proto.budget_action = self.budget_action.to_proto()
        proto.created_by = self.created_by or ""
        proto.created_at = self.created_at
        proto.last_updated_by = self.last_updated_by or ""
        proto.last_updated_at = self.last_updated_at
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            budget_policy_id=proto.budget_policy_id,
            budget_unit=BudgetUnit.from_proto(proto.budget_unit),
            budget_amount=proto.budget_amount,
            duration_unit=BudgetDurationUnit.from_proto(proto.duration_unit),
            duration_value=proto.duration_value,
            target_type=BudgetTargetType.from_proto(proto.target_type),
            budget_action=BudgetAction.from_proto(proto.budget_action),
            created_by=proto.created_by or None,
            created_at=proto.created_at,
            last_updated_by=proto.last_updated_by or None,
            last_updated_at=proto.last_updated_at,
        )
