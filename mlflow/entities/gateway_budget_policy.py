from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import BudgetAction as ProtoBudgetAction
from mlflow.protos.service_pb2 import BudgetDuration as ProtoBudgetDuration
from mlflow.protos.service_pb2 import BudgetDurationUnit as ProtoBudgetDurationUnit
from mlflow.protos.service_pb2 import BudgetTargetScope as ProtoBudgetTargetScope
from mlflow.protos.service_pb2 import BudgetUnit as ProtoBudgetUnit
from mlflow.protos.service_pb2 import GatewayBudgetPolicy as ProtoGatewayBudgetPolicy
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class BudgetDurationUnit(str, Enum):
    """Duration unit for budget policy fixed windows."""

    MINUTES = "MINUTES"
    HOURS = "HOURS"
    DAYS = "DAYS"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"

    @classmethod
    def from_proto(cls, proto: ProtoBudgetDurationUnit) -> BudgetDurationUnit | None:
        try:
            return cls(ProtoBudgetDurationUnit.Name(proto))
        except ValueError:
            return None

    def to_proto(self) -> ProtoBudgetDurationUnit:
        # `EnumTypeWrapper.Value` is untyped upstream, hence the cast.
        return cast(ProtoBudgetDurationUnit, ProtoBudgetDurationUnit.Value(self.value))


class BudgetTargetScope(str, Enum):
    """Target scope for a budget policy."""

    GLOBAL = "GLOBAL"
    WORKSPACE = "WORKSPACE"
    ENDPOINT = "ENDPOINT"
    USER = "USER"

    @classmethod
    def from_proto(cls, proto: ProtoBudgetTargetScope) -> BudgetTargetScope | None:
        try:
            return cls(ProtoBudgetTargetScope.Name(proto))
        except ValueError:
            return None

    def to_proto(self) -> ProtoBudgetTargetScope:
        # `EnumTypeWrapper.Value` is untyped upstream, hence the cast.
        return cast(ProtoBudgetTargetScope, ProtoBudgetTargetScope.Value(self.value))


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
        # `EnumTypeWrapper.Value` is untyped upstream, hence the cast.
        return cast(ProtoBudgetAction, ProtoBudgetAction.Value(self.value))


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
        # `EnumTypeWrapper.Value` is untyped upstream, hence the cast.
        return cast(ProtoBudgetUnit, ProtoBudgetUnit.Value(self.value))


@dataclass
class BudgetDuration:
    """Fixed window duration: a (unit, value) pair defining the length of a budget window."""

    unit: BudgetDurationUnit
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.unit, str):
            self.unit = BudgetDurationUnit(self.unit)

    def to_proto(self) -> ProtoBudgetDuration:
        proto = ProtoBudgetDuration()
        proto.unit = self.unit.to_proto()
        proto.value = self.value
        return proto

    @classmethod
    def from_proto(cls, proto: ProtoBudgetDuration) -> BudgetDuration:
        # Well-formed protos always carry concrete enum values, so the
        # optional results of the enum converters cannot occur here.
        return cls(
            unit=cast(BudgetDurationUnit, BudgetDurationUnit.from_proto(proto.unit)),
            value=proto.value,
        )


@dataclass
class GatewayBudgetPolicy(_MlflowObject):
    """
    Represents a budget policy for the AI Gateway.

    Budget policies set limits with fixed time windows,
    supporting global, per-workspace, per-endpoint, or per-user scoping.

    Args:
        budget_policy_id: Unique identifier for this budget policy.
        budget_unit: Budget measurement unit (e.g. USD).
        budget_amount: Budget limit amount.
        duration: Fixed time window (unit + length pair).
        target_scope: Scope of the budget (GLOBAL, WORKSPACE, ENDPOINT, or USER).
        budget_action: Action when budget is exceeded (ALERT, REJECT).
        created_at: Timestamp (milliseconds) when the policy was created.
        last_updated_at: Timestamp (milliseconds) when the policy was last updated.
        created_by: User ID who created the policy.
        last_updated_by: User ID who last updated the policy.
        workspace: Workspace that owns the policy.
        target_value: Target the policy applies to, interpreted per ``target_scope``:
            a gateway endpoint ID for ENDPOINT scope, a username for
            USER scope. ``None`` for GLOBAL and WORKSPACE scopes.
    """

    budget_policy_id: str
    budget_unit: BudgetUnit
    budget_amount: float
    duration: BudgetDuration
    target_scope: BudgetTargetScope
    budget_action: BudgetAction
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
    workspace: str | None = None
    target_value: str | None = None

    def __post_init__(self) -> None:
        self.workspace = resolve_entity_workspace_name(self.workspace)
        if isinstance(self.budget_unit, str):
            self.budget_unit = BudgetUnit(self.budget_unit)
        if isinstance(self.target_scope, str):
            self.target_scope = BudgetTargetScope(self.target_scope)
        if isinstance(self.budget_action, str):
            self.budget_action = BudgetAction(self.budget_action)

    def to_proto(self) -> ProtoGatewayBudgetPolicy:
        proto = ProtoGatewayBudgetPolicy()
        proto.budget_policy_id = self.budget_policy_id
        proto.budget_unit = self.budget_unit.to_proto()
        proto.budget_amount = self.budget_amount
        proto.duration.CopyFrom(self.duration.to_proto())
        proto.target_scope = self.target_scope.to_proto()
        proto.budget_action = self.budget_action.to_proto()
        proto.created_by = self.created_by or ""
        proto.created_at = self.created_at
        proto.last_updated_by = self.last_updated_by or ""
        proto.last_updated_at = self.last_updated_at
        if self.target_value is not None:
            proto.target_value = self.target_value
        return proto

    @classmethod
    def from_proto(cls, proto: ProtoGatewayBudgetPolicy) -> GatewayBudgetPolicy:
        # Well-formed protos always carry concrete enum values, so the
        # optional results of the enum converters cannot occur here.
        return cls(
            budget_policy_id=proto.budget_policy_id,
            budget_unit=cast(BudgetUnit, BudgetUnit.from_proto(proto.budget_unit)),
            budget_amount=proto.budget_amount,
            duration=BudgetDuration.from_proto(proto.duration),
            target_scope=cast(BudgetTargetScope, BudgetTargetScope.from_proto(proto.target_scope)),
            budget_action=cast(BudgetAction, BudgetAction.from_proto(proto.budget_action)),
            created_by=proto.created_by or None,
            created_at=proto.created_at,
            last_updated_by=proto.last_updated_by or None,
            last_updated_at=proto.last_updated_at,
            target_value=proto.target_value or None,
        )
