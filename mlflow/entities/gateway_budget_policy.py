from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class BudgetDurationUnit(str, Enum):
    """Duration unit for budget policy fixed windows."""

    MINUTES = "MINUTES"
    HOURS = "HOURS"
    DAYS = "DAYS"
    MONTHS = "MONTHS"


class BudgetTargetScope(str, Enum):
    """Target scope for a budget policy."""

    GLOBAL = "GLOBAL"
    WORKSPACE = "WORKSPACE"


class BudgetAction(str, Enum):
    """Action to take when a budget is exceeded."""

    ALERT = "ALERT"
    REJECT = "REJECT"


class BudgetUnit(str, Enum):
    """Budget measurement unit."""

    USD = "USD"


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
