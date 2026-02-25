from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class BudgetDurationType(str, Enum):
    """Duration type for budget policy fixed windows."""

    HOURS = "HOURS"
    DAYS = "DAYS"
    MONTHS = "MONTHS"


class BudgetTargetType(str, Enum):
    """Target scope for a budget policy."""

    GLOBAL = "GLOBAL"
    WORKSPACE = "WORKSPACE"


class BudgetOnExceeded(str, Enum):
    """Action to take when a budget is exceeded."""

    ALERT = "ALERT"
    REJECT = "REJECT"
    ALERT_AND_REJECT = "ALERT_AND_REJECT"


@dataclass
class GatewayBudgetPolicy(_MlflowObject):
    """
    Represents a budget policy for the AI Gateway.

    Budget policies set cost-based limits (USD) with fixed time windows,
    supporting global or per-workspace scoping.

    Args:
        budget_policy_id: Unique identifier for this budget policy.
        name: User-friendly name for the budget policy.
        limit_usd: Budget limit in USD.
        duration_type: Type of time window (HOURS, DAYS, MONTHS).
        duration_value: Length of the window in units of duration_type.
        target_type: Scope of the budget (GLOBAL or WORKSPACE).
        on_exceeded: Action when budget is exceeded (ALERT, REJECT, ALERT_AND_REJECT).
        created_at: Timestamp (milliseconds) when the policy was created.
        last_updated_at: Timestamp (milliseconds) when the policy was last updated.
        created_by: User ID who created the policy.
        last_updated_by: User ID who last updated the policy.
        workspace: Workspace that owns the policy.
    """

    budget_policy_id: str
    name: str
    limit_usd: float
    duration_type: BudgetDurationType
    duration_value: int
    target_type: BudgetTargetType
    on_exceeded: BudgetOnExceeded
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
    workspace: str | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)
        if isinstance(self.duration_type, str):
            self.duration_type = BudgetDurationType(self.duration_type)
        if isinstance(self.target_type, str):
            self.target_type = BudgetTargetType(self.target_type)
        if isinstance(self.on_exceeded, str):
            self.on_exceeded = BudgetOnExceeded(self.on_exceeded)
