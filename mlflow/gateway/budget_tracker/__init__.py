"""Budget tracker for AI Gateway cost management.

Provides an abstract BudgetTracker interface and window computation helpers.
The concrete InMemoryBudgetTracker lives in ``budget_tracker.in_memory``.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mlflow.entities.gateway_budget_policy import (
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    GatewayBudgetPolicy,
)
from mlflow.environment_variables import (
    MLFLOW_GATEWAY_BUDGET_REDIS_URL,
    MLFLOW_GATEWAY_BUDGET_REFRESH_INTERVAL,
)
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)
# Sunday-aligned epoch for WEEKS windows (Dec 28, 1969 is the Sunday before Jan 1, 1970)
_EPOCH_SUNDAY = _EPOCH - timedelta(days=4)

# Module-level singleton
_budget_tracker: BudgetTracker | None = None
_tracker_lock = threading.Lock()


def get_budget_tracker() -> BudgetTracker:
    """Get or create the module-level BudgetTracker singleton."""
    global _budget_tracker
    if _budget_tracker is None:
        with _tracker_lock:
            if _budget_tracker is None:
                if redis_url := MLFLOW_GATEWAY_BUDGET_REDIS_URL.get():
                    from mlflow.gateway.budget_tracker.redis import RedisBudgetTracker

                    _budget_tracker = RedisBudgetTracker(_redis_url=redis_url)
                else:
                    from mlflow.gateway.budget_tracker.in_memory import InMemoryBudgetTracker

                    _budget_tracker = InMemoryBudgetTracker()
    return _budget_tracker


@dataclass
class BudgetWindow:
    """Tracks spend within a single fixed time window for one policy."""

    policy: GatewayBudgetPolicy
    window_start: datetime
    window_end: datetime
    cumulative_spend: float = 0.0
    exceeded: bool = False


class BudgetTracker(ABC):
    """Abstract base class for budget trackers.

    Defines the interface for tracking cumulative cost per budget policy
    within fixed time windows. Concrete implementations may store state
    in memory, Redis, or other backends.
    """

    _last_refresh_time: float = float("-inf")

    def needs_refresh(self) -> bool:
        """Check whether policies should be re-fetched from the database."""
        return (
            time.monotonic() - self._last_refresh_time
        ) >= MLFLOW_GATEWAY_BUDGET_REFRESH_INTERVAL.get()

    def mark_refreshed(self) -> None:
        """Mark the tracker as just refreshed."""
        self._last_refresh_time = time.monotonic()

    def invalidate(self) -> None:
        """Reset the refresh timer so the next needs_refresh() call returns True."""
        self._last_refresh_time = float("-inf")

    @abstractmethod
    def refresh_policies(self, policies: list[GatewayBudgetPolicy]) -> list[BudgetWindow]:
        """Load or refresh policies from the database.

        Preserves accumulated cost for unchanged windows. Removes windows
        for policies that no longer exist.

        Returns:
            List of newly created windows (cumulative_spend=0) that may need
            backfilling from historical trace data.
        """

    @abstractmethod
    def record_cost(
        self,
        cost_usd: float,
        workspace: str | None = None,
    ) -> list[BudgetWindow]:
        """Record a cost against all applicable policies.

        Args:
            cost_usd: The cost in USD to record.
            workspace: The workspace the request was made from (None for default).

        Returns:
            List of windows that were newly exceeded (limit exceeded for the first
            time in this window). Used to trigger webhook alerts.
        """

    @abstractmethod
    def should_reject_request(
        self,
        workspace: str | None = None,
    ) -> tuple[bool, BudgetWindow | None]:
        """Check if any REJECT-capable policy is exceeded.

        Args:
            workspace: The workspace to check against.

        Returns:
            Tuple of (exceeded, window). If exceeded is True, window is the
            first exceeded window found.
        """

    @abstractmethod
    def backfill_spend(self, spend_by_policy: dict[str, float]) -> None:
        """Set cumulative spend on windows from historical data.

        Used to seed newly created windows with spend from trace data
        so that budget tracking survives server restarts.

        Args:
            spend_by_policy: Dict mapping budget_policy_id to historical spend amount.
        """

    @abstractmethod
    def get_all_windows(self) -> list[BudgetWindow]:
        """Get the current window info for all tracked policies."""

    @abstractmethod
    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        """Get the current window info for a policy (for payload construction)."""


def _compute_window_start(
    duration: BudgetDuration,
    now: datetime,
) -> datetime:
    """Compute the start of the current fixed window for a given policy.

    Windows are aligned to:
    - MINUTES: aligned to epoch minutes
    - HOURS: aligned to epoch hours (e.g., duration.value=2 → 0:00, 2:00, 4:00, …)
    - DAYS: aligned to epoch days (e.g., duration.value=7 → weekly from epoch)
    - WEEKS: aligned to Sunday-based weeks (e.g., duration.value=1 → every Sunday from epoch)
    - MONTHS: aligned to first of months (e.g., duration.value=3 → Jan 1, Apr 1, Jul 1, …)
    """
    if duration.value <= 0:
        raise ValueError(f"duration.value must be positive, got {duration.value}")

    if duration.unit == BudgetDurationUnit.MINUTES:
        epoch = _EPOCH
        minutes_since_epoch = (now - epoch).total_seconds() / 60
        window_index = int(minutes_since_epoch) // duration.value
        window_start_minutes = window_index * duration.value
        return epoch + timedelta(minutes=window_start_minutes)

    elif duration.unit == BudgetDurationUnit.HOURS:
        epoch = _EPOCH
        hours_since_epoch = (now - epoch).total_seconds() / 3600
        window_index = int(hours_since_epoch) // duration.value
        window_start_hours = window_index * duration.value
        return epoch + timedelta(hours=window_start_hours)

    elif duration.unit == BudgetDurationUnit.DAYS:
        epoch = _EPOCH
        days_since_epoch = (now - epoch).days
        window_index = days_since_epoch // duration.value
        window_start_days = window_index * duration.value
        return epoch + timedelta(days=window_start_days)

    elif duration.unit == BudgetDurationUnit.WEEKS:
        days_since_sunday_epoch = (now - _EPOCH_SUNDAY).days
        window_index = days_since_sunday_epoch // (7 * duration.value)
        window_start_days = window_index * (7 * duration.value)
        return _EPOCH_SUNDAY + timedelta(days=window_start_days)

    elif duration.unit == BudgetDurationUnit.MONTHS:
        year = now.year
        month = now.month
        total_months = (year - 1970) * 12 + (month - 1)
        window_index = total_months // duration.value
        window_start_months = window_index * duration.value
        start_year = 1970 + window_start_months // 12
        start_month = (window_start_months % 12) + 1
        return datetime(start_year, start_month, 1, tzinfo=timezone.utc)

    raise ValueError(f"Unknown duration type: {duration.unit}")


def _compute_window_end(
    duration: BudgetDuration,
    window_start: datetime,
) -> datetime:
    """Compute the end of the current fixed window."""
    if duration.unit == BudgetDurationUnit.MINUTES:
        return window_start + timedelta(minutes=duration.value)
    elif duration.unit == BudgetDurationUnit.HOURS:
        return window_start + timedelta(hours=duration.value)
    elif duration.unit == BudgetDurationUnit.DAYS:
        return window_start + timedelta(days=duration.value)
    elif duration.unit == BudgetDurationUnit.WEEKS:
        return window_start + timedelta(weeks=duration.value)
    elif duration.unit == BudgetDurationUnit.MONTHS:
        year = window_start.year
        month = window_start.month + duration.value
        while month > 12:
            month -= 12
            year += 1
        return datetime(year, month, 1, tzinfo=timezone.utc)

    raise ValueError(f"Unknown duration type: {duration.unit}")


def _policy_applies(policy: GatewayBudgetPolicy, workspace: str | None) -> bool:
    """Check if a policy applies to a given workspace.

    GLOBAL policies apply to all workspaces. WORKSPACE policies only apply
    when the request workspace matches the policy's workspace.
    """
    if policy.target_scope == BudgetTargetScope.GLOBAL:
        return True
    effective_workspace = workspace or DEFAULT_WORKSPACE_NAME
    return policy.workspace == effective_workspace
