"""In-memory budget tracker for AI Gateway cost management.

Tracks cumulative cost per budget policy and fixed time window.
Cost accumulation lives in memory and resets on server restart (acceptable for P0).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from mlflow.entities.gateway_budget_policy import (
    BudgetDurationType,
    BudgetOnExceeded,
    BudgetTargetType,
    GatewayBudgetPolicy,
)

_logger = logging.getLogger(__name__)

# How often (seconds) to re-fetch policies from the database
_REFRESH_INTERVAL_SECONDS = 60


def _compute_window_start(
    duration_type: BudgetDurationType,
    duration_value: int,
    now: datetime,
) -> datetime:
    """Compute the start of the current fixed window for a given policy.

    Windows are aligned to:
    - HOURS: aligned to epoch hours (e.g., duration_value=2 → 0:00, 2:00, 4:00, …)
    - DAYS: aligned to epoch days (e.g., duration_value=7 → weekly from epoch)
    - MONTHS: aligned to first of the month
    """
    if duration_type == BudgetDurationType.HOURS:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        hours_since_epoch = (now - epoch).total_seconds() / 3600
        window_index = int(hours_since_epoch) // duration_value
        window_start_hours = window_index * duration_value
        return epoch + timedelta(hours=window_start_hours)

    elif duration_type == BudgetDurationType.DAYS:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        days_since_epoch = (now - epoch).days
        window_index = days_since_epoch // duration_value
        window_start_days = window_index * duration_value
        return epoch + timedelta(days=window_start_days)

    elif duration_type == BudgetDurationType.MONTHS:
        # Align to first of month, stepping by duration_value months
        year = now.year
        month = now.month
        # Find the window start: step back to the aligned month
        # Month numbering: 0-indexed from some epoch month
        total_months = (year - 1970) * 12 + (month - 1)
        window_index = total_months // duration_value
        window_start_months = window_index * duration_value
        start_year = 1970 + window_start_months // 12
        start_month = (window_start_months % 12) + 1
        return datetime(start_year, start_month, 1, tzinfo=timezone.utc)

    raise ValueError(f"Unknown duration type: {duration_type}")


def _compute_window_end(
    duration_type: BudgetDurationType,
    duration_value: int,
    window_start: datetime,
) -> datetime:
    """Compute the end of the current fixed window."""
    if duration_type == BudgetDurationType.HOURS:
        return window_start + timedelta(hours=duration_value)
    elif duration_type == BudgetDurationType.DAYS:
        return window_start + timedelta(days=duration_value)
    elif duration_type == BudgetDurationType.MONTHS:
        # Advance by duration_value months
        year = window_start.year
        month = window_start.month + duration_value
        while month > 12:
            month -= 12
            year += 1
        return datetime(year, month, 1, tzinfo=timezone.utc)

    raise ValueError(f"Unknown duration type: {duration_type}")


@dataclass
class _BudgetWindow:
    """Tracks spend within a single fixed time window for one policy."""

    policy: GatewayBudgetPolicy
    window_start: datetime
    window_end: datetime
    cumulative_cost_usd: float = 0.0
    crossed: bool = False


@dataclass
class BudgetTracker:
    """Thread-safe in-memory budget tracker.

    Tracks cumulative cost per budget policy within fixed time windows.
    Policies are periodically refreshed from the database.
    """

    _windows: dict[str, _BudgetWindow] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_refresh_time: float = 0.0

    def needs_refresh(self) -> bool:
        """Check whether policies should be re-fetched from the database."""
        return (time.monotonic() - self._last_refresh_time) >= _REFRESH_INTERVAL_SECONDS

    def load_policies(self, policies: list[GatewayBudgetPolicy]) -> None:
        """Load or refresh policies from the database.

        Preserves accumulated cost for unchanged windows. Removes windows
        for policies that no longer exist.
        """
        now = datetime.now(timezone.utc)
        new_windows: dict[str, _BudgetWindow] = {}

        with self._lock:
            for policy in policies:
                pid = policy.budget_policy_id
                window_start = _compute_window_start(
                    policy.duration_type, policy.duration_value, now
                )
                window_end = _compute_window_end(
                    policy.duration_type, policy.duration_value, window_start
                )

                existing = self._windows.get(pid)
                if existing and existing.window_start == window_start:
                    # Same window: preserve accumulated cost and crossed flag
                    existing.policy = policy
                    existing.window_end = window_end
                    new_windows[pid] = existing
                else:
                    # New window or new policy: reset
                    new_windows[pid] = _BudgetWindow(
                        policy=policy,
                        window_start=window_start,
                        window_end=window_end,
                    )

            self._windows = new_windows
            self._last_refresh_time = time.monotonic()

    def record_cost(
        self,
        cost_usd: float,
        workspace: str | None = None,
    ) -> list[_BudgetWindow]:
        """Record a cost against all applicable policies.

        Args:
            cost_usd: The cost in USD to record.
            workspace: The workspace the request was made from (None for default).

        Returns:
            List of windows that were newly crossed (limit exceeded for the first
            time in this window). Used to trigger webhook alerts.
        """
        newly_crossed: list[_BudgetWindow] = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for window in self._windows.values():
                # Check if window is still active
                if now >= window.window_end:
                    # Window expired; reset it
                    window.window_start = _compute_window_start(
                        window.policy.duration_type,
                        window.policy.duration_value,
                        now,
                    )
                    window.window_end = _compute_window_end(
                        window.policy.duration_type,
                        window.policy.duration_value,
                        window.window_start,
                    )
                    window.cumulative_cost_usd = 0.0
                    window.crossed = False

                if not _policy_applies(window.policy, workspace):
                    continue

                window.cumulative_cost_usd += cost_usd

                if (
                    not window.crossed
                    and window.cumulative_cost_usd >= window.policy.limit_usd
                ):
                    window.crossed = True
                    newly_crossed.append(window)

        return newly_crossed

    def is_budget_exceeded(
        self,
        workspace: str | None = None,
    ) -> tuple[bool, GatewayBudgetPolicy | None]:
        """Check if any REJECT-capable policy is exceeded.

        Args:
            workspace: The workspace to check against.

        Returns:
            Tuple of (exceeded, policy). If exceeded is True, policy is the
            first exceeded policy found.
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            for window in self._windows.values():
                if now >= window.window_end:
                    continue

                if not _policy_applies(window.policy, workspace):
                    continue

                if window.policy.on_exceeded not in (
                    BudgetOnExceeded.REJECT,
                    BudgetOnExceeded.ALERT_AND_REJECT,
                ):
                    continue

                if window.cumulative_cost_usd >= window.policy.limit_usd:
                    return True, window.policy

        return False, None

    def get_window_info(self, budget_policy_id: str) -> _BudgetWindow | None:
        """Get the current window info for a policy (for payload construction)."""
        with self._lock:
            return self._windows.get(budget_policy_id)


def _policy_applies(policy: GatewayBudgetPolicy, workspace: str | None) -> bool:
    """Check if a policy applies to a given workspace.

    GLOBAL policies apply to all workspaces. WORKSPACE policies only apply
    when the request workspace matches the policy's workspace.
    """
    if policy.target_type == BudgetTargetType.GLOBAL:
        return True
    # WORKSPACE-scoped policy: match on workspace
    if workspace and policy.workspace:
        return policy.workspace == workspace
    return False


# Module-level singleton
_budget_tracker: BudgetTracker | None = None
_tracker_lock = threading.Lock()


def get_budget_tracker() -> BudgetTracker:
    """Get or create the module-level BudgetTracker singleton."""
    global _budget_tracker
    if _budget_tracker is None:
        with _tracker_lock:
            if _budget_tracker is None:
                _budget_tracker = BudgetTracker()
    return _budget_tracker
