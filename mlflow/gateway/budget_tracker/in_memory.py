"""In-memory budget tracker implementation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mlflow.entities.gateway_budget_policy import BudgetAction, GatewayBudgetPolicy
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    BudgetWindow,
    _compute_window_end,
    _compute_window_start,
    _policy_applies,
)


@dataclass
class InMemoryBudgetTracker(BudgetTracker):
    """Thread-safe in-memory budget tracker.

    Tracks cumulative cost per budget policy within fixed time windows.
    Policies are periodically refreshed from the database.
    Cost accumulation resets on server restart.
    """

    _windows: dict[str, BudgetWindow] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def refresh_policies(self, policies: list[GatewayBudgetPolicy]) -> list[BudgetWindow]:
        """Load or refresh policies from the database.

        Preserves accumulated cost for unchanged windows. Removes windows
        for policies that no longer exist.

        Returns:
            All current windows. In a multi-worker setup each worker tracks
            spend independently, so all windows are returned so callers can
            sync cumulative spend from authoritative trace data on every
            refresh interval.
        """
        now = datetime.now(timezone.utc)
        active_windows: dict[str, BudgetWindow] = {}

        with self._lock:
            for policy in policies:
                pid = policy.budget_policy_id
                window_start = _compute_window_start(
                    policy.duration_unit, policy.duration_value, now
                )
                window_end = _compute_window_end(
                    policy.duration_unit, policy.duration_value, window_start
                )

                existing = self._windows.get(pid)
                if existing and existing.window_start == window_start:
                    existing.policy = policy
                    existing.window_end = window_end
                    active_windows[pid] = existing
                else:
                    active_windows[pid] = BudgetWindow(
                        policy=policy,
                        window_start=window_start,
                        window_end=window_end,
                    )

            self._windows = active_windows
            self.mark_refreshed()

        return list(active_windows.values())

    def record_cost(
        self,
        cost_usd: float,
        workspace: str | None = None,
    ) -> list[BudgetWindow]:
        """Record a cost against all applicable policies.

        Args:
            cost_usd: The cost in USD to record.
            workspace: The workspace the request was made from.

        Returns:
            List of windows that were newly exceeded (limit exceeded for the first
            time in this window). Used to trigger webhook alerts.
        """
        newly_exceeded: list[BudgetWindow] = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for window in self._windows.values():
                if now >= window.window_end:
                    window.window_start = _compute_window_start(
                        window.policy.duration_unit,
                        window.policy.duration_value,
                        now,
                    )
                    window.window_end = _compute_window_end(
                        window.policy.duration_unit,
                        window.policy.duration_value,
                        window.window_start,
                    )
                    window.cumulative_spend = 0.0
                    window.exceeded = False

                if not _policy_applies(window.policy, workspace):
                    continue

                window.cumulative_spend += cost_usd

                if not window.exceeded and window.cumulative_spend >= window.policy.budget_amount:
                    window.exceeded = True
                    newly_exceeded.append(window)

        return newly_exceeded

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
        now = datetime.now(timezone.utc)

        with self._lock:
            for window in self._windows.values():
                if now >= window.window_end:
                    continue

                if not _policy_applies(window.policy, workspace):
                    continue

                if window.policy.budget_action != BudgetAction.REJECT:
                    continue

                if window.cumulative_spend >= window.policy.budget_amount:
                    return True, window

        return False, None

    def backfill_spend(self, spend_by_policy: dict[str, float]) -> None:
        """Sync cumulative spend on windows from authoritative trace data.

        Uses max(current, db_value) so that in-process spend recorded since
        the last trace flush is never lost due to DB write lag.
        """
        with self._lock:
            for budget_policy_id, spend in spend_by_policy.items():
                window = self._windows.get(budget_policy_id)
                if window is None:
                    continue
                window.cumulative_spend = max(window.cumulative_spend, spend)
                window.exceeded = window.cumulative_spend >= window.policy.budget_amount

    def get_all_windows(self) -> list[BudgetWindow]:
        with self._lock:
            return list(self._windows.values())

    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        """Get the current window info for a policy (for payload construction)."""
        with self._lock:
            return self._windows.get(budget_policy_id)
