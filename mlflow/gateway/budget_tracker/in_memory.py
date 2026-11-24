"""In-memory budget tracker implementation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetWindowType,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    BudgetWindow,
    _advance_sliding_window,
    _compute_window_end,
    _compute_window_start,
    _policy_applies,
    _sliding_bucket_start_ms,
    _sliding_bucket_width_ms,
    _to_epoch_ms,
)


def _is_sliding(window: BudgetWindow) -> bool:
    return window.policy.window_type == BudgetWindowType.SLIDING


@dataclass
class InMemoryBudgetTracker(BudgetTracker):
    """Thread-safe in-memory budget tracker.

    Tracks cumulative cost per budget policy within fixed or sliding time windows.
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
                existing = self._windows.get(pid)

                if policy.window_type == BudgetWindowType.SLIDING:
                    # Buckets are sized from the duration, so they only carry over while
                    # the duration is unchanged.
                    if (
                        existing
                        and _is_sliding(existing)
                        and existing.policy.duration == policy.duration
                    ):
                        existing.policy = policy
                        window = existing
                    else:
                        window = BudgetWindow(policy=policy, window_start=now, window_end=now)
                    _advance_sliding_window(window, now)
                    active_windows[pid] = window
                    continue

                window_start = _compute_window_start(policy.duration, now)
                window_end = _compute_window_end(policy.duration, window_start)

                if existing and not _is_sliding(existing) and existing.window_start == window_start:
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
        endpoint_id: str | None = None,
        username: str | None = None,
        timestamp_ms: int | None = None,
    ) -> list[BudgetWindow]:
        """Record a cost against all applicable policies.

        Args:
            cost_usd: The cost in USD to record.
            workspace: The workspace the request was made from.
            endpoint_id: The gateway endpoint the request was routed to.
            username: The authenticated username the request was made by.
            timestamp_ms: When the request started, in epoch milliseconds (defaults to
                now). Sliding windows file the cost under this time.

        Returns:
            List of windows that were newly exceeded (limit exceeded for the first
            time in this window). Used to trigger webhook alerts.
        """
        newly_exceeded: list[BudgetWindow] = []
        now = datetime.now(timezone.utc)
        if timestamp_ms is None:
            timestamp_ms = _to_epoch_ms(now)

        with self._lock:
            for window in self._windows.values():
                if _is_sliding(window):
                    cutoff_ms = _advance_sliding_window(window, now)
                    if not _policy_applies(
                        window.policy, workspace, endpoint_id=endpoint_id, username=username
                    ):
                        continue
                    width_ms = _sliding_bucket_width_ms(window.policy.duration)
                    bucket_start_ms = _sliding_bucket_start_ms(timestamp_ms, width_ms)
                    # A request that started before the window no longer counts toward it,
                    # just as its trace would not.
                    if bucket_start_ms < cutoff_ms:
                        continue
                    window.bucket_spend[bucket_start_ms] = (
                        window.bucket_spend.get(bucket_start_ms, 0.0) + cost_usd
                    )
                    window.cumulative_spend += cost_usd
                    if (
                        not window.exceeded
                        and window.cumulative_spend >= window.policy.budget_amount
                    ):
                        window.exceeded = True
                        newly_exceeded.append(window)
                    continue

                if now >= window.window_end:
                    window.window_start = _compute_window_start(window.policy.duration, now)
                    window.window_end = _compute_window_end(
                        window.policy.duration, window.window_start
                    )
                    window.cumulative_spend = 0.0
                    window.exceeded = False

                if not _policy_applies(
                    window.policy, workspace, endpoint_id=endpoint_id, username=username
                ):
                    continue

                window.cumulative_spend += cost_usd

                if not window.exceeded and window.cumulative_spend >= window.policy.budget_amount:
                    window.exceeded = True
                    newly_exceeded.append(window)

        return newly_exceeded

    def should_reject_request(
        self,
        workspace: str | None = None,
        endpoint_id: str | None = None,
        username: str | None = None,
    ) -> tuple[bool, BudgetWindow | None]:
        """Check if any REJECT-capable policy is exceeded.

        Args:
            workspace: The workspace to check against.
            endpoint_id: The gateway endpoint to check against.
            username: The authenticated username to check against.

        Returns:
            Tuple of (exceeded, window). If exceeded is True, window is the
            first exceeded window found.
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            for window in self._windows.values():
                if _is_sliding(window):
                    _advance_sliding_window(window, now)
                elif now >= window.window_end:
                    continue

                if not _policy_applies(
                    window.policy, workspace, endpoint_id=endpoint_id, username=username
                ):
                    continue

                if window.policy.budget_action != BudgetAction.REJECT:
                    continue

                if window.cumulative_spend >= window.policy.budget_amount:
                    return True, window

        return False, None

    def backfill_spend(self, spend_by_policy: dict[str, float]) -> list[BudgetWindow]:
        """Sync cumulative spend on windows from authoritative trace data.

        Uses max(current, db_value) so that in-process spend recorded since
        the last trace flush is never lost due to DB write lag.

        Returns:
            Windows whose limit this sync crossed for the first time.
        """
        newly_exceeded: list[BudgetWindow] = []

        with self._lock:
            for budget_policy_id, spend in spend_by_policy.items():
                window = self._windows.get(budget_policy_id)
                if window is None or _is_sliding(window):
                    continue
                was_exceeded = window.exceeded
                window.cumulative_spend = max(window.cumulative_spend, spend)
                window.exceeded = window.cumulative_spend >= window.policy.budget_amount
                if window.exceeded and not was_exceeded:
                    newly_exceeded.append(window)

        return newly_exceeded

    def backfill_bucket_spend(
        self, bucket_spend_by_policy: dict[str, dict[int, float]]
    ) -> list[BudgetWindow]:
        """Sync the buckets of sliding windows from authoritative trace data.

        Takes max(current, db_value) per bucket, for the same reason ``backfill_spend``
        does for whole windows.

        Returns:
            Windows whose limit this sync crossed.
        """
        newly_exceeded: list[BudgetWindow] = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for budget_policy_id, buckets in bucket_spend_by_policy.items():
                window = self._windows.get(budget_policy_id)
                if window is None or not _is_sliding(window):
                    continue
                # Age out first: a flag set by spend that has since aged out must not hide
                # a crossing made by the seeded spend.
                _advance_sliding_window(window, now)
                was_exceeded = window.exceeded
                for bucket_start_ms, spend in buckets.items():
                    window.bucket_spend[bucket_start_ms] = max(
                        window.bucket_spend.get(bucket_start_ms, 0.0), spend
                    )
                _advance_sliding_window(window, now)
                window.exceeded = window.cumulative_spend >= window.policy.budget_amount
                if window.exceeded and not was_exceeded:
                    newly_exceeded.append(window)

        return newly_exceeded

    def get_all_windows(self) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        with self._lock:
            for window in self._windows.values():
                if _is_sliding(window):
                    _advance_sliding_window(window, now)
            return list(self._windows.values())

    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        """Get the current window info for a policy (for payload construction)."""
        with self._lock:
            window = self._windows.get(budget_policy_id)
            if window is not None and _is_sliding(window):
                _advance_sliding_window(window, datetime.now(timezone.utc))
            return window
