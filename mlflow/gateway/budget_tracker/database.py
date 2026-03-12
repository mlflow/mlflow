"""Database-backed budget tracker for multi-process deployments.

Queries the database for current spend instead of tracking in-memory,
ensuring accurate budget enforcement across multiple worker processes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetTargetScope,
    GatewayBudgetPolicy,
)
from mlflow.environment_variables import MLFLOW_GATEWAY_BUDGET_CACHE_TTL
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    BudgetWindow,
    _compute_window_end,
    _compute_window_start,
    _policy_applies,
)

if TYPE_CHECKING:
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

_logger = logging.getLogger(__name__)


@dataclass
class _CachedRejectResult:
    """Cached result for should_reject_request."""

    mono_time: float
    exceeded: bool
    window: BudgetWindow | None


@dataclass
class DatabaseBudgetTracker(BudgetTracker):
    """Database-backed budget tracker for multi-process deployments.

    Queries the database for current spend when checking REJECT policies,
    ensuring accurate cross-worker budget enforcement. A short TTL cache
    avoids excessive DB queries on rapid request bursts.

    ALERT webhook detection also queries the database to detect threshold
    crossings across all workers.
    """

    _store: SqlAlchemyStore
    _policies: list[GatewayBudgetPolicy] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reject_cache: dict[str | None, _CachedRejectResult] = field(default_factory=dict)
    # Track which ALERT policies have already fired in the current window
    _alerted: dict[str, datetime] = field(default_factory=dict)

    def refresh_policies(self, policies: list[GatewayBudgetPolicy]) -> list[BudgetWindow]:
        with self._lock:
            self._policies = list(policies)
            self._reject_cache.clear()
            # Clean up alerted state for removed policies
            active_ids = {p.budget_policy_id for p in policies}
            self._alerted = {k: v for k, v in self._alerted.items() if k in active_ids}
            self.mark_refreshed()
        return []  # No backfill needed for DB-backed tracker

    def _query_spend(
        self,
        policy: GatewayBudgetPolicy,
        window_start: datetime,
        window_end: datetime,
    ) -> float:
        start_ms = int(window_start.timestamp() * 1000)
        end_ms = int(window_end.timestamp() * 1000)
        workspace = policy.workspace if policy.target_scope == BudgetTargetScope.WORKSPACE else None
        return self._store.sum_gateway_trace_cost(
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            workspace=workspace,
        )

    def should_reject_request(
        self,
        workspace: str | None = None,
    ) -> tuple[bool, BudgetWindow | None]:
        now_mono = time.monotonic()
        cache_ttl = MLFLOW_GATEWAY_BUDGET_CACHE_TTL.get()

        with self._lock:
            cached = self._reject_cache.get(workspace)
            if cached and (now_mono - cached.mono_time) < cache_ttl:
                return cached.exceeded, cached.window
            policies = list(self._policies)

        now = datetime.now(timezone.utc)
        for policy in policies:
            if policy.budget_action != BudgetAction.REJECT:
                continue
            if not _policy_applies(policy, workspace):
                continue

            window_start = _compute_window_start(policy.duration_unit, policy.duration_value, now)
            window_end = _compute_window_end(
                policy.duration_unit, policy.duration_value, window_start
            )
            if now >= window_end:
                continue

            try:
                spend = self._query_spend(policy, window_start, window_end)
            except Exception:
                _logger.debug(
                    "Failed to query spend for policy %s",
                    policy.budget_policy_id,
                    exc_info=True,
                )
                continue

            if spend >= policy.budget_amount:
                window = BudgetWindow(
                    policy=policy,
                    window_start=window_start,
                    window_end=window_end,
                    cumulative_spend=spend,
                    exceeded=True,
                )
                with self._lock:
                    self._reject_cache[workspace] = _CachedRejectResult(now_mono, True, window)
                return True, window

        with self._lock:
            self._reject_cache[workspace] = _CachedRejectResult(now_mono, False, None)
        return False, None

    def record_cost(
        self,
        cost_usd: float,
        workspace: str | None = None,
    ) -> list[BudgetWindow]:
        with self._lock:
            self._reject_cache.clear()
            policies = list(self._policies)

        newly_exceeded: list[BudgetWindow] = []
        now = datetime.now(timezone.utc)

        for policy in policies:
            if policy.budget_action != BudgetAction.ALERT:
                continue
            if not _policy_applies(policy, workspace):
                continue

            window_start = _compute_window_start(policy.duration_unit, policy.duration_value, now)
            window_end = _compute_window_end(
                policy.duration_unit, policy.duration_value, window_start
            )
            pid = policy.budget_policy_id

            with self._lock:
                alerted_start = self._alerted.get(pid)
                if alerted_start == window_start:
                    continue  # Already alerted in this window

            try:
                spend = self._query_spend(policy, window_start, window_end)
            except Exception:
                _logger.debug("Failed to query spend for policy %s", pid, exc_info=True)
                continue

            # Add cost_usd to account for the in-flight request whose trace
            # may not yet be persisted to the database.
            spend += cost_usd

            if spend >= policy.budget_amount:
                with self._lock:
                    if self._alerted.get(pid) == window_start:
                        continue
                    self._alerted[pid] = window_start
                window = BudgetWindow(
                    policy=policy,
                    window_start=window_start,
                    window_end=window_end,
                    cumulative_spend=spend,
                    exceeded=True,
                )
                newly_exceeded.append(window)

        return newly_exceeded

    def backfill_spend(self, spend_by_policy: dict[str, float]) -> None:
        pass  # Not needed — spend is always queried from the database

    def get_all_windows(self) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        with self._lock:
            policies = list(self._policies)

        windows: list[BudgetWindow] = []
        for policy in policies:
            window_start = _compute_window_start(policy.duration_unit, policy.duration_value, now)
            window_end = _compute_window_end(
                policy.duration_unit, policy.duration_value, window_start
            )
            spend = self._query_spend(policy, window_start, window_end)
            windows.append(
                BudgetWindow(
                    policy=policy,
                    window_start=window_start,
                    window_end=window_end,
                    cumulative_spend=spend,
                    exceeded=spend >= policy.budget_amount,
                )
            )
        return windows

    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            policy = next(
                (p for p in self._policies if p.budget_policy_id == budget_policy_id),
                None,
            )
        if policy is None:
            return None

        window_start = _compute_window_start(policy.duration_unit, policy.duration_value, now)
        window_end = _compute_window_end(policy.duration_unit, policy.duration_value, window_start)
        spend = self._query_spend(policy, window_start, window_end)
        return BudgetWindow(
            policy=policy,
            window_start=window_start,
            window_end=window_end,
            cumulative_spend=spend,
            exceeded=spend >= policy.budget_amount,
        )
