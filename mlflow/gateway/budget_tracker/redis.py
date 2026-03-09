"""Redis-backed budget tracker implementation."""

from __future__ import annotations

import json
import logging
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

_logger = logging.getLogger(__name__)

# Redis key prefix for all budget tracker keys
_KEY_PREFIX = "mlflow:budget:"


def _window_key(policy_id: str) -> str:
    return f"{_KEY_PREFIX}window:{policy_id}"


def _policy_key(policy_id: str) -> str:
    return f"{_KEY_PREFIX}policy:{policy_id}"


def _policy_set_key() -> str:
    return f"{_KEY_PREFIX}policies"


def _serialize_policy(policy: GatewayBudgetPolicy) -> str:
    return json.dumps({
        "budget_policy_id": policy.budget_policy_id,
        "budget_unit": policy.budget_unit.value,
        "budget_amount": policy.budget_amount,
        "duration_unit": policy.duration_unit.value,
        "duration_value": policy.duration_value,
        "target_scope": policy.target_scope.value,
        "budget_action": policy.budget_action.value,
        "workspace": policy.workspace,
        "created_at": policy.created_at,
        "last_updated_at": policy.last_updated_at,
    })


def _deserialize_policy(data: str) -> GatewayBudgetPolicy:
    from mlflow.entities.gateway_budget_policy import (
        BudgetDurationUnit,
        BudgetTargetScope,
        BudgetUnit,
    )

    d = json.loads(data)
    return GatewayBudgetPolicy(
        budget_policy_id=d["budget_policy_id"],
        budget_unit=BudgetUnit(d["budget_unit"]),
        budget_amount=d["budget_amount"],
        duration_unit=BudgetDurationUnit(d["duration_unit"]),
        duration_value=d["duration_value"],
        target_scope=BudgetTargetScope(d["target_scope"]),
        budget_action=BudgetAction(d["budget_action"]),
        workspace=d.get("workspace"),
        created_at=d.get("created_at", 0),
        last_updated_at=d.get("last_updated_at", 0),
    )


@dataclass
class RedisBudgetTracker(BudgetTracker):
    """Redis-backed budget tracker for distributed deployments.

    Uses Redis to store budget windows, enabling shared budget tracking
    across multiple gateway instances. Atomic operations (INCRBYFLOAT)
    ensure correct cost accumulation under concurrent access.
    """

    _redis_url: str = "redis://localhost:6379/0"
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        import redis

        if self._client is None:
            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)

    def _ensure_window(self, policy: GatewayBudgetPolicy, now: datetime) -> BudgetWindow:
        """Ensure a valid window exists in Redis for the given policy, returning it."""
        pid = policy.budget_policy_id
        wkey = _window_key(pid)

        window_start = _compute_window_start(policy.duration_unit, policy.duration_value, now)
        window_end = _compute_window_end(policy.duration_unit, policy.duration_value, window_start)

        stored = self._client.hgetall(wkey)

        if stored and stored.get("window_start") == window_start.isoformat():
            return BudgetWindow(
                policy=policy,
                window_start=window_start,
                window_end=window_end,
                cumulative_spend=float(stored.get("cumulative_spend", 0.0)),
                exceeded=stored.get("exceeded", "0") == "1",
            )

        # Window expired or doesn't exist — create a new one
        pipe = self._client.pipeline()
        pipe.hset(
            wkey,
            mapping={
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "cumulative_spend": "0.0",
                "exceeded": "0",
            },
        )
        ttl_seconds = int((window_end - now).total_seconds()) + 3600
        pipe.expire(wkey, ttl_seconds)
        pipe.execute()

        return BudgetWindow(
            policy=policy,
            window_start=window_start,
            window_end=window_end,
            cumulative_spend=0.0,
            exceeded=False,
        )

    def refresh_policies(self, policies: list[GatewayBudgetPolicy]) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        fresh_windows: list[BudgetWindow] = []

        new_policy_ids = set()
        pipe = self._client.pipeline()

        for policy in policies:
            pid = policy.budget_policy_id
            new_policy_ids.add(pid)
            pipe.hset(_policy_key(pid), mapping={"data": _serialize_policy(policy)})

        pipe.execute()

        # Remove stale policies
        existing_ids = self._client.smembers(_policy_set_key())
        stale_ids = existing_ids - new_policy_ids
        if stale_ids:
            pipe = self._client.pipeline()
            for stale_id in stale_ids:
                pipe.delete(_window_key(stale_id))
                pipe.delete(_policy_key(stale_id))
                pipe.srem(_policy_set_key(), stale_id)
            pipe.execute()

        # Update the policy set and ensure windows exist
        if new_policy_ids:
            self._client.sadd(_policy_set_key(), *new_policy_ids)

        for policy in policies:
            window = self._ensure_window(policy, now)
            if window.cumulative_spend == 0.0:
                fresh_windows.append(window)

        self.mark_refreshed()
        return fresh_windows

    def record_cost(
        self,
        cost_usd: float,
        workspace: str | None = None,
    ) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        newly_exceeded: list[BudgetWindow] = []

        policy_ids = self._client.smembers(_policy_set_key())

        for pid in policy_ids:
            policy_data = self._client.hget(_policy_key(pid), "data")
            if policy_data is None:
                continue
            policy = _deserialize_policy(policy_data)

            if not _policy_applies(policy, workspace):
                continue

            window = self._ensure_window(policy, now)

            # Atomically increment spend
            new_spend = self._client.hincrbyfloat(_window_key(pid), "cumulative_spend", cost_usd)

            if new_spend >= policy.budget_amount:
                was_exceeded = self._client.hget(_window_key(pid), "exceeded")
                if was_exceeded != "1":
                    self._client.hset(_window_key(pid), "exceeded", "1")
                    window.cumulative_spend = new_spend
                    window.exceeded = True
                    newly_exceeded.append(window)

        return newly_exceeded

    def should_reject_request(
        self,
        workspace: str | None = None,
    ) -> tuple[bool, GatewayBudgetPolicy | None]:
        now = datetime.now(timezone.utc)

        policy_ids = self._client.smembers(_policy_set_key())

        for pid in policy_ids:
            policy_data = self._client.hget(_policy_key(pid), "data")
            if policy_data is None:
                continue
            policy = _deserialize_policy(policy_data)

            if not _policy_applies(policy, workspace):
                continue

            if policy.budget_action != BudgetAction.REJECT:
                continue

            wkey = _window_key(pid)
            stored = self._client.hgetall(wkey)
            if not stored:
                continue

            window_end_str = stored.get("window_end")
            if window_end_str:
                window_end = datetime.fromisoformat(window_end_str)
                if now >= window_end:
                    continue

            spend = float(stored.get("cumulative_spend", 0.0))
            if spend >= policy.budget_amount:
                return True, policy

        return False, None

    def backfill_spend(self, spend_by_policy: dict[str, float]) -> None:
        pipe = self._client.pipeline()

        for budget_policy_id, spend in spend_by_policy.items():
            wkey = _window_key(budget_policy_id)
            if not self._client.exists(wkey):
                continue

            policy_data = self._client.hget(_policy_key(budget_policy_id), "data")
            if policy_data is None:
                continue
            policy = _deserialize_policy(policy_data)

            pipe.hset(wkey, "cumulative_spend", str(spend))
            exceeded = "1" if spend >= policy.budget_amount else "0"
            pipe.hset(wkey, "exceeded", exceeded)

        pipe.execute()

    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        wkey = _window_key(budget_policy_id)
        stored = self._client.hgetall(wkey)
        if not stored:
            return None

        policy_data = self._client.hget(_policy_key(budget_policy_id), "data")
        if policy_data is None:
            return None

        policy = _deserialize_policy(policy_data)

        return BudgetWindow(
            policy=policy,
            window_start=datetime.fromisoformat(stored["window_start"]),
            window_end=datetime.fromisoformat(stored["window_end"]),
            cumulative_spend=float(stored.get("cumulative_spend", 0.0)),
            exceeded=stored.get("exceeded", "0") == "1",
        )
