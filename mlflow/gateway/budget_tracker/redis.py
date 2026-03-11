"""Redis-backed budget tracker implementation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.exceptions import MlflowException
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

# Lua script: atomically initialize or roll a window only if the stored
# window_start differs (or the key doesn't exist). Returns 1 if the window
# was created/rolled, 0 if it already matched.
_ENSURE_WINDOW_LUA = """
local wkey = KEYS[1]
local new_start = ARGV[1]
local new_end = ARGV[2]
local ttl = tonumber(ARGV[3])

local current_start = redis.call('HGET', wkey, 'window_start')
if current_start == new_start then
    return 0
end

redis.call('HSET', wkey,
    'window_start', new_start,
    'window_end', new_end,
    'cumulative_spend', '0.0',
    'exceeded', '0')

if ttl and ttl > 0 then
    redis.call('EXPIRE', wkey, ttl)
end

return 1
"""

# Lua script: atomically increment spend and flip the exceeded flag exactly
# once. Returns [new_spend, just_flipped] where just_flipped is 1 if this
# call was the one that transitioned exceeded from 0 to 1.
_RECORD_COST_LUA = """
local wkey = KEYS[1]
local cost = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

local new_spend = redis.call('HINCRBYFLOAT', wkey, 'cumulative_spend', cost)
new_spend = tonumber(new_spend)

if new_spend >= limit then
    local was_exceeded = redis.call('HGET', wkey, 'exceeded')
    if was_exceeded ~= '1' then
        redis.call('HSET', wkey, 'exceeded', '1')
        return {tostring(new_spend), 1}
    end
end

return {tostring(new_spend), 0}
"""


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
    across multiple gateway instances. Lua scripts ensure atomicity for
    window initialization, cost accumulation, and exceeded-flag transitions.
    """

    _redis_url: str = "redis://localhost:6379/0"
    _client: redis.Redis | None = field(default=None, repr=False)

    def __post_init__(self):
        if self._client is None:
            try:
                import redis
            except ImportError:
                raise MlflowException(
                    "The `redis` package is required for RedisBudgetTracker. "
                    "Install it with: pip install redis"
                )
            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)

    def _ensure_window(
        self, policy: GatewayBudgetPolicy, now: datetime
    ) -> tuple[BudgetWindow, bool]:
        """Ensure a valid window exists in Redis for the given policy.

        Returns:
            Tuple of (window, created) where created is True if the window
            was just initialized or rolled over.
        """
        pid = policy.budget_policy_id
        wkey = _window_key(pid)

        window_start = _compute_window_start(policy.duration_unit, policy.duration_value, now)
        window_end = _compute_window_end(policy.duration_unit, policy.duration_value, window_start)

        ttl_seconds = int((window_end - now).total_seconds()) + 3600
        created = self._client.eval(
            _ENSURE_WINDOW_LUA,
            1,
            wkey,
            window_start.isoformat(),
            window_end.isoformat(),
            ttl_seconds,
        )

        # Read back final state to capture any concurrent writes
        stored = self._client.hgetall(wkey)
        cumulative_spend = float(stored.get("cumulative_spend", 0.0))
        exceeded = stored.get("exceeded", "0") == "1"

        window = BudgetWindow(
            policy=policy,
            window_start=window_start,
            window_end=window_end,
            cumulative_spend=cumulative_spend,
            exceeded=exceeded,
        )
        return window, bool(created)

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
        if stale_ids := existing_ids - new_policy_ids:
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
            window, created = self._ensure_window(policy, now)
            if created:
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
            if not (policy_data := self._client.hget(_policy_key(pid), "data")):
                continue
            policy = _deserialize_policy(policy_data)

            if not _policy_applies(policy, workspace):
                continue

            window, _created = self._ensure_window(policy, now)

            # Atomically increment spend and flip exceeded flag
            result = self._client.eval(
                _RECORD_COST_LUA,
                1,
                _window_key(pid),
                str(cost_usd),
                str(policy.budget_amount),
            )
            new_spend = float(result[0])
            just_flipped = int(result[1])

            if just_flipped:
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
            if not (policy_data := self._client.hget(_policy_key(pid), "data")):
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

            if window_end_str := stored.get("window_end"):
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

            if not (policy_data := self._client.hget(_policy_key(budget_policy_id), "data")):
                continue
            policy = _deserialize_policy(policy_data)

            pipe.hset(wkey, "cumulative_spend", str(spend))
            exceeded = "1" if spend >= policy.budget_amount else "0"
            pipe.hset(wkey, "exceeded", exceeded)

        pipe.execute()

    def get_all_windows(self) -> list[BudgetWindow]:
        return [
            window
            for pid in self._client.smembers(_policy_set_key())
            if (window := self._get_window_info(pid))
        ]

    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        wkey = _window_key(budget_policy_id)
        stored = self._client.hgetall(wkey)
        if not stored:
            return None

        if not (policy_data := self._client.hget(_policy_key(budget_policy_id), "data")):
            return None

        policy = _deserialize_policy(policy_data)

        return BudgetWindow(
            policy=policy,
            window_start=datetime.fromisoformat(stored["window_start"]),
            window_end=datetime.fromisoformat(stored["window_end"]),
            cumulative_spend=float(stored.get("cumulative_spend", 0.0)),
            exceeded=stored.get("exceeded", "0") == "1",
        )
