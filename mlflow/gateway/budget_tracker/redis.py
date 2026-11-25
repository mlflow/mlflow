"""Redis-backed budget tracker implementation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import redis

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    BudgetWindowType,
    GatewayBudgetPolicy,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    BudgetWindow,
    SlidingWindowHistory,
    _compute_window_end,
    _compute_window_start,
    _from_epoch_ms,
    _policy_applies,
    _sliding_bucket_expiry,
    _sliding_bucket_start_ms,
    _sliding_bucket_width_ms,
    _sliding_window_cutoff_ms,
    _to_epoch_ms,
)

_logger = logging.getLogger(__name__)

# Redis key prefix for all budget tracker keys
_KEY_PREFIX = "mlflow:budget:"

# Lua script: atomically initialize or roll a window only if the stored window
# no longer describes the same budget period (or the key doesn't exist).
# The duration is stored with the bounds so that a duration edit rolls the
# window even when the new bounds happen to start at the same instant.
# Rolling deletes the hash first so that no field of a sliding window that
# used to live under this key survives into the fixed window.
# The window also records the revision (last_updated_at) of the policy version that
# last wrote it. A caller holding an older version gets stale=1 and no writes, so a
# process that has not picked up an edit yet cannot roll the window back to the old
# duration or replace the sliding window the policy switched to. It reloads the
# policy and retries instead.
# Returns {created, cumulative_spend, exceeded, stale} so callers don't need a
# separate round-trip to read back the state.
_ENSURE_WINDOW_LUA = """
local wkey = KEYS[1]
local new_start = ARGV[1]
local new_end = ARGV[2]
local ttl = tonumber(ARGV[3])
local duration_unit = ARGV[4]
local duration_value = ARGV[5]
local revision = tonumber(ARGV[6])

local stored_revision = tonumber(redis.call('HGET', wkey, 'policy_revision'))
if stored_revision and revision < stored_revision then
    return {0, '0.0', '0', 1}
end

local current = redis.call('HMGET', wkey, 'window_start', 'duration_unit', 'duration_value')
if current[1] == new_start
    and current[2] == duration_unit
    and current[3] == duration_value then
    if not stored_revision or revision > stored_revision then
        redis.call('HSET', wkey, 'policy_revision', ARGV[6])
    end
    local spend = redis.call('HGET', wkey, 'cumulative_spend') or '0.0'
    local exceeded = redis.call('HGET', wkey, 'exceeded') or '0'
    return {0, spend, exceeded, 0}
end

redis.call('DEL', wkey)
redis.call('HSET', wkey,
    'window_start', new_start,
    'window_end', new_end,
    'duration_unit', duration_unit,
    'duration_value', duration_value,
    'cumulative_spend', '0.0',
    'exceeded', '0',
    'policy_revision', ARGV[6])

if ttl and ttl > 0 then
    redis.call('EXPIRE', wkey, ttl)
end

return {1, '0.0', '0', 0}
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


def _window_duration_matches(stored: dict[str, str], policy: GatewayBudgetPolicy) -> bool:
    """Whether a stored window's bounds were computed from this policy's duration.

    Window bounds come from the duration, so a window written by a process that has
    not picked up a duration edit yet describes a different budget period than the
    one this policy defines.
    """
    stored_duration = (stored.get("duration_unit"), stored.get("duration_value"))
    return stored_duration == (policy.duration.unit.value, str(policy.duration.value))


# Lua script: seed a window's spend from trace history and set the exceeded flag,
# reporting whether this call was the one that flipped it. Mirrors _RECORD_COST_LUA
# so that a crossing first observed through backfill is reported exactly once, even
# when several gateway processes refresh at the same time.
#
# Takes max(stored, seeded) like InMemoryBudgetTracker.backfill_spend: this hash is
# the shared counter record_cost bumps with HINCRBYFLOAT, and the trace history the
# seed comes from lags behind it. Writing the seed unconditionally would regress the
# counter, which un-sets exceeded, lets the same window alert twice as live spend
# re-crosses the limit, and drops a REJECT policy below its limit until it does.
#
# Returns {flipped, cumulative_spend, window_start, window_end} so the caller can
# build the window from this same atomic read rather than a second round-trip that
# could observe a rolled window.
_BACKFILL_SPEND_LUA = """
local wkey = KEYS[1]
local spend = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

if redis.call('EXISTS', wkey) == 0 then
    return {0, '0', '', ''}
end

local was_exceeded = redis.call('HGET', wkey, 'exceeded')
local current = tonumber(redis.call('HGET', wkey, 'cumulative_spend') or '0')
local new_spend = math.max(current, spend)

local exceeded = '0'
if new_spend >= limit then
    exceeded = '1'
end

redis.call('HSET', wkey, 'cumulative_spend', tostring(new_spend), 'exceeded', exceeded)

local flipped = 0
if exceeded == '1' and was_exceeded ~= '1' then
    flipped = 1
end

return {
    flipped,
    tostring(new_spend),
    redis.call('HGET', wkey, 'window_start'),
    redis.call('HGET', wkey, 'window_end'),
}
"""


# Lua script: bring a sliding window up to date, optionally adding spend to it.
#
# A sliding window is one hash holding its config plus a 'b:<bucket_start_ms>' field per
# bucket. The config is stored so that an edit to the duration (and with it the bucket
# width) or a switch from a fixed window discards buckets measured under the old config.
# Buckets that start before the cutoff have aged out and are deleted.
#
# ARGV: duration_unit, duration_value, bucket_width_ms, cutoff_ms, limit, ttl,
# policy_revision, mode, ... where mode is one of:
#   'check'    - never create or reset the window: one that is missing or written under
#                another config reads as empty
#   'ensure'   - create or reset the window if needed
#   'record'   - also add ARGV[10] to the bucket starting at ARGV[9]
#   'backfill' - also raise each (bucket_start, spend) pair in ARGV[9..] to at least
#                that spend, taking max(stored, seeded) like _BACKFILL_SPEND_LUA
#
# As in _ENSURE_WINDOW_LUA, a caller holding an older version of the policy than the one
# that last wrote the window gets stale=1 and no writes. Otherwise it could reset the
# window to the old config, or judge the exceeded flag against the old limit and clear it
# so that the next crossing alerts a second time.
#
# Aging out only lowers spend, so 'check' and 'ensure' clear the exceeded flag once spend
# drops below the limit but never set it. 'record' and 'backfill' set it and report the
# crossing exactly once, even with several gateway processes writing at the same time.
#
# Returns {created, flipped, spend, buckets, stale}, where buckets is a flat
# [start, spend, start, spend, ...] list of the live buckets.
_SLIDING_WINDOW_LUA = """
local wkey = KEYS[1]
local duration_unit = ARGV[1]
local duration_value = ARGV[2]
local width = ARGV[3]
local cutoff = tonumber(ARGV[4])
local limit = tonumber(ARGV[5])
local ttl = tonumber(ARGV[6])
local revision = tonumber(ARGV[7])
local mode = ARGV[8]

local stored_revision = tonumber(redis.call('HGET', wkey, 'policy_revision'))
if stored_revision and revision < stored_revision then
    return {0, 0, '0', {}, 1}
end

local created = 0
local config = redis.call(
    'HMGET', wkey, 'window_type', 'duration_unit', 'duration_value', 'bucket_width_ms')
if config[1] ~= 'SLIDING'
    or config[2] ~= duration_unit
    or config[3] ~= duration_value
    or config[4] ~= width then
    if mode == 'check' then
        return {0, 0, '0', {}, 0}
    end
    redis.call('DEL', wkey)
    redis.call('HSET', wkey,
        'window_type', 'SLIDING',
        'duration_unit', duration_unit,
        'duration_value', duration_value,
        'bucket_width_ms', width,
        'exceeded', '0',
        'policy_revision', ARGV[7])
    created = 1
elseif not stored_revision or revision > stored_revision then
    redis.call('HSET', wkey, 'policy_revision', ARGV[7])
end

-- Deletes aged-out buckets and returns the spend and flat bucket list of the live ones.
local function live_buckets()
    local spend = 0
    local buckets = {}
    local fields = redis.call('HGETALL', wkey)
    for i = 1, #fields, 2 do
        local field = fields[i]
        if string.sub(field, 1, 2) == 'b:' then
            local start = string.sub(field, 3)
            if tonumber(start) < cutoff then
                redis.call('HDEL', wkey, field)
            else
                spend = spend + tonumber(fields[i + 1])
                table.insert(buckets, start)
                table.insert(buckets, fields[i + 1])
            end
        end
    end
    return spend, buckets
end

-- Age out first: a flag set by spend that has since aged out must not hide a new crossing.
local spend, buckets = live_buckets()
local was_exceeded = redis.call('HGET', wkey, 'exceeded') == '1' and spend >= limit
local exceeded = was_exceeded

if mode == 'record' then
    if tonumber(ARGV[9]) >= cutoff then
        redis.call('HINCRBYFLOAT', wkey, 'b:' .. ARGV[9], ARGV[10])
    end
elseif mode == 'backfill' then
    for i = 9, #ARGV, 2 do
        local field = 'b:' .. ARGV[i]
        local stored = tonumber(redis.call('HGET', wkey, field) or '0')
        if tonumber(ARGV[i + 1]) > stored then
            redis.call('HSET', wkey, field, ARGV[i + 1])
        end
    end
end

if mode == 'record' or mode == 'backfill' then
    spend, buckets = live_buckets()
    exceeded = spend >= limit
end
redis.call('HSET', wkey, 'exceeded', exceeded and '1' or '0')

local flipped = 0
if exceeded and not was_exceeded then
    flipped = 1
end

if ttl and ttl > 0 then
    redis.call('EXPIRE', wkey, ttl)
end

return {created, flipped, tostring(spend), buckets, 0}
"""

# Lua script: store a policy unless Redis already holds a newer version of it. A refresh
# that read the database before an edit must not overwrite the edited version another
# process stored after it: stale callers reload the policy from here, and windows rely on
# it being at least as new as any version that wrote them.
_STORE_POLICY_LUA = """
local stored = tonumber(redis.call('HGET', KEYS[1], 'revision'))
if stored and tonumber(ARGV[2]) < stored then
    return 0
end
redis.call('HSET', KEYS[1], 'data', ARGV[1], 'revision', ARGV[2])
return 1
"""

_SlidingMode = Literal["check", "ensure", "record", "backfill"]


@dataclass
class _WindowUpdate:
    """Outcome of running a window script for one version of a policy.

    ``stale`` means a newer version of the policy last wrote the window, so the script left
    it untouched and ``window`` is None.
    """

    window: BudgetWindow | None
    created: bool = False
    flipped: bool = False
    stale: bool = False


def _stored_revision(stored: dict[str, str]) -> int:
    """Revision of the policy version that last wrote a window (-1 if not recorded)."""
    return int(stored.get("policy_revision", -1))


def _is_sliding_hash(stored: dict[str, str]) -> bool:
    return stored.get("window_type") == BudgetWindowType.SLIDING.value


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
        "duration_unit": policy.duration.unit.value,
        "duration_value": policy.duration.value,
        "target_scope": policy.target_scope.value,
        "budget_action": policy.budget_action.value,
        "workspace": policy.workspace,
        "target_value": policy.target_value,
        "created_at": policy.created_at,
        "last_updated_at": policy.last_updated_at,
        "window_type": policy.window_type.value,
    })


def _deserialize_policy(data: str) -> GatewayBudgetPolicy:
    d = json.loads(data)
    return GatewayBudgetPolicy(
        budget_policy_id=d["budget_policy_id"],
        budget_unit=BudgetUnit(d["budget_unit"]),
        budget_amount=d["budget_amount"],
        duration=BudgetDuration(
            unit=BudgetDurationUnit(d["duration_unit"]),
            value=d["duration_value"],
        ),
        target_scope=BudgetTargetScope(d["target_scope"]),
        budget_action=BudgetAction(d["budget_action"]),
        workspace=d.get("workspace"),
        target_value=d.get("target_value"),
        created_at=d.get("created_at", 0),
        last_updated_at=d.get("last_updated_at", 0),
        # Written by a process that predates sliding windows.
        window_type=BudgetWindowType(d.get("window_type", BudgetWindowType.FIXED.value)),
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
    # In-process cache of deserialized policies, populated by refresh_policies
    # to avoid repeated Redis round-trips and JSON deserialization in record_cost.
    _policy_cache: dict[str, GatewayBudgetPolicy] = field(default_factory=dict, repr=False)

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

    def _policy_for(self, budget_policy_id: str) -> GatewayBudgetPolicy | None:
        # Use in-process cache when available to avoid Redis round-trips
        if policy := self._policy_cache.get(budget_policy_id):
            return policy
        if not (policy_data := self._client.hget(_policy_key(budget_policy_id), "data")):
            return None
        return _deserialize_policy(policy_data)

    def _with_latest_policy(
        self,
        policy: GatewayBudgetPolicy,
        update: Callable[[GatewayBudgetPolicy], _WindowUpdate],
    ) -> _WindowUpdate:
        """Run ``update`` for ``policy``, and again with the newer version if it was stale.

        Redis holds a version of each policy at least as new as any version that wrote its
        window (``_STORE_POLICY_LUA`` keeps it from regressing), so one retry is enough.
        """
        result = update(policy)
        if not result.stale:
            return result
        if not (policy_data := self._client.hget(_policy_key(policy.budget_policy_id), "data")):
            return result
        latest = _deserialize_policy(policy_data)
        if latest.last_updated_at <= policy.last_updated_at:
            return result
        self._policy_cache[latest.budget_policy_id] = latest
        return update(latest)

    def _ensure_window(self, policy: GatewayBudgetPolicy, now: datetime) -> _WindowUpdate:
        """Ensure a valid fixed window exists in Redis for the given policy.

        ``created`` is True if the window was just initialized or rolled over.
        """
        window_start = _compute_window_start(policy.duration, now)
        window_end = _compute_window_end(policy.duration, window_start)

        # Keep the key for 1 hour past window_end so late-arriving requests
        # can still read/update the window before it expires from Redis.
        ttl_seconds = int((window_end - now).total_seconds()) + 3600
        created, cumulative_spend, exceeded, stale = self._client.eval(
            _ENSURE_WINDOW_LUA,
            1,
            _window_key(policy.budget_policy_id),
            window_start.isoformat(),
            window_end.isoformat(),
            ttl_seconds,
            policy.duration.unit.value,
            str(policy.duration.value),
            str(policy.last_updated_at),
        )
        if int(stale):
            return _WindowUpdate(window=None, stale=True)

        window = BudgetWindow(
            policy=policy,
            window_start=window_start,
            window_end=window_end,
            cumulative_spend=float(cumulative_spend),
            exceeded=exceeded == "1",
        )
        return _WindowUpdate(window=window, created=bool(int(created)))

    def _update_sliding_window(
        self,
        policy: GatewayBudgetPolicy,
        now: datetime,
        mode: _SlidingMode,
        *mode_args: str,
    ) -> _WindowUpdate:
        """Run ``_SLIDING_WINDOW_LUA`` on a sliding policy's window."""
        width_ms = _sliding_bucket_width_ms(policy.duration)
        cutoff_ms = _sliding_window_cutoff_ms(policy.duration, width_ms, now)
        newest_bucket_ms = _sliding_bucket_start_ms(_to_epoch_ms(now), width_ms)
        # Keep the key for 1 hour past the point where its newest bucket ages out.
        expires_in = _sliding_bucket_expiry(policy.duration, newest_bucket_ms, width_ms) - now
        ttl_seconds = int(expires_in.total_seconds()) + 3600
        created, flipped, spend, flat_buckets, stale = self._client.eval(
            _SLIDING_WINDOW_LUA,
            1,
            _window_key(policy.budget_policy_id),
            policy.duration.unit.value,
            str(policy.duration.value),
            str(width_ms),
            str(cutoff_ms),
            str(policy.budget_amount),
            ttl_seconds,
            str(policy.last_updated_at),
            mode,
            *mode_args,
        )
        if int(stale):
            return _WindowUpdate(window=None, stale=True)

        bucket_spend = {
            int(flat_buckets[i]): float(flat_buckets[i + 1]) for i in range(0, len(flat_buckets), 2)
        }
        window = BudgetWindow(
            policy=policy,
            window_start=_from_epoch_ms(cutoff_ms),
            window_end=now,
            cumulative_spend=float(spend),
            exceeded=float(spend) >= policy.budget_amount,
            bucket_spend=bucket_spend,
        )
        return _WindowUpdate(window=window, created=bool(int(created)), flipped=bool(int(flipped)))

    def _ensure(self, policy: GatewayBudgetPolicy, now: datetime) -> _WindowUpdate:
        if policy.window_type == BudgetWindowType.SLIDING:
            return self._update_sliding_window(policy, now, "ensure")
        return self._ensure_window(policy, now)

    def _read_fixed_window(self, policy: GatewayBudgetPolicy, now: datetime) -> _WindowUpdate:
        """Read a fixed policy's current window without writing to it."""
        if not (stored := self._client.hgetall(_window_key(policy.budget_policy_id))):
            return _WindowUpdate(window=None)

        if _stored_revision(stored) > policy.last_updated_at:
            return _WindowUpdate(window=None, stale=True)

        # Still holds the buckets of an older, sliding version of this policy that nothing
        # has replaced yet; it has no fixed window's bounds to compare against.
        if _is_sliding_hash(stored):
            return _WindowUpdate(window=None)

        if not _window_duration_matches(stored, policy):
            # The window was written by a process still on an older version of this
            # policy. Its bounds and this budget_amount measure different periods,
            # so the next refresh or recorded cost has to roll it before it can be
            # compared against this limit.
            return _WindowUpdate(window=None)

        window = self._build_window(policy, stored)
        if now >= window.window_end:
            # Stale window: the budget period has ended; don't reject based on old data.
            return _WindowUpdate(window=None)
        return _WindowUpdate(window=window)

    def refresh_policies(self, policies: list[GatewayBudgetPolicy]) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        fresh_windows: list[BudgetWindow] = []

        new_policy_ids = {p.budget_policy_id for p in policies}

        # Remove stale policies before writing new data to avoid deleting
        # a re-added policy's data in the same refresh call.
        existing_ids = self._client.smembers(_policy_set_key())
        if stale_ids := existing_ids - new_policy_ids:
            pipe = self._client.pipeline()
            for stale_id in stale_ids:
                pipe.delete(_window_key(stale_id))
                pipe.delete(_policy_key(stale_id))
                pipe.srem(_policy_set_key(), stale_id)
            pipe.execute()

        # Write new/updated policy data and update the policy set
        pipe = self._client.pipeline()
        for policy in policies:
            pipe.eval(
                _STORE_POLICY_LUA,
                1,
                _policy_key(policy.budget_policy_id),
                _serialize_policy(policy),
                str(policy.last_updated_at),
            )
        pipe.execute()

        if new_policy_ids:
            self._client.sadd(_policy_set_key(), *new_policy_ids)

        # Update in-process policy cache
        self._policy_cache = {p.budget_policy_id: p for p in policies}

        for policy in policies:
            update = self._with_latest_policy(policy, lambda p: self._ensure(p, now))
            if update.window is None:
                continue
            # A sliding window never rolls over, so syncing it with trace history is the
            # only way to recover spend it missed: a failed history query, traces exported
            # after the last sync, or a window first created by a recorded cost. The
            # per-bucket max makes syncing it on every refresh idempotent. A fixed window
            # only needs seeding when it is created.
            if update.created or update.window.policy.window_type == BudgetWindowType.SLIDING:
                fresh_windows.append(update.window)

        self.mark_refreshed()
        return fresh_windows

    def record_cost(
        self,
        cost_usd: float,
        workspace: str | None = None,
        endpoint_id: str | None = None,
        username: str | None = None,
        timestamp_ms: int | None = None,
    ) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        newly_exceeded: list[BudgetWindow] = []
        if timestamp_ms is None:
            timestamp_ms = _to_epoch_ms(now)

        def record(policy: GatewayBudgetPolicy) -> _WindowUpdate:
            if not _policy_applies(policy, workspace, endpoint_id=endpoint_id, username=username):
                return _WindowUpdate(window=None)

            if policy.window_type == BudgetWindowType.SLIDING:
                bucket_start_ms = _sliding_bucket_start_ms(
                    timestamp_ms, _sliding_bucket_width_ms(policy.duration)
                )
                return self._update_sliding_window(
                    policy, now, "record", str(bucket_start_ms), str(cost_usd)
                )

            update = self._ensure_window(policy, now)
            if update.window is None:
                return update

            # Atomically increment spend and flip exceeded flag
            new_spend, just_flipped = self._client.eval(
                _RECORD_COST_LUA,
                1,
                _window_key(policy.budget_policy_id),
                str(cost_usd),
                str(policy.budget_amount),
            )
            if int(just_flipped):
                update.window.cumulative_spend = float(new_spend)
                update.window.exceeded = True
                update.flipped = True
            return update

        for pid in self._client.smembers(_policy_set_key()):
            if policy := self._policy_for(pid):
                update = self._with_latest_policy(policy, record)
                if update.flipped:
                    newly_exceeded.append(update.window)

        return newly_exceeded

    def should_reject_request(
        self,
        workspace: str | None = None,
        endpoint_id: str | None = None,
        username: str | None = None,
    ) -> tuple[bool, BudgetWindow | None]:
        now = datetime.now(timezone.utc)

        def rejecting_window(policy: GatewayBudgetPolicy) -> _WindowUpdate:
            if not _policy_applies(policy, workspace, endpoint_id=endpoint_id, username=username):
                return _WindowUpdate(window=None)

            if policy.budget_action != BudgetAction.REJECT:
                return _WindowUpdate(window=None)

            if policy.window_type == BudgetWindowType.SLIDING:
                update = self._update_sliding_window(policy, now, "check")
            else:
                update = self._read_fixed_window(policy, now)
            if update.window and update.window.cumulative_spend < policy.budget_amount:
                return _WindowUpdate(window=None)
            return update

        for pid in self._client.smembers(_policy_set_key()):
            if policy := self._policy_for(pid):
                if window := self._with_latest_policy(policy, rejecting_window).window:
                    return True, window

        return False, None

    def backfill_spend(self, spend_by_policy: dict[str, float]) -> list[BudgetWindow]:
        newly_exceeded: list[BudgetWindow] = []

        for budget_policy_id, spend in spend_by_policy.items():
            if not (policy_data := self._client.hget(_policy_key(budget_policy_id), "data")):
                continue
            policy = _deserialize_policy(policy_data)
            if policy.window_type == BudgetWindowType.SLIDING:
                continue
            wkey = _window_key(budget_policy_id)

            # The window may be absent (never created, or expired); the script writes
            # nothing in that case, matching the previous existence check.
            flipped, new_spend, window_start, window_end = self._client.eval(
                _BACKFILL_SPEND_LUA,
                1,
                wkey,
                str(spend),
                str(policy.budget_amount),
            )

            if int(flipped):
                newly_exceeded.append(
                    BudgetWindow(
                        policy=policy,
                        window_start=datetime.fromisoformat(window_start),
                        window_end=datetime.fromisoformat(window_end),
                        cumulative_spend=float(new_spend),
                        exceeded=True,
                    )
                )

        return newly_exceeded

    def backfill_bucket_spend(self, histories: list[SlidingWindowHistory]) -> list[BudgetWindow]:
        now = datetime.now(timezone.utc)
        newly_exceeded: list[BudgetWindow] = []

        for history in histories:
            policy = history.policy
            if policy.window_type != BudgetWindowType.SLIDING:
                continue
            if not self._client.exists(_policy_key(policy.budget_policy_id)):
                continue
            seeded = [str(value) for item in sorted(history.bucket_spend.items()) for value in item]
            # A history read for an older version of the policy comes back stale and is
            # dropped rather than retried, since its buckets may have another width.
            # Sliding windows are synced on every refresh, so the next one reads them again.
            update = self._update_sliding_window(policy, now, "backfill", *seeded)
            if update.flipped:
                newly_exceeded.append(update.window)

        return newly_exceeded

    def get_all_windows(self) -> list[BudgetWindow]:
        return [
            window
            for pid in self._client.smembers(_policy_set_key())
            if (window := self._get_window_info(pid))
        ]

    def _build_window(self, policy: GatewayBudgetPolicy, stored: dict[str, str]) -> BudgetWindow:
        return BudgetWindow(
            policy=policy,
            window_start=datetime.fromisoformat(stored["window_start"]),
            window_end=datetime.fromisoformat(stored["window_end"]),
            cumulative_spend=float(stored.get("cumulative_spend", 0.0)),
            exceeded=stored.get("exceeded", "0") == "1",
        )

    def _get_window_info(self, budget_policy_id: str) -> BudgetWindow | None:
        wkey = _window_key(budget_policy_id)
        if not (stored := self._client.hgetall(wkey)):
            return None

        if not (policy_data := self._client.hget(_policy_key(budget_policy_id), "data")):
            return None

        policy = _deserialize_policy(policy_data)
        if policy.window_type == BudgetWindowType.SLIDING:
            return self._update_sliding_window(policy, datetime.now(timezone.utc), "check").window
        if _is_sliding_hash(stored):
            return None
        return self._build_window(policy, stored)
