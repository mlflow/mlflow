"""Budget tracker for AI Gateway cost management.

Provides an abstract BudgetTracker interface and window computation helpers.
The concrete InMemoryBudgetTracker lives in ``budget_tracker.in_memory``.
"""

from __future__ import annotations

import calendar
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from mlflow.entities.gateway_budget_policy import (
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetWindowType,
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

# A sliding window keeps its spend in this many epoch-aligned buckets. A bucket counts until
# all of it has aged out of the window, so a cost counts for between one duration and one
# duration plus a bucket width (1/60 of the duration) after its request started. It never
# stops counting early, which keeps REJECT policies conservative.
_SLIDING_WINDOW_BUCKETS = 60

_TIMEDELTA_UNITS = {
    BudgetDurationUnit.MINUTES: "minutes",
    BudgetDurationUnit.HOURS: "hours",
    BudgetDurationUnit.DAYS: "days",
    BudgetDurationUnit.WEEKS: "weeks",
}

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
    """Tracks spend within the current time window for one policy.

    For a FIXED policy the window spans ``[window_start, window_end)``. For a SLIDING policy
    the window ends at the time it was last evaluated (``window_end``) and starts at its
    oldest counted bucket (``window_start``), and ``bucket_spend`` holds the spend of each
    live bucket keyed by the bucket's start in epoch milliseconds.
    """

    policy: GatewayBudgetPolicy
    window_start: datetime
    window_end: datetime
    cumulative_spend: float = 0.0
    exceeded: bool = False
    bucket_spend: dict[int, float] = field(default_factory=dict)

    @property
    def resets_at(self) -> datetime:
        """When the spend next drops below the limit, assuming no new spend.

        A fixed window resets to zero when it ends. A sliding window frees up gradually as
        its oldest buckets age out, so this is when enough of them have aged out.
        """
        if self.policy.window_type != BudgetWindowType.SLIDING:
            return self.window_end
        width_ms = _sliding_bucket_width_ms(self.policy.duration)
        remaining = self.cumulative_spend
        for bucket_start_ms in sorted(self.bucket_spend):
            remaining -= self.bucket_spend[bucket_start_ms]
            if remaining < self.policy.budget_amount:
                return _sliding_bucket_expiry(self.policy.duration, bucket_start_ms, width_ms)
        return self.window_end


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
            Windows that should be synced against authoritative trace data.
        """

    @abstractmethod
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
            workspace: The workspace the request was made from (None for default).
            endpoint_id: The gateway endpoint the request was routed to. Used to
                match ENDPOINT-scoped policies.
            username: The authenticated username the request was made by. Used to
                match USER-scoped policies.
            timestamp_ms: When the request started, in epoch milliseconds (defaults to
                now). Sliding windows file the cost under this time, the same time the
                request's trace records, so that spend synced from trace history lands
                in the same bucket.

        Returns:
            List of windows that were newly exceeded (limit exceeded for the first
            time in this window). Used to trigger webhook alerts.
        """

    @abstractmethod
    def should_reject_request(
        self,
        workspace: str | None = None,
        endpoint_id: str | None = None,
        username: str | None = None,
    ) -> tuple[bool, BudgetWindow | None]:
        """Check if any REJECT-capable policy is exceeded.

        Args:
            workspace: The workspace to check against.
            endpoint_id: The gateway endpoint to check against. Used to match
                ENDPOINT-scoped policies.
            username: The authenticated username to check against. Used to match
                USER-scoped policies.

        Returns:
            Tuple of (exceeded, window). If exceeded is True, window is the
            first exceeded window found.
        """

    @abstractmethod
    def backfill_spend(self, spend_by_policy: dict[str, float]) -> list[BudgetWindow]:
        """Set cumulative spend on windows from historical data.

        Used to seed newly created windows with spend from trace data
        so that budget tracking survives server restarts.

        Args:
            spend_by_policy: Dict mapping budget_policy_id to historical spend amount.

        Returns:
            List of windows that were newly exceeded (the seeded spend carried them past
            their limit for the first time in this window). Used to trigger webhook
            alerts, the same way ``record_cost`` reports a crossing made by a request.
        """

    @abstractmethod
    def backfill_bucket_spend(
        self, bucket_spend_by_policy: dict[str, dict[int, float]]
    ) -> list[BudgetWindow]:
        """Seed the buckets of sliding windows from historical data.

        The sliding-window counterpart of ``backfill_spend``: a sliding window needs to know
        when its spend happened, not just how much, to age it out.

        Args:
            bucket_spend_by_policy: Dict mapping budget_policy_id to historical spend per
                bucket, keyed by bucket start in epoch milliseconds.

        Returns:
            List of windows that the seeded spend carried past their limit.
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


def _shift_by_duration(duration: BudgetDuration, moment: datetime, sign: int = 1) -> datetime:
    """Move ``moment`` forward (``sign=1``) or back (``sign=-1``) by ``duration``.

    Months move by calendar month, clamping the day to the length of the target month.
    """
    if duration.unit == BudgetDurationUnit.MONTHS:
        month_index = moment.year * 12 + moment.month - 1 + sign * duration.value
        year, month = divmod(month_index, 12)
        day = min(moment.day, calendar.monthrange(year, month + 1)[1])
        return moment.replace(year=year, month=month + 1, day=day)
    return moment + sign * timedelta(**{_TIMEDELTA_UNITS[duration.unit]: duration.value})


def _to_epoch_ms(moment: datetime) -> int:
    return (moment - _EPOCH) // timedelta(milliseconds=1)


def _from_epoch_ms(epoch_ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=epoch_ms)


def _sliding_bucket_width_ms(duration: BudgetDuration) -> int:
    """Width of one bucket of a sliding window, in milliseconds."""
    if duration.value <= 0:
        raise ValueError(f"duration.value must be positive, got {duration.value}")
    # Months vary in length, so their buckets are sized from 30-day months to keep the
    # width, and with it the bucket boundaries, stable from one month to the next.
    if duration.unit == BudgetDurationUnit.MONTHS:
        length = timedelta(days=30 * duration.value)
    else:
        length = timedelta(**{_TIMEDELTA_UNITS[duration.unit]: duration.value})
    return max(1, length // timedelta(milliseconds=1) // _SLIDING_WINDOW_BUCKETS)


def _sliding_bucket_start_ms(timestamp_ms: int, width_ms: int) -> int:
    """Start of the epoch-aligned bucket that contains ``timestamp_ms``."""
    return timestamp_ms - timestamp_ms % width_ms


def _sliding_bucket_expiry(
    duration: BudgetDuration, bucket_start_ms: int, width_ms: int
) -> datetime:
    """When a bucket stops counting: one duration after the bucket ends."""
    return _shift_by_duration(duration, _from_epoch_ms(bucket_start_ms + width_ms))


def _sliding_window_cutoff_ms(duration: BudgetDuration, width_ms: int, now: datetime) -> int:
    """Start of the oldest bucket that still counts toward a sliding window at ``now``.

    Every bucket that starts at or after the cutoff is live, and every earlier one expired.
    """
    window_start_ms = _to_epoch_ms(_shift_by_duration(duration, now, sign=-1))
    cutoff_ms = _sliding_bucket_start_ms(window_start_ms, width_ms)
    # Clamping the day makes shifting by months lossy (Mar 31 minus a month is Feb 28, plus
    # a month is Mar 28), so a bucket that ends after ``now`` minus the duration can still
    # be expired. Expiry only grows with the bucket start, so step past those.
    while _sliding_bucket_expiry(duration, cutoff_ms, width_ms) <= now:
        cutoff_ms += width_ms
    return cutoff_ms


def _advance_sliding_window(window: BudgetWindow, now: datetime) -> int:
    """Age out a sliding window's expired buckets as of ``now``.

    Recomputes the window's bounds and spend. Aging out can only lower spend, so it clears
    the exceeded flag once spend drops below the limit but never sets it: setting it is
    left to the callers that add spend, so they can report the crossing.

    Returns:
        The window's cutoff in epoch milliseconds.
    """
    width_ms = _sliding_bucket_width_ms(window.policy.duration)
    cutoff_ms = _sliding_window_cutoff_ms(window.policy.duration, width_ms, now)
    window.bucket_spend = {
        start: spend for start, spend in window.bucket_spend.items() if start >= cutoff_ms
    }
    window.window_start = _from_epoch_ms(cutoff_ms)
    window.window_end = now
    window.cumulative_spend = sum(window.bucket_spend.values())
    window.exceeded = window.exceeded and window.cumulative_spend >= window.policy.budget_amount
    return cutoff_ms


def _policy_applies(
    policy: GatewayBudgetPolicy,
    workspace: str | None,
    endpoint_id: str | None = None,
    username: str | None = None,
) -> bool:
    """Check if a policy applies to a given request.

    - GLOBAL policies apply to all requests.
    - WORKSPACE policies only apply when the request workspace matches the
      policy's workspace.
    - ENDPOINT policies only apply when the request endpoint matches the
      policy's ``target_value``.
    - USER policies only apply when the request username matches the
      policy's ``target_value``.
    """
    if policy.target_scope == BudgetTargetScope.GLOBAL:
        return True
    if policy.target_scope == BudgetTargetScope.ENDPOINT:
        return endpoint_id is not None and policy.target_value == endpoint_id
    if policy.target_scope == BudgetTargetScope.USER:
        return policy.target_value is not None and policy.target_value == username
    effective_workspace = workspace or DEFAULT_WORKSPACE_NAME
    return policy.workspace == effective_workspace
