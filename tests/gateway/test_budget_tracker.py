from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    BudgetWindowType,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    SlidingWindowHistory,
    _compute_window_end,
    _compute_window_start,
    _policy_applies,
    _shift_by_duration,
    _sliding_bucket_expiry,
    _sliding_bucket_width_ms,
    _sliding_window_cutoff_ms,
)
from mlflow.gateway.budget_tracker.in_memory import InMemoryBudgetTracker


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    duration=None,
    target_scope=BudgetTargetScope.GLOBAL,
    budget_action=BudgetAction.ALERT,
    workspace=None,
    target_value=None,
    window_type=BudgetWindowType.FIXED,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration=duration or BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope=target_scope,
        budget_action=budget_action,
        created_at=0,
        last_updated_at=0,
        workspace=workspace,
        target_value=target_value,
        window_type=window_type,
    )


# --- _compute_window_start tests ---


def test_compute_window_start_minutes():
    now = datetime(2025, 6, 15, 10, 37, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.MINUTES, value=15), now)
    assert start == datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)


def test_compute_window_start_hours():
    now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2), now)
    assert start == datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)


def test_compute_window_start_days():
    now = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.DAYS, value=7), now)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    days_since_epoch = (now - epoch).days
    window_index = days_since_epoch // 7
    expected = epoch + timedelta(days=window_index * 7)
    assert start == expected


def test_compute_window_start_weeks():
    # June 15, 2025 is a Sunday — window should start on that Sunday
    now = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.WEEKS, value=1), now)
    assert start == datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    assert start.weekday() == 6  # Sunday

    # Wednesday mid-week — window should start on preceding Sunday
    now = datetime(2025, 6, 18, 14, 30, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.WEEKS, value=1), now)
    assert start == datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    assert start.weekday() == 6  # Sunday

    # Multi-week (2-week) windows also start on Sundays
    now = datetime(2025, 6, 20, 0, 0, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.WEEKS, value=2), now)
    assert start.weekday() == 6  # Sunday


def test_compute_window_start_months():
    now = datetime(2025, 8, 15, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=3), now)
    # Total months from epoch: (2025-1970)*12 + (8-1) = 660 + 7 = 667
    # Window index: 667 // 3 = 222, window_start_months = 666
    # start_year = 1970 + 666//12 = 1970 + 55 = 2025, start_month = (666%12)+1 = 7
    assert start == datetime(2025, 7, 1, tzinfo=timezone.utc)


# --- _compute_window_end tests ---


def test_compute_window_end_minutes():
    start = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDuration(unit=BudgetDurationUnit.MINUTES, value=15), start)
    assert end == datetime(2025, 6, 15, 10, 45, 0, tzinfo=timezone.utc)


def test_compute_window_end_hours():
    start = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2), start)
    assert end == datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_compute_window_end_days():
    start = datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDuration(unit=BudgetDurationUnit.DAYS, value=7), start)
    assert end == datetime(2025, 6, 22, 0, 0, 0, tzinfo=timezone.utc)


def test_compute_window_end_weeks():
    start = datetime(2025, 6, 12, 0, 0, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDuration(unit=BudgetDurationUnit.WEEKS, value=2), start)
    assert end == datetime(2025, 6, 26, 0, 0, 0, tzinfo=timezone.utc)


def test_compute_window_end_months():
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=3), start)
    assert end == datetime(2025, 10, 1, tzinfo=timezone.utc)


def test_compute_window_end_months_crosses_year():
    start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=3), start)
    assert end == datetime(2026, 2, 1, tzinfo=timezone.utc)


# --- _policy_applies tests ---


def test_policy_applies_global():
    policy = _make_policy(target_scope=BudgetTargetScope.GLOBAL)
    assert _policy_applies(policy, None) is True
    assert _policy_applies(policy, "ws1") is True


def test_policy_applies_workspace_match():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, "ws1") is True


def test_policy_applies_workspace_no_match():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, "ws2") is False


def test_policy_applies_workspace_none():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, None) is False


def test_policy_applies_workspace_none_matches_default():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE)
    # policy.workspace resolves to "default" via __post_init__
    assert _policy_applies(policy, None) is True


# --- _policy_applies ENDPOINT scope tests ---


def test_policy_applies_endpoint_match():
    policy = _make_policy(target_scope=BudgetTargetScope.ENDPOINT, target_value="ep-1")
    assert _policy_applies(policy, workspace=None, endpoint_id="ep-1") is True


def test_policy_applies_endpoint_no_match():
    policy = _make_policy(target_scope=BudgetTargetScope.ENDPOINT, target_value="ep-1")
    assert _policy_applies(policy, workspace=None, endpoint_id="ep-2") is False


def test_policy_applies_endpoint_none_request():
    policy = _make_policy(target_scope=BudgetTargetScope.ENDPOINT, target_value="ep-1")
    # A request without an endpoint (e.g. the windows listing) never matches.
    assert _policy_applies(policy, workspace=None, endpoint_id=None) is False


def test_policy_applies_endpoint_ignores_workspace():
    # ENDPOINT scope matches purely on endpoint_id; workspace is irrelevant.
    policy = _make_policy(
        target_scope=BudgetTargetScope.ENDPOINT, target_value="ep-1", workspace="ws-a"
    )
    assert _policy_applies(policy, workspace="ws-b", endpoint_id="ep-1") is True
    assert _policy_applies(policy, workspace="ws-a", endpoint_id="ep-other") is False


def test_policy_applies_global_ignores_endpoint():
    policy = _make_policy(target_scope=BudgetTargetScope.GLOBAL)
    assert _policy_applies(policy, workspace=None, endpoint_id="ep-1") is True


def test_policy_applies_workspace_ignores_endpoint():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE, workspace="ws1")
    # WORKSPACE scope still matches on workspace regardless of endpoint_id.
    assert _policy_applies(policy, workspace="ws1", endpoint_id="ep-anything") is True
    assert _policy_applies(policy, workspace="ws2", endpoint_id="ep-anything") is False


# --- InMemoryBudgetTracker tests ---


def test_in_memory_tracker_is_budget_tracker():
    tracker = InMemoryBudgetTracker()
    assert isinstance(tracker, BudgetTracker)


def test_record_cost_below_limit():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.record_cost(50.0)
    assert newly_exceeded == []

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 50.0
    assert window.exceeded is False


def test_record_cost_exceeds_threshold():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.record_cost(150.0)
    assert len(newly_exceeded) == 1
    assert newly_exceeded[0].policy.budget_policy_id == "bp-test"

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True


def test_record_cost_exceeds_only_once():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    exceeded1 = tracker.record_cost(150.0)
    assert len(exceeded1) == 1

    exceeded2 = tracker.record_cost(50.0)
    assert exceeded2 == []

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 200.0


def test_record_cost_incremental_exceeding():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert tracker.record_cost(60.0) == []
    exceeded = tracker.record_cost(50.0)
    assert len(exceeded) == 1
    assert tracker._get_window_info("bp-test").cumulative_spend == 110.0


def test_should_reject_request_reject():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])

    tracker.record_cost(150.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"


def test_should_reject_request_alert_only():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])

    tracker.record_cost(150.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


def test_should_reject_request_not_yet():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])

    tracker.record_cost(50.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


def test_window_resets_on_expiry():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.MINUTES, value=5),
    )
    tracker.refresh_policies([policy])
    tracker.record_cost(80.0)

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 80.0

    # Simulate time passing beyond window end
    with patch(
        "mlflow.gateway.budget_tracker.in_memory.datetime",
    ) as mock_dt:
        mock_dt.now.return_value = window.window_end + timedelta(seconds=1)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        newly_exceeded = tracker.record_cost(10.0)

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 10.0
    assert window.exceeded is False
    assert newly_exceeded == []


def test_refresh_policies_preserves_spend_in_same_window():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(budget_amount=100.0)
    tracker.refresh_policies([policy])
    tracker.record_cost(60.0)

    # Reload same policy — spend should be preserved
    tracker.refresh_policies([policy])
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 60.0


def test_refresh_policies_removes_deleted_policy():
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.refresh_policies([policy1, policy2])
    tracker.record_cost(50.0)

    # Reload with only policy1 — policy2 window should be gone
    tracker.refresh_policies([policy1])
    assert tracker._get_window_info("bp-1") is not None
    assert tracker._get_window_info("bp-2") is None


def test_multiple_policies_independent():
    tracker = InMemoryBudgetTracker()
    policy_alert = _make_policy(
        budget_policy_id="bp-alert",
        budget_amount=50.0,
        budget_action=BudgetAction.ALERT,
    )
    policy_reject = _make_policy(
        budget_policy_id="bp-reject",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy_alert, policy_reject])

    exceeded = tracker.record_cost(75.0)
    # Only the alert policy should be exceeded (50 < 75)
    assert len(exceeded) == 1
    assert exceeded[0].policy.budget_policy_id == "bp-alert"

    # Reject policy should be at 75, not exceeded yet
    exceeded, _ = tracker.should_reject_request()
    assert exceeded is False

    # Push reject over threshold
    tracker.record_cost(30.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-reject"


def test_workspace_scoped_cost_recording():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.WORKSPACE,
        workspace="ws1",
        budget_amount=100.0,
    )
    tracker.refresh_policies([policy])

    # Cost from different workspace — should not apply
    tracker.record_cost(200.0, workspace="ws2")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0

    # Cost from matching workspace — should apply
    tracker.record_cost(50.0, workspace="ws1")
    assert window.cumulative_spend == 50.0


@pytest.mark.parametrize("duration_unit", list(BudgetDurationUnit))
def test_all_duration_units_window_consistency(duration_unit):
    now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    duration = BudgetDuration(unit=duration_unit, value=1)
    start = _compute_window_start(duration, now)
    end = _compute_window_end(duration, start)
    assert start < end
    assert start <= now < end


# --- refresh_policies return value tests ---


def test_refresh_policies_returns_all_windows():
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1")
    policy2 = _make_policy(budget_policy_id="bp-2")

    windows = tracker.refresh_policies([policy1, policy2])
    assert len(windows) == 2
    ids = {w.policy.budget_policy_id for w in windows}
    assert ids == {"bp-1", "bp-2"}


def test_refresh_policies_returns_all_windows_on_reload():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy()

    windows = tracker.refresh_policies([policy])
    assert len(windows) == 1

    # Reload same policy within same window — still returns the existing window
    windows = tracker.refresh_policies([policy])
    assert len(windows) == 1


def test_refresh_policies_returns_all_windows_on_mixed():
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1")
    tracker.refresh_policies([policy1])

    policy2 = _make_policy(budget_policy_id="bp-2")
    windows = tracker.refresh_policies([policy1, policy2])
    assert len(windows) == 2
    ids = {w.policy.budget_policy_id for w in windows}
    assert ids == {"bp-1", "bp-2"}


# --- backfill_spend tests ---


def test_backfill_spend_sets_cumulative():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 42.5})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 42.5
    assert window.exceeded is False


def test_backfill_spend_sets_exceeded_when_exceeds():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 150.0})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True


def test_backfill_spend_sets_exceeded_at_exact_limit():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 100.0})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 100.0
    assert window.exceeded is True


def test_backfill_spend_nonexistent_is_noop():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy()])
    # Should not raise
    tracker.backfill_spend({"nonexistent-policy": 50.0})


def test_backfill_spend_uses_max_to_protect_in_process_spend():
    # Simulate trace-flush lag: in-process spend is ahead of what DB reports.
    # backfill_spend must not decrease cumulative_spend below the in-process value.
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])
    tracker.record_cost(30.0)  # in-process spend = 30

    tracker.backfill_spend({"bp-test": 10.0})  # DB is behind
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 30.0  # in-process value preserved


def test_backfill_spend_returns_the_window_it_carried_over_the_limit():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.backfill_spend({"bp-test": 150.0})
    assert [w.policy.budget_policy_id for w in newly_exceeded] == ["bp-test"]
    assert newly_exceeded[0].cumulative_spend == 150.0
    assert newly_exceeded[0].exceeded is True


def test_backfill_spend_returns_nothing_below_the_limit():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert tracker.backfill_spend({"bp-test": 42.5}) == []


def test_backfill_spend_reports_a_crossing_only_once():
    # The alert is edge-triggered, so a window already over its limit stays quiet.
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert len(tracker.backfill_spend({"bp-test": 150.0})) == 1
    assert tracker.backfill_spend({"bp-test": 175.0}) == []


def test_backfill_spend_stays_quiet_when_record_cost_already_reported():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])
    assert len(tracker.record_cost(120.0)) == 1

    assert tracker.backfill_spend({"bp-test": 150.0}) == []


# --- get_all_windows tests ---


def test_get_all_windows_empty():
    tracker = InMemoryBudgetTracker()
    assert tracker.get_all_windows() == []


def test_get_all_windows_returns_all_policies():
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1")
    policy2 = _make_policy(budget_policy_id="bp-2")
    tracker.refresh_policies([policy1, policy2])

    windows = tracker.get_all_windows()
    assert len(windows) == 2
    ids = {w.policy.budget_policy_id for w in windows}
    assert ids == {"bp-1", "bp-2"}


def test_get_all_windows_reflects_current_spend():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(budget_policy_id="bp-spend", budget_amount=100.0)
    tracker.refresh_policies([policy])
    tracker.record_cost(42.5)

    windows = tracker.get_all_windows()
    assert len(windows) == 1
    assert windows[0].cumulative_spend == 42.5


def test_get_all_windows_after_policy_removed():
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1")
    policy2 = _make_policy(budget_policy_id="bp-2")
    tracker.refresh_policies([policy1, policy2])

    tracker.refresh_policies([policy1])

    windows = tracker.get_all_windows()
    assert len(windows) == 1
    assert windows[0].policy.budget_policy_id == "bp-1"


# --- InMemoryBudgetTracker endpoint-scoped tests ---


def _endpoint_policy(budget_policy_id="bp-ep", endpoint_id="ep-1", **kwargs):
    return _make_policy(
        budget_policy_id=budget_policy_id,
        target_scope=BudgetTargetScope.ENDPOINT,
        target_value=endpoint_id,
        **kwargs,
    )


def test_endpoint_scoped_cost_recording_matches_endpoint():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_endpoint_policy(endpoint_id="ep-1", budget_amount=100.0)])

    # Cost routed to a different endpoint must not accumulate.
    tracker.record_cost(200.0, endpoint_id="ep-2")
    assert tracker._get_window_info("bp-ep").cumulative_spend == 0.0

    # Cost routed to the matching endpoint accumulates.
    tracker.record_cost(40.0, endpoint_id="ep-1")
    assert tracker._get_window_info("bp-ep").cumulative_spend == 40.0


def test_endpoint_scoped_cost_recording_ignores_no_endpoint():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([_endpoint_policy(endpoint_id="ep-1", budget_amount=100.0)])

    tracker.record_cost(50.0, endpoint_id=None)
    assert tracker._get_window_info("bp-ep").cumulative_spend == 0.0


def test_endpoint_scoped_exceed_and_reject():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _endpoint_policy(endpoint_id="ep-1", budget_amount=50.0, budget_action=BudgetAction.REJECT)
    ])

    newly_exceeded = tracker.record_cost(60.0, endpoint_id="ep-1")
    assert len(newly_exceeded) == 1
    assert newly_exceeded[0].policy.budget_policy_id == "bp-ep"

    # Only requests to the matching endpoint are rejected.
    exceeded, window = tracker.should_reject_request(endpoint_id="ep-1")
    assert exceeded is True
    assert window.policy.target_value == "ep-1"

    exceeded_other, window_other = tracker.should_reject_request(endpoint_id="ep-2")
    assert exceeded_other is False
    assert window_other is None

    # A request with no endpoint (e.g. non-gateway path) is never rejected by an
    # endpoint-scoped policy.
    exceeded_none, _ = tracker.should_reject_request(endpoint_id=None)
    assert exceeded_none is False


def test_endpoint_policies_are_isolated_from_each_other():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _endpoint_policy(budget_policy_id="bp-a", endpoint_id="ep-a", budget_amount=100.0),
        _endpoint_policy(budget_policy_id="bp-b", endpoint_id="ep-b", budget_amount=100.0),
    ])

    tracker.record_cost(70.0, endpoint_id="ep-a")
    assert tracker._get_window_info("bp-a").cumulative_spend == 70.0
    assert tracker._get_window_info("bp-b").cumulative_spend == 0.0


def test_global_and_endpoint_policies_both_apply():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _make_policy(budget_policy_id="bp-global", budget_amount=1000.0),
        _endpoint_policy(budget_policy_id="bp-ep", endpoint_id="ep-1", budget_amount=100.0),
    ])

    # A request to ep-1 counts against both the GLOBAL and the ENDPOINT policy.
    tracker.record_cost(30.0, endpoint_id="ep-1")
    assert tracker._get_window_info("bp-global").cumulative_spend == 30.0
    assert tracker._get_window_info("bp-ep").cumulative_spend == 30.0

    # A request to ep-2 only counts against the GLOBAL policy.
    tracker.record_cost(20.0, endpoint_id="ep-2")
    assert tracker._get_window_info("bp-global").cumulative_spend == 50.0
    assert tracker._get_window_info("bp-ep").cumulative_spend == 30.0


def test_endpoint_and_workspace_policies_independent():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _make_policy(
            budget_policy_id="bp-ws",
            target_scope=BudgetTargetScope.WORKSPACE,
            workspace="ws1",
            budget_amount=100.0,
        ),
        _endpoint_policy(budget_policy_id="bp-ep", endpoint_id="ep-1", budget_amount=100.0),
    ])

    # Request from ws1 to ep-1 hits both.
    tracker.record_cost(25.0, workspace="ws1", endpoint_id="ep-1")
    assert tracker._get_window_info("bp-ws").cumulative_spend == 25.0
    assert tracker._get_window_info("bp-ep").cumulative_spend == 25.0

    # Request from ws2 to ep-1 hits only the endpoint policy.
    tracker.record_cost(25.0, workspace="ws2", endpoint_id="ep-1")
    assert tracker._get_window_info("bp-ws").cumulative_spend == 25.0
    assert tracker._get_window_info("bp-ep").cumulative_spend == 50.0


# --- _policy_applies USER scope tests ---


def test_policy_applies_user_match():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    assert _policy_applies(policy, None, username="alice") is True
    # workspace is irrelevant for a USER-scoped policy
    assert _policy_applies(policy, "ws1", username="alice") is True


def test_policy_applies_user_no_match():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    assert _policy_applies(policy, None, username="bob") is False


def test_policy_applies_user_scope_no_request_username():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    # An unauthenticated request (username=None) never matches a USER policy.
    assert _policy_applies(policy, None, username=None) is False


def test_policy_applies_user_policy_without_username_never_matches():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value=None)
    assert _policy_applies(policy, None, username=None) is False
    assert _policy_applies(policy, None, username="alice") is False


def test_policy_applies_global_ignores_username():
    policy = _make_policy(target_scope=BudgetTargetScope.GLOBAL)
    assert _policy_applies(policy, None, username="alice") is True
    assert _policy_applies(policy, None, username=None) is True


def test_policy_applies_workspace_ignores_username():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, "ws1", username="alice") is True
    assert _policy_applies(policy, "ws2", username="alice") is False


# --- InMemoryBudgetTracker USER-scope tests ---


def test_user_scoped_cost_recording_matches_username():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.USER, target_value="alice", budget_amount=100.0
    )
    tracker.refresh_policies([policy])

    # Cost from a different user should not apply
    tracker.record_cost(200.0, username="bob")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0

    # Cost from the matching user applies
    tracker.record_cost(50.0, username="alice")
    assert window.cumulative_spend == 50.0


def test_user_scoped_cost_recording_ignores_unauthenticated_request():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    tracker.refresh_policies([policy])

    tracker.record_cost(50.0, username=None)
    assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_user_scoped_should_reject_only_for_matching_username():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.USER,
        target_value="alice",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy])
    tracker.record_cost(150.0, username="alice")

    exceeded, window = tracker.should_reject_request(username="alice")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"

    # Other users are unaffected by alice's budget
    exceeded, window = tracker.should_reject_request(username="bob")
    assert exceeded is False
    assert window is None


def test_multiple_user_policies_are_independent():
    tracker = InMemoryBudgetTracker()
    alice = _make_policy(
        budget_policy_id="bp-alice",
        target_scope=BudgetTargetScope.USER,
        target_value="alice",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    bob = _make_policy(
        budget_policy_id="bp-bob",
        target_scope=BudgetTargetScope.USER,
        target_value="bob",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([alice, bob])

    tracker.record_cost(150.0, username="alice")
    assert tracker._get_window_info("bp-alice").cumulative_spend == 150.0
    assert tracker._get_window_info("bp-bob").cumulative_spend == 0.0

    exceeded, _ = tracker.should_reject_request(username="alice")
    assert exceeded is True
    exceeded, _ = tracker.should_reject_request(username="bob")
    assert exceeded is False


def test_user_and_global_policies_stack():
    tracker = InMemoryBudgetTracker()
    global_policy = _make_policy(budget_policy_id="bp-global", budget_amount=1000.0)
    user_policy = _make_policy(
        budget_policy_id="bp-user",
        target_scope=BudgetTargetScope.USER,
        target_value="alice",
        budget_amount=100.0,
    )
    tracker.refresh_policies([global_policy, user_policy])

    exceeded = tracker.record_cost(150.0, username="alice")
    # Both the global and user windows accrue; only the user policy is exceeded.
    assert {w.policy.budget_policy_id for w in exceeded} == {"bp-user"}
    assert tracker._get_window_info("bp-global").cumulative_spend == 150.0
    assert tracker._get_window_info("bp-user").cumulative_spend == 150.0


def test_user_scoped_zero_budget_rejects_immediately():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _make_policy(
            target_scope=BudgetTargetScope.USER,
            target_value="alice",
            budget_amount=0.0,
            budget_action=BudgetAction.REJECT,
        )
    ])
    # No spend recorded yet, but a $0 budget is already met (0 >= 0), so the very
    # first request from alice is rejected.
    exceeded, window = tracker.should_reject_request(username="alice")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"
    # Other users are unaffected.
    assert tracker.should_reject_request(username="bob") == (False, None)


def test_user_scoped_rejects_at_exact_limit():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _make_policy(
            target_scope=BudgetTargetScope.USER,
            target_value="alice",
            budget_amount=100.0,
            budget_action=BudgetAction.REJECT,
        )
    ])
    tracker.record_cost(99.99, username="alice")
    assert tracker.should_reject_request(username="alice")[0] is False

    tracker.record_cost(0.01, username="alice")  # now exactly at the $100 limit
    assert tracker.should_reject_request(username="alice")[0] is True


def test_user_scoped_exceeded_flips_once_then_keeps_accumulating():
    tracker = InMemoryBudgetTracker()
    tracker.refresh_policies([
        _make_policy(
            target_scope=BudgetTargetScope.USER,
            target_value="alice",
            budget_amount=100.0,
            budget_action=BudgetAction.ALERT,
        )
    ])
    # First crossing reports the window as newly exceeded (fires the alert once).
    assert len(tracker.record_cost(150.0, username="alice")) == 1
    # Continued overuse accumulates but does not re-report.
    assert tracker.record_cost(50.0, username="alice") == []
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 200.0
    assert window.exceeded is True


# --- Sliding windows ---

_NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
_ONE_HOUR = BudgetDuration(unit=BudgetDurationUnit.HOURS, value=1)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


@contextmanager
def _clock(moment: datetime):
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    with patch("mlflow.gateway.budget_tracker.in_memory.datetime", _FrozenDatetime):
        yield


def _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.REJECT, **kwargs):
    kwargs.setdefault("duration", _ONE_HOUR)
    return _make_policy(
        budget_amount=budget_amount,
        budget_action=budget_action,
        window_type=BudgetWindowType.SLIDING,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("duration", "expected_width_ms"),
    [
        (BudgetDuration(unit=BudgetDurationUnit.MINUTES, value=1), 1_000),
        (BudgetDuration(unit=BudgetDurationUnit.HOURS, value=1), 60_000),
        (BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2), 120_000),
        (BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1), 1_440_000),
        (BudgetDuration(unit=BudgetDurationUnit.WEEKS, value=1), 10_080_000),
        (BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=1), 43_200_000),
    ],
)
def test_sliding_bucket_width_is_a_sixtieth_of_the_duration(duration, expected_width_ms):
    assert _sliding_bucket_width_ms(duration) == expected_width_ms


@pytest.mark.parametrize(
    ("moment", "months", "expected"),
    [
        # Forward onto a day the target month lacks: the start of the month after it.
        (datetime(2026, 1, 31, tzinfo=timezone.utc), 1, datetime(2026, 3, 1, tzinfo=timezone.utc)),
        (
            datetime(2026, 1, 29, 0, 0, tzinfo=timezone.utc),
            1,
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        ),
        (datetime(2028, 1, 31, tzinfo=timezone.utc), 1, datetime(2028, 3, 1, tzinfo=timezone.utc)),
        (
            datetime(2028, 1, 29, 6, 0, tzinfo=timezone.utc),
            1,
            datetime(2028, 2, 29, 6, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 12, 31, tzinfo=timezone.utc),
            2,
            datetime(2027, 3, 1, tzinfo=timezone.utc),
        ),
        # Back onto a day the target month lacks: clamped to its last day.
        (
            datetime(2026, 3, 31, tzinfo=timezone.utc),
            -1,
            datetime(2026, 2, 28, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 12, 15, tzinfo=timezone.utc),
            1,
            datetime(2027, 1, 15, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 1, 15, tzinfo=timezone.utc),
            -1,
            datetime(2025, 12, 15, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 5, 10, tzinfo=timezone.utc),
            -14,
            datetime(2025, 3, 10, tzinfo=timezone.utc),
        ),
    ],
)
def test_shift_by_duration_months(moment, months, expected):
    duration = BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=abs(months))
    assert _shift_by_duration(duration, moment, sign=1 if months > 0 else -1) == expected


def test_sliding_window_cutoff_is_the_bucket_containing_the_window_start():
    now = _NOW + timedelta(seconds=30)
    width_ms = _sliding_bucket_width_ms(_ONE_HOUR)
    # The window starts at 11:00:30, inside the 11:00 bucket.
    assert _sliding_window_cutoff_ms(_ONE_HOUR, width_ms, now) == _ms(_NOW - timedelta(hours=1))


def test_sliding_window_cutoff_skips_buckets_expired_by_month_clamping():
    duration = BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=1)
    width_ms = _sliding_bucket_width_ms(duration)
    now = datetime(2026, 3, 31, 10, 0, tzinfo=timezone.utc)
    # A month before now is Feb 28 10:00, but the bucket ending Feb 28 12:00 already expired
    # on Mar 28 12:00. The first live bucket ends Mar 1, so it expires Apr 1.
    cutoff_ms = _sliding_window_cutoff_ms(duration, width_ms, now)
    assert cutoff_ms == _ms(datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc))
    assert _sliding_bucket_expiry(duration, cutoff_ms, width_ms) > now
    assert _sliding_bucket_expiry(duration, cutoff_ms - width_ms, width_ms) <= now


def test_sliding_window_counts_only_the_trailing_duration():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        tracker.record_cost(30.0, timestamp_ms=_ms(_NOW - timedelta(minutes=50)))
        tracker.record_cost(40.0, timestamp_ms=_ms(_NOW - timedelta(minutes=10)))
        assert tracker._get_window_info("bp-test").cumulative_spend == 70.0

    # 15 minutes later the cost from 50 minutes ago is more than an hour old.
    with _clock(_NOW + timedelta(minutes=15)):
        window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 40.0
    assert window.window_start == _NOW - timedelta(minutes=45)
    assert window.window_end == _NOW + timedelta(minutes=15)


def test_sliding_window_counts_a_cost_for_at_least_the_duration():
    tracker = InMemoryBudgetTracker()
    started = _NOW - timedelta(seconds=1)  # 11:59:59, in the 11:59 bucket
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        tracker.record_cost(10.0, timestamp_ms=_ms(started))

    # Still counted a full hour after the request started...
    with _clock(started + timedelta(hours=1)):
        assert tracker._get_window_info("bp-test").cumulative_spend == 10.0
    # ...and until the whole 11:59 bucket is an hour old.
    with _clock(_NOW + timedelta(hours=1) - timedelta(microseconds=1)):
        assert tracker._get_window_info("bp-test").cumulative_spend == 10.0
    with _clock(_NOW + timedelta(hours=1)):
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_sliding_window_ignores_cost_of_request_started_before_the_window():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        assert tracker.record_cost(10.0, timestamp_ms=_ms(_NOW - timedelta(hours=2))) == []
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_sliding_window_rejects_until_enough_spend_ages_out():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=50.0)])
        tracker.record_cost(30.0, timestamp_ms=_ms(_NOW - timedelta(minutes=40)))
        tracker.record_cost(30.0, timestamp_ms=_ms(_NOW - timedelta(minutes=5)))
        exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    # The 11:20 bucket ends at 11:21 and ages out an hour later, leaving $30 < $50.
    assert window.resets_at == _NOW + timedelta(minutes=21)

    with _clock(_NOW + timedelta(minutes=21) - timedelta(seconds=1)):
        assert tracker.should_reject_request()[0] is True
    with _clock(_NOW + timedelta(minutes=21)):
        assert tracker.should_reject_request() == (False, None)


def test_sliding_window_resets_at_skips_buckets_that_do_not_free_enough():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=50.0)])
        tracker.record_cost(5.0, timestamp_ms=_ms(_NOW - timedelta(minutes=50)))
        tracker.record_cost(60.0, timestamp_ms=_ms(_NOW - timedelta(minutes=30)))
        _, window = tracker.should_reject_request()
    # Aging out the $5 bucket still leaves $60, so the $60 bucket (11:30) has to go too.
    assert window.resets_at == _NOW + timedelta(minutes=31)


def test_sliding_window_alert_rearms_after_spend_drops_below_the_limit():
    tracker = InMemoryBudgetTracker()
    policy = _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
    with _clock(_NOW):
        tracker.refresh_policies([policy])
        assert len(tracker.record_cost(60.0)) == 1
        # Still over the limit: no second alert.
        assert tracker.record_cost(10.0) == []

    later = _NOW + timedelta(hours=1, minutes=1)
    with _clock(later):
        assert tracker._get_window_info("bp-test").exceeded is False
        newly_exceeded = tracker.record_cost(60.0)
    assert [w.policy.budget_policy_id for w in newly_exceeded] == ["bp-test"]
    assert newly_exceeded[0].cumulative_spend == 60.0


def test_sliding_window_respects_scope():
    tracker = InMemoryBudgetTracker()
    policy = _sliding_policy(target_scope=BudgetTargetScope.ENDPOINT, target_value="ep-1")
    with _clock(_NOW):
        tracker.refresh_policies([policy])
        tracker.record_cost(60.0, endpoint_id="ep-2")
        assert tracker.should_reject_request(endpoint_id="ep-1") == (False, None)
        tracker.record_cost(60.0, endpoint_id="ep-1")
        assert tracker.should_reject_request(endpoint_id="ep-1")[0] is True
        assert tracker.should_reject_request(endpoint_id="ep-2") == (False, None)


def test_refresh_preserves_sliding_window_buckets():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=100.0)])
        tracker.record_cost(30.0)
        # An edit that keeps the duration, such as a new amount, keeps the spend.
        tracker.refresh_policies([_sliding_policy(budget_amount=20.0)])
        window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 30.0
    assert window.policy.budget_amount == 20.0


def test_refresh_resets_sliding_window_when_duration_changes():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=100.0)])
        tracker.record_cost(30.0)
        two_hours = BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2)
        tracker.refresh_policies([_sliding_policy(budget_amount=100.0, duration=two_hours)])
        window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0
    assert window.bucket_spend == {}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (BudgetWindowType.FIXED, BudgetWindowType.SLIDING),
        (BudgetWindowType.SLIDING, BudgetWindowType.FIXED),
    ],
)
def test_refresh_resets_spend_when_window_type_changes(before, after):
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_make_policy(duration=_ONE_HOUR, window_type=before)])
        tracker.record_cost(30.0)
        tracker.refresh_policies([_make_policy(duration=_ONE_HOUR, window_type=after)])
        window = tracker._get_window_info("bp-test")
    assert window.policy.window_type == after
    assert window.cumulative_spend == 0.0
    assert window.bucket_spend == {}


def test_backfill_bucket_spend_takes_max_per_bucket_and_reports_crossing():
    tracker = InMemoryBudgetTracker()
    policy = _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
    bucket_a = _ms(_NOW - timedelta(minutes=30))
    bucket_b = _ms(_NOW - timedelta(minutes=10))
    with _clock(_NOW):
        tracker.refresh_policies([policy])
        tracker.record_cost(20.0, timestamp_ms=bucket_a)
        # Trace history lags behind on bucket_a, and saw another worker's spend in bucket_b.
        newly_exceeded = tracker.backfill_bucket_spend([
            SlidingWindowHistory(policy=policy, bucket_spend={bucket_a: 5.0, bucket_b: 40.0})
        ])
        window = tracker._get_window_info("bp-test")
    assert window.bucket_spend == {bucket_a: 20.0, bucket_b: 40.0}
    assert window.cumulative_spend == 60.0
    assert newly_exceeded == [window]

    with _clock(_NOW):
        history = SlidingWindowHistory(policy=policy, bucket_spend={bucket_b: 40.0})
        assert tracker.backfill_bucket_spend([history]) == []


def test_backfill_bucket_spend_reports_crossing_after_earlier_spend_aged_out():
    tracker = InMemoryBudgetTracker()
    policy = _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
    with _clock(_NOW):
        tracker.refresh_policies([policy])
        assert len(tracker.record_cost(60.0)) == 1

    later = _NOW + timedelta(hours=1, minutes=1)
    with _clock(later):
        crossed = tracker.backfill_bucket_spend([
            SlidingWindowHistory(policy=policy, bucket_spend={_ms(later): 60.0})
        ])
    assert [w.policy.budget_policy_id for w in crossed] == ["bp-test"]


def test_backfill_bucket_spend_drops_buckets_outside_the_window():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        policy = _sliding_policy(budget_amount=1000.0)
        tracker.refresh_policies([policy])
        tracker.backfill_bucket_spend([
            SlidingWindowHistory(policy=policy, bucket_spend={_ms(_NOW - timedelta(hours=3)): 10.0})
        ])
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_backfill_methods_ignore_the_other_window_type():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([
            _make_policy(budget_policy_id="bp-fixed"),
            _sliding_policy(budget_policy_id="bp-sliding"),
        ])
        assert tracker.backfill_spend({"bp-sliding": 100.0}) == []
        fixed_history = SlidingWindowHistory(
            policy=_make_policy(budget_policy_id="bp-fixed"), bucket_spend={_ms(_NOW): 100.0}
        )
        assert tracker.backfill_bucket_spend([fixed_history]) == []
        assert tracker._get_window_info("bp-fixed").cumulative_spend == 0.0
        assert tracker._get_window_info("bp-sliding").cumulative_spend == 0.0


def test_get_all_windows_ages_out_sliding_windows():
    tracker = InMemoryBudgetTracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        tracker.record_cost(10.0)
    with _clock(_NOW + timedelta(hours=2)):
        [window] = tracker.get_all_windows()
    assert window.cumulative_spend == 0.0
    assert window.window_end == _NOW + timedelta(hours=2)
    assert window.window_start == _NOW + timedelta(hours=1)


def test_backfill_bucket_spend_drops_history_read_for_another_duration():
    tracker = InMemoryBudgetTracker()
    two_hours = BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2)
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0, duration=two_hours)])
        # Read while the policy still had a 1-hour duration, so its buckets are 1 minute wide
        # instead of 2.
        stale_history = SlidingWindowHistory(
            policy=_sliding_policy(budget_amount=1000.0),
            bucket_spend={_ms(_NOW - timedelta(minutes=1)): 25.0},
        )
        assert tracker.backfill_bucket_spend([stale_history]) == []
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_sliding_window_resets_at_does_not_drift_with_float_subtraction():
    tracker = InMemoryBudgetTracker()
    first = _NOW - timedelta(minutes=40)
    second = _NOW - timedelta(minutes=20)
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=0.30)])
        tracker.record_cost(0.60, timestamp_ms=_ms(first))
        tracker.record_cost(0.30, timestamp_ms=_ms(second))
        _, window = tracker.should_reject_request()
    # 0.60 + 0.30 - 0.60 is 0.2999..., but once the first bucket ages out the tracker sums
    # what is left to exactly 0.30, which still rejects. So the reset is when the second
    # bucket ages out, not the first.
    assert window.resets_at == second + timedelta(hours=1, minutes=1)
    with _clock(first + timedelta(hours=1, minutes=1)):
        assert tracker.should_reject_request()[0] is True


_ONE_MONTH = BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=1)


def test_sliding_month_window_counts_a_request_for_its_full_calendar_month():
    tracker = InMemoryBudgetTracker()
    started = datetime(2026, 1, 28, 23, 0, tzinfo=timezone.utc)
    with _clock(started):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0, duration=_ONE_MONTH)])
        tracker.record_cost(10.0, timestamp_ms=_ms(started))

    # Its 12-hour bucket ends Jan 29 00:00, a day February lacks. Clamping that to Feb 28
    # 00:00 would drop the request 23 hours before a month has passed.
    with _clock(datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)):
        assert tracker._get_window_info("bp-test").cumulative_spend == 10.0
    with _clock(datetime(2026, 2, 28, 23, 30, tzinfo=timezone.utc)):
        assert tracker._get_window_info("bp-test").cumulative_spend == 10.0
    with _clock(datetime(2026, 3, 1, tzinfo=timezone.utc)):
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def _one_calendar_month_after(moment: datetime) -> datetime:
    """The same day and time next month, or the end of next month if it lacks that day."""
    year, month = divmod(moment.year * 12 + moment.month, 12)
    try:
        return moment.replace(year=year, month=month + 1)
    except ValueError:
        year, month = divmod(moment.year * 12 + moment.month + 1, 12)
        return datetime(year, month + 1, 1, tzinfo=moment.tzinfo)


@pytest.mark.parametrize("year", [2026, 2028])
def test_sliding_month_bucket_expiry_grows_and_never_precedes_a_month(year):
    width_ms = _sliding_bucket_width_ms(_ONE_MONTH)
    start_ms = _ms(datetime(year, 1, 1, tzinfo=timezone.utc))
    end_ms = _ms(datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    previous_expiry = None
    for bucket_start_ms in range(start_ms, end_ms, width_ms):
        expiry = _sliding_bucket_expiry(_ONE_MONTH, bucket_start_ms, width_ms)
        if previous_expiry is not None:
            assert expiry >= previous_expiry
        previous_expiry = expiry
        for instant_ms in (bucket_start_ms, bucket_start_ms + width_ms - 1):
            instant = datetime.fromtimestamp(instant_ms / 1000, tz=timezone.utc)
            assert expiry >= _one_calendar_month_after(instant)
