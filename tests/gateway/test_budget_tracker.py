from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    _compute_window_end,
    _compute_window_start,
    _policy_applies,
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
    assert _policy_applies(policy, None, principal="alice") is True
    # workspace is irrelevant for a USER-scoped policy
    assert _policy_applies(policy, "ws1", principal="alice") is True


def test_policy_applies_user_no_match():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    assert _policy_applies(policy, None, principal="bob") is False


def test_policy_applies_user_principal_none_request():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    # An unauthenticated request (principal=None) never matches a USER policy.
    assert _policy_applies(policy, None, principal=None) is False


def test_policy_applies_user_policy_without_principal_never_matches():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value=None)
    assert _policy_applies(policy, None, principal=None) is False
    assert _policy_applies(policy, None, principal="alice") is False


def test_policy_applies_global_ignores_principal():
    policy = _make_policy(target_scope=BudgetTargetScope.GLOBAL)
    assert _policy_applies(policy, None, principal="alice") is True
    assert _policy_applies(policy, None, principal=None) is True


def test_policy_applies_workspace_ignores_principal():
    policy = _make_policy(target_scope=BudgetTargetScope.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, "ws1", principal="alice") is True
    assert _policy_applies(policy, "ws2", principal="alice") is False


# --- InMemoryBudgetTracker USER-scope tests ---


def test_user_scoped_cost_recording_matches_principal():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.USER, target_value="alice", budget_amount=100.0
    )
    tracker.refresh_policies([policy])

    # Cost from a different user should not apply
    tracker.record_cost(200.0, principal="bob")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0

    # Cost from the matching user applies
    tracker.record_cost(50.0, principal="alice")
    assert window.cumulative_spend == 50.0


def test_user_scoped_cost_recording_ignores_unauthenticated_request():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    tracker.refresh_policies([policy])

    tracker.record_cost(50.0, principal=None)
    assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_user_scoped_should_reject_only_for_matching_principal():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.USER,
        target_value="alice",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy])
    tracker.record_cost(150.0, principal="alice")

    exceeded, window = tracker.should_reject_request(principal="alice")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"

    # Other users are unaffected by alice's budget
    exceeded, window = tracker.should_reject_request(principal="bob")
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

    tracker.record_cost(150.0, principal="alice")
    assert tracker._get_window_info("bp-alice").cumulative_spend == 150.0
    assert tracker._get_window_info("bp-bob").cumulative_spend == 0.0

    exceeded, _ = tracker.should_reject_request(principal="alice")
    assert exceeded is True
    exceeded, _ = tracker.should_reject_request(principal="bob")
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

    exceeded = tracker.record_cost(150.0, principal="alice")
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
    exceeded, window = tracker.should_reject_request(principal="alice")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"
    # Other users are unaffected.
    assert tracker.should_reject_request(principal="bob") == (False, None)


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
    tracker.record_cost(99.99, principal="alice")
    assert tracker.should_reject_request(principal="alice")[0] is False

    tracker.record_cost(0.01, principal="alice")  # now exactly at the $100 limit
    assert tracker.should_reject_request(principal="alice")[0] is True


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
    assert len(tracker.record_cost(150.0, principal="alice")) == 1
    # Continued overuse accumulates but does not re-report.
    assert tracker.record_cost(50.0, principal="alice") == []
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 200.0
    assert window.exceeded is True
