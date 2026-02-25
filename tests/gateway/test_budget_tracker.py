from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetDurationType,
    BudgetOnExceeded,
    BudgetTargetType,
    BudgetType,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import (
    BudgetTracker,
    InMemoryBudgetTracker,
    _compute_window_end,
    _compute_window_start,
    _policy_applies,
)


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    duration_type=BudgetDurationType.DAYS,
    duration_value=1,
    target_type=BudgetTargetType.GLOBAL,
    on_exceeded=BudgetOnExceeded.ALERT,
    workspace=None,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_type=BudgetType.USD,
        budget_amount=budget_amount,
        duration_type=duration_type,
        duration_value=duration_value,
        target_type=target_type,
        on_exceeded=on_exceeded,
        created_at=0,
        last_updated_at=0,
        workspace=workspace,
    )


# --- _compute_window_start tests ---


def test_compute_window_start_minutes():
    now = datetime(2025, 6, 15, 10, 37, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDurationType.MINUTES, 15, now)
    assert start == datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)


def test_compute_window_start_hours():
    now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDurationType.HOURS, 2, now)
    assert start == datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)


def test_compute_window_start_days():
    now = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDurationType.DAYS, 7, now)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    days_since_epoch = (now - epoch).days
    window_index = days_since_epoch // 7
    expected = epoch + timedelta(days=window_index * 7)
    assert start == expected


def test_compute_window_start_months():
    now = datetime(2025, 8, 15, tzinfo=timezone.utc)
    start = _compute_window_start(BudgetDurationType.MONTHS, 3, now)
    # Total months from epoch: (2025-1970)*12 + (8-1) = 660 + 7 = 667
    # Window index: 667 // 3 = 222, window_start_months = 666
    # start_year = 1970 + 666//12 = 1970 + 55 = 2025, start_month = (666%12)+1 = 7
    assert start == datetime(2025, 7, 1, tzinfo=timezone.utc)


# --- _compute_window_end tests ---


def test_compute_window_end_minutes():
    start = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDurationType.MINUTES, 15, start)
    assert end == datetime(2025, 6, 15, 10, 45, 0, tzinfo=timezone.utc)


def test_compute_window_end_hours():
    start = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDurationType.HOURS, 2, start)
    assert end == datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_compute_window_end_days():
    start = datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDurationType.DAYS, 7, start)
    assert end == datetime(2025, 6, 22, 0, 0, 0, tzinfo=timezone.utc)


def test_compute_window_end_months():
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDurationType.MONTHS, 3, start)
    assert end == datetime(2025, 10, 1, tzinfo=timezone.utc)


def test_compute_window_end_months_crosses_year():
    start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end = _compute_window_end(BudgetDurationType.MONTHS, 3, start)
    assert end == datetime(2026, 2, 1, tzinfo=timezone.utc)


# --- _policy_applies tests ---


def test_policy_applies_global():
    policy = _make_policy(target_type=BudgetTargetType.GLOBAL)
    assert _policy_applies(policy, None) is True
    assert _policy_applies(policy, "ws1") is True


def test_policy_applies_workspace_match():
    policy = _make_policy(target_type=BudgetTargetType.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, "ws1") is True


def test_policy_applies_workspace_no_match():
    policy = _make_policy(target_type=BudgetTargetType.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, "ws2") is False


def test_policy_applies_workspace_none():
    policy = _make_policy(target_type=BudgetTargetType.WORKSPACE, workspace="ws1")
    assert _policy_applies(policy, None) is False


# --- InMemoryBudgetTracker tests ---


def test_in_memory_tracker_is_budget_tracker():
    tracker = InMemoryBudgetTracker()
    assert isinstance(tracker, BudgetTracker)


def test_record_cost_below_limit():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])

    newly_crossed = tracker.record_cost(50.0)
    assert newly_crossed == []

    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 50.0
    assert window.crossed is False


def test_record_cost_crosses_threshold():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])

    newly_crossed = tracker.record_cost(150.0)
    assert len(newly_crossed) == 1
    assert newly_crossed[0].policy.budget_policy_id == "bp-test"

    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.crossed is True


def test_record_cost_crosses_only_once():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])

    crossed1 = tracker.record_cost(150.0)
    assert len(crossed1) == 1

    crossed2 = tracker.record_cost(50.0)
    assert crossed2 == []

    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 200.0


def test_record_cost_incremental_crossing():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])

    assert tracker.record_cost(60.0) == []
    crossed = tracker.record_cost(50.0)
    assert len(crossed) == 1
    assert tracker.get_window_info("bp-test").cumulative_spend == 110.0


def test_is_budget_exceeded_reject():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0, on_exceeded=BudgetOnExceeded.REJECT)])

    tracker.record_cost(150.0)
    exceeded, policy = tracker.is_budget_exceeded()
    assert exceeded is True
    assert policy.budget_policy_id == "bp-test"


def test_is_budget_exceeded_alert_only():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0, on_exceeded=BudgetOnExceeded.ALERT)])

    tracker.record_cost(150.0)
    exceeded, policy = tracker.is_budget_exceeded()
    assert exceeded is False
    assert policy is None


def test_is_budget_exceeded_not_yet():
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0, on_exceeded=BudgetOnExceeded.REJECT)])

    tracker.record_cost(50.0)
    exceeded, policy = tracker.is_budget_exceeded()
    assert exceeded is False
    assert policy is None


def test_window_resets_on_expiry():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        budget_amount=100.0,
        duration_type=BudgetDurationType.MINUTES,
        duration_value=5,
    )
    tracker.load_policies([policy])
    tracker.record_cost(80.0)

    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 80.0

    # Simulate time passing beyond window end
    with patch(
        "mlflow.gateway.budget_tracker.datetime",
    ) as mock_dt:
        mock_dt.now.return_value = window.window_end + timedelta(seconds=1)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        newly_crossed = tracker.record_cost(10.0)

    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 10.0
    assert window.crossed is False
    assert newly_crossed == []


def test_load_policies_preserves_spend_in_same_window():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(budget_amount=100.0)
    tracker.load_policies([policy])
    tracker.record_cost(60.0)

    # Reload same policy — spend should be preserved
    tracker.load_policies([policy])
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 60.0


def test_load_policies_removes_deleted_policy():
    tracker = InMemoryBudgetTracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.load_policies([policy1, policy2])
    tracker.record_cost(50.0)

    # Reload with only policy1 — policy2 window should be gone
    tracker.load_policies([policy1])
    assert tracker.get_window_info("bp-1") is not None
    assert tracker.get_window_info("bp-2") is None


def test_multiple_policies_independent():
    tracker = InMemoryBudgetTracker()
    policy_alert = _make_policy(
        budget_policy_id="bp-alert",
        budget_amount=50.0,
        on_exceeded=BudgetOnExceeded.ALERT,
    )
    policy_reject = _make_policy(
        budget_policy_id="bp-reject",
        budget_amount=100.0,
        on_exceeded=BudgetOnExceeded.REJECT,
    )
    tracker.load_policies([policy_alert, policy_reject])

    crossed = tracker.record_cost(75.0)
    # Only the alert policy should be crossed (50 < 75)
    assert len(crossed) == 1
    assert crossed[0].policy.budget_policy_id == "bp-alert"

    # Reject policy should be at 75, not exceeded yet
    exceeded, _ = tracker.is_budget_exceeded()
    assert exceeded is False

    # Push reject over threshold
    tracker.record_cost(30.0)
    exceeded, policy = tracker.is_budget_exceeded()
    assert exceeded is True
    assert policy.budget_policy_id == "bp-reject"


def test_workspace_scoped_cost_recording():
    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        target_type=BudgetTargetType.WORKSPACE,
        workspace="ws1",
        budget_amount=100.0,
    )
    tracker.load_policies([policy])

    # Cost from different workspace — should not apply
    tracker.record_cost(200.0, workspace="ws2")
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 0.0

    # Cost from matching workspace — should apply
    tracker.record_cost(50.0, workspace="ws1")
    assert window.cumulative_spend == 50.0


@pytest.mark.parametrize("duration_type", list(BudgetDurationType))
def test_all_duration_types_window_consistency(duration_type):
    now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    start = _compute_window_start(duration_type, 1, now)
    end = _compute_window_end(duration_type, 1, start)
    assert start < end
    assert start <= now < end
