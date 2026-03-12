from datetime import datetime, timezone
from unittest import mock

import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker.database import DatabaseBudgetTracker


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    duration_unit=BudgetDurationUnit.DAYS,
    duration_value=1,
    target_scope=BudgetTargetScope.GLOBAL,
    budget_action=BudgetAction.ALERT,
    workspace=None,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration_unit=duration_unit,
        duration_value=duration_value,
        target_scope=target_scope,
        budget_action=budget_action,
        created_at=0,
        last_updated_at=0,
        workspace=workspace,
    )


@pytest.fixture
def mock_store():
    return mock.MagicMock()


@pytest.fixture
def tracker(mock_store):
    return DatabaseBudgetTracker(_store=mock_store)


# --- refresh_policies ---


def test_refresh_policies_returns_empty(tracker):
    new_windows = tracker.refresh_policies([_make_policy()])
    assert new_windows == []


def test_refresh_policies_clears_cache(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    policy = _make_policy(budget_action=BudgetAction.REJECT)
    tracker.refresh_policies([policy])

    tracker.should_reject_request()
    assert len(tracker._reject_cache) == 1

    tracker.refresh_policies([policy])
    assert len(tracker._reject_cache) == 0


# --- should_reject_request ---


def test_should_reject_under_budget(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 50.0
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])

    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None
    mock_store.sum_gateway_trace_cost.assert_called_once()


def test_should_reject_over_budget(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])

    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"
    assert window.cumulative_spend == 150.0


def test_should_reject_skips_alert_policies(tracker, mock_store):
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.ALERT)])

    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    mock_store.sum_gateway_trace_cost.assert_not_called()


def test_should_reject_uses_ttl_cache(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 50.0
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])

    tracker.should_reject_request()
    tracker.should_reject_request()
    mock_store.sum_gateway_trace_cost.assert_called_once()


def test_should_reject_cache_expires(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 50.0
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])

    tracker.should_reject_request()
    # Expire the cache by manipulating the timestamp
    for cached in tracker._reject_cache.values():
        cached.mono_time -= 100
    tracker.should_reject_request()
    assert mock_store.sum_gateway_trace_cost.call_count == 2


def test_should_reject_workspace_scoped(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    policy = _make_policy(
        budget_action=BudgetAction.REJECT,
        target_scope=BudgetTargetScope.WORKSPACE,
        workspace="ws1",
    )
    tracker.refresh_policies([policy])

    # Different workspace — not applicable
    exceeded, _ = tracker.should_reject_request(workspace="ws2")
    assert exceeded is False
    mock_store.sum_gateway_trace_cost.assert_not_called()

    # Matching workspace
    exceeded, window = tracker.should_reject_request(workspace="ws1")
    assert exceeded is True
    mock_store.sum_gateway_trace_cost.assert_called_once()


def test_should_reject_db_error_is_handled(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.side_effect = RuntimeError("DB down")
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])

    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


# --- record_cost ---


def test_record_cost_clears_reject_cache(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 50.0
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])
    tracker.should_reject_request()
    assert len(tracker._reject_cache) == 1

    mock_store.sum_gateway_trace_cost.return_value = 0.0
    tracker.record_cost(10.0)
    assert len(tracker._reject_cache) == 0


def test_record_cost_detects_alert_exceeded(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])

    newly_exceeded = tracker.record_cost(10.0)
    assert len(newly_exceeded) == 1
    assert newly_exceeded[0].policy.budget_policy_id == "bp-test"


def test_record_cost_alert_fires_only_once_per_window(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])

    assert len(tracker.record_cost(10.0)) == 1
    assert len(tracker.record_cost(10.0)) == 0


def test_record_cost_skips_reject_policies(tracker, mock_store):
    tracker.refresh_policies([_make_policy(budget_action=BudgetAction.REJECT)])

    newly_exceeded = tracker.record_cost(200.0)
    assert newly_exceeded == []
    mock_store.sum_gateway_trace_cost.assert_not_called()


# --- backfill_spend ---


def test_backfill_spend_is_noop(tracker):
    tracker.refresh_policies([_make_policy()])
    tracker.backfill_spend({"bp-test": 50.0})
    # No error, no effect


# --- get_all_windows ---


def test_get_all_windows_queries_db(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 42.5
    tracker.refresh_policies([_make_policy(budget_policy_id="bp-1")])

    windows = tracker.get_all_windows()
    assert len(windows) == 1
    assert windows[0].cumulative_spend == 42.5
    assert windows[0].policy.budget_policy_id == "bp-1"


def test_get_all_windows_empty(tracker):
    assert tracker.get_all_windows() == []


# --- _get_window_info ---


def test_get_window_info_queries_db(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 75.0
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 75.0
    assert window.exceeded is False


def test_get_window_info_exceeded(tracker, mock_store):
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    window = tracker._get_window_info("bp-test")
    assert window.exceeded is True


def test_get_window_info_nonexistent(tracker):
    tracker.refresh_policies([_make_policy()])
    assert tracker._get_window_info("nonexistent") is None


# --- multiple policies ---


def test_multiple_reject_policies_first_exceeded_wins(tracker, mock_store):
    policy1 = _make_policy(
        budget_policy_id="bp-1", budget_amount=50.0, budget_action=BudgetAction.REJECT
    )
    policy2 = _make_policy(
        budget_policy_id="bp-2", budget_amount=200.0, budget_action=BudgetAction.REJECT
    )
    tracker.refresh_policies([policy1, policy2])

    mock_store.sum_gateway_trace_cost.return_value = 75.0
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-1"
    # Should short-circuit — only one DB query needed
    mock_store.sum_gateway_trace_cost.assert_called_once()


def test_alert_window_resets_on_new_window(tracker, mock_store):
    policy = _make_policy(
        budget_amount=100.0,
        budget_action=BudgetAction.ALERT,
        duration_unit=BudgetDurationUnit.MINUTES,
        duration_value=5,
    )
    mock_store.sum_gateway_trace_cost.return_value = 150.0
    tracker.refresh_policies([policy])

    # First call — should fire
    assert len(tracker.record_cost(10.0)) == 1
    # Second call same window — should not fire
    assert len(tracker.record_cost(10.0)) == 0

    # Simulate window rollover by changing the alerted window_start
    now = datetime.now(timezone.utc)
    tracker._alerted["bp-test"] = now.replace(year=2000)

    # Should fire again in new window
    assert len(tracker.record_cost(10.0)) == 1
