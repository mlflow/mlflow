from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import BudgetTracker

fakeredis = pytest.importorskip("fakeredis")


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


def _make_tracker():
    from mlflow.gateway.budget_tracker.redis import RedisBudgetTracker

    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisBudgetTracker(_client=client)


def test_redis_tracker_is_budget_tracker():
    tracker = _make_tracker()
    assert isinstance(tracker, BudgetTracker)


def test_record_cost_below_limit():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.record_cost(50.0)
    assert newly_exceeded == []

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 50.0
    assert window.exceeded is False


def test_record_cost_exceeds_threshold():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.record_cost(150.0)
    assert len(newly_exceeded) == 1
    assert newly_exceeded[0].policy.budget_policy_id == "bp-test"

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True


def test_record_cost_exceeds_only_once():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    exceeded1 = tracker.record_cost(150.0)
    assert len(exceeded1) == 1

    exceeded2 = tracker.record_cost(50.0)
    assert exceeded2 == []

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 200.0


def test_record_cost_incremental_exceeding():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert tracker.record_cost(60.0) == []
    exceeded = tracker.record_cost(50.0)
    assert len(exceeded) == 1
    assert tracker._get_window_info("bp-test").cumulative_spend == 110.0


def test_should_reject_request_reject():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])

    tracker.record_cost(150.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"


def test_should_reject_request_alert_only():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])

    tracker.record_cost(150.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


def test_should_reject_request_not_yet():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])

    tracker.record_cost(50.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


def test_refresh_policies_removes_deleted_policy():
    tracker = _make_tracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.refresh_policies([policy1, policy2])
    tracker.record_cost(50.0)

    tracker.refresh_policies([policy1])
    assert tracker._get_window_info("bp-1") is not None
    assert tracker._get_window_info("bp-2") is None


def test_multiple_policies_independent():
    tracker = _make_tracker()
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
    assert len(exceeded) == 1
    assert exceeded[0].policy.budget_policy_id == "bp-alert"

    exceeded, _ = tracker.should_reject_request()
    assert exceeded is False

    tracker.record_cost(30.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-reject"


def test_workspace_scoped_cost_recording():
    tracker = _make_tracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.WORKSPACE,
        workspace="ws1",
        budget_amount=100.0,
    )
    tracker.refresh_policies([policy])

    tracker.record_cost(200.0, workspace="ws2")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0

    tracker.record_cost(50.0, workspace="ws1")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 50.0


def test_backfill_spend_sets_cumulative():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 42.5})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 42.5
    assert window.exceeded is False


def test_backfill_spend_sets_exceeded_when_exceeds():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 150.0})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True


def test_backfill_spend_sets_exceeded_at_exact_limit():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 100.0})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 100.0
    assert window.exceeded is True


def test_backfill_spend_nonexistent_is_noop():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy()])
    tracker.backfill_spend({"nonexistent-policy": 50.0})


def test_refresh_policies_returns_new_windows():
    tracker = _make_tracker()
    policy1 = _make_policy(budget_policy_id="bp-1")
    policy2 = _make_policy(budget_policy_id="bp-2")

    new_windows = tracker.refresh_policies([policy1, policy2])
    assert len(new_windows) == 2
    ids = {w.policy.budget_policy_id for w in new_windows}
    assert ids == {"bp-1", "bp-2"}


def test_refresh_policies_is_idempotent_for_existing_policies():
    tracker = _make_tracker()
    policy = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)

    first_windows = tracker.refresh_policies([policy])
    assert len(first_windows) == 1

    tracker.backfill_spend({"bp-1": 42.5})
    window_before = tracker._get_window_info("bp-1")
    assert window_before.cumulative_spend == 42.5

    # Second call with the same policy should not create a new window
    second_windows = tracker.refresh_policies([policy])
    assert len(second_windows) == 0

    # Existing window state should be preserved
    window_after = tracker._get_window_info("bp-1")
    assert window_after.cumulative_spend == window_before.cumulative_spend
    assert window_after.exceeded == window_before.exceeded


def test_get_all_windows():
    tracker = _make_tracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.refresh_policies([policy1, policy2])

    tracker.record_cost(75.0)

    windows = tracker.get_all_windows()
    assert len(windows) == 2
    by_id = {w.policy.budget_policy_id: w for w in windows}
    assert by_id["bp-1"].cumulative_spend == 75.0
    assert by_id["bp-1"].exceeded is False
    assert by_id["bp-2"].cumulative_spend == 75.0
    assert by_id["bp-2"].exceeded is False


def test_should_reject_request_workspace_filtering():
    tracker = _make_tracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.WORKSPACE,
        workspace="ws1",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy])

    tracker.record_cost(150.0, workspace="ws1")

    exceeded, window = tracker.should_reject_request(workspace="ws2")
    assert exceeded is False
    assert window is None

    exceeded, window = tracker.should_reject_request(workspace="ws1")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"


def test_record_cost_at_exact_budget_boundary():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    exceeded = tracker.record_cost(100.0)
    assert len(exceeded) == 1
    assert exceeded[0].policy.budget_policy_id == "bp-test"

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 100.0
    assert window.exceeded is True


def test_window_rollover_resets_spend():
    tracker = _make_tracker()
    tracker.refresh_policies(
        [
            _make_policy(
                budget_amount=100.0, duration_unit=BudgetDurationUnit.MINUTES, duration_value=1
            )
        ]
    )

    tracker.record_cost(150.0)
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True

    # Simulate time advancing past the window boundary
    future = datetime.now(timezone.utc) + timedelta(minutes=2)
    with patch(
        "mlflow.gateway.budget_tracker.redis.datetime",
    ) as mock_dt:
        mock_dt.now.return_value = future
        mock_dt.fromisoformat = datetime.fromisoformat

        tracker.record_cost(10.0)

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 10.0
    assert window.exceeded is False


def test_get_budget_tracker_returns_redis_when_configured():
    from mlflow.gateway.budget_tracker.redis import RedisBudgetTracker

    with (
        patch(
            "mlflow.gateway.budget_tracker.MLFLOW_GATEWAY_BUDGET_REDIS_URL.get",
            return_value="redis://localhost:6379/0",
        ),
        patch(
            "mlflow.gateway.budget_tracker.redis.RedisBudgetTracker.__post_init__",
        ) as mock_init,
        patch(
            "mlflow.gateway.budget_tracker._budget_tracker",
            new=None,
        ),
    ):
        from mlflow.gateway.budget_tracker import get_budget_tracker

        tracker = get_budget_tracker()
        assert isinstance(tracker, RedisBudgetTracker)
        mock_init.assert_called_once()

        # Reset the singleton
        import mlflow.gateway.budget_tracker

        mlflow.gateway.budget_tracker._budget_tracker = None
