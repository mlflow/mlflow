from unittest.mock import MagicMock, patch

import fastapi
import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import InMemoryBudgetTracker
from mlflow.server.gateway_budget import check_budget_limit

_TRACKER_FUNC = "mlflow.server.gateway_budget.get_budget_tracker"


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    budget_action=BudgetAction.REJECT,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration_unit=BudgetDurationUnit.DAYS,
        duration_value=1,
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=budget_action,
        created_at=0,
        last_updated_at=0,
    )


def test_check_budget_limit_no_policies():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = []

        # No policies → no exception
        check_budget_limit(store)


def test_check_budget_limit_not_exceeded():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        tracker.load_policies([_make_policy(budget_amount=100.0)])
        tracker.record_cost(50.0)
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

        # Under budget → no exception
        check_budget_limit(store)


def test_check_budget_limit_exceeded_rejects():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        tracker.load_policies(
            [_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)]
        )
        tracker.record_cost(150.0)
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [
            _make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)
        ]

        with pytest.raises(fastapi.HTTPException, match="Request rejected"):
            check_budget_limit(store)


def test_check_budget_limit_alert_does_not_reject():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        tracker.load_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])
        tracker.record_cost(150.0)
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [
            _make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)
        ]

        # ALERT policy exceeded → no exception (only webhooks, no rejection)
        check_budget_limit(store)


def test_check_budget_limit_error_message_format():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        tracker.load_policies(
            [
                _make_policy(
                    budget_policy_id="bp-monthly",
                    budget_amount=500.0,
                    budget_action=BudgetAction.REJECT,
                )
            ]
        )
        tracker.record_cost(600.0)
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [
            _make_policy(
                budget_policy_id="bp-monthly",
                budget_amount=500.0,
                budget_action=BudgetAction.REJECT,
            )
        ]

        with pytest.raises(fastapi.HTTPException, match="Request rejected") as exc_info:
            check_budget_limit(store)

        detail = exc_info.value.detail
        assert "bp-monthly" in detail
        assert "$500.00" in detail
        assert "1 days" in detail
        assert "Request rejected" in detail


def test_check_budget_limit_with_workspace():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        policy = GatewayBudgetPolicy(
            budget_policy_id="bp-ws",
            budget_unit=BudgetUnit.USD,
            budget_amount=50.0,
            duration_unit=BudgetDurationUnit.DAYS,
            duration_value=1,
            target_scope=BudgetTargetScope.WORKSPACE,
            budget_action=BudgetAction.REJECT,
            created_at=0,
            last_updated_at=0,
            workspace="ws1",
        )
        tracker.load_policies([policy])
        tracker.record_cost(100.0, workspace="ws1")
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [policy]

        # ws1 is exceeded → should raise
        with pytest.raises(fastapi.HTTPException, match="Request rejected"):
            check_budget_limit(store, workspace="ws1")

        # ws2 should not be affected
        check_budget_limit(store, workspace="ws2")


def test_check_budget_limit_multiple_policies():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        alert_policy = _make_policy(
            budget_policy_id="bp-alert",
            budget_amount=50.0,
            budget_action=BudgetAction.ALERT,
        )
        reject_policy = _make_policy(
            budget_policy_id="bp-reject",
            budget_amount=100.0,
            budget_action=BudgetAction.REJECT,
        )
        tracker.load_policies([alert_policy, reject_policy])
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [alert_policy, reject_policy]

        # 75 exceeds alert (50) but not reject (100) → no rejection
        tracker.record_cost(75.0)
        check_budget_limit(store)

        # Push to 105 → exceeds reject policy → should raise
        tracker.record_cost(30.0)
        with pytest.raises(fastapi.HTTPException, match="Request rejected"):
            check_budget_limit(store)
