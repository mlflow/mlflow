import json

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
)
from mlflow.store.tracking.gateway.rest_mixin import RestGatewayStoreMixin


class _StubRestStore(RestGatewayStoreMixin):
    def __init__(self):
        self.calls = []

    def _call_endpoint(self, api, json_body):
        self.calls.append((api, json.loads(json_body)))
        return api.Response()


def test_create_budget_policy_passes_endpoint_target():
    store = _StubRestStore()
    store.create_budget_policy(
        budget_unit=BudgetUnit.USD,
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope=BudgetTargetScope.ENDPOINT,
        budget_action=BudgetAction.REJECT,
        target_value="ep-1",
    )
    (_, body) = store.calls[0]
    assert body["target_value"] == "ep-1"
    assert body["target_scope"] == "ENDPOINT"


def test_create_budget_policy_omits_target_value_when_unset():
    store = _StubRestStore()
    store.create_budget_policy(
        budget_unit=BudgetUnit.USD,
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=BudgetAction.ALERT,
    )
    (_, body) = store.calls[0]
    assert "target_value" not in body


def test_update_budget_policy_passes_endpoint_target():
    store = _StubRestStore()
    store.update_budget_policy(
        budget_policy_id="bp-1",
        target_scope=BudgetTargetScope.ENDPOINT,
        target_value="ep-2",
    )
    (_, body) = store.calls[0]
    assert body["budget_policy_id"] == "bp-1"
    assert body["target_value"] == "ep-2"


def test_update_budget_policy_omits_target_value_when_unset():
    store = _StubRestStore()
    store.update_budget_policy(budget_policy_id="bp-1", budget_amount=50.0)
    (_, body) = store.calls[0]
    assert "target_value" not in body
