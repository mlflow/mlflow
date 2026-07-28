import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)


def _make_policy(**overrides):
    kwargs = {
        "budget_policy_id": "bp-1",
        "budget_unit": BudgetUnit.USD,
        "budget_amount": 42.0,
        "duration": BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        "target_scope": BudgetTargetScope.GLOBAL,
        "budget_action": BudgetAction.ALERT,
        "created_at": 10,
        "last_updated_at": 20,
    }
    kwargs.update(overrides)
    return GatewayBudgetPolicy(**kwargs)


def test_endpoint_scope_enum_value():
    assert BudgetTargetScope.ENDPOINT.value == "ENDPOINT"
    assert BudgetTargetScope("ENDPOINT") is BudgetTargetScope.ENDPOINT


def test_target_value_defaults_none():
    policy = _make_policy()
    assert policy.target_value is None


def test_to_proto_from_proto_round_trip_endpoint_scope():
    policy = _make_policy(target_scope=BudgetTargetScope.ENDPOINT, target_value="ep-42")
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == BudgetTargetScope.ENDPOINT
    assert restored.target_value == "ep-42"


@pytest.mark.parametrize(
    "target_scope",
    [BudgetTargetScope.GLOBAL, BudgetTargetScope.WORKSPACE],
)
def test_to_proto_from_proto_round_trip_no_target(target_scope):
    policy = _make_policy(target_scope=target_scope)
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == target_scope
    # Untargeted policies serialize target_value as unset (empty) -> None.
    assert restored.target_value is None


def test_target_scope_from_string_in_post_init():
    policy = GatewayBudgetPolicy(
        budget_policy_id="bp-1",
        budget_unit=BudgetUnit.USD,
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope="ENDPOINT",
        budget_action=BudgetAction.REJECT,
        created_at=0,
        last_updated_at=0,
        target_value="ep-1",
    )
    assert policy.target_scope is BudgetTargetScope.ENDPOINT


def test_to_proto_omits_target_value_when_none():
    policy = _make_policy(target_scope=BudgetTargetScope.GLOBAL, target_value=None)
    proto = policy.to_proto()
    assert proto.target_value == ""
    # Unset optional fields must not be marked present, so JSON responses omit them.
    assert not proto.HasField("target_value")
