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


def test_endpoint_id_defaults_none():
    policy = _make_policy()
    assert policy.endpoint_id is None


def test_to_proto_from_proto_round_trip_endpoint_scope():
    policy = _make_policy(target_scope=BudgetTargetScope.ENDPOINT, endpoint_id="ep-42")
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == BudgetTargetScope.ENDPOINT
    assert restored.endpoint_id == "ep-42"


@pytest.mark.parametrize(
    "target_scope",
    [BudgetTargetScope.GLOBAL, BudgetTargetScope.WORKSPACE],
)
def test_to_proto_from_proto_round_trip_no_endpoint(target_scope):
    policy = _make_policy(target_scope=target_scope)
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == target_scope
    # Non-endpoint policies serialize endpoint_id as unset (empty) -> None.
    assert restored.endpoint_id is None


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
        endpoint_id="ep-1",
    )
    assert policy.target_scope is BudgetTargetScope.ENDPOINT


def test_to_proto_omits_endpoint_id_when_none():
    policy = _make_policy(target_scope=BudgetTargetScope.GLOBAL, endpoint_id=None)
    proto = policy.to_proto()
    assert proto.endpoint_id == ""


def test_budget_target_scope_user_proto_roundtrip():
    assert BudgetTargetScope.from_proto(BudgetTargetScope.USER.to_proto()) == BudgetTargetScope.USER


def test_user_policy_proto_roundtrip_preserves_principal():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, principal="alice@example.com")
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == BudgetTargetScope.USER
    assert restored.principal == "alice@example.com"


def test_non_user_policy_principal_defaults_none():
    policy = _make_policy()
    assert policy.principal is None
    proto = policy.to_proto()
    # Unset optional fields must not be marked present, so JSON responses omit them.
    assert not proto.HasField("principal")
    assert not proto.HasField("endpoint_id")
    restored = GatewayBudgetPolicy.from_proto(proto)
    assert restored.principal is None


def test_string_target_scope_coerced_to_enum():
    policy = _make_policy(target_scope="USER", principal="bob")
    assert policy.target_scope is BudgetTargetScope.USER
