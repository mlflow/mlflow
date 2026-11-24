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
from mlflow.protos.service_pb2 import BudgetWindowType as ProtoBudgetWindowType


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


def test_budget_target_scope_user_proto_roundtrip():
    assert BudgetTargetScope.from_proto(BudgetTargetScope.USER.to_proto()) == BudgetTargetScope.USER


def test_user_policy_proto_roundtrip_preserves_username_target():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice@example.com")
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == BudgetTargetScope.USER
    assert restored.target_value == "alice@example.com"


def test_string_target_scope_coerced_to_enum():
    policy = _make_policy(target_scope="USER", target_value="bob")
    assert policy.target_scope is BudgetTargetScope.USER


def test_window_type_defaults_to_fixed():
    assert _make_policy().window_type == BudgetWindowType.FIXED


@pytest.mark.parametrize("window_type", list(BudgetWindowType))
def test_to_proto_from_proto_round_trip_window_type(window_type):
    policy = _make_policy(window_type=window_type)
    assert GatewayBudgetPolicy.from_proto(policy.to_proto()).window_type == window_type


def test_from_proto_without_window_type_is_fixed():
    # Servers that predate sliding windows never set the field.
    proto = _make_policy().to_proto()
    proto.ClearField("window_type")
    assert not proto.HasField("window_type")
    assert GatewayBudgetPolicy.from_proto(proto).window_type == BudgetWindowType.FIXED


def test_window_type_from_string_in_post_init():
    assert _make_policy(window_type="SLIDING").window_type is BudgetWindowType.SLIDING


def test_window_type_proto_conversion():
    assert BudgetWindowType.SLIDING.to_proto() == ProtoBudgetWindowType.Value("SLIDING")
    assert (
        BudgetWindowType.from_proto(ProtoBudgetWindowType.Value("FIXED")) is BudgetWindowType.FIXED
    )
    assert (
        BudgetWindowType.from_proto(ProtoBudgetWindowType.Value("WINDOW_TYPE_UNSPECIFIED")) is None
    )
