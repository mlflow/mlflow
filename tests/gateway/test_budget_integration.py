from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import InMemoryBudgetTracker
from mlflow.server.gateway_api import (
    _fire_budget_crossed_webhooks,
    _get_model_info,
    _make_cost_recording_reducer,
    _maybe_record_budget_cost,
    _maybe_refresh_budget_policies,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig, GatewayModelConfig
from mlflow.tracing.constant import CostKey

_COST_FUNC = "mlflow.server.gateway_api.calculate_cost_by_model_and_token_usage"
_TRACKER_FUNC = "mlflow.server.gateway_api.get_budget_tracker"
_DELIVER_FUNC = "mlflow.server.gateway_api.deliver_webhook"
_REGISTRY_FUNC = "mlflow.server.gateway_api._get_model_registry_store"


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    budget_action=BudgetAction.ALERT,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration_unit=BudgetDurationUnit.DAYS,
        duration_value=1,
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=on_exceeded,
        created_at=0,
        last_updated_at=0,
    )


def _make_endpoint_config(model_name="gpt-4o", provider="openai"):
    return GatewayEndpointConfig(
        endpoint_id="ep-1",
        endpoint_name="test-endpoint",
        models=[
            GatewayModelConfig(
                model_definition_id="md-1",
                provider=provider,
                model_name=model_name,
                secret_value={"api_key": "test"},
            )
        ],
    )


# --- _get_model_info tests ---


def test_get_model_info_with_models():
    config = _make_endpoint_config(model_name="gpt-4o", provider="openai")
    model_name, provider = _get_model_info(config)
    assert model_name == "gpt-4o"
    assert provider == "openai"


def test_get_model_info_empty():
    config = GatewayEndpointConfig(
        endpoint_id="ep-1", endpoint_name="test", models=[]
    )
    model_name, provider = _get_model_info(config)
    assert model_name is None
    assert provider is None


# --- _maybe_record_budget_cost tests ---


@dataclass
class _FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


@dataclass
class _FakeResponse:
    usage: _FakeUsage = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = _FakeUsage()


@patch(_TRACKER_FUNC)
@patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.05})
def test_maybe_record_budget_cost_with_usage_object(mock_cost, mock_tracker):
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])
    mock_tracker.return_value = tracker

    store = MagicMock()
    store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

    response = _FakeResponse()
    _maybe_record_budget_cost(
        store, response, model_name="gpt-4o", model_provider="openai"
    )

    mock_cost.assert_called_once()
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 0.05


@patch(_TRACKER_FUNC)
@patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.03})
def test_maybe_record_budget_cost_with_usage_dict(mock_cost, mock_tracker):
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])
    mock_tracker.return_value = tracker

    store = MagicMock()
    store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

    response = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "choices": [],
    }
    _maybe_record_budget_cost(
        store, response, model_name="gpt-4o", model_provider="openai"
    )

    mock_cost.assert_called_once()
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 0.03


@patch(_TRACKER_FUNC)
@patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.01})
def test_maybe_record_budget_cost_with_anthropic_dict_keys(
    mock_cost, mock_tracker
):
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])
    mock_tracker.return_value = tracker

    store = MagicMock()
    store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

    response = {
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    _maybe_record_budget_cost(
        store, response, model_name="claude-3", model_provider="anthropic"
    )

    mock_cost.assert_called_once()
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 0.01


@patch(_COST_FUNC, return_value=None)
def test_maybe_record_budget_cost_no_cost_available(mock_cost):
    store = MagicMock()
    response = _FakeResponse()
    _maybe_record_budget_cost(
        store, response, model_name="unknown-model", model_provider="unknown"
    )


def test_maybe_record_budget_cost_no_usage():
    store = MagicMock()
    response = {"choices": []}
    _maybe_record_budget_cost(store, response, model_name="gpt-4o")


def test_maybe_record_budget_cost_none_usage_attr():
    store = MagicMock()

    @dataclass
    class NoUsageResponse:
        usage: None = None

    _maybe_record_budget_cost(store, NoUsageResponse(), model_name="gpt-4o")


# --- _fire_budget_crossed_webhooks tests ---


@patch(_DELIVER_FUNC)
@patch(_REGISTRY_FUNC)
def test_fire_budget_crossed_webhooks_alert(mock_registry, mock_deliver):
    mock_registry.return_value = MagicMock()

    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        budget_amount=50.0, budget_action=BudgetAction.ALERT
    )
    tracker.load_policies([policy])
    crossed = tracker.record_cost(60.0)
    assert len(crossed) == 1

    _fire_budget_crossed_webhooks(crossed, workspace=None)
    mock_deliver.assert_called_once()

    payload = mock_deliver.call_args.kwargs["payload"]
    assert payload["budget_policy_id"] == "bp-test"
    assert payload["budget_amount"] == 50.0
    assert payload["current_spend"] == 60.0


@patch(_DELIVER_FUNC)
@patch(_REGISTRY_FUNC)
def test_fire_budget_crossed_webhooks_reject_skipped(
    mock_registry, mock_deliver
):
    mock_registry.return_value = MagicMock()

    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        budget_amount=50.0, budget_action=BudgetAction.REJECT
    )
    tracker.load_policies([policy])
    crossed = tracker.record_cost(60.0)
    assert len(crossed) == 1

    _fire_budget_crossed_webhooks(crossed, workspace=None)
    mock_deliver.assert_not_called()


@patch(_DELIVER_FUNC)
@patch(_REGISTRY_FUNC)
def test_fire_budget_crossed_webhooks_with_workspace(
    mock_registry, mock_deliver
):
    mock_registry.return_value = MagicMock()

    tracker = InMemoryBudgetTracker()
    policy = _make_policy(
        budget_amount=50.0, budget_action=BudgetAction.ALERT
    )
    tracker.load_policies([policy])
    crossed = tracker.record_cost(60.0)

    _fire_budget_crossed_webhooks(crossed, workspace="my-ws")
    payload = mock_deliver.call_args.kwargs["payload"]
    assert payload["workspace"] == "my-ws"


# --- _maybe_refresh_budget_policies tests ---


@patch(_TRACKER_FUNC)
def test_maybe_refresh_budget_policies(mock_get_tracker):
    tracker = InMemoryBudgetTracker()
    mock_get_tracker.return_value = tracker

    store = MagicMock()
    policy = _make_policy()
    store.list_budget_policies.return_value = [policy]

    _maybe_refresh_budget_policies(store)

    store.list_budget_policies.assert_called_once()
    window = tracker.get_window_info("bp-test")
    assert window is not None


@patch(_TRACKER_FUNC)
def test_maybe_refresh_skips_when_not_needed(mock_get_tracker):
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy()])
    mock_get_tracker.return_value = tracker

    store = MagicMock()
    _maybe_refresh_budget_policies(store)
    store.list_budget_policies.assert_not_called()


# --- _make_cost_recording_reducer tests ---


@patch(_TRACKER_FUNC)
@patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.10})
@patch(
    "mlflow.server.gateway_api.aggregate_chat_stream_chunks"
)
def test_cost_recording_reducer(
    mock_aggregate, mock_cost, mock_get_tracker
):
    tracker = InMemoryBudgetTracker()
    tracker.load_policies([_make_policy(budget_amount=100.0)])
    mock_get_tracker.return_value = tracker

    store = MagicMock()
    store.list_budget_policies.return_value = [
        _make_policy(budget_amount=100.0)
    ]

    mock_aggregate.return_value = _FakeResponse()

    endpoint_config = _make_endpoint_config()
    reducer = _make_cost_recording_reducer(
        store, endpoint_config, workspace=None
    )

    chunks = ["chunk1", "chunk2"]
    result = reducer(chunks)

    mock_aggregate.assert_called_once_with(chunks)
    assert result is not None
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 0.10


@patch("mlflow.server.gateway_api.aggregate_chat_stream_chunks")
def test_cost_recording_reducer_no_result(mock_aggregate):
    mock_aggregate.return_value = None

    store = MagicMock()
    endpoint_config = _make_endpoint_config()
    reducer = _make_cost_recording_reducer(
        store, endpoint_config, workspace=None
    )

    result = reducer(["chunk1"])
    assert result is None


# --- End-to-end cost → webhook test ---


@patch(_DELIVER_FUNC)
@patch(_REGISTRY_FUNC)
@patch(_TRACKER_FUNC)
@patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 200.0})
def test_record_cost_triggers_webhook(
    mock_cost, mock_get_tracker, mock_registry, mock_deliver
):
    mock_registry.return_value = MagicMock()

    tracker = InMemoryBudgetTracker()
    tracker.load_policies([
        _make_policy(
            budget_amount=100.0, budget_action=BudgetAction.ALERT
        )
    ])
    mock_get_tracker.return_value = tracker

    store = MagicMock()
    store.list_budget_policies.return_value = [
        _make_policy(
            budget_amount=100.0, budget_action=BudgetAction.ALERT
        )
    ]

    response = _FakeResponse()
    _maybe_record_budget_cost(
        store, response, model_name="gpt-4o", model_provider="openai"
    )

    mock_deliver.assert_called_once()
    window = tracker.get_window_info("bp-test")
    assert window.cumulative_spend == 200.0
    assert window.crossed is True


@patch(_DELIVER_FUNC)
@patch(_REGISTRY_FUNC)
@patch(_TRACKER_FUNC)
@patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 200.0})
def test_record_cost_no_webhook_for_reject(
    mock_cost, mock_get_tracker, mock_registry, mock_deliver
):
    mock_registry.return_value = MagicMock()

    tracker = InMemoryBudgetTracker()
    tracker.load_policies([
        _make_policy(
            budget_amount=100.0, budget_action=BudgetAction.REJECT
        )
    ])
    mock_get_tracker.return_value = tracker

    store = MagicMock()
    store.list_budget_policies.return_value = [
        _make_policy(
            budget_amount=100.0, budget_action=BudgetAction.REJECT
        )
    ]

    response = _FakeResponse()
    _maybe_record_budget_cost(
        store, response, model_name="gpt-4o", model_provider="openai"
    )

    # Crossed but REJECT → no webhook fired
    mock_deliver.assert_not_called()
    assert tracker.get_window_info("bp-test").crossed is True
