from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker.in_memory import InMemoryBudgetTracker
from mlflow.server.gateway_budget import (
    calculate_existing_cost_for_new_windows,
    fire_budget_exceeded_webhooks,
    make_cost_recording_reducer,
    maybe_refresh_budget_policies,
    record_budget_cost,
)
from mlflow.tracing.constant import CostKey

_COST_FUNC = "mlflow.server.gateway_budget.calculate_cost_by_model_and_token_usage"
_TRACKER_FUNC = "mlflow.server.gateway_budget.get_budget_tracker"
_DELIVER_FUNC = "mlflow.server.gateway_budget.deliver_webhook"
_REGISTRY_FUNC = "mlflow.server.gateway_budget._get_model_registry_store"


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
        budget_action=budget_action,
        created_at=0,
        last_updated_at=0,
    )


# --- record_budget_cost tests ---


@dataclass
class _FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


@dataclass
class _FakeResponse:
    model: str = "gpt-4o"
    usage: _FakeUsage | None = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = _FakeUsage()


def test_record_budget_cost_with_usage_object():
    with (
        patch(_TRACKER_FUNC) as mock_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.05}) as mock_cost,
    ):
        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy(budget_amount=100.0)])
        mock_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

        response = _FakeResponse()
        record_budget_cost(store, response)

        mock_cost.assert_called_once()
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 0.05


def test_record_budget_cost_with_usage_dict():
    with (
        patch(_TRACKER_FUNC) as mock_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.03}) as mock_cost,
    ):
        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy(budget_amount=100.0)])
        mock_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

        response = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "choices": [],
        }
        record_budget_cost(store, response)

        mock_cost.assert_called_once()
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 0.03


def test_record_budget_cost_with_anthropic_dict_keys():
    with (
        patch(_TRACKER_FUNC) as mock_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.01}) as mock_cost,
    ):
        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy(budget_amount=100.0)])
        mock_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

        response = {
            "model": "claude-3-haiku",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        record_budget_cost(store, response)

        mock_cost.assert_called_once()
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 0.01


def test_record_budget_cost_no_cost_available():
    with patch(_COST_FUNC, return_value=None):
        store = MagicMock()
        response = _FakeResponse()
        record_budget_cost(store, response)


def test_record_budget_cost_no_usage():
    store = MagicMock()
    response = {"choices": []}
    record_budget_cost(store, response)


def test_record_budget_cost_none_usage_attr():
    store = MagicMock()

    @dataclass
    class NoUsageResponse:
        usage: object | None = None

    record_budget_cost(store, NoUsageResponse())


# --- fire_budget_exceeded_webhooks tests ---


def test_fire_budget_exceeded_webhooks_alert():
    with patch(_DELIVER_FUNC) as mock_deliver, patch(_REGISTRY_FUNC) as mock_registry:
        mock_registry.return_value = MagicMock()

        tracker = InMemoryBudgetTracker()
        policy = _make_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
        tracker.refresh_policies([policy])
        crossed = tracker.record_cost(60.0)
        assert len(crossed) == 1

        fire_budget_exceeded_webhooks(crossed, workspace=None)
        mock_deliver.assert_called_once()

        payload = mock_deliver.call_args.kwargs["payload"]
        assert payload["budget_policy_id"] == "bp-test"
        assert payload["budget_amount"] == 50.0
        assert payload["current_spend"] == 60.0


def test_fire_budget_exceeded_webhooks_reject_skipped():
    with patch(_DELIVER_FUNC) as mock_deliver, patch(_REGISTRY_FUNC) as mock_registry:
        mock_registry.return_value = MagicMock()

        tracker = InMemoryBudgetTracker()
        policy = _make_policy(budget_amount=50.0, budget_action=BudgetAction.REJECT)
        tracker.refresh_policies([policy])
        crossed = tracker.record_cost(60.0)
        assert len(crossed) == 1

        fire_budget_exceeded_webhooks(crossed, workspace=None)
        mock_deliver.assert_not_called()


def test_fire_budget_exceeded_webhooks_with_workspace():
    with patch(_DELIVER_FUNC) as mock_deliver, patch(_REGISTRY_FUNC) as mock_registry:
        mock_registry.return_value = MagicMock()

        tracker = InMemoryBudgetTracker()
        policy = _make_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
        tracker.refresh_policies([policy])
        crossed = tracker.record_cost(60.0)

        fire_budget_exceeded_webhooks(crossed, workspace="my-ws")
        payload = mock_deliver.call_args.kwargs["payload"]
        assert payload["workspace"] == "my-ws"


# --- _maybe_refresh_budget_policies tests ---


def test_maybe_refresh_budget_policies():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        policy = _make_policy()
        store.list_budget_policies.return_value = [policy]

        maybe_refresh_budget_policies(store)

        store.list_budget_policies.assert_called_once()
        window = tracker._get_window_info("bp-test")
        assert window is not None


def test_maybe_refresh_skips_when_not_needed():
    with patch(_TRACKER_FUNC) as mock_get_tracker:
        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy()])
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        maybe_refresh_budget_policies(store)
        store.list_budget_policies.assert_not_called()


# --- _make_cost_recording_reducer tests ---


def test_cost_recording_reducer():
    with (
        patch(_TRACKER_FUNC) as mock_get_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.10}),
        patch("mlflow.server.gateway_budget.aggregate_chat_stream_chunks") as mock_aggregate,
    ):
        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy(budget_amount=100.0)])
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

        mock_aggregate.return_value = _FakeResponse()

        reducer = make_cost_recording_reducer(store, workspace=None)

        chunks = ["chunk1", "chunk2"]
        result = reducer(chunks)

        mock_aggregate.assert_called_once_with(chunks)
        assert result is not None
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 0.10


def test_cost_recording_reducer_no_result():
    with patch("mlflow.server.gateway_budget.aggregate_chat_stream_chunks") as mock_aggregate:
        mock_aggregate.return_value = None

        store = MagicMock()
        reducer = make_cost_recording_reducer(store, workspace=None)

        result = reducer(["chunk1"])
        assert result is None


# --- End-to-end cost → webhook test ---


def test_record_cost_triggers_webhook():
    with (
        patch(_DELIVER_FUNC) as mock_deliver,
        patch(_REGISTRY_FUNC) as mock_registry,
        patch(_TRACKER_FUNC) as mock_get_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 200.0}),
    ):
        mock_registry.return_value = MagicMock()

        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [
            _make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)
        ]

        response = _FakeResponse()
        record_budget_cost(store, response)

        mock_deliver.assert_called_once()
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 200.0
        assert window.exceeded is True


def test_record_cost_no_webhook_for_reject():
    with (
        patch(_DELIVER_FUNC) as mock_deliver,
        patch(_REGISTRY_FUNC) as mock_registry,
        patch(_TRACKER_FUNC) as mock_get_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 200.0}),
    ):
        mock_registry.return_value = MagicMock()

        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies(
            [_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)]
        )
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [
            _make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)
        ]

        response = _FakeResponse()
        record_budget_cost(store, response)

        # Crossed but REJECT → no webhook fired
        mock_deliver.assert_not_called()
        assert tracker._get_window_info("bp-test").exceeded is True


# --- calculate_existing_cost_for_new_windows tests ---


def test_calculate_existing_cost_on_new_windows():
    tracker = InMemoryBudgetTracker()
    new_windows = tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    store = MagicMock()
    store.sum_gateway_trace_cost.return_value = 42.0

    existing_spend = calculate_existing_cost_for_new_windows(store, new_windows)
    tracker.backfill_spend(existing_spend)

    store.sum_gateway_trace_cost.assert_called_once()
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 42.0


def test_calculate_existing_cost_skipped_when_no_new_windows():
    store = MagicMock()

    result = calculate_existing_cost_for_new_windows(store, [])

    assert result == {}
    store.sum_gateway_trace_cost.assert_not_called()


def test_calculate_existing_cost_handles_store_error():
    tracker = InMemoryBudgetTracker()
    new_windows = tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    store = MagicMock()
    store.sum_gateway_trace_cost.side_effect = Exception("DB error")

    # Should not raise
    existing_spend = calculate_existing_cost_for_new_windows(store, new_windows)
    tracker.backfill_spend(existing_spend)

    # Window should remain at 0 since calculation failed
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0


def test_calculate_existing_cost_zero_spend_excluded():
    tracker = InMemoryBudgetTracker()
    new_windows = tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    store = MagicMock()
    store.sum_gateway_trace_cost.return_value = 0.0

    existing_spend = calculate_existing_cost_for_new_windows(store, new_windows)
    assert existing_spend == {}

    tracker.backfill_spend(existing_spend)
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0


def test_refresh_triggers_backfill():
    tracker = InMemoryBudgetTracker()

    store = MagicMock()
    policy = _make_policy(budget_amount=100.0)
    store.list_budget_policies.return_value = [policy]
    store.sum_gateway_trace_cost.return_value = 25.0

    with patch(_TRACKER_FUNC, return_value=tracker):
        maybe_refresh_budget_policies(store)

    store.sum_gateway_trace_cost.assert_called_once()
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 25.0
