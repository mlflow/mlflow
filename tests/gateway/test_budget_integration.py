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
    make_budget_on_complete,
    maybe_refresh_budget_policies,
)
from mlflow.tracing.constant import CostKey, SpanAttributeKey

_COST_FUNC = "mlflow.server.gateway_budget.calculate_cost_by_model_and_token_usage"
_TRACKER_FUNC = "mlflow.server.gateway_budget.get_budget_tracker"
_DELIVER_FUNC = "mlflow.server.gateway_budget.deliver_webhook"
_REGISTRY_FUNC = "mlflow.server.gateway_budget._get_model_registry_store"
_MODEL_SPAN_INFO_FUNC = "mlflow.gateway.tracing_utils._get_model_span_info"


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


def _make_model_span_info(
    model="gpt-4o",
    provider="openai",
    usage=None,
):
    """Create a mock _ModelSpanInfo with the given span attributes."""
    from mlflow.gateway.tracing_utils import _ModelSpanInfo

    attrs = {}
    if model:
        attrs[SpanAttributeKey.MODEL] = model
    if provider:
        attrs[SpanAttributeKey.MODEL_PROVIDER] = provider
    if usage is not None:
        attrs[SpanAttributeKey.CHAT_USAGE] = usage
    return _ModelSpanInfo(name=f"model/{provider}/{model}", attributes=attrs)


def _mock_active_span():
    """Create a mock span with a trace_id."""
    span = MagicMock()
    span.trace_id = "test-trace-id"
    return span


# --- make_budget_on_complete tests ---


def test_budget_on_complete():
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    with (
        patch(_TRACKER_FUNC) as mock_get_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 0.05}),
        patch(_MODEL_SPAN_INFO_FUNC, return_value=[_make_model_span_info(usage=usage)]),
        patch("mlflow.get_current_active_span", return_value=_mock_active_span()),
    ):
        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies([_make_policy(budget_amount=100.0)])
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]

        on_complete = make_budget_on_complete(store, workspace=None)
        on_complete()

        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 0.05


def test_budget_on_complete_no_span():
    with patch("mlflow.get_current_active_span", return_value=None):
        store = MagicMock()
        on_complete = make_budget_on_complete(store, workspace=None)
        on_complete()  # should not raise


def test_budget_on_complete_no_child_spans():
    with (
        patch(_MODEL_SPAN_INFO_FUNC, return_value=[]),
        patch("mlflow.get_current_active_span", return_value=_mock_active_span()),
    ):
        store = MagicMock()
        on_complete = make_budget_on_complete(store, workspace=None)
        on_complete()  # should not raise


def test_budget_on_complete_no_cost():
    usage = {"input_tokens": 100, "output_tokens": 50}
    with (
        patch(_COST_FUNC, return_value=None),
        patch(_MODEL_SPAN_INFO_FUNC, return_value=[_make_model_span_info(usage=usage)]),
        patch("mlflow.get_current_active_span", return_value=_mock_active_span()),
    ):
        store = MagicMock()
        on_complete = make_budget_on_complete(store, workspace=None)
        on_complete()  # should not raise


def test_budget_on_complete_triggers_webhook():
    usage = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
    with (
        patch(_DELIVER_FUNC) as mock_deliver,
        patch(_REGISTRY_FUNC) as mock_registry,
        patch(_TRACKER_FUNC) as mock_get_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 200.0}),
        patch(_MODEL_SPAN_INFO_FUNC, return_value=[_make_model_span_info(usage=usage)]),
        patch("mlflow.get_current_active_span", return_value=_mock_active_span()),
    ):
        mock_registry.return_value = MagicMock()

        tracker = InMemoryBudgetTracker()
        tracker.refresh_policies(
            [_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)]
        )
        mock_get_tracker.return_value = tracker

        store = MagicMock()
        store.list_budget_policies.return_value = [
            _make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)
        ]

        on_complete = make_budget_on_complete(store, workspace=None)
        on_complete()

        mock_deliver.assert_called_once()
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == 200.0
        assert window.exceeded is True


def test_budget_on_complete_no_webhook_for_reject():
    usage = {"input_tokens": 1000, "output_tokens": 500}
    with (
        patch(_DELIVER_FUNC) as mock_deliver,
        patch(_REGISTRY_FUNC) as mock_registry,
        patch(_TRACKER_FUNC) as mock_get_tracker,
        patch(_COST_FUNC, return_value={CostKey.TOTAL_COST: 200.0}),
        patch(_MODEL_SPAN_INFO_FUNC, return_value=[_make_model_span_info(usage=usage)]),
        patch("mlflow.get_current_active_span", return_value=_mock_active_span()),
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

        on_complete = make_budget_on_complete(store, workspace=None)
        on_complete()

        mock_deliver.assert_not_called()
        assert tracker._get_window_info("bp-test").exceeded is True


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
