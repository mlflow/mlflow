from unittest.mock import MagicMock, patch

import fastapi
import pytest

import mlflow
import mlflow.gateway.budget_tracker as _bt_module
from mlflow.entities import SpanStatusCode, SpanType
from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget import (
    calculate_existing_cost_for_windows,
    check_budget_limit,
    fire_budget_exceeded_webhooks,
    make_budget_on_complete,
    maybe_refresh_budget_policies,
)
from mlflow.gateway.budget_tracker import get_budget_tracker
from mlflow.gateway.tracing_utils import maybe_traced_gateway_call
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracing.constant import CostKey, SpanAttributeKey
from mlflow.tracking.fluent import _get_experiment_id

_DELIVER_FUNC = "mlflow.gateway.budget.deliver_webhook"


@pytest.fixture(autouse=True)
def _reset_budget_tracker():
    _bt_module._budget_tracker = None
    yield
    _bt_module._budget_tracker = None


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    budget_action=BudgetAction.ALERT,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=budget_action,
        created_at=0,
        last_updated_at=0,
    )


def _make_endpoint_config(experiment_id=None):
    return GatewayEndpointConfig(
        endpoint_id="ep-test",
        endpoint_name="test-endpoint",
        experiment_id=experiment_id or _get_experiment_id(),
        models=[],
    )


_NO_TRACE_CONFIG = GatewayEndpointConfig(
    endpoint_id="ep-test",
    endpoint_name="test-endpoint",
    experiment_id=None,
    models=[],
)


def _make_store(policies=None):
    store = MagicMock()
    store.list_budget_policies.return_value = policies or []
    store.sum_gateway_trace_cost.return_value = 0.0
    return store


async def maybe_traced_call(provider_func, endpoint_config, on_complete):
    traced = maybe_traced_gateway_call(provider_func, endpoint_config, on_complete=on_complete)
    return await traced({"messages": [{"role": "user", "content": "test"}]})


async def _provider_with_cost(payload):
    """Simulates a provider that sets LLM_COST on its child span."""
    with mlflow.start_span("provider/openai/gpt-4o", span_type=SpanType.LLM) as span:
        span.set_attributes({
            SpanAttributeKey.MODEL: "gpt-4o",
            SpanAttributeKey.MODEL_PROVIDER: "openai",
            SpanAttributeKey.LLM_COST: {
                CostKey.INPUT_COST: 0.025,
                CostKey.OUTPUT_COST: 0.050,
                CostKey.TOTAL_COST: 0.075,
            },
        })
    return {"choices": [{"message": {"content": "Hello"}}]}


async def _provider_no_cost(payload):
    """Simulates a provider that creates a span without cost attributes."""
    with mlflow.start_span("provider/custom/no-cost", span_type=SpanType.LLM) as span:
        span.set_attribute(SpanAttributeKey.MODEL, "custom-model")
    return {"choices": [{"message": {"content": "Hello"}}]}


# --- make_budget_on_complete integration tests ---


@pytest.mark.asyncio
async def test_budget_on_complete_records_cost():
    policy = _make_policy(budget_amount=100.0)
    store = _make_store(policies=[policy])

    on_complete = make_budget_on_complete(store, workspace=None)
    await maybe_traced_call(_provider_with_cost, _make_endpoint_config(), on_complete)

    tracker = get_budget_tracker()
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == pytest.approx(0.075)


@pytest.mark.asyncio
async def test_budget_on_complete_no_span():
    store = _make_store()
    on_complete = make_budget_on_complete(store, workspace=None)
    on_complete()  # called outside trace context — should not raise


@pytest.mark.asyncio
async def test_budget_on_complete_no_cost():
    policy = _make_policy(budget_amount=100.0)
    store = _make_store(policies=[policy])

    on_complete = make_budget_on_complete(store, workspace=None)
    await maybe_traced_call(_provider_no_cost, _make_endpoint_config(), on_complete)

    # No cost was computed, so record_cost was never called.
    # The tracker may or may not have refreshed policies (early return before refresh).
    tracker = get_budget_tracker()
    window = tracker._get_window_info("bp-test")
    if window is not None:
        assert window.cumulative_spend == 0.0


@pytest.mark.asyncio
async def test_budget_on_complete_triggers_webhook():
    with patch(_DELIVER_FUNC) as mock_deliver:
        policy = _make_policy(budget_amount=0.05, budget_action=BudgetAction.ALERT)
        store = _make_store(policies=[policy])
        endpoint_config = _make_endpoint_config()

        on_complete = make_budget_on_complete(store, workspace=None)
        await maybe_traced_call(_provider_with_cost, endpoint_config, on_complete)

        mock_deliver.assert_called_once()
        payload = mock_deliver.call_args.kwargs["payload"]
        assert payload["budget_policy_id"] == "bp-test"
        assert payload["budget_amount"] == 0.05
        assert payload["current_spend"] == pytest.approx(0.075)


@pytest.mark.asyncio
async def test_budget_on_complete_no_webhook_for_reject():
    with patch(_DELIVER_FUNC) as mock_deliver:
        policy = _make_policy(budget_amount=0.05, budget_action=BudgetAction.REJECT)
        store = _make_store(policies=[policy])
        endpoint_config = _make_endpoint_config()

        on_complete = make_budget_on_complete(store, workspace=None)
        await maybe_traced_call(_provider_with_cost, endpoint_config, on_complete)

        mock_deliver.assert_not_called()
        tracker = get_budget_tracker()
        assert tracker._get_window_info("bp-test").exceeded is True


# --- multi-invocation integration test ---


@pytest.mark.asyncio
async def test_budget_accumulates_over_multiple_invocations():
    with patch(_DELIVER_FUNC) as mock_deliver:
        policy = _make_policy(budget_amount=0.20, budget_action=BudgetAction.ALERT)
        store = _make_store(policies=[policy])
        endpoint_config = _make_endpoint_config()

        # Call 1: 0.075 spend, under budget
        on_complete = make_budget_on_complete(store, workspace=None)
        await maybe_traced_call(_provider_with_cost, endpoint_config, on_complete)
        mock_deliver.assert_not_called()

        # Call 2: 0.15 spend, still under budget
        on_complete = make_budget_on_complete(store, workspace=None)
        await maybe_traced_call(_provider_with_cost, endpoint_config, on_complete)
        mock_deliver.assert_not_called()

        # Call 3: 0.225 spend, exceeds $0.20 budget → webhook fires
        on_complete = make_budget_on_complete(store, workspace=None)
        await maybe_traced_call(_provider_with_cost, endpoint_config, on_complete)
        mock_deliver.assert_called_once()

        tracker = get_budget_tracker()
        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == pytest.approx(0.225)
        assert window.exceeded is True

        # Call 4: already exceeded, webhook should not fire again
        on_complete = make_budget_on_complete(store, workspace=None)
        await maybe_traced_call(_provider_with_cost, endpoint_config, on_complete)
        mock_deliver.assert_called_once()  # still just the one call

        window = tracker._get_window_info("bp-test")
        assert window.cumulative_spend == pytest.approx(0.30)


# --- fire_budget_exceeded_webhooks tests ---


def test_fire_budget_exceeded_webhooks_alert():
    with patch(_DELIVER_FUNC) as mock_deliver:
        tracker = get_budget_tracker()
        policy = _make_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
        tracker.refresh_policies([policy])
        crossed = tracker.record_cost(60.0)
        assert len(crossed) == 1

        fire_budget_exceeded_webhooks(crossed, workspace=None, registry_store=MagicMock())
        mock_deliver.assert_called_once()

        payload = mock_deliver.call_args.kwargs["payload"]
        assert payload["budget_policy_id"] == "bp-test"
        assert payload["budget_amount"] == 50.0
        assert payload["current_spend"] == 60.0


def test_fire_budget_exceeded_webhooks_reject_skipped():
    with patch(_DELIVER_FUNC) as mock_deliver:
        tracker = get_budget_tracker()
        policy = _make_policy(budget_amount=50.0, budget_action=BudgetAction.REJECT)
        tracker.refresh_policies([policy])
        crossed = tracker.record_cost(60.0)
        assert len(crossed) == 1

        fire_budget_exceeded_webhooks(crossed, workspace=None, registry_store=MagicMock())
        mock_deliver.assert_not_called()


def test_fire_budget_exceeded_webhooks_with_workspace():
    with patch(_DELIVER_FUNC) as mock_deliver:
        tracker = get_budget_tracker()
        policy = _make_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
        tracker.refresh_policies([policy])
        crossed = tracker.record_cost(60.0)

        fire_budget_exceeded_webhooks(crossed, workspace="my-ws", registry_store=MagicMock())
        payload = mock_deliver.call_args.kwargs["payload"]
        assert payload["workspace"] == "my-ws"


# --- maybe_refresh_budget_policies tests ---


def test_maybe_refresh_budget_policies():
    store = MagicMock()
    store.list_budget_policies.return_value = [_make_policy()]
    store.sum_gateway_trace_cost.return_value = 0.0

    maybe_refresh_budget_policies(store)

    store.list_budget_policies.assert_called_once()
    tracker = get_budget_tracker()
    assert tracker._get_window_info("bp-test") is not None


def test_maybe_refresh_skips_when_not_needed():
    tracker = get_budget_tracker()
    tracker.refresh_policies([_make_policy()])

    store = MagicMock()
    maybe_refresh_budget_policies(store)
    store.list_budget_policies.assert_not_called()


# --- calculate_existing_cost_for_windows tests ---


def test_calculate_existing_cost_on_new_windows():
    tracker = get_budget_tracker()
    new_windows = tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    store = MagicMock()
    store.sum_gateway_trace_cost.return_value = 42.0

    existing_spend = calculate_existing_cost_for_windows(store, new_windows)
    tracker.backfill_spend(existing_spend)

    store.sum_gateway_trace_cost.assert_called_once()
    assert tracker._get_window_info("bp-test").cumulative_spend == 42.0


def test_calculate_existing_cost_skipped_when_no_new_windows():
    store = MagicMock()
    result = calculate_existing_cost_for_windows(store, [])
    assert result == {}
    store.sum_gateway_trace_cost.assert_not_called()


def test_calculate_existing_cost_handles_store_error():
    tracker = get_budget_tracker()
    new_windows = tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    store = MagicMock()
    store.sum_gateway_trace_cost.side_effect = Exception("DB error")

    existing_spend = calculate_existing_cost_for_windows(store, new_windows)
    tracker.backfill_spend(existing_spend)

    assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_calculate_existing_cost_zero_spend_excluded():
    tracker = get_budget_tracker()
    new_windows = tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    store = MagicMock()
    store.sum_gateway_trace_cost.return_value = 0.0

    existing_spend = calculate_existing_cost_for_windows(store, new_windows)
    assert existing_spend == {}

    tracker.backfill_spend(existing_spend)
    assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_refresh_triggers_backfill():
    store = MagicMock()
    store.list_budget_policies.return_value = [_make_policy(budget_amount=100.0)]
    store.sum_gateway_trace_cost.return_value = 25.0

    maybe_refresh_budget_policies(store)

    store.sum_gateway_trace_cost.assert_called_once()
    tracker = get_budget_tracker()
    assert tracker._get_window_info("bp-test").cumulative_spend == 25.0


# --- check_budget_limit tests ---


def test_check_budget_limit_no_policies():
    store = _make_store(policies=[])
    check_budget_limit(store, _NO_TRACE_CONFIG)


def test_check_budget_limit_not_exceeded():
    policy = _make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(50.0)

    check_budget_limit(store, _NO_TRACE_CONFIG)


def test_check_budget_limit_exceeded_rejects():
    policy = _make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(150.0)

    with pytest.raises(fastapi.HTTPException, match="Request rejected"):
        check_budget_limit(store, _NO_TRACE_CONFIG)


def test_check_budget_limit_alert_does_not_reject():
    policy = _make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(150.0)

    check_budget_limit(store, _NO_TRACE_CONFIG)


def test_check_budget_limit_error_message_format():
    policy = _make_policy(
        budget_policy_id="bp-monthly",
        budget_amount=500.0,
        budget_action=BudgetAction.REJECT,
    )
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(600.0)

    with pytest.raises(fastapi.HTTPException, match="Request rejected") as exc_info:
        check_budget_limit(store, _NO_TRACE_CONFIG)

    detail = exc_info.value.detail
    assert "$500.00" in detail
    assert "1 day" in detail
    assert "resets at" in detail
    assert "Request rejected" in detail


def test_check_budget_limit_error_message_plural():
    policy = GatewayBudgetPolicy(
        budget_policy_id="bp-plural",
        budget_unit=BudgetUnit.USD,
        budget_amount=200.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=3),
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=BudgetAction.REJECT,
        created_at=0,
        last_updated_at=0,
    )
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(300.0)

    with pytest.raises(fastapi.HTTPException, match="Request rejected") as exc_info:
        check_budget_limit(store, _NO_TRACE_CONFIG)

    detail = exc_info.value.detail
    assert "$200.00" in detail
    assert "3 months" in detail
    assert "resets at" in detail
    assert "Request rejected" in detail


def test_check_budget_limit_with_workspace():
    policy = GatewayBudgetPolicy(
        budget_policy_id="bp-ws",
        budget_unit=BudgetUnit.USD,
        budget_amount=50.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope=BudgetTargetScope.WORKSPACE,
        budget_action=BudgetAction.REJECT,
        created_at=0,
        last_updated_at=0,
        workspace="ws1",
    )
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(100.0, workspace="ws1")

    with pytest.raises(fastapi.HTTPException, match="Request rejected"):
        check_budget_limit(store, _NO_TRACE_CONFIG, workspace="ws1")

    check_budget_limit(store, _NO_TRACE_CONFIG, workspace="ws2")


def test_check_budget_limit_multiple_policies():
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
    store = _make_store(policies=[alert_policy, reject_policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([alert_policy, reject_policy])

    # 75 exceeds alert (50) but not reject (100) → no rejection
    tracker.record_cost(75.0)
    check_budget_limit(store, _NO_TRACE_CONFIG)

    # Push to 105 → exceeds reject policy → should raise
    tracker.record_cost(30.0)
    with pytest.raises(fastapi.HTTPException, match="Request rejected"):
        check_budget_limit(store, _NO_TRACE_CONFIG)


# --- _create_budget_error_trace tests ---


def test_check_budget_limit_creates_error_trace_when_exceeded():
    policy = _make_policy(budget_amount=10.0, budget_action=BudgetAction.REJECT)
    store = _make_store(policies=[policy])
    endpoint_config = _make_endpoint_config()

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(20.0)

    with pytest.raises(fastapi.HTTPException, match="Request rejected"):
        check_budget_limit(store, endpoint_config)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    root_span = trace.data.spans[0]
    assert root_span.name == "gateway/test-endpoint"
    assert root_span.status.status_code == SpanStatusCode.ERROR
    assert len(root_span.events) == 1
    assert root_span.events[0].name == "exception"


def test_check_budget_limit_no_trace_without_experiment_id():
    policy = _make_policy(budget_amount=10.0, budget_action=BudgetAction.REJECT)
    store = _make_store(policies=[policy])

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(20.0)

    with pytest.raises(fastapi.HTTPException, match="Request rejected"):
        check_budget_limit(store, _NO_TRACE_CONFIG)

    assert mlflow.get_last_active_trace_id() is None


def test_check_budget_limit_no_trace_when_under_budget():
    policy = _make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)
    store = _make_store(policies=[policy])
    endpoint_config = _make_endpoint_config()

    tracker = get_budget_tracker()
    tracker.refresh_policies([policy])
    tracker.record_cost(50.0)

    check_budget_limit(store, endpoint_config)

    assert mlflow.get_last_active_trace_id() is None
