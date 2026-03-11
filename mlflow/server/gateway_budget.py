"""Budget tracking and enforcement for the MLflow Gateway.

This module provides budget-related functions for recording costs, refreshing policies,
firing exceeded-budget webhooks, and creating on_complete callbacks for budget recording.
"""

import logging

from fastapi import HTTPException

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.gateway_budget_policy import BudgetAction, BudgetTargetScope
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.gateway.budget_tracker import BudgetWindow, get_budget_tracker
from mlflow.gateway.tracing_utils import (
    _gateway_span_attributes,
    _gateway_span_name,
    _get_model_span_info,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import CostKey, SpanAttributeKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME
from mlflow.webhooks.delivery import deliver_webhook
from mlflow.webhooks.types import BudgetPolicyExceededPayload

_logger = logging.getLogger(__name__)


def calculate_existing_cost_for_new_windows(
    store: SqlAlchemyStore, new_windows: list[BudgetWindow]
) -> dict[str, float]:
    """Calculate existing spend for newly created budget windows from trace history.

    When a new BudgetWindow is created (server restart or new policy), its
    cumulative_spend starts at 0. This queries historical trace cost data so
    that budget tracking survives server restarts.

    Returns:
        Dict mapping budget_policy_id to historical spend amount.
    """
    result: dict[str, float] = {}
    if not new_windows:
        return result

    for window in new_windows:
        try:
            start_ms = int(window.window_start.timestamp() * 1000)
            end_ms = int(window.window_end.timestamp() * 1000)
            workspace = (
                window.policy.workspace
                if window.policy.target_scope == BudgetTargetScope.WORKSPACE
                else None
            )
            spend = store.sum_gateway_trace_cost(
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                workspace=workspace,
            )
            if spend > 0:
                result[window.policy.budget_policy_id] = spend
        except Exception:
            _logger.debug(
                "Failed to calculate existing cost for policy %s",
                window.policy.budget_policy_id,
                exc_info=True,
            )
    return result


def maybe_refresh_budget_policies(store: SqlAlchemyStore) -> None:
    """Refresh budget policies from the database if stale."""
    tracker = get_budget_tracker()
    if tracker.needs_refresh():
        try:
            policies = store.list_budget_policies()
            new_windows = tracker.refresh_policies(policies)
            existing_spend = calculate_existing_cost_for_new_windows(store, new_windows)
            tracker.backfill_spend(existing_spend)
        except Exception:
            _logger.debug("Failed to refresh budget policies", exc_info=True)


def _compute_cost_from_child_spans(trace_id: str) -> float:
    """Sum total cost across child spans.

    Prefers ``LLM_COST`` if already set on the span (computed at span.end()),
    otherwise falls back to calculating from MODEL + CHAT_USAGE via LiteLLM.
    """
    total = 0.0
    for info in _get_model_span_info(trace_id):
        if llm_cost := info.attributes.get(SpanAttributeKey.LLM_COST):
            total += llm_cost.get(CostKey.TOTAL_COST, 0.0)
        else:
            model_name = info.attributes.get(SpanAttributeKey.MODEL)
            usage = info.attributes.get(SpanAttributeKey.CHAT_USAGE)
            if not usage:
                continue
            model_provider = info.attributes.get(SpanAttributeKey.MODEL_PROVIDER)
            if cost := calculate_cost_by_model_and_token_usage(model_name, usage, model_provider):
                total += cost.get(CostKey.TOTAL_COST, 0.0)
    return total


def fire_budget_exceeded_webhooks(
    newly_exceeded: list[BudgetWindow],
    workspace: str | None,
    registry_store,
) -> None:
    """Fire budget_policy.exceeded webhooks for newly-exceeded budget windows."""
    event = WebhookEvent(WebhookEntity.BUDGET_POLICY, WebhookAction.EXCEEDED)

    for window in newly_exceeded:
        policy = window.policy
        if policy.budget_action != BudgetAction.ALERT:
            continue

        payload = BudgetPolicyExceededPayload(
            budget_policy_id=policy.budget_policy_id,
            budget_unit=policy.budget_unit.value,
            budget_amount=policy.budget_amount,
            current_spend=window.cumulative_spend,
            duration_unit=policy.duration_unit.value,
            duration_value=policy.duration_value,
            target_scope=policy.target_scope.value,
            workspace=workspace or (policy.workspace or DEFAULT_WORKSPACE_NAME),
            window_start=int(window.window_start.timestamp() * 1000),
        )
        deliver_webhook(event=event, payload=payload, store=registry_store)


def _create_budget_error_trace(
    endpoint_config: GatewayEndpointConfig,
    exception: HTTPException,
) -> None:
    """Create an error trace for a budget limit rejection.

    Only creates a trace when usage tracking is enabled (the endpoint has an
    experiment_id), matching the guard in ``maybe_traced_gateway_call``.
    """
    if not endpoint_config.experiment_id:
        return
    try:
        with mlflow.start_span(
            name=_gateway_span_name(endpoint_config),
            span_type=SpanType.LLM,
            trace_destination=MlflowExperimentLocation(endpoint_config.experiment_id),
            attributes=_gateway_span_attributes(endpoint_config),
        ) as span:
            span.record_exception(exception)
    except Exception:
        _logger.debug("Failed to create budget error trace", exc_info=True)


def check_budget_limit(
    store: SqlAlchemyStore,
    endpoint_config: GatewayEndpointConfig,
    workspace: str | None = None,
) -> None:
    """Check if any REJECT-capable budget policy is exceeded.

    Raises HTTPException(429) with an error trace if the budget limit is exceeded.
    """
    maybe_refresh_budget_policies(store)
    tracker = get_budget_tracker()
    exceeded, window = tracker.should_reject_request(workspace=workspace)
    if exceeded:
        policy = window.policy
        unit = policy.duration_unit.value.lower()
        if policy.duration_value == 1:
            unit = unit.rstrip("s")
        reset_time = window.window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        detail = (
            f"Budget limit exceeded. "
            f"Limit: ${policy.budget_amount:.2f} USD per "
            f"{policy.duration_value} {unit}. "
            f"Budget resets at {reset_time}. "
            "Request rejected."
        )
        exc = HTTPException(status_code=429, detail=detail)
        _create_budget_error_trace(endpoint_config, exc)
        raise exc


def make_budget_on_complete(
    store: SqlAlchemyStore,
    workspace: str | None,
):
    """Create an on_complete callback that records budget cost from child span attributes."""
    from mlflow.server.handlers import _get_model_registry_store

    try:
        registry_store = _get_model_registry_store()
    except Exception:
        registry_store = None

    def on_complete():
        try:
            span = mlflow.get_current_active_span()
            if not span:
                return

            total_cost = _compute_cost_from_child_spans(span.trace_id)
            if total_cost <= 0:
                return

            maybe_refresh_budget_policies(store)
            tracker = get_budget_tracker()
            if newly_exceeded := tracker.record_cost(total_cost, workspace=workspace):
                if registry_store:
                    fire_budget_exceeded_webhooks(newly_exceeded, workspace, registry_store)
        except Exception:
            _logger.debug("Failed to record budget cost", exc_info=True)

    return on_complete
