"""Budget tracking and enforcement for the MLflow Gateway.

This module provides budget-related functions for recording costs, refreshing policies,
firing exceeded-budget webhooks, and creating streaming cost-recording reducers.
"""

import logging
from typing import Any

from mlflow.entities.gateway_budget_policy import BudgetTargetScope
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.gateway.budget_tracker import BudgetWindow, get_budget_tracker
from mlflow.gateway.tracing_utils import aggregate_chat_stream_chunks
from mlflow.server.handlers import _get_model_registry_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import CostKey
from mlflow.tracing.utils import (
    calculate_cost_by_model_and_token_usage,
    extract_token_usage_from_response,
)
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


def _extract_model_name(response: Any) -> str | None:
    """Extract model name from a gateway response (object or dict)."""
    if hasattr(response, "model"):
        return response.model
    if isinstance(response, dict):
        return response.get("model")
    return None


def record_budget_cost(
    store: SqlAlchemyStore,
    response: Any,
    workspace: str | None = None,
) -> None:
    """Record cost from a gateway response against budget policies.

    Extracts token usage and model name from the response, calculates cost
    via LiteLLM, records it in the budget tracker, and fires webhooks for
    newly-exceeded budget windows.

    This is a best-effort operation; errors are logged but not raised.
    """
    try:
        usage_dict = extract_token_usage_from_response(response)
        if not usage_dict:
            return

        model_name = _extract_model_name(response)
        cost = calculate_cost_by_model_and_token_usage(
            model_name=model_name,
            usage=usage_dict,
        )
        if not cost:
            return

        total_cost = cost.get(CostKey.TOTAL_COST, 0.0)
        if total_cost <= 0:
            return

        maybe_refresh_budget_policies(store)
        tracker = get_budget_tracker()
        # Fire webhooks for newly-exceeded budget windows
        if newly_exceeded := tracker.record_cost(total_cost, workspace=workspace):
            fire_budget_exceeded_webhooks(newly_exceeded, workspace)
    except Exception:
        _logger.debug("Failed to record budget cost", exc_info=True)


def fire_budget_exceeded_webhooks(
    newly_exceeded: list[BudgetWindow], workspace: str | None
) -> None:
    """Fire budget_policy.exceeded webhooks for newly-exceeded budget windows."""
    try:
        registry_store = _get_model_registry_store()
    except Exception:
        _logger.debug("Failed to get model registry store for webhook delivery", exc_info=True)
        return

    event = WebhookEvent(WebhookEntity.BUDGET_POLICY, WebhookAction.EXCEEDED)

    for window in newly_exceeded:
        policy = window.policy
        if policy.budget_action.value != "ALERT":
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


def make_cost_recording_reducer(
    store: SqlAlchemyStore,
    workspace: str | None,
):
    """Create an output_reducer that aggregates stream chunks and records budget cost."""

    def reducer(chunks):
        result = aggregate_chat_stream_chunks(chunks)
        if result:
            record_budget_cost(store, result, workspace=workspace)
        return result

    return reducer
