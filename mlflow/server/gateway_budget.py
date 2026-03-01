"""Budget tracking and enforcement for the MLflow Gateway.

This module provides budget-related functions for recording costs, refreshing policies,
firing exceeded-budget webhooks, and creating streaming cost-recording reducers.
"""

import logging
from typing import Any

from fastapi import HTTPException

from mlflow.entities.gateway_budget_policy import BudgetTargetScope
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.gateway.budget_tracker import BudgetWindow, get_budget_tracker
from mlflow.gateway.tracing_utils import aggregate_chat_stream_chunks
from mlflow.server.handlers import _get_model_registry_store
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import CostKey, TokenUsageKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage
from mlflow.webhooks.delivery import deliver_webhook
from mlflow.webhooks.types import BudgetPolicyExceededPayload

_logger = logging.getLogger(__name__)


def backfill_budget_spend(store: SqlAlchemyStore, tracker, new_windows) -> None:
    """Backfill cumulative spend for newly created budget windows from trace history.

    When a new BudgetWindow is created (server restart or new policy), its
    cumulative_spend starts at 0. This queries historical trace cost data to
    seed the spend so budgets aren't effectively reset on restart.
    """
    if not new_windows:
        return

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
                tracker.backfill_spend(window.policy.budget_policy_id, spend)
        except Exception:
            _logger.debug(
                "Failed to backfill budget spend for policy %s",
                window.policy.budget_policy_id,
                exc_info=True,
            )


def maybe_refresh_budget_policies(store: SqlAlchemyStore) -> None:
    """Refresh budget policies from the database if stale."""
    tracker = get_budget_tracker()
    if tracker.needs_refresh():
        try:
            policies = store.list_budget_policies()
            new_windows = tracker.load_policies(policies)
            backfill_budget_spend(store, tracker, new_windows)
        except Exception:
            _logger.debug("Failed to refresh budget policies", exc_info=True)


def maybe_record_budget_cost(
    store: SqlAlchemyStore,
    response: Any,
    model_name: str | None = None,
    model_provider: str | None = None,
    workspace: str | None = None,
) -> None:
    """Record cost from a gateway response against budget policies.

    Extracts token usage from the response, calculates cost via LiteLLM,
    records it in the budget tracker, and fires webhooks for newly-exceeded
    budget windows.

    This is a best-effort operation; errors are logged but not raised.
    """
    try:
        # Extract token usage from response
        usage_dict = None
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage
            usage_dict = {}
            if hasattr(usage, "prompt_tokens") and usage.prompt_tokens is not None:
                usage_dict[TokenUsageKey.INPUT_TOKENS] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens") and usage.completion_tokens is not None:
                usage_dict[TokenUsageKey.OUTPUT_TOKENS] = usage.completion_tokens
            if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
                usage_dict[TokenUsageKey.TOTAL_TOKENS] = usage.total_tokens
        elif isinstance(response, dict):
            if raw_usage := response.get("usage"):
                usage_dict = {
                    TokenUsageKey.INPUT_TOKENS: raw_usage.get(
                        "prompt_tokens", raw_usage.get("input_tokens", 0)
                    ),
                    TokenUsageKey.OUTPUT_TOKENS: raw_usage.get(
                        "completion_tokens", raw_usage.get("output_tokens", 0)
                    ),
                }

        if not usage_dict:
            return

        cost = calculate_cost_by_model_and_token_usage(
            model_name=model_name,
            usage=usage_dict,
            model_provider=model_provider,
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
            workspace=workspace or (policy.workspace or "default"),
            window_start=int(window.window_start.timestamp() * 1000),
        )
        deliver_webhook(event=event, payload=payload, store=registry_store)


def check_budget_limit(store: SqlAlchemyStore, workspace: str | None = None) -> None:
    """Check if any REJECT-capable budget policy is exceeded.

    Raises HTTPException(429) if the budget limit is exceeded.
    """
    maybe_refresh_budget_policies(store)
    tracker = get_budget_tracker()
    exceeded, policy = tracker.should_reject_request(workspace=workspace)
    if exceeded:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Budget limit exceeded for policy '{policy.budget_policy_id}'. "
                f"Limit: ${policy.budget_amount:.2f} USD per "
                f"{policy.duration_value} {policy.duration_unit.value.lower()}. "
                "Request rejected."
            ),
        )


def get_model_info(
    endpoint_config: GatewayEndpointConfig,
) -> tuple[str | None, str | None]:
    """Extract model_name and provider from endpoint config."""
    if endpoint_config.models:
        m = endpoint_config.models[0]
        return m.model_name, m.provider
    return None, None


def make_cost_recording_reducer(
    store: SqlAlchemyStore,
    endpoint_config: GatewayEndpointConfig,
    workspace: str | None,
):
    """Create an output_reducer that aggregates stream chunks and records budget cost."""

    def reducer(chunks):
        result = aggregate_chat_stream_chunks(chunks)
        if result:
            model_name, model_provider = get_model_info(endpoint_config)
            maybe_record_budget_cost(
                store,
                result,
                model_name=model_name,
                model_provider=model_provider,
                workspace=workspace,
            )
        return result

    return reducer
