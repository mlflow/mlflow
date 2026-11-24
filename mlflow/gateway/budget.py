"""Budget tracking and enforcement for the MLflow Gateway.

This module provides budget-related functions for recording costs, refreshing policies,
firing exceeded-budget webhooks, and creating on_complete callbacks for budget recording.
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetTargetScope,
    BudgetWindowType,
    GatewayBudgetPolicy,
)
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.gateway.budget_tracker import (
    BudgetWindow,
    _sliding_bucket_start_ms,
    _sliding_bucket_width_ms,
    _sliding_window_cutoff_ms,
    _to_epoch_ms,
    get_budget_tracker,
)
from mlflow.gateway.tracing_utils import (
    _gateway_span_attributes,
    _gateway_span_name,
    _get_model_span_info,
    build_gateway_trace_metadata,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import CostKey, SpanAttributeKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME
from mlflow.webhooks.delivery import deliver_webhook
from mlflow.webhooks.types import BudgetPolicyExceededPayload

_logger = logging.getLogger(__name__)


def _trace_cost_filters(policy: GatewayBudgetPolicy) -> dict[str, str | None]:
    """Trace-history filters selecting the gateway traces whose cost a policy counts."""
    # The policy's target_value is interpreted per scope: it filters trace history by
    # gateway endpoint for ENDPOINT policies and by recorded auth username for USER
    # policies.
    return {
        "workspace": (
            policy.workspace if policy.target_scope == BudgetTargetScope.WORKSPACE else None
        ),
        "endpoint_id": (
            policy.target_value if policy.target_scope == BudgetTargetScope.ENDPOINT else None
        ),
        "username": (
            policy.target_value if policy.target_scope == BudgetTargetScope.USER else None
        ),
    }


def calculate_existing_cost_for_windows(
    store: SqlAlchemyStore, windows: list[BudgetWindow]
) -> dict[str, float]:
    """Calculate spend for fixed budget windows from trace history.

    Queries historical trace cost data for each window so that budget tracking
    accounts for spend across all gateway workers and survives server restarts.
    Sliding windows are covered by ``calculate_existing_bucket_cost_for_windows``.

    Returns:
        Dict mapping budget_policy_id to historical spend amount.
    """
    result: dict[str, float] = {}
    if not windows:
        return result

    for window in windows:
        if window.policy.window_type == BudgetWindowType.SLIDING:
            continue
        try:
            start_ms = int(window.window_start.timestamp() * 1000)
            end_ms = int(window.window_end.timestamp() * 1000)
            spend = store.sum_gateway_trace_cost(
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                **_trace_cost_filters(window.policy),
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


def calculate_existing_bucket_cost_for_windows(
    store: SqlAlchemyStore, windows: list[BudgetWindow]
) -> dict[str, dict[int, float]]:
    """Calculate the spend per bucket of sliding budget windows from trace history.

    The sliding-window counterpart of ``calculate_existing_cost_for_windows``. Spend is
    bucketed by trace start time, the same time ``record_cost`` files a request's cost
    under, so the trace history and the tracker agree on which bucket a request is in.

    Returns:
        Dict mapping budget_policy_id to spend per bucket, keyed by bucket start in epoch
        milliseconds.
    """
    result: dict[str, dict[int, float]] = {}
    now = datetime.now(timezone.utc)

    for window in windows:
        policy = window.policy
        if policy.window_type != BudgetWindowType.SLIDING:
            continue
        try:
            width_ms = _sliding_bucket_width_ms(policy.duration)
            spend = store.sum_gateway_trace_cost_by_bucket(
                start_time_ms=_sliding_window_cutoff_ms(policy.duration, width_ms, now),
                end_time_ms=_sliding_bucket_start_ms(_to_epoch_ms(now), width_ms) + width_ms,
                bucket_width_ms=width_ms,
                **_trace_cost_filters(policy),
            )
            if buckets := {start: cost for start, cost in spend.items() if cost > 0}:
                result[policy.budget_policy_id] = buckets
        except Exception:
            _logger.debug(
                "Failed to calculate existing bucket cost for policy %s",
                policy.budget_policy_id,
                exc_info=True,
            )
    return result


def _get_registry_store():
    """Return the model registry store used for webhook delivery, or None."""
    from mlflow.server.handlers import _get_model_registry_store

    try:
        return _get_model_registry_store()
    except Exception:
        return None


def maybe_refresh_budget_policies(store: SqlAlchemyStore) -> None:
    """Refresh budget policies from the database if stale.

    Seeding a window from trace history can carry it past its limit without any single
    request observing the crossing, so the windows that crossed here are alerted on too.
    Otherwise the flag would already be set by the time the next request records a cost,
    and the transition ``record_cost`` reports would never happen.
    """
    tracker = get_budget_tracker()
    if tracker.needs_refresh():
        try:
            policies = store.list_budget_policies_across_workspaces()
            windows = tracker.refresh_policies(policies)
            existing_spend = calculate_existing_cost_for_windows(store, windows)
            newly_exceeded = tracker.backfill_spend(existing_spend)
            if existing_bucket_spend := calculate_existing_bucket_cost_for_windows(store, windows):
                newly_exceeded += tracker.backfill_bucket_spend(existing_bucket_spend)
            if newly_exceeded:
                # The flag is committed by the time we get here, so delivery is
                # best-effort: if it fails the window stays muted. This is the tradeoff
                # ``record_cost`` already makes, where the flag flips atomically and
                # ``on_complete`` then delivers. Making delivery recoverable needs
                # "already alerted" persisted apart from "currently exceeded", which is
                # a larger change than this fix.
                if registry_store := _get_registry_store():
                    # No request workspace here: the payload falls back to the policy's own.
                    fire_budget_exceeded_webhooks(newly_exceeded, None, registry_store)
                else:
                    _logger.debug(
                        "%d budget window(s) crossed their limit during refresh, but no "
                        "model registry store is available to deliver the webhook",
                        len(newly_exceeded),
                    )
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
            duration_unit=policy.duration.unit.value,
            duration_value=policy.duration.value,
            target_scope=policy.target_scope.value,
            workspace=workspace or (policy.workspace or DEFAULT_WORKSPACE_NAME),
            target_value=policy.target_value,
            window_start=int(window.window_start.timestamp() * 1000),
            window_type=policy.window_type.value,
        )
        deliver_webhook(event=event, payload=payload, store=registry_store)


def _create_budget_error_trace(
    endpoint_config: GatewayEndpointConfig,
    exception: HTTPException,
    metadata: dict[str, str] | None = None,
) -> None:
    """Create an error trace for a budget limit rejection.

    Only creates a trace when usage tracking is enabled, matching the guard
    in ``maybe_traced_gateway_call``.
    """
    if not endpoint_config.usage_tracking:
        return
    try:
        with mlflow.start_span(
            name=_gateway_span_name(endpoint_config),
            span_type=SpanType.LLM,
            trace_destination=MlflowExperimentLocation(endpoint_config.experiment_id),
            attributes=_gateway_span_attributes(endpoint_config),
        ) as span:
            if metadata:
                mlflow.update_current_trace(metadata=metadata)
            span.record_exception(exception)
    except Exception:
        _logger.debug("Failed to create budget error trace", exc_info=True)


def check_budget_limit(
    store: SqlAlchemyStore,
    endpoint_config: GatewayEndpointConfig,
    workspace: str | None = None,
    username: str | None = None,
    request_headers: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    """Check if any REJECT-capable budget policy is exceeded.

    Raises HTTPException(429) with an error trace if the budget limit is exceeded.

    Args:
        store: The tracking store holding the budget policies.
        endpoint_config: The endpoint the request is routed to.
        workspace: The workspace the request was made from.
        username: The authenticated username the request was made by.
        request_headers: HTTP request headers, which identify the request's caller.
        metadata: Trace metadata for the request (e.g. auth user info), recorded on the
            error trace of a rejected request.
    """
    maybe_refresh_budget_policies(store)
    tracker = get_budget_tracker()
    exceeded, window = tracker.should_reject_request(
        workspace=workspace, endpoint_id=endpoint_config.endpoint_id, username=username
    )
    if exceeded:
        policy = window.policy
        unit = policy.duration.unit.value.lower()
        if policy.duration.value == 1:
            unit = unit.rstrip("s")
        reset_time = window.resets_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if policy.window_type == BudgetWindowType.SLIDING:
            detail = (
                f"Budget limit exceeded. "
                f"Limit: ${policy.budget_amount:.2f} USD per sliding "
                f"{policy.duration.value} {unit} window. "
                f"Spend falls back under the limit at {reset_time}. "
                "Request rejected."
            )
        else:
            detail = (
                f"Budget limit exceeded. "
                f"Limit: ${policy.budget_amount:.2f} USD per "
                f"{policy.duration.value} {unit}. "
                f"Budget resets at {reset_time}. "
                "Request rejected."
            )
        exc = HTTPException(status_code=429, detail=detail)
        trace_metadata = build_gateway_trace_metadata(endpoint_config, metadata, request_headers)
        _create_budget_error_trace(endpoint_config, exc, trace_metadata)
        raise exc


def make_budget_on_complete(
    store: SqlAlchemyStore,
    workspace: str | None,
    endpoint_id: str | None = None,
    username: str | None = None,
):
    """Create an on_complete callback that records budget cost from child span attributes."""
    registry_store = _get_registry_store()

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
            if newly_exceeded := tracker.record_cost(
                total_cost,
                workspace=workspace,
                endpoint_id=endpoint_id,
                username=username,
                # The gateway span is the trace's root span, so its start is the trace
                # timestamp that trace-history backfill buckets this request by.
                timestamp_ms=span.start_time_ns // 1_000_000,
            ):
                if registry_store:
                    fire_budget_exceeded_webhooks(newly_exceeded, workspace, registry_store)
        except Exception:
            _logger.debug("Failed to record budget cost", exc_info=True)

    return on_complete
