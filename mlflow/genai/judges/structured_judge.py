"""Model-agnostic routing for structured (Boolean/categorical) judges.

Built-in judges and ``make_judge`` scorers evaluate with a plain chat model by default. When
the configured model is a TypeSafe System One model -- addressed directly (``typesafe:/``) or
through an AI Gateway endpoint (``gateway:/``) -- the same judge must instead go through the
System One evaluation API. This module owns that routing decision and the gateway chat-first
fallback, keeping ``typesafe.py`` a pure System One client.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import SYSTEM_ONE_CHAT_ROUTE_REJECTION_DETAIL
from mlflow.genai.judges.adapters.utils import ChatCompletionError
from mlflow.genai.judges.typesafe import (
    _GATEWAY_PROVIDER,
    _GatewayEndpointNotSystemOne,
    _invoke_typesafe_judge,
    _is_typesafe_model,
)
from mlflow.genai.utils.gateway_utils import _resolve_gateway_uri

# Remember which gateway endpoints serve System One models so jev judges skip the chat
# attempt on every row after the first detection. Membership is permanent for the life of the
# process -- an endpoint essentially never changes its model type mid-run -- so no TTL is
# needed. If one is reconfigured away from System One, the entry is dropped lazily when the
# System One route rejects it (see ``_invoke_gateway_judge``), so the cache still self-heals.
#
# No lock is used. Under a thread-pool evaluation several threads can race before the entry
# lands, so a few may each make the chat-first probe. That is harmless: the probe is idempotent
# and the worst case is a handful of extra rejected chat calls on the first batch of rows.
_gateway_system_one_cache: set[tuple[str, str]] = set()


def _is_gateway_model(model_uri: str) -> bool:
    provider, separator, _ = model_uri.partition(":/")
    return bool(separator) and provider == _GATEWAY_PROVIDER


def _gateway_system_one_cache_key(model_uri: str) -> tuple[str, str]:
    return (_resolve_gateway_uri(), model_uri)


def _is_gateway_system_one_rejection(exc: BaseException) -> bool:
    """True when a chat invocation failed because the endpoint only serves System One models.

    The gateway chat adapter raises ``ChatCompletionError`` for a non-2xx chat response and
    wraps it in an ``MlflowException`` (``raise ... from e``), so the signal lives on the
    cause chain: a 400 whose body ``{"detail": ...}`` equals the shared rejection constant.
    """
    cause = getattr(exc, "__cause__", None)
    if not isinstance(cause, ChatCompletionError) or cause.status_code != 400:
        return False
    message = cause.message
    if message == SYSTEM_ONE_CHAT_ROUTE_REJECTION_DETAIL:
        return True
    try:
        detail = json.loads(message).get("detail")
    except (TypeError, ValueError, AttributeError):
        return False
    return detail == SYSTEM_ONE_CHAT_ROUTE_REJECTION_DETAIL


def _invoke_gateway_judge(model_uri: str, *, chat_invoker, **kwargs) -> Feedback:
    """Evaluate a ``gateway:/`` judge, preferring chat and falling back to System One.

    Non-jev endpoints (the pre-existing majority) keep going straight to chat, so they see no
    regression. A System One endpoint rejects chat with a known 400; the judge then switches to
    the System One route and remembers the endpoint so later rows skip the chat attempt.
    """
    cache_key = _gateway_system_one_cache_key(model_uri)
    if cache_key in _gateway_system_one_cache:
        try:
            return _invoke_typesafe_judge(model_uri, **kwargs)
        except _GatewayEndpointNotSystemOne:
            # Endpoint was reconfigured away from System One; drop the entry and fall back to chat.
            _gateway_system_one_cache.discard(cache_key)
    try:
        return chat_invoker()
    except MlflowException as e:
        if not _is_gateway_system_one_rejection(e):
            raise
    _gateway_system_one_cache.add(cache_key)
    return _invoke_typesafe_judge(model_uri, **kwargs)


def _invoke_structured_builtin_judge(
    model_uri: str,
    *,
    chat_invoker,
    decision_invoke_params: dict[str, Any],
) -> Feedback:
    """Route a built-in judge across TypeSafe-compatible and ordinary chat models.

    ``chat_invoker`` runs the judge as a plain chat completion; ``decision_invoke_params`` holds
    the System One inputs (``instructions``/``state`` and friends) that mirror the direct
    ``typesafe:/`` branch each judge already defines. Direct ``typesafe:/`` goes straight to
    System One. ``gateway:/`` prefers chat and falls back to System One only on the specific
    rejection (chat-first, so chat endpoints are unchanged). Every other model uses
    ``chat_invoker`` unchanged.
    """
    params = {"feedback_value_type": Literal["yes", "no"], **decision_invoke_params}
    if _is_typesafe_model(model_uri):
        return _invoke_typesafe_judge(model_uri, **params)
    if _is_gateway_model(model_uri):
        return _invoke_gateway_judge(model_uri, chat_invoker=chat_invoker, **params)
    return chat_invoker()


__all__ = [
    "_invoke_gateway_judge",
    "_invoke_structured_builtin_judge",
    "_is_gateway_model",
    "_is_gateway_system_one_rejection",
]
