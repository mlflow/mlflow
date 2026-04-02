"""Shared utilities for judge adapters."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from mlflow.genai.judges.adapters.base_adapter import BaseJudgeAdapter
    from mlflow.types.llm import ChatMessage

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CLIENT_QUERY_RETRY_CODES,
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR

# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------


def get_adapter(
    model_uri: str,
    prompt: str | list["ChatMessage"],
) -> "BaseJudgeAdapter":
    """
    Factory function to get the appropriate adapter for a given model configuration. Tries adapters
    in order of priority.

    Args:
        model_uri: The full model URI (e.g., "openai:/gpt-4", "databricks").
        prompt: The prompt to evaluate (string or list of ChatMessages).

    Returns:
        An instance of the appropriate adapter.

    Raises:
        MlflowException: If no suitable adapter is found.
    """
    # Lazy imports to avoid circular imports and heavyweight dependency chains.
    # Importing mlflow.metrics.genai.model_utils triggers mlflow.metrics.__init__
    # → mlflow.metrics.genai → genai_metric → pandas, breaking the skinny client.
    from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
        DatabricksManagedJudgeAdapter,
    )
    from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
    from mlflow.genai.judges.adapters.litellm_adapter import LiteLLMAdapter

    adapters = [
        DatabricksManagedJudgeAdapter,
        GatewayAdapter,
        LiteLLMAdapter,
    ]

    for adapter_class in adapters:
        if adapter_class.is_applicable(model_uri=model_uri, prompt=prompt):
            return adapter_class()

    raise MlflowException(
        f"No suitable adapter found for model_uri='{model_uri}'.",
        error_code=BAD_REQUEST,
    )


# ---------------------------------------------------------------------------
# Shared HTTP / error handling
# ---------------------------------------------------------------------------


class ChatCompletionError(Exception):
    def __init__(self, status_code: int, message: str, is_context_window_error: bool = False):
        self.status_code = status_code
        self.message = message
        self.is_context_window_error = is_context_window_error
        super().__init__(message)


def send_chat_request(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    num_retries: int,
) -> dict[str, Any]:
    """Send a chat completions request with retry logic."""
    last_exception = None
    for attempt in range(1 + num_retries):
        try:
            resp = requests.post(
                url=endpoint,
                headers={"Content-Type": "application/json", **headers},
                json=payload,
                timeout=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
            )

            if resp.status_code == 400:
                # Don't retry 400s — raise immediately for caller to handle
                error_body = _safe_parse_error(resp)
                raise ChatCompletionError(
                    status_code=400,
                    message=error_body,
                    is_context_window_error=is_context_window_error(error_body),
                )

            if resp.status_code in MLFLOW_GATEWAY_CLIENT_QUERY_RETRY_CODES:
                last_exception = ChatCompletionError(
                    status_code=resp.status_code,
                    message=_safe_parse_error(resp),
                )
                if attempt < num_retries:
                    _sleep_with_backoff(attempt)
                    continue
                raise last_exception

            if resp.status_code >= 400:
                raise ChatCompletionError(
                    status_code=resp.status_code,
                    message=_safe_parse_error(resp),
                )

            return resp.json()

        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < num_retries:
                _sleep_with_backoff(attempt)
                continue
            raise MlflowException(
                f"Request to {endpoint} timed out after {MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS}s",
                error_code=INTERNAL_ERROR,
            ) from e
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            if attempt < num_retries:
                _sleep_with_backoff(attempt)
                continue
            raise MlflowException(
                f"Failed to connect to {endpoint}: {e}",
                error_code=INTERNAL_ERROR,
            ) from e

    raise last_exception


def _safe_parse_error(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if "error" in body:
            error = body["error"]
            if isinstance(error, dict):
                return error.get("message", resp.text)
            return str(error)
    except Exception:
        pass
    return resp.text


def is_context_window_error(error_message: str) -> bool:
    lower = error_message.lower()
    return "context length" in lower or "too many tokens" in lower or "maximum context" in lower


def is_response_format_error(error_message: str) -> bool:
    lower = error_message.lower()
    return (
        "response_format" in lower
        or "json_schema" in lower
        or "structured output" in lower
        or "output_config" in lower
    )


def _sleep_with_backoff(attempt: int) -> None:
    delay = min(2**attempt, 30)
    time.sleep(delay)
