from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
    create_litellm_message_from_databricks_response,
    serialize_messages_to_databricks_prompts,
)
from mlflow.genai.judges.constants import (
    _DATABRICKS_AGENTIC_JUDGE_MODEL,
    _DATABRICKS_DEFAULT_JUDGE_MODEL,
)
from mlflow.genai.utils.gateway_utils import get_gateway_litellm_config
from mlflow.tracking import get_tracking_uri
from mlflow.utils.uri import is_databricks_uri

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

_logger = logging.getLogger(__name__)

_DEFAULT_SIMULATION_MODEL = "openai:/gpt-5"


def get_default_simulation_model() -> str:
    if is_databricks_uri(get_tracking_uri()):
        return _DATABRICKS_AGENTIC_JUDGE_MODEL
    return _DEFAULT_SIMULATION_MODEL


@contextmanager
def delete_trace_if_created():
    """Delete at most one trace created within this context to avoid polluting user traces."""
    trace_id_before = mlflow.get_last_active_trace_id(thread_local=True)
    try:
        yield
    finally:
        trace_id_after = mlflow.get_last_active_trace_id(thread_local=True)
        if trace_id_after and trace_id_after != trace_id_before:
            try:
                mlflow.delete_trace(trace_id_after)
            except Exception as e:
                _logger.debug(f"Failed to delete trace {trace_id_after}: {e}")


def invoke_model_without_tracing(
    model_uri: str,
    messages: list[ChatMessage],
    num_retries: int = 3,
    inference_params: dict[str, Any] | None = None,
    response_format: type | None = None,
) -> str:
    """
    Invoke a model without tracing. This method will delete the last trace created by the
    invocation, if any.
    """
    import litellm

    from mlflow.metrics.genai.model_utils import _parse_model_uri

    with delete_trace_if_created():
        if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
            user_prompt, system_prompt = serialize_messages_to_databricks_prompts(messages)

            result = call_chat_completions(
                user_prompt=user_prompt,
                system_prompt=system_prompt or ".",
                model=_DATABRICKS_AGENTIC_JUDGE_MODEL,
            )
            if getattr(result, "error_code", None):
                raise MlflowException(
                    f"Failed to get chat completions result from Databricks managed endpoint: "
                    f"[{result.error_code}] {result.error_message}"
                )

            output_json = result.output_json
            if not output_json:
                raise MlflowException("Empty response from Databricks managed endpoint")

            parsed_json = json.loads(output_json) if isinstance(output_json, str) else output_json
            return create_litellm_message_from_databricks_response(parsed_json).content

        provider, model_name = _parse_model_uri(model_uri)

        litellm_messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

        kwargs = {
            "messages": litellm_messages,
            "max_retries": num_retries,
            "drop_params": True,
        }

        if provider == "gateway":
            config = get_gateway_litellm_config(model_name)
            kwargs["api_base"] = config.api_base
            kwargs["api_key"] = config.api_key
            kwargs["model"] = config.model
        else:
            kwargs["model"] = f"{provider}/{model_name}"
        if inference_params:
            kwargs.update(inference_params)
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if inference_params and "Unsupported value: 'temperature'" in error_str:
                kwargs.pop("temperature", None)
                response = litellm.completion(**kwargs)
                return response.choices[0].message.content
            else:
                raise


def format_history(history: list[dict[str, Any]]) -> str | None:
    if not history:
        return None
    formatted = []
    for msg in history:
        role = msg.get("role") or "unknown"
        content = msg.get("content") or ""
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)
