from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    import litellm

    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_GATEWAY_URI, MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.base_adapter import (
    AdapterInvocationInput,
    AdapterInvocationOutput,
    BaseJudgeAdapter,
)
from mlflow.genai.judges.utils.parsing_utils import (
    _sanitize_justification,
    _strip_markdown_code_blocks,
)
from mlflow.genai.judges.utils.telemetry_utils import (
    _record_judge_model_usage_failure_databricks_telemetry,
    _record_judge_model_usage_success_databricks_telemetry,
)
from mlflow.genai.judges.utils.tool_calling_utils import (
    _process_tool_calls,
    _raise_iteration_limit_exceeded,
)
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.tracking import get_tracking_uri
from mlflow.utils.uri import append_to_uri_path, is_http_uri

_logger = logging.getLogger(__name__)

# Global cache to track model capabilities across function calls
# Key: model URI (e.g., "openai/gpt-4"), Value: boolean indicating response_format support
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}


@dataclass
class InvokeLiteLLMOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None
    cost: float | None


class _SuppressLiteLLMNonfatalErrors(ContextDecorator):
    """
    Thread-safe context manager and decorator to suppress LiteLLM's "Give Feedback" and
    "Provider List" messages. These messages indicate nonfatal bugs in the LiteLLM library;
    they are often noisy and can be safely ignored.

    Uses reference counting to ensure suppression remains active while any thread is running,
    preventing race conditions in parallel execution.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.count = 0
        self.original_litellm_settings = {}

    def __enter__(self) -> "_SuppressLiteLLMNonfatalErrors":
        try:
            import litellm
        except ImportError:
            return self

        with self.lock:
            if self.count == 0:
                # First caller - store original settings and enable suppression
                self.original_litellm_settings = {
                    "set_verbose": getattr(litellm, "set_verbose", None),
                    "suppress_debug_info": getattr(litellm, "suppress_debug_info", None),
                }
                litellm.set_verbose = False
                litellm.suppress_debug_info = True
            self.count += 1

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any | None,
    ) -> bool:
        try:
            import litellm
        except ImportError:
            return False

        with self.lock:
            self.count -= 1
            if self.count == 0:
                # Last caller - restore original settings
                if (
                    original_verbose := self.original_litellm_settings.get("set_verbose")
                ) is not None:
                    litellm.set_verbose = original_verbose
                if (
                    original_suppress := self.original_litellm_settings.get("suppress_debug_info")
                ) is not None:
                    litellm.suppress_debug_info = original_suppress
                self.original_litellm_settings.clear()

        return False


# Global instance for use as threadsafe decorator
_suppress_litellm_nonfatal_errors = _SuppressLiteLLMNonfatalErrors()


def _invoke_litellm(
    litellm_model: str,
    messages: list["litellm.Message"],
    tools: list[dict[str, Any]],
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None,
    include_response_format: bool,
    inference_params: dict[str, Any] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> "litellm.ModelResponse":
    """
    Invoke litellm completion with retry support.

    Args:
        litellm_model: The LiteLLM model identifier
            (e.g., "openai/gpt-4" or endpoint name for gateway).
        messages: List of litellm Message objects.
        tools: List of tool definitions (empty list if no tools).
        num_retries: Number of retries with exponential backoff.
        response_format: Optional Pydantic model class for structured output.
        include_response_format: Whether to include response_format in the request.
        inference_params: Optional dictionary of additional inference parameters to pass
            to the model (e.g., temperature, top_p, max_tokens).
        api_base: Optional API base URL (used for gateway routing).
        api_key: Optional API key (used for gateway routing).

    Returns:
        The litellm ModelResponse object.

    Raises:
        Various litellm exceptions on failure.
    """
    import litellm

    kwargs = {
        "model": litellm_model,
        "messages": messages,
        "tools": tools or None,
        "tool_choice": "auto" if tools else None,
        "retry_policy": _get_litellm_retry_policy(num_retries),
        "retry_strategy": "exponential_backoff_retry",
        # In LiteLLM version 1.55.3+, max_retries is stacked on top of retry_policy.
        # To avoid double-retry, we set max_retries=0
        "max_retries": 0,
        # Drop any parameters that are known to be unsupported by the LLM.
        # This is important for compatibility with certain models that don't support
        # certain call parameters (e.g. GPT-4 doesn't support 'response_format')
        "drop_params": True,
    }

    if api_base is not None:
        kwargs["api_base"] = api_base
    if api_key is not None:
        kwargs["api_key"] = api_key

    if include_response_format:
        # LiteLLM supports passing Pydantic models directly for response_format
        kwargs["response_format"] = response_format or _get_default_judge_response_schema()

    # Apply any additional inference parameters (e.g., temperature, top_p, max_tokens)
    if inference_params:
        kwargs.update(inference_params)

    return litellm.completion(**kwargs)


@_suppress_litellm_nonfatal_errors
def _invoke_litellm_and_handle_tools(
    provider: str,
    model_name: str,
    messages: list["ChatMessage"],
    trace: Trace | None,
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None = None,
    inference_params: dict[str, Any] | None = None,
) -> InvokeLiteLLMOutput:
    """
    Invoke litellm with retry support and handle tool calling loop.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic', 'gateway').
        model_name: The model name (or endpoint name for gateway provider).
        messages: List of ChatMessage objects.
        trace: Optional trace object for context with tool calling support.
        num_retries: Number of retries with exponential backoff on transient failures.
        response_format: Optional Pydantic model class for structured output format.
                       Used by get_chat_completions_with_structured_output for
                       schema-based extraction.
        inference_params: Optional dictionary of additional inference parameters to pass
                       to the model (e.g., temperature, top_p, max_tokens).

    Returns:
        InvokeLiteLLMOutput containing:
        - response: The model's response content
        - request_id: The request ID for telemetry (if available)
        - num_prompt_tokens: Number of prompt tokens used (if available)
        - num_completion_tokens: Number of completion tokens used (if available)
        - cost: The total cost of the request (if available)

    Raises:
        MlflowException: If the request fails after all retries.
    """
    import litellm

    from mlflow.genai.judges.tools import list_judge_tools

    messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

    # Construct model URI and gateway params
    if provider == "gateway":
        # MLFLOW_GATEWAY_URI takes precedence over tracking URI for gateway routing.
        # This is needed for async job workers: the job infrastructure passes the HTTP
        # tracking URI (e.g., http://127.0.0.1:5000) to workers, but _get_tracking_store()
        # overwrites MLFLOW_TRACKING_URI with the backend store URI (e.g., sqlite://).
        # Job workers set MLFLOW_GATEWAY_URI to preserve the HTTP URI for gateway calls.
        tracking_uri = MLFLOW_GATEWAY_URI.get() or get_tracking_uri()

        # Validate that tracking URI is a valid HTTP(S) URL for gateway
        if not is_http_uri(tracking_uri):
            raise MlflowException(
                f"Gateway provider requires an HTTP(S) tracking URI, but got: '{tracking_uri}'. "
                "The gateway provider routes requests through the MLflow tracking server. "
                "Please set MLFLOW_TRACKING_URI to a valid HTTP(S) URL "
                "(e.g., 'http://localhost:5000' or 'https://your-mlflow-server.com')."
            )

        api_base = append_to_uri_path(tracking_uri, "gateway/mlflow/v1/")

        # Use openai/ prefix for LiteLLM to use OpenAI-compatible format.
        # LiteLLM strips the prefix, so gateway receives model_name as the endpoint.
        model = f"openai/{model_name}"
        # LiteLLM requires api_key to be set when using custom api_base, otherwise it
        # raises AuthenticationError looking for OPENAI_API_KEY env var. Gateway handles
        # auth in the server layer, so we pass a dummy value to satisfy LiteLLM.
        api_key = "mlflow-gateway-auth"
    else:
        model = f"{provider}/{model_name}"
        api_base = None
        api_key = None

    tools = []
    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [tool.get_definition().to_dict() for tool in judge_tools]

    def _prune_messages_for_context_window() -> list[litellm.Message] | None:
        if provider == "gateway":
            # For gateway provider, we don't know the underlying model,
            # so simply remove the oldest tool call pair.
            return _prune_messages_exceeding_context_window_length(messages)

        # For direct providers, use token-counting based pruning.
        try:
            max_context_length = litellm.get_model_info(model)["max_input_tokens"]
        except Exception:
            max_context_length = None

        return _prune_messages_exceeding_context_window_length(
            messages, model=model, max_tokens=max_context_length or 100000
        )

    include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(model, True)

    max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
    iteration_count = 0
    total_cost = None

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            _raise_iteration_limit_exceeded(max_iterations)
        try:
            try:
                response = _invoke_litellm(
                    litellm_model=model,
                    messages=messages,
                    tools=tools,
                    num_retries=num_retries,
                    response_format=response_format,
                    include_response_format=include_response_format,
                    inference_params=inference_params,
                    api_base=api_base,
                    api_key=api_key,
                )
            except (litellm.BadRequestError, litellm.UnsupportedParamsError) as e:
                error_str = str(e).lower()
                is_context_window_error = (
                    isinstance(e, litellm.ContextWindowExceededError)
                    or "context length" in error_str
                    or "too many tokens" in error_str
                )
                if is_context_window_error:
                    pruned = _prune_messages_for_context_window()
                    if pruned is None:
                        raise MlflowException(
                            "Context window exceeded and there are no tool calls to truncate. "
                            "The initial prompt may be too long for the model's context window."
                        ) from e
                    messages = pruned
                    continue
                # Check whether the request attempted to use structured outputs, rather than
                # checking whether the model supports structured outputs in the capabilities cache,
                # since the capabilities cache may have been updated between the time that
                # include_response_format was set and the request was made
                if include_response_format:
                    # Retry without response_format if the request failed due to unsupported params.
                    # Some models don't support structured outputs (response_format) at all,
                    # and some models don't support both tool calling and structured outputs.
                    _logger.debug(
                        f"Model {model} may not support structured outputs "
                        f"or combined tool calling + structured outputs. Error: {e}. "
                        f"Falling back to unstructured response.",
                        exc_info=True,
                    )
                    _MODEL_RESPONSE_FORMAT_CAPABILITIES[model] = False
                    include_response_format = False
                    continue
                else:
                    raise

            if cost := _extract_response_cost(response):
                if total_cost is None:
                    total_cost = 0
                total_cost += cost

            message = response.choices[0].message
            if not message.tool_calls:
                request_id = getattr(response, "id", None)
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                return InvokeLiteLLMOutput(
                    response=message.content,
                    request_id=request_id,
                    num_prompt_tokens=prompt_tokens,
                    num_completion_tokens=completion_tokens,
                    cost=total_cost,
                )

            messages.append(message)
            tool_response_messages = _process_tool_calls(tool_calls=message.tool_calls, trace=trace)
            messages.extend(tool_response_messages)

        except MlflowException:
            raise
        except Exception as e:
            error_message, error_code = _extract_litellm_error(e)
            raise MlflowException(
                f"Failed to invoke the judge via litellm: {error_message}",
                error_code=error_code,
            ) from e


def _extract_litellm_error(e: Exception) -> tuple[str, str]:
    """
    Extract the detail message and error code from an exception.

    Tries to parse structured error info from the exception message if it contains
    a gateway error in the format: {'detail': {'error_code': '...', 'message': '...'}}.
    Falls back to str(e) if parsing fails.

    Returns (message, error_code).
    """
    error_str = str(e)
    if match := re.search(r"\{'detail':\s*\{[^}]+\}\}", error_str):
        try:
            parsed = json.loads(match.group(0).replace("'", '"'))
            detail = parsed.get("detail", {})
            if isinstance(detail, dict):
                return detail.get("message", error_str), detail.get("error_code", INTERNAL_ERROR)
        except json.JSONDecodeError:
            pass

    return error_str, INTERNAL_ERROR


def _extract_response_cost(response: "litellm.Completion") -> float | None:
    if hidden_params := getattr(response, "_hidden_params", None):
        return hidden_params.get("response_cost")


def _remove_oldest_tool_call_pair(
    messages: list["litellm.Message"],
) -> list["litellm.Message"] | None:
    """
    Remove the oldest assistant message with tool calls and its corresponding tool responses.

    Args:
        messages: List of LiteLLM message objects.

    Returns:
        Modified messages with oldest tool call pair removed, or None if no tool calls to remove.
    """
    result = next(
        ((i, msg) for i, msg in enumerate(messages) if msg.role == "assistant" and msg.tool_calls),
        None,
    )
    if result is None:
        return None

    assistant_idx, assistant_msg = result
    modified = messages[:]
    modified.pop(assistant_idx)

    tool_call_ids = {tc.id if hasattr(tc, "id") else tc["id"] for tc in assistant_msg.tool_calls}
    return [
        msg for msg in modified if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
    ]


def _get_default_judge_response_schema() -> type[pydantic.BaseModel]:
    """
    Get the default Pydantic schema for judge evaluations.

    Returns:
        A Pydantic BaseModel class defining the standard judge output format.
    """
    # Import here to avoid circular imports
    from mlflow.genai.judges.base import Judge

    output_fields = Judge.get_output_fields()

    field_definitions = {}
    for field in output_fields:
        field_definitions[field.name] = (str, pydantic.Field(description=field.description))

    return pydantic.create_model("JudgeEvaluation", **field_definitions)


def _prune_messages_exceeding_context_window_length(
    messages: list["litellm.Message"],
    model: str | None = None,
    max_tokens: int | None = None,
) -> list["litellm.Message"] | None:
    """
    Prune messages from history to stay under token limit.

    When max_tokens is provided and model supports token counting, uses proactive
    token-counting based pruning. Otherwise, uses reactive truncation by removing
    a single tool call pair (useful when the underlying model is unknown).

    Args:
        messages: List of LiteLLM message objects.
        model: Model name for token counting. Required for token-based pruning.
        max_tokens: Maximum token limit. If None, removes the oldest tool call pair.

    Returns:
        Pruned list of LiteLLM message objects, or None if no tool calls to remove.
    """
    import litellm

    if max_tokens is None or model is None:
        return _remove_oldest_tool_call_pair(messages)

    initial_tokens = litellm.token_counter(model=model, messages=messages)
    if initial_tokens <= max_tokens:
        return messages

    pruned_messages = messages[:]
    # Remove tool call pairs until we're under limit
    while litellm.token_counter(model=model, messages=pruned_messages) > max_tokens:
        result = _remove_oldest_tool_call_pair(pruned_messages)
        if result is None:
            break
        pruned_messages = result

    final_tokens = litellm.token_counter(model=model, messages=pruned_messages)
    _logger.info(f"Pruned message history from {initial_tokens} to {final_tokens} tokens")
    return pruned_messages


def _get_litellm_retry_policy(num_retries: int) -> "litellm.RetryPolicy":
    """
    Get a LiteLLM retry policy for retrying requests when transient API errors occur.

    Args:
        num_retries: The number of times to retry a request if it fails transiently due to
                     network error, rate limiting, etc. Requests are retried with exponential
                     backoff.

    Returns:
        A LiteLLM RetryPolicy instance.
    """
    from litellm import RetryPolicy

    return RetryPolicy(
        TimeoutErrorRetries=num_retries,
        RateLimitErrorRetries=num_retries,
        InternalServerErrorRetries=num_retries,
        ContentPolicyViolationErrorRetries=num_retries,
        # We don't retry on errors that are unlikely to be transient
        # (e.g. bad request, invalid auth credentials)
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
    )


def _is_litellm_available() -> bool:
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


class LiteLLMAdapter(BaseJudgeAdapter):
    """Adapter for LiteLLM-supported providers."""

    @classmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        return _is_litellm_available()

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        from mlflow.types.llm import ChatMessage

        messages = (
            [ChatMessage(role="user", content=input_params.prompt)]
            if isinstance(input_params.prompt, str)
            else input_params.prompt
        )

        is_model_provider_databricks = input_params.model_provider in ("databricks", "endpoints")
        try:
            output = _invoke_litellm_and_handle_tools(
                provider=input_params.model_provider,
                model_name=input_params.model_name,
                messages=messages,
                trace=input_params.trace,
                num_retries=input_params.num_retries,
                response_format=input_params.response_format,
                inference_params=input_params.inference_params,
            )

            cleaned_response = _strip_markdown_code_blocks(output.response)

            try:
                response_dict = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                raise MlflowException(
                    f"Failed to parse response from judge model. Response: {output.response}"
                ) from e

            metadata = {AssessmentMetadataKey.JUDGE_COST: output.cost} if output.cost else None

            if "error" in response_dict:
                raise MlflowException(
                    f"Judge evaluation failed with error: {response_dict['error']}"
                )

            feedback = Feedback(
                name=input_params.assessment_name,
                value=response_dict["result"],
                rationale=_sanitize_justification(response_dict.get("rationale", "")),
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id=input_params.model_uri
                ),
                trace_id=input_params.trace.info.trace_id
                if input_params.trace is not None
                else None,
                metadata=metadata,
            )

            if is_model_provider_databricks:
                try:
                    _record_judge_model_usage_success_databricks_telemetry(
                        request_id=output.request_id,
                        model_provider=input_params.model_provider,
                        endpoint_name=input_params.model_name,
                        num_prompt_tokens=output.num_prompt_tokens,
                        num_completion_tokens=output.num_completion_tokens,
                    )
                except Exception:
                    _logger.debug("Failed to record judge model usage success telemetry")

            return AdapterInvocationOutput(feedback=feedback, cost=output.cost)

        except MlflowException as e:
            if is_model_provider_databricks:
                try:
                    _record_judge_model_usage_failure_databricks_telemetry(
                        model_provider=input_params.model_provider,
                        endpoint_name=input_params.model_name,
                        error_code=e.error_code or "UNKNOWN",
                        error_message=str(e),
                    )
                except Exception:
                    _logger.debug("Failed to record judge model usage failure telemetry")
            raise
