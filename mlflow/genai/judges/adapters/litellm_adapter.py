from __future__ import annotations

import json
import logging
import threading
from contextlib import ContextDecorator
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    import litellm

    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
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
from mlflow.genai.judges.utils.tool_calling_utils import _process_tool_calls
from mlflow.protos.databricks_pb2 import REQUEST_LIMIT_EXCEEDED
from mlflow.tracing.constant import AssessmentMetadataKey

_logger = logging.getLogger(__name__)

# Global cache to track model capabilities across function calls
# Key: model URI (e.g., "openai/gpt-4"), Value: boolean indicating response_format support
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}


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
    litellm_model_uri: str,
    messages: list["litellm.Message"],
    tools: list[dict[str, Any]],
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None,
    include_response_format: bool,
    inference_params: dict[str, Any] | None = None,
) -> "litellm.ModelResponse":
    """
    Invoke litellm completion with retry support.

    Args:
        litellm_model_uri: Full model URI for litellm (e.g., "openai/gpt-4").
        messages: List of litellm Message objects.
        tools: List of tool definitions (empty list if no tools).
        num_retries: Number of retries with exponential backoff.
        response_format: Optional Pydantic model class for structured output.
        include_response_format: Whether to include response_format in the request.
        inference_params: Optional dictionary of additional inference parameters to pass
            to the model (e.g., temperature, top_p, max_tokens).

    Returns:
        The litellm ModelResponse object.

    Raises:
        Various litellm exceptions on failure.
    """
    import litellm

    kwargs = {
        "model": litellm_model_uri,
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
) -> tuple[str, float | None]:
    """
    Invoke litellm with retry support and handle tool calling loop.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_name: The model name.
        messages: List of ChatMessage objects.
        trace: Optional trace object for context with tool calling support.
        num_retries: Number of retries with exponential backoff on transient failures.
        response_format: Optional Pydantic model class for structured output format.
                       Used by get_chat_completions_with_structured_output for
                       schema-based extraction.
        inference_params: Optional dictionary of additional inference parameters to pass
                       to the model (e.g., temperature, top_p, max_tokens).

    Returns:
        Tuple of the model's response content and the total cost.

    Raises:
        MlflowException: If the request fails after all retries.
    """
    import litellm

    from mlflow.genai.judges.tools import list_judge_tools

    messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

    litellm_model_uri = f"{provider}/{model_name}"
    tools = []

    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [tool.get_definition().to_dict() for tool in judge_tools]

    def _prune_messages_for_context_window():
        try:
            max_context_length = litellm.get_max_tokens(litellm_model_uri)
        except Exception:
            max_context_length = None

        return _prune_messages_exceeding_context_window_length(
            messages=messages,
            model=litellm_model_uri,
            max_tokens=max_context_length or 100000,
        )

    include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(litellm_model_uri, True)

    max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
    iteration_count = 0
    total_cost = None

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            raise MlflowException(
                f"Completion iteration limit of {max_iterations} exceeded. "
                f"This usually indicates the model is not powerful enough to effectively "
                f"analyze the trace. Consider using a more intelligent/powerful model. "
                f"In rare cases, for very complex traces where a large number of completion "
                f"iterations might be required, you can increase the number of iterations by "
                f"modifying the {MLFLOW_JUDGE_MAX_ITERATIONS.name} environment variable.",
                error_code=REQUEST_LIMIT_EXCEEDED,
            )
        try:
            try:
                response = _invoke_litellm(
                    litellm_model_uri=litellm_model_uri,
                    messages=messages,
                    tools=tools,
                    num_retries=num_retries,
                    response_format=response_format,
                    include_response_format=include_response_format,
                    inference_params=inference_params,
                )
            except (litellm.BadRequestError, litellm.UnsupportedParamsError) as e:
                if isinstance(e, litellm.ContextWindowExceededError) or "context length" in str(e):
                    messages = _prune_messages_for_context_window()
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
                        f"Model {litellm_model_uri} may not support structured outputs or combined "
                        f"tool calling + structured outputs. Error: {e}. "
                        f"Falling back to unstructured response.",
                        exc_info=True,
                    )
                    _MODEL_RESPONSE_FORMAT_CAPABILITIES[litellm_model_uri] = False
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
                return message.content, total_cost

            messages.append(message)
            tool_response_messages = _process_tool_calls(tool_calls=message.tool_calls, trace=trace)
            messages.extend(tool_response_messages)

        except MlflowException:
            raise
        except Exception as e:
            raise MlflowException(f"Failed to invoke the judge via litellm: {e}") from e


def _extract_response_cost(response: "litellm.Completion") -> float | None:
    if hidden_params := getattr(response, "_hidden_params", None):
        return hidden_params.get("response_cost")


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
    model: str,
    max_tokens: int,
) -> list["litellm.Message"]:
    """
    Prune messages from history to stay under token limit.

    Args:
        messages: List of LiteLLM message objects.
        model: Model name for token counting.
        max_tokens: Maximum token limit.

    Returns:
        Pruned list of LiteLLM message objects under the token limit
    """
    import litellm

    initial_tokens = litellm.token_counter(model=model, messages=messages)
    if initial_tokens <= max_tokens:
        return messages

    pruned_messages = messages[:]
    # Remove tool call pairs until we're under limit
    while litellm.token_counter(model=model, messages=pruned_messages) > max_tokens:
        # Find first assistant message with tool calls
        result = next(
            (
                (i, msg)
                for i, msg in enumerate(pruned_messages)
                if msg.role == "assistant" and msg.tool_calls
            ),
            None,
        )
        if result is None:
            break  # No more tool calls to remove
        assistant_idx, assistant_msg = result
        pruned_messages.pop(assistant_idx)
        # Remove corresponding tool response messages
        tool_call_ids = {
            tc.id if hasattr(tc, "id") else tc["id"] for tc in assistant_msg.tool_calls
        }
        pruned_messages = [
            msg
            for msg in pruned_messages
            if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
        ]

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

        response, total_cost = _invoke_litellm_and_handle_tools(
            provider=input_params.model_provider,
            model_name=input_params.model_name,
            messages=messages,
            trace=input_params.trace,
            num_retries=input_params.num_retries,
            response_format=input_params.response_format,
            inference_params=input_params.inference_params,
        )

        cleaned_response = _strip_markdown_code_blocks(response)

        try:
            response_dict = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse response from judge model. Response: {response}"
            ) from e

        metadata = {AssessmentMetadataKey.JUDGE_COST: total_cost} if total_cost else None

        if "error" in response_dict:
            raise MlflowException(f"Judge evaluation failed with error: {response_dict['error']}")

        feedback = Feedback(
            name=input_params.assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=input_params.model_uri
            ),
            trace_id=input_params.trace.info.trace_id if input_params.trace is not None else None,
            metadata=metadata,
        )

        return AdapterInvocationOutput(feedback=feedback, cost=total_cost)
