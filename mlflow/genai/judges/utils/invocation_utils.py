"""Main invocation utilities for judge models."""

from __future__ import annotations

import json
import logging
import traceback
import warnings
from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _record_judge_model_usage_failure_databricks_telemetry,
    _record_judge_model_usage_success_databricks_telemetry,
)
from mlflow.genai.judges.adapters.base_adapter import AdapterInvocationInput, get_adapter
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
from mlflow.genai.judges.utils.parsing_utils import _strip_markdown_code_blocks
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.telemetry.events import InvokeCustomJudgeModelEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.telemetry.utils import _is_in_databricks

_logger = logging.getLogger(__name__)


class FieldExtraction(pydantic.BaseModel):
    """Schema for extracting inputs and outputs from traces using LLM."""

    inputs: str = pydantic.Field(description="The user's original request or question")
    outputs: str = pydantic.Field(description="The system's final response")


@record_usage_event(InvokeCustomJudgeModelEvent)
def invoke_judge_model(
    model_uri: str,
    prompt: str | list["ChatMessage"],
    assessment_name: str,
    trace: Trace | None = None,
    num_retries: int = 10,
    response_format: type[pydantic.BaseModel] | None = None,
    use_case: str | None = None,
) -> Feedback:
    """
    Invoke the judge model.

    Routes to the appropriate adapter based on the model URI and configuration.
    Uses a factory pattern to select the correct adapter:
    - DatabricksManagedJudgeAdapter: For the default Databricks judge
    - DatabricksServingEndpointAdapter: For Databricks serving endpoints
    - LiteLLMAdapter: For LiteLLM-supported providers
    - GatewayAdapter: Fallback for native providers

    Args:
        model_uri: The model URI.
        prompt: The prompt to evaluate. Can be a string (single prompt) or
                a list of ChatMessage objects.
        assessment_name: The name of the assessment.
        trace: Optional trace object for context.
        num_retries: Number of retries on transient failures when using litellm.
        response_format: Optional Pydantic model class for structured output format.
        use_case: The use case for the chat completion. Only applicable when using the
            Databricks default judge and only used if supported by the installed
            databricks-agents version.

    Returns:
        Feedback object with the judge's assessment.

    Raises:
        MlflowException: If the model cannot be invoked or dependencies are missing.
    """
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        return _invoke_databricks_default_judge(prompt, assessment_name, use_case=use_case)

    in_databricks = _is_in_databricks()

    # Get the appropriate adapter (it will parse model_uri if needed)
    adapter = get_adapter(model_uri=model_uri, prompt=prompt)

    # Create input parameters
    input_params = AdapterInvocationInput(
        model_uri=model_uri,
        prompt=prompt,
        assessment_name=assessment_name,
        trace=trace,
        num_retries=num_retries,
        response_format=response_format,
    )

    # Parse model URI for telemetry and deprecation warning (only when needed)
    model_provider, model_name = _parse_model_uri(model_uri)

    # Show deprecation warning for legacy 'endpoints' provider
    if model_provider == "endpoints":
        warnings.warn(
            "The legacy provider 'endpoints' is deprecated and will be removed in a future "
            "release. Please update your code to use the 'databricks' provider instead.",
            FutureWarning,
            stacklevel=2,
        )

    # Invoke the adapter with telemetry for Databricks endpoints
    if model_provider in {"databricks", "endpoints"} and isinstance(prompt, str):
        try:
            output = adapter.invoke(input_params)
            feedback = output.feedback
            feedback.trace_id = trace.info.trace_id if trace is not None else None

            # Record success telemetry only when in Databricks
            if in_databricks:
                try:
                    provider = "databricks" if model_provider == "endpoints" else model_provider
                    _record_judge_model_usage_success_databricks_telemetry(
                        request_id=output.request_id,
                        model_provider=provider,
                        endpoint_name=model_name,
                        num_prompt_tokens=output.num_prompt_tokens,
                        num_completion_tokens=output.num_completion_tokens,
                    )
                except Exception as telemetry_error:
                    _logger.debug(
                        "Failed to record judge model usage success telemetry. Error: %s",
                        telemetry_error,
                        exc_info=True,
                    )

            return feedback

        except Exception:
            # Record failure telemetry only when in Databricks
            if in_databricks:
                try:
                    provider = "databricks" if model_provider == "endpoints" else model_provider
                    _record_judge_model_usage_failure_databricks_telemetry(
                        model_provider=provider,
                        endpoint_name=model_name,
                        error_code="UNKNOWN",
                        error_message=traceback.format_exc(),
                    )
                except Exception as telemetry_error:
                    _logger.debug(
                        "Failed to record judge model usage failure telemetry. Error: %s",
                        telemetry_error,
                        exc_info=True,
                    )
            raise

    # For all other cases, invoke the adapter directly
    output = adapter.invoke(input_params)
    return output.feedback


def get_chat_completions_with_structured_output(
    model_uri: str,
    messages: list["ChatMessage"],
    output_schema: type[pydantic.BaseModel],
    trace: Trace | None = None,
    num_retries: int = 10,
) -> pydantic.BaseModel:
    """
    Get chat completions from an LLM with structured output conforming to a Pydantic schema.

    This function invokes an LLM and ensures the response matches the provided Pydantic schema.
    When a trace is provided, the LLM can use tool calling to examine trace spans.

    Args:
        model_uri: The model URI (e.g., "openai:/gpt-4", "anthropic:/claude-3").
        messages: List of ChatMessage objects for the conversation with the LLM.
        output_schema: Pydantic model class defining the expected output structure.
                       The LLM will be instructed to return data matching this schema.
        trace: Optional trace object for context. When provided, enables tool
               calling to examine trace spans.
        num_retries: Number of retries on transient failures. Defaults to 10 with
                     exponential backoff.

    Returns:
        Instance of output_schema with the structured data from the LLM.

    Raises:
        ImportError: If LiteLLM is not installed.
        JSONDecodeError: If the LLM response cannot be parsed as JSON.
        ValidationError: If the LLM response does not match the output schema.

    Example:
        .. code-block:: python

            from pydantic import BaseModel, Field
            from mlflow.genai.judges.utils import get_chat_completions_with_structured_output
            from mlflow.types.llm import ChatMessage


            class FieldExtraction(BaseModel):
                inputs: str = Field(description="The user's original request")
                outputs: str = Field(description="The system's final response")


            # Extract fields from a trace where root span lacks input/output
            # but nested spans contain the actual data
            result = get_chat_completions_with_structured_output(
                model_uri="openai:/gpt-4",
                messages=[
                    ChatMessage(role="system", content="Extract fields from the trace"),
                    ChatMessage(role="user", content="Find the inputs and outputs"),
                ],
                output_schema=FieldExtraction,
                trace=trace,  # Trace with nested spans containing actual data
            )
            print(result.inputs)  # Extracted from inner span
            print(result.outputs)  # Extracted from inner span
    """
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    model_provider, model_name = _parse_model_uri(model_uri)

    # TODO: The cost measurement is discarded here from the parsing of the
    # tool handling response. We should eventually pass this cost estimation through
    # so that the total cost of the usage of the scorer incorporates tool call usage.
    # Deferring for initial implementation due to complexity.
    response, _ = _invoke_litellm_and_handle_tools(
        provider=model_provider,
        model_name=model_name,
        messages=messages,
        trace=trace,
        num_retries=num_retries,
        response_format=output_schema,
    )

    cleaned_response = _strip_markdown_code_blocks(response)
    response_dict = json.loads(cleaned_response)
    return output_schema(**response_dict)
