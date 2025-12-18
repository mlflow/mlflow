"""Main invocation utilities for judge models."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges.adapters.base_adapter import AdapterInvocationInput
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
from mlflow.genai.judges.adapters.utils import get_adapter
from mlflow.genai.judges.utils.parsing_utils import _strip_markdown_code_blocks
from mlflow.telemetry.events import InvokeCustomJudgeModelEvent
from mlflow.telemetry.track import record_usage_event

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
    inference_params: dict[str, Any] | None = None,
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
        inference_params: Optional dictionary of inference parameters to pass to the
            model (e.g., temperature, top_p, max_tokens). These parameters allow
            fine-grained control over the model's behavior during evaluation.

    Returns:
        Feedback object with the judge's assessment.

    Raises:
        MlflowException: If the model cannot be invoked or dependencies are missing.
    """
    adapter = get_adapter(model_uri=model_uri, prompt=prompt)

    input_params = AdapterInvocationInput(
        model_uri=model_uri,
        prompt=prompt,
        assessment_name=assessment_name,
        trace=trace,
        num_retries=num_retries,
        response_format=response_format,
        use_case=use_case,
        inference_params=inference_params,
    )

    output = adapter.invoke(input_params)
    return output.feedback


def get_chat_completions_with_structured_output(
    model_uri: str,
    messages: list["ChatMessage"],
    output_schema: type[pydantic.BaseModel],
    trace: Trace | None = None,
    num_retries: int = 10,
    inference_params: dict[str, Any] | None = None,
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
        inference_params: Optional dictionary of inference parameters to pass to the
                       model (e.g., temperature, top_p, max_tokens).

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
        inference_params=inference_params,
    )

    cleaned_response = _strip_markdown_code_blocks(response)
    response_dict = json.loads(cleaned_response)
    return output_schema(**response_dict)
