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
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    _invoke_databricks_default_judge,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint_judge,
    _record_judge_model_usage_failure_databricks_telemetry,
    _record_judge_model_usage_success_databricks_telemetry,
)
from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.utils.parsing_utils import (
    _sanitize_justification,
    _strip_markdown_code_blocks,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.telemetry.events import InvokeCustomJudgeModelEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.telemetry.utils import _is_in_databricks
from mlflow.tracing.constant import AssessmentMetadataKey

_logger = logging.getLogger(__name__)


def _is_litellm_available() -> bool:
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


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
) -> Feedback:
    """
    Invoke the judge model.

    Routes to the appropriate implementation based on the model URI:
    - "databricks": Uses databricks.agents.evals library for default judge,
                    direct API for regular endpoints
    - LiteLLM-supported providers: Uses LiteLLM if available
    - Native providers: Falls back to AI Gateway adapters

    Args:
        model_uri: The model URI.
        prompt: The prompt to evaluate. Can be a string (single prompt) or
                a list of ChatMessage objects.
        assessment_name: The name of the assessment.
        trace: Optional trace object for context.
        num_retries: Number of retries on transient failures when using litellm.
        response_format: Optional Pydantic model class for structured output format.

    Returns:
        Feedback object with the judge's assessment.

    Raises:
        MlflowException: If the model cannot be invoked or dependencies are missing.
    """
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        return _invoke_databricks_default_judge(prompt, assessment_name)

    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.types.llm import ChatMessage

    model_provider, model_name = _parse_model_uri(model_uri)
    in_databricks = _is_in_databricks()

    # Handle Databricks endpoints (not the default judge) with proper telemetry
    if model_provider in {"databricks", "endpoints"} and isinstance(prompt, str):
        if model_provider == "endpoints":
            warnings.warn(
                "The legacy provider 'endpoints' is deprecated and will be removed in a future"
                "release. Please update your code to use the 'databricks' provider instead.",
                FutureWarning,
                stacklevel=2,
            )
        try:
            output = _invoke_databricks_serving_endpoint_judge(
                model_name=model_name,
                prompt=prompt,
                assessment_name=assessment_name,
                num_retries=num_retries,
                response_format=response_format,
            )
            feedback = output.feedback
            feedback.trace_id = trace.info.trace_id if trace is not None else None

            # Record success telemetry only when in Databricks
            if in_databricks:
                try:
                    _record_judge_model_usage_success_databricks_telemetry(
                        request_id=output.request_id,
                        model_provider=output.model_provider,
                        endpoint_name=output.model_name,
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

    # Handle all other cases (including non-Databricks, ChatMessage prompts, traces)
    messages = [ChatMessage(role="user", content=prompt)] if isinstance(prompt, str) else prompt
    total_cost = None
    if _is_litellm_available():
        response, total_cost = _invoke_litellm_and_handle_tools(
            provider=model_provider,
            model_name=model_name,
            messages=messages,
            trace=trace,
            num_retries=num_retries,
            response_format=response_format,
        )
    elif trace is not None:
        raise MlflowException(
            "LiteLLM is required for using traces with judges. "
            "Please install it with `pip install litellm`.",
            error_code=BAD_REQUEST,
        )
    else:
        if not isinstance(prompt, str):
            raise MlflowException(
                "This judge is not supported by native LLM providers. Please install "
                "LiteLLM with `pip install litellm` to use this judge.",
                error_code=BAD_REQUEST,
            )
        if response_format is not None:
            _logger.warning(
                "Structured output is not supported by native LLM providers. Please install "
                "LiteLLM with `pip install litellm` to use this judge.",
            )
        response = _invoke_via_gateway(model_uri, model_provider, prompt)

    cleaned_response = _strip_markdown_code_blocks(response)

    try:
        response_dict = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise MlflowException(
            f"Failed to parse response from judge model. Response: {response}",
            error_code=BAD_REQUEST,
        ) from e

    metadata = {AssessmentMetadataKey.JUDGE_COST: total_cost} if total_cost else None

    feedback = Feedback(
        name=assessment_name,
        value=response_dict["result"],
        rationale=_sanitize_justification(response_dict.get("rationale", "")),
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=model_uri),
        trace_id=trace.info.trace_id if trace is not None else None,
        metadata=metadata,
    )

    if "error" in response_dict:
        feedback.error = response_dict["error"]
        raise MlflowException(
            f"Judge evaluation failed with error: {response_dict['error']}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return feedback


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
