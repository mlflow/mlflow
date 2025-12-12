from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
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
from mlflow.protos.databricks_pb2 import BAD_REQUEST

# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]


def _invoke_via_gateway(
    model_uri: str,
    provider: str,
    prompt: str,
    inference_params: dict[str, Any] | None = None,
) -> str:
    """
    Invoke the judge model via native AI Gateway adapters.

    Args:
        model_uri: The full model URI.
        provider: The provider name.
        prompt: The prompt to evaluate.
        inference_params: Optional dictionary of inference parameters to pass to the
            model (e.g., temperature, top_p, max_tokens).

    Returns:
        The JSON response string from the model.

    Raises:
        MlflowException: If the provider is not natively supported or invocation fails.
    """
    from mlflow.metrics.genai.model_utils import get_endpoint_type, score_model_on_payload

    if provider not in _NATIVE_PROVIDERS:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    return score_model_on_payload(
        model_uri=model_uri,
        payload=prompt,
        eval_parameters=inference_params,
        endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
    )


class GatewayAdapter(BaseJudgeAdapter):
    """Adapter for native AI Gateway providers (fallback when LiteLLM is not available)."""

    @classmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        model_provider, _ = _parse_model_uri(model_uri)
        return model_provider in _NATIVE_PROVIDERS and isinstance(prompt, str)

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        if input_params.trace is not None:
            raise MlflowException(
                "LiteLLM is required for using traces with judges. "
                "Please install it with `pip install litellm`.",
            )

        # Validate structured output support
        if input_params.response_format is not None:
            raise MlflowException(
                "Structured output is not supported by native LLM providers. "
                "Please install LiteLLM with `pip install litellm` to use structured output.",
            )

        response = _invoke_via_gateway(
            input_params.model_uri,
            input_params.model_provider,
            input_params.prompt,
            input_params.inference_params,
        )

        cleaned_response = _strip_markdown_code_blocks(response)

        try:
            response_dict = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse response from judge model. Response: {response}",
                error_code=BAD_REQUEST,
            ) from e

        feedback = Feedback(
            name=input_params.assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=input_params.model_uri
            ),
        )

        return AdapterInvocationOutput(feedback=feedback)
