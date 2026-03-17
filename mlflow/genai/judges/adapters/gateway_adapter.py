from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pydantic

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
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE

# "endpoints" is a special case for MLflow deployment endpoints (e.g. Databricks model serving).
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]


def _invoke_via_gateway(
    model_uri: str,
    provider: str,
    prompt: str,
    inference_params: dict[str, Any] | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """
    Invoke the judge model via native AI Gateway adapters.

    Args:
        model_uri: The full model URI.
        provider: The provider name.
        prompt: The prompt to evaluate.
        inference_params: Optional dictionary of inference parameters to pass to the
            model (e.g., temperature, top_p, max_tokens).
        base_url: Optional base URL to route requests through.
        extra_headers: Optional dictionary of additional HTTP headers to include
            in requests to the LLM provider.

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
        extra_headers=extra_headers,
        proxy_url=base_url,
        endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
    )


def _invoke_chat_via_gateway(
    model_uri: str,
    messages: list[dict[str, str]],
    response_format: type[pydantic.BaseModel] | None = None,
    inference_params: dict[str, Any] | None = None,
) -> str:
    """Invoke the judge model via the gateway provider infrastructure with ChatMessage support."""
    from mlflow.gateway.config import Provider
    from mlflow.genai.discovery.utils import _pydantic_to_response_format
    from mlflow.metrics.genai.model_utils import (
        _get_provider_instance,
        _parse_model_uri,
        _send_request,
    )

    provider_name, model_name = _parse_model_uri(model_uri)
    provider = _get_provider_instance(provider_name, model_name)

    payload: dict[str, Any] = {"messages": messages}
    if inference_params:
        payload.update(inference_params)
    if response_format is not None:
        payload["response_format"] = _pydantic_to_response_format(response_format)

    chat_payload = provider.adapter_class.chat_to_model(payload, provider.config)

    if provider_name in (Provider.AMAZON_BEDROCK, Provider.BEDROCK):
        raw_response = provider._request(chat_payload)
    else:
        raw_response = _send_request(
            endpoint=provider.get_endpoint_url("llm/v1/chat"),
            headers=provider.headers,
            payload=chat_payload,
        )

    response = provider.adapter_class.model_to_chat(raw_response, provider.config)
    return response.choices[0].message.content or ""


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
        return model_provider in _NATIVE_PROVIDERS

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        if input_params.trace is not None:
            raise MlflowException(
                "LiteLLM is required for using traces with judges. "
                "Please install it with `pip install litellm`.",
            )

        if isinstance(input_params.prompt, str):
            # base_url and extra_headers are not supported for deployment endpoints
            if input_params.model_provider == "endpoints" and (
                input_params.base_url is not None or input_params.extra_headers is not None
            ):
                raise MlflowException(
                    "base_url and extra_headers are not supported for deployment "
                    "endpoints (endpoints:/...). The endpoint URL is determined by the "
                    "deployment target configuration.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            response = _invoke_via_gateway(
                input_params.model_uri,
                input_params.model_provider,
                input_params.prompt,
                input_params.inference_params,
                input_params.base_url,
                input_params.extra_headers,
            )
        else:
            messages = [{"role": msg.role, "content": msg.content} for msg in input_params.prompt]
            response = _invoke_chat_via_gateway(
                input_params.model_uri,
                messages,
                response_format=input_params.response_format,
                inference_params=input_params.inference_params,
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
