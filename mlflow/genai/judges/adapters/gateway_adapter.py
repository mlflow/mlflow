from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointType
from mlflow.genai.discovery.utils import _pydantic_to_response_format
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
# "gateway" routes requests through the MLflow tracking server's gateway proxy.
_NATIVE_PROVIDERS = ["openai", "anthropic", "gemini", "mistral", "endpoints", "gateway"]


def _invoke_via_gateway(
    model_uri: str,
    provider: str,
    prompt: str | list[dict[str, str]],
    inference_params: dict[str, Any] | None = None,
    response_format: type[pydantic.BaseModel] | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """
    Invoke the judge model via native AI Gateway adapters.

    Supports both string prompts (via ``score_model_on_payload``) and
    ChatMessage-style message lists (via the provider infrastructure).

    Args:
        model_uri: The full model URI.
        provider: The provider name.
        prompt: The prompt to evaluate. Either a string or a list of message dicts.
        inference_params: Optional dictionary of inference parameters to pass to the
            model (e.g., temperature, top_p, max_tokens).
        response_format: Optional Pydantic model class for structured output.
            Only used for ChatMessage-style prompts.

    Returns:
        The JSON response string from the model.

    Raises:
        MlflowException: If the provider is not natively supported or invocation fails.
    """
    from mlflow.metrics.genai.model_utils import (
        _call_llm_provider_api,
        _parse_model_uri,
        get_endpoint_type,
        score_model_on_payload,
    )

    if provider not in _NATIVE_PROVIDERS:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    if isinstance(prompt, str):
        return score_model_on_payload(
            model_uri=model_uri,
            payload=prompt,
            eval_parameters=inference_params,
            extra_headers=extra_headers,
            proxy_url=base_url,
            endpoint_type=get_endpoint_type(model_uri) or EndpointType.LLM_V1_CHAT,
        )

    _, model_name = _parse_model_uri(model_uri)

    # gateway:/ routes through the MLflow tracking server's gateway HTTP API
    if provider == "gateway":
        return _invoke_gateway_chat(model_name, prompt, inference_params, response_format)

    rf_dict = _pydantic_to_response_format(response_format) if response_format else None
    return _call_llm_provider_api(
        provider,
        model_name,
        messages=prompt,
        eval_parameters=inference_params,
        response_format=rf_dict,
    )


def _invoke_gateway_chat(
    endpoint_name: str,
    messages: list[dict[str, str]],
    inference_params: dict[str, Any] | None = None,
    response_format: type[pydantic.BaseModel] | None = None,
) -> str:
    """
    Invoke a gateway endpoint via the MLflow tracking server's chat completions API.

    Sends a POST request to ``/gateway/mlflow/v1/chat/completions`` on the tracking
    server, using the OpenAI-compatible format where the endpoint name is passed as
    the ``model`` field in the request body.

    Args:
        endpoint_name: The gateway endpoint name (e.g., "my-chat" from "gateway:/my-chat").
        messages: List of message dicts with "role" and "content" keys.
        inference_params: Optional inference parameters (temperature, etc.).
        response_format: Optional Pydantic model for structured output.

    Returns:
        The model's response content as a string.
    """
    import requests

    from mlflow.environment_variables import MLFLOW_GATEWAY_URI
    from mlflow.tracking import get_tracking_uri
    from mlflow.utils.uri import is_http_uri

    gateway_uri = MLFLOW_GATEWAY_URI.get() or get_tracking_uri()
    if not is_http_uri(gateway_uri):
        raise MlflowException(
            f"Gateway provider requires an HTTP(S) tracking URI, but got: '{gateway_uri}'. "
            "Please set MLFLOW_TRACKING_URI to a valid HTTP(S) URL "
            "(e.g., 'http://localhost:5000').",
            error_code=BAD_REQUEST,
        )
    url = f"{gateway_uri.rstrip('/')}/gateway/mlflow/v1/chat/completions"

    payload: dict[str, Any] = {"model": endpoint_name, "messages": messages}
    if inference_params:
        payload.update(inference_params)
    if response_format is not None:
        rf_dict = _pydantic_to_response_format(response_format)
        if rf_dict is not None:
            payload["response_format"] = rf_dict

    try:
        resp = requests.post(url=url, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        body = getattr(e.response, "text", "")
        raise MlflowException(
            f"Failed to call gateway endpoint '{endpoint_name}' at {url}.\n"
            f"- Error: {e}\n- Response body: {body}",
            error_code=BAD_REQUEST,
        ) from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise MlflowException(
            f"Unexpected response format from gateway endpoint '{endpoint_name}': {data}",
            error_code=BAD_REQUEST,
        ) from e


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
        if model_provider not in _NATIVE_PROVIDERS:
            return False
        # "endpoints" (Databricks model serving) only supports string prompts
        # via score_model_on_payload; _get_provider_instance doesn't handle it.
        if isinstance(prompt, list) and model_provider == "endpoints":
            return False
        return True

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        if input_params.trace is not None:
            raise MlflowException(
                "LiteLLM is required for using traces with judges. "
                "Please install it with `pip install litellm`.",
            )

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

        # gateway:/ routes through the MLflow tracking server; base_url and extra_headers
        # should be configured on the server, not per-request.
        if input_params.model_provider == "gateway" and (
            input_params.base_url is not None or input_params.extra_headers is not None
        ):
            raise MlflowException(
                "base_url and extra_headers are not supported for the gateway provider "
                "(gateway:/...). Configure the MLflow AI Gateway on the tracking server instead.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if isinstance(input_params.prompt, str):
            prompt = input_params.prompt
        else:
            prompt = [{"role": msg.role, "content": msg.content} for msg in input_params.prompt]

        response = _invoke_via_gateway(
            input_params.model_uri,
            input_params.model_provider,
            prompt,
            inference_params=input_params.inference_params,
            response_format=input_params.response_format,
            base_url=input_params.base_url,
            extra_headers=input_params.extra_headers,
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
