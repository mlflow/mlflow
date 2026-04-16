import logging
import os
from typing import TYPE_CHECKING, Any

import requests
from pydantic import BaseModel

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.openai import OpenAIConfig, OpenAIProvider
from mlflow.genai.utils.gateway_utils import get_gateway_config
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.provider_filter import normalize_provider_name
from mlflow.utils.providers import (
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
)

if TYPE_CHECKING:
    from mlflow.gateway.providers import BaseProvider

_logger = logging.getLogger(__name__)


def get_endpoint_type(endpoint_uri: str) -> str | None:
    """
    Get the type of the endpoint if it is MLflow deployment
    endpoint. For other endpoints e.g. OpenAI, or if the
    endpoint does not specify type, return None.
    """
    schema, path = _parse_model_uri(endpoint_uri)

    if schema != "endpoints":
        return None

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client()

    endpoint = client.get_endpoint(path)
    # TODO: Standardize the return type of `get_endpoint` and remove this check
    endpoint = endpoint.dict() if isinstance(endpoint, BaseModel) else endpoint
    return endpoint.get("task", endpoint.get("endpoint_type"))


# TODO: improve this name
def score_model_on_payload(
    model_uri: str,
    payload: str,
    eval_parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    endpoint_type: str | None = None,
):
    """Call the model identified by the given uri with the given string prompt."""
    eval_parameters = eval_parameters or {}
    extra_headers = extra_headers or {}

    prefix, suffix = _parse_model_uri(model_uri)

    if prefix == "gateway":
        return _call_llm_provider_api(
            "gateway",
            suffix,
            input_data=payload,
            eval_parameters=eval_parameters,
            extra_headers=extra_headers,
            proxy_url=proxy_url,
        )

    elif prefix == "endpoints":
        from mlflow.deployments import get_deploy_client

        if isinstance(payload, str) and endpoint_type is None:
            client = get_deploy_client()
            endpoint_type = client.get_endpoint(suffix).endpoint_type
        return call_deployments_api(suffix, payload, eval_parameters, endpoint_type)
    elif prefix in ("model", "runs"):
        # TODO: call _load_model_or_server
        raise NotImplementedError

    # Import here to avoid loading gateway module at the top level
    from mlflow.gateway.provider_registry import is_supported_provider

    if is_supported_provider(prefix):
        return _call_llm_provider_api(
            prefix, suffix, payload, eval_parameters, extra_headers, proxy_url
        )

    raise MlflowException(
        f"Unknown model uri prefix '{prefix}'",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _parse_model_uri(model_uri: str) -> tuple[str, str]:
    """Parse a model URI of the form "<provider>:/<model-name>"."""
    # urllib.parse.urlparse is not used because provider names with underscores
    # (e.g., vertex_ai) are invalid in RFC 3986 URI schemes and would fail parsing.
    match model_uri.split(":/", 1):
        case [provider, model_path] if provider and model_path.lstrip("/"):
            return provider, model_path.lstrip("/")
        case _:
            raise MlflowException(
                f"Malformed model uri '{model_uri}'. The URI must be in the format of "
                "<provider>:/<model-name>, e.g., 'openai:/gpt-4.1-mini'.",
                error_code=INVALID_PARAMETER_VALUE,
            )


def convert_mlflow_uri_to_litellm(model_uri: str) -> str:
    """
    Convert MLflow model URI format to LiteLLM format.

    MLflow uses URIs like 'openai:/gpt-4' while LiteLLM expects 'openai/gpt-4'.
    For Databricks endpoints, MLflow uses 'endpoints:/endpoint-name' which needs
    to be converted to 'databricks/endpoints/endpoint-name' for LiteLLM.

    Args:
        model_uri: MLflow model URI (e.g., 'openai:/gpt-4', 'endpoints:/my-endpoint')

    Returns:
        LiteLLM-compatible model string (e.g., 'openai/gpt-4', 'databricks/endpoints/my-endpoint')
    """
    try:
        scheme, path = _parse_model_uri(model_uri)
    except Exception as e:
        raise MlflowException(f"Failed to convert MLflow model URI to LiteLLM format: {e}")
    if scheme in ("endpoints", "databricks"):
        return f"databricks/{path}"
    return f"{scheme}/{path}"


_PREDICT_ERROR_MSG = """\
Failed to call the deployment endpoint. Please check the deployment URL \
is set correctly and the input payload is valid.\n
- Error: {e}\n
- Deployment URI: {uri}\n
- Input payload: {payload}"""


def _is_supported_llm_provider(schema: str) -> bool:
    from mlflow.gateway.provider_registry import provider_registry

    return schema in provider_registry.keys()


_MODELS_WITHOUT_OUTPUT_CONFIG: set[tuple[str, str]] = set()


def _is_unsupported_output_format_error(exc: MlflowException) -> bool:
    """Check if the error indicates the model doesn't support structured output.

    Older Anthropic models (e.g. claude-sonnet-4-20250514) don't support ``output_config``
    and return a 400 with::

        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "'claude-sonnet-4-20250514' does not support output format.",
            },
        }

    Newer models (e.g. claude-sonnet-4-5-20250929) support it.
    """
    match exc.__cause__:
        case requests.exceptions.HTTPError(
            response=requests.Response(status_code=400) as response,
        ):
            try:
                body = response.json()
            except Exception:
                return False
            match body:
                case {
                    "error": {
                        "type": "invalid_request_error",
                        "message": str(msg),
                    }
                }:
                    return "does not support output format" in msg.lower()
    return False


def _call_llm_provider_api(
    provider_name: str,
    model: str,
    input_data: str | None = None,
    eval_parameters: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    Invoke chat endpoint of various LLM providers.

    Under the hood, this function uses the MLflow Gateway to transform the input/output data
    for different LLM providers.

    Args:
        provider_name: The provider name, e.g., "anthropic".
        model: The model name, e.g., "claude-3-5-sonnet"
        input_data: The input string prompt to send to the model as a chat message.
            Mutually exclusive with ``messages``.
        eval_parameters: The additional parameters to send to the model, e.g. temperature.
        extra_headers: The additional headers to send to the provider.
        proxy_url: Proxy URL to be used for the judge model. If not specified, the default
            URL for the LLM provider will be used.
        messages: Pre-built list of message dicts (``[{"role": ..., "content": ...}]``).
            Mutually exclusive with ``input_data``.
        response_format: Response format dict (e.g. from ``_pydantic_to_response_format``).
    """
    from mlflow.gateway.config import Provider
    from mlflow.gateway.schemas import chat

    if (input_data is None) == (messages is None):
        raise MlflowException.invalid_parameter_value(
            "Exactly one of input_data or messages must be provided."
        )

    eval_parameters = eval_parameters or {}
    extra_headers = extra_headers or {}
    provider = _get_provider_instance(provider_name, model)

    if messages is not None:
        payload = {"messages": messages} | eval_parameters
        if response_format is not None:
            payload["response_format"] = response_format
    else:
        chat_request = chat.RequestPayload(
            model=model,
            messages=[
                chat.RequestMessage(role="user", content=input_data),
            ],
            **eval_parameters,
        )

        # Filter out keys in the payload to the specified ones + "messages".
        # Does not include "model" key here because some providers do not accept it as a
        # part of the payload. Whether or not to include "model" key must be determined
        # by each provider implementation.
        filtered_keys = {"messages", *eval_parameters.keys()}

        payload = {
            k: v
            for k, v in chat_request.model_dump(exclude_none=True).items()
            if (v is not None) and (k in filtered_keys)
        }

    chat_payload = provider.adapter_class.chat_to_model(payload, provider.config)
    if messages is None:
        # eval_parameters were filtered out by the RequestPayload serialization;
        # re-apply them. When messages is not None, they're already in the payload.
        chat_payload.update(eval_parameters)

    if normalize_provider_name(provider_name) == Provider.BEDROCK:
        if proxy_url or extra_headers:
            _logger.warning(
                "Proxy URL and extra headers are not supported for Bedrock LLMs. "
                "Ignoring the provided proxy URL and extra headers.",
            )
        response = provider._request(chat_payload)
    else:
        if (provider_name, model) in _MODELS_WITHOUT_OUTPUT_CONFIG:
            chat_payload.pop("output_config", None)
            chat_payload.pop("response_format", None)

        try:
            response = _send_request(
                endpoint=proxy_url or provider.get_endpoint_url("llm/v1/chat"),
                headers=provider.headers | extra_headers,
                payload=chat_payload,
            )
        except MlflowException as e:
            if provider_name != "anthropic" or not _is_unsupported_output_format_error(e):
                raise
            # Model doesn't support structured output; remember and retry.
            _MODELS_WITHOUT_OUTPUT_CONFIG.add((provider_name, model))
            chat_payload.pop("output_config", None)
            chat_payload.pop("response_format", None)
            response = _send_request(
                endpoint=proxy_url or provider.get_endpoint_url("llm/v1/chat"),
                headers=provider.headers | extra_headers,
                payload=chat_payload,
            )
    chat_response = provider.adapter_class.model_to_chat(response, provider.config)
    if len(chat_response.choices) == 0:
        raise MlflowException(
            "Failed to score the provided input as the judge LLM did not return "
            "any chat completion results in the response."
        )
    content = chat_response.choices[0].message.content

    # NB: Evaluation only handles text content for now.
    return content[0].text if isinstance(content, list) else content


class _MlflowGatewayProvider(OpenAIProvider):
    """OpenAI-compatible provider for MLflow AI Gateway endpoints.

    Overrides ``headers`` to use gateway auth headers instead of
    the standard ``Bearer {api_key}`` used by OpenAIProvider.
    """

    def __init__(self, config: EndpointConfig, extra_headers: dict[str, str] | None = None):
        super().__init__(config)
        self._extra_headers = extra_headers

    @property
    def headers(self) -> dict[str, str]:
        return {**(self._extra_headers or {})}


def _get_provider_instance(provider: str, model: str) -> "BaseProvider":
    """Get the provider instance for the given provider name and the model name."""
    from mlflow.gateway.config import Provider

    def _get_route_config(config):
        return EndpointConfig(
            name=provider,
            endpoint_type="llm/v1/chat",
            model={
                "provider": provider,
                "name": model,
                "config": config.model_dump(),
            },
        )

    # NB: Not all LLM providers in MLflow Gateway are supported here. We can add
    # new ones as requested, as long as the provider support chat endpoints.
    if provider == Provider.OPENAI:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MlflowException.invalid_parameter_value(
                "OPENAI_API_KEY environment variable must be set to use the openai provider."
            )
        config = OpenAIConfig(openai_api_key=os.environ["OPENAI_API_KEY"])
        return OpenAIProvider(_get_route_config(config))

    elif provider == Provider.AZURE:
        if not os.environ.get(AZURE_API_KEY_ENV_VAR):
            raise MlflowException.invalid_parameter_value(
                f"{AZURE_API_KEY_ENV_VAR} environment variable must be set "
                "to use the azure provider."
            )
        config = OpenAIConfig(
            openai_api_key=os.environ[AZURE_API_KEY_ENV_VAR],
            openai_api_type="azure",
            openai_api_base=os.environ.get(AZURE_API_BASE_ENV_VAR),
            openai_api_version=os.environ.get(AZURE_API_VERSION_ENV_VAR),
            openai_deployment_name=model,
        )
        return OpenAIProvider(_get_route_config(config))

    elif provider == Provider.ANTHROPIC:
        from mlflow.gateway.providers.anthropic import AnthropicConfig, AnthropicProvider

        config = AnthropicConfig(anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return AnthropicProvider(_get_route_config(config))

    elif normalize_provider_name(provider) == Provider.BEDROCK:
        from mlflow.gateway.config import AWSBearerToken, AWSIdAndKey, AWSRole
        from mlflow.gateway.providers.bedrock import AmazonBedrockConfig, AmazonBedrockProvider

        if bearer_token := os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            aws_config = AWSBearerToken(
                aws_region=os.environ.get("AWS_REGION"),
                aws_bearer_token=bearer_token,
            )
        elif aws_role_arn := os.environ.get("AWS_ROLE_ARN"):
            aws_config = AWSRole(
                aws_region=os.environ.get("AWS_REGION"),
                aws_role_arn=aws_role_arn,
            )
        else:
            aws_config = AWSIdAndKey(
                aws_region=os.environ.get("AWS_REGION"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            )
        config = AmazonBedrockConfig(aws_config=aws_config)
        return AmazonBedrockProvider(_get_route_config(config))

    # # Cohere provider implementation seems to be broken and does not work with
    # # their latest APIs. Uncomment once the provider implementation is fixed.
    # elif provider == Provider.COHERE:
    #     from mlflow.gateway.providers.cohere import CohereConfig, CohereProvider

    #     config = CohereConfig(cohere_api_key=os.environ.get("COHERE_API_KEY"))
    #     return CohereProvider(_get_route_config(config))

    elif provider == Provider.GEMINI:
        from mlflow.gateway.providers.gemini import GeminiConfig, GeminiProvider
        from mlflow.utils.providers import _CORE_PROVIDER_ENV_VARS

        config = GeminiConfig(gemini_api_key=os.environ.get(_CORE_PROVIDER_ENV_VARS["gemini"]))
        return GeminiProvider(_get_route_config(config))

    elif provider == Provider.MISTRAL:
        from mlflow.gateway.providers.mistral import MistralConfig, MistralProvider

        config = MistralConfig(mistral_api_key=os.environ.get("MISTRAL_API_KEY"))
        return MistralProvider(_get_route_config(config))

    elif provider == Provider.TOGETHERAI:
        from mlflow.gateway.providers.togetherai import TogetherAIConfig, TogetherAIProvider

        config = TogetherAIConfig(togetherai_api_key=os.environ.get("TOGETHERAI_API_KEY"))
        return TogetherAIProvider(_get_route_config(config))

    elif provider == "gateway":
        gw_config = get_gateway_config(model)
        openai_config = OpenAIConfig(
            openai_api_key="mlflow-gateway-auth",
            openai_api_base=gw_config.api_base.rstrip("/"),
        )
        route_config = EndpointConfig(
            name="gateway",
            endpoint_type="llm/v1/chat",
            model={
                "provider": "openai",
                "name": model,
                "config": openai_config.model_dump(),
            },
        )
        return _MlflowGatewayProvider(route_config, extra_headers=gw_config.extra_headers)

    elif provider == Provider.GROQ:
        from mlflow.gateway.config import _OpenAICompatibleConfig
        from mlflow.gateway.providers.groq import GroqProvider

        config = _OpenAICompatibleConfig(api_key=os.environ.get("GROQ_API_KEY"))
        return GroqProvider(_get_route_config(config))

    elif provider == Provider.DEEPSEEK:
        from mlflow.gateway.config import _OpenAICompatibleConfig
        from mlflow.gateway.providers.deepseek import DeepSeekProvider

        config = _OpenAICompatibleConfig(api_key=os.environ.get("DEEPSEEK_API_KEY"))
        return DeepSeekProvider(_get_route_config(config))

    elif provider == Provider.XAI:
        from mlflow.gateway.config import _OpenAICompatibleConfig
        from mlflow.gateway.providers.xai import XAIProvider

        config = _OpenAICompatibleConfig(api_key=os.environ.get("XAI_API_KEY"))
        return XAIProvider(_get_route_config(config))

    elif provider == Provider.OPENROUTER:
        from mlflow.gateway.config import _OpenAICompatibleConfig
        from mlflow.gateway.providers.openrouter import OpenRouterProvider

        config = _OpenAICompatibleConfig(api_key=os.environ.get("OPENROUTER_API_KEY"))
        return OpenRouterProvider(_get_route_config(config))

    elif provider == Provider.OLLAMA:
        from mlflow.gateway.providers.ollama import OllamaConfig, OllamaProvider

        config = OllamaConfig(api_key=os.environ.get("OLLAMA_API_KEY", "ollama"))
        return OllamaProvider(_get_route_config(config))

    elif provider == Provider.DATABRICKS:
        from mlflow.gateway.providers.databricks import DatabricksConfig, DatabricksProvider

        config = DatabricksConfig(
            host=os.environ.get("DATABRICKS_HOST"),
            token=os.environ.get("DATABRICKS_TOKEN"),
            client_id=os.environ.get("DATABRICKS_CLIENT_ID"),
            client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET"),
        )
        return DatabricksProvider(_get_route_config(config))

    elif provider == Provider.VERTEX_AI:
        from mlflow.gateway.config import VertexAIConfig
        from mlflow.gateway.providers.vertex_ai import VertexAIProvider

        config = VertexAIConfig(
            vertex_project=os.environ.get("VERTEX_PROJECT", ""),
            vertex_location=os.environ.get("VERTEX_LOCATION"),
            vertex_credentials=os.environ.get("VERTEX_CREDENTIALS"),
        )
        return VertexAIProvider(_get_route_config(config))

    raise MlflowException(
        f"Provider '{provider}' is not supported for evaluation.",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _send_request(
    endpoint: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        response = requests.post(
            url=endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        body = getattr(e.response, "text", "")
        raise MlflowException(
            f"Failed to call LLM endpoint at {endpoint}.\n- Error: {e}\n"
            f"- Response body: {body}\n- Input payload: {payload}."
        ) from e

    return response.json()


def call_deployments_api(
    deployment_uri: str,
    input_data: str | dict[str, Any],
    eval_parameters: dict[str, Any] | None = None,
    endpoint_type: str | None = None,
):
    """Call the deployment endpoint with the given payload and parameters.

    Args:
        deployment_uri: The URI of the deployment endpoint.
        input_data: The input string or dictionary to send to the endpoint.
            - If it is a string, MLflow tries to construct the payload based on the endpoint type.
            - If it is a dictionary, MLflow directly sends it to the endpoint.
        eval_parameters: The evaluation parameters to send to the endpoint.
        endpoint_type: The type of the endpoint. If specified, must be 'llm/v1/completions'
            or 'llm/v1/chat'. If not specified, MLflow tries to get the endpoint type
            from the endpoint, and if not found, directly sends the payload to the endpoint.

    Returns:
        The unpacked response from the endpoint.
    """
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client()

    if isinstance(input_data, str):
        payload = _construct_payload_from_str(input_data, endpoint_type)
    elif isinstance(input_data, dict):
        # If the input is a dictionary, we assume it is already in the correct format
        payload = input_data
    else:
        raise MlflowException(
            f"Invalid input data type {type(input_data)}. Must be a string or a dictionary.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    payload = {**payload, **(eval_parameters or {})}

    try:
        response = client.predict(endpoint=deployment_uri, inputs=payload)
    except Exception as e:
        raise MlflowException(
            _PREDICT_ERROR_MSG.format(e=e, uri=deployment_uri, payload=payload)
        ) from e

    return _parse_response(response, endpoint_type)


def _construct_payload_from_str(prompt: str, endpoint_type: str) -> dict[str, Any]:
    """
    Construct the payload from the input string based on the endpoint type.
    If the endpoint type is not specified or unsupported one, raise an exception.
    """
    if endpoint_type == "llm/v1/completions":
        return {"prompt": prompt}
    elif endpoint_type == "llm/v1/chat":
        return {"messages": [{"role": "user", "content": prompt}]}
    else:
        raise MlflowException(
            f"Unsupported endpoint type: {endpoint_type}. If string input is provided, "
            "the endpoint type must be 'llm/v1/completions' or 'llm/v1/chat'.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _parse_response(
    response: dict[str, Any], endpoint_type: str | None
) -> str | None | dict[str, Any]:
    if endpoint_type == "llm/v1/completions":
        return _parse_completions_response_format(response)
    elif endpoint_type == "llm/v1/chat":
        return _parse_chat_response_format(response)
    else:
        return response


def _parse_chat_response_format(response):
    try:
        text = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = None
    return text


def _parse_completions_response_format(response):
    try:
        text = response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        text = None
    return text
