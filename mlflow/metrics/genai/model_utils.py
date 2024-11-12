import logging
import os
import urllib.parse
from typing import TYPE_CHECKING, Any, Optional, Union

import requests

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import REQUEST_URL_CHAT

if TYPE_CHECKING:
    from mlflow.gateway.providers import BaseProvider

_logger = logging.getLogger(__name__)


def get_endpoint_type(endpoint_uri: str) -> Optional[str]:
    """
    Get the type of the endpoint if it is MLflow deployment
    endpoint. For other endpoints e.g. OpenAI, or if the
    endpoint does not specify type, return None.
    """
    schema, path = _parse_model_uri(endpoint_uri)

    if schema != "endpoints":
        return None

    from pydantic import BaseModel

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client()

    endpoint = client.get_endpoint(path)
    # TODO: Standardize the return type of `get_endpoint` and remove this check
    endpoint = endpoint.dict() if isinstance(endpoint, BaseModel) else endpoint
    return endpoint.get("task", endpoint.get("endpoint_type"))


# TODO: improve this name
def score_model_on_payload(
    model_uri,
    payload,
    eval_parameters=None,
    extra_headers=None,
    proxy_url=None,
    endpoint_type=None,
):
    """Call the model identified by the given uri with the given string prompt."""

    eval_parameters = eval_parameters or {}
    extra_headers = extra_headers or {}

    prefix, suffix = _parse_model_uri(model_uri)

    if prefix == "openai":
        # TODO: Migrate OpenAI schema to use _call_llm_provider_api.
        return _call_openai_api(suffix, payload, eval_parameters, extra_headers, proxy_url)
    elif prefix == "gateway":
        return _call_gateway_api(suffix, payload, eval_parameters)
    elif prefix == "endpoints":
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


def _parse_model_uri(model_uri):
    parsed = urllib.parse.urlparse(model_uri, allow_fragments=False)
    scheme = parsed.scheme
    path = parsed.path
    if not path.startswith("/") or len(path) <= 1:
        raise MlflowException(
            f"Malformed model uri '{model_uri}'", error_code=INVALID_PARAMETER_VALUE
        )
    path = path.lstrip("/")
    return scheme, path


def _call_openai_api(openai_uri, payload, eval_parameters, extra_headers, proxy_url):
    if "OPENAI_API_KEY" not in os.environ:
        raise MlflowException(
            "OPENAI_API_KEY environment variable not set",
            error_code=INVALID_PARAMETER_VALUE,
        )

    from mlflow.openai import _get_api_config
    from mlflow.openai.api_request_parallel_processor import process_api_requests
    from mlflow.utils.openai_utils import _OAITokenHolder

    api_config = _get_api_config()
    api_token = _OAITokenHolder(api_config.api_type)

    payload = {
        "messages": [{"role": "user", "content": payload}],
        **eval_parameters,
    }

    if api_config.api_type in ("azure", "azure_ad", "azuread"):
        api_base = getattr(api_config, "api_base")
        api_version = getattr(api_config, "api_version")
        engine = getattr(api_config, "engine")
        deployment_id = getattr(api_config, "deployment_id")

        if engine:
            # Avoid using both parameters as they serve the same purpose
            # Invalid inputs:
            #   - Wrong engine + correct/wrong deployment_id
            #   - No engine + wrong deployment_id
            # Valid inputs:
            #   - Correct engine + correct/wrong deployment_id
            #   - No engine + correct deployment_id
            if deployment_id is not None:
                _logger.warning(
                    "Both engine and deployment_id are set. Using engine as it takes precedence."
                )
            payload = {"engine": engine, **payload}
        elif deployment_id is None:
            raise MlflowException(
                "Either engine or deployment_id must be set for Azure OpenAI API",
            )
        payload = payload

        request_url = proxy_url or (
            f"{api_base}/openai/deployments/{deployment_id}"
            f"/chat/completions?api-version={api_version}"
        )
    else:
        payload = {"model": openai_uri, **payload}
        request_url = proxy_url or REQUEST_URL_CHAT

    try:
        resp = process_api_requests(
            [payload],
            request_url,
            api_token=api_token,
            throw_original_error=True,
            max_workers=1,
            extra_headers=extra_headers,
        )[0]
    except MlflowException as e:
        raise e
    except Exception as e:
        raise MlflowException(f"Error response from OpenAI:\n {e}")

    return _parse_chat_response_format(resp)


_PREDICT_ERROR_MSG = """\
Failed to call the deployment endpoint. Please check the deployment URL \
is set correctly and the input payload is valid.\n
- Error: {e}\n
- Deployment URI: {uri}\n
- Input payload: {payload}"""


def _is_supported_llm_provider(schema: str) -> bool:
    from mlflow.gateway.provider_registry import provider_registry

    return schema in provider_registry.keys()


def _call_llm_provider_api(
    provider_name: str,
    model: str,
    input_data: str,
    eval_parameters: dict[str, Any],
    extra_headers: dict[str, str],
    proxy_url: Optional[str] = None,
) -> str:
    """
    Invoke chat endpoint of various LLM providers.

    Under the hood, this function uses the MLflow Gateway to transform the input/output data
    for different LLM providers.

    Args:
        provider_name: The provider name, e.g., "anthropic".
        model: The model name, e.g., "claude-3-5-sonnet"
        input_data: The input string prompt to send to the model as a chat message.
        eval_parameters: The additional parameters to send to the model, e.g. temperature.
        extra_headers: The additional headers to send to the provider.
        proxy_url: Proxy URL to be used for the judge model. If not specified, the default
            URL for the LLM provider will be used.
    """
    from mlflow.gateway.config import Provider
    from mlflow.gateway.schemas import chat

    provider = _get_provider_instance(provider_name, model)

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
        for k, v in chat_request.model_dump().items()
        if (v is not None) and (k in filtered_keys)
    }
    chat_payload = provider.adapter_class.chat_to_model(payload, provider.config)
    chat_payload.update(eval_parameters)

    if provider_name in [Provider.AMAZON_BEDROCK, Provider.BEDROCK]:
        if proxy_url or extra_headers:
            _logger.warning(
                "Proxy URL and extra headers are not supported for Bedrock LLMs. "
                "Ignoring the provided proxy URL and extra headers.",
            )
        response = provider._request(chat_payload)
    else:
        response = _send_request(
            endpoint=proxy_url or provider.get_endpoint_url("llm/v1/chat"),
            headers={**provider.headers, **extra_headers},
            payload=chat_payload,
        )
    chat_response = provider.adapter_class.model_to_chat(response, provider.config)
    if len(chat_response.choices) == 0:
        raise MlflowException(
            "Failed to score the provided input as the judge LLM did not return "
            "any chat completion results in the response."
        )
    return chat_response.choices[0].message.content


def _get_provider_instance(provider: str, model: str) -> "BaseProvider":
    """Get the provider instance for the given provider name and the model name."""
    from mlflow.gateway.config import Provider, RouteConfig

    def _get_route_config(config):
        return RouteConfig(
            name=provider,
            route_type="llm/v1/chat",
            model={
                "provider": provider,
                "name": model,
                "config": config.model_dump(),
            },
        )

    # NB: Not all LLM providers in MLflow Gateway are supported here. We can add
    # new ones as requested, as long as the provider support chat endpoints.
    if provider == Provider.ANTHROPIC:
        from mlflow.gateway.providers.anthropic import AnthropicConfig, AnthropicProvider

        config = AnthropicConfig(anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return AnthropicProvider(_get_route_config(config))

    elif provider in [Provider.AMAZON_BEDROCK, Provider.BEDROCK]:
        from mlflow.gateway.config import AWSIdAndKey, AWSRole
        from mlflow.gateway.providers.bedrock import AmazonBedrockConfig, AmazonBedrockProvider

        if aws_role_arn := os.environ.get("AWS_ROLE_ARN"):
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

    elif provider == Provider.MISTRAL:
        from mlflow.gateway.providers.mistral import MistralConfig, MistralProvider

        config = MistralConfig(mistral_api_key=os.environ.get("MISTRAL_API_KEY"))
        return MistralProvider(_get_route_config(config))

    elif provider == Provider.TOGETHERAI:
        from mlflow.gateway.providers.togetherai import TogetherAIConfig, TogetherAIProvider

        config = TogetherAIConfig(togetherai_api_key=os.environ.get("TOGETHERAI_API_KEY"))
        return TogetherAIProvider(_get_route_config(config))

    raise MlflowException(f"Provider '{provider}' is not supported for evaluation.")


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
        raise MlflowException(
            f"Failed to call LLM endpoint at {endpoint}.\n- Error: {e}\n- Input payload: {payload}."
        )

    return response.json()


def call_deployments_api(
    deployment_uri: str,
    input_data: Union[str, dict[str, Any]],
    eval_parameters: Optional[dict[str, Any]] = None,
    endpoint_type: Optional[str] = None,
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


def _call_gateway_api(gateway_uri, payload, eval_parameters):
    from mlflow.gateway import get_route, query

    route_info = get_route(gateway_uri).dict()
    if route_info["endpoint_type"] == "llm/v1/completions":
        completions_payload = {
            "prompt": payload,
            **eval_parameters,
        }
        response = query(gateway_uri, completions_payload)
        return _parse_completions_response_format(response)
    elif route_info["endpoint_type"] == "llm/v1/chat":
        chat_payload = {
            "messages": [{"role": "user", "content": payload}],
            **eval_parameters,
        }
        response = query(gateway_uri, chat_payload)
        return _parse_chat_response_format(response)
    else:
        raise MlflowException(
            f"Unsupported gateway route type: {route_info['endpoint_type']}. Use a "
            "route of type 'llm/v1/completions' or 'llm/v1/chat' instead.",
            error_code=INVALID_PARAMETER_VALUE,
        )


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
    response: dict[str, Any], endpoint_type: Optional[str]
) -> Union[Optional[str], dict[str, Any]]:
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
