import logging
import os
import urllib.parse
from typing import Any, Dict, Optional, Union

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import REQUEST_URL_CHAT

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
def score_model_on_payload(model_uri, payload, eval_parameters=None, endpoint_type=None):
    """Call the model identified by the given uri with the given string prompt."""

    if eval_parameters is None:
        eval_parameters = {}
    prefix, suffix = _parse_model_uri(model_uri)

    if prefix == "openai":
        return _call_openai_api(suffix, payload, eval_parameters)
    elif prefix == "gateway":
        return _call_gateway_api(suffix, payload, eval_parameters)
    elif prefix == "endpoints":
        return call_deployments_api(suffix, payload, eval_parameters, endpoint_type)
    elif prefix in ("model", "runs"):
        # TODO: call _load_model_or_server
        raise NotImplementedError
    else:
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


def _call_openai_api(openai_uri, payload, eval_parameters):
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

        request_url = (
            f"{api_base}/openai/deployments/{deployment_id}"
            f"/chat/completions?api-version={api_version}"
        )
    else:
        payload = {"model": openai_uri, **payload}
        request_url = REQUEST_URL_CHAT

    try:
        resp = process_api_requests(
            [payload],
            request_url,
            api_token=api_token,
            throw_original_error=True,
            max_workers=1,
        )[0]
    except MlflowException as e:
        raise e
    except Exception as e:
        raise MlflowException(f"Error response from OpenAI:\n {e}")

    return _parse_chat_response_format(resp)


_PREDICT_ERROR_MSG = """\
Failed to call the deployment endpoint. Please check the deployment URL\
is set correctly and the input payload is valid.\n
- Error: {e}\n
- Deployment URI: {uri}\n
- Input payload: {payload}"""


def call_deployments_api(
    deployment_uri: str,
    input_data: Union[str, Dict[str, Any]],
    eval_parameters: Optional[Dict[str, Any]] = None,
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


def _construct_payload_from_str(prompt: str, endpoint_type: str) -> Dict[str, Any]:
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
    response: Dict[str, Any], endpoint_type: Optional[str]
) -> Union[Optional[str], Dict[str, Any]]:
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
