import logging
import os
import urllib.parse

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE, UNAUTHENTICATED

ROUTE_TYPE = "llm/v1/completions"

_logger = logging.getLogger(__name__)


# TODO: improve this name
def score_model_on_payload(model_uri, payload, eval_parameters=None):
    """Call the model identified by the given uri with the given payload."""

    if eval_parameters is None:
        eval_parameters = {}
    prefix, suffix = _parse_model_uri(model_uri)

    if prefix == "openai":
        return _call_openai_api(suffix, payload)
    elif prefix == "gateway":
        return _call_gateway_api(suffix, payload, eval_parameters)
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


def _call_openai_api(openai_uri, payload):
    """Wrapper around the OpenAI API to make it compatible with the MLflow Gateway API."""

    if "OPENAI_API_KEY" not in os.environ:
        raise MlflowException(
            "OPENAI_API_KEY environment variable not set",
            error_code=INVALID_PARAMETER_VALUE,
        )

    import openai

    from mlflow.openai import _get_api_config
    from mlflow.openai.api_request_parallel_processor import process_api_requests
    from mlflow.openai.utils import _OAITokenHolder

    api_config = _get_api_config()
    api_token = _OAITokenHolder(api_config.api_type)
    envs = {
        x: getattr(api_config, x)
        for x in ["api_base", "api_version", "api_type", "engine", "deployment_id"]
        if getattr(api_config, x) is not None
    }

    payload = {{"candidate_count": "n"}.get(k, k): v for k, v in payload.items()}
    # The range of OpenAI's temperature is 0-2, but ours is 0-1, so we double it.
    payload["temperature"] = 2 * payload["temperature"]
    payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]

    if api_config.api_type in ("azure", "azure_ad", "azuread"):
        deployment_id = envs.get("deployment_id")
        if envs.get("engine"):
            # Avoid using both parameters as they serve the same purpose
            # Invalid inputs:
            #   - Wrong engine + correct/wrong deployment_id
            #   - No engine + wrong deployment_id
            # Valid inputs:
            #   - Correct engine + correct/wrong deployment_id
            #   - No engine + correct deployment_id
            if deployment_id is not None:
                _logger.warning(
                    "Both engine and deployment_id are set. " "Using engine as it takes precedence."
                )
        elif deployment_id is None:
            raise MlflowException(
                "Either engine or deployment_id must be set for Azure OpenAI API",
            )
        payload = payload
    else:
        payload = {"model": openai_uri, **payload}

    payload_with_envs = {**payload, **envs}

    try:
        resp = process_api_requests(
            [payload_with_envs],
            openai.ChatCompletion,
            api_token=api_token,
            throw_original_error=True,
            max_workers=1,
        )[0]
    except openai.error.AuthenticationError as e:
        raise MlflowException(
            f"Authentication Error for OpenAI. Error response:\n {e}",
            error_code=UNAUTHENTICATED,
        )
    except openai.error.InvalidRequestError as e:
        raise MlflowException(
            f"Invalid Request to OpenAI. Error response:\n {e}", error_code=BAD_REQUEST
        )
    except MlflowException as e:
        raise e
    except Exception as e:
        raise MlflowException(f"Error response from OpenAI:\n {e}")

    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = None
    return text


def _call_gateway_api(gateway_uri, payload, eval_parameters):
    from mlflow.gateway import get_route, query

    route_info = get_route(gateway_uri).dict()
    if route_info["route_type"] == "llm/v1/completions":
        completions_payload = {
            "prompt": payload,
            **eval_parameters,
        }
        response = query(gateway_uri, completions_payload)
        try:
            text = response["candidates"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = None
        return text
    elif route_info["route_type"] == "llm/v1/chat":
        chat_payload = {
            "messages": [{"role": "user", "content": payload}],
            **eval_parameters,
        }
        response = query(gateway_uri, chat_payload)
        try:
            text = response["candidates"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = None
        return text
    else:
        raise MlflowException(
            f"Unsupported gateway route type: {route_info['route_type']}. Use a "
            "route of type 'llm/v1/completions' or 'llm/v1/chat' instead.",
            error_code=INVALID_PARAMETER_VALUE,
        )
