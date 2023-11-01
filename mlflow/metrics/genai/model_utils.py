import json
import os
import urllib.parse

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE, UNAUTHENTICATED

ROUTE_TYPE = "llm/v1/completions"


# TODO: improve this name
def score_model_on_payload(model_uri, payload):
    """Call the model identified by the given uri with the given payload."""

    prefix, suffix = _parse_model_uri(model_uri)

    if prefix == "openai":
        return _call_openai_api(suffix, payload)
    elif prefix == "gateway":
        return _call_gateway_api(suffix, payload)
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
    from mlflow.gateway.config import RouteConfig
    from mlflow.gateway.providers.openai import OpenAIProvider

    if "OPENAI_API_KEY" not in os.environ:
        raise MlflowException(
            "OPENAI_API_KEY environment variable not set",
            error_code=INVALID_PARAMETER_VALUE,
        )

    config = {"openai_api_key": os.environ["OPENAI_API_KEY"]}
    if "OPENAI_API_BASE" in os.environ:
        config["openai_api_base"] = os.environ["OPENAI_API_BASE"]
    if "OPENAI_API_TYPE" in os.environ:
        config["openai_api_type"] = os.environ["OPENAI_API_TYPE"]
    if "OPENAI_API_VERSION" in os.environ:
        config["openai_api_version"] = os.environ["OPENAI_API_VERSION"]
    if "OPENAI_DEPLOYMENT_NAME" in os.environ:
        config["openai_deployment_name"] = os.environ["OPENAI_DEPLOYMENT_NAME"]

    route_config = RouteConfig(
        name="openai",
        route_type=ROUTE_TYPE,
        model={
            "name": openai_uri,
            "provider": "openai",
            "config": config,
        },
    )
    openai_provider = OpenAIProvider(route_config)

    payload = openai_provider._prepare_completion_request_payload(payload)

    import openai

    from mlflow.openai.api_request_parallel_processor import process_api_requests
    from mlflow.openai.utils import _OAITokenHolder

    api_token = _OAITokenHolder(os.environ.get("OPENAI_API_TYPE", "openai"))

    try:
        resp = process_api_requests(
            [openai_provider._add_model_to_payload_if_necessary(payload)],
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

    return json.loads(openai_provider._prepare_completion_response_payload(resp).json())


def _call_gateway_api(gateway_uri, payload):
    from mlflow.gateway import query

    return query(gateway_uri, payload)
