import os
import urllib.parse

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

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

    route_config = RouteConfig(
        name="openai",
        route_type=ROUTE_TYPE,
        model={
            "name": openai_uri,
            "provider": "openai",
            "config": {"openai_api_key": os.environ["OPENAI_API_KEY"]},
        },
    )
    openai_provider = OpenAIProvider(route_config)

    return openai_provider.sync_completions(payload)


def _call_gateway_api(gateway_uri, payload):
    from mlflow.gateway import query

    return query(gateway_uri, payload)
