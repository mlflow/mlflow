from __future__ import annotations

from dataclasses import dataclass

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.tracking import get_tracking_uri
from mlflow.utils.uri import append_to_uri_path, is_http_uri


@dataclass
class GatewayLiteLLMConfig:
    api_base: str
    api_key: str
    model: str


def get_gateway_litellm_config(endpoint_name: str) -> GatewayLiteLLMConfig:
    """
    Get the LiteLLM configuration for invoking an MLflow Gateway endpoint.

    Args:
        endpoint_name: The name of the gateway endpoint (e.g., "chat" from "gateway:/chat").

    Returns:
        A GatewayLiteLLMConfig with api_base, api_key, and model configured for LiteLLM.

    Raises:
        MlflowException: If the gateway URI is not a valid HTTP(S) URL.
    """
    # MLFLOW_GATEWAY_URI takes precedence over tracking URI for gateway routing.
    # This is needed for async job workers: the job infrastructure passes the HTTP
    # tracking URI (e.g., http://127.0.0.1:5000) to workers, but _get_tracking_store()
    # overwrites MLFLOW_TRACKING_URI with the backend store URI (e.g., sqlite://).
    # Job workers set MLFLOW_GATEWAY_URI to preserve the HTTP URI for gateway calls.
    gateway_uri = MLFLOW_GATEWAY_URI.get() or get_tracking_uri()

    if not is_http_uri(gateway_uri):
        raise MlflowException(
            f"Gateway provider requires an HTTP(S) tracking URI, but got: '{gateway_uri}'. "
            "The gateway provider routes requests through the MLflow tracking server. "
            "Please set MLFLOW_TRACKING_URI to a valid HTTP(S) URL "
            "(e.g., 'http://localhost:5000' or 'https://your-mlflow-server.com')."
        )

    return GatewayLiteLLMConfig(
        api_base=append_to_uri_path(gateway_uri, "gateway/mlflow/v1/"),
        # LiteLLM requires api_key when using custom api_base. Gateway handles
        # auth in the server layer, so we pass a dummy value to satisfy LiteLLM.
        api_key="mlflow-gateway-auth",
        # Use openai/ prefix for LiteLLM to use OpenAI-compatible format.
        # LiteLLM strips the prefix, so gateway receives endpoint_name as the model.
        model=f"openai/{endpoint_name}",
    )
