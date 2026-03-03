from __future__ import annotations

import os
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


def _get_gateway_api_key() -> str:
    """Return the API key for internal gateway requests.

    When basic-auth is enabled, the server generates a random token at startup and
    stores it in ``_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN``.  Job subprocesses inherit
    this env var and use it as a Bearer token so the auth middleware can recognise
    them as trusted internal callers.  When the env var is absent (auth disabled or
    running outside the server) we fall back to a static placeholder that satisfies
    LiteLLM's requirement for a non-empty ``api_key``.
    """
    return os.environ.get("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", "mlflow-gateway-auth")


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
        api_key=_get_gateway_api_key(),
        # Use openai/ prefix for LiteLLM to use OpenAI-compatible format.
        # LiteLLM strips the prefix, so gateway receives endpoint_name as the model.
        model=f"openai/{endpoint_name}",
    )
