from __future__ import annotations

import base64
from dataclasses import dataclass

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.tracking import get_tracking_uri
from mlflow.utils.credentials import read_mlflow_creds
from mlflow.utils.uri import append_to_uri_path, is_http_uri


@dataclass
class GatewayConfig:
    """Generic gateway endpoint configuration for direct HTTP calls."""

    api_base: str
    endpoint_name: str
    extra_headers: dict[str, str] | None


@dataclass
class GatewayLiteLLMConfig:
    """Gateway configuration with LiteLLM-specific fields (api_key, model format)."""

    api_base: str
    api_key: str
    model: str
    extra_headers: dict[str, str] | None


def get_gateway_config(endpoint_name: str) -> GatewayConfig:
    """
    Get the gateway configuration for invoking an MLflow Gateway endpoint directly.

    This is the generic version that returns the raw gateway config without
    LiteLLM-specific fields. Use this for direct HTTP calls to the gateway.

    Args:
        endpoint_name: The name of the gateway endpoint (e.g., "chat" from "gateway:/chat").

    Returns:
        A GatewayConfig with api_base, endpoint_name, and extra_headers.

    Raises:
        MlflowException: If the gateway URI is not a valid HTTP(S) URL.
    """
    gateway_uri = _resolve_gateway_uri()

    return GatewayConfig(
        api_base=append_to_uri_path(gateway_uri, "gateway/mlflow/v1/"),
        endpoint_name=endpoint_name,
        extra_headers=_build_auth_headers(),
    )


def get_gateway_litellm_config(endpoint_name: str) -> GatewayLiteLLMConfig:
    """
    Get the LiteLLM configuration for invoking an MLflow Gateway endpoint.

    Wraps ``get_gateway_config`` with LiteLLM-specific fields: a dummy api_key
    (required by LiteLLM) and an ``openai/`` model prefix (LiteLLM routing format).

    Args:
        endpoint_name: The name of the gateway endpoint (e.g., "chat" from "gateway:/chat").

    Returns:
        A GatewayLiteLLMConfig with api_base, api_key, model, and extra_headers
        configured for LiteLLM.

    Raises:
        MlflowException: If the gateway URI is not a valid HTTP(S) URL.
    """
    config = get_gateway_config(endpoint_name)

    return GatewayLiteLLMConfig(
        api_base=config.api_base,
        # Static dummy key to satisfy LiteLLM's requirement for a non-empty api_key.
        api_key="mlflow-gateway-auth",
        # Use openai/ prefix for LiteLLM to use OpenAI-compatible format.
        # LiteLLM strips the prefix, so gateway receives endpoint_name as the model.
        model=f"openai/{endpoint_name}",
        extra_headers=config.extra_headers,
    )


def _resolve_gateway_uri() -> str:
    """Resolve the gateway URI from environment or tracking URI."""
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

    return gateway_uri


def _build_auth_headers() -> dict[str, str] | None:
    """Build authentication headers from MLflow credentials if available."""
    creds = read_mlflow_creds()
    if creds.username and creds.password:
        encoded = base64.b64encode(f"{creds.username}:{creds.password}".encode()).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}
    return None
