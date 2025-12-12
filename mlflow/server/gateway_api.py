"""
Database-backed Gateway API endpoints for MLflow Server.

This module provides dynamic gateway endpoints that are configured from the database
rather than from a static YAML configuration file. It integrates the AI Gateway
functionality directly into the MLflow tracking server.
"""

import functools
import logging

from fastapi import APIRouter, HTTPException, Request

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    AmazonBedrockConfig,
    AnthropicConfig,
    AWSBaseConfig,
    EndpointConfig,
    EndpointType,
    OpenAIConfig,
    Provider,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import make_streaming_response
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.store.tracking.gateway.config_resolver import get_endpoint_config
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

# Create router for gateway endpoints
gateway_router = APIRouter(prefix="/gateway", tags=["gateway"])


def _translate_http_exception(func):
    """
    Decorator for translating MLflow exceptions to HTTP exceptions.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise  # Re-raise HTTPException as-is to preserve status codes
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            _logger.exception(f"Unexpected error in gateway endpoint: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}")

    return wrapper


def _create_provider_from_endpoint_config(
    endpoint_id: str, store: SqlAlchemyStore, endpoint_type: EndpointType
) -> BaseProvider:
    """
    Create a provider instance from database endpoint configuration.

    Args:
        endpoint_id: The endpoint ID to retrieve configuration for.
        store: The SQLAlchemy store instance.
        endpoint_type: Endpoint type (chat or embeddings).

    Returns:
        Provider instance

    Raises:
        MlflowException: If endpoint not found or configuration is invalid.
    """
    # Get endpoint config with decrypted secrets
    endpoint_config = get_endpoint_config(endpoint_id=endpoint_id, store=store)

    if not endpoint_config.models:
        raise MlflowException(
            f"Endpoint '{endpoint_id}' has no models configured",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    # For now, use the first model (later we can support traffic routing)
    model_config = endpoint_config.models[0]

    # Build provider-specific configuration
    provider_enum = Provider(model_config.provider)

    if provider_enum == Provider.OPENAI:
        provider_config = OpenAIConfig(
            openai_api_key=model_config.secret_value.get("api_key"),
            openai_api_base=model_config.auth_config.get("api_base")
            if model_config.auth_config
            else None,
        )
    elif provider_enum == Provider.ANTHROPIC:
        provider_config = AnthropicConfig(
            anthropic_api_key=model_config.secret_value.get("api_key"),
            anthropic_version=model_config.auth_config.get("version")
            if model_config.auth_config
            else None,
        )
    elif provider_enum in (Provider.BEDROCK, Provider.AMAZON_BEDROCK):
        auth_config = model_config.auth_config or {}
        provider_config = AmazonBedrockConfig(
            aws_config=AWSBaseConfig(**auth_config) if auth_config else AWSBaseConfig(),
        )
    else:
        # TODO: Support long-tail providers with LiteLLM
        raise MlflowException(
            f"Provider '{model_config.provider}' is not yet supported for database-backed endpoints"
        )

    # Create a temporary EndpointConfig for the provider
    # This mimics what the gateway does with yaml-based configs
    gateway_endpoint_config = EndpointConfig(
        name=endpoint_config.endpoint_name,
        endpoint_type=endpoint_type,
        model={
            "name": model_config.model_name,
            "provider": provider_enum.value,
            "config": provider_config.model_dump(),
        },
    )

    provider_class = get_provider(provider_enum)

    return provider_class(gateway_endpoint_config)


def _create_invocations_handler(endpoint_id: str, store: SqlAlchemyStore):
    """
    Create a unified invocations endpoint handler that supports both chat and embeddings.

    The handler automatically detects the request type based on the payload structure:
    - If payload has "messages" field -> chat endpoint
    - If payload has "input" field -> embeddings endpoint
    """

    @_translate_http_exception
    async def _invocations(request: Request):
        # Get raw JSON to determine request type
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

        # Detect request type based on payload structure
        if "messages" in body:
            # Chat request
            endpoint_type = EndpointType.LLM_V1_CHAT
            try:
                payload = chat.RequestPayload(**body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid chat payload: {e!s}")

            provider = _create_provider_from_endpoint_config(endpoint_id, store, endpoint_type)

            if payload.stream:
                return await make_streaming_response(provider.chat_stream(payload))
            else:
                return await provider.chat(payload)

        elif "input" in body:
            # Embeddings request
            endpoint_type = EndpointType.LLM_V1_EMBEDDINGS
            try:
                payload = embeddings.RequestPayload(**body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid embeddings payload: {e!s}")

            provider = _create_provider_from_endpoint_config(endpoint_id, store, endpoint_type)

            return await provider.embeddings(payload)

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: payload format must be either chat or embeddings",
            )

    return _invocations


def register_gateway_endpoints(store: SqlAlchemyStore) -> APIRouter:
    """
    Register dynamic gateway endpoints from the database.

    This function queries the database for all configured endpoints and dynamically
    registers FastAPI routes for each one.

    Args:
        store: The SQLAlchemy store instance to query for endpoints.

    Returns:
        The configured APIRouter with all gateway endpoints registered.
    """
    # Get all endpoints from the database
    try:
        endpoints = store.list_gateway_endpoints()
    except Exception as e:
        _logger.warning(f"Failed to load gateway endpoints from database: {e}")
        endpoints = []

    _logger.info(f"Registering {len(endpoints)} gateway endpoints from database")

    for endpoint in endpoints:
        endpoint_name = endpoint.name
        endpoint_id = endpoint.endpoint_id

        # Register unified invocations endpoint (supports both chat and embeddings)
        invocations_path = f"/{endpoint_name}/mlflow/invocations"
        gateway_router.add_api_route(
            path=invocations_path,
            endpoint=_create_invocations_handler(endpoint_id, store),
            methods=["POST"],
            response_model=None,  # Dynamic response based on request type
            name=f"invocations_{endpoint_name}",
            summary=f"Invocations endpoint for {endpoint_name} (supports chat and embeddings)",
        )
        _logger.info(f"Registered invocations endpoint: /gateway{invocations_path}")

    return gateway_router


def get_gateway_router() -> APIRouter:
    """
    Get the configured gateway router.

    This is the main entry point for integrating the gateway API into the MLflow server.

    Returns:
        APIRouter configured with all database-backed gateway endpoints.
    """
    store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        _logger.warning(
            f"Gateway endpoints require SqlAlchemyStore, got {type(store).__name__}. "
            "Gateway endpoints will not be available."
        )
        return gateway_router

    return register_gateway_endpoints(store)
