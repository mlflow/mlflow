"""
Database-backed Gateway API endpoints for MLflow Server.

This module provides dynamic gateway endpoints that are configured from the database
rather than from a static YAML configuration file. It integrates the AI Gateway
functionality directly into the MLflow tracking server.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    AmazonBedrockConfig,
    AnthropicConfig,
    AWSBaseConfig,
    AWSIdAndKey,
    AWSRole,
    EndpointConfig,
    EndpointType,
    GeminiConfig,
    MistralConfig,
    OpenAIConfig,
    Provider,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import make_streaming_response, translate_http_exception
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.tracking.gateway.config_resolver import get_endpoint_config
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

gateway_router = APIRouter(prefix="/gateway", tags=["gateway"])


def _create_provider_from_endpoint_name(
    store: SqlAlchemyStore, endpoint_name: str, endpoint_type: EndpointType
) -> BaseProvider:
    """
    Create a provider instance from database endpoint configuration.

    Args:
        store: The SQLAlchemy store instance.
        endpoint_name: The endpoint name to retrieve configuration for.
        endpoint_type: Endpoint type (chat or embeddings).

    Returns:
        Provider instance

    Raises:
        MlflowException: If endpoint not found or configuration is invalid.
    """
    # Get endpoint config with decrypted secrets
    endpoint_config = get_endpoint_config(endpoint_name=endpoint_name, store=store)

    if not endpoint_config.models:
        raise MlflowException(
            f"Endpoint '{endpoint_name}' has no models configured",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    # For now, use the first model (TODO: Support traffic routing)
    model_config = endpoint_config.models[0]

    if model_config.provider == Provider.OPENAI:
        auth_config = model_config.auth_config or {}
        openai_config = {
            "openai_api_key": model_config.secret_value.get("api_key"),
        }

        # Check if this is Azure OpenAI (requires api_type, deployment_name, api_base, api_version)
        if "api_type" in auth_config and auth_config["api_type"] in ("azure", "azuread"):
            openai_config["openai_api_type"] = auth_config["api_type"]
            openai_config["openai_api_base"] = auth_config.get("api_base")
            openai_config["openai_deployment_name"] = auth_config.get("deployment_name")
            openai_config["openai_api_version"] = auth_config.get("api_version")
        else:
            # Standard OpenAI
            if "api_base" in auth_config:
                openai_config["openai_api_base"] = auth_config["api_base"]
            if "organization" in auth_config:
                openai_config["openai_organization"] = auth_config["organization"]

        provider_config = OpenAIConfig(**openai_config)
    elif model_config.provider == Provider.ANTHROPIC:
        anthropic_config = {
            "anthropic_api_key": model_config.secret_value.get("api_key"),
        }
        if model_config.auth_config and "version" in model_config.auth_config:
            anthropic_config["anthropic_version"] = model_config.auth_config["version"]
        provider_config = AnthropicConfig(**anthropic_config)
    elif model_config.provider in (Provider.BEDROCK, Provider.AMAZON_BEDROCK):
        # Bedrock supports multiple auth modes
        auth_config = model_config.auth_config or {}
        secret_value = model_config.secret_value or {}

        # Check for role-based auth (aws_role_arn in auth_config)
        if "aws_role_arn" in auth_config:
            aws_config = AWSRole(
                aws_role_arn=auth_config["aws_role_arn"],
                session_length_seconds=auth_config.get("session_length_seconds", 15 * 60),
                aws_region=auth_config.get("aws_region"),
            )
        # Check for access key auth (credentials in secret_value)
        elif "aws_access_key_id" in secret_value:
            aws_config = AWSIdAndKey(
                aws_access_key_id=secret_value["aws_access_key_id"],
                aws_secret_access_key=secret_value["aws_secret_access_key"],
                aws_session_token=secret_value.get("aws_session_token"),
                aws_region=auth_config.get("aws_region"),
            )
        else:
            aws_config = AWSBaseConfig(
                aws_region=auth_config.get("aws_region"),
            )

        provider_config = AmazonBedrockConfig(aws_config=aws_config)
    elif model_config.provider == Provider.MISTRAL:
        provider_config = MistralConfig(
            mistral_api_key=model_config.secret_value.get("api_key"),
        )
    elif model_config.provider == Provider.GEMINI:
        provider_config = GeminiConfig(
            gemini_api_key=model_config.secret_value.get("api_key"),
        )
    else:
        # TODO: Support long-tail providers with LiteLLM
        raise NotImplementedError(f"Provider {model_config.provider} is not supported")

    # Create an EndpointConfig for the provider
    gateway_endpoint_config = EndpointConfig(
        name=endpoint_config.endpoint_name,
        endpoint_type=endpoint_type,
        model={
            "name": model_config.model_name,
            "provider": model_config.provider,
            "config": provider_config.model_dump(),
        },
    )

    provider_class = get_provider(model_config.provider)

    return provider_class(gateway_endpoint_config)


@gateway_router.post("/{endpoint_name}/mlflow/invocations")
@translate_http_exception
async def invocations(endpoint_name: str, request: Request):
    """
    Create a unified invocations endpoint handler that supports both chat and embeddings.

    The handler automatically detects the request type based on the payload structure:
    - If payload has "messages" field -> chat endpoint
    - If payload has "input" field -> embeddings endpoint
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    store = _get_store()

    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(
            status_code=500,
            detail="Gateway endpoints are only available with SqlAlchemyStore, "
            f"got {type(store).__name__}.",
        )

    # Detect request type based on payload structure
    if "messages" in body:
        # Chat request
        endpoint_type = EndpointType.LLM_V1_CHAT
        try:
            payload = chat.RequestPayload(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid chat payload: {e!s}")

        provider = _create_provider_from_endpoint_name(store, endpoint_name, endpoint_type)

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

        provider = _create_provider_from_endpoint_name(store, endpoint_name, endpoint_type)

        return await provider.embeddings(payload)

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: payload format must be either chat or embeddings",
        )
