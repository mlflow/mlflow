"""
Database-backed Gateway API endpoints for MLflow Server.

This module provides dynamic gateway endpoints that are configured from the database
rather than from a static YAML configuration file. It integrates the AI Gateway
functionality directly into the MLflow tracking server.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

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
    LiteLLMConfig,
    MistralConfig,
    OpenAIConfig,
    Provider,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.providers.base import PASSTHROUGH_ROUTES, BaseProvider, PassthroughAction
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import make_streaming_response, translate_http_exception
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.tracking.abstract_store import AbstractStore
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
        # Use LiteLLM as fallback for unsupported providers
        # Store the original provider name for LiteLLM's provider/model format
        original_provider = model_config.provider
        litellm_config = {
            "litellm_provider": original_provider,
            "litellm_api_key": model_config.secret_value.get("api_key"),
        }
        auth_config = model_config.auth_config or {}
        if "api_base" in auth_config:
            litellm_config["litellm_api_base"] = auth_config["api_base"]
        provider_config = LiteLLMConfig(**litellm_config)
        model_config.provider = Provider.LITELLM

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


def _validate_store(store: AbstractStore):
    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(
            status_code=500,
            detail="Gateway endpoints are only available with SqlAlchemyStore, "
            f"got {type(store).__name__}.",
        )


def _extract_endpoint_name_from_model(body: dict[str, Any]) -> str:
    """
    Extract and validate the endpoint name from the 'model' parameter in the request body.

    Args:
        body: The request body dictionary

    Returns:
        The endpoint name extracted from the 'model' parameter

    Raises:
        HTTPException: If the 'model' parameter is missing
    """
    endpoint_name = body.get("model")
    if not endpoint_name:
        raise HTTPException(
            status_code=400,
            detail="Missing required 'model' parameter in request body",
        )
    return endpoint_name


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

    _validate_store(store)

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


@gateway_router.post("/mlflow/v1/chat/completions")
@translate_http_exception
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    This endpoint follows the OpenAI API format where the endpoint name is specified
    via the "model" parameter in the request body, allowing clients to use the
    standard OpenAI SDK.

    Example:
        POST /gateway/mlflow/v1/chat/completions
        {
            "model": "my-endpoint-name",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    # Extract endpoint name from "model" parameter
    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")

    store = _get_store()

    _validate_store(store)

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


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_CHAT])
@translate_http_exception
async def openai_passthrough_chat(request: Request):
    """
    OpenAI passthrough endpoint for chat completions.

    This endpoint accepts raw OpenAI API format and passes it through to the
    OpenAI provider with the configured API key and model. The 'model' parameter
    in the request specifies which MLflow endpoint to use.

    Supports streaming responses when the 'stream' parameter is set to true.

    Example:
        POST /gateway/openai/v1/chat/completions
        {
            "model": "my-openai-endpoint",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "stream": true
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.OPENAI_CHAT, body)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_EMBEDDINGS])
@translate_http_exception
async def openai_passthrough_embeddings(request: Request):
    """
    OpenAI passthrough endpoint for embeddings.

    This endpoint accepts raw OpenAI API format and passes it through to the
    OpenAI provider with the configured API key and model. The 'model' parameter
    in the request specifies which MLflow endpoint to use.

    Example:
        POST /gateway/openai/v1/embeddings
        {
            "model": "my-openai-endpoint",
            "input": "The food was delicious and the waiter..."
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    provider = _create_provider_from_endpoint_name(
        store, endpoint_name, EndpointType.LLM_V1_EMBEDDINGS
    )
    return await provider.passthrough(PassthroughAction.OPENAI_EMBEDDINGS, body)


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_RESPONSES])
@translate_http_exception
async def openai_passthrough_responses(request: Request):
    """
    OpenAI passthrough endpoint for the Responses API.

    This endpoint accepts raw OpenAI Responses API format and passes it through to the
    OpenAI provider with the configured API key and model. The 'model' parameter
    in the request specifies which MLflow endpoint to use.

    Supports streaming responses when the 'stream' parameter is set to true.

    Example:
        POST /gateway/openai/v1/responses
        {
            "model": "my-openai-endpoint",
            "input": [{"type": "text", "text": "Hello"}],
            "instructions": "You are a helpful assistant",
            "stream": true
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.OPENAI_RESPONSES, body)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.ANTHROPIC_MESSAGES])
@translate_http_exception
async def anthropic_passthrough_messages(request: Request):
    """
    Anthropic passthrough endpoint for the Messages API.

    This endpoint accepts raw Anthropic API format and passes it through to the
    Anthropic provider with the configured API key and model. The 'model' parameter
    in the request specifies which MLflow endpoint to use.

    Supports streaming responses when the 'stream' parameter is set to true.

    Example:
        POST /gateway/anthropic/v1/messages
        {
            "model": "my-anthropic-endpoint",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "stream": true
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, body)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.GEMINI_GENERATE_CONTENT])
@translate_http_exception
async def gemini_passthrough_generate_content(endpoint_name: str, request: Request):
    """
    Gemini passthrough endpoint for generateContent API (non-streaming).

    This endpoint accepts raw Gemini API format and passes it through to the
    Gemini provider with the configured API key. The endpoint_name in the URL path
    specifies which MLflow endpoint to use.

    Example:
        POST /gateway/gemini/v1beta/models/my-gemini-endpoint:generateContent
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello"}]
                }
            ]
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    store = _get_store()
    _validate_store(store)

    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    return await provider.passthrough(PassthroughAction.GEMINI_GENERATE_CONTENT, body)


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT])
@translate_http_exception
async def gemini_passthrough_stream_generate_content(endpoint_name: str, request: Request):
    """
    Gemini passthrough endpoint for streamGenerateContent API (streaming).

    This endpoint accepts raw Gemini API format and passes it through to the
    Gemini provider with the configured API key. The endpoint_name in the URL path
    specifies which MLflow endpoint to use.

    Example:
        POST /gateway/gemini/v1beta/models/my-gemini-endpoint:streamGenerateContent
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello"}]
                }
            ]
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")

    store = _get_store()
    _validate_store(store)

    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT, body)
    return StreamingResponse(response, media_type="text/event-stream")
