"""
Database-backed Gateway API endpoints for MLflow Server.

This module provides dynamic gateway endpoints that are configured from the database
rather than from a static YAML configuration file. It integrates the AI Gateway
functionality directly into the MLflow tracking server.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from mlflow.entities.gateway_endpoint import GatewayModelLinkageType
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    AnthropicConfig,
    EndpointConfig,
    EndpointType,
    GeminiConfig,
    LiteLLMConfig,
    MistralConfig,
    OpenAIAPIType,
    OpenAIConfig,
    Provider,
    _AuthConfigKey,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.providers.base import (
    PASSTHROUGH_ROUTES,
    BaseProvider,
    FallbackProvider,
    PassthroughAction,
    TrafficRouteProvider,
)
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import make_streaming_response, translate_http_exception
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.gateway.config_resolver import get_endpoint_config
from mlflow.store.tracking.gateway.entities import (
    GatewayEndpointConfig,
    GatewayModelConfig,
    RoutingStrategy,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.telemetry.events import GatewayInvocationEvent, GatewayInvocationType
from mlflow.telemetry.track import _record_event
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

gateway_router = APIRouter(prefix="/gateway", tags=["gateway"])


async def _get_request_body(request: Request) -> dict:
    """
    Get request body, using cached version if available.

    The auth middleware may have already parsed the request body for permission
    validation. Since Starlette request body can only be read once, we cache
    the parsed body in request.state.cached_body for reuse by route handlers.

    Args:
        request: The FastAPI Request object.

    Returns:
        Parsed JSON body as a dictionary.

    Raises:
        HTTPException: If the request body is not valid JSON.
    """
    # Check if body was already parsed by auth middleware
    cached_body = getattr(request.state, "cached_body", None)
    if isinstance(cached_body, dict):
        return cached_body

    # Otherwise parse it now
    try:
        return await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e!s}")


def _record_gateway_invocation(invocation_type: GatewayInvocationType) -> Callable[..., Any]:
    """
    Decorator to record telemetry for gateway invocation endpoints.

    Automatically tracks success/failure status, duration, and streaming mode
    (determined by checking if the response is a StreamingResponse).

    Args:
        invocation_type: The type of invocation endpoint.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None

            try:
                result = await func(*args, **kwargs)
                return result  # noqa: RET504
            except Exception:
                success = False
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)
                _record_event(
                    GatewayInvocationEvent,
                    params={
                        "is_streaming": isinstance(result, StreamingResponse),
                        "invocation_type": invocation_type,
                    },
                    success=success,
                    duration_ms=duration_ms,
                )

        return wrapper

    return decorator


def _build_endpoint_config(
    endpoint_name: str,
    model_config: GatewayModelConfig,
    endpoint_type: EndpointType,
) -> EndpointConfig:
    """
    Build an EndpointConfig from model configuration.

    This function combines provider config building and endpoint config building
    into a single operation.

    Args:
        endpoint_name: The endpoint name.
        model_config: The model configuration object with decrypted secrets.
        endpoint_type: Endpoint type (chat or embeddings).

    Returns:
        EndpointConfig instance ready for provider instantiation.

    Raises:
        MlflowException: If provider configuration is invalid.
    """
    provider_config = None

    if model_config.provider == Provider.OPENAI:
        auth_config = model_config.auth_config or {}
        openai_config = {
            "openai_api_key": model_config.secret_value.get(_AuthConfigKey.API_KEY),
        }

        # Check if this is Azure OpenAI (requires api_type, deployment_name, api_base, api_version)
        if "api_type" in auth_config and auth_config["api_type"] in ("azure", "azuread"):
            openai_config["openai_api_type"] = auth_config["api_type"]
            openai_config["openai_api_base"] = auth_config.get(_AuthConfigKey.API_BASE)
            openai_config["openai_deployment_name"] = auth_config.get("deployment_name")
            openai_config["openai_api_version"] = auth_config.get("api_version")
        else:
            # Standard OpenAI
            if _AuthConfigKey.API_BASE in auth_config:
                openai_config["openai_api_base"] = auth_config[_AuthConfigKey.API_BASE]
            if "organization" in auth_config:
                openai_config["openai_organization"] = auth_config["organization"]

        provider_config = OpenAIConfig(**openai_config)
    elif model_config.provider == Provider.AZURE:
        auth_config = model_config.auth_config or {}
        model_config.provider = Provider.OPENAI
        provider_config = OpenAIConfig(
            openai_api_type=OpenAIAPIType.AZURE,
            openai_api_key=model_config.secret_value.get(_AuthConfigKey.API_KEY),
            openai_api_base=auth_config.get(_AuthConfigKey.API_BASE),
            openai_deployment_name=model_config.model_name,
            openai_api_version=auth_config.get("api_version"),
        )
    elif model_config.provider == Provider.ANTHROPIC:
        anthropic_config = {
            "anthropic_api_key": model_config.secret_value.get(_AuthConfigKey.API_KEY),
        }
        if model_config.auth_config and "version" in model_config.auth_config:
            anthropic_config["anthropic_version"] = model_config.auth_config["version"]
        provider_config = AnthropicConfig(**anthropic_config)
    elif model_config.provider == Provider.MISTRAL:
        provider_config = MistralConfig(
            mistral_api_key=model_config.secret_value.get(_AuthConfigKey.API_KEY),
        )
    elif model_config.provider == Provider.GEMINI:
        provider_config = GeminiConfig(
            gemini_api_key=model_config.secret_value.get(_AuthConfigKey.API_KEY),
        )
    else:
        # Use LiteLLM as fallback for unsupported providers
        # Store the original provider name for LiteLLM's provider/model format
        original_provider = model_config.provider
        auth_config = model_config.auth_config or {}
        # Merge auth_config with secret_value (secret_value contains api_key and other secrets)
        litellm_config = {
            "litellm_provider": original_provider,
            "litellm_auth_config": auth_config | model_config.secret_value,
        }
        provider_config = LiteLLMConfig(**litellm_config)
        model_config.provider = Provider.LITELLM

    # Build and return EndpointConfig
    return EndpointConfig(
        name=endpoint_name,
        endpoint_type=endpoint_type,
        model={
            "name": model_config.model_name,
            "provider": model_config.provider,
            "config": provider_config.model_dump(),
        },
    )


def _create_provider(
    endpoint_config: GatewayEndpointConfig,
    endpoint_type: EndpointType,
) -> BaseProvider:
    """
    Create a provider instance based on endpoint routing strategy.

    Fallback is independent of routing strategy - if fallback_config is present,
    the provider is wrapped with FallbackProvider.

    Args:
        endpoint_config: The endpoint configuration with model details and routing config.
        endpoint_type: Endpoint type (chat or embeddings).

    Returns:
        Provider instance (standard provider, TrafficRouteProvider, or FallbackProvider).

    Raises:
        MlflowException: If endpoint configuration is invalid or has no models.
    """
    # Get PRIMARY models
    primary_models = [
        model
        for model in endpoint_config.models
        if model.linkage_type == GatewayModelLinkageType.PRIMARY
    ]

    if not primary_models:
        raise MlflowException(
            f"Endpoint '{endpoint_config.endpoint_name}' has no PRIMARY models configured",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    # Create base provider based on routing strategy
    if endpoint_config.routing_strategy == RoutingStrategy.REQUEST_BASED_TRAFFIC_SPLIT:
        # Traffic split: distribute requests based on weights
        configs = []
        weights = []
        for model_config in primary_models:
            gateway_endpoint_config = _build_endpoint_config(
                endpoint_name=endpoint_config.endpoint_name,
                model_config=model_config,
                endpoint_type=endpoint_type,
            )
            configs.append(gateway_endpoint_config)
            weights.append(int(model_config.weight * 100))  # Convert to percentage

        primary_provider = TrafficRouteProvider(
            configs=configs,
            traffic_splits=weights,
            routing_strategy="TRAFFIC_SPLIT",
        )
    else:
        # Default: use the first PRIMARY model
        model_config = primary_models[0]
        gateway_endpoint_config = _build_endpoint_config(
            endpoint_config.endpoint_name, model_config, endpoint_type
        )
        provider_class = get_provider(model_config.provider)
        primary_provider = provider_class(gateway_endpoint_config)

    # Wrap with FallbackProvider if fallback configuration exists
    if endpoint_config.fallback_config:
        fallback_models = [
            model
            for model in endpoint_config.models
            if model.linkage_type == GatewayModelLinkageType.FALLBACK
        ]

        if not fallback_models:
            _logger.warning(
                f"Endpoint '{endpoint_config.endpoint_name}' has fallback_config "
                "but no FALLBACK models configured"
            )
            return primary_provider

        # Sort fallback models by fallback_order
        fallback_models.sort(
            key=lambda m: m.fallback_order if m.fallback_order is not None else float("inf")
        )

        fallback_providers = [
            get_provider(model_config.provider)(
                _build_endpoint_config(
                    endpoint_name=endpoint_config.endpoint_name,
                    model_config=model_config,
                    endpoint_type=endpoint_type,
                )
            )
            for model_config in fallback_models
        ]

        max_attempts = endpoint_config.fallback_config.max_attempts or len(fallback_models)

        # FallbackProvider expects all providers (primary + fallback)
        # We need to create a combined provider that tries primary first, then fallbacks
        all_providers = [primary_provider] + fallback_providers

        return FallbackProvider(
            providers=all_providers,
            max_attempts=max_attempts + 1,  # +1 to include primary
            strategy=endpoint_config.fallback_config.strategy,
        )

    return primary_provider


def _create_provider_from_endpoint_name(
    store: SqlAlchemyStore,
    endpoint_name: str,
    endpoint_type: EndpointType,
) -> BaseProvider:
    """
    Create a provider from an endpoint name (backward compatibility helper for tests).

    Args:
        store: The SQLAlchemy store instance.
        endpoint_name: The endpoint name.
        endpoint_type: Endpoint type (chat or embeddings).

    Returns:
        Provider instance
    """
    endpoint_config = get_endpoint_config(endpoint_name=endpoint_name, store=store)
    return _create_provider(endpoint_config, endpoint_type)


def _validate_store(store: AbstractStore) -> None:
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


@gateway_router.post("/{endpoint_name}/mlflow/invocations", response_model=None)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.MLFLOW_INVOCATIONS)
async def invocations(endpoint_name: str, request: Request):
    """
    Unified invocations endpoint handler that supports both chat and embeddings.

    The handler automatically detects the request type based on the payload structure:
    - If payload has "messages" field -> chat endpoint
    - If payload has "input" field -> embeddings endpoint
    """
    body = await _get_request_body(request)

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


@gateway_router.post("/mlflow/v1/chat/completions", response_model=None)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.MLFLOW_CHAT_COMPLETIONS)
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
    body = await _get_request_body(request)

    # Extract endpoint name from "model" parameter
    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")

    store = _get_store()

    _validate_store(store)

    try:
        payload = chat.RequestPayload(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid chat payload: {e!s}")

    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)

    if payload.stream:
        return await make_streaming_response(provider.chat_stream(payload))
    else:
        return await provider.chat(payload)


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_CHAT], response_model=None)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.OPENAI_PASSTHROUGH_CHAT)
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
    body = await _get_request_body(request)

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    headers = dict(request.headers)
    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.OPENAI_CHAT, body, headers)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_EMBEDDINGS], response_model=None)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.OPENAI_PASSTHROUGH_EMBEDDINGS)
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
    body = await _get_request_body(request)

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    headers = dict(request.headers)
    provider = _create_provider_from_endpoint_name(
        store, endpoint_name, EndpointType.LLM_V1_EMBEDDINGS
    )
    return await provider.passthrough(PassthroughAction.OPENAI_EMBEDDINGS, body, headers)


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_RESPONSES], response_model=None)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.OPENAI_PASSTHROUGH_RESPONSES)
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
    body = await _get_request_body(request)

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    headers = dict(request.headers)
    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.OPENAI_RESPONSES, body, headers)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.ANTHROPIC_MESSAGES], response_model=None)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.ANTHROPIC_PASSTHROUGH_MESSAGES)
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
    body = await _get_request_body(request)

    endpoint_name = _extract_endpoint_name_from_model(body)
    body.pop("model")
    store = _get_store()
    _validate_store(store)

    headers = dict(request.headers)
    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, body, headers)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(
    PASSTHROUGH_ROUTES[PassthroughAction.GEMINI_GENERATE_CONTENT], response_model=None
)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.GEMINI_PASSTHROUGH_GENERATE_CONTENT)
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
    body = await _get_request_body(request)

    store = _get_store()
    _validate_store(store)

    headers = dict(request.headers)
    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    return await provider.passthrough(PassthroughAction.GEMINI_GENERATE_CONTENT, body, headers)


@gateway_router.post(
    PASSTHROUGH_ROUTES[PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT], response_model=None
)
@translate_http_exception
@_record_gateway_invocation(GatewayInvocationType.GEMINI_PASSTHROUGH_STREAM_GENERATE_CONTENT)
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
    body = await _get_request_body(request)

    store = _get_store()
    _validate_store(store)

    headers = dict(request.headers)
    provider = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(
        PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT, body, headers
    )
    return StreamingResponse(response, media_type="text/event-stream")
