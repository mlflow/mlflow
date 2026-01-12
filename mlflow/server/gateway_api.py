"""
Database-backed Gateway API endpoints for MLflow Server.

This module provides dynamic gateway endpoints that are configured from the database
rather than from a static YAML configuration file. It integrates the AI Gateway
functionality directly into the MLflow tracking server.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from mlflow.entities.gateway_endpoint import GatewayModelLinkageType
from mlflow.entities.gateway_usage import InvocationStatus, ProviderCallInput, ProviderCallStatus
from mlflow.exceptions import MlflowException
from mlflow.gateway.rate_limiter import get_rate_limiter
from mlflow.gateway.config import (
    AnthropicConfig,
    EndpointConfig,
    EndpointType,
    GeminiConfig,
    LiteLLMConfig,
    MistralConfig,
    OpenAIConfig,
    Provider,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.providers.base import (
    PASSTHROUGH_ROUTES,
    BaseProvider,
    FallbackProvider,
    ProviderCallResult,
    PassthroughAction,
    ProviderCallAttempt,
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
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

gateway_router = APIRouter(prefix="/gateway", tags=["gateway"])


def _check_rate_limit(
    store: AbstractStore,
    endpoint_id: str,
    request: Request,
) -> None:
    """
    Check rate limit for a request and raise HTTPException if exceeded.

    Args:
        store: The tracking store.
        endpoint_id: The endpoint ID being called.
        request: The FastAPI request object (used to extract username).

    Raises:
        HTTPException: With status 429 if rate limit is exceeded.
    """
    # Extract username from request headers if available
    # Common patterns: X-User-Id, X-Username, Authorization header parsing
    username = request.headers.get("X-Username") or request.headers.get("X-User-Id")

    rate_limiter = get_rate_limiter()
    allowed, limit, remaining = rate_limiter.check_rate_limit(store, endpoint_id, username)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded: {limit} queries per minute",
                "limit": limit,
                "remaining": 0,
                "retry_after": 60,
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": "60",
            },
        )


def _extract_usage_from_response(response: Any) -> dict[str, int]:
    """Extract token usage from a provider response."""
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if response is None:
        return usage

    usage_obj = None
    if hasattr(response, "usage") and response.usage is not None:
        usage_obj = response.usage
    elif isinstance(response, dict) and "usage" in response:
        usage_obj = response["usage"]

    if usage_obj is not None:
        if hasattr(usage_obj, "prompt_tokens"):
            usage["prompt_tokens"] = usage_obj.prompt_tokens or 0
            usage["completion_tokens"] = usage_obj.completion_tokens or 0
            usage["total_tokens"] = usage_obj.total_tokens or 0
        elif isinstance(usage_obj, dict):
            usage["prompt_tokens"] = usage_obj.get("prompt_tokens", 0) or 0
            usage["completion_tokens"] = usage_obj.get("completion_tokens", 0) or 0
            usage["total_tokens"] = usage_obj.get("total_tokens", 0) or 0

    return usage


def _calculate_cost(
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, float]:
    """Calculate the cost of a provider call using litellm if available."""
    cost = {
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "total_cost": 0.0,
    }

    try:
        from litellm import cost_per_token

        provider_model = f"{provider}/{model}" if provider else model
        prompt_cost, completion_cost = cost_per_token(
            model=provider_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        cost["prompt_cost"] = prompt_cost
        cost["completion_cost"] = completion_cost
        cost["total_cost"] = prompt_cost + completion_cost
    except ImportError:
        _logger.debug("litellm not installed, skipping cost calculation")
    except Exception as e:
        _logger.debug(f"Failed to calculate cost: {e}")

    return cost


def _create_provider_call_input(
    attempt: ProviderCallAttempt,
) -> ProviderCallInput:
    """Create ProviderCallInput from a ProviderCallAttempt, extracting usage and cost."""
    if attempt.success and attempt.response is not None:
        usage = _extract_usage_from_response(attempt.response)
        cost = _calculate_cost(
            model=attempt.model_name,
            provider=attempt.provider_name,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
    else:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = {"prompt_cost": 0.0, "completion_cost": 0.0, "total_cost": 0.0}

    return ProviderCallInput(
        provider=attempt.provider_name,
        model_name=attempt.model_name,
        attempt_number=attempt.attempt_number,
        status=ProviderCallStatus.SUCCESS if attempt.success else ProviderCallStatus.ERROR,
        latency_ms=attempt.latency_ms,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        prompt_cost=cost["prompt_cost"],
        completion_cost=cost["completion_cost"],
        total_cost=cost["total_cost"],
        error_message=attempt.error_message,
    )


def _log_invocation(
    store: SqlAlchemyStore,
    endpoint_id: str,
    endpoint_type: str,
    fallback_result: ProviderCallResult,
    total_latency_ms: int,
) -> None:
    """
    Log a gateway invocation with all provider attempts to the tracking store.

    This unified function handles both single-provider and fallback scenarios.
    All provider call attempts (including failed ones) are logged for metrics tracking.

    Args:
        store: The SQLAlchemy store instance.
        endpoint_id: ID of the gateway endpoint.
        endpoint_type: Type of the endpoint (e.g., "llm/v1/chat").
        fallback_result: The ProviderCallResult containing all provider attempts.
        total_latency_ms: Total latency in milliseconds for the entire invocation.
    """
    try:
        provider_calls = [
            _create_provider_call_input(attempt)
            for attempt in fallback_result.attempts
        ]

        # Determine overall status
        if fallback_result.success:
            failed_attempts = sum(1 for a in fallback_result.attempts if not a.success)
            status = InvocationStatus.PARTIAL if failed_attempts > 0 else InvocationStatus.SUCCESS
            error_message = None
        else:
            status = InvocationStatus.ERROR
            error_messages = [
                a.error_message for a in fallback_result.attempts if a.error_message
            ]
            error_message = (
                "; ".join(error_messages) if error_messages else "All attempts failed"
            )

        store.log_gateway_invocation(
            endpoint_id=endpoint_id,
            endpoint_type=endpoint_type,
            status=status,
            provider_calls=provider_calls,
            total_latency_ms=total_latency_ms,
            username=None,  # TODO: Extract from request context if available
            error_message=error_message,
        )
    except Exception as e:
        _logger.warning(f"Failed to log gateway invocation: {e}")


def _create_single_attempt_result(
    provider_name: str,
    model_name: str,
    response: Any,
    latency_ms: int,
    success: bool,
    error_message: str | None = None,
) -> ProviderCallResult:
    """
    Create a ProviderCallResult for a single provider call (non-fallback scenario).

    This allows unified logging for both single-provider and fallback scenarios.
    """
    attempt = ProviderCallAttempt(
        provider_name=provider_name,
        model_name=model_name,
        attempt_number=1,
        success=success,
        latency_ms=latency_ms,
        response=response,
        error_message=error_message,
    )
    return ProviderCallResult(
        response=response,
        attempts=[attempt],
        success=success,
    )


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
) -> tuple[BaseProvider, str]:
    """
    Create a provider from an endpoint name.

    Args:
        store: The SQLAlchemy store instance.
        endpoint_name: The endpoint name.
        endpoint_type: Endpoint type (chat or embeddings).

    Returns:
        Tuple of (provider instance, endpoint_id)
    """
    endpoint_config = get_endpoint_config(endpoint_name=endpoint_name, store=store)
    return _create_provider(endpoint_config, endpoint_type), endpoint_config.endpoint_id


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


def _get_provider_info(provider: BaseProvider) -> tuple[str, str]:
    """Extract provider name and model name from a provider instance."""
    if hasattr(provider, "NAME"):
        provider_name = provider.NAME
    elif hasattr(provider, "config") and hasattr(provider.config, "model"):
        provider_name = provider.config.model.provider
    else:
        provider_name = "unknown"

    if hasattr(provider, "config") and hasattr(provider.config, "model"):
        model_name = provider.config.model.name
    else:
        model_name = "unknown"

    return provider_name, model_name


@gateway_router.post("/{endpoint_name}/mlflow/invocations", response_model=None)
@translate_http_exception
async def invocations(endpoint_name: str, request: Request):
    """
    Unified invocations endpoint handler that supports both chat and embeddings.

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

        provider, endpoint_id = _create_provider_from_endpoint_name(store, endpoint_name, endpoint_type)
        provider_name, model_name = _get_provider_info(provider)

        # Check rate limit before processing
        _check_rate_limit(store, endpoint_id, request)

        if payload.stream:
            # For streaming, we can't easily track usage as tokens come incrementally
            return await make_streaming_response(provider.chat_stream(payload))
        else:
            start_time = time.time()
            try:
                # FallbackProvider returns (response, fallback_result) tuple
                if isinstance(provider, FallbackProvider):
                    response, fallback_result = await provider.chat(payload)
                else:
                    response = await provider.chat(payload)
                    latency_ms = int((time.time() - start_time) * 1000)
                    fallback_result = _create_single_attempt_result(
                        provider_name, model_name, response, latency_ms, success=True
                    )
                latency_ms = int((time.time() - start_time) * 1000)
                _log_invocation(
                    store, endpoint_id, endpoint_type.value, fallback_result, latency_ms
                )
                return response
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                # Check if exception has fallback_result attached (from FallbackProvider)
                fallback_result = getattr(e, "fallback_result", None)
                if not fallback_result:
                    fallback_result = _create_single_attempt_result(
                        provider_name, model_name, None, latency_ms,
                        success=False, error_message=str(e)
                    )
                _log_invocation(
                    store, endpoint_id, endpoint_type.value, fallback_result, latency_ms
                )
                raise

    elif "input" in body:
        # Embeddings request
        endpoint_type = EndpointType.LLM_V1_EMBEDDINGS
        try:
            payload = embeddings.RequestPayload(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid embeddings payload: {e!s}")

        provider, endpoint_id = _create_provider_from_endpoint_name(store, endpoint_name, endpoint_type)
        provider_name, model_name = _get_provider_info(provider)

        # Check rate limit before processing
        _check_rate_limit(store, endpoint_id, request)

        start_time = time.time()
        try:
            # FallbackProvider returns (response, fallback_result) tuple
            if isinstance(provider, FallbackProvider):
                response, fallback_result = await provider.embeddings(payload)
            else:
                response = await provider.embeddings(payload)
                latency_ms = int((time.time() - start_time) * 1000)
                fallback_result = _create_single_attempt_result(
                    provider_name, model_name, response, latency_ms, success=True
                )
            latency_ms = int((time.time() - start_time) * 1000)
            _log_invocation(
                store, endpoint_id, endpoint_type.value, fallback_result, latency_ms
            )
            return response
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            # Check if exception has fallback_result attached (from FallbackProvider)
            fallback_result = getattr(e, "fallback_result", None)
            if not fallback_result:
                fallback_result = _create_single_attempt_result(
                    provider_name, model_name, None, latency_ms,
                    success=False, error_message=str(e)
                )
            _log_invocation(
                store, endpoint_id, endpoint_type.value, fallback_result, latency_ms
            )
            raise

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: payload format must be either chat or embeddings",
        )


@gateway_router.post("/mlflow/v1/chat/completions", response_model=None)
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

    provider, endpoint_id = _create_provider_from_endpoint_name(store, endpoint_name, endpoint_type)
    provider_name, model_name = _get_provider_info(provider)

    # Check rate limit before processing
    _check_rate_limit(store, endpoint_id, request)

    if payload.stream:
        # For streaming, we can't easily track usage as tokens come incrementally
        return await make_streaming_response(provider.chat_stream(payload))
    else:
        start_time = time.time()
        try:
            # FallbackProvider returns (response, fallback_result) tuple
            if isinstance(provider, FallbackProvider):
                response, fallback_result = await provider.chat(payload)
            else:
                response = await provider.chat(payload)
                latency_ms = int((time.time() - start_time) * 1000)
                fallback_result = _create_single_attempt_result(
                    provider_name, model_name, response, latency_ms, success=True
                )
            latency_ms = int((time.time() - start_time) * 1000)
            _log_invocation(
                store, endpoint_id, endpoint_type.value, fallback_result, latency_ms
            )
            return response
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            # Check if exception has fallback_result attached (from FallbackProvider)
            fallback_result = getattr(e, "fallback_result", None)
            if not fallback_result:
                fallback_result = _create_single_attempt_result(
                    provider_name, model_name, None, latency_ms,
                    success=False, error_message=str(e)
                )
            _log_invocation(
                store, endpoint_id, endpoint_type.value, fallback_result, latency_ms
            )
            raise


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_CHAT], response_model=None)
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

    headers = dict(request.headers)
    provider, _ = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.OPENAI_CHAT, body, headers)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_EMBEDDINGS], response_model=None)
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

    headers = dict(request.headers)
    provider, _ = _create_provider_from_endpoint_name(
        store, endpoint_name, EndpointType.LLM_V1_EMBEDDINGS
    )
    return await provider.passthrough(PassthroughAction.OPENAI_EMBEDDINGS, body, headers)


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.OPENAI_RESPONSES], response_model=None)
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

    headers = dict(request.headers)
    provider, _ = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.OPENAI_RESPONSES, body, headers)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(PASSTHROUGH_ROUTES[PassthroughAction.ANTHROPIC_MESSAGES], response_model=None)
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

    headers = dict(request.headers)
    provider, _ = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, body, headers)

    if body.get("stream"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@gateway_router.post(
    PASSTHROUGH_ROUTES[PassthroughAction.GEMINI_GENERATE_CONTENT], response_model=None
)
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

    headers = dict(request.headers)
    provider, _ = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    return await provider.passthrough(PassthroughAction.GEMINI_GENERATE_CONTENT, body, headers)


@gateway_router.post(
    PASSTHROUGH_ROUTES[PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT], response_model=None
)
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

    headers = dict(request.headers)
    provider, _ = _create_provider_from_endpoint_name(store, endpoint_name, EndpointType.LLM_V1_CHAT)
    response = await provider.passthrough(
        PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT, body, headers
    )
    return StreamingResponse(response, media_type="text/event-stream")
