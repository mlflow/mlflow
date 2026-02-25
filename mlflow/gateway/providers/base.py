from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterable

import numpy as np

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.gateway_endpoint import FallbackStrategy
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.utils.annotations import developer_stable


class PassthroughAction(str, Enum):
    """
    Enum for passthrough endpoint actions.
    """

    OPENAI_CHAT = "openai_chat"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE_CONTENT = "gemini_generate_content"
    GEMINI_STREAM_GENERATE_CONTENT = "gemini_stream_generate_content"


# Mapping of passthrough actions to their gateway API routes
PASSTHROUGH_ROUTES = {
    PassthroughAction.OPENAI_CHAT: "/openai/v1/chat/completions",
    PassthroughAction.OPENAI_EMBEDDINGS: "/openai/v1/embeddings",
    PassthroughAction.OPENAI_RESPONSES: "/openai/v1/responses",
    PassthroughAction.ANTHROPIC_MESSAGES: "/anthropic/v1/messages",
    PassthroughAction.GEMINI_GENERATE_CONTENT: "/gemini/v1beta/models/{endpoint_name}:generateContent",  # noqa: E501
    PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT: "/gemini/v1beta/models/{endpoint_name}:streamGenerateContent",  # noqa: E501
}


def _get_nested(d: dict[str, Any], key: str) -> Any:
    """Look up a value by key, supporting one level of nesting."""
    match key.split("."):
        case [parent, child]:
            match d.get(parent):
                case dict(nested) if child in nested:
                    return nested[child]
                case _:
                    return None
        case _:
            return d.get(key)


@developer_stable
class BaseProvider(ABC):
    """
    Base class for MLflow Gateway providers.

    Args:
        config: The endpoint configuration.
        enable_tracing: If True, wraps method calls with MLflow tracing spans.
    """

    NAME: str = ""
    SUPPORTED_ROUTE_TYPES: tuple[str, ...]
    CONFIG_TYPE: type[ConfigModel]
    PASSTHROUGH_PROVIDER_PATHS: dict[PassthroughAction, str] = {}

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False):
        if self.NAME == "":
            raise ValueError(
                f"{self.__class__.__name__} is a subclass of BaseProvider and must "
                f"override 'NAME' attribute as a non-empty string."
            )

        if not hasattr(self, "CONFIG_TYPE") or not issubclass(self.CONFIG_TYPE, ConfigModel):
            raise ValueError(
                f"{self.__class__.__name__} is a subclass of BaseProvider and must "
                f"override 'CONFIG_TYPE' attribute as a subclass of ConfigModel."
            )

        self.config = config
        self._enable_tracing = enable_tracing

    # -------------------------------------------------------------------------
    # Internal implementation methods (override these in subclasses)
    # -------------------------------------------------------------------------

    async def _chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        raise AIGatewayException(
            status_code=501,
            detail=f"The chat streaming route is not implemented for {self.NAME} models.",
        )

    async def _chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise AIGatewayException(
            status_code=501,
            detail=f"The chat route is not implemented for {self.NAME} models.",
        )

    async def _completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        raise AIGatewayException(
            status_code=501,
            detail=f"The completions streaming route is not implemented for {self.NAME} models.",
        )

    async def _completions(
        self, payload: completions.RequestPayload
    ) -> completions.ResponsePayload:
        raise AIGatewayException(
            status_code=501,
            detail=f"The completions route is not implemented for {self.NAME} models.",
        )

    async def _embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise AIGatewayException(
            status_code=501,
            detail=f"The embeddings route is not implemented for {self.NAME} models.",
        )

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        route = PASSTHROUGH_ROUTES.get(action)
        raise AIGatewayException(
            status_code=501,
            detail=f"The passthrough route '{route}' is not implemented for {self.NAME} models.",
        )

    # -------------------------------------------------------------------------
    # Public methods (with optional tracing)
    # -------------------------------------------------------------------------

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        async for chunk in self._maybe_trace_stream_method(
            "chat_stream", self._chat_stream, payload
        ):
            yield chunk

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        return await self._maybe_trace_method("chat", self._chat, payload)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        async for chunk in self._maybe_trace_stream_method(
            "completions_stream", self._completions_stream, payload
        ):
            yield chunk

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        return await self._maybe_trace_method("completions", self._completions, payload)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return await self._maybe_trace_method("embeddings", self._embeddings, payload)

    async def passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        if not self._enable_tracing:
            return await self._passthrough(action, payload, headers)

        try:
            result = await self._passthrough(action, payload, headers)
            if isinstance(result, AsyncIterable):

                @mlflow.trace(span_type=SpanType.LLM, name=self._get_span_name())
                async def passthrough():
                    span = mlflow.get_current_active_span()
                    if span is not None:
                        span.set_attributes(
                            {**self._get_provider_attributes(), "action": action.value}
                        )
                    async for chunk in result:
                        yield chunk

                return passthrough()
            else:

                @mlflow.trace(span_type=SpanType.LLM, name=self._get_span_name())
                async def passthrough():
                    span = mlflow.get_current_active_span()
                    if span is not None:
                        span.set_attributes(
                            {**self._get_provider_attributes(), "action": action.value}
                        )
                    if span is not None:
                        if token_usage := self._extract_passthrough_token_usage(action, result):
                            span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)
                    return result

                return await passthrough()
        except Exception as e:
            with mlflow.start_span(span_type=SpanType.LLM, name=self._get_span_name()) as span:
                span.set_attributes({**self._get_provider_attributes(), "action": action.value})
                raise e

    # -------------------------------------------------------------------------
    # Tracing helper methods
    # -------------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """
        Get the provider name for tracing and metrics.

        Override this method to return a different provider name than the class NAME.
        For example, LiteLLM provider overrides this to return the actual underlying
        provider name (e.g., "anthropic", "openai") instead of "litellm".

        Returns:
            The provider name string.
        """
        return self.NAME

    def _get_span_name(self) -> str:
        """Generate span name based on provider and model."""
        provider_name = self.get_provider_name()
        model_name = ""
        if hasattr(self, "config") and hasattr(self.config, "model"):
            model_name = getattr(self.config.model, "name", "")

        span_name = f"provider/{provider_name}"
        if model_name:
            span_name = f"{span_name}/{model_name}"
        return span_name

    def _get_provider_attributes(self) -> dict[str, str]:
        """Get provider attributes for span."""
        attrs = {
            SpanAttributeKey.MODEL_PROVIDER: self.get_provider_name(),
        }
        if hasattr(self, "config") and hasattr(self.config, "model"):
            if model_name := getattr(self.config.model, "name", ""):
                attrs[SpanAttributeKey.MODEL] = model_name
        return attrs

    def _extract_token_usage(self, result) -> dict[str, int] | None:
        """Extract token usage from a response object if available."""
        if not hasattr(result, "usage") or result.usage is None:
            return None

        usage = result.usage
        token_usage = {}
        if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
            token_usage[TokenUsageKey.INPUT_TOKENS] = prompt_tokens
        if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
            token_usage[TokenUsageKey.OUTPUT_TOKENS] = completion_tokens
        if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
            token_usage[TokenUsageKey.TOTAL_TOKENS] = total_tokens

        # Extract cached token details if available
        if (cached := getattr(usage, "cache_read_input_tokens", None)) is not None:
            token_usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
        elif details := getattr(usage, "prompt_tokens_details", None):
            if (cached := getattr(details, "cached_tokens", None)) is not None:
                token_usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached

        if (created := getattr(usage, "cache_creation_input_tokens", None)) is not None:
            token_usage[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created

        return token_usage or None

    def _extract_passthrough_token_usage(
        self, action: PassthroughAction, result: dict[str, Any]
    ) -> dict[str, int] | None:
        """
        Extract token usage from a passthrough response dictionary.

        Override this method in provider subclasses to handle provider-specific
        response formats for passthrough endpoints.

        Args:
            action: The passthrough action that was performed.
            result: The raw response dictionary from the provider API.

        Returns:
            A dictionary with token usage keys (input_tokens, output_tokens, total_tokens)
            or None if usage information is not available.
        """
        return None

    @staticmethod
    def _extract_token_usage_from_dict(
        usage_dict: dict[str, Any] | None,
        input_tokens_key: str,
        output_tokens_key: str,
        total_tokens_key: str | None = None,
        cache_read_key: str | None = None,
        cache_creation_key: str | None = None,
    ) -> dict[str, int] | None:
        """
        Extract token usage from a dictionary with configurable key names.

        This is a helper method to reduce code duplication across providers.
        Each provider uses different key names for token usage, but the extraction
        logic is the same.

        Args:
            usage_dict: The dictionary containing token usage information.
            input_tokens_key: Key name for input/prompt tokens (e.g., "input_tokens",
                "prompt_tokens", "promptTokenCount").
            output_tokens_key: Key name for output/completion tokens (e.g., "output_tokens",
                "completion_tokens", "candidatesTokenCount").
            total_tokens_key: Optional key name for total tokens. If None or not present
                in usage_dict, total will be calculated from input + output.
            cache_read_key: Optional key for cache read tokens. Supports "key1.key2"
                notation for nested dicts (e.g., "prompt_tokens_details.cached_tokens").
            cache_creation_key: Optional key for cache creation tokens. Supports
                "key1.key2" notation for nested dicts.

        Returns:
            A dictionary with normalized token usage keys (input_tokens, output_tokens,
            total_tokens) or None if usage_dict is None or empty.
        """
        if not usage_dict:
            return None

        token_usage = {}

        if (input_tokens := usage_dict.get(input_tokens_key)) is not None:
            token_usage[TokenUsageKey.INPUT_TOKENS] = input_tokens
        if (output_tokens := usage_dict.get(output_tokens_key)) is not None:
            token_usage[TokenUsageKey.OUTPUT_TOKENS] = output_tokens

        if total_tokens_key and (total_tokens := usage_dict.get(total_tokens_key)) is not None:
            token_usage[TokenUsageKey.TOTAL_TOKENS] = total_tokens
        elif (
            TokenUsageKey.INPUT_TOKENS in token_usage and TokenUsageKey.OUTPUT_TOKENS in token_usage
        ):
            token_usage[TokenUsageKey.TOTAL_TOKENS] = (
                token_usage[TokenUsageKey.INPUT_TOKENS] + token_usage[TokenUsageKey.OUTPUT_TOKENS]
            )

        if token_usage:
            if cache_read_key is not None:
                if (cached := _get_nested(usage_dict, cache_read_key)) is not None:
                    token_usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
            if cache_creation_key is not None:
                if (created := _get_nested(usage_dict, cache_creation_key)) is not None:
                    token_usage[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created

        return token_usage or None

    def _set_span_token_usage(self, token_usage: dict[str, int]) -> None:
        """
        Set token usage on the current active span if tracing is enabled.

        This is a helper for providers to call at the end of streaming passthrough.
        """
        if self._enable_tracing and (span := mlflow.get_current_active_span()) and token_usage:
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

    async def _maybe_trace_method(self, method_name: str, method, *args, **kwargs):
        """Execute a method with optional tracing span based on _enable_tracing."""
        if not self._enable_tracing:
            return await method(*args, **kwargs)

        active_span = mlflow.get_current_active_span()
        if active_span is None:
            return await method(*args, **kwargs)

        span_name = self._get_span_name()
        with mlflow.start_span(span_type=SpanType.LLM, name=span_name) as span:
            span.set_attributes({**self._get_provider_attributes(), "method": method_name})

            result = await method(*args, **kwargs)

            if token_usage := self._extract_token_usage(result):
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

            span.set_status("OK")
            return result

    async def _maybe_trace_stream_method(self, method_name: str, method, *args, **kwargs):
        """Execute a streaming method with optional tracing span based on _enable_tracing."""
        if not self._enable_tracing:
            async for chunk in method(*args, **kwargs):
                yield chunk
            return

        active_span = mlflow.get_current_active_span()
        if active_span is None:
            async for chunk in method(*args, **kwargs):
                yield chunk
            return

        span_name = self._get_span_name()
        # Use start_span_no_context to get a LiveSpan that can be manually ended
        span = start_span_no_context(
            name=span_name,
            span_type=SpanType.LLM,
            parent_span=active_span,
            attributes={
                **self._get_provider_attributes(),
                "method": method_name,
                "streaming": True,
            },
        )

        try:
            last_chunk = None
            async for chunk in method(*args, **kwargs):
                last_chunk = chunk
                yield chunk

            # Extract usage from the final chunk if available (OpenAI includes this
            # when stream_options.include_usage=true)
            if last_chunk is not None:
                if token_usage := self._extract_token_usage(last_chunk):
                    span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

            span.set_status("OK")
            span.end()
        except Exception as e:
            span.record_exception(e)
            span.end()
            raise

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def _validate_passthrough_action(self, action: PassthroughAction) -> str:
        """
        Validates that the passthrough action is supported by this provider
        and returns the provider path.

        Args:
            action: The passthrough action to validate

        Returns:
            The provider path for the action
        """
        provider_path = self.PASSTHROUGH_PROVIDER_PATHS.get(action)
        if provider_path is None:
            requested_route = PASSTHROUGH_ROUTES.get(action, action.value)
            supported_routes = ", ".join(
                f"/gateway{supported_route} (provider_path: {path})"
                for act in self.PASSTHROUGH_PROVIDER_PATHS.keys()
                if (supported_route := PASSTHROUGH_ROUTES.get(act))
                and (path := self.PASSTHROUGH_PROVIDER_PATHS.get(act))
            )
            raise AIGatewayException(
                status_code=400,
                detail="Unsupported passthrough endpoint "
                f"'{requested_route}' for {self.NAME} provider. "
                f"Supported endpoints: {supported_routes}",
            )
        return provider_path

    async def _stream_passthrough_with_usage(
        self, stream: AsyncIterable[Any]
    ) -> AsyncIterable[Any]:
        """Stream passthrough response while accumulating token usage."""
        accumulated_usage: dict[str, int] = {}
        try:
            async for chunk in stream:
                chunk_usage = self._extract_streaming_token_usage(chunk)
                accumulated_usage.update(chunk_usage)
                yield chunk
        finally:
            # Calculate total if we have input and output but no total
            if (
                TokenUsageKey.INPUT_TOKENS in accumulated_usage
                and TokenUsageKey.OUTPUT_TOKENS in accumulated_usage
                and TokenUsageKey.TOTAL_TOKENS not in accumulated_usage
            ):
                accumulated_usage[TokenUsageKey.TOTAL_TOKENS] = (
                    accumulated_usage[TokenUsageKey.INPUT_TOKENS]
                    + accumulated_usage[TokenUsageKey.OUTPUT_TOKENS]
                )
            self._set_span_token_usage(accumulated_usage)

    def _extract_streaming_token_usage(self, chunk: Any) -> dict[str, int]:
        """Extract token usage from a streaming chunk.

        Override this method in provider subclasses to handle provider-specific
        streaming formats for passthrough endpoints.

        Returns:
            A dictionary with token usage keys found in this chunk.
            May be partial (e.g., only input_tokens or only output_tokens).
        """
        return {}

    @staticmethod
    def check_for_model_field(payload):
        if "model" in payload:
            raise AIGatewayException(
                status_code=422,
                detail="The parameter 'model' is not permitted to be passed. The route being "
                "queried already defines a model instance.",
            )


class TrafficRouteProvider(BaseProvider):
    """
    A provider that split traffic and forward to multiple providers
    """

    NAME: str = "TrafficRoute"

    def __init__(
        self,
        configs: list[EndpointConfig],
        traffic_splits: list[int],
        routing_strategy: str,
        enable_tracing: bool = False,
    ):
        from mlflow.gateway.providers import get_provider

        if len(configs) != len(traffic_splits):
            raise MlflowException.invalid_parameter_value(
                "'configs' and 'traffic_splits' should have the same length."
            )

        if routing_strategy != "TRAFFIC_SPLIT":
            raise MlflowException.invalid_parameter_value(
                "'routing_strategy' must be 'TRAFFIC_SPLIT'."
            )

        self._providers = [
            get_provider(config.model.provider)(config, enable_tracing=enable_tracing)
            for config in configs
        ]
        # Normalize the weights to sum to 1
        self._weights = np.array(traffic_splits, dtype=np.float32) / np.sum(traffic_splits)
        self._indices = np.arange(len(self._providers))
        self._enable_tracing = enable_tracing

    def _get_provider(self):
        chosen_index = np.random.choice(self._indices, p=self._weights)
        return self._providers[chosen_index]

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        prov = self._get_provider()
        async for i in prov.chat_stream(payload):
            yield i

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        prov = self._get_provider()
        return await prov.chat(payload)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        prov = self._get_provider()
        async for i in prov.completions_stream(payload):
            yield i

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        prov = self._get_provider()
        return await prov.completions(payload)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        prov = self._get_provider()
        return await prov.embeddings(payload)

    async def passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        prov = self._get_provider()
        return await prov.passthrough(action, payload, headers)


class FallbackProvider(BaseProvider):
    """
    A provider that implements fallback routing across multiple providers.

    Attempts to call providers in order until one succeeds or max_attempts is reached.
    """

    NAME: str = "Fallback"

    def __init__(
        self,
        providers: list[BaseProvider],
        strategy: FallbackStrategy | None = None,
        max_attempts: int | None = None,
        enable_tracing: bool = False,
    ):
        if not providers:
            raise MlflowException.invalid_parameter_value(
                "'providers' must contain at least one provider."
            )

        self._providers = providers
        self._enable_tracing = enable_tracing

        max_attempts = max_attempts if max_attempts is not None else len(self._providers)
        self._max_attempts = min(max_attempts, len(self._providers))
        self._strategy = strategy

    async def _execute_with_fallback(self, method_name: str, *args, **kwargs):
        """
        Execute a method on providers with fallback logic.

        Args:
            method_name: Name of the method to call on each provider
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result from the first successful provider call

        Raises:
            AIGatewayException: If all fallback attempts fail, with status code
                propagated from the last exception if it was an AIGatewayException
                or HTTPException
        """
        from fastapi import HTTPException

        last_error = None

        for attempt, provider in enumerate(self._providers[: self._max_attempts], 1):
            try:
                method = getattr(provider, method_name)
                return await method(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self._max_attempts:
                    continue
                break

        # Propagate HTTP status code from the last exception if available
        status_code = 500
        if isinstance(last_error, (AIGatewayException, HTTPException)):
            status_code = last_error.status_code

        raise AIGatewayException(
            status_code=status_code,
            detail=f"All {self._max_attempts} fallback attempts failed. Last error: {last_error!s}",
        )

    async def _execute_stream_with_fallback(self, method_name: str, *args, **kwargs):
        """
        Execute a streaming method on providers with fallback logic.

        Args:
            method_name: Name of the streaming method to call on each provider
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Yields:
            Stream chunks from the first successful provider call

        Raises:
            AIGatewayException: If all fallback attempts fail, with status code
                propagated from the last exception if it was an AIGatewayException
                or HTTPException
        """
        from fastapi import HTTPException

        last_error = None

        for attempt, provider in enumerate(self._providers[: self._max_attempts], 1):
            try:
                method = getattr(provider, method_name)
                async for chunk in method(*args, **kwargs):
                    yield chunk
                # Stream completed successfully
                return
            except Exception as e:
                last_error = e
                if attempt < self._max_attempts:
                    continue
                break

        # Propagate HTTP status code from the last exception if available
        status_code = 500
        if isinstance(last_error, (AIGatewayException, HTTPException)):
            status_code = last_error.status_code

        raise AIGatewayException(
            status_code=status_code,
            detail=f"All {self._max_attempts} fallback attempts failed. Last error: {last_error!s}",
        )

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        async for chunk in self._execute_stream_with_fallback("chat_stream", payload):
            yield chunk

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        return await self._execute_with_fallback("chat", payload)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        async for chunk in self._execute_stream_with_fallback("completions_stream", payload):
            yield chunk

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        return await self._execute_with_fallback("completions", payload)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return await self._execute_with_fallback("embeddings", payload)

    async def passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        return await self._execute_with_fallback("passthrough", action, payload, headers)


class ProviderAdapter(ABC):
    @classmethod
    @abstractmethod
    def model_to_embeddings(cls, resp, config): ...

    @classmethod
    @abstractmethod
    def model_to_completions(cls, resp, config): ...

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def completions_to_model(cls, payload, config): ...

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def chat_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def embeddings_to_model(cls, payload, config): ...

    @classmethod
    def check_keys_against_mapping(cls, mapping, payload):
        for k1, k2 in mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
