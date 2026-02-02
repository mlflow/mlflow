from typing import Any, AsyncIterable

import mlflow
from mlflow.gateway.providers.base import BaseProvider, PassthroughAction
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.fluent import start_span_no_context


class TracingProviderWrapper(BaseProvider):
    """
    A wrapper provider that adds MLflow tracing spans to all provider method calls.

    This wrapper automatically instruments any provider (including FallbackProvider
    and TrafficRouteProvider) with tracing spans for each method invocation.

    Usage:
        provider = TracingProviderWrapper(original_provider)
        result = await provider.chat(payload)  # Automatically traced

    The wrapper creates spans with:
        - Provider name and model information
        - Method name being called
        - Success/error status
        - Error details on failure
    """

    NAME: str = "TracingWrapper"

    def __init__(self, provider: BaseProvider):
        self._provider = provider
        # Expose underlying provider attributes for compatibility
        if hasattr(provider, "config"):
            self.config = provider.config

    @property
    def wrapped_provider(self) -> BaseProvider:
        """Access the underlying wrapped provider."""
        return self._provider

    def _get_provider_name(self) -> str:
        """Get the provider name"""
        if hasattr(self._provider, "get_provider_name"):
            return self._provider.get_provider_name()
        return self._provider.NAME

    def _get_span_name(self) -> str:
        """Generate span name based on wrapped provider."""
        provider_name = self._get_provider_name()
        model_name = ""
        if hasattr(self._provider, "config") and hasattr(self._provider.config, "model"):
            model_name = getattr(self._provider.config.model, "name", "")

        span_name = f"provider/{provider_name}"
        if model_name:
            span_name = f"{span_name}/{model_name}"
        return span_name

    def _get_provider_attributes(self) -> dict[str, str]:
        """Get provider attributes for span."""
        attrs = {
            SpanAttributeKey.MODEL_PROVIDER: self._get_provider_name(),
        }
        if hasattr(self._provider, "config") and hasattr(self._provider.config, "model"):
            if model_name := getattr(self._provider.config.model, "name", ""):
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

        return token_usage or None

    async def _trace_method(self, method_name: str, method, *args, **kwargs):
        """Execute a method with tracing span."""
        active_span = mlflow.get_current_active_span()
        if active_span is None:
            return await method(*args, **kwargs)

        span_name = self._get_span_name()
        with mlflow.start_span(name=span_name) as span:
            span.set_attributes({**self._get_provider_attributes(), "method": method_name})

            try:
                result = await method(*args, **kwargs)

                # Extract and log token usage if available
                if token_usage := self._extract_token_usage(result):
                    span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

                span.set_status("OK")
                return result
            except Exception as e:
                span.record_exception(e)
                raise

    async def _trace_stream_method(self, method_name: str, method, *args, **kwargs):
        """Execute a streaming method with tracing span."""
        active_span = mlflow.get_current_active_span()
        if active_span is None:
            async for chunk in method(*args, **kwargs):
                yield chunk
            return

        span_name = self._get_span_name()
        # Use start_span_no_context to get a LiveSpan that can be manually ended
        span = start_span_no_context(
            name=span_name,
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

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        async for chunk in self._trace_stream_method(
            "chat_stream", self._provider.chat_stream, payload
        ):
            yield chunk

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        return await self._trace_method("chat", self._provider.chat, payload)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        async for chunk in self._trace_stream_method(
            "completions_stream", self._provider.completions_stream, payload
        ):
            yield chunk

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        return await self._trace_method("completions", self._provider.completions, payload)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return await self._trace_method("embeddings", self._provider.embeddings, payload)

    async def passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[bytes]:
        return await self._trace_method(
            "passthrough", self._provider.passthrough, action, payload, headers
        )
