"""
Tracing utilities for MLflow Gateway.

This module provides tracing functionality for gateway API endpoints,
including span management for both synchronous and streaming requests.
"""

import contextvars
import logging
from contextlib import contextmanager
from typing import Any

from fastapi.responses import StreamingResponse

import mlflow
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context

_logger = logging.getLogger(__name__)


def _configure_gateway_span(
    span: LiveSpan,
    endpoint_config: GatewayEndpointConfig,
    request_type: str,
    inputs: dict[str, Any],
):
    """
    Configure a gateway span with standard attributes and tags.

    Args:
        span: The span to configure.
        endpoint_config: The gateway endpoint configuration.
        request_type: Type of request (e.g., "chat", "embeddings", "passthrough").
        inputs: The request inputs to log.
    """
    span.set_inputs(inputs)
    span.set_attribute("request_type", request_type)
    span.set_attribute("endpoint_id", endpoint_config.endpoint_id)
    span.set_attribute("endpoint_name", endpoint_config.endpoint_name)

    # Set trace-level tags for filtering in metrics API
    tags = {
        TraceMetadataKey.GATEWAY_ENDPOINT_ID: endpoint_config.endpoint_id,
        TraceMetadataKey.GATEWAY_REQUEST_TYPE: request_type,
    }
    mlflow.update_current_trace(tags=tags)


def _start_gateway_span(
    endpoint_config: GatewayEndpointConfig,
    request_type: str,
    inputs: dict[str, Any],
) -> tuple[LiveSpan | None, contextvars.Token | None]:
    """
    Start a gateway trace span for streaming scenarios.

    This creates a span with proper context management, suitable for streaming
    where the span lifecycle extends beyond the initial function call.

    Args:
        endpoint_config: The gateway endpoint configuration.
        request_type: Type of request (e.g., "chat", "embeddings", "passthrough").
        inputs: The request inputs to log.

    Returns:
        Tuple of (span, context_token) if tracing is enabled, (None, None) otherwise.
        The context_token must be passed to _end_gateway_span to restore context.
    """
    if not endpoint_config.experiment_id:
        return None, None

    try:
        span = start_span_no_context(
            name=f"gateway/{endpoint_config.endpoint_name}",
            experiment_id=endpoint_config.experiment_id,
        )
        _configure_gateway_span(span, endpoint_config, request_type, inputs)

        # Set as active span so child spans can be created
        token = set_span_in_context(span)
        return span, token
    except Exception as e:
        _logger.debug(f"Failed to start gateway span: {e}")
        return None, None


def _end_gateway_span(span: LiveSpan | None, token: contextvars.Token | None) -> None:
    if span is None:
        return

    try:
        span.end()
    except Exception as e:
        _logger.debug(f"Failed to end span: {e}")

    if token is not None:
        try:
            detach_span_from_context(token)
        except Exception as e:
            _logger.debug(f"Failed to detach context: {e}")


@contextmanager
def _create_gateway_trace(
    endpoint_config: GatewayEndpointConfig,
    request_type: str,
    inputs: dict[str, Any],
):
    """
    Context manager for gateway tracing with proper context management.

    This uses mlflow.start_span which sets the span as active in the context,
    allowing child spans to be created properly.

    Args:
        endpoint_config: The gateway endpoint configuration.
        request_type: Type of request (e.g., "chat", "embeddings", "passthrough").
        inputs: The request inputs to log.

    Yields:
        The span if tracing is enabled, None otherwise.
    """
    if not endpoint_config.experiment_id:
        yield None
        return

    with mlflow.start_span(
        name=f"gateway/{endpoint_config.endpoint_name}",
        trace_destination=MlflowExperimentLocation(experiment_id=endpoint_config.experiment_id),
    ) as span:
        _configure_gateway_span(span, endpoint_config, request_type, inputs)
        yield span


def _set_trace_outputs(span: LiveSpan, outputs: Any):
    """Helper to set outputs on a span if it exists."""
    if span is None:
        return
    try:
        span.set_outputs(outputs)
    except Exception as e:
        _logger.debug(f"Failed to set trace outputs: {e}")


async def _make_traced_streaming_response(
    stream,
    span: LiveSpan | None,
    token: contextvars.Token | None,
    is_sse: bool = True,
):
    """
    Create a streaming response that captures outputs for tracing.

    This wraps the stream generator to collect outputs and set them on the span
    after streaming completes. The span is ended when the stream finishes.

    Args:
        stream: The async generator producing stream chunks.
        span: The span to set outputs on (can be None if tracing is disabled).
        token: The context token from _start_gateway_span (for restoring context).
        is_sse: If True, format chunks as SSE data events. If False, yield raw bytes.

    Returns:
        A StreamingResponse that traces the streamed outputs.
    """
    from mlflow.gateway.utils import to_sse_chunk

    async def traced_stream():
        chunks = []
        try:
            async for chunk in stream:
                chunks.append(chunk)
                if is_sse:
                    yield to_sse_chunk(chunk.model_dump_json())
                else:
                    yield chunk

            # Set all chunks as output
            if span is not None:
                try:
                    if chunks:
                        span.set_outputs(chunks)
                    span.set_status("OK")
                except Exception as e:
                    _logger.debug(f"Failed to set stream outputs: {e}")
        except Exception as e:
            if span is not None:
                try:
                    span.record_exception(e)
                except Exception:
                    pass
            raise
        finally:
            _end_gateway_span(span, token)

    return StreamingResponse(traced_stream(), media_type="text/event-stream")
