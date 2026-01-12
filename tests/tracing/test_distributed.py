import re

import pytest
from opentelemetry import trace as otel_trace

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)


def _parse_traceparent(header_value: str) -> tuple[int, int]:
    """
    Parse W3C traceparent header into (trace_id_int, span_id_int).
    Format: version-traceid-spanid-flags (all lowercase hex, no 0x prefix).
    """
    parts = header_value.split("-")
    assert len(parts) == 4, f"Invalid traceparent format: {header_value}"
    version, trace_id_hex, span_id_hex, flags = parts
    assert re.fullmatch(r"[0-9a-f]{2}", version), f"Invalid version: {version}"
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id_hex), f"Invalid trace id: {trace_id_hex}"
    assert re.fullmatch(r"[0-9a-f]{16}", span_id_hex), f"Invalid span id: {span_id_hex}"
    assert re.fullmatch(r"[0-9a-f]{2}", flags), f"Invalid flags: {flags}"
    return int(trace_id_hex, 16), int(span_id_hex, 16)


def test_get_tracing_context_headers_for_http_request_in_active_span():
    with mlflow.start_span("client-span"):
        current_span = otel_trace.get_current_span()
        assert current_span.get_span_context().is_valid
        client_trace_id = current_span.get_span_context().trace_id
        client_span_id = current_span.get_span_context().span_id

        headers: dict[str, str] = get_tracing_context_headers_for_http_request()
        assert isinstance(headers, dict)
        assert "traceparent" in headers

        # Validate that the header encodes the same trace and span IDs
        header_trace_id, header_span_id = _parse_traceparent(headers["traceparent"])
        assert header_trace_id == client_trace_id
        assert header_span_id == client_span_id


def test_get_tracing_context_headers_for_http_request_without_active_span():
    with pytest.raises(
        MlflowException,
        match=(
            "'get_tracing_context_headers_for_http_request' must be called within the scope "
            "of an active span."
        ),
    ):
        get_tracing_context_headers_for_http_request()


def test_set_tracing_context_from_http_request_headers():
    # Create headers from a client context first
    with mlflow.start_span("client-to-generate-headers") as client_span:
        client_headers = get_tracing_context_headers_for_http_request()
        client_otel_trace_id, client_otel_span_id = _parse_traceparent(
            client_headers["traceparent"]
        )
        client_trace_id = client_span.trace_id
        client_span_id = client_span.span_id

    assert mlflow.get_current_active_span() is None

    # Attach the context from headers and verify it becomes current inside the block
    with set_tracing_context_from_http_request_headers(client_headers):
        current = otel_trace.get_current_span()
        assert current.get_span_context().is_valid
        assert current.get_span_context().trace_id == client_otel_trace_id
        # span_id in context is the parent span id from the header
        assert current.get_span_context().span_id == client_otel_span_id

        # get_current_active_span returns None because it is a `NonRecordingSpan`
        assert mlflow.get_current_active_span() is None

        with mlflow.start_span("child-span") as child_span:
            assert child_span.parent_id == client_span_id
            assert child_span.trace_id == client_trace_id
