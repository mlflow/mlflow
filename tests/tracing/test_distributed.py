import re

from opentelemetry import trace as otel_trace

import mlflow
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)


def _parse_traceparent(header_value: str) -> tuple[int, int]:
    """
    Parse W3C traceparent header into (trace_id_int, span_id_int).
    Format: version-traceid-spanid-flags (all lowercase hex, no 0x prefix).
    """
    # Basic validation to provide clearer error messages in case of malformed headers
    assert isinstance(header_value, str) and header_value, (
        "traceparent header must be a non-empty string"
    )
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
        assert "traceparent" in headers and headers["traceparent"]

        # Validate that the header encodes the same trace and span IDs
        header_trace_id, header_span_id = _parse_traceparent(headers["traceparent"])
        assert header_trace_id == client_trace_id
        assert header_span_id == client_span_id


def test_get_tracing_context_headers_for_http_request_without_active_span():
    # No active span: injection should not add a traceparent header
    headers: dict[str, str] = get_tracing_context_headers_for_http_request()
    # OpenTelemetry inject typically omits headers when context is invalid
    assert "traceparent" not in headers
    assert headers == {} or not headers.get("traceparent")


def test_set_tracing_context_from_http_request_headers_attaches_and_detaches():
    # Create headers from a client context first
    with mlflow.start_span("client-to-generate-headers"):
        client_headers = get_tracing_context_headers_for_http_request()
        assert "traceparent" in client_headers
        client_trace_id, client_span_id = _parse_traceparent(client_headers["traceparent"])
        assert client_trace_id != 0 and client_span_id != 0

    # Outside the client span, there should be no active span
    assert not otel_trace.get_current_span().get_span_context().is_valid
    assert mlflow.get_current_active_span() is None

    # Attach the context from headers and verify it becomes current inside the block
    with set_tracing_context_from_http_request_headers(client_headers):
        current = otel_trace.get_current_span()
        assert current.get_span_context().is_valid
        assert current.get_span_context().trace_id == client_trace_id
        # span_id in context is the parent span id from the header
        assert current.get_span_context().span_id == client_span_id

    # After exiting, the previously invalid context should be restored
    assert not otel_trace.get_current_span().get_span_context().is_valid
    assert mlflow.get_current_active_span() is None


def test_end_to_end_inject_extract_and_create_child_span():
    # Client side: start a span and create headers to send downstream
    with mlflow.start_span("client-root") as client_mlflow_span:
        client_current = otel_trace.get_current_span()
        assert client_current.get_span_context().is_valid
        client_trace_id = client_current.get_span_context().trace_id
        client_span_id = client_current.get_span_context().span_id

        headers = get_tracing_context_headers_for_http_request()
        assert "traceparent" in headers
        h_trace_id, h_parent_span_id = _parse_traceparent(headers["traceparent"])
        assert h_trace_id == client_trace_id
        assert h_parent_span_id == client_span_id

    # Server side: extract headers, set context, and start a child span
    with set_tracing_context_from_http_request_headers(headers):
        # Starting a new MLflow span should create a child under the extracted context
        with mlflow.start_span("server-handler"):
            server_current = otel_trace.get_current_span()
            assert server_current.get_span_context().is_valid
            # Same trace, different (new) span id
            assert server_current.get_span_context().trace_id == client_trace_id
            assert server_current.get_span_context().span_id != h_parent_span_id

            # MLflow API should see an active span
            assert mlflow.get_current_active_span() is not None
