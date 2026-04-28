"""
Tests for session ID propagation in distributed tracing scenarios.
"""

import mlflow
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.distributed import (
    MLFLOW_SESSION_ID_HEADER,
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)
from mlflow.tracing.session_context import get_session_id, set_session


def test_session_id_included_in_outgoing_headers():
    with set_session("distributed-session-123"):
        headers = get_tracing_context_headers_for_http_request()

        assert MLFLOW_SESSION_ID_HEADER in headers
        assert headers[MLFLOW_SESSION_ID_HEADER] == "distributed-session-123"


def test_session_id_not_included_when_not_set():
    # Ensure no session is set
    assert get_session_id() is None

    headers = get_tracing_context_headers_for_http_request()
    assert MLFLOW_SESSION_ID_HEADER not in headers


def test_session_id_extracted_from_incoming_headers():
    # First, generate valid traceparent headers from an active span
    with mlflow.start_span("client-span"):
        base_headers = get_tracing_context_headers_for_http_request()

    # Add session ID to the headers
    incoming_headers = {
        **base_headers,
        MLFLOW_SESSION_ID_HEADER: "incoming-session-456",
    }

    with set_tracing_context_from_http_request_headers(incoming_headers):
        # Session ID should be set from headers
        assert get_session_id() == "incoming-session-456"

    # Should be cleared after context exits
    assert get_session_id() is None


def test_session_id_header_case_insensitive():
    # First, generate valid traceparent headers from an active span
    with mlflow.start_span("client-span"):
        base_headers = get_tracing_context_headers_for_http_request()

    # Add session ID with title-case header (like some HTTP servers do)
    incoming_headers = {
        **base_headers,
        "Mlflow-Session-Id": "case-insensitive-session",
    }

    with set_tracing_context_from_http_request_headers(incoming_headers):
        assert get_session_id() == "case-insensitive-session"

    assert get_session_id() is None


def test_end_to_end_session_propagation():
    """
    Simulates a complete distributed tracing scenario:
    1. Service A sets a session ID and makes a traced call
    2. Service A extracts headers for HTTP request to Service B
    3. Service B receives headers and sets context
    4. Session ID should be available in Service B's context
    """
    # Service A: Set session and get headers for outbound request within an active span
    with set_session("e2e-session-test"):
        with mlflow.start_span("service-a-request"):
            outbound_headers = get_tracing_context_headers_for_http_request()

    # Verify headers contain both traceparent and session ID
    assert "traceparent" in outbound_headers
    assert outbound_headers.get(MLFLOW_SESSION_ID_HEADER) == "e2e-session-test"

    # Service B: Receive headers and verify session context is set
    with set_tracing_context_from_http_request_headers(outbound_headers):
        # The session ID should be available in Service B's context
        assert get_session_id() == "e2e-session-test"

        # Verify we can create spans within this context
        with mlflow.start_span("service-b-handler") as span:
            span.set_attribute("key", "value")
            # Span should be created successfully with the parent from headers
            assert span is not None

    # After context exit, session should be cleared
    assert get_session_id() is None


def test_session_id_propagation_with_trace_creation():
    """
    Verify that a locally created trace (not from remote headers)
    properly captures the session ID from context.
    """
    with set_session("local-trace-session"):

        @mlflow.trace
        def local_function():
            return "result"

        local_function()

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        assert (
            trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) == "local-trace-session"
        )
