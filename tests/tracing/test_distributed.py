import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import requests

import mlflow
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)

from tests.helper_functions import get_safe_port
from tests.tracing.helper import skip_when_testing_trace_sdk

REQUEST_TIMEOUT = 10


@contextmanager
def flask_server(
    server_script_path: Path,
    port: int,
    *,
    wait_timeout: int = 30,
    health_endpoint: str = "/health",
) -> Iterator[str]:
    """Context manager to run a Flask server in a subprocess."""
    with subprocess.Popen([sys.executable, str(server_script_path), str(port)]) as proc:
        base_url = f"http://127.0.0.1:{port}"

        try:
            # Wait for server to be ready
            for _ in range(wait_timeout):
                try:
                    response = requests.get(f"{base_url}{health_endpoint}", timeout=1.0)
                    if response.ok:
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"Flask server failed to start within {wait_timeout} seconds")

            yield base_url
        finally:
            proc.terminate()


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
        current_span = mlflow.get_current_active_span()._span
        assert current_span.get_span_context().is_valid
        client_trace_id = current_span.get_span_context().trace_id
        client_span_id = current_span.get_span_context().span_id

        headers = get_tracing_context_headers_for_http_request()
        assert isinstance(headers, dict)
        assert "traceparent" in headers

        # Validate that the header encodes the same trace and span IDs
        header_trace_id, header_span_id = _parse_traceparent(headers["traceparent"])
        assert header_trace_id == client_trace_id
        assert header_span_id == client_span_id


def test_get_tracing_context_headers_for_http_request_without_active_span():
    headers = get_tracing_context_headers_for_http_request()
    assert headers == {}


def test_set_tracing_context_from_http_request_headers():
    # Create headers from a client context first
    with mlflow.start_span("client-to-generate-headers") as client_span:
        client_headers = get_tracing_context_headers_for_http_request()
        client_trace_id = client_span.trace_id
        client_span_id = client_span.span_id

    assert mlflow.get_current_active_span() is None

    # Attach the context from headers and verify it becomes current inside the block
    with set_tracing_context_from_http_request_headers(client_headers):
        # get_current_active_span returns None because it is a `NonRecordingSpan`
        assert mlflow.get_current_active_span() is None

        with mlflow.start_span("child-span") as child_span:
            assert child_span.parent_id == client_span_id
            assert child_span.trace_id == client_trace_id


@skip_when_testing_trace_sdk
def test_distributed_tracing_e2e(tmp_path):
    # Path to the Flask server script
    server_path = Path(__file__).parent / "fixtures" / "flask_tracing_server.py"
    port = get_safe_port()

    # Start Flask server using the context manager
    with flask_server(server_path, port) as base_url:
        # Client side: create a span and send headers to server
        with mlflow.start_span("client-root") as client_span:
            headers = get_tracing_context_headers_for_http_request()
            resp = requests.post(f"{base_url}/handle", headers=headers, timeout=REQUEST_TIMEOUT)
            assert resp.ok, f"Server returned {resp.status_code}: {resp.text}"
            payload = resp.json()

            # Validate server span is a child in the same trace
            assert payload["trace_id"] == client_span.trace_id
            assert payload["parent_id"] == client_span.span_id

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(client_span.trace_id)

    assert trace is not None, "Trace not found"
    spans = trace.data.spans
    assert len(spans) == 2

    # Identify root and child
    root_span = next(s for s in spans if s.parent_id is None)
    child_span = next(s for s in spans if s.parent_id is not None)

    assert root_span.name == "client-root"
    assert child_span.name == "server-handler"
    assert child_span.parent_id == root_span.span_id


@skip_when_testing_trace_sdk
def test_distributed_tracing_e2e_nested_call(tmp_path):
    port = get_safe_port()
    port2 = get_safe_port()

    # Path to the Flask server script
    server_path = Path(__file__).parent / "fixtures" / "flask_tracing_server.py"

    # Start both Flask servers using the context manager
    with flask_server(server_path, port) as base_url, flask_server(server_path, port2) as base_url2:
        # Client side: create a span and send headers to server
        with mlflow.start_span("client-root") as client_span:
            headers = get_tracing_context_headers_for_http_request()
            # Pass the second server URL as a query parameter
            resp = requests.post(
                f"{base_url}/handle1",
                headers=headers,
                params={"second_server_url": base_url2},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.ok, f"Server returned {resp.status_code}: {resp.text}"
            payload = resp.json()

            # Validate server span is a child in the same trace
            assert payload["trace_id"] == client_span.trace_id
            assert payload["parent_id"] == client_span.span_id
            child_span1_id = payload["span_id"]
            assert payload["nested_call_resp"]["trace_id"] == client_span.trace_id
            assert payload["nested_call_resp"]["parent_id"] == child_span1_id
            child_span2_id = payload["nested_call_resp"]["span_id"]

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(client_span.trace_id)

    assert trace is not None, "Trace not found"
    spans = trace.data.spans
    assert len(spans) == 3

    # Identify root and child
    root_span = next(s for s in spans if s.parent_id is None)
    child_span1 = next(s for s in spans if s.parent_id == root_span.span_id)
    child_span2 = next(s for s in spans if s.parent_id == child_span1.span_id)

    assert root_span.name == "client-root"
    assert child_span1.name == "server-handler1"
    assert child_span2.name == "server-handler2"
    assert child_span1.span_id == child_span1_id
    assert child_span2.span_id == child_span2_id
