import re
import sys
import textwrap
import time
from pathlib import Path
from subprocess import Popen

import pytest
import requests
from opentelemetry import trace as otel_trace

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)

from tests.helper_functions import get_safe_port


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
        breakpoint()
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


def test_distributed_e2e_in_subprocess(tmp_path):
    # Prepare a minimal Flask server script that extracts headers and starts a child span
    server_code = textwrap.dedent(
        """
        import sys
        import json
        from flask import Flask, request, jsonify
        import mlflow
        from mlflow.tracing.distributed import set_tracing_context_from_http_request_headers

        app = Flask(__name__)

        @app.get("/health")
        def health():
            return "ok", 200

        @app.post("/handle")
        def handle():
            # Forward all headers for extraction
            headers = dict(request.headers)
            with set_tracing_context_from_http_request_headers(headers):
                with mlflow.start_span("server-handler") as span:
                    return jsonify({
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "parent_id": span.parent_id,
                    })

        if __name__ == "__main__":
            port = int(sys.argv[1])
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        """
    )

    server_path = Path(tmp_path) / "flask_server.py"
    server_path.write_text(server_code)

    port = get_safe_port()

    # Start server in a separate process
    proc = Popen([sys.executable, str(server_path), str(port)])
    try:
        base_url = f"http://127.0.0.1:{port}"
        # Wait until server is ready
        for _ in range(30):
            try:
                r = requests.get(f"{base_url}/health", timeout=1.0)
                if r.ok:
                    break
            except requests.exceptions.RequestException:
                time.sleep(0.2)
        else:
            raise RuntimeError("Flask server failed to start")

        # Client side: create a span and send headers to server
        with mlflow.start_span("client-root") as client_span:
            headers = get_tracing_context_headers_for_http_request()
            resp = requests.post(f"{base_url}/handle", headers=headers, timeout=5)
            assert resp.ok, f"Server returned {resp.status_code}: {resp.text}"
            payload = resp.json()

            # Validate server span is a child in the same trace
            assert payload["trace_id"] == client_span.trace_id
            assert payload["parent_id"] == client_span.span_id
    finally:
        proc.terminate()
