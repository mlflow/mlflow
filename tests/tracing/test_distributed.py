import re
import sys
import textwrap
import time
from pathlib import Path
from subprocess import Popen

import requests
from opentelemetry import trace as otel_trace

import mlflow
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)

from tests.helper_functions import get_safe_port
from tests.tracing.helper import skip_when_testing_trace_sdk


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
    headers = get_tracing_context_headers_for_http_request()
    assert headers == {}


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


@skip_when_testing_trace_sdk
def test_distributed_tracing_e2e(tmp_path):
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
        proc.wait()

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
    base_url = f"http://127.0.0.1:{port}"
    base_url2 = f"http://127.0.0.1:{port2}"

    server_code = textwrap.dedent(
        f"""
        import sys
        import json
        import requests
        from flask import Flask, request, jsonify
        import mlflow
        from mlflow.tracing.distributed import (
            set_tracing_context_from_http_request_headers,
            get_tracing_context_headers_for_http_request,
        )

        app = Flask(__name__)

        @app.get("/health")
        def health():
            return "ok", 200

        @app.post("/handle1")
        def handle1():
            # Forward all headers for extraction
            headers = dict(request.headers)
            with set_tracing_context_from_http_request_headers(headers):
                with mlflow.start_span("server-handler1") as span:
                    headers2 = get_tracing_context_headers_for_http_request()
                    resp2 = requests.post("{base_url2}/handle2", headers=headers2, timeout=5)
                    assert resp2.ok
                    payload2 = resp2.json()
                    return jsonify({{
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "parent_id": span.parent_id,
                        "nested_call_resp": payload2,
                    }})

        @app.post("/handle2")
        def handle2():
            # Forward all headers for extraction
            headers = dict(request.headers)
            with set_tracing_context_from_http_request_headers(headers):
                with mlflow.start_span("server-handler2") as span:
                    return jsonify({{
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "parent_id": span.parent_id,
                    }})

        if __name__ == "__main__":
            port = int(sys.argv[1])
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        """
    )

    server_path = Path(tmp_path) / "flask_server.py"
    server_path.write_text(server_code)

    # Start server in a separate process
    proc = Popen([sys.executable, str(server_path), str(port)])
    proc2 = Popen([sys.executable, str(server_path), str(port2)])
    try:
        # Wait until server is ready
        for _ in range(30):
            try:
                r = requests.get(f"{base_url}/health", timeout=1.0)
                r2 = requests.get(f"{base_url2}/health", timeout=1.0)
                if r.ok and r2.ok:
                    break
            except requests.exceptions.RequestException:
                time.sleep(0.2)
        else:
            raise RuntimeError("Flask server failed to start")

        # Client side: create a span and send headers to server
        with mlflow.start_span("client-root") as client_span:
            headers = get_tracing_context_headers_for_http_request()
            resp = requests.post(f"{base_url}/handle1", headers=headers, timeout=5)
            assert resp.ok, f"Server returned {resp.status_code}: {resp.text}"
            payload = resp.json()

            # Validate server span is a child in the same trace
            assert payload["trace_id"] == client_span.trace_id
            assert payload["parent_id"] == client_span.span_id
            child_span1_id = payload["span_id"]
            assert payload["nested_call_resp"]["trace_id"] == client_span.trace_id
            assert payload["nested_call_resp"]["parent_id"] == child_span1_id
            child_span2_id = payload["nested_call_resp"]["span_id"]
    finally:
        proc.terminate()
        proc2.terminate()
        proc.wait()
        proc2.wait()

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
