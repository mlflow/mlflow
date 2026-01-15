"""
Flask server for distributed tracing tests.

This server is used to test distributed tracing functionality by accepting
HTTP requests with tracing headers and creating child spans.
"""

import sys

import requests
from flask import Flask, jsonify, request

import mlflow
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)

app = Flask(__name__)


@app.get("/health")
def health():
    """Health check endpoint."""
    return "ok", 200


@app.post("/handle")
def handle():
    """
    Handle a request with distributed tracing context.

    Extracts tracing headers from the request and creates a child span
    within the distributed trace context.
    """
    headers = dict(request.headers)
    with set_tracing_context_from_http_request_headers(headers):
        with mlflow.start_span("server-handler") as span:
            return jsonify(
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_id": span.parent_id,
                }
            )


@app.post("/handle1")
def handle1():
    """
    Handle a request and make a nested call to another server.

    This endpoint demonstrates distributed tracing across multiple services.
    It receives a request with tracing headers, creates a span, makes a nested
    call to another service (/handle2), and returns combined results.
    """
    headers = dict(request.headers)
    with set_tracing_context_from_http_request_headers(headers):
        with mlflow.start_span("server-handler1") as span:
            # Get the URL for the second handler from environment or command line
            # In nested tests, this will be passed via environment
            second_server_url = request.args.get("second_server_url")
            if not second_server_url:
                return jsonify({"error": "second_server_url parameter required"}), 400

            headers2 = get_tracing_context_headers_for_http_request()
            resp2 = requests.post(f"{second_server_url}/handle2", headers=headers2, timeout=5)
            if not resp2.ok:
                return jsonify({"error": f"Nested call failed: {resp2.status_code}"}), 502

            payload2 = resp2.json()
            return jsonify(
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_id": span.parent_id,
                    "nested_call_resp": payload2,
                }
            )


@app.post("/handle2")
def handle2():
    """
    Handle a nested request in a distributed trace.

    This is the second level handler that receives requests from /handle1
    and creates its own span in the distributed trace.
    """
    headers = dict(request.headers)
    with set_tracing_context_from_http_request_headers(headers):
        with mlflow.start_span("server-handler2") as span:
            return jsonify(
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_id": span.parent_id,
                }
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: flask_tracing_server.py <port>")

    port = int(sys.argv[1])
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
