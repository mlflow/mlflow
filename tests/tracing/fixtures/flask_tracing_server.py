"""Flask server for distributed tracing tests."""

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
    return "ok", 200


@app.post("/handle")
def handle():
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
