"""Flask server for distributed tracing tests."""

import logging
import os
import sys
import time

import requests
from flask import Flask, jsonify, request

import mlflow
from mlflow.tracing.distributed import (
    get_tracing_context_headers_for_http_request,
    set_tracing_context_from_http_request_headers,
)

REQUEST_TIMEOUT = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

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
    t0 = time.perf_counter()
    headers = dict(request.headers)
    with set_tracing_context_from_http_request_headers(headers):
        t1 = time.perf_counter()
        with mlflow.start_span("server-handler1") as span:
            t2 = time.perf_counter()
            second_server_url = request.args.get("second_server_url")
            if not second_server_url:
                return jsonify({"error": "second_server_url parameter required"}), 400

            headers2 = get_tracing_context_headers_for_http_request()
            t3 = time.perf_counter()
            resp2 = requests.post(
                f"{second_server_url}/handle2", headers=headers2, timeout=REQUEST_TIMEOUT
            )
            t4 = time.perf_counter()
            if not resp2.ok:
                return jsonify({"error": f"Nested call failed: {resp2.status_code}"}), 502

            payload2 = resp2.json()
            t5 = time.perf_counter()
            logger.info(
                "/handle1 timing: set_context=%.3fs, start_span=%.3fs, "
                "get_headers=%.3fs, nested_call=%.3fs, parse_resp=%.3fs, total=%.3fs",
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
                t5 - t0,
            )
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
    t0 = time.perf_counter()
    headers = dict(request.headers)
    with set_tracing_context_from_http_request_headers(headers):
        t1 = time.perf_counter()
        with mlflow.start_span("server-handler2") as span:
            t2 = time.perf_counter()
            logger.info(
                "/handle2 timing: set_context=%.3fs, start_span=%.3fs, total=%.3fs",
                t1 - t0,
                t2 - t1,
                t2 - t0,
            )
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
    logger.info("Server starting on port %d (pid=%d)", port, os.getpid())
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
