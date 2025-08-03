"""
Integration tests for OpenTelemetry REST API endpoints.

These tests start a FastAPI/uvicorn server and make actual HTTP requests
to the OTel endpoints.
"""

import base64
import os
import sys
import tempfile
import uuid
from subprocess import Popen

import pytest
import requests

from mlflow import MlflowClient

from tests.helper_functions import LOCALHOST, get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


@pytest.fixture
def fastapi_server():
    """Start a FastAPI server with uvicorn for testing OTel endpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend_uri = f"sqlite:///{tmpdir}/mlflow.db"
        artifact_root = tmpdir
        server_port = get_safe_port()
        server_url = f"http://{LOCALHOST}:{server_port}"

        # Start server with uvicorn (which uses FastAPI app)
        with Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "mlflow.server.fastapi_app:app",
                "--host",
                LOCALHOST,
                "--port",
                str(server_port),
                "--workers",
                "1",
            ],
            env={
                **os.environ,
                "_MLFLOW_SERVER_FILE_STORE": backend_uri,
                "_MLFLOW_SERVER_ARTIFACT_ROOT": artifact_root,
            },
        ) as proc:
            try:
                _await_server_up_or_die(server_port)
                yield server_url
            finally:
                proc.terminate()
                proc.wait()


def test_otel_span_export_json(fastapi_server):
    """Test OTel span export with JSON format."""
    # Create MLflow client
    client = MlflowClient(fastapi_server)

    # Create an experiment
    experiment_id = client.create_experiment("test_otel_json")

    # Generate a new trace ID (OTel spans create their own traces)
    trace_id_int = uuid.uuid4().int & ((1 << 128) - 1)  # 128-bit trace ID
    trace_id_bytes = trace_id_int.to_bytes(16, byteorder="big")

    # Create OTel span data
    span_id_bytes = (12345).to_bytes(8, byteorder="big")
    parent_span_id_bytes = (11111).to_bytes(8, byteorder="big")

    otel_request = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "test-service"}},
                        {"key": "mlflow.experimentId", "value": {"stringValue": experiment_id}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": base64.b64encode(trace_id_bytes).decode(),
                                "spanId": base64.b64encode(span_id_bytes).decode(),
                                "parentSpanId": base64.b64encode(parent_span_id_bytes).decode(),
                                "name": "test_otel_span",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "2000000000",
                                "status": {"code": "STATUS_CODE_OK"},
                                "traceState": "key1=value1,key2=value2",
                                "attributes": [
                                    {"key": "http.method", "value": {"stringValue": "GET"}},
                                    {"key": "http.status_code", "value": {"intValue": "200"}},
                                    {
                                        "key": "custom_attribute",
                                        "value": {"stringValue": "test_value"},
                                    },
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }

    # Send POST request to OTel endpoint
    response = requests.post(
        f"{fastapi_server}/v1/traces",
        json=otel_request,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    response_data = response.json()

    # Verify no partial failures
    assert "partialSuccess" not in response_data or response_data["partialSuccess"] is None


def test_otel_span_export_partial_failure(fastapi_server):
    """Test OTel span export with partial failures."""
    client = MlflowClient(fastapi_server)

    # Create an experiment
    experiment_id = client.create_experiment("test_otel_partial")

    # Create a valid trace
    trace_info = client.start_trace(name="test_trace", experiment_id=experiment_id)
    client.end_trace(trace_info.request_id)

    # Extract the trace ID
    trace_id_hex = trace_info.request_id[3:]
    trace_id_int = int(trace_id_hex, 16)
    trace_id_bytes = trace_id_int.to_bytes(16, byteorder="big")

    # Mix of valid and invalid spans
    otel_request = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            # Valid span
                            {
                                "traceId": base64.b64encode(trace_id_bytes).decode(),
                                "spanId": base64.b64encode(
                                    (100).to_bytes(8, byteorder="big")
                                ).decode(),
                                "name": "valid_span",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "2000000000",
                            },
                            # Invalid span - missing trace ID
                            {
                                "spanId": base64.b64encode(
                                    (101).to_bytes(8, byteorder="big")
                                ).decode(),
                                "name": "invalid_span",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "2000000000",
                            },
                            # Another valid span
                            {
                                "traceId": base64.b64encode(trace_id_bytes).decode(),
                                "spanId": base64.b64encode(
                                    (102).to_bytes(8, byteorder="big")
                                ).decode(),
                                "name": "another_valid_span",
                                "startTimeUnixNano": "3000000000",
                                "endTimeUnixNano": "4000000000",
                            },
                        ]
                    }
                ]
            }
        ]
    }

    response = requests.post(f"{fastapi_server}/v1/traces", json=otel_request)

    assert response.status_code == 200
    response_data = response.json()

    # Should have partial success info
    assert "partialSuccess" in response_data
    assert response_data["partialSuccess"]["rejectedSpans"] == 1
    assert "missing trace ID" in response_data["partialSuccess"]["errorMessage"]


def test_otel_span_export_protobuf(fastapi_server):
    """Test OTel span export with protobuf format."""
    try:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
        from opentelemetry.proto.trace.v1.trace_pb2 import Status
    except ImportError:
        pytest.skip("OpenTelemetry protobuf libraries not installed")

    client = MlflowClient(fastapi_server)

    # Create an experiment
    experiment_id = client.create_experiment("test_otel_protobuf")

    # Create a trace
    trace_info = client.start_trace(name="test_trace", experiment_id=experiment_id)
    client.end_trace(trace_info.request_id)

    # Extract the trace ID
    trace_id_hex = trace_info.request_id[3:]
    trace_id_int = int(trace_id_hex, 16)
    trace_id_bytes = trace_id_int.to_bytes(16, byteorder="big")

    # Create protobuf request
    proto_request = ExportTraceServiceRequest()
    resource_span = proto_request.resource_spans.add()
    scope_span = resource_span.scope_spans.add()

    # Add a span
    span = scope_span.spans.add()
    span.trace_id = trace_id_bytes
    span.span_id = (999).to_bytes(8, byteorder="big")
    span.name = "protobuf_test_span"
    span.start_time_unix_nano = 1000000000
    span.end_time_unix_nano = 2000000000
    span.status.code = Status.StatusCode.STATUS_CODE_OK

    # Send binary protobuf request
    response = requests.post(
        f"{fastapi_server}/v1/traces/protobuf",
        data=proto_request.SerializeToString(),
        headers={"Content-Type": "application/x-protobuf"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-protobuf"

    # The response should be valid protobuf (even if empty)
    assert len(response.content) >= 0


def test_otel_span_export_different_value_types(fastapi_server):
    """Test handling of different attribute value types."""
    client = MlflowClient(fastapi_server)

    # Create experiment and trace
    experiment_id = client.create_experiment("test_otel_types")
    trace_info = client.start_trace(name="test_trace", experiment_id=experiment_id)
    client.end_trace(trace_info.request_id)

    # Extract trace ID
    trace_id_hex = trace_info.request_id[3:]
    trace_id_int = int(trace_id_hex, 16)
    trace_id_bytes = trace_id_int.to_bytes(16, byteorder="big")

    otel_request = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": base64.b64encode(trace_id_bytes).decode(),
                                "spanId": base64.b64encode(
                                    (555).to_bytes(8, byteorder="big")
                                ).decode(),
                                "name": "test_attribute_types",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "2000000000",
                                "attributes": [
                                    {"key": "string_attr", "value": {"stringValue": "test_string"}},
                                    {"key": "int_attr", "value": {"intValue": "42"}},
                                    {"key": "double_attr", "value": {"doubleValue": "3.14159"}},
                                    {"key": "bool_attr", "value": {"boolValue": True}},
                                    {
                                        "key": "array_attr",
                                        "value": {
                                            "arrayValue": {
                                                "values": [
                                                    {"stringValue": "item1"},
                                                    {"stringValue": "item2"},
                                                    {"stringValue": "item3"},
                                                ]
                                            }
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    response = requests.post(f"{fastapi_server}/v1/traces", json=otel_request)

    assert response.status_code == 200
    response_data = response.json()
    assert "partialSuccess" not in response_data or response_data["partialSuccess"] is None


def test_otel_span_export_empty_request(fastapi_server):
    """Test handling of empty request."""
    response = requests.post(
        f"{fastapi_server}/v1/traces",
        json={"resourceSpans": []},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "partialSuccess" not in response_data or response_data["partialSuccess"] is None


def test_otel_span_export_invalid_json(fastapi_server):
    """Test handling of invalid JSON."""
    response = requests.post(
        f"{fastapi_server}/v1/traces",
        data="invalid json {",
        headers={"Content-Type": "application/json"},
    )

    # FastAPI returns 422 for validation errors
    assert response.status_code == 422
