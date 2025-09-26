import json
from unittest import mock

import pytest
from google.protobuf.json_format import MessageToDict

import mlflow
from mlflow.entities import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_tracing_pb2 import CreateTrace, GetTraces
from mlflow.protos.service_pb2 import StartTraceV3
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.tracing.constant import TRACE_ID_V4_PREFIX
from mlflow.utils.databricks_tracing_utils import (
    trace_info_to_proto,
    trace_location_from_databricks_uc_schema,
    trace_to_proto,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds


def _args(host_creds, endpoint, method, json_body, version, retry_timeout_seconds=None):
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/{version}/mlflow/{endpoint}",
        "method": method,
    }
    if retry_timeout_seconds is not None:
        res["retry_timeout_seconds"] = retry_timeout_seconds
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(
    http_request,
    host_creds,
    endpoint,
    method,
    json_body,
    version="4.0",
    retry_timeout_seconds=None,
):
    """
    Verify HTTP requests in tests.

    Args:
        http_request: The mocked HTTP request object
        host_creds: MlflowHostCreds object
        endpoint: The endpoint being called (e.g., "traces/123")
        method: The HTTP method (e.g., "GET", "POST")
        json_body: The request body as a JSON string
        version: The version of the API to use (e.g., "2.0", "3.0", "4.0")
        retry_timeout_seconds: The retry timeout seconds to use for the request
    """
    http_request.assert_any_call(
        **(_args(host_creds, endpoint, method, json_body, version, retry_timeout_seconds))
    )


def test_create_trace_v4_uc_location(monkeypatch):
    monkeypatch.setenv(MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.name, "1")
    monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, "test-warehouse")

    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    trace_info = TraceInfo(
        trace_id="trace:/catalog.schema/123",
        trace_location=trace_location_from_databricks_uc_schema("catalog", "schema"),
        request_time=123,
        execution_duration=10,
        state=TraceState.OK,
        request_preview="",
        response_preview="",
        trace_metadata={},
    )

    # Mock successful v4 response
    response = mock.MagicMock()
    response.status_code = 200
    expected_trace_info = MessageToDict(
        trace_info_to_proto(trace_info), preserving_proto_field_name=True
    )
    # The returned trace_id in proto should be otel_trace_id
    expected_trace_info.update({"trace_id": "123"})
    response.text = json.dumps({"trace_info": expected_trace_info})

    expected_request = CreateTrace(
        trace_info=trace_info_to_proto(trace_info),
        sql_warehouse_id="test-warehouse",
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        result = store.start_trace(trace_info)
        _verify_requests(
            mock_http,
            creds,
            "traces/catalog.schema",
            "POST",
            message_to_json(expected_request),
            version="4.0",
            retry_timeout_seconds=1,
        )
        assert result.trace_id == "trace:/catalog.schema/123"


def test_create_trace_v4_experiment_location(monkeypatch):
    monkeypatch.setenv(MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.name, "1")
    monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, "test-warehouse")

    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    trace_info = TraceInfo(
        trace_id="tr-123",
        trace_location=TraceLocation.from_experiment_id("123"),
        request_time=123,
        execution_duration=10,
        state=TraceState.OK,
        request_preview="",
        response_preview="",
        trace_metadata={},
    )

    # Mock successful v4 response
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps({"trace_info": trace_info.to_dict()})

    expected_request = CreateTrace(
        trace_info=trace_info_to_proto(trace_info),
        sql_warehouse_id="test-warehouse",
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        result = store.start_trace(trace_info)
        _verify_requests(
            mock_http,
            creds,
            "traces/123",
            "POST",
            message_to_json(expected_request),
            version="4.0",
            retry_timeout_seconds=1,
        )
        assert result.trace_id == "tr-123"


def test_create_trace_v4_fallback_to_v3(monkeypatch):
    monkeypatch.setenv(MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.name, "1")
    monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, "test-warehouse")

    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    trace_info = TraceInfo(
        trace_id="tr-456",
        trace_location=TraceLocation.from_experiment_id("456"),
        request_time=456,
        execution_duration=20,
        state=TraceState.OK,
        request_preview="preview",
        response_preview="response",
        trace_metadata={"key": "value"},
    )

    trace = Trace(info=trace_info, data=TraceData())

    v4_error = MlflowException("Endpoint not found", error_code=databricks_pb2.ENDPOINT_NOT_FOUND)
    v3_response = StartTraceV3.Response(trace=trace.to_proto())

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        mock_call_endpoint.side_effect = [v4_error, v3_response]

        result = store.start_trace(trace_info)

        assert mock_call_endpoint.call_count == 2
        first_call = mock_call_endpoint.call_args_list[0]
        assert first_call[0][0] == CreateTrace
        second_call = mock_call_endpoint.call_args_list[1]
        assert second_call[0][0] == StartTraceV3
        assert result.trace_id == "tr-456"


def test_get_traces(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, "test-warehouse")
    with mlflow.start_span(name="test_span_1") as span1:
        span1.set_inputs({"input": "test_value_1"})
        span1.set_outputs({"output": "result_1"})

    with mlflow.start_span(name="test_span_2") as span2:
        span2.set_inputs({"input": "test_value_2"})
        span2.set_outputs({"output": "result_2"})

    trace1 = mlflow.get_trace(span1.trace_id)
    trace2 = mlflow.get_trace(span2.trace_id)

    mock_response = GetTraces.Response()
    mock_response.traces.extend([trace_to_proto(trace1), trace_to_proto(trace2)])

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("https://test"))

    location = "catalog.schema"
    v4_trace_id_1 = f"{TRACE_ID_V4_PREFIX}{location}/{span1.trace_id}"
    v4_trace_id_2 = f"{TRACE_ID_V4_PREFIX}{location}/{span2.trace_id}"
    trace_ids = [v4_trace_id_1, v4_trace_id_2]

    with (
        mock.patch.object(store, "_call_endpoint", return_value=mock_response) as mock_call,
    ):
        result = store.get_traces(trace_ids)

        mock_call.assert_called_once()
        call_args = mock_call.call_args

        assert call_args[0][0] == GetTraces

        request_body = call_args[0][1]
        request_data = json.loads(request_body)
        assert request_data["sql_warehouse_id"] == "test-warehouse"
        assert "trace_ids" in request_data
        assert len(request_data["trace_ids"]) == 2

        endpoint = call_args[1]["endpoint"]
        assert "/mlflow/traces/batch" in endpoint

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(trace, Trace) for trace in result)
        assert result[0].info.trace_id == span1.trace_id
        assert result[1].info.trace_id == span2.trace_id


def test_log_spans_to_uc_table_empty_spans():
    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("http://localhost"))
    result = store._log_spans_to_uc_table("catalog.schema.table", [], "databricks")
    assert result == []


def test_log_spans_to_uc_table_multiple_trace_ids():
    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("http://localhost"))
    spans = [
        mock.MagicMock(spec=Span, trace_id="trace1"),
        mock.MagicMock(spec=Span, trace_id="trace2"),
    ]

    with pytest.raises(MlflowException, match="All spans must belong to the same trace"):
        store._log_spans_to_uc_table("catalog.schema.table", spans, "databricks")


@mock.patch("mlflow.store.tracking.databricks_rest_store.get_databricks_workspace_client_config")
@mock.patch("mlflow.store.tracking.databricks_rest_store.http_request")
@mock.patch("mlflow.store.tracking.databricks_rest_store.verify_rest_response")
def test_log_spans_to_uc_table_success(mock_verify, mock_http_request, mock_get_config):
    from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan

    # Mock configuration
    mock_config = mock.MagicMock()
    mock_config.authenticate.return_value = {"Authorization": "Bearer token"}
    mock_get_config.return_value = mock_config

    # Create real OTel proto spans
    otel_span1 = OTelProtoSpan()
    otel_span1.name = "span1"
    otel_span1.trace_id = b"trace123"

    otel_span2 = OTelProtoSpan()
    otel_span2.name = "span2"
    otel_span2.trace_id = b"trace123"

    # Mock spans
    mock_span1 = mock.MagicMock(spec=Span)
    mock_span1.trace_id = "trace123"
    mock_span1.to_otel_proto.return_value = otel_span1

    mock_span2 = mock.MagicMock(spec=Span)
    mock_span2.trace_id = "trace123"
    mock_span2.to_otel_proto.return_value = otel_span2

    spans = [mock_span1, mock_span2]

    # Mock HTTP response
    mock_response = mock.MagicMock()
    mock_http_request.return_value = mock_response

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("http://localhost"))

    # Execute
    store._log_spans_to_uc_table("catalog.schema.spans", spans, "databricks")

    # Verify calls
    mock_get_config.assert_called_once_with("databricks")
    mock_http_request.assert_called_once()
    mock_verify.assert_called_once_with(mock_response, "/api/2.0/tracing/otel/v1/traces")

    # Verify HTTP request details
    call_kwargs = mock_http_request.call_args
    assert call_kwargs[1]["method"] == "POST"
    assert call_kwargs[1]["endpoint"] == "/api/2.0/tracing/otel/v1/traces"
    assert "Content-Type" in call_kwargs[1]["extra_headers"]
    assert call_kwargs[1]["extra_headers"]["Content-Type"] == "application/x-protobuf"
    assert "X-Databricks-UC-Table-Name" in call_kwargs[1]["extra_headers"]
    assert call_kwargs[1]["extra_headers"]["X-Databricks-UC-Table-Name"] == "catalog.schema.spans"


@mock.patch("mlflow.store.tracking.databricks_rest_store.get_databricks_workspace_client_config")
def test_log_spans_to_uc_table_config_error(mock_get_config):
    mock_get_config.side_effect = Exception("Config failed")

    mock_span = mock.MagicMock(spec=Span, trace_id="trace123")
    spans = [mock_span]

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("http://localhost"))

    with pytest.raises(MlflowException, match="Failed to log spans to UC table"):
        store._log_spans_to_uc_table("catalog.schema.spans", spans, "databricks")
