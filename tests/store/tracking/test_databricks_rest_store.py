import json
import time
from unittest import mock

import pytest
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan

import mlflow
from mlflow.entities import Span
from mlflow.entities.assessment import (
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
    FeedbackValue,
)
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation, UCSchemaLocation
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_TRACING_SQL_WAREHOUSE_ID,
)
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND
from mlflow.protos.databricks_tracing_pb2 import (
    BatchGetTraces,
    CreateAssessment,
    CreateTraceUCStorageLocation,
    DeleteAssessment,
    DeleteTraceTag,
    GetAssessment,
    GetTraceInfo,
    LinkExperimentToUCTraceLocation,
    SetTraceTag,
    UnLinkExperimentToUCTraceLocation,
)
from mlflow.protos.databricks_tracing_pb2 import UCSchemaLocation as ProtoUCSchemaLocation
from mlflow.protos.service_pb2 import DeleteTraceTag as DeleteTraceTagV3
from mlflow.protos.service_pb2 import GetTraceInfoV3, StartTraceV3
from mlflow.protos.service_pb2 import SetTraceTag as SetTraceTagV3
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.tracing.constant import TRACE_ID_V4_PREFIX
from mlflow.utils.databricks_tracing_utils import (
    assessment_to_proto,
    trace_info_to_proto,
    trace_to_proto,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _V3_TRACE_REST_API_PATH_PREFIX,
    _V4_TRACE_REST_API_PATH_PREFIX,
    MlflowHostCreds,
)


def create_mock_spans(diff_trace_id=False):
    otel_span1 = OTelProtoSpan()
    otel_span1.name = "span1"
    otel_span1.trace_id = b"trace123"

    otel_span2 = OTelProtoSpan()
    otel_span2.name = "span2"
    otel_span2.trace_id = b"trace456" if diff_trace_id else b"trace123"

    # Mock spans
    mock_span1 = mock.MagicMock(spec=Span)
    mock_span1.trace_id = "trace123"
    mock_span1.to_otel_proto.return_value = otel_span1

    mock_span2 = mock.MagicMock(spec=Span)
    mock_span2.trace_id = "trace456" if diff_trace_id else "trace123"
    mock_span2.to_otel_proto.return_value = otel_span2

    return [mock_span1, mock_span2]


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
        trace_location=TraceLocation.from_databricks_uc_schema("catalog", "schema"),
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
    response.text = json.dumps(expected_trace_info)

    expected_request = trace_info_to_proto(trace_info)
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        result = store.start_trace(trace_info)
        _verify_requests(
            mock_http,
            creds,
            "traces/catalog.schema/123/info",
            "POST",
            message_to_json(expected_request),
            version="4.0",
            retry_timeout_seconds=1,
        )
        assert result.trace_id == "trace:/catalog.schema/123"


def test_create_trace_experiment_location_fallback_to_v3(monkeypatch):
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
    v3_response = StartTraceV3.Response(trace=trace.to_proto())

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        mock_call_endpoint.side_effect = [v3_response]

        result = store.start_trace(trace_info)

        assert mock_call_endpoint.call_count == 1
        call_args = mock_call_endpoint.call_args_list[0]
        assert call_args[0][0] == StartTraceV3
        assert result.trace_id == "tr-456"


def test_get_trace_info(monkeypatch):
    with mlflow.start_span(name="test_span_v4") as span:
        span.set_inputs({"input": "test_value"})
        span.set_outputs({"output": "result"})

    trace = mlflow.get_trace(span.trace_id)
    mock_response = GetTraceInfo.Response(trace=trace_to_proto(trace))

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("https://test"))

    location = "catalog.schema"
    v4_trace_id = f"{TRACE_ID_V4_PREFIX}{location}/{span.trace_id}"

    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "test-warehouse")
    with mock.patch.object(store, "_call_endpoint", return_value=mock_response) as mock_call:
        result = store.get_trace_info(v4_trace_id)

        mock_call.assert_called_once()
        call_args = mock_call.call_args

        assert call_args[0][0] == GetTraceInfo

        request_body = call_args[0][1]
        request_data = json.loads(request_body)
        assert request_data["trace_id"] == span.trace_id
        assert request_data["location"] == location
        assert request_data["sql_warehouse_id"] == "test-warehouse"

        endpoint = call_args[1]["endpoint"]
        assert f"/traces/{location}/{span.trace_id}/info" in endpoint

        assert isinstance(result, TraceInfo)
        assert result.trace_id == span.trace_id


def test_get_trace_info_fallback_to_v3():
    with mlflow.start_span(name="test_span_v3") as span:
        span.set_inputs({"input": "test_value"})

    trace = mlflow.get_trace(span.trace_id)
    mock_v3_response = GetTraceInfoV3.Response(trace=trace.to_proto())

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("https://test"))

    with mock.patch.object(store, "_call_endpoint", return_value=mock_v3_response) as mock_call:
        result = store.get_trace_info(span.trace_id)

        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][0] == GetTraceInfoV3

        request_body = call_args[0][1]
        request_data = json.loads(request_body)
        assert request_data["trace_id"] == span.trace_id

        assert isinstance(result, TraceInfo)
        assert result.trace_id == span.trace_id


def test_set_trace_tag():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    location = "catalog.schema"
    trace_id = "tr-1234"
    request = SetTraceTag(
        key="k",
        value="v",
    )
    response.text = "{}"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.set_trace_tag(
            trace_id=f"{TRACE_ID_V4_PREFIX}{location}/{trace_id}",
            key=request.key,
            value=request.value,
        )
        expected_json = {
            "key": request.key,
            "value": request.value,
        }
        mock_http.assert_called_once_with(
            host_creds=creds,
            endpoint=f"/api/4.0/mlflow/traces/{location}/{trace_id}/tags",
            method="PATCH",
            json=expected_json,
        )
        assert res is None


def test_set_trace_tag_fallback():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    trace_id = "tr-1234"
    response.text = "{}"

    with mock.patch.object(
        store, "_call_endpoint", return_value=SetTraceTagV3.Response()
    ) as mock_call:
        result = store.set_trace_tag(
            trace_id=trace_id,
            key="k",
            value="v",
        )

        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][0] == SetTraceTagV3

        request_body = call_args[0][1]
        request_data = json.loads(request_body)
        assert request_data["key"] == "k"
        assert request_data["value"] == "v"
        assert result is None


def test_delete_trace_tag(monkeypatch):
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    location = "catalog.schema"
    trace_id = "tr-1234"
    sql_warehouse_id = "warehouse_456"
    request = DeleteTraceTag(
        trace_id=trace_id,
        location=location,
        key="k",
    )
    response.text = "{}"

    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", sql_warehouse_id)
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.delete_trace_tag(
            trace_id=f"{TRACE_ID_V4_PREFIX}{location}/{trace_id}",
            key=request.key,
        )
        expected_json = {
            "sql_warehouse_id": sql_warehouse_id,
        }
        mock_http.assert_called_once_with(
            host_creds=creds,
            endpoint=f"/api/4.0/mlflow/traces/{location}/{trace_id}/tags/{request.key}",
            method="DELETE",
            json=expected_json,
        )
        assert res is None


def test_delete_trace_tag_fallback():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    trace_id = "tr-1234"
    response.text = "{}"

    with mock.patch.object(
        store, "_call_endpoint", return_value=DeleteTraceTagV3.Response()
    ) as mock_call:
        result = store.delete_trace_tag(
            trace_id=trace_id,
            key="k",
        )

        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][0] == DeleteTraceTagV3

        request_body = call_args[0][1]
        request_data = json.loads(request_body)
        assert request_data["key"] == "k"
        assert result is None


@pytest.mark.parametrize("sql_warehouse_id", [None, "warehouse_override"])
def test_batch_get_traces(monkeypatch, sql_warehouse_id):
    monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, "test-warehouse")
    with mlflow.start_span(name="test_span_1") as span1:
        span1.set_inputs({"input": "test_value_1"})
        span1.set_outputs({"output": "result_1"})

    with mlflow.start_span(name="test_span_2") as span2:
        span2.set_inputs({"input": "test_value_2"})
        span2.set_outputs({"output": "result_2"})

    trace1 = mlflow.get_trace(span1.trace_id)
    trace2 = mlflow.get_trace(span2.trace_id)

    mock_response = BatchGetTraces.Response()
    mock_response.traces.extend([trace_to_proto(trace1), trace_to_proto(trace2)])

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("https://test"))

    location = "catalog.schema"
    v4_trace_id_1 = f"{TRACE_ID_V4_PREFIX}{location}/{span1.trace_id}"
    v4_trace_id_2 = f"{TRACE_ID_V4_PREFIX}{location}/{span2.trace_id}"
    trace_ids = [v4_trace_id_1, v4_trace_id_2]

    with (
        mock.patch.object(store, "_call_endpoint", return_value=mock_response) as mock_call,
    ):
        result = store.batch_get_traces(trace_ids, location)

        mock_call.assert_called_once()
        call_args = mock_call.call_args

        assert call_args[0][0] == BatchGetTraces

        request_body = call_args[0][1]
        request_data = json.loads(request_body)
        assert request_data["sql_warehouse_id"] == "test-warehouse"
        assert request_data["trace_ids"] == [span1.trace_id, span2.trace_id]

        endpoint = call_args[1]["endpoint"]
        assert endpoint == f"{_V4_TRACE_REST_API_PATH_PREFIX}/{location}/batchGet"

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(trace, Trace) for trace in result)
        assert result[0].info.trace_id == span1.trace_id
        assert result[1].info.trace_id == span2.trace_id


def test_search_traces_uc_schema(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACING_SQL_WAREHOUSE_ID.name, "test-warehouse")

    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    response.text = json.dumps(
        {
            "trace_infos": [
                {
                    # REST API uses raw otel id as trace_id
                    "trace_id": "1234",
                    "trace_location": {
                        "type": "UC_SCHEMA",
                        "uc_schema": {"catalog_name": "catalog", "schema_name": "schema"},
                    },
                    "request_time": "1970-01-01T00:00:00.123Z",
                    "execution_duration_ms": 456,
                    "state": "OK",
                    "trace_metadata": {"key": "value"},
                    "tags": {"k": "v"},
                }
            ],
            "next_page_token": "token",
        }
    )

    filter_string = "state = 'OK'"
    max_results = 50
    order_by = ["request_time ASC", "execution_duration_ms DESC"]
    locations = ["catalog.schema"]
    page_token = "12345abcde"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            locations=locations,
            page_token=page_token,
        )

    # V4 endpoint should be called for UC schema locations
    assert mock_http.call_count == 1
    call_args = mock_http.call_args[1]
    assert call_args["endpoint"] == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search"

    json_body = call_args["json"]
    assert "locations" in json_body
    assert len(json_body["locations"]) == 1
    assert json_body["locations"][0]["uc_schema"]["catalog_name"] == "catalog"
    assert json_body["locations"][0]["uc_schema"]["schema_name"] == "schema"
    assert json_body["filter"] == filter_string
    assert json_body["max_results"] == max_results
    assert json_body["order_by"] == order_by
    assert json_body["page_token"] == page_token
    assert json_body["sql_warehouse_id"] == "test-warehouse"

    assert len(trace_infos) == 1
    assert isinstance(trace_infos[0], TraceInfo)
    assert trace_infos[0].trace_id == "trace:/catalog.schema/1234"
    assert trace_infos[0].trace_location.uc_schema.catalog_name == "catalog"
    assert trace_infos[0].trace_location.uc_schema.schema_name == "schema"
    assert trace_infos[0].request_time == 123
    assert trace_infos[0].state == TraceStatus.OK.to_state()
    assert trace_infos[0].tags == {"k": "v"}
    assert trace_infos[0].trace_metadata == {"key": "value"}
    assert token == "token"


@pytest.mark.parametrize(
    "exception",
    [
        # Workspace where SearchTracesV4 is not supported yet
        RestException(
            json={
                "error_code": databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND),
                "message": "Not found",
            }
        ),
        # V4 endpoint does not support searching by experiment ID (yet)
        RestException(
            json={
                "error_code": databricks_pb2.ErrorCode.Name(databricks_pb2.INVALID_PARAMETER_VALUE),
                "message": "MLFLOW_EXPERIMENT locations not yet supported",
            }
        ),
    ],
)
def test_search_traces_experiment_id(exception):
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    response.text = json.dumps(
        {
            "traces": [
                {
                    "trace_id": "tr-1234",
                    "trace_location": {
                        "type": "MLFLOW_EXPERIMENT",
                        "mlflow_experiment": {"experiment_id": "1"},
                    },
                    "request_time": "1970-01-01T00:00:00.123Z",
                    "execution_duration_ms": 456,
                    "state": "OK",
                    "trace_metadata": {"key": "value"},
                    "tags": {"k": "v"},
                }
            ],
            "next_page_token": "token",
        }
    )
    filter_string = "state = 'OK'"
    page_token = "12345abcde"
    locations = ["1"]

    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        # v4 call -> exception, v3 call -> response
        mock_http.side_effect = [exception, response]
        trace_infos, token = store.search_traces(
            filter_string=filter_string,
            page_token=page_token,
            locations=locations,
        )

    # MLflow first tries V4 endpoint, then falls back to V3
    assert mock_http.call_count == 2

    first_call_args = mock_http.call_args_list[0][1]
    assert first_call_args["endpoint"] == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search"

    json_body = first_call_args["json"]
    assert "locations" in json_body
    assert len(json_body["locations"]) == 1
    assert json_body["locations"][0]["mlflow_experiment"]["experiment_id"] == "1"
    assert json_body["filter"] == filter_string
    assert json_body["max_results"] == 100

    second_call_args = mock_http.call_args_list[1][1]
    assert second_call_args["endpoint"] == f"{_V3_TRACE_REST_API_PATH_PREFIX}/search"
    json_body = second_call_args["json"]
    assert len(json_body["locations"]) == 1
    assert json_body["locations"][0]["mlflow_experiment"]["experiment_id"] == "1"
    assert json_body["filter"] == filter_string
    assert json_body["max_results"] == 100

    assert len(trace_infos) == 1
    assert isinstance(trace_infos[0], TraceInfo)
    assert trace_infos[0].trace_id == "tr-1234"
    assert trace_infos[0].experiment_id == "1"
    assert trace_infos[0].request_time == 123
    assert trace_infos[0].state == TraceStatus.OK.to_state()
    assert trace_infos[0].tags == {"k": "v"}
    assert trace_infos[0].trace_metadata == {"key": "value"}
    assert token == "token"


@pytest.mark.parametrize(
    "exception",
    [
        # Workspace where SearchTracesV4 is not supported yet
        RestException(
            json={
                "error_code": databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND),
                "message": "Not found",
            }
        ),
        # V4 endpoint does not support searching by experiment ID (yet)
        RestException(
            json={
                "error_code": databricks_pb2.ErrorCode.Name(databricks_pb2.INVALID_PARAMETER_VALUE),
                "message": "MLFLOW_EXPERIMENT locations not yet supported",
            }
        ),
    ],
)
def test_search_traces_with_mixed_locations(exception):
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    expected_error_message = (
        "Searching traces in UC tables is not supported yet."
        if exception.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND)
        else "The `locations` parameter cannot contain both MLflow experiment and UC schema "
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", side_effect=exception) as mock_http:
        with pytest.raises(MlflowException, match=expected_error_message):
            store.search_traces(
                filter_string="state = 'OK'",
                locations=["1", "catalog.schema"],
            )

    # V4 endpoint should be called first. Not fallback to V3 because location includes UC schema.
    mock_http.assert_called_once()
    call_args = mock_http.call_args[1]
    assert call_args["endpoint"] == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search"

    json_body = call_args["json"]
    assert "locations" in json_body
    assert len(json_body["locations"]) == 2
    assert json_body["locations"][0]["mlflow_experiment"]["experiment_id"] == "1"
    assert json_body["locations"][1]["uc_schema"]["catalog_name"] == "catalog"
    assert json_body["locations"][1]["uc_schema"]["schema_name"] == "schema"


def test_search_traces_does_not_fallback_when_uc_schemas_are_specified():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    def mock_http_request(*args, **kwargs):
        if kwargs.get("endpoint") == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search":
            raise MlflowException("V4 endpoint not supported", error_code=ENDPOINT_NOT_FOUND)
        return mock.MagicMock()

    with mock.patch("mlflow.utils.rest_utils.http_request", side_effect=mock_http_request):
        with pytest.raises(
            MlflowException,
            match="Searching traces in UC tables is not supported yet.",
        ):
            store.search_traces(locations=["catalog.schema"])


def test_search_traces_non_fallback_errors():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        mock_http.side_effect = MlflowException("Random error")
        with pytest.raises(MlflowException, match="Random error"):
            store.search_traces(locations=["catalog.schema"])


def test_search_traces_experiment_ids_deprecated():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    # Test that using experiment_ids raises error saying it's deprecated
    with pytest.raises(
        MlflowException,
        match="experiment_ids.*deprecated.*use.*locations",
    ):
        store.search_traces(
            experiment_ids=["123"],
        )


def test_search_traces_with_missing_location():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    with pytest.raises(MlflowException, match="location.*must be specified"):
        store.search_traces()

    with pytest.raises(MlflowException, match="location.*must be specified"):
        store.search_traces(locations=[])


def test_search_traces_with_invalid_location():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    with pytest.raises(MlflowException, match="Invalid location type:"):
        store.search_traces(locations=["catalog.schema.table_name"])


def test_search_unified_traces():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response (using TraceInfo format for online path)
    response.text = json.dumps(
        {
            "traces": [
                {
                    "request_id": "tr-1234",
                    "experiment_id": "1234",
                    "timestamp_ms": 123,
                    "execution_time_ms": 456,
                    "status": "OK",
                    "tags": [
                        {"key": "k", "value": "v"},
                    ],
                    "request_metadata": [
                        {"key": "key", "value": "value"},
                    ],
                }
            ],
            "next_page_token": "token",
        }
    )

    # Parameters for search_traces
    experiment_ids = ["1234"]
    filter_string = "status = 'OK'"
    max_results = 10
    order_by = ["timestamp_ms DESC"]
    page_token = "12345abcde"
    model_id = "model123"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            locations=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            model_id=model_id,
        )

        # Verify the correct endpoint was called
        call_args = mock_http.call_args[1]
        assert call_args["endpoint"] == "/api/2.0/mlflow/unified-traces"

        # Verify the correct trace info objects were returned
        assert len(trace_infos) == 1
        assert isinstance(trace_infos[0], TraceInfo)
        assert trace_infos[0].trace_id == "tr-1234"
        assert trace_infos[0].experiment_id == "1234"
        assert trace_infos[0].request_time == 123
        # V3's state maps to V2's status
        assert trace_infos[0].state == TraceStatus.OK.to_state()
        assert trace_infos[0].tags == {"k": "v"}
        assert trace_infos[0].trace_metadata == {"key": "value"}
        assert token == "token"


def test_set_experiment_trace_location():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    experiment_id = "123"
    uc_schema = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    sql_warehouse_id = "test-warehouse-id"

    # Mock response for CreateTraceUCStorageLocation
    create_location_response = mock.MagicMock()
    create_location_response.uc_schema = ProtoUCSchemaLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
        otel_spans_table_name="test_spans",
        otel_logs_table_name="test_logs",
    )

    # Mock response for LinkExperimentToUCTraceLocation
    link_response = mock.MagicMock()
    link_response.status_code = 200
    link_response.text = "{}"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        mock_call.side_effect = [create_location_response, link_response]

        result = store.set_experiment_trace_location(
            location=uc_schema,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )

        assert mock_call.call_count == 2

        # Verify CreateTraceUCStorageLocation call
        first_call = mock_call.call_args_list[0]
        assert first_call[0][0] == CreateTraceUCStorageLocation
        create_request_body = json.loads(first_call[0][1])
        assert create_request_body["uc_schema"]["catalog_name"] == "test_catalog"
        assert create_request_body["uc_schema"]["schema_name"] == "test_schema"
        assert create_request_body["sql_warehouse_id"] == sql_warehouse_id
        assert first_call[1]["endpoint"] == f"{_V4_TRACE_REST_API_PATH_PREFIX}/location"

        # Verify LinkExperimentToUCTraceLocation call
        second_call = mock_call.call_args_list[1]
        assert second_call[0][0] == LinkExperimentToUCTraceLocation
        link_request_body = json.loads(second_call[0][1])
        assert link_request_body["experiment_id"] == experiment_id
        assert link_request_body["uc_schema"]["catalog_name"] == "test_catalog"
        assert link_request_body["uc_schema"]["schema_name"] == "test_schema"
        assert link_request_body["uc_schema"]["otel_spans_table_name"] == "test_spans"
        assert link_request_body["uc_schema"]["otel_logs_table_name"] == "test_logs"
        assert (
            second_call[1]["endpoint"]
            == f"{_V4_TRACE_REST_API_PATH_PREFIX}/{experiment_id}/link-location"
        )

        assert isinstance(result, UCSchemaLocation)
        assert result.catalog_name == "test_catalog"
        assert result.schema_name == "test_schema"
        assert result.full_otel_spans_table_name == "test_catalog.test_schema.test_spans"
        assert result.full_otel_logs_table_name == "test_catalog.test_schema.test_logs"


def test_set_experiment_trace_location_with_existing_location():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    experiment_id = "123"
    uc_schema = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    sql_warehouse_id = "test-warehouse-id"

    create_location_response = MlflowException(
        "Location already exists", error_code=databricks_pb2.ALREADY_EXISTS
    )

    # Mock response for LinkExperimentToUCTraceLocation
    link_response = mock.MagicMock()
    link_response.status_code = 200
    link_response.text = "{}"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        mock_call.side_effect = [create_location_response, link_response]

        result = store.set_experiment_trace_location(
            location=uc_schema,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )

        assert mock_call.call_count == 2

        # Verify CreateTraceUCStorageLocation call
        first_call = mock_call.call_args_list[0]
        assert first_call[0][0] == CreateTraceUCStorageLocation
        create_request_body = json.loads(first_call[0][1])
        assert create_request_body["uc_schema"]["catalog_name"] == "test_catalog"
        assert create_request_body["uc_schema"]["schema_name"] == "test_schema"
        assert create_request_body["sql_warehouse_id"] == sql_warehouse_id
        assert first_call[1]["endpoint"] == f"{_V4_TRACE_REST_API_PATH_PREFIX}/location"

        # Verify LinkExperimentToUCTraceLocation call
        second_call = mock_call.call_args_list[1]
        assert second_call[0][0] == LinkExperimentToUCTraceLocation
        link_request_body = json.loads(second_call[0][1])
        assert link_request_body["experiment_id"] == experiment_id
        assert link_request_body["uc_schema"]["catalog_name"] == "test_catalog"
        assert link_request_body["uc_schema"]["schema_name"] == "test_schema"
        assert (
            second_call[1]["endpoint"]
            == f"{_V4_TRACE_REST_API_PATH_PREFIX}/{experiment_id}/link-location"
        )

        assert isinstance(result, UCSchemaLocation)
        assert result.catalog_name == "test_catalog"
        assert result.schema_name == "test_schema"


def test_unset_experiment_trace_location_with_uc_schema():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    experiment_id = "123"

    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    with mock.patch.object(store, "_call_endpoint", return_value=response) as mock_call:
        store.unset_experiment_trace_location(
            experiment_id=experiment_id,
            location=UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema"),
        )

        mock_call.assert_called_once()
        call_args = mock_call.call_args

        assert call_args[0][0] == UnLinkExperimentToUCTraceLocation
        request_body = json.loads(call_args[0][1])
        assert request_body["experiment_id"] == experiment_id
        assert request_body["uc_schema"]["catalog_name"] == "test_catalog"
        assert request_body["uc_schema"]["schema_name"] == "test_schema"
        expected_endpoint = f"{_V4_TRACE_REST_API_PATH_PREFIX}/{experiment_id}/unlink-location"
        assert call_args[1]["endpoint"] == expected_endpoint


def test_log_spans_to_uc_table_empty_spans():
    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("http://localhost"))
    result = store.log_spans("catalog.schema.table", [], tracking_uri="databricks")
    assert result == []


@mock.patch("mlflow.store.tracking.databricks_rest_store.get_databricks_workspace_client_config")
@mock.patch("mlflow.store.tracking.databricks_rest_store.http_request")
@mock.patch("mlflow.store.tracking.databricks_rest_store.verify_rest_response")
@pytest.mark.parametrize("diff_trace_id", [True, False])
def test_log_spans_to_uc_table_success(
    mock_verify, mock_http_request, mock_get_config, diff_trace_id
):
    # Mock configuration
    mock_config = mock.MagicMock()
    mock_config.authenticate.return_value = {"Authorization": "Bearer token"}
    mock_get_config.return_value = mock_config

    spans = create_mock_spans(diff_trace_id)

    # Mock HTTP response
    mock_response = mock.MagicMock()
    mock_http_request.return_value = mock_response

    store = DatabricksTracingRestStore(lambda: MlflowHostCreds("http://localhost"))

    # Execute
    store.log_spans("catalog.schema.spans", spans, tracking_uri="databricks")

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
        store.log_spans("catalog.schema.spans", spans, tracking_uri="databricks")


def test_create_assessment():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_identifier": {
                    "uc_schema": {
                        "catalog_name": "catalog",
                        "schema_name": "schema",
                    },
                    "trace_id": "1234",
                },
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-20T05:47:23Z",
                "feedback": {"value": True},
                "rationale": "rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )

    feedback = Feedback(
        trace_id="trace:/catalog.schema/1234",
        name="assessment_name",
        value=True,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4o-mini"
        ),
        create_time_ms=int(time.time() * 1000),
        last_update_time_ms=int(time.time() * 1000),
        rationale="rationale",
        metadata={"model": "gpt-4o-mini"},
        span_id=None,
    )

    request = CreateAssessment(
        assessment=assessment_to_proto(feedback),
    )
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.create_assessment(
            assessment=feedback,
        )

        _verify_requests(
            mock_http,
            creds,
            "traces/catalog.schema/1234/assessment",
            "POST",
            message_to_json(request),
            version="4.0",
        )
        assert isinstance(res, Feedback)
        assert res.assessment_id is not None
        assert res.value == feedback.value


def test_get_assessment():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    trace_id = "trace:/catalog.schema/1234"
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": trace_id,
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-20T05:47:23Z",
                "feedback": {"value": True},
                "rationale": "rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )
    request = GetAssessment()
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.get_assessment(
            trace_id=trace_id,
            assessment_id="1234",
        )

        _verify_requests(
            mock_http,
            creds,
            "traces/catalog.schema/1234/assessment/1234",
            "GET",
            message_to_json(request),
            version="4.0",
        )
        assert isinstance(res, Feedback)
        assert res.assessment_id == "1234"
        assert res.value is True


def test_update_assessment():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    trace_id = "trace:/catalog.schema/1234"
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "updated_assessment_name",
                "trace_location": {
                    "type": "UC_SCHEMA",
                    "uc_schema": {
                        "catalog_name": "catalog",
                        "schema_name": "schema",
                    },
                },
                "trace_id": "1234",
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-20T05:47:23Z",
                "feedback": {"value": False},
                "rationale": "updated_rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )

    request = {
        "assessment": {
            "assessment_id": "1234",
            "trace_location": {
                "type": "UC_SCHEMA",
                "uc_schema": {
                    "catalog_name": "catalog",
                    "schema_name": "schema",
                    "otel_spans_table_name": "mlflow_experiment_trace_otel_spans",
                    "otel_logs_table_name": "mlflow_experiment_trace_otel_logs",
                },
            },
            "trace_id": "1234",
            "feedback": {"value": False},
            "rationale": "updated_rationale",
            "metadata": {"model": "gpt-4o-mini"},
        },
        "update_mask": "feedback,rationale,metadata",
    }
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.update_assessment(
            trace_id=trace_id,
            assessment_id="1234",
            feedback=FeedbackValue(value=False),
            rationale="updated_rationale",
            metadata={"model": "gpt-4o-mini"},
        )

        _verify_requests(
            mock_http,
            creds,
            "traces/catalog.schema/1234/assessment/1234",
            "PATCH",
            json.dumps(request),
            version="4.0",
        )
        assert isinstance(res, Feedback)
        assert res.assessment_id == "1234"
        assert res.value is False
        assert res.rationale == "updated_rationale"


def test_delete_assessment():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps({})
    trace_id = "trace:/catalog.schema/1234"
    request = DeleteAssessment()
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        store.delete_assessment(
            trace_id=trace_id,
            assessment_id="1234",
        )

        _verify_requests(
            mock_http,
            creds,
            "traces/catalog.schema/1234/assessment/1234",
            "DELETE",
            message_to_json(request),
            version="4.0",
        )
