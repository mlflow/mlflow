import json
from unittest import mock

import pytest
from google.protobuf.json_format import MessageToDict

import mlflow
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
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND
from mlflow.protos.databricks_tracing_pb2 import (
    CreateTrace,
    CreateTraceUCStorageLocation,
    DeleteTraceTag,
    GetTraceInfo,
    GetTraces,
    LinkExperimentToUCTraceLocation,
    SearchTraces,
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
    trace_info_to_proto,
    trace_location_from_databricks_uc_schema,
    trace_location_to_proto,
    trace_to_proto,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _V3_TRACE_REST_API_PATH_PREFIX,
    _V4_TRACE_REST_API_PATH_PREFIX,
    MlflowHostCreds,
)


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
        assert endpoint == f"{_V4_TRACE_REST_API_PATH_PREFIX}/batch"

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(trace, Trace) for trace in result)
        assert result[0].info.trace_id == span1.trace_id
        assert result[1].info.trace_id == span2.trace_id


def test_search_traces():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response
    response.text = json.dumps(
        {
            "trace_infos": [
                {
                    "trace_id": "tr-1234",
                    "trace_location": {
                        "type": "MLFLOW_EXPERIMENT",
                        "mlflow_experiment": {"experiment_id": "1234"},
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

    # Parameters for search_traces
    filter_string = "state = 'OK'"
    max_results = 10
    order_by = ["request_time DESC"]
    page_token = "12345abcde"
    locations = ["catalog.schema"]

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            locations=locations,
        )

        # Verify the correct endpoint was called (now V4 by default)
        call_args = mock_http.call_args[1]
        assert call_args["endpoint"] == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search"

        # Verify the correct parameters were passed
        json_body = call_args["json"]
        # The field name should now be 'locations' instead of 'trace_locations'
        assert "locations" in json_body
        # The experiment_ids are converted to trace_locations
        assert len(json_body["locations"]) == 1
        assert json_body["locations"][0]["uc_schema"]["catalog_name"] == "catalog"
        assert json_body["locations"][0]["uc_schema"]["schema_name"] == "schema"
        assert json_body["filter"] == filter_string
        assert json_body["max_results"] == max_results
        assert json_body["order_by"] == order_by
        assert json_body["page_token"] == page_token

    # Verify the correct parameters were passed and the correct trace info objects were returned
    # for either endpoint
    assert len(trace_infos) == 1
    assert isinstance(trace_infos[0], TraceInfo)
    assert trace_infos[0].trace_id == "tr-1234"
    assert trace_infos[0].experiment_id == "1234"
    assert trace_infos[0].request_time == 123
    # V3's state maps to V2's status
    assert trace_infos[0].state == TraceStatus.OK.to_state()
    # This is correct because TraceInfoV3.from_proto converts the repeated field tags to a dict
    assert trace_infos[0].tags == {"k": "v"}
    assert trace_infos[0].trace_metadata == {"key": "value", "mlflow.trace_schema.version": "3"}
    assert token == "token"


def test_search_traces_with_mixed_locations():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response
    response.text = json.dumps(
        {
            "trace_infos": [
                {
                    "trace_id": "tr-1234",
                    "trace_location": {
                        "type": "MLFLOW_EXPERIMENT",
                        "mlflow_experiment": {"experiment_id": "1234"},
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

    # Parameters for search_traces
    filter_string = "state = 'OK'"
    max_results = 10
    order_by = ["request_time DESC"]
    page_token = "12345abcde"
    locations = ["1234", "catalog.schema"]
    trace_locations = []
    for location in locations:
        if "." not in location:
            trace_locations.append(
                trace_location_to_proto(TraceLocation.from_experiment_id(location))
            )
        else:
            catalog, schema = location.split(".")
            trace_locations.append(
                trace_location_to_proto(trace_location_from_databricks_uc_schema(catalog, schema))
            )

    expected_request = SearchTraces(
        locations=trace_locations,
        filter=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            locations=locations,
        )

        _verify_requests(
            mock_http,
            creds,
            "traces/search",
            "POST",
            message_to_json(expected_request),
            version="4.0",
        )

    # Verify the correct parameters were passed and the correct trace info objects were returned
    # for either endpoint
    assert len(trace_infos) == 1
    assert isinstance(trace_infos[0], TraceInfo)
    assert trace_infos[0].trace_id == "tr-1234"
    assert trace_infos[0].experiment_id == "1234"
    assert trace_infos[0].request_time == 123
    # V3's state maps to V2's status
    assert trace_infos[0].state == TraceStatus.OK.to_state()
    # This is correct because TraceInfoV3.from_proto converts the repeated field tags to a dict
    assert trace_infos[0].tags == {"k": "v"}
    assert trace_infos[0].trace_metadata == {"key": "value", "mlflow.trace_schema.version": "3"}
    assert token == "token"


def test_search_traces_errors():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)

    def mock_http_request(*args, **kwargs):
        if kwargs.get("endpoint") == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search":
            raise MlflowException("V4 endpoint not supported", error_code=ENDPOINT_NOT_FOUND)
        return mock.MagicMock()

    with mock.patch("mlflow.utils.rest_utils.http_request", side_effect=mock_http_request):
        with pytest.raises(
            MlflowException,
            match="Searching traces by locations including UC schemas is not supported",
        ):
            store.search_traces(
                locations=["catalog.schema"],
            )


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


def test_search_traces_fallback_to_v3():
    creds = MlflowHostCreds("https://hello")
    store = DatabricksTracingRestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response
    response.text = json.dumps(
        {
            "traces": [
                # v3 API result only includes trace_info
                {
                    "trace_id": "tr-1234",
                    "trace_location": {
                        "type": "MLFLOW_EXPERIMENT",
                        "mlflow_experiment": {"experiment_id": "1234"},
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

    # Parameters for search_traces
    experiment_ids = ["1234"]
    filter_string = "state = 'OK'"
    max_results = 10
    order_by = ["request_time DESC"]
    page_token = "12345abcde"

    def mock_http_request(*args, **kwargs):
        if kwargs.get("endpoint") == f"{_V4_TRACE_REST_API_PATH_PREFIX}/search":
            raise MlflowException("V4 endpoint not supported", error_code=ENDPOINT_NOT_FOUND)
        return response

    # Test with databricks tracking URI (using v3 endpoint)
    with mock.patch(
        "mlflow.utils.rest_utils.http_request", side_effect=mock_http_request
    ) as mock_http:
        trace_infos, token = store.search_traces(
            locations=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

        # Verify the correct endpoint was called
        call_args = mock_http.call_args[1]
        assert call_args["endpoint"] == f"{_V3_TRACE_REST_API_PATH_PREFIX}/search"

        # Verify the correct parameters were passed
        json_body = call_args["json"]
        # The field name should now be 'locations' instead of 'trace_locations'
        assert "locations" in json_body
        # The experiment_ids are converted to trace_locations
        assert len(json_body["locations"]) == 1
        assert json_body["locations"][0]["mlflow_experiment"]["experiment_id"] == experiment_ids[0]
        assert json_body["filter"] == filter_string
        assert json_body["max_results"] == max_results
        assert json_body["order_by"] == order_by
        assert json_body["page_token"] == page_token

    # Verify the correct parameters were passed and the correct trace info objects were returned
    # for either endpoint
    assert len(trace_infos) == 1
    assert isinstance(trace_infos[0], TraceInfo)
    assert trace_infos[0].trace_id == "tr-1234"
    assert trace_infos[0].experiment_id == "1234"
    assert trace_infos[0].request_time == 123
    # V3's state maps to V2's status
    assert trace_infos[0].state == TraceStatus.OK.to_state()
    # This is correct because TraceInfoV3.from_proto converts the repeated field tags to a dict
    assert trace_infos[0].tags == {"k": "v"}
    assert trace_infos[0].trace_metadata == {"key": "value", "mlflow.trace_schema.version": "3"}
    assert token == "token"


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
    sql_warehouse_id = "warehouse123"
    model_id = "model123"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            locations=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
            sql_warehouse_id=sql_warehouse_id,
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
        assert trace_infos[0].trace_metadata == {"key": "value", "mlflow.trace_schema.version": "3"}
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
            uc_schema=uc_schema,
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
            == f"{_V4_TRACE_REST_API_PATH_PREFIX}/location/{experiment_id}"
        )

        assert isinstance(result, UCSchemaLocation)
        assert result.catalog_name == "test_catalog"
        assert result.schema_name == "test_schema"
        assert result.otel_spans_table_name == "test_spans"
        assert result.otel_logs_table_name == "test_logs"


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
            uc_schema=uc_schema,
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
            == f"{_V4_TRACE_REST_API_PATH_PREFIX}/location/{experiment_id}"
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
            location="test_catalog.test_schema",
        )

        mock_call.assert_called_once()
        call_args = mock_call.call_args

        assert call_args[0][0] == UnLinkExperimentToUCTraceLocation
        request_body = json.loads(call_args[0][1])
        assert request_body["experiment_id"] == experiment_id
        assert request_body["location"] == "test_catalog.test_schema"
        expected_endpoint = (
            f"{_V4_TRACE_REST_API_PATH_PREFIX}/location/{experiment_id}/test_catalog.test_schema"
        )
        assert call_args[1]["endpoint"] == expected_endpoint
