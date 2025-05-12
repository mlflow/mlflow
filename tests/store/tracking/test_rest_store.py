import datetime
import json
import time
from unittest import mock

import pytest

import mlflow
from mlflow.entities import (
    Dataset,
    DatasetInput,
    Experiment,
    ExperimentTag,
    InputTag,
    LifecycleStage,
    Metric,
    Param,
    RunTag,
    SourceType,
    ViewType,
)
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.protos.service_pb2 import (
    CreateAssessment,
    CreateRun,
    DeleteExperiment,
    DeleteRun,
    DeleteTag,
    DeleteTraces,
    EndTrace,
    GetExperimentByName,
    GetTraceInfo,
    GetTraceInfoV3,
    LogBatch,
    LogInputs,
    LogMetric,
    LogModel,
    LogParam,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SearchRuns,
    SetExperimentTag,
    SetTag,
    SetTraceTag,
    StartTrace,
    StartTraceV3,
)
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking.request_header.default_request_header_provider import (
    DefaultRequestHeaderProvider,
)
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _V3_TRACE_REST_API_PATH_PREFIX,
    MlflowHostCreds,
    get_search_traces_v3_endpoint,
)


class MyCoolException(Exception):
    pass


class CustomErrorHandlingRestStore(RestStore):
    def _call_endpoint(self, api, json_body):
        raise MyCoolException("cool")


def mock_http_request():
    return mock.patch(
        "mlflow.utils.rest_utils.http_request",
        return_value=mock.MagicMock(status_code=200, text="{}"),
    )


@mock.patch("requests.Session.request")
def test_successful_http_request(request):
    def mock_request(*args, **kwargs):
        # Filter out None arguments
        assert args == ("POST", "https://hello/api/2.0/mlflow/experiments/search")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        assert kwargs == {
            "allow_redirects": True,
            "json": {"view_type": "ACTIVE_ONLY"},
            "headers": DefaultRequestHeaderProvider().request_headers(),
            "verify": True,
            "timeout": 120,
        }
        response = mock.MagicMock()
        response.status_code = 200
        response.text = '{"experiments": [{"name": "Exp!", "lifecycle_stage": "active"}]}'
        return response

    request.side_effect = mock_request

    store = RestStore(lambda: MlflowHostCreds("https://hello"))
    experiments = store.search_experiments()
    assert experiments[0].name == "Exp!"


@mock.patch("requests.Session.request")
def test_failed_http_request(request):
    response = mock.MagicMock()
    response.status_code = 404
    response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
    request.return_value = response

    store = RestStore(lambda: MlflowHostCreds("https://hello"))
    with pytest.raises(MlflowException, match="RESOURCE_DOES_NOT_EXIST: No experiment"):
        store.search_experiments()


@mock.patch("requests.Session.request")
def test_failed_http_request_custom_handler(request):
    response = mock.MagicMock()
    response.status_code = 404
    response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
    request.return_value = response

    store = CustomErrorHandlingRestStore(lambda: MlflowHostCreds("https://hello"))
    with pytest.raises(MyCoolException, match="cool"):
        store.search_experiments()


@mock.patch("requests.Session.request")
def test_response_with_unknown_fields(request):
    experiment_json = {
        "experiment_id": "1",
        "name": "My experiment",
        "artifact_location": "foo",
        "lifecycle_stage": "deleted",
        "OMG_WHAT_IS_THIS_FIELD": "Hooly cow",
    }

    response = mock.MagicMock()
    response.status_code = 200
    experiments = {"experiments": [experiment_json]}
    response.text = json.dumps(experiments)
    request.return_value = response

    store = RestStore(lambda: MlflowHostCreds("https://hello"))
    experiments = store.search_experiments()
    assert len(experiments) == 1
    assert experiments[0].name == "My experiment"


def _args(host_creds, endpoint, method, json_body, use_v3=False, retry_timeout_seconds=None):
    version = "3.0" if use_v3 else "2.0"
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
    http_request, host_creds, endpoint, method, json_body, use_v3=False, retry_timeout_seconds=None
):
    """
    Verify HTTP requests in tests.

    Args:
        http_request: The mocked HTTP request object
        host_creds: MlflowHostCreds object
        endpoint: The endpoint being called (e.g., "traces/123")
        method: The HTTP method (e.g., "GET", "POST")
        json_body: The request body as a JSON string
        use_v3: If True, verify using /api/3.0/mlflow/ prefix instead of /api/2.0/mlflow/
                This is used for trace-related endpoints that use the V3 API.
        retry_timeout_seconds: The retry timeout seconds to use for the request
    """
    http_request.assert_any_call(
        **(_args(host_creds, endpoint, method, json_body, use_v3, retry_timeout_seconds))
    )


def test_requestor():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    user_name = "mock user"
    source_name = "rest test"
    run_name = "my name"

    source_name_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_name", return_value=source_name
    )
    source_type_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_type",
        return_value=SourceType.LOCAL,
    )
    with (
        mock_http_request() as mock_http,
        mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store),
        mock.patch("mlflow.tracking.context.default_context._get_user", return_value=user_name),
        mock.patch("time.time", return_value=13579),
        source_name_patch,
        source_type_patch,
    ):
        with mlflow.start_run(experiment_id="43", run_name=run_name):
            cr_body = message_to_json(
                CreateRun(
                    experiment_id="43",
                    user_id=user_name,
                    run_name=run_name,
                    start_time=13579000,
                    tags=[
                        ProtoRunTag(key="mlflow.source.name", value=source_name),
                        ProtoRunTag(key="mlflow.source.type", value="LOCAL"),
                        ProtoRunTag(key="mlflow.user", value=user_name),
                        ProtoRunTag(key="mlflow.runName", value=run_name),
                    ],
                )
            )
            expected_kwargs = _args(creds, "runs/create", "POST", cr_body)

            assert mock_http.call_count == 1
            actual_kwargs = mock_http.call_args[1]

            # Test the passed tag values separately from the rest of the request
            # Tag order is inconsistent on Python 2 and 3, but the order does not matter
            expected_tags = expected_kwargs["json"].pop("tags")
            actual_tags = actual_kwargs["json"].pop("tags")

            assert sorted(expected_tags, key=lambda t: t["key"]) == sorted(
                actual_tags, key=lambda t: t["key"]
            )
            assert expected_kwargs == actual_kwargs

    with mock_http_request() as mock_http:
        store.log_param("some_uuid", Param("k1", "v1"))
        body = message_to_json(
            LogParam(run_uuid="some_uuid", run_id="some_uuid", key="k1", value="v1")
        )
        _verify_requests(mock_http, creds, "runs/log-parameter", "POST", body)

    with mock_http_request() as mock_http:
        store.set_experiment_tag("some_id", ExperimentTag("t1", "abcd" * 1000))
        body = message_to_json(
            SetExperimentTag(experiment_id="some_id", key="t1", value="abcd" * 1000)
        )
        _verify_requests(mock_http, creds, "experiments/set-experiment-tag", "POST", body)

    with mock_http_request() as mock_http:
        store.set_tag("some_uuid", RunTag("t1", "abcd" * 1000))
        body = message_to_json(
            SetTag(run_uuid="some_uuid", run_id="some_uuid", key="t1", value="abcd" * 1000)
        )
        _verify_requests(mock_http, creds, "runs/set-tag", "POST", body)

    with mock_http_request() as mock_http:
        store.delete_tag("some_uuid", "t1")
        body = message_to_json(DeleteTag(run_id="some_uuid", key="t1"))
        _verify_requests(mock_http, creds, "runs/delete-tag", "POST", body)

    with mock_http_request() as mock_http:
        store.log_metric("u2", Metric("m1", 0.87, 12345, 3))
        body = message_to_json(
            LogMetric(run_uuid="u2", run_id="u2", key="m1", value=0.87, timestamp=12345, step=3)
        )
        _verify_requests(mock_http, creds, "runs/log-metric", "POST", body)

    with mock_http_request() as mock_http:
        metrics = [
            Metric("m1", 0.87, 12345, 0),
            Metric("m2", 0.49, 12345, -1),
            Metric("m3", 0.58, 12345, 2),
        ]
        params = [Param("p1", "p1val"), Param("p2", "p2val")]
        tags = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
        store.log_batch(run_id="u2", metrics=metrics, params=params, tags=tags)
        metric_protos = [metric.to_proto() for metric in metrics]
        param_protos = [param.to_proto() for param in params]
        tag_protos = [tag.to_proto() for tag in tags]
        body = message_to_json(
            LogBatch(run_id="u2", metrics=metric_protos, params=param_protos, tags=tag_protos)
        )
        _verify_requests(mock_http, creds, "runs/log-batch", "POST", body)

    with mock_http_request() as mock_http:
        dataset = Dataset(name="name", digest="digest", source_type="st", source="source")
        tag = InputTag(key="k1", value="v1")
        dataset_input = DatasetInput(dataset=dataset, tags=[tag])
        store.log_inputs("some_uuid", [dataset_input])
        body = message_to_json(LogInputs(run_id="some_uuid", datasets=[dataset_input.to_proto()]))
        _verify_requests(mock_http, creds, "runs/log-inputs", "POST", body)

    with mock_http_request() as mock_http:
        store.delete_run("u25")
        _verify_requests(
            mock_http, creds, "runs/delete", "POST", message_to_json(DeleteRun(run_id="u25"))
        )

    with mock_http_request() as mock_http:
        store.restore_run("u76")
        _verify_requests(
            mock_http, creds, "runs/restore", "POST", message_to_json(RestoreRun(run_id="u76"))
        )

    with mock_http_request() as mock_http:
        store.delete_experiment("0")
        _verify_requests(
            mock_http,
            creds,
            "experiments/delete",
            "POST",
            message_to_json(DeleteExperiment(experiment_id="0")),
        )

    with mock_http_request() as mock_http:
        store.restore_experiment("0")
        _verify_requests(
            mock_http,
            creds,
            "experiments/restore",
            "POST",
            message_to_json(RestoreExperiment(experiment_id="0")),
        )

    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        response = mock.MagicMock()
        response.status_code = 200
        response.text = '{"runs": ["1a", "2b", "3c"], "next_page_token": "67890fghij"}'
        mock_http.return_value = response
        result = store.search_runs(
            ["0", "1"],
            "params.p1 = 'a'",
            ViewType.ACTIVE_ONLY,
            max_results=10,
            order_by=["a"],
            page_token="12345abcde",
        )

        expected_message = SearchRuns(
            experiment_ids=["0", "1"],
            filter="params.p1 = 'a'",
            run_view_type=ViewType.to_proto(ViewType.ACTIVE_ONLY),
            max_results=10,
            order_by=["a"],
            page_token="12345abcde",
        )
        _verify_requests(mock_http, creds, "runs/search", "POST", message_to_json(expected_message))
        assert result.token == "67890fghij"

    with mock_http_request() as mock_http:
        run_id = "run_id"
        m = Model(artifact_path="model/path", run_id="run_id", flavors={"tf": "flavor body"})
        store.record_logged_model("run_id", m)
        expected_message = LogModel(run_id=run_id, model_json=json.dumps(m.get_tags_dict()))
        _verify_requests(
            mock_http, creds, "runs/log-model", "POST", message_to_json(expected_message)
        )

    # if model has config, it should be removed from the model_json before sending to the server
    with mock_http_request() as mock_http:
        run_id = "run_id"
        flavors_with_config = {
            "tf": "flavor body",
            "python_function": {"config": {"a": 1}, "code": "code"},
        }
        m_with_config = Model(
            artifact_path="model/path", run_id="run_id", flavors=flavors_with_config
        )
        store.record_logged_model("run_id", m_with_config)
        flavors = m_with_config.get_tags_dict().get("flavors", {})
        assert all("config" not in v for v in flavors.values())
        expected_message = LogModel(
            run_id=run_id, model_json=json.dumps(m_with_config.get_tags_dict())
        )
        _verify_requests(
            mock_http,
            creds,
            "runs/log-model",
            "POST",
            message_to_json(expected_message),
        )

    with mock_http_request() as mock_http:
        request_id = "tr-123"
        # Regular call, which will use V2 API
        store.get_trace_info(request_id)
        v2_expected_message = GetTraceInfo(request_id=request_id)
        _verify_requests(
            mock_http,
            creds,
            "traces/tr-123/info",
            "GET",
            message_to_json(v2_expected_message),
        )

    # For V3 call, we need to ensure the mock's behavior matches expectations
    with mock_http_request() as mock_http:
        request_id = "tr-123"
        # Successful V3 API call (no fallback)
        store.get_trace_info(request_id, should_query_v3=True)

        # Verify the V3 API was called
        v3_expected_message = GetTraceInfoV3(trace_id=request_id)
        _verify_requests(
            mock_http,
            creds,
            "traces/tr-123",
            "GET",
            message_to_json(v3_expected_message),
            use_v3=True,
        )

    # Now test the fallback path by raising an exception from the V3 call
    with mock_http_request() as mock_http:
        request_id = "tr-123"
        # Make the first call raise an exception
        calls = []

        def side_effect(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:  # First call (V3 API)
                raise MlflowException("V3 API not available")
            # Second call (fallback to V2 API) returns a normal response
            response = mock.MagicMock()
            response.status_code = 200
            response.text = "{}"
            return response

        mock_http.side_effect = side_effect

        # Now when we call get_trace_info, it should try V3 first, fail, then fall back to V2
        store.get_trace_info(request_id, should_query_v3=True)

        # Check call arguments to verify V2 fallback was used
        assert len(mock_http.call_args_list) == 2

        # First call should be to V3 API
        v3_call = mock_http.call_args_list[0]
        assert v3_call[1]["endpoint"] == "/api/3.0/mlflow/traces/tr-123"


def test_get_experiment_by_name():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        response = mock.MagicMock()
        response.status_code = 200
        experiment = Experiment(
            experiment_id="123",
            name="abc",
            artifact_location="/abc",
            lifecycle_stage=LifecycleStage.ACTIVE,
        )
        response.text = json.dumps(
            {"experiment": json.loads(message_to_json(experiment.to_proto()))}
        )
        mock_http.return_value = response
        result = store.get_experiment_by_name("abc")
        expected_message0 = GetExperimentByName(experiment_name="abc")
        _verify_requests(
            mock_http,
            creds,
            "experiments/get-by-name",
            "GET",
            message_to_json(expected_message0),
        )
        assert result.experiment_id == experiment.experiment_id
        assert result.name == experiment.name
        assert result.artifact_location == experiment.artifact_location
        assert result.lifecycle_stage == experiment.lifecycle_stage
        # Test GetExperimentByName against nonexistent experiment
        mock_http.reset_mock()
        nonexistent_exp_response = mock.MagicMock()
        nonexistent_exp_response.status_code = 404
        nonexistent_exp_response.text = MlflowException(
            "Exp doesn't exist!", RESOURCE_DOES_NOT_EXIST
        ).serialize_as_json()
        mock_http.return_value = nonexistent_exp_response
        assert store.get_experiment_by_name("nonexistent-experiment") is None
        expected_message1 = GetExperimentByName(experiment_name="nonexistent-experiment")
        _verify_requests(
            mock_http,
            creds,
            "experiments/get-by-name",
            "GET",
            message_to_json(expected_message1),
        )
        assert mock_http.call_count == 1


def test_search_experiments():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.search_experiments(
            view_type=ViewType.DELETED_ONLY,
            max_results=5,
            filter_string="name",
            order_by=["name"],
            page_token="abc",
        )
        _verify_requests(
            mock_http,
            creds,
            "experiments/search",
            "POST",
            message_to_json(
                SearchExperiments(
                    view_type=ViewType.DELETED_ONLY,
                    max_results=5,
                    filter="name",
                    order_by=["name"],
                    page_token="abc",
                )
            ),
        )


def _mock_response_with_200_status_code():
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    return mock_response


def test_get_metric_history_paginated():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    response_1 = _mock_response_with_200_status_code()
    response_2 = _mock_response_with_200_status_code()
    response_payload_1 = {
        "metrics": [
            {"key": "a_metric", "value": 42, "timestamp": 123456777, "step": 0},
            {"key": "a_metric", "value": 46, "timestamp": 123456797, "step": 1},
        ],
        "next_page_token": "AcursorForTheRestofTheData",
    }
    response_1.text = json.dumps(response_payload_1)
    response_payload_2 = {
        "metrics": [
            {"key": "a_metric", "value": 40, "timestamp": 123456877, "step": 2},
            {"key": "a_metric", "value": 56, "timestamp": 123456897, "step": 3},
        ],
        "next_page_token": "",
    }
    response_2.text = json.dumps(response_payload_2)
    with mock.patch(
        "requests.Session.request", side_effect=[response_1, response_2]
    ) as mock_request:
        # Fetch the first page
        metrics = store.get_metric_history(
            run_id="2", metric_key="a_metric", max_results=2, page_token=None
        )
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["params"] == {
            "max_results": 2,
            "metric_key": "a_metric",
            "run_id": "2",
            "run_uuid": "2",
        }
        assert len(metrics) == 2
        assert metrics[0] == Metric(key="a_metric", value=42, timestamp=123456777, step=0)
        assert metrics[1] == Metric(key="a_metric", value=46, timestamp=123456797, step=1)
        assert metrics.token == "AcursorForTheRestofTheData"
        # Fetch the second page
        mock_request.reset_mock()
        metrics = store.get_metric_history(
            run_id="2", metric_key="a_metric", max_results=2, page_token=metrics.token
        )
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["params"] == {
            "max_results": 2,
            "page_token": "AcursorForTheRestofTheData",
            "metric_key": "a_metric",
            "run_id": "2",
            "run_uuid": "2",
        }
        assert len(metrics) == 2
        assert metrics[0] == Metric(key="a_metric", value=40, timestamp=123456877, step=2)
        assert metrics[1] == Metric(key="a_metric", value=56, timestamp=123456897, step=3)
        assert metrics.token is None


def test_get_metric_history_on_non_existent_metric_key():
    creds = MlflowHostCreds("https://hello")
    rest_store = RestStore(lambda: creds)
    empty_metric_response = _mock_response_with_200_status_code()
    empty_metric_response.text = json.dumps({})
    with mock.patch(
        "requests.Session.request", side_effect=[empty_metric_response]
    ) as mock_request:
        metrics = rest_store.get_metric_history(run_id="1", metric_key="test_metric")
        mock_request.assert_called_once()
        assert metrics == []


def test_start_trace():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    request_id = "tr-123"
    experiment_id = "447585625682310"
    timestamp_ms = 123
    # Metadata/tags values should be string, but should not break for other types too
    metadata = {"key1": "val1", "key2": "val2", "key3": 123}
    tags = {"tag1": "tv1", "tag2": "tv2", "tag3": None}
    expected_request = StartTrace(
        experiment_id=experiment_id,
        timestamp_ms=123,
        request_metadata=[
            ProtoTraceRequestMetadata(key=k, value=str(v)) for k, v in metadata.items()
        ],
        tags=[ProtoTraceTag(key=k, value=str(v)) for k, v in tags.items()],
    )
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "trace_info": {
                "request_id": request_id,
                "experiment_id": experiment_id,
                "timestamp_ms": timestamp_ms,
                "execution_time_ms": None,
                "status": 0,  # Running
                "request_metadata": [{"key": k, "value": str(v)} for k, v in metadata.items()],
                "tags": [{"key": k, "value": str(v)} for k, v in tags.items()],
            }
        }
    )
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.start_trace(
            experiment_id=experiment_id,
            timestamp_ms=timestamp_ms,
            request_metadata=metadata,
            tags=tags,
        )
        _verify_requests(mock_http, creds, "traces", "POST", message_to_json(expected_request))
        assert isinstance(res, TraceInfoV2)
        assert res.request_id == request_id
        assert res.experiment_id == experiment_id
        assert res.timestamp_ms == timestamp_ms
        assert res.execution_time_ms == 0
        assert res.status == TraceStatus.UNSPECIFIED
        assert res.request_metadata == {k: str(v) for k, v in metadata.items()}
        assert res.tags == {k: str(v) for k, v in tags.items()}


def test_start_trace_v3(monkeypatch):
    monkeypatch.setenv(MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.name, "1")

    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    trace = Trace(
        info=TraceInfo(
            trace_id="tr-123",
            trace_location=TraceLocation.from_experiment_id("123"),
            request_time=123,
            execution_duration=10,
            state=TraceState.OK,
            request_preview="",
            response_preview="",
            trace_metadata={},
        ),
        data=TraceData(),
    )

    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps({})

    expected_request = StartTraceV3(trace=trace.to_proto())

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        store.start_trace_v3(trace)
        _verify_requests(
            mock_http,
            creds,
            "traces",
            "POST",
            message_to_json(expected_request),
            use_v3=True,
            retry_timeout_seconds=1,
        )


def test_end_trace():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    experiment_id = "447585625682310"
    request_id = "tr-123"
    timestamp_ms = 123
    status = TraceStatus.OK
    metadata = {"key1": "val1", "key2": "val2"}
    tags = {"tag1": "tv1", "tag2": "tv2"}
    expected_request = EndTrace(
        request_id=request_id,
        timestamp_ms=123,
        status=status,
        request_metadata=[ProtoTraceRequestMetadata(key=k, value=v) for k, v in metadata.items()],
        tags=[ProtoTraceTag(key=k, value=v) for k, v in tags.items()],
    )
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "trace_info": {
                "request_id": request_id,
                "experiment_id": experiment_id,
                "timestamp_ms": timestamp_ms,
                "execution_time_ms": 12345,
                "status": 1,  # OK
                "request_metadata": [{"key": k, "value": v} for k, v in metadata.items()],
                "tags": [{"key": k, "value": v} for k, v in tags.items()],
            }
        }
    )

    with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=True):
        with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
            res = store.end_trace(
                request_id=request_id,
                timestamp_ms=timestamp_ms,
                status=status,
                request_metadata=metadata,
                tags=tags,
            )
            _verify_requests(
                mock_http,
                creds,
                f"traces/{request_id}",
                "PATCH",
                message_to_json(expected_request),
                use_v3=False,
            )
            assert isinstance(res, TraceInfoV2)
            assert res.request_id == request_id
            assert res.experiment_id == experiment_id
            assert res.timestamp_ms == timestamp_ms
            assert res.execution_time_ms == 12345
            assert res.status == TraceStatus.OK
            assert res.request_metadata == metadata
            assert res.tags == tags


def test_search_traces():
    """Test the search_traces method with default behavior using SearchTracesV3Request."""
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response
    response.text = json.dumps(
        {
            "traces": [
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

    # Test with databricks tracking URI (using v3 endpoint)
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=True):
            trace_infos, token = store.search_traces(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )

            # Verify the correct endpoint was called
            endpoint = get_search_traces_v3_endpoint(is_databricks=True)
            call_args = mock_http.call_args[1]
            assert call_args["endpoint"] == endpoint
            assert endpoint == f"{_V3_TRACE_REST_API_PATH_PREFIX}/search"

            # Verify the correct parameters were passed
            json_body = call_args["json"]
            # The field name should now be 'locations' instead of 'trace_locations'
            assert "locations" in json_body
            # The experiment_ids are converted to trace_locations
            assert len(json_body["locations"]) == 1
            assert (
                json_body["locations"][0]["mlflow_experiment"]["experiment_id"] == experiment_ids[0]
            )
            assert json_body["filter"] == filter_string
            assert json_body["max_results"] == max_results
            assert json_body["order_by"] == order_by
            assert json_body["page_token"] == page_token

    # Test with non-databricks tracking URI (using v2 endpoint)
    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=False):
            # For V2 API, use a different response with tags in the list format
            v2_response = mock.MagicMock()
            v2_response.status_code = 200
            v2_response.text = json.dumps(
                {
                    "traces": [
                        {
                            "request_id": "tr-1234",  # V2 uses request_id instead of trace_id
                            "experiment_id": "1234",
                            "timestamp_ms": 123,  # V2 uses timestamp_ms instead of request_time
                            "execution_time_ms": 456,
                            "status": "OK",  # V2 uses status instead of state
                            "request_metadata": [{"key": "key", "value": "value"}],
                            "tags": [{"key": "k", "value": "v"}],
                        }
                    ],
                    "next_page_token": "token",
                }
            )
            mock_http.return_value = v2_response

            trace_infos, token = store.search_traces(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
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
    assert trace_infos[0].trace_metadata == {"key": "value"}
    assert token == "token"


def test_search_unified_traces():
    """Test the search_traces method when using SearchUnifiedTraces with sql_warehouse_id."""
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
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
            experiment_ids=experiment_ids,
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
        assert trace_infos[0].trace_metadata == {"key": "value"}
        assert token == "token"


def test_get_artifact_uri_for_trace_compatibility():
    """Test that get_artifact_uri_for_trace works with both TraceInfo and TraceInfoV3 objects."""
    from mlflow.tracing.utils.artifact_utils import get_artifact_uri_for_trace

    # Create a TraceInfo (v2) object
    trace_info_v2 = TraceInfoV2(
        request_id="tr-1234",
        experiment_id="1234",
        timestamp_ms=123,
        execution_time_ms=456,
        status=TraceStatus.OK,
        request_metadata={"key": "value"},
        tags={MLFLOW_ARTIFACT_LOCATION: "s3://bucket/trace-v2-path"},
    )

    # Create a TraceInfoV3 object
    trace_location = TraceLocation.from_experiment_id("5678")
    trace_info_v3 = TraceInfo(
        trace_id="tr-5678",
        trace_location=trace_location,
        request_time=789,
        state=TraceState.OK,
        trace_metadata={"key3": "value3"},
        tags={MLFLOW_ARTIFACT_LOCATION: "s3://bucket/trace-v3-path"},
    )

    # Test that get_artifact_uri_for_trace works with TraceInfo (v2)
    v2_uri = get_artifact_uri_for_trace(trace_info_v2)
    assert v2_uri == "s3://bucket/trace-v2-path"

    # Test that get_artifact_uri_for_trace works with TraceInfoV3
    v3_uri = get_artifact_uri_for_trace(trace_info_v3)
    assert v3_uri == "s3://bucket/trace-v3-path"

    # Test that get_artifact_uri_for_trace raises the expected exception when tag is missing
    trace_info_no_tag = TraceInfoV2(
        request_id="tr-1234",
        experiment_id="1234",
        timestamp_ms=123,
        execution_time_ms=456,
        status=TraceStatus.OK,
        tags={},
    )
    with pytest.raises(MlflowException, match="Unable to determine trace artifact location"):
        get_artifact_uri_for_trace(trace_info_no_tag)


@pytest.mark.parametrize(
    "delete_traces_kwargs",
    [
        {"experiment_id": "0", "request_ids": ["tr-1234"]},
        {"experiment_id": "0", "max_timestamp_millis": 1, "max_traces": 2},
    ],
)
def test_delete_traces(delete_traces_kwargs):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    request = DeleteTraces(**delete_traces_kwargs)
    response.text = json.dumps({"traces_deleted": 1})
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.delete_traces(**delete_traces_kwargs)
        _verify_requests(mock_http, creds, "traces/delete-traces", "POST", message_to_json(request))
        assert res == 1


def test_set_trace_tag():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    request_id = "tr-1234"
    request = SetTraceTag(
        key="k",
        value="v",
    )
    response.text = "{}"

    with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=True):
        with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
            res = store.set_trace_tag(
                request_id=request_id,
                key=request.key,
                value=request.value,
            )
            _verify_requests(
                mock_http,
                creds,
                f"traces/{request_id}/tags",
                "PATCH",
                message_to_json(request),
                use_v3=False,
            )
            assert res is None


def test_log_assessment():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": "tr-1234",
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

    assessment = Assessment(
        trace_id="tr-1234",
        name="assessment_name",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4o-mini"
        ),
        create_time_ms=int(time.time() * 1000),
        last_update_time_ms=int(time.time() * 1000),
        feedback=Feedback(value=True),
        rationale="rationale",
        metadata={"model": "gpt-4o-mini"},
        span_id=None,
    )

    request = CreateAssessment(assessment=assessment.to_proto())
    with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=True):
        with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
            res = store.create_assessment(assessment)

            _verify_requests(
                mock_http,
                creds,
                "traces/tr-1234/assessments",
                "POST",
                message_to_json(request),
                use_v3=True,
            )
            assert isinstance(res, Assessment)


@pytest.mark.parametrize(
    ("updates", "expected_request_json"),
    [
        (
            {"name": "updated_name"},
            {
                "assessment": {
                    "assessment_id": "1234",
                    "trace_id": "tr-1234",
                    "assessment_name": "updated_name",
                },
                "update_mask": "assessmentName",
            },
        ),
        (
            {"expectation": Expectation(value="updated_value")},
            {
                "assessment": {
                    "assessment_id": "1234",
                    "trace_id": "tr-1234",
                    "expectation": {"value": "updated_value"},
                },
                "update_mask": "expectation",
            },
        ),
        (
            {
                "feedback": Feedback(value=0.5),
                "rationale": "update",
                "metadata": {"model": "gpt-4o-mini"},
            },
            {
                "assessment": {
                    "assessment_id": "1234",
                    "trace_id": "tr-1234",
                    "feedback": {"value": 0.5},
                    "rationale": "update",
                    "metadata": {"model": "gpt-4o-mini"},
                },
                "update_mask": "feedback,rationale,metadata",
            },
        ),
    ],
)
def test_update_assessment(updates, expected_request_json):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": "tr-1234",
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-25T01:23:45Z",
                "feedback": {"value": True},
                "rationale": "rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )

    with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=True):
        with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
            res = store.update_assessment(
                trace_id="tr-1234",
                assessment_id="1234",
                **updates,
            )

            _verify_requests(
                mock_http,
                creds,
                "traces/tr-1234/assessments/1234",
                "PATCH",
                json.dumps(expected_request_json),
                use_v3=True,
            )
            assert isinstance(res, Assessment)


def test_delete_assessment():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    with mock.patch.object(store, "_is_databricks_tracking_uri", return_value=True):
        with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
            store.delete_assessment(trace_id="tr-1234", assessment_id="1234")

        expected_request_json = {"assessment_id": "1234", "trace_id": "tr-1234"}
        _verify_requests(
            mock_http,
            creds,
            "traces/tr-1234/assessments/1234",
            "DELETE",
            json.dumps(expected_request_json),
            use_v3=True,
        )


def test_update_assessment_invalid_update():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with pytest.raises(MlflowException, match="Exactly one of `expectation` or `feedback`"):
        store.update_assessment(
            trace_id="tr-1234",
            assessment_id="1234",
            expectation=Expectation(value="updated_value"),
            feedback=Feedback(value=0.5),
        )


def test_get_trace_info_v3_api():
    """
    Test that get_trace_info with should_query_v3=True correctly extracts the trace_info
    from the nested structure in the V3 API response.
    """
    trace_id = "tr-123"
    trace_location = TraceLocation.from_experiment_id("exp-123")
    trace_info_v3 = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location,
        request_time=int(datetime.datetime(2023, 5, 1, 12, 0, 0).timestamp() * 1000),
        state=TraceState.OK,
        trace_metadata={"key1": "value1"},
        tags={"tag1": "value1"},
    )

    with mock.patch(
        "mlflow.entities.trace_info_v3.TraceInfoV3.from_proto", return_value=trace_info_v3
    ) as mock_from_proto:
        store = RestStore(lambda: MlflowHostCreds("https://hello"))

        with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
            # Set up the mock to return a dummy response with a trace field
            mock_response = mock.MagicMock()
            mock_response.trace.trace_info = mock.MagicMock()
            mock_call_endpoint.return_value = mock_response

            # Call the method we're testing
            result = store.get_trace_info(trace_id, should_query_v3=True)

            # Verify mock_from_proto was called with the trace_info from the response
            mock_from_proto.assert_called_once_with(mock_response.trace.trace_info)

            # Verify we get the expected object back
            assert result is trace_info_v3
            assert isinstance(result, TraceInfo)
            assert result.trace_id == trace_id
            assert result.experiment_id == "exp-123"
            assert result.trace_metadata == {"key1": "value1"}
            assert result.tags == {"tag1": "value1"}
            assert result.state == TraceState.OK
